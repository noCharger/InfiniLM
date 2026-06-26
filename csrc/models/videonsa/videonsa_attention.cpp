#include "videonsa_attention.hpp"
#include "../../global_state/global_state.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/matmul.hpp"
#include "infinicore/ops/mha_varlen.hpp"
#include "infinicore/ops/mul.hpp"
#include "infinicore/ops/nsa_compress_paged_cache.hpp"
#include "infinicore/ops/nsa_paged_attention.hpp"
#include "infinicore/ops/paged_caching.hpp"
#include "infinicore/ops/sigmoid.hpp"
#include "infinicore/ops/silu.hpp"
#include "infinicore/ops/softmax.hpp"
#include "infinicore/ops/sum.hpp"
#include "infinicore/ops/take.hpp"
#include <algorithm>

namespace infinilm::models::videonsa {

namespace {

constexpr size_t kNsaBlockSize = 64;
constexpr int kNsaSelectBlocks = 1;

infinicore::Tensor scalar_tensor(float value, const infinicore::Device &device) {
    auto cpu = infinicore::Tensor::from_blob(&value, {1}, infinicore::DataType::F32, infinicore::Device::cpu());
    return cpu->to(device);
}

infinicore::Tensor mean_pool_blocks(const infinicore::Tensor &x, size_t block_size) {
    const size_t seq_len = x->size(0);
    std::vector<infinicore::Tensor> blocks;
    blocks.reserve((seq_len + block_size - 1) / block_size);
    for (size_t start = 0; start < seq_len; start += block_size) {
        const size_t len = std::min(block_size, seq_len - start);
        auto block = x->narrow({{0, start, len}});
        auto pooled = infinicore::op::sum(block, {0}, false);
        blocks.push_back(pooled->unsqueeze(0));
    }
    return blocks.size() == 1 ? blocks.front() : infinicore::op::cat(blocks, 0);
}

infinicore::Tensor repeat_group_tensor(const infinicore::Tensor &x, size_t repeats) {
    std::vector<infinicore::Tensor> copies;
    copies.reserve(repeats);
    for (size_t i = 0; i < repeats; ++i) {
        copies.push_back(x);
    }
    return copies.size() == 1 ? copies.front() : infinicore::op::cat(copies, 0);
}

infinicore::Tensor grouped_dense_attention(const infinicore::Tensor &q,
                                           const infinicore::Tensor &k,
                                           const infinicore::Tensor &v,
                                           size_t num_heads,
                                           size_t num_kv_heads,
                                           size_t head_dim,
                                           float scale) {
    const size_t seq_len = q->size(0);
    const size_t kv_len = k->size(0);
    const size_t heads_per_group = num_heads / num_kv_heads;
    std::vector<infinicore::Tensor> group_outputs;
    group_outputs.reserve(num_kv_heads);
    for (size_t g = 0; g < num_kv_heads; ++g) {
        auto q_group = q->narrow({{1, g * heads_per_group, heads_per_group}})
                           ->permute({1, 0, 2})
                           ->contiguous();                               // [heads_per_group, seq, dim]
        auto k_group = k->narrow({{1, g, 1}})->squeeze(1)->unsqueeze(0); // [1, kv, dim]
        auto v_group = v->narrow({{1, g, 1}})->squeeze(1)->unsqueeze(0); // [1, kv, dim]
        auto k_repeated = repeat_group_tensor(k_group, heads_per_group);
        auto v_repeated = repeat_group_tensor(v_group, heads_per_group);
        auto scores = infinicore::op::matmul(q_group, k_repeated->permute({0, 2, 1}), scale);
        infinicore::op::softmax_(scores, scores, -1);
        auto out = infinicore::op::matmul(scores, v_repeated)
                       ->view({heads_per_group, seq_len, head_dim})
                       ->permute({1, 0, 2})
                       ->contiguous();
        group_outputs.push_back(out);
    }
    return group_outputs.size() == 1 ? group_outputs.front() : infinicore::op::cat(group_outputs, 1);
}

infinicore::Tensor expand_head_gate(const infinicore::Tensor &gate, size_t head_dim) {
    const size_t seq_len = gate->size(1);
    const size_t num_heads = gate->size(2);
    auto flat_gate = gate->contiguous()->view({seq_len * num_heads});
    std::vector<int64_t> expand_indices(seq_len * num_heads * head_dim);
    for (size_t row = 0; row < seq_len * num_heads; ++row) {
        for (size_t d = 0; d < head_dim; ++d) {
            expand_indices[row * head_dim + d] = static_cast<int64_t>(row);
        }
    }
    auto indices = infinicore::Tensor::empty({seq_len, num_heads, head_dim}, infinicore::DataType::I64, gate->device());
    infinicore::context::memcpyH2D(indices->data(), expand_indices.data(), expand_indices.size() * sizeof(int64_t), false);
    return infinicore::op::take(flat_gate, indices);
}

} // namespace

VideoNSAAttention::VideoNSAAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     size_t layer_idx,
                                     const infinicore::Device &device)
    : infinilm::layers::attention::Attention(model_config, layer_idx, device) {
    const auto &dtype{model_config->get_dtype()};
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    max_position_embeddings_ = model_config->get<size_t>("max_position_embeddings");

    INFINICORE_NN_MODULE_INIT(g_proj_1, hidden_size, hidden_size, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(g_proj_2, hidden_size, 3 * total_num_heads, true, dtype, device);
}

infinicore::Tensor VideoNSAAttention::forward(const infinicore::Tensor &positions,
                                              const infinicore::Tensor &hidden_states) const {
    const auto &forward_context = infinilm::global_state::get_forward_context();
    const auto &mm_metadata = forward_context.mm_metadata;
    const bool has_visual_ranges = mm_metadata.visual_token_ranges.has_value() && !mm_metadata.visual_token_ranges->empty();
    const auto &attn_metadata = forward_context.attn_metadata;
    const bool is_paged = ::infinilm::backends::AttentionBackend::PAGED_ATTN == attention_backend_
                       || ::infinilm::backends::AttentionBackend::FLASH_ATTN == attention_backend_;
    const bool is_flash_attn = ::infinilm::backends::AttentionBackend::FLASH_ATTN == attention_backend_;
    const bool has_paged_metadata = attn_metadata.total_sequence_lengths.has_value()
                                 && attn_metadata.slot_mapping.has_value()
                                 && attn_metadata.block_tables.has_value();
    const bool is_decode = has_paged_metadata
                        && hidden_states->size(0) == 1
                        && hidden_states->size(1) == attn_metadata.total_sequence_lengths.value()->shape()[0];
    const bool can_use_paged_decode_nsa = is_paged && has_paged_metadata && is_decode;
    const bool can_use_fast_prefill = is_flash_attn
                                   && has_paged_metadata
                                   && hidden_states->size(0) == 1
                                   && !is_decode;
    const bool can_use_paged_prefill_nsa = false && has_visual_ranges
                                        && is_paged
                                        && has_paged_metadata
                                        && hidden_states->size(0) == 1
                                        && hidden_states->size(1) != attn_metadata.total_sequence_lengths.value()->shape()[0];

    const bool can_use_static_scattered_nsa = ::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_;
    if (!can_use_static_scattered_nsa && !can_use_fast_prefill && !can_use_paged_prefill_nsa && !can_use_paged_decode_nsa) {
        return infinilm::layers::attention::Attention::forward(positions, hidden_states);
    }

    if (can_use_fast_prefill && !can_use_paged_prefill_nsa) {
        return infinilm::layers::attention::Attention::forward(positions, hidden_states);
    }

    auto hidden_states_mutable = hidden_states;
    const size_t batch_size = hidden_states->size(0);
    const size_t seq_len = hidden_states->size(1);
    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    auto pos_shape = positions->shape();
    infinicore::Tensor pos_ids_for_rope = positions;
    if (pos_shape.size() == 2) {
        pos_ids_for_rope = positions->narrow({{0, 0, 1}})->view({pos_shape[1]});
    } else if (pos_shape.size() != 1) {
        throw std::runtime_error("VideoNSAAttention: Unexpected position_ids shape");
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    if (can_use_static_scattered_nsa) {
        auto q_static = q->view({batch_size, seq_len, num_attention_heads_, head_dim_});
        auto k_static = k->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
        auto v_static = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

        auto q_rope = infinicore::Tensor::empty({batch_size, num_attention_heads_, seq_len, head_dim_}, q_static->dtype(), q_static->device())
                          ->permute({0, 2, 1, 3});
        rotary_emb_->forward(q_rope, q_static, pos_ids_for_rope);
        rotary_emb_->forward(k_static, pos_ids_for_rope, true);

        auto &kv_cache = forward_context.kv_cache_vec[layer_idx_];
        auto k_cache_layer = kv_cache->narrow({{0, 0, 1}})->squeeze(0);
        auto v_cache_layer = kv_cache->narrow({{0, 1, 1}})->squeeze(0);
        const size_t cache_pos = reinterpret_cast<int32_t *>(attn_metadata.past_sequence_lengths.value()->to(infinicore::Device::cpu())->data())[0];
        const size_t total_seq_len = cache_pos + seq_len;
        k_cache_layer->narrow({{2, cache_pos, seq_len}})->copy_from(k_static->permute({0, 2, 1, 3}));
        v_cache_layer->narrow({{2, cache_pos, seq_len}})->copy_from(v_static->permute({0, 2, 1, 3}));
        auto k_total = k_cache_layer->narrow({{2, 0, total_seq_len}});
        auto v_total = v_cache_layer->narrow({{2, 0, total_seq_len}});

        auto gate_hidden = g_proj_1_->forward(hidden_states_mutable);
        gate_hidden = infinicore::op::silu(gate_hidden);
        auto gates = g_proj_2_->forward(gate_hidden);
        gates = infinicore::op::sigmoid(gates)->view({batch_size, seq_len, 3, num_attention_heads_});

        std::vector<infinicore::Tensor> batch_outputs;
        batch_outputs.reserve(batch_size);
        for (size_t b = 0; b < batch_size; ++b) {
            auto q_b = q_rope->narrow({{0, b, 1}})->squeeze(0);
            auto k_b = k_total->narrow({{0, b, 1}})->squeeze(0)->permute({1, 0, 2});
            auto v_b = v_total->narrow({{0, b, 1}})->squeeze(0)->permute({1, 0, 2});

            auto k_cmp = mean_pool_blocks(k_b, kNsaBlockSize);
            auto v_cmp = mean_pool_blocks(v_b, kNsaBlockSize);
            auto comp_heads = grouped_dense_attention(q_b, k_cmp, v_cmp, num_attention_heads_, num_key_value_heads_, head_dim_, scale);

            const size_t win_len = std::min<size_t>(256, total_seq_len);
            auto k_win = k_b->narrow({{0, total_seq_len - win_len, win_len}});
            auto v_win = v_b->narrow({{0, total_seq_len - win_len, win_len}});
            auto win_heads = grouped_dense_attention(q_b, k_win, v_win, num_attention_heads_, num_key_value_heads_, head_dim_, scale);

            auto gates_b = gates->narrow({{0, b, 1}});
            auto g_cmp = expand_head_gate(gates_b->narrow({{2, 0, 1}})->squeeze(2), head_dim_);
            auto g_sel = expand_head_gate(gates_b->narrow({{2, 1, 1}})->squeeze(2), head_dim_);
            auto g_win = expand_head_gate(gates_b->narrow({{2, 2, 1}})->squeeze(2), head_dim_);

            // Experimental scattered path: selected-block attention reuses the compressed output
            // to keep this branch expressible with existing dense ops only.
            auto comp_part = infinicore::op::mul(comp_heads, g_cmp);
            auto sel_part = infinicore::op::mul(comp_heads, g_sel);
            auto win_part = infinicore::op::mul(win_heads, g_win);
            auto mixed = infinicore::op::add(infinicore::op::add(comp_part, sel_part), win_part);
            batch_outputs.push_back(mixed->view({1, seq_len, num_attention_heads_ * head_dim_}));
        }
        auto attn_output = batch_outputs.size() == 1 ? batch_outputs.front() : infinicore::op::cat(batch_outputs, 0);
        return o_proj_->forward(attn_output);
    }

    auto q_reshaped = q->view({seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({seq_len, num_key_value_heads_, head_dim_});
    rotary_emb_->forward(q_reshaped, pos_ids_for_rope, true);
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

    auto &kv_cache = forward_context.kv_cache_vec[layer_idx_];
    auto k_cache_layer = kv_cache->narrow({{0, 0, 1}})->squeeze(0);
    auto v_cache_layer = kv_cache->narrow({{0, 1, 1}})->squeeze(0);
    auto k_cache_for_nsa = is_flash_attn ? k_cache_layer->permute({0, 2, 1, 3}) : k_cache_layer;
    auto v_cache_for_nsa = is_flash_attn ? v_cache_layer->permute({0, 2, 1, 3}) : v_cache_layer;

    infinicore::op::paged_caching_(k_cache_for_nsa, v_cache_for_nsa, k_reshaped, v_reshaped, attn_metadata.slot_mapping.value());

    if (can_use_paged_decode_nsa) {
        auto gate_hidden = g_proj_1_->forward(hidden_states_mutable);
        gate_hidden = infinicore::op::silu(gate_hidden);
        auto gates = g_proj_2_->forward(gate_hidden);
        gates = infinicore::op::sigmoid(gates)->view({seq_len, 3, num_attention_heads_});

        const size_t page_block_size = k_cache_for_nsa->size(2);
        const size_t subblocks_per_page = page_block_size / kNsaBlockSize;
        const size_t cmp_blocks = k_cache_for_nsa->size(0) * subblocks_per_page;
        const bool need_cmp_alloc = !nsa_k_cmp_cache_.has_value()
                                 || nsa_k_cmp_cache_.value()->size(0) != cmp_blocks
                                 || nsa_k_cmp_cache_.value()->size(1) != num_key_value_heads_
                                 || nsa_k_cmp_cache_.value()->size(2) != head_dim_;
        if (need_cmp_alloc) {
            nsa_k_cmp_cache_ = infinicore::Tensor::empty({cmp_blocks, num_key_value_heads_, head_dim_}, k_cache_for_nsa->dtype(), k_cache_layer->device());
            nsa_v_cmp_cache_ = infinicore::Tensor::empty({cmp_blocks, num_key_value_heads_, head_dim_}, v_cache_for_nsa->dtype(), v_cache_layer->device());
            nsa_cmp_cache_ready_ = false;
        }
        const size_t num_decode_seqs = attn_metadata.total_sequence_lengths.value()->size(0);
        const bool update_last_only = nsa_cmp_cache_ready_ && nsa_cmp_cached_num_seqs_ == num_decode_seqs;
        infinicore::op::nsa_compress_paged_cache_(
            nsa_k_cmp_cache_.value(),
            nsa_v_cmp_cache_.value(),
            k_cache_for_nsa,
            v_cache_for_nsa,
            attn_metadata.block_tables.value(),
            attn_metadata.total_sequence_lengths.value(),
            static_cast<int>(kNsaBlockSize),
            update_last_only);
        nsa_cmp_cache_ready_ = true;
        nsa_cmp_cached_num_seqs_ = num_decode_seqs;

        auto nsa_heads = infinicore::Tensor::empty({seq_len, num_attention_heads_, head_dim_}, q_reshaped->dtype(), q_reshaped->device());
        infinicore::op::nsa_paged_attention_(
            nsa_heads,
            q_reshaped,
            nsa_k_cmp_cache_.value(),
            nsa_v_cmp_cache_.value(),
            k_cache_for_nsa,
            v_cache_for_nsa,
            attn_metadata.block_tables.value(),
            attn_metadata.total_sequence_lengths.value(),
            gates,
            scale,
            static_cast<int>(kNsaBlockSize),
            64,
            kNsaSelectBlocks);
        auto attn_output = nsa_heads->view({1, seq_len, num_attention_heads_ * head_dim_});
        return o_proj_->forward(attn_output);
    }

    auto k_cmp = mean_pool_blocks(k_reshaped, kNsaBlockSize);
    auto v_cmp = mean_pool_blocks(v_reshaped, kNsaBlockSize);
    auto nsa_heads = grouped_dense_attention(q_reshaped, k_cmp, v_cmp, num_attention_heads_, num_key_value_heads_, head_dim_, scale);

    auto gate_hidden = g_proj_1_->forward(hidden_states_mutable);
    gate_hidden = infinicore::op::silu(gate_hidden);
    auto gates = g_proj_2_->forward(gate_hidden);
    gates = infinicore::op::sigmoid(gates)->view({1, seq_len, 3, num_attention_heads_});
    auto gate_sum = infinicore::op::sum(gates, {2}, false); // [1, seq, heads]
    auto gate_expanded = expand_head_gate(gate_sum, head_dim_);
    nsa_heads = infinicore::op::mul(nsa_heads, gate_expanded);

    auto attn_output = nsa_heads->view({1, seq_len, num_attention_heads_ * head_dim_});
    return o_proj_->forward(attn_output);
}

void VideoNSAAttention::process_weights_after_loading() {
    infinilm::layers::attention::Attention::process_weights_after_loading();
    g_proj_1_->process_weights_after_loading();
    g_proj_2_->process_weights_after_loading();
}

void VideoNSAAttention::reset_runtime_state() const {
    infinilm::layers::attention::Attention::reset_runtime_state();
    g_proj_1_->reset_runtime_state();
    g_proj_2_->reset_runtime_state();
    nsa_k_cmp_cache_.reset();
    nsa_v_cmp_cache_.reset();
    nsa_cmp_cache_ready_ = false;
    nsa_cmp_cached_num_seqs_ = 0;
}

} // namespace infinilm::models::videonsa
