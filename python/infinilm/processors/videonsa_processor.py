import os

from typing_extensions import override

import torch
from transformers import AutoProcessor

from .processor import InfinilmProcessor, register_processor


@register_processor("qwen2_5_vl")
class VideoNSAProcessor(InfinilmProcessor):
    def __init__(self, model_dir_path: str):
        self.processor = AutoProcessor.from_pretrained(
            model_dir_path, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        self.image_token_id = getattr(self.processor, "image_token_id", None)
        if self.image_token_id is None:
            self.image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.video_token_id = getattr(self.processor, "video_token_id", None)
        if self.video_token_id is None:
            self.video_token_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        self.pixel_values_dtype = torch.bfloat16
        self._configure_media_processors_from_env()

    def _configure_media_processors_from_env(self):
        image_processor = getattr(self.processor, "image_processor", None)
        image_max_pixels = os.getenv("INFINILM_VIDEONSA_IMAGE_MAX_PIXELS")
        image_min_pixels = os.getenv("INFINILM_VIDEONSA_IMAGE_MIN_PIXELS")
        if image_processor is not None and image_max_pixels:
            value = int(image_max_pixels)
            image_processor.max_pixels = value
            image_processor.size = dict(getattr(image_processor, "size", {}) or {})
            image_processor.size["longest_edge"] = value
        if image_processor is not None and image_min_pixels:
            value = int(image_min_pixels)
            image_processor.min_pixels = value
            image_processor.size = dict(getattr(image_processor, "size", {}) or {})
            image_processor.size["shortest_edge"] = value

        video_processor = getattr(self.processor, "video_processor", None)
        if video_processor is None:
            return
        num_frames = os.getenv("INFINILM_VIDEONSA_VIDEO_NUM_FRAMES")
        max_pixels = os.getenv("INFINILM_VIDEONSA_VIDEO_MAX_PIXELS")
        min_pixels = os.getenv("INFINILM_VIDEONSA_VIDEO_MIN_PIXELS")
        if num_frames:
            video_processor.num_frames = int(num_frames)
            video_processor.do_sample_frames = True
        if max_pixels:
            value = int(max_pixels)
            video_processor.max_pixels = value
            video_processor.size = dict(getattr(video_processor, "size", {}) or {})
            video_processor.size["longest_edge"] = value
        if min_pixels:
            value = int(min_pixels)
            video_processor.min_pixels = value
            video_processor.size = dict(getattr(video_processor, "size", {}) or {})
            video_processor.size["shortest_edge"] = value

    def _normalize_messages(self, conversation):
        normalized = []
        for msg in conversation:
            content = msg.get("content", [])
            if isinstance(content, str):
                normalized.append({"role": msg.get("role", "user"), "content": content})
                continue
            items = []
            for item in content:
                typ = item.get("type")
                if typ == "text":
                    items.append({"type": "text", "text": item.get("text", "")})
                elif typ == "image_url":
                    items.append({"type": "image", "image": item["image_url"]["url"]})
                elif typ == "video_url":
                    items.append({"type": "video", "video": item["video_url"]["url"]})
                else:
                    raise NotImplementedError(
                        f"Unsupported VideoNSA content type: {typ}"
                    )
            normalized.append({"role": msg.get("role", "user"), "content": items})
        return normalized

    @override
    def __call__(
        self,
        prompt,
        images=None,
        videos=None,
        audios=None,
        return_tensors: str = None,
        **kwargs,
    ) -> dict:
        results = self.processor(
            text=prompt,
            images=images or None,
            videos=videos or None,
            return_tensors=return_tensors,
            return_mm_token_type_ids=False,
            **kwargs,
        )
        input_ids = results["input_ids"]
        bounds = []
        for row in input_ids:
            row_bounds = []
            start = None
            current = None
            for idx, token_id in enumerate(row.tolist()):
                if token_id in (self.image_token_id, self.video_token_id):
                    if start is None:
                        start = idx
                        current = token_id
                    elif current != token_id:
                        row_bounds.append([start, idx])
                        start = idx
                        current = token_id
                elif start is not None:
                    row_bounds.append([start, idx])
                    start = None
                    current = None
            if start is not None:
                row_bounds.append([start, len(row)])
            bounds.append(row_bounds)

        if bounds and any(bounds):
            max_bounds = max(len(b) for b in bounds)
            image_bound = torch.zeros((len(bounds), max_bounds, 2), dtype=torch.long)
            for i, row_bounds in enumerate(bounds):
                if row_bounds:
                    image_bound[i, : len(row_bounds), :] = torch.tensor(
                        row_bounds, dtype=torch.long
                    )
            results["image_bound"] = image_bound
        return results

    @override
    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        return self.processor.apply_chat_template(
            self._normalize_messages(conversation),
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            **kwargs,
        )

    @override
    def build_model_inputs(
        self,
        scheduler_output,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 1,
        **kwargs,
    ) -> dict:
        import infinicore

        if not scheduler_output.scheduled_requests:
            raise RuntimeError(
                "build_model_inputs called with empty scheduled_requests"
            )

        tokens = []
        seq_lens = []
        seq_offsets = [0]
        block_tables = []
        slot_mapping = []
        cached_lens = []
        position_ids = []
        cu_seqlens = [0]
        mm_data = {}
        max_block_table_len = max(
            1, *(len(req.block_table) for req in scheduler_output.scheduled_requests)
        )
        current_offset = 0

        for req_id, req in enumerate(scheduler_output.scheduled_requests):
            num_cached = req.num_local_cached_tokens
            if scheduler_output.is_prefill:
                req_tokens = req.get_input_tokens()
                tokens_to_compute = req_tokens[num_cached:]
                tokens.extend(tokens_to_compute)
                compute_len = len(tokens_to_compute)
                seq_len = len(req_tokens)
                seq_lens.append(seq_len)
                current_offset += compute_len
                seq_offsets.append(current_offset)
                cached_lens.append(num_cached)
                position_ids.extend(range(num_cached, num_cached + compute_len))

                if (
                    req.processed_inputs is not None
                    and "image_bound" in req.processed_inputs
                ):
                    image_bound = req.processed_inputs["image_bound"]
                    bounds = image_bound[0]
                    num_cached_media = (bounds[:, 1] <= num_cached).sum().item()
                    if num_cached_media < len(bounds):
                        grids = []
                        tensors = []
                        offset = 0
                        pixel_values = req.processed_inputs.get("pixel_values")
                        image_grid = req.processed_inputs.get("image_grid_thw")
                        if pixel_values is not None:
                            for media_idx, grid in enumerate(image_grid):
                                patch_count = int(grid.prod().item())
                                if media_idx >= num_cached_media:
                                    grids.append(grid)
                                    tensors.append(
                                        pixel_values[offset : offset + patch_count]
                                    )
                                offset += patch_count
                        pixel_values_videos = req.processed_inputs.get(
                            "pixel_values_videos"
                        )
                        video_grid = req.processed_inputs.get("video_grid_thw")
                        if pixel_values_videos is not None:
                            offset = 0
                            base = 0 if image_grid is None else len(image_grid)
                            for media_idx, grid in enumerate(video_grid):
                                patch_count = int(grid.prod().item())
                                if base + media_idx >= num_cached_media:
                                    grids.append(grid)
                                    tensors.append(
                                        pixel_values_videos[
                                            offset : offset + patch_count
                                        ]
                                    )
                                offset += patch_count
                        if tensors:
                            pixel_tensor = torch.cat(tensors, dim=0).to(
                                self.pixel_values_dtype
                            )
                            grid_tensor = torch.stack(grids).to(torch.int64)
                            bound = (
                                bounds[num_cached_media:].unsqueeze(0).to(torch.int64)
                            )
                            mm_data.setdefault("pixel_values", []).append(
                                infinicore.from_torch(pixel_tensor)
                            )
                            mm_data.setdefault("tgt_sizes", []).append(
                                infinicore.from_torch(grid_tensor)
                            )
                            mm_data.setdefault("image_bound", []).append(
                                infinicore.from_torch(bound)
                            )
                            mm_data.setdefault("image_req_ids", []).append(req_id)
            else:
                seq_len = req.get_total_length()
                last_token = (
                    req.generated_token_ids[-1]
                    if req.generated_token_ids
                    else req.prompt_token_ids[-1]
                )
                tokens.append(last_token)
                seq_lens.append(seq_len)
                current_offset += 1
                seq_offsets.append(current_offset)
                cached_lens.append(num_cached)
                position_ids.append(seq_len - 1)

            if req.slot_mapping:
                padded_slot_mapping = req.slot_mapping
            else:
                padded_slot_mapping = (
                    list(range(cu_seqlens[-1], cu_seqlens[-1] + len(tokens_to_compute)))
                    if scheduler_output.is_prefill
                    else [current_offset - 1]
                )
            slot_mapping.extend(padded_slot_mapping)

            padded_block_table = req.block_table + [-1] * (
                max_block_table_len - len(req.block_table)
            )
            block_tables.append(padded_block_table)
            cu_seqlens.append(cu_seqlens[-1] + seq_lens[-1])

        return {
            "input_ids": infinicore.from_list([tokens], dtype=infinicore.int64),
            "position_ids": infinicore.from_list(position_ids, dtype=infinicore.int64),
            "past_kv_lengths": infinicore.from_list(
                cached_lens, dtype=infinicore.int32
            ),
            "total_kv_lengths": infinicore.from_list(seq_lens, dtype=infinicore.int32),
            "input_offsets": infinicore.from_list(seq_offsets, dtype=infinicore.int32),
            "cu_seqlens": infinicore.from_list(cu_seqlens, dtype=infinicore.int32),
            "block_tables": infinicore.from_list(block_tables, dtype=infinicore.int32),
            "slot_mapping": infinicore.from_list(slot_mapping, dtype=infinicore.int64),
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            **mm_data,
        }

    @override
    def get_tokenizer(self):
        return self.tokenizer

    @override
    def get_mm_token_index_list(
        self, prompt_token_ids, image_ids=None, video_ids=None, **kwargs
    ):
        ids = list(image_ids or []) + list(video_ids or [])
        out = []
        media_idx = 0
        start = None
        current = None
        for i, token_id in enumerate(prompt_token_ids):
            if token_id in (self.image_token_id, self.video_token_id):
                if start is None:
                    start = i
                    current = token_id
                elif current != token_id:
                    out.append(
                        {
                            "start_index": start,
                            "end_index": i,
                            "identifier": ids[media_idx],
                        }
                    )
                    media_idx += 1
                    start = i
                    current = token_id
            elif start is not None:
                out.append(
                    {"start_index": start, "end_index": i, "identifier": ids[media_idx]}
                )
                media_idx += 1
                start = None
                current = None
        if start is not None:
            out.append(
                {
                    "start_index": start,
                    "end_index": len(prompt_token_ids),
                    "identifier": ids[media_idx],
                }
            )
        return out
