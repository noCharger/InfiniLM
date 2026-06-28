# Serving-layer profiler

Instruments the InfiniLM engine loop (`LLMEngine.step` → `Scheduler.schedule` →
`ModelRunner.execute_model`) so you can drill `vllm bench serve` client metrics
(TTFT / ITL / throughput) **down one layer** into their serving-side causes:
scheduling overhead, batch occupancy, KV-cache pressure, prefill/decode mix, and
the host-vs-GPU split of every step.

It is **opt-in** and **zero-cost when off** (every hook is an early-return when
`INFINILM_PROFILE_STEPS` is unset). It does **not** time individual CUDA kernels —
that lives below the C++/stream boundary and, under `--enable-graph`, inside a
captured graph, so reach for a vendor profiler (Nsight Systems / `msprof`) there.

## What it captures (per step)

`{phase, batch, num_tokens, q_wait, q_run, kv_used, kv_total, kv_util,
schedule_ms, build_inputs_ms, forward_ms, to_list_ms, update_ms, step_ms}`

Beyond upstream `feat/test-profiling`'s running totals, it keeps a per-step trace,
which is what makes ITL **distributions** (p50/p99), KV utilization over time, and
batch-occupancy possible — a sum can't give percentiles.

## Use

```bash
# 1. start the server with profiling on (works for your prod launch, just add envs)
INFINILM_PROFILE_STEPS=1 \
INFINILM_PROFILE_SYNC=1 \              # sync_device() before timing forward -> true GPU ms
INFINILM_PROFILE_TRACE=/tmp/trace.jsonl \
python python/infinilm/server/inference_server.py --device nvidia --model ... \
    --port 8102 --tp 1 --enable-graph --enable-paged-attn --attn flash-attn ...

# 2. drive load as usual
vllm bench serve --backend openai-chat --model 9g_8b_thinking --port 8102 \
    --save-result --result-filename result.json ...

# 3. drill down (optionally line up against the client numbers)
python -m infinilm.profiler.analyze_trace /tmp/trace.jsonl --bench result.json
```

On shutdown an `infinilm_profile*` summary is printed to stderr (atexit), and the
JSONL trace holds every step for offline analysis.

## Env vars

| var | default | meaning |
|---|---|---|
| `INFINILM_PROFILE_STEPS` | off | master switch |
| `INFINILM_PROFILE_SYNC` | off | `sync_device()` before stopping the forward clock → accurate GPU time under async/graph (adds one sync per step; leave off for max-throughput runs) |
| `INFINILM_PROFILE_TRACE` | none | path for the per-step JSONL trace |
| `INFINILM_PROFILE_MAX_STEPS` | 200000 | in-memory trace cap (JSONL file is unbounded) |

## How to read it

- **`forward_ms` dominates** → the model is the cost; go to the kernel layer (Nsight/`msprof` on `mha_kvcache`, the GEMMs).
- **host overhead high** (`schedule`/`update`/`to_list` big) → the serving loop is the cost; batch detokenize, trim the scheduler path, or enable graph.
- **client ITL ≫ engine step_ms** → the gap is *above* the engine (queueing / network / SSE / asyncio), not the model.
- **KV util near 100%** → preemption/recompute; raise `num_blocks` / lower `max_cache_len`. Usually the source of ITL p99 spikes.
- **decode batch mean low** → concurrency-bound; push continuous batching before micro-optimizing kernels.

## Files

- `serving_profiler.py` — `ServingProfiler` recorder (no infinilm/infinicore deps; importable anywhere).
- `analyze_trace.py` — standalone drill-down + `vllm bench serve` correlation.
- hooks: `llm.py` (`LLMEngine.step`), `model_runner.py` (`_model_forward`).
