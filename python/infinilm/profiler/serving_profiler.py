"""Serving-layer step profiler for the InfiniLM engine loop.

Why this exists
---------------
`vllm bench serve` (or any OpenAI-compatible load generator) measures the engine
as a black box from the client: TTFT / ITL / throughput / percentiles. Those
numbers tell you *what* is slow but not *why*. The cause lives one layer below the
endpoint, in the Python serving loop (`LLMEngine.step` -> `Scheduler.schedule` ->
`ModelRunner.execute_model`): scheduling overhead, batch composition, KV-cache
pressure, prefill-vs-decode mix, and the host-vs-GPU split of each step.

This profiler instruments that loop. It is the right layer for the *first* drill
down from client metrics. It deliberately does NOT try to time individual CUDA
kernels inside the C++ engine -- that lives below another stream/thread boundary
and (with `--enable-graph`) inside a captured graph, so it belongs to a vendor
profiler (Nsight / msprof), not to Python instrumentation.

Design
------
- Opt-in and zero-cost when off: gated by `INFINILM_PROFILE_STEPS` (same env name
  as the upstream `feat/test-profiling` convention). When disabled, every method is
  an early-return no-op and no per-step state is gathered.
- One instance is shared by `LLMEngine` and its `ModelRunner` so sub-phase timings
  (build_inputs / forward / readback) and engine-phase timings (schedule / update)
  land in the same per-step record.
- Beyond upstream's *aggregate* counters, it keeps a *per-step trace* so the
  analyzer can compute ITL distributions (p50/p99), KV utilization over time, and
  batch-size distributions -- none of which a running sum can produce.
- Single writer: the engine step runs on one thread (offline loop, or the single
  `AsyncLLMEngineStepThread`), so no lock is needed.

Env vars
--------
INFINILM_PROFILE_STEPS   enable (1/true)                       [default off]
INFINILM_PROFILE_SYNC    sync_device() before timing forward   [default off]
                         -> accurate GPU forward time under async/graph; adds a
                            sync per step so leave off for throughput runs.
INFINILM_PROFILE_TRACE   path to write per-step JSONL trace    [default none]
INFINILM_PROFILE_MAX_STEPS  cap in-memory trace (ring)         [default 200000]
"""

from __future__ import annotations

import atexit
import json
import os
import statistics
import sys
import time


def _truthy(v: str) -> bool:
    return v not in ("", "0", "false", "False", "no", "off")


def _pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


class ServingProfiler:
    """Records per-step serving state + phase timings for the engine loop."""

    # Aggregate keys kept compatible with feat/test-profiling's summary lines.
    _TIME_KEYS = (
        "schedule_ms", "build_inputs_ms", "forward_ms", "to_list_ms", "update_ms",
        "prefill_forward_ms", "decode_forward_ms",
    )
    _COUNT_KEYS = (
        "steps", "prefill_steps", "decode_steps",
        "scheduled_requests", "scheduled_tokens",
    )

    def __init__(self, enabled=False, sync=False, trace_path=None, max_steps=200000):
        self.enabled = enabled
        self.sync = sync and enabled
        self.trace_path = trace_path if enabled else None
        self.max_steps = max_steps
        self.agg = {k: 0.0 for k in self._TIME_KEYS}
        self.agg.update({k: 0 for k in self._COUNT_KEYS})
        self.trace = []
        self._scratch = {}
        self._trace_fh = None
        self._flushed = False
        if self.enabled:
            if self.trace_path:
                try:
                    self._trace_fh = open(self.trace_path, "w", buffering=1)
                except OSError as e:  # don't let a bad path kill the server
                    print(f"[serving_profiler] cannot open {self.trace_path}: {e}", file=sys.stderr)
                    self._trace_fh = None

    @classmethod
    def from_env(cls):
        """Production entry point: builds from env and wires atexit flushing.

        Lifecycle (atexit) lives here rather than in __init__ so that bare
        construction (tests, ad-hoc use) yields a pure recorder with no global
        side effects.
        """
        enabled = _truthy(os.getenv("INFINILM_PROFILE_STEPS", "0"))
        prof = cls(
            enabled=enabled,
            sync=_truthy(os.getenv("INFINILM_PROFILE_SYNC", "0")),
            trace_path=os.getenv("INFINILM_PROFILE_TRACE") or None,
            max_steps=int(os.getenv("INFINILM_PROFILE_MAX_STEPS", "200000")),
        )
        if prof.enabled:
            atexit.register(prof.flush)
        return prof

    @classmethod
    def disabled(cls):
        return cls(enabled=False)

    # ---- per-step recording (called from the engine/runner hooks) ----

    def begin_step(self):
        if not self.enabled:
            return
        self._scratch = {k: 0.0 for k in ("schedule_ms", "build_inputs_ms",
                                          "forward_ms", "to_list_ms", "update_ms")}

    def mark(self, name, ms):
        """Record a sub-phase duration for the current step (ms)."""
        if not self.enabled:
            return
        self._scratch[name] = self._scratch.get(name, 0.0) + ms
        if name in self.agg:
            self.agg[name] += ms

    def record_step(self, *, phase, batch, num_tokens,
                    q_wait=None, q_run=None, kv_used=None, kv_total=None):
        """Finalize the current step into the trace + aggregates."""
        if not self.enabled:
            return
        s = self._scratch
        step_ms = (s.get("schedule_ms", 0.0) + s.get("build_inputs_ms", 0.0)
                   + s.get("forward_ms", 0.0) + s.get("to_list_ms", 0.0)
                   + s.get("update_ms", 0.0))
        rec = {
            "step": self.agg["steps"],
            "phase": phase,
            "batch": batch,
            "num_tokens": num_tokens,
            "q_wait": q_wait,
            "q_run": q_run,
            "kv_used": kv_used,
            "kv_total": kv_total,
            "kv_util": (kv_used / kv_total) if (kv_used is not None and kv_total) else None,
            "schedule_ms": round(s.get("schedule_ms", 0.0), 4),
            "build_inputs_ms": round(s.get("build_inputs_ms", 0.0), 4),
            "forward_ms": round(s.get("forward_ms", 0.0), 4),
            "to_list_ms": round(s.get("to_list_ms", 0.0), 4),
            "update_ms": round(s.get("update_ms", 0.0), 4),
            "step_ms": round(step_ms, 4),
        }
        # aggregates
        self.agg["steps"] += 1
        is_prefill = phase == "prefill"
        self.agg["prefill_steps" if is_prefill else "decode_steps"] += 1
        self.agg["scheduled_requests"] += batch
        self.agg["scheduled_tokens"] += num_tokens
        self.agg["prefill_forward_ms" if is_prefill else "decode_forward_ms"] += s.get("forward_ms", 0.0)

        if self._trace_fh is not None:
            self._trace_fh.write(json.dumps(rec) + "\n")
        if len(self.trace) < self.max_steps:
            self.trace.append(rec)

    # ---- summary / lifecycle ----

    def reset(self):
        for k in self._TIME_KEYS:
            self.agg[k] = 0.0
        for k in self._COUNT_KEYS:
            self.agg[k] = 0
        self.trace.clear()

    def _decode_itl(self):
        d = [r["step_ms"] for r in self.trace if r["phase"] == "decode"]
        df = [r["forward_ms"] for r in self.trace if r["phase"] == "decode"]
        return d, df

    def format_summary(self) -> str:
        if not self.enabled:
            return "[serving_profiler] disabled (set INFINILM_PROFILE_STEPS=1)"
        a = self.agg
        total = a["schedule_ms"] + a["build_inputs_ms"] + a["forward_ms"] + a["to_list_ms"] + a["update_ms"]
        lines = [
            "infinilm_profile "
            f"steps={a['steps']} prefill_steps={a['prefill_steps']} decode_steps={a['decode_steps']} "
            f"scheduled_requests={a['scheduled_requests']} scheduled_tokens={a['scheduled_tokens']} "
            f"total_ms={total:.2f}",
            "infinilm_profile_phase "
            f"schedule_ms={a['schedule_ms']:.2f} build_inputs_ms={a['build_inputs_ms']:.2f} "
            f"forward_ms={a['forward_ms']:.2f} to_list_ms={a['to_list_ms']:.2f} update_ms={a['update_ms']:.2f}",
        ]
        # host vs gpu split + decode ITL distribution (the value-add over aggregates)
        host_ms = a["schedule_ms"] + a["build_inputs_ms"] + a["to_list_ms"] + a["update_ms"]
        host_pct = 100.0 * host_ms / total if total else 0.0
        d, df = self._decode_itl()
        if d:
            lines.append(
                "infinilm_profile_decode_itl "
                f"n={len(d)} mean_ms={statistics.fmean(d):.3f} p50_ms={_pct(d,50):.3f} "
                f"p99_ms={_pct(d,99):.3f} max_ms={max(d):.3f} "
                f"forward_mean_ms={statistics.fmean(df):.3f}")
        kv = [r["kv_util"] for r in self.trace if r["kv_util"] is not None]
        bat = [r["batch"] for r in self.trace if r["phase"] == "decode"]
        lines.append(
            "infinilm_profile_serving "
            f"host_overhead_pct={host_pct:.1f} "
            + (f"kv_util_mean={statistics.fmean(kv):.3f} kv_util_max={max(kv):.3f} " if kv else "")
            + (f"decode_batch_mean={statistics.fmean(bat):.2f} decode_batch_max={max(bat)}" if bat else ""))
        return "\n".join(lines)

    def flush(self):
        if not self.enabled or self._flushed:
            return
        self._flushed = True
        try:
            if self.agg["steps"]:
                print(self.format_summary(), file=sys.stderr)
        finally:
            if self._trace_fh is not None:
                try:
                    self._trace_fh.close()
                except OSError:
                    pass
                self._trace_fh = None
