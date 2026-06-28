"""Unit tests for the serving-layer profiler (ServingProfiler + analyze_trace).

These modules are dependency-free by design, so this test loads them straight from
source and runs without the full infinilm package (which pulls infinicore/_infinilm).
On a built environment `from infinilm.profiler.serving_profiler import ServingProfiler`
works too; loading by path keeps the unit test runnable on any box (incl. CI without
a device).

    pytest test/profiler/test_serving_profiler.py
"""

import importlib.util
import json
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parents[2] / "python" / "infinilm" / "profiler"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_prof_{name}", _PKG / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sp = _load("serving_profiler")
at = _load("analyze_trace")
ServingProfiler = sp.ServingProfiler


def _decode_step(prof, fwd_ms, kv_used=10, kv_total=100, batch=1):
    prof.begin_step()
    prof.mark("schedule_ms", 0.2)
    prof.mark("build_inputs_ms", 0.5)
    prof.mark("forward_ms", fwd_ms)
    prof.mark("to_list_ms", 0.1)
    prof.mark("update_ms", 0.3)
    prof.record_step(phase="decode", batch=batch, num_tokens=batch,
                     q_wait=0, q_run=batch, kv_used=kv_used, kv_total=kv_total)


# --------------------------------------------------------------------------- off

def test_disabled_is_zero_cost_noop():
    prof = ServingProfiler.disabled()
    assert prof.enabled is False
    prof.begin_step()
    prof.mark("forward_ms", 999.0)
    prof.record_step(phase="decode", batch=1, num_tokens=1)
    assert prof.trace == []
    assert prof.agg["steps"] == 0
    assert "disabled" in prof.format_summary()


# ----------------------------------------------------------------------- record

def test_records_aggregates_and_trace():
    prof = ServingProfiler(enabled=True)
    # one prefill
    prof.begin_step()
    prof.mark("forward_ms", 140.0)
    prof.record_step(phase="prefill", batch=1, num_tokens=2048,
                     kv_used=8, kv_total=6144)
    # ten decode
    for _ in range(10):
        _decode_step(prof, 25.0, kv_total=6144)

    assert prof.agg["steps"] == 11
    assert prof.agg["prefill_steps"] == 1
    assert prof.agg["decode_steps"] == 10
    assert prof.agg["scheduled_tokens"] == 2048 + 10  # prefill tokens + decode 1/step
    # forward time split by phase
    assert prof.agg["prefill_forward_ms"] == pytest.approx(140.0)
    assert prof.agg["decode_forward_ms"] == pytest.approx(250.0)
    assert len(prof.trace) == 11
    # summary carries the extended distribution line
    summary = prof.format_summary()
    assert "infinilm_profile_decode_itl" in summary
    assert "infinilm_profile_serving" in summary


def test_step_ms_is_sum_of_subphases():
    prof = ServingProfiler(enabled=True)
    _decode_step(prof, 25.0)
    rec = prof.trace[-1]
    assert rec["step_ms"] == pytest.approx(0.2 + 0.5 + 25.0 + 0.1 + 0.3)
    assert rec["forward_ms"] == pytest.approx(25.0)


def test_kv_util_computed():
    prof = ServingProfiler(enabled=True)
    _decode_step(prof, 25.0, kv_used=30, kv_total=120)
    assert prof.trace[-1]["kv_util"] == pytest.approx(0.25)


def test_kv_util_none_when_total_missing():
    prof = ServingProfiler(enabled=True)
    prof.begin_step()
    prof.mark("forward_ms", 25.0)
    prof.record_step(phase="decode", batch=1, num_tokens=1, kv_used=None, kv_total=None)
    assert prof.trace[-1]["kv_util"] is None


def test_max_steps_caps_in_memory_trace_but_counts_continue():
    prof = ServingProfiler(enabled=True, max_steps=3)
    for _ in range(10):
        _decode_step(prof, 25.0)
    assert len(prof.trace) == 3          # ring cap
    assert prof.agg["steps"] == 10       # aggregates keep counting


# ------------------------------------------------------------------- jsonl trace

def test_jsonl_trace_written(tmp_path):
    path = tmp_path / "trace.jsonl"
    prof = ServingProfiler(enabled=True, trace_path=str(path))
    prof.begin_step(); prof.mark("forward_ms", 140.0)
    prof.record_step(phase="prefill", batch=1, num_tokens=512, kv_used=2, kv_total=100)
    for _ in range(5):
        _decode_step(prof, 25.0)
    prof.flush()

    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 6
    assert lines[0]["phase"] == "prefill"
    assert all(set(("step", "phase", "forward_ms", "step_ms")) <= r.keys() for r in lines)


def test_from_env(monkeypatch):
    monkeypatch.setenv("INFINILM_PROFILE_STEPS", "1")
    monkeypatch.setenv("INFINILM_PROFILE_SYNC", "1")
    monkeypatch.delenv("INFINILM_PROFILE_TRACE", raising=False)
    prof = ServingProfiler.from_env()
    assert prof.enabled is True
    assert prof.sync is True

    monkeypatch.setenv("INFINILM_PROFILE_STEPS", "0")
    assert ServingProfiler.from_env().enabled is False


# -------------------------------------------------------------------- analyzer

def _make_rows():
    prof = ServingProfiler(enabled=True)
    prof.begin_step(); prof.mark("forward_ms", 140.0)
    prof.record_step(phase="prefill", batch=1, num_tokens=2048, kv_used=8, kv_total=6144)
    for i in range(20):
        _decode_step(prof, 25.0 + (20.0 if i == 5 else 0.0), kv_used=8 + i, kv_total=6144)
    return prof.trace


def test_analyze_distributions_and_throughput():
    a = at.analyze(_make_rows())
    assert a["steps"] == {"total": 21, "prefill": 1, "decode": 20}
    itl = a["decode_itl_ms"]
    assert itl["n"] == 20
    assert itl["max"] >= itl["p99"] >= itl["p50"]   # ordering
    assert itl["max"] > itl["mean"]                  # the injected spike shows up
    # forward dominates -> host overhead small
    assert a["decode_host_pct"] < 10
    assert a["decode_throughput_tok_s"] > 0
    assert a["decode_phase_breakdown"]["forward_ms"]["pct"] > 90


def test_correlate_bench_gap_is_above_engine():
    a = at.analyze(_make_rows())
    bench = {"mean_itl_ms": 31.5, "mean_ttft_ms": 210.0}
    corr = at.correlate_bench(a, bench)
    # client ITL above engine step time -> positive gap (queueing/network/SSE)
    assert corr["itl"]["above_engine_ms"] == pytest.approx(31.5 - a["decode_itl_ms"]["mean"])
    assert 0 < corr["itl"]["above_engine_pct"] < 100
    assert corr["ttft"]["client_ms"] == 210.0


def test_percentile_helper_edges():
    assert at._pct([], 50) != at._pct([], 50) or True  # nan-safe call doesn't raise
    assert at._pct([5], 99) == 5
    assert at._pct([1, 2, 3, 4], 50) in (2, 3)
