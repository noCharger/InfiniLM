#!/usr/bin/env python3
"""Drill `vllm bench serve` client metrics down into the serving loop.

Reads the per-step JSONL trace written by ServingProfiler (INFINILM_PROFILE_TRACE)
and reports the decode ITL / prefill TTFT distributions split into GPU-forward vs
host overhead, plus KV utilization and batch occupancy over the run. Optionally
ingests a `vllm bench serve --save-result` JSON to line up CLIENT-observed numbers
against ENGINE-observed ones -- the gap is queueing + network + SSE, i.e. the part
that lives above the engine.

    python -m infinilm.profiler.analyze_trace trace.jsonl
    python -m infinilm.profiler.analyze_trace trace.jsonl --bench result.json --out report.json
"""

import argparse
import json
import statistics
import sys


def _pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def _dist(xs):
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "p50": _pct(xs, 50),
        "p99": _pct(xs, 99),
        "max": max(xs),
    }


def load_trace(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def analyze(rows):
    decode = [r for r in rows if r["phase"] == "decode"]
    prefill = [r for r in rows if r["phase"] == "prefill"]

    def phase_breakdown(rs):
        keys = ("schedule_ms", "build_inputs_ms", "forward_ms", "to_list_ms", "update_ms")
        tot = {k: sum(r.get(k, 0.0) for r in rs) for k in keys}
        grand = sum(tot.values()) or 1.0
        return {k: {"ms": tot[k], "pct": 100.0 * tot[k] / grand} for k in keys}

    d_step = [r["step_ms"] for r in decode]
    d_fwd = [r["forward_ms"] for r in decode]
    d_host = [r["step_ms"] - r["forward_ms"] for r in decode]
    p_step = [r["step_ms"] for r in prefill]
    kv = [r["kv_util"] for r in rows if r.get("kv_util") is not None]
    d_batch = [r["batch"] for r in decode]

    decode_tokens = sum(r["num_tokens"] for r in decode)
    decode_secs = sum(d_step) / 1000.0
    return {
        "steps": {"total": len(rows), "prefill": len(prefill), "decode": len(decode)},
        "decode_itl_ms": _dist(d_step),
        "decode_forward_ms": _dist(d_fwd),
        "decode_host_ms": _dist(d_host),
        "decode_host_pct": (100.0 * sum(d_host) / sum(d_step)) if d_step else None,
        "prefill_step_ms": _dist(p_step),
        "decode_phase_breakdown": phase_breakdown(decode),
        "prefill_phase_breakdown": phase_breakdown(prefill),
        "kv_util": {"mean": statistics.fmean(kv), "max": max(kv)} if kv else None,
        "decode_batch": {"mean": statistics.fmean(d_batch), "max": max(d_batch)} if d_batch else None,
        "decode_throughput_tok_s": (decode_tokens / decode_secs) if decode_secs else None,
    }


def correlate_bench(a, bench):
    """Line up client (vllm bench serve) vs engine numbers. Gap = above-engine."""
    def g(*names):
        for n in names:
            if n in bench and bench[n] is not None:
                return bench[n]
        return None

    out = {}
    cli_itl = g("mean_itl_ms", "median_itl_ms")
    eng_itl = a["decode_itl_ms"].get("mean")
    if cli_itl and eng_itl:
        out["itl"] = {"client_ms": cli_itl, "engine_step_ms": eng_itl,
                      "above_engine_ms": cli_itl - eng_itl,
                      "above_engine_pct": 100.0 * (cli_itl - eng_itl) / cli_itl}
    cli_ttft = g("mean_ttft_ms", "median_ttft_ms")
    eng_pf = a["prefill_step_ms"].get("mean")
    if cli_ttft and eng_pf:
        out["ttft"] = {"client_ms": cli_ttft, "engine_prefill_ms": eng_pf,
                       "above_engine_ms": cli_ttft - eng_pf,
                       "above_engine_pct": 100.0 * (cli_ttft - eng_pf) / cli_ttft}
    return out


def hints(a, corr):
    h = []
    hp = a.get("decode_host_pct")
    if hp is not None and hp > 15:
        h.append(f"[host {hp:.0f}% of decode step] Python 调度/detokenize/KV 管理占比高 → "
                 "看 decode_phase_breakdown 里 schedule/update/to_list 哪个大;考虑 detokenize 批量化、"
                 "调度路径精简。GPU 已不是这部分的瓶颈。")
    bd = a.get("decode_phase_breakdown", {})
    big = max(bd.items(), key=lambda kv: kv[1]["pct"], default=None)
    if big and big[0] != "forward_ms" and big[1]["pct"] > 25:
        h.append(f"[{big[0]} 占 decode {big[1]['pct']:.0f}%] 非 forward 的单段就吃掉 1/4+,是 host 侧首要优化点。")
    kv = a.get("kv_util")
    if kv and kv["max"] > 0.9:
        h.append(f"[KV 峰值利用 {kv['max']*100:.0f}%] 接近满 → 可能触发 preempt/recompute,P99 尖刺多半来自这;"
                 "加 num_blocks 或减 max_cache_len。")
    db = a.get("decode_batch")
    if db and db["mean"] < 2:
        h.append(f"[decode batch 均值 {db['mean']:.1f}] 并发很低,单请求 decode 受访存限,"
                 "吞吐主要靠连续批处理把并发顶上去——先压并发,再谈 kernel。")
    if corr.get("itl") and corr["itl"]["above_engine_pct"] > 30:
        h.append(f"[客户端 ITL 比引擎 step 高 {corr['itl']['above_engine_pct']:.0f}%] 大头在引擎之上:"
                 "排队/网络/SSE。若并发不高,多半是 step 之间的调度间隙或 asyncio 回传——往服务框架查,别往 kernel 查。")
    if not h:
        h.append("未触发阈值:host 占比低、KV 不紧、并发/对齐正常。要再深需到 kernel 层(Nsight/msprof)。")
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="ServingProfiler 写出的 JSONL(INFINILM_PROFILE_TRACE)")
    ap.add_argument("--bench", help="vllm bench serve --save-result 的 JSON(可选,做客户端对齐)")
    ap.add_argument("--out", help="把完整结果写成 JSON")
    args = ap.parse_args()

    rows = load_trace(args.trace)
    if not rows:
        sys.exit(f"空 trace: {args.trace}")
    a = analyze(rows)
    corr = {}
    if args.bench:
        with open(args.bench) as f:
            bench = json.load(f)
        corr = correlate_bench(a, bench)
    h = hints(a, corr)

    _print(a, corr, h)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"analysis": a, "correlation": corr, "hints": h}, f, indent=2, ensure_ascii=False)
        print(f"\n已写出 {args.out}")


def _print(a, corr, h):
    line = "─" * 70
    s = a["steps"]
    print(f"{line}\n服务层下钻  (steps: {s['total']} = prefill {s['prefill']} + decode {s['decode']})\n{line}")
    itl, fwd = a["decode_itl_ms"], a["decode_forward_ms"]
    if itl.get("n"):
        print(f"  decode ITL (step)   : mean {itl['mean']:.2f} / p50 {itl['p50']:.2f} / "
              f"p99 {itl['p99']:.2f} / max {itl['max']:.2f} ms")
        print(f"    └ GPU forward     : mean {fwd['mean']:.2f} ms   host 开销 {a['decode_host_pct']:.0f}%")
    pf = a["prefill_step_ms"]
    if pf.get("n"):
        print(f"  prefill step (TTFT~): mean {pf['mean']:.2f} / p99 {pf['p99']:.2f} ms")
    if a.get("decode_throughput_tok_s"):
        print(f"  decode 吞吐          : {a['decode_throughput_tok_s']:.1f} tok/s")
    if a.get("kv_util"):
        print(f"  KV 利用率           : mean {a['kv_util']['mean']*100:.0f}% / max {a['kv_util']['max']*100:.0f}%")
    if a.get("decode_batch"):
        print(f"  decode batch        : mean {a['decode_batch']['mean']:.2f} / max {a['decode_batch']['max']}")

    print(f"\n  decode 每步分解(host 侧分项):")
    for k, v in sorted(a["decode_phase_breakdown"].items(), key=lambda kv: -kv[1]["pct"]):
        bar = "█" * int(v["pct"] / 4)
        print(f"    {k:<16} {v['pct']:>5.1f}%  {bar}")

    if corr:
        print(f"\n  客户端 vs 引擎(gap = 引擎之上:排队/网络/SSE):")
        for k, v in corr.items():
            print(f"    {k.upper():<5} client {v['client_ms']:.2f} ms  vs  engine "
                  f"{v.get('engine_step_ms', v.get('engine_prefill_ms')):.2f} ms  "
                  f"→ 上层 {v['above_engine_ms']:.2f} ms ({v['above_engine_pct']:.0f}%)")

    print(f"\n{line}\n💡 下钻结论\n{line}")
    for x in h:
        print(f"  • {x}")
    print(line)


if __name__ == "__main__":
    main()
