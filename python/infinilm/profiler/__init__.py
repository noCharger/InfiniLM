"""Serving-layer profiling for the InfiniLM engine loop.

Opt-in via `INFINILM_PROFILE_STEPS=1`. See `serving_profiler.ServingProfiler` for
the env vars and design. Offline analysis lives in `analyze_trace.py`.
"""

from infinilm.profiler.serving_profiler import ServingProfiler

__all__ = ["ServingProfiler"]
