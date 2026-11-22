# Gemma 3 1B Q8_0 at 6abf2a8, quiet battery run (fused norm kernels)

- Command: `KV_PRECISION=f16 IGNORE_EOS=1 SKIP_LLAMA_BENCH=1
  scripts/bench-baseline.sh artifacts-local/gemma-3-1b-it-Q8_0.gguf
  gemma3-1b-fusednorm-quiet` (ROUNDS=3), run from a Terminal with the
  Claude app closed, on battery power (recorded in `environment.json`).
- Tree clean at `6abf2a8` — the acceptance run for the fused post-block
  norm + residual add and the merged Q/K norms (ADR-011 note
  2026-11-22), both bit-identical to what they replace. The TuckCore
  records carry `ADR013:gemma3-q8_0+f16:metal:f16op`.
- Reclaimable 3.9-4.3 GiB before each invocation; no swap growth; all 6
  invocations valid on the first attempt.

## Results (locally-calibrated)

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| tuckcore-metal | 3 | 326 ms | 60.50 | 65.29 | 1227 MiB |
| llama-metal | 3 | 289 ms | 70.02 | 75.75 | 199 MiB |

- Paired TTFT: median **1.13x** — prefill **~88.5%** of llama.cpp (pairs
  1.128 / 1.128 / 1.126). The >= 90% bar is **not crossed**.
- Paired decode: median **0.862** (0.858 / 0.862 / 0.864). TuckCore
  steady 65.21-65.40 tok/s; llama.cpp 75.70-76.04 this run (its fast
  mode, as in the f16op-quiet runs).

## Reading — the fusion did not move the bench number

- Warm TTFT is **326/326/325 ms, identical to the run before the
  fusion** (330 -> 326 across the edge-stage tuning; fused norms: no
  further change). This is the second projection to fail at the bench
  level, and the finding is recorded in the ADR-011 note: deltas measured
  in the UNGROUPED profiling regime do not translate to the grouped
  production encoder. The ungrouped norm stack (~30 ms -> ~10 ms) was
  dominated by per-dispatch overhead that the grouped encoder never
  pays; the fused kernels' true production saving is the launch/encode
  count (208 fewer per prefill), worth ~1 ms, not the ~6-7 ms the
  kernel arithmetic projected.
- The kernels are kept: they are bit-identical, strictly less work per
  row, fewer launches, and they do shrink the real (grouped) cost
  slightly — but they are not the request-level lever.
- With GEMM at 1.6-1.9% behind llama and the elementwise norms now
  minimal, the remaining ~11.5% TTFT gap (326 vs 289 ms) must live in
  the grouped regime's other operators and host dispatch: the chunk
  attention kernel and GeGLU (each ~24 ms ungrouped), plus encoder
  overhead. The recorded next step is a GROUPED-regime profile (the
  current profiler measures ungrouped), before any further kernel work
  — no more projections from ungrouped deltas.
