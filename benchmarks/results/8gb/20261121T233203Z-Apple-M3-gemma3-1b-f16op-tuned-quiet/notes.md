# Gemma 3 1B Q8_0 at 5f242d0, quiet battery run (tuned edge stage)

- Command: `KV_PRECISION=f16 IGNORE_EOS=1 SKIP_LLAMA_BENCH=1
  scripts/bench-baseline.sh artifacts-local/gemma-3-1b-it-Q8_0.gguf
  gemma3-1b-f16op-tuned-quiet` (ROUNDS=3), run from a Terminal with the
  Claude app closed, on battery power (recorded in `environment.json`).
- Tree clean at `5f242d0` — first recorded run of the tuned edge stage
  (the f16op kernel's 8 KB static whole-tile stage replaced by
  per-SIMD-group 256 B stages; ADR-011 engine note 2026-11-22). The
  TuckCore records carry `ADR013:gemma3-q8_0+f16:metal:f16op`.
- Reclaimable 3.5-4.3 GiB before each invocation; no swap growth; all 6
  invocations valid on the first attempt.

## Results (locally-calibrated)

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| tuckcore-metal | 3 | 326 ms | 60.29 | 65.06 | 1227 MiB |
| llama-metal | 3 | 289 ms | 69.93 | 75.64 | 200 MiB |

- Paired TTFT: median **1.13x** — prefill **~88.5%** of llama.cpp
  (round pairs 1.128 / 1.128 / 1.062; llama's round 3 ran slow at
  306 ms TTFT and 62.5 tok/s decode). The >= 90% prefill bar is **not
  crossed** by the median.
- Paired decode: median **0.860** (0.852 / 0.860 / 1.042). TuckCore
  steady at 64.73-65.13 tok/s across every run ever recorded (65.0
  +/- 0.2); llama.cpp swings 62.5-76.1 across runs, and that swing is
  what moves the ratio.

## Reading

- The tuned kernel's request-level effect is real but small: warm TTFT
  330 -> 326 ms (-1.2%) versus the previous quiet run. That matches the
  same-window kernel arithmetic (FFN shapes -3.4%, GEMM stack -2.65%,
  GEMM 79% of prefill -> ~-2%); it does not reproduce the -9.5% quoted
  from in-process profiles, which compared different machine-load
  windows and was load-contaminated. This run is the acceptance record
  and it says: the GEMM work closed the FFN kernel gap (5% -> 1.8-1.9%
  behind llama in the same window) but that was worth ~1-2% of request
  prefill, not ~10%.
- With the GEMM stack now at 1.6-1.9% behind llama, the remaining ~11%
  request-level gap is dominated by the non-GEMM stack and host
  dispatch: per the 512-token profile, gelu_tanh_mul 5.6%,
  attention chunk 5.6%, rmsnorm 3.8%, axpy 3.3%, conv/recurrence ~5% —
  ~21% of prefill where llama.cpp (and exo's Flint work) run fused
  stacked kernels and fewer launches. The next recorded lever is fused
  GeGLU/norm kernels, not further GEMM tuning.
- Battery power: TTFT reproduced the AC run's values (326 vs 330 ms),
  so power state is not driving the ratio.
