# Gemma 3 1B Q8_0 at 2d03884, quiet AC run (metal-f16-act)

- Command: `KV_PRECISION=f16 IGNORE_EOS=1 SKIP_LLAMA_BENCH=1
  scripts/bench-baseline.sh artifacts-local/gemma-3-1b-it-Q8_0.gguf
  gemma3-1b-f16op-quiet` (ROUNDS=3), run from a Terminal with the Claude
  app closed, on AC power (recorded in `environment.json`).
- Tree clean at `2d03884` — first recorded run with the metal-f16-act
  GEMM operand profile as the engine default (ADR-011 engine note
  2026-11-22). The TuckCore records carry
  `ADR013:gemma3-q8_0+f16:metal:f16op`. f16 KV on both engines;
  ignore-EOS workload (ADR-012) so decode is measurable.
- Reclaimable 3.6-4.3 GiB before each invocation; no swap growth; all 6
  invocations valid on the first attempt.

## Results (locally-calibrated)

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| tuckcore-metal | 3 | 330 ms | 60.22 | 65.04 | 1227 MiB |
| llama-metal | 3 | 290 ms | 70.14 | 75.89 | 199 MiB |

- Paired decode: median **0.857** (0.857, 0.856, 0.857).
- Paired TTFT: median **1.14** — prefill ~88%.

## Reading

- llama.cpp ran its fastest recorded decode this run (75.85-76.09
  tok/s; across the three Gemma runs it swings 61.4-76.1), which alone
  moved the paired ratio; TuckCore's own decode is unchanged and
  steady as in every run (64.90-65.19 tok/s, median 65.04). The
  profile does not touch decode (per-token GEMVs).
- Prefill ~88%: within the recorded 85-90% band. The f16-operand
  kernel's residual at the Gemma FFN shapes (6912x1152 / 1152x6912,
  recorded 5% behind llama at f16 operands) keeps Gemma prefill just
  below the bar; that residual is the recorded follow-up.
- No Gemma-specific tuning has been done.
