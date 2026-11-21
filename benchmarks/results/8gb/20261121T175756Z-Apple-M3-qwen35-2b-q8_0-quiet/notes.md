# Qwen3.5-2B Q8_0 at 546531a, quiet run (3 GiB budget)

- Command: `MEMORY_BUDGET=3GiB SKIP_LLAMA_BENCH=1 scripts/bench-baseline.sh
  artifacts-local/Qwen3.5-2B-Q8_0.gguf` (ROUNDS=3), run from a Terminal
  with the Claude app closed.
- MacBook Air M3 8 GB, macOS 27.0.1 (26A434), **battery power** (recorded
  in `environment.json`; the earlier runs at this model were on AC).
  Reclaimable memory 2.5-3.9 GiB before each invocation; no swap growth in
  any round; all 6 invocations valid on the first attempt.
- Artifact: Qwen3.5-2B-Q8_0.gguf, sha256 `1b04acba…21f2c1`, 3 GiB budget
  for both engines, each engine's default KV precision (f16 for both).

## Results (locally-calibrated)

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| tuckcore-metal | 3 | 829 ms | 34.30 | 38.35 | 2148 MiB |
| llama-metal | 3 | 610 ms | 36.85 | 40.24 | 249 MiB |

- Paired decode: median **0.956** (0.956, 0.949, 1.020).
- Paired TTFT: median **1.16** (1.07, 1.38, 1.16).
- Cold-load TTFT: TuckCore 1.75-2.27 s, llama.cpp 0.98-2.19 s.

## Reading

- **Round 3 slowed both engines together** (TuckCore TTFT 1.257 s, decode
  32.3 tok/s; llama.cpp 1.086 s, 31.7 tok/s), a machine-wide slowdown
  (battery power / other load), not one engine. The paired ratios in that
  round are still consistent (decode 1.020, TTFT 1.16).
- **Decode** is at parity: paired medians 0.956 here, 0.971 and 1.030 in
  the two AC runs (0.95-1.03 per pair across three runs).
- **Prefill** is 86-94% of llama.cpp across runs (paired medians 1.06,
  1.06, 1.16), so the Phase 1 prefill bar (>= 90%) is **not consistently
  met** on this model. Round 2 here was the weakest pair (1.38).
- Footprint (2,148 vs 249 MiB) is not like-for-like: llama.cpp's
  memory-mapped weights are excluded from its footprint.
