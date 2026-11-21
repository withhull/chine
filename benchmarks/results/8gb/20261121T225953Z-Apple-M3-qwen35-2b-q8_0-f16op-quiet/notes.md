# Qwen3.5-2B Q8_0 at 2d03884, quiet AC run (3 GiB budget, metal-f16-act)

- Command: `MEMORY_BUDGET=3GiB SKIP_LLAMA_BENCH=1
  scripts/bench-baseline.sh artifacts-local/Qwen3.5-2B-Q8_0.gguf
  qwen35-2b-q8_0-f16op-quiet` (ROUNDS=3), run from a Terminal with the
  Claude app closed, on AC power (recorded in `environment.json`).
- Tree clean at `2d03884` — the first recorded run of the
  `metal-f16-act` GEMM operand profile as the engine default (ADR-011
  engine note 2026-11-22): batched prefill GEMMs stage operands as half
  with f32 accumulation. The TuckCore records carry
  `Q35-2C:qwen35-q8_0+f16:metal:f16op`.
- Reclaimable 3.1-4.1 GiB before each invocation; no swap growth in any
  round; all 6 invocations valid on the first attempt. Each engine's
  default KV precision (f16 for both).

## Results (locally-calibrated)

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| tuckcore-metal | 3 | 601 ms | 36.54 | 39.82 | 2148 MiB |
| llama-metal | 3 | 587 ms | 37.28 | 40.60 | 248 MiB |

- Paired decode: median **0.980** (0.982, 0.980, 0.980).
- Paired TTFT: median **1.02** — prefill **98%** of llama.cpp, up from
  94% (the two AC runs with f32 operands) and 86% (the quiet battery
  run). Both engines were very steady this run (TuckCore 600-602 ms,
  llama.cpp 587-588 ms).
- Cold-load TTFT: TuckCore 1.75-1.83 s, llama.cpp 0.96-1.57 s.

## Reading

- The metal-f16-act profile moved the 2B prefill from 94% to **98%**,
  which is what the kernel-level measurements predicted (f16-operand
  GEMM 2-3.5% faster at the 2B shapes, GEMM ~78-79% of prefill). The
  Phase 1 prefill bar (>= 90%) is **met on this model in this run**;
  a second run would make it "met and reproduced" by the repo's
  convention.
- Decode stays at parity: paired 0.956-1.030 across the four runs.
- Footprint (2,148 vs 248 MiB) is not like-for-like: llama.cpp's
  memory-mapped weights are excluded from its footprint.
