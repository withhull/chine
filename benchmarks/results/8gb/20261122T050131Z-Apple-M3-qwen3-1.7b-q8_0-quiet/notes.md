# Qwen3-1.7B Q8_0 at 2a21431, quiet AC run (acceptance)

- Command: `KV_PRECISION=f16 IGNORE_EOS=1 SKIP_LLAMA_BENCH=1 ROUNDS=3
  MEMORY_BUDGET=4GiB scripts/bench-baseline.sh
  artifacts-local/Qwen3-1.7B-Q8_0.gguf qwen3-1.7b-q8_0-quiet`, run from
  Terminal with the Hull app closed, AC power, Mac untouched. Zero swap
  growth; all 6 invocations valid; reclaimable 3.0-4.1 GiB.
- The acceptance run for the fourth model (dense qwen3 family one size
  up: 28 layers, hidden 2048, head dim 128, 8 KV heads; verify
  --reference PASS at max |Δ| 3.189e-1 under the registered 5e-1 Q8_0
  policy, top-1 100%, texts equal — see the compatibility record
  `20261122T044954Z-…-qwen3-1.7b-q8_0-loaded` and the claims row).

## Results (locally-calibrated)

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| chinecore-metal | 3 | 588 ms | 40.89 | 44.93 | 2233 MiB |
| llama-metal | 3 | 559 ms | 41.44 | 45.38 | 380 MiB |

- Paired decode: median **0.990** (0.991 / 0.989 / 0.990) — the
  tightest paired band recorded in this harness on any model: both
  engines steady to within ±0.05 tok/s (ChineCore 44.88-44.96,
  llama.cpp 45.37-45.39, no bimodality this run). Decode at 99%, the
  >= 90% bar met in every pair.
- Paired TTFT: median **1.05x** — prefill **~95%** (585-589 vs
  559 ms), the >= 90% bar met.
- Footprint: 2233 MiB resident (weights 1.83 GiB + KV + scratch) vs
  llama.cpp's 380 MiB peak (mmap; not like-for-like, as recorded for
  every model).

## Reading

- Both Phase 1 bars are met on the fourth model with zero engine
  changes: the dense family scales one size up (hidden 1024 -> 2048,
  bigger FFN, new GQA geometry) and lands at decode parity and 95%
  prefill, matching the 0.8B/2B pattern.
- The earlier loaded run (`…-qwen3-1.7b-q8_0-loaded`, same code) showed
  paired 1.091 decode / 0.92 TTFT under active-session load — recorded
  as the load asymmetry (llama.cpp's host loop degrades more), not a
  lead; the quiet numbers are the acceptance record.
- The 1.7B's decode at 4 GiB resident is bandwidth-bound as expected:
  ~45 tok/s against ~2.0 GiB of per-token weight traffic on a 100
  GB/s part; both engines sit at the same wall, which is why the pairs
  are so tight.
