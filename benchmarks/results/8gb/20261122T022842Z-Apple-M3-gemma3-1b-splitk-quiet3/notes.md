# Gemma 3 1B Q8_0 at 118a711, quiet AC run (split-K decode attention)

- Command: `KV_PRECISION=f16 IGNORE_EOS=1 SKIP_LLAMA_BENCH=1 ROUNDS=3
  scripts/bench-baseline.sh artifacts-local/gemma-3-1b-it-Q8_0.gguf
  gemma3-1b-splitk-quiet3` (default ROUNDS), run from Terminal with the
  Hull app closed, AC power (recorded in `environment.json`).
- Tree clean at `118a711` — the acceptance run for the split-K decode
  attention (ADR-011 note 2026-11-22): n = 1 attention dispatches up to
  16 slice groups per (KV head, row) writing [m, l, acc] partials, then a
  merge kernel folds them; prefill keeps the one-group-per-row kernel, so
  the win is decode-only by construction. The TuckCore records carry
  `ADR013:gemma3-q8_0+f16:metal:f16op`.
- Reclaimable 3.5-4.3 GiB before each invocation; zero swap growth; all
  6 invocations valid on the first attempt.
- Two earlier runs at the same code (`…-splitk-quiet`,
  `…-splitk-quiet2`) were measured with the machine in active use
  (load average 2.3-3.4, both engines ~20-25% below their quiet
  baselines, llama.cpp hit harder: decode medians 57.65/58.37 vs
  55.81/56.69 tok/s). They are recorded as load-contamination evidence —
  paired decode 1.033/1.031, TuckCore ahead in all 6 pairs — not as the
  acceptance number.

## Results (locally-calibrated)

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| tuckcore-metal | 3 | 324 ms | 68.07 | 74.21 | 1229 MiB |
| llama-metal | 3 | 290 ms | 64.68 | 69.58 | 199 MiB |

- Paired TTFT: median **1.12x** — prefill ~89% of llama.cpp (unchanged by
  design: the split-K serves n = 1 only, so the prefill kernels and their
  cost are the pre-split-K ones; 325/324/324 vs the 326/326/325 ms of the
  fusednorm acceptance run — within its ± 0.2 band once round 1's
  cold-load tail is excluded).
- Paired decode: median **1.067** (0.977 / 1.197 / 1.067). TuckCore
  74.13/74.21/74.22 tok/s — the same ± 0.2 steadiness the 65.0 baseline
  had, now 14.2% higher (65.0 -> 74.2, the projected ~71.5 exceeded);
  llama.cpp swung 75.85/62.00/69.58 (its recorded bimodality, 62-76
  tok/s). Against llama.cpp's FASTEST round the pair is 0.977; the
  >= 90% decode bar is met in every pair.

## Reading — the 65-vs-76 gap is closed at the bench level

- Decode attention was the measured gap (2.77-2.89 ms/token, one
  128-thread group per row at ~5.4 GB/s); the split-K pair cut it to
  1.12-1.16 ms/token in-engine, and the request-level number moves
  exactly as that arithmetic says: 65.0 -> 74.2 tok/s, +14.2%.
- For the first time in this harness on Gemma 3, TuckCore's decode is
  paired AHEAD of llama.cpp's median (1.067), and at 0.977 against its
  fastest round. Decode parity-or-ahead now holds on every measured model
  (0.8B: 0.99-1.26; 2B: 0.95-1.03; Gemma 3: 0.977-1.197).
- Prefill is unchanged at ~89% (324 vs 290 ms, paired 1.12) — the >= 90%
  prefill bar remains the open Gemma item, unchanged by this work and
  bounded by the earlier ablation at ~1.5% of prefill.
- Footprint peak 1229 MiB (+2 MiB vs the fusednorm run — the decode
  split-K partials scratch; prefill unchanged).
