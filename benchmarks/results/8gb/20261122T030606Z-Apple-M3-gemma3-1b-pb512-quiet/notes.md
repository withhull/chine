# Gemma 3 1B Q8_0 at 1837737, quiet AC run (prefill chunks of 512 tokens)

- Command: `KV_PRECISION=f16 IGNORE_EOS=1 SKIP_LLAMA_BENCH=1 ROUNDS=3
  scripts/bench-baseline.sh artifacts-local/gemma-3-1b-it-Q8_0.gguf
  gemma3-1b-pb512-quiet`, run from Terminal with the Hull app closed, AC
  power (recorded in `environment.json`), Mac untouched.
- Tree clean at `1837737` (the code is the one-line `PREFILL_BATCH` 128
  -> 512 at `70d8701`; the two later commits are docs-only). The
  acceptance run for the prefill M-scaling lever (ADR-011 note
  2026-11-22): llama.cpp's request runs its prefill GEMMs at the full
  512-row batch; our four 128-row chunks paid the M-scaling difference
  at every Gemma shape (isolated, same window: llama +5.4% per token on
  FFN up, +25% on FFN down, +29-77% on the small attention shapes from
  128 to 512 rows; interleaved in-model A/B 4x128 vs 1x512 on the loaded
  machine: 384.4 vs 375.6 ms). The TuckCore records carry
  `ADR013:gemma3-q8_0+f16:metal:f16op`.
- Reclaimable 3.8-4.3 GiB before each invocation; zero swap growth; all
  6 invocations valid on the first attempt.

## Results (locally-calibrated)

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| tuckcore-metal | 3 | 301 ms | 68.40 | 74.09 | 1277 MiB |
| llama-metal | 3 | 289 ms | 70.06 | 75.79 | 200 MiB |

- Paired TTFT: median **1.04x** - prefill **~96% of llama.cpp** (pairs
  0.99 / 1.04 / 1.04). The >= 90% prefill bar is **met on Gemma 3 for
  the first time** (previous acceptance runs: 324 vs 290 ms, paired
  1.12-1.13x, ~88.5-89%). Warm TTFT 303/301/301 - the request-level
  move is -23 ms, larger than the loaded in-process A/B's -8.9 ms: the
  GEMM M-scaling gain is paid in full on the quiet machine (the loaded
  A/B's cache contention compressed it), consistent with the isolated
  same-window arithmetic direction.
- Paired decode: median 0.979 (1.193 / 0.977 / 0.979). TuckCore
  73.97-74.21 - the split-K decode number, unchanged as designed (the
  change touches prefill chunking only). Round 1's 1.193 is llama.cpp's
  recorded bimodality (61.99 that round); against its fast rounds
  (75.8) the pair is 0.977-0.979 - the >= 90% decode bar stays met on
  every pair.
- Both Phase 1 bars (>= 90% decode AND prefill) are now met on Gemma 3
  in this run: 0.977-0.979 decode, 0.99-1.04 prefill.

## Reading

- The remaining ~4% prefill gap (301 vs 289 ms) is the at-512 GEMM
  kernel deficit measured isolated (FFN down 1152x6912: ours 3717 us /
  2193 GFLOP/s vs llama 3101 / 2629, 1.199x; small attention shapes
  1.17-1.20x) - the recorded follow-up lever, llama.cpp pipelines its
  K-loop where our 64x32 tile re-reads W per token-tile behind
  barriers.
- Footprint peak 1277 MiB (+48 vs the split-K acceptance's 1229; the
  512-row prefill scratch is 56.1 vs 17.6 MiB at capacity 520) - inside
  the planner's 192 MiB resident workspaces reservation, admission
  unchanged.
- Decode regression check: none (74.09 vs 74.21 medians; the split-K
  path is untouched by chunk size).
