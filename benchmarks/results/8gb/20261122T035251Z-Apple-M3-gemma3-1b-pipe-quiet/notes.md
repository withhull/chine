# Gemma 3 1B Q8_0 at 22f8c90, quiet AC run (software-pipelined GEMM K-loop)

- Command: `KV_PRECISION=f16 IGNORE_EOS=1 SKIP_LLAMA_BENCH=1 ROUNDS=3
  scripts/bench-baseline.sh artifacts-local/gemma-3-1b-it-Q8_0.gguf
  gemma3-1b-pipe-quiet`, run from Terminal with the Hull app closed, AC
  power, Mac untouched. Zero swap; all 6 invocations valid.
- Tree clean at `22f8c90` — the acceptance run for the software-pipelined
  MMA GEMM K-loop (both kernels: block blk+1's device loads issued under
  block blk's mma). The isolated same-window numbers said -7-11% at
  every deficit shape (FFN down 3717.4 -> 3353.7 us at 512 rows,
  1.199x -> 1.082x vs llama; k/v 1.194x -> 1.062x; q 1.175x -> 1.091x;
  o 1.200x -> 1.083x), projecting ~-14 ms at the request level.

## Results (locally-calibrated)

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| tuckcore-metal | 3 | 317 ms | 68.07 | 74.06 | 1277 MiB |
| llama-metal | 3 | 289 ms | 70.11 | 75.86 | 200 MiB |

- Paired TTFT: median **1.10x** — prefill regressed vs the pb512
  acceptance at the pre-pipeline kernels (301 ms median, paired 1.04x):
  warm TTFT 319/317/317 vs 303/301/301, **+16 ms**.
- The control is decisive: llama.cpp reproduced its pb512-run numbers
  exactly (289/290/289 TTFT, 75.7-75.9 decode) — the machine was quiet
  and consistent, so the regression is attributable to the kernel
  change alone.
- Decode unchanged (74.06 vs 74.09 medians — the pipelining touches
  the batched GEMM path, not the decode GEMV).

## Reading — the isolated projection reversed at the request level

- Reverted at `52e269c`. The standing acceptance records remain the
  pb512 run: TTFT 301 ms (paired 1.04x, prefill ~96%), decode 74.09
  (paired 0.977-0.979).
- The likely mechanism, recorded as hypothesis (not proven): in the
  model, GEMMs interleave with attention/norm kernels in one grouped
  command buffer, where the prefetch registers held live across the mma
  section cost occupancy or issue slots; in the isolated 16-identical-
  dispatch harness the same pattern wins. The isolated harness ran on
  a loaded machine, the acceptance on a quiet one.
- This is the **third** isolated/ungrouped projection to fail at the
  request level (the fused-norm arc, the edge-stage cross-window
  comparison, now this), and the first to actively REGRESS. The rule
  going forward, recorded in ADR-011: no kernel change ships on
  isolated evidence alone — it needs an in-model interleaved A/B (as
  the batch-512 change had) or the quiet acceptance run BEFORE the
  commit is considered accepted.
