# Gemma 3 1B Q8_0 at 546531a, quiet run

- Command: `KV_PRECISION=f16 IGNORE_EOS=1 SKIP_LLAMA_BENCH=1
  scripts/bench-baseline.sh artifacts-local/gemma-3-1b-it-Q8_0.gguf`
  (ROUNDS=3), run from a Terminal with the Claude app closed.
- MacBook Air M3 8 GB, macOS 27.0.1 (26A434), **battery power** (recorded;
  the first Gemma 3 run was on AC). Reclaimable 2.8-3.8 GiB before each
  invocation; no swap growth; all 6 invocations valid on the first
  attempt.
- Artifact: gemma-3-1b-it-Q8_0.gguf, sha256 `b205840c…10d5`; f16 KV on
  both engines; ignore-EOS workload (ADR-012) so decode is measurable.

## Results (locally-calibrated)

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| tuckcore-metal | 3 | 341 ms | 60.07 | 65.04 | 1226 MiB |
| llama-metal | 3 | 291 ms | 63.68 | 68.44 | 199 MiB |

- Paired decode: median **0.950** (1.043, 0.950, 0.861).
- Paired TTFT: median **1.17** (1.14, 1.17, 1.18).
- Cold-load TTFT: TuckCore 0.78-0.93 s, llama.cpp 0.59-0.67 s.

## Reading

- TuckCore's own numbers are stable (TTFT 349/341/341 ms, decode
  64.0/65.0/65.1 tok/s). llama.cpp's decode varies widely (61.4 / 68.4 /
  75.6 tok/s), so the paired decode ratio moves with it: 1.043 in the
  round llama.cpp was slow, 0.861 in the round it was fast.
- Against the first (AC) run: decode paired median 0.896 -> 0.950, TTFT
  1.11x -> 1.17x. **Decode is at the ≥ 90% bar** (paired medians 0.90 and
  0.95); **prefill is at 85-90%**, just below it.
- No Gemma-specific tuning has been done.
