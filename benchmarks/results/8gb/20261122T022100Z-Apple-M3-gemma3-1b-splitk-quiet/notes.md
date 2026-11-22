# Gemma 3 1B Q8_0 at 118a711 (dirty tree: the split-K engine change, pre-commit), loaded-machine run

- Same command shape as the acceptance runs, but measured with the
  machine in active use (the Hull session open, load average 2.3-3.4,
  WindowServer ~31%): both engines ran ~20-25% below their quiet
  baselines. NOT the acceptance number; kept as load-contamination
  evidence at the same code as `…-splitk-quiet3`.
- Paired decode 1.033 (1.033 / 1.031 / 1.101), TuckCore ahead in every
  pair even though both engines were depressed (TuckCore 57.65 vs
  llama.cpp 55.81 medians). TTFT paired 1.07 (389 vs 359 ms).
- The clean-tree rerun (`…-splitk-quiet2`, same load conditions)
  reproduced it: paired decode 1.031, TuckCore 58.37 vs 56.69.
