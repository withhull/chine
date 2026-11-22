# Gemma 3 1B Q8_0 at 118a711 (clean tree), loaded-machine rerun

- Clean tree but the machine stayed in active use (same session load as
  `…-splitk-quiet`): both engines again ~20-25% below quiet baselines,
  llama.cpp hit harder. Paired decode 1.031 (1.008 / 1.031 / 1.180),
  TuckCore ahead in every pair; TTFT paired 1.09 (387 vs 352 ms).
- The acceptance number at this code is `…-splitk-quiet3` (quiet machine,
  app closed): decode 74.21 tok/s, paired 1.067.
