# Qwen3-1.7B Q8_0 at dcce36f, loaded-memory-tight run (compatibility check)

- Purpose: the first run of a fourth model — Qwen3-1.7B Q8_0, the
  dense qwen3 family one size up (28 layers, hidden 2048, head dim
  128, 8 KV heads, GQA; downloaded 2026-11-22,
  sha256 per store import `6b470519913961bb…`) — as a compatibility
  check of the engine's dense path at a new geometry, not an
  acceptance record.
- Compatibility chain before the bench: import clean (architecture
  qwen3, no experts); `verify --reference` PASS (12 exact ids, 32
  generated tokens, texts equal, max |Δ| 3.189e-1 under the registered
  5e-1 Q8_0 policy, top-1 100%, top-5 overlap 0.97, mean KL 1.202e-3);
  Metal resident generation verified (resident plan at 4 GiB budget,
  coherent text; the default 2 GiB plan correctly streams and the
  Metal path correctly refuses it — resident-only by design).
- Conditions: mid-morning, session open, reclaimable only 2.3-3.3 GiB;
  one ChineCore round discarded for +275 MiB swap growth and re-run ok.
  Absolute numbers are load- and memory-contaminated for BOTH engines
  (warm decode 26-34 tok/s, warm TTFT 786-1387 ms, both engines
  swinging round to round). Read the paired ratios, not the absolutes.

## Results (locally-calibrated, loaded conditions)

| engine | rounds | warm TTFT | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- |
| chinecore-metal | 3 | 1225 ms | 28.26 | 2233 MiB |
| llama-metal | 3 | 1277 ms | 26.02 | 381 MiB |

- Paired decode: median **1.091** (1.091 / 1.232 / 1.077) — ChineCore
  ahead in every valid round.
- Paired TTFT: median **0.92** — ahead of llama.cpp on this model even
  under load.
- Footprint: ChineCore 2233 MiB resident (weights + KV + scratch) vs
  llama.cpp 381 MiB peak (mmap; not like-for-like, as recorded for the
  other models).

## Reading

- The dense family scales one size up with no engine change: the
  adapter, planner, resident admission and Metal path all held, and
  the reference envelope matches the other Q8_0 models (max |Δ| 0.32
  vs 0.34-0.35 on the 0.8B/2B/0.6B).
- The paired-ahead result under load is consistent with the recorded
  load behavior (llama.cpp's host loop degrades more). The quiet
  acceptance run is the follow-up for the absolute record.
- New-model caveat recorded: the 1.7B GGUF declares the qwen2
  pre-tokenizer; text prompts need CHINE_LLAMA_LIB (pinned llama
  tokenizer) — the typed error says so. The bench harness needs no
  tokenizer (frozen synthetic ids).
