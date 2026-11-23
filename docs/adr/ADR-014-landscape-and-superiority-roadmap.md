# ADR-014: The competitive landscape and the superiority program

Status: **accepted** (2026-11-23). Builds on ADR-001 (product contract),
ADR-006 (tiers), ADR-011 (unified engine, including its Splash landscape
note), ADR-012 (serving roadmap). Extends ADR-012's step order; supersedes
nothing.

## Context

Six references reviewed 2026-11-23. Every competitor number below is
**their published claim at read date**, not a Chine measurement; Chine
rows cite records already in `benchmarks/results/` and
`docs/certification-matrix.md`.

- **llama.cpp** (ggml-org): pinned here as oracle and baseline (b11695).
  Every comparison runs under the four-way identity rules. The parity
  target others measure against.
- **Ollama** (187.9k stars): distribution, ecosystem and agent
  integrations over llama.cpp (plus an MLX runner). The convenience king;
  not the Apple-Silicon performance frontier. Its contract violations
  (context shrink on OOM, free-memory admission, per-family forward
  passes) are already refused in ADR-012.
- **oMLX** (23.9k stars): MLX server — continuous batching (mlx-lm
  BatchGenerator), vLLM-style paged KV with a hot RAM + cold SSD tier,
  Affine4/8 KV compression, multi-model LRU, VLM/OCR/embeddings/rerankers,
  multi-Mac pipeline split (Ring/Thunderbolt RDMA), a prefill accelerator
  on a second Mac, admin UI. Per-family custom kernels with **documented
  silent fallbacks** (GLM-5.2 fused DSA prefill "roughly 30× faster with
  the kernels, 845 vs ~29 tok/s on an M3 Ultra" — a user without full
  Xcode silently gets the slow generic path).
- **mlx-dspark** (779 stars): DSpark/DFlash/DFlash2 speculative decoding
  on MLX. Measured 1.13–4.31× decode across 15+ targets, **lossless by
  verify**; per-machine calibrated draft caps (~5 s, cached per model +
  quant); drafter-free n-gram **lookup** speculation for any target
  (4.5× on Gemma-12B, 3.6× on Ornith-9B on copy-heavy code editing);
  continuous batching ~2.5× aggregate; CPU co-prefill ~1.3–1.4×; OpenAI +
  Anthropic + Responses APIs. Their Claude Code finding: **prefix-cache
  compatibility dominates drafter quality** on agent workloads
  (Qwen3-8B ~2:20 vs Ornith 4:10 wall clock despite a lower accept
  length — 18–26k-token prompts re-prefilled every request).
- **Splash** (2.7k stars, Inco AI): per-family bespoke Metal kernels +
  trained DFlash2 drafts + ANE FFN prefill. Measured on M5 Pro / M3 Max:
  74 tok/s vs llama.cpp's 16–17 on Qwen3.8-27B (4.5–5.3×; 2.7–4.6×
  against MTP), 175–209 vs 66–69 on the 35B-A3B MoE (2.5–3.2×); prefill
  2,011 tok/s on 32K (1.3×); **cached TTFT 6.6–7.3×**; aggregate decode
  2.0–3.9× on 4 concurrent; 8-bit KV default; SSD cache to 32 GiB.
  Fidelity is published as next-token **agreement 99.30–99.45%** (27B) —
  i.e. not bit-exact. All of their headline ratios are on 27B+ MoE at
  48–64 GB tiers; no small-model or 8 GB number is published anywhere.
- **airllm** (lyogavin/airllm, 34.9k stars): layer-wise weight streaming
  — the checkpoint is split into per-decoder-layer files on disk, one
  layer at a time is loaded, computed and freed, and the next is
  prefetched asynchronously. Peak memory is the largest layer plus KV,
  not the model: 70B on a single 4 GB GPU, DeepSeek-V3 (671B) in
  ~12 GB, Kimi K3 (2.8T) under 4 GB — MoE streams only the routed
  experts per token. The README publishes capacity, never absolute
  speed; community measurements span ~13 s to ~2 min per token:
  access at the price of throughput. Optional 4/8-bit block-wise
  compression at split time (their "up to 3×", contested in their own
  issues) is a requantization of the model. macOS runs via MLX,
  Llama-style architectures only, with community-documented garbled
  Llama-3 output on unfixed versions — the silent-degradation
  anti-example again. Their split doubles disk usage
  (`delete_original` reclaims it).

**Where Chine is (locally-calibrated, one M3 8 GB, identity rules):**
decode and prefill at parity or parity-plus with pinned llama.cpp on five
pins ≤ 2B (decode paired 0.93–1.11, prefill 92–106%, both Phase 1 bars
met, the dense 1.7B as the sixth model at 0.990/95%); MTP greedy
speculation ≈ 1.30× decode on the 0.8B pin; prefix cache bit-identical
to a cold prefill (0.33–0.37 s hit vs 0.87–0.93 s cold on a 984-token
chat); model lifecycle with real macOS pressure unload (ahead of Ollama
per ADR-012 step 4b's own comparison). **Not built:** batching,
paged/multi-snapshot/SSD caches, KV compression, drafts beyond one MTP
token, tool calls, Anthropic endpoint, pull; MoE is specified, not
executing.

**The gap map.** The levers others measurably exploit that Chine has not
measured: (1) drafts at higher accept-length × cost ratios, (2)
concurrency, (3) cache depth (paged blocks, SSD tiers), (4) long-context
KV cost, (5) big MoE at 32 GB+ tiers, (6) capacity-class execution of
oversized artifacts — airllm's whole premise, at their
throughput-sacrificing extreme. Nobody publishes an advantage on
our measured tier — the ≤ 2B dense/hybrid class at 8 GB — where our
decode is already at the DRAM ceiling and paired at parity-plus.

## Decision

Adopt the levers; never their contract violations. Every item lands as
its own gated step with a named bar and a named competitor shape; a step
that fails its gate stays ⛔ in the README. Order is by measured leverage
per engineering cost on the measured tier. Profiling precedes every
kernel change (repo rule 7); comparisons run only under the identity
rules (rule 6).

1. **Concurrency: continuous batching behind admission.** The supervisor
   admits a decode batch only when every member's plan holds; one
   generation at a time becomes one batch at a time with per-request
   cancellation and bounded queues unchanged. Gate: batched decode
   logits within the same registered envelopes; pressure/eviction
   semantics unchanged; `scripts/test.sh` green with crash and cancel
   injection on batches. Bar: aggregate decode ≥ 2× on 4 concurrent
   short prompts (omlx's and Splash's own headline shape) or the honest
   record of why not.
2. **Cache depth.** The recorded ADR-012 step-2 follow-ups first —
   several snapshots, partial-prefix rungs, snapshots during decode
   (multi-turn reuse) — then a cold SSD tier with per-block checksums
   through the existing packer discipline, bounded by plan. The gate
   stays bit-identical to a cold prefill; hits stay labeled, never mixed
   with cold numbers. Bar: cached TTFT on a 32K replay in Splash's own
   shape; the dspark finding (prefix reuse dominates drafter quality on
   agent prompts) makes this the agent-workload lever.
3. **Drafts beyond one MTP token.** First n-gram/lookup speculation for
   every family — lossless through the existing verify machinery,
   dspark's 1.13–4.5× on copy-heavy agent text is the shape to measure;
   then DSpark/DFlash-family drafters as adapters on the unified engine;
   with per-machine calibrated draft caps (dspark's calibration is our
   admission philosophy applied to speculation — the cap derives from
   this Mac's measured verify/drafter cost curves, never hand-set).
   Gates: token-identical to plain greedy; sampled requests only with
   exact rejection sampling (the recorded ADR-012 step-3 follow-up);
   a real-text acceptance profile replaces the frozen synthetic
   workload before any ratio is quoted.
4. **Long-context KV: a declared 8-bit KV representation.** Priced,
   planned and recorded like f16/f32 packed KV today, with its own
   registered tolerance policy before any speedup is read (Splash,
   omlx and dspark all ship 8-bit KV; ours keeps the declared-profile
   contract — representation in the identity, drift bounded in the
   record).
5. **The located kernel deficit.** The at-512 FFN-down GEMM (1.199×
   isolated; llama pipelines its K-loop) — the only remaining measured
   prefill gap on our pins. Profile first, port second, quiet-accept
   third. No other kernel work without a measured gap (rule 7).
6. **MoE execution** (the adapter exists) — the class every competitor
   headline ratio above 2× actually runs on. Gate: exact expert routing,
   no dropped experts (rule 2), admission across expert working sets.
   Then, only on real larger-tier hardware: the two ADR-011 Phase-2
   levers — bf16-in-MMA decode operands and ANE FFN prefill — evaluated,
   not ported blind.
7. **Capacity-class execution for oversized artifacts** — Phase 2's GPU
   partial residency and streaming, now with airllm as the reference
   to design against. Admission may plan a resident-head +
   streamed-tail hybrid per ADR-006: resident layers within the
   budget, the tail streamed through the mapped, verified artifact in
   bounded blocks — the CPU path's existing streaming discipline, on
   the GPU. Two refusals from the airllm read: the all-streaming
   whole-model path, whose decode floor is disk bandwidth (their
   ~13 s–2 min per token), and split-time weight requantization
   (rule 2 — the artifact's own computation or nothing). Their
   per-layer split doubles disk; GGUF tensor blocks plus the mapped
   handle need no split copy at all. Gate: the plan record carries an
   `estimated` decode rate derived from measured disk bandwidth, and
   the executed run must land inside its registered band — an
   honest-to-the-tier admission, never a speed claim.
8. **Distribution and API surface (ADR-012 step 5, unchanged):** pull
   with verified content hashes; an Anthropic-compatible endpoint;
   tool calls and structured output only through declared templates,
   never by prompt injection.
9. **Robustness campaigns gate every public claim above:** the G4 items
   already in the certification matrix (GGUF fuzzing, worker crash
   injection, pressure soak, streaming-disconnect soak) plus multi-tier
   measurement (M1/M2 minimum cohorts, 16 GB+) — no tier claim exists
   until its record does.

**Refused, from the same read:** silent kernel fallbacks (oMLX — a user
must never silently get the slow path; ours fails closed with a typed
error); context truncation and free-memory heuristics (Ollama, ADR-012);
per-model bespoke kernel suites replacing the unified engine (Splash
pairs each model with its own kernels; Chine keeps one engine and typed
adapters); remote-code execution (dspark refuses it by default — agreed);
split-time weight requantization and the all-streaming whole-model
decode path (airllm's compression option and operating point —
rule 2 and the honest-latency gate above); and every unmeasured
speedup claim — all five publish self-measured
numbers on their own workloads; `docs/claims.md` line 28 stands until a
matched record exists under our identity rules.

## Consequences

- The program is specified against named competitors, their own numbers
  and their own benchmark shapes; each bar above names the shape it must
  beat and the gate that makes it honest.
- No new public claim arises from this ADR: competitor numbers are
  theirs (read-dated), Chine numbers were already recorded.
- ADR-012's step order is extended, not superseded: step 5 remains;
  its steps 2–3 follow-ups are absorbed into items 2–3 here.
- The differentiation thesis is explicit: others ship the features with
  silent fallbacks and unquantified fidelity; Chine ships the same
  features with losslessness proven, admission exact, and every number
  labeled — on the tier we measure, against the engine they all measure
  against.
