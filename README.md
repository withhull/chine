# Chine

Local inference for Apple Silicon — 8 GB MacBook Airs through high-memory
Macs — built around one engine this project owns end to end.

- **One unified Metal engine runs every supported family** (ADR-011). A
  small adapter describes a model as a typed graph — which tensors are
  which projections, the attention window, the activation — and the
  engine executes it from shared blocks: matrix-unit Q8_0/Q4/F16
  GEMMs,
  chunk attention with or without sliding windows, Gated DeltaNet
  recurrence, RMSNorm, GeGLU, sampling. There is no per-model forward
  pass and no vendored foreign engine on this path.
- **llama.cpp b11695 is pinned as both oracle and baseline**
  (`scripts/fetch-llama.sh`, loaded at runtime, never vendored). Every
  comparison runs under identity rules — identical artifact, token IDs,
  declared numerical profile and cache state — and every displayed
  number is labeled `estimated`, `locally-calibrated` or `certified`
  (ADR-001, [docs/claims.md](docs/claims.md)).
- **The model you chose runs with its own computation.** No dropped
  experts, no precision substitution, no truncated context. The
  numerical profile is declared per load and recorded in every
  identity, stat and verify record (ADR-010).

Product contract: [ADR-001](docs/adr/ADR-001-product-contract.md).
Positioning and memory tiers: [ADR-006](docs/adr/ADR-006-positioning-and-memory-tiers.md).

## Status

Engineering build; nothing is certified. The full state, with evidence
per row, is [docs/certification-matrix.md](docs/certification-matrix.md).

- **Orchestration**: implemented and tested — inspect, import, pack, plan
  (`R + U + H ≤ B` admission), an isolated worker, an OpenAI-style
  `/v1/chat/completions` endpoint with SSE, `verify --reference` against
  pinned llama.cpp, labeled bench records.
- **ChineCore executor**: Qwen3.5 0.8B and 2B, Gemma 3 1B (ADR-013) and
  dense Qwen3 all run on the unified Metal engine with batched prefill
  and pipelined greedy decoding. The CPU path (scalar + NEON, bounded
  weight streaming) stays as the CPU executor and arithmetic oracle.
- **Not built yet**: GPU partial residency and streaming, MoE execution,
  K-quant kernels, speculative sampling (greedy MTP speculation is
  built).
- **Hardware exercised**: one MacBook Air M3 8 GB. Other tiers are
  pending; all numbers below are locally-calibrated on that machine.

## Measured against pinned llama.cpp (M3 8 GB, 512 tokens in / 256 out, ABBA)

| Model (Q8_0) | Decode, paired | First token, paired |
| --- | --- | --- |
| Qwen3.5-0.8B | 0.99–1.26× (two idle runs + a rerun; 84.2–84.3 vs 66.8–84.7 tok/s) | 94–106% — parity |
| Qwen3.5-2B (3 GiB budget) | 0.956–1.030 across four runs — parity | 94% with f32-operand GEMMs, **98%** on the f16-operand profile (0.601 vs 0.587 s, quiet AC run) |
| Qwen3-1.7B (4 GiB budget) | **0.990 (0.989–0.991)** — the tightest paired band recorded on any model (both engines ± 0.05 tok/s) | **95% (588 vs 559 ms, paired 1.05×)** — the dense family one size up, zero engine changes; both Phase 1 bars met |
| Qwen3-0.6B | not measurable in this harness (the frozen prompt hits EOS after 1–2 tokens on both engines) | 83% (0.254 vs 0.212 s) |
| Gemma 3 1B (f16 KV, ignore-EOS) | **1.067 median (0.977–1.197)** — split-K decode attention: 74.2 tok/s steady, up from 65.0; ≥ 0.9 even against llama.cpp's fastest round | **96% (301 vs 289 ms, paired 1.04×)** — prefill chunks of 512 tokens: 324 → 301 ms; **both Phase 1 bars met in this run** |
| Gemma 3 1B **Q4_0** (with Q4_1 tensors) | **0.932 (0.931/0.932/0.938) — the bar met with margin in every clean fast-mode pair**: 94.2–95.7 steady vs llama's 101.2–102.2. The arc: vectorized gemv loads 86.4 → 91.7, then the shape-adaptive redesign (one-row-per-simdgroup below 2048 rows, multi-row above) → 95.3 (+3.9%); the head is measured at the DRAM ceiling (closed) and the norm stack at its floor | **92% (312 vs 287 ms, paired 1.09×)** — prefill bar met on the first Q4 artifact |
| Qwen3.5-0.8B **Q4_K_M** (K-quants: Q4_K + Q6_K + Q5_K + F32 GDN projections) | **1.018 (1.012/1.018/1.018) — parity-plus**: 113.15 (± 0.35) vs 111.25, ahead in every pair. The arc: first-cut 0.866 → three invented/derived forms quiet-reverted → **the verbatim port of the reference lane decomposition** (+17.2% decode) | ⛔ **72% (387 vs 279 ms, paired 1.39×)** — prefill bar not yet met; the reference mul_mm staging port is the recorded follow-up |

The Phase 1 bar is ≥ 90% of llama.cpp for decode and prefill. Decode
meets it on the 0.8B (reproduced), at parity-or-ahead on the 2B, on
Gemma 3 (worst pair 0.977), and on the Q4_0 artifact **with margin
(0.931–0.938, all clean fast-mode pairs, held by the shape-adaptive
gemv redesign)**. Prefill is
met and reproduced on the 0.8B, met at 98% on the 2B (reproduction
pending), and met at 96% on Gemma 3 and 92% on its Q4_0 pin —
llama.cpp's own first-token spread (0.269–0.300 s on the
0.8B) is as wide as several of these margins, so parity is stated
where a lead is not established.

**GEMM operand profile (`metal-f16-act`, ADR-010/ADR-011).** Batched
prefill GEMMs stage dequantized weights and activations as half and
multiply in half with f32 accumulation — llama.cpp `kernel_mul_mm`
numerics. This is the default; `--chinecore-profile oracle-f32` (with
`--kernels metal`) restores f32 operands. Qualified: `verify
--reference` PASS on the 0.8B / 2B / Qwen3-0.6B (max |Δ| 0.34–0.35
under the registered 5e-1 Q8_0 policy, generated texts equal); the
Gemma 3 real-text Metal gate is unchanged (max |Δ| 0.026, top-1 equal
at every length). The resolved profile appears in every record.

**Where Gemma prefill stands, and what the record says to do next.**
The grouped-regime ablation (ADR-011 note, 2026-11-22) measured the
production encoder directly: the 512-token prefill is **~85% GEMM**
(303.0/357.9 and 314.7/370.7 ms across two passes); attention ≤ 11.9%,
GeGLU ≤ 3.4%, all norms ≤ 3.7%, host skeleton ~1.4%. The isolated FFN
kernels are 1.6–1.9% behind llama's and the small attention shapes are
ahead of it, which bounds further per-shape kernel tuning at ~1.5% of
prefill. Node-concurrent encoding is worth less than the early ADR-011
note hoped: llama.cpp's own merged concurrency reordering (PR #19555,
Feb 2026) measures 0.98–1.03× on dense prefill — the GEMMs already
saturate the GPU, so overlap has little idle capacity to fill. The
headroom other engines measurably exploit on Apple Silicon is
decode-side (Rec's fused whole-decode-pass Metal kernel claims
1.4–2.6×; MLX leads llama.cpp decode by ~9%), workload-level (prefix
caching — built here, bit-identical), and long-context KV pruning. The
Gemma decode gap is closed: the per-row GEMV kernels measured at
parity with llama.cpp, the actual cost was the decode attention (one
128-thread group per row at ~5.4 GB/s), and the shipped split-K decode
attention (up to 16 slice groups + merge at n = 1, llama.cpp's own
split) moved the request-level number exactly as its −60% in-engine
attention cost says: 65.0 → 74.2 tok/s steady, paired 0.977–1.197.
The Gemma prefill gap closed the same way, by matching the request
shape: llama.cpp prefills as one 512-row GEMM batch and its kernels
scale better with M (FFN down +25% per token from 128 to 512 rows);
shipping 512-token chunks (one line, every dependent derives from the
constant) moved TTFT 324 → 301 ms, paired 1.04× — both Phase 1 bars
met. The remaining ~4% is the at-512 GEMM kernel deficit (FFN down
1.199× isolated; llama pipelines its K-loop), the recorded follow-up.

**Beyond decode and prefill:**

- **MTP speculative greedy** (`--speculative`, ADR-012): the artifact's
  own MTP head drafts one token, the engine verifies with its exact
  decode arithmetic, output is token-identical to plain greedy. 0.8B
  decode 98–101 vs 76–78 tok/s plain, TTFT +4% (indicative). The 2B has
  no MTP block and is refused.
- **Prefix cache** (ADR-012): a request that extends the previous
  request's prefix resumes from a snapshot at a chunk boundary,
  bit-identical to a cold prefill. A 984-token chat: 0.87–0.93 s cold vs
  0.33–0.37 s on a hit (0.8B, indicative). Off in `bench`.
- **Cold first request** (worker start, full-payload identity check, GPU
  upload, first prefill): 1.13–1.26 s vs llama.cpp's 0.75–1.40 s in the
  same harness (0.8B).
- **Not measured:** other models, contexts and memory tiers. Records are
  in [benchmarks/results/](benchmarks/results/), each with the command,
  conditions and a validity check per round.

## Quickstart

```bash
scripts/build.sh         # cargo build + native ABI stub check
scripts/test.sh          # fmt + clippy -D warnings + full test suite
scripts/demo.sh          # end-to-end on a generated fixture
scripts/fetch-llama.sh   # pinned llama.cpp reference + tm shim into .deps/
```

```bash
chine doctor                                    # device profile, honest labels
chine inspect model.gguf                        # validate without loading weights
chine import model.gguf --alias my-model        # atomic, resumable, verified
chine plan my-model                             # admission, mode, assumptions
chine run my-model --prompt "..." --engine chinecore --kernels metal
chine run my-model --prompt "..." --speculative     # MTP greedy, opt-in
chine verify my-model --reference              # ChineCore vs llama.cpp logits
chine bench my-model --engine chinecore          # labeled benchmark record
chine serve --engine chinecore --kernels metal   # loopback API on 127.0.0.1:8378
scripts/bench-baseline.sh model.gguf               # ABBA: ChineCore vs llama.cpp
```

`serve` accepts `--keep-alive 5m` (or `0`, or `forever`), per-request
`keep_alive`/`ttl`, `--max-loaded-models N` within the budget, and
`POST /control/load` / `/control/unload`; idle models are released under
real macOS memory pressure. The default process budget is a quarter of
unified memory, at least 2 GiB and at most 24 GiB; `--memory-budget`
overrides it.

## Execution strategy by memory tier

One engine; the planner picks a strategy per model and per Mac by
admission (`R + U + H ≤ B`). Execution is resident-first: weights load
once and stay resident when they fit. Streaming is a strategy for
constrained machines, not the architecture (ADR-006).

| Unified memory | Strategy | State |
| --- | --- | --- |
| 8 GB | Compact weights, strict budgeting, selective streaming | budgeting and admission ✅; resident Metal ✅ (≤ 2B); CPU block streaming ✅ (dense Qwen3); GPU streaming ⛔ Phase 2 |
| 16–24 GB | Hybrid resident and streamed execution, adaptive caching | ⛔ Phases 2–3 |
| 32–48 GB | Larger resident models, fast prefill, speculative decoding | resident Metal ✅ (small models); speculation ✅ experimental (MTP greedy) |
| 64–128 GB+ | Maximum resident execution, batching, low-latency decoding | ⛔ Phases 1, 4 |

8 GB is the minimum tier this must handle correctly, not the product's
ambition — the tier table above is the scope.

## What works today

| Component | State |
| --- | --- |
| `doctor`: device profile from the system; unavailable signals are labeled, never guessed | ✅ |
| `inspect`: GGUF v2/v3 and Safetensors, validated with checked arithmetic before any allocation | ✅ |
| `import` / `pack`: resumable chunked-digest imports, atomic publish, lossless packer with per-tile checksums | ✅ |
| `plan`: `R + U + H ≤ B` admission, KV math, mode selection, typed infeasibility, recorded assumptions | ✅ |
| `run` / `serve`: isolated worker, framed protocol, streaming, cancellation, bounded queue, loopback HTTP with bearer auth and SSE | ✅ |
| `verify`: resident vs streamed, and ChineCore vs llama.cpp on identical token IDs with registered tolerances | ✅ |
| `bench`: standard-v1 records with cache-state labels and caveats | ✅ |
| Governor: Normal / Constrained / Paused / Draining / Failed, driven by real memory-pressure and thermal signals | ✅ |
| Qwen3.5 hybrid on Metal (unified engine): Gated DeltaNet and gated attention, packed f16 KV, batched prefill, device greedy sampling | ✅ experimental; chat and raw completion (pinned 0.8B and 2B templates, text-only subset, thinking off by default) |
| Dense Qwen3 on Metal (unified engine): f32 or packed f16 KV, batched prefill, device greedy sampling | ✅ experimental; qualified on the pinned Qwen3-0.6B (CPU oracle and llama.cpp envelopes) |
| Gemma 3 text on Metal (unified engine, ADR-013): sliding-window attention, sandwich norms, GeGLU; raw completion + chat (`gemma3-turns` template recognized, `/v1/chat/completions` gated) | ✅ experimental; logits match pinned llama.cpp Metal (max diff ≤ 0.022–0.026 on the Q8_0 pin, ≤ 0.0114 on the Q4_0 pin); chat renders byte-exactly like Jinja2 and answers on the real pin (image content refused) |
| Dense Qwen3 on CPU: scalar f32 oracle and NEON Q8_0/F16 kernels, bounded weight streaming | ✅ experimental |
| Formats executed | CPU: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K. Metal: F32, F16, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K. Other K-quants (Q2/Q3/Q8_K) and IQ formats are refused with a typed error |
| GPU partial residency and streaming, MoE execution, compressed (Bonsai) execution | ⛔ Phases 2–3 |

## Architecture

```text
CLI / local API client
        |
Rust supervisor: admission, queue, governor, cancellation, non-content logs
        |
planner -> versioned ExecutionPlan (mode, budget, assumptions, confidence)
        |
InferenceBackend interface (ADR-005)
   ├── chinecore         custom executor: unified Metal engine (ADR-011), CPU oracle
   ├── reference        pinned llama.cpp/ggml (compatibility and oracle)
   └── reference-test   synthetic pipeline engine (disclosed)
```

- **Supervisor** (`crates/supervisor`): never computes tensors; stays
  responsive when a worker fails and fabricates nothing.
- **Planner** (`crates/planner`): prices memory and latency from artifact
  metadata and device measurements; projections stay `estimated` until
  they are measured.
- **Store** (`crates/store`): keeps data under
  `~/Library/Application Support/Chine` (override with
  `CHINE_DATA_DIR`), installs atomically and never deletes sources.

## Repository layout

```text
crates/manifest     schemas, GGUF/Safetensors inspection, adapters, fixtures
crates/backend      versioned inference-backend interface
crates/planner      device profiler, admission and KV math, cost model
crates/store        imports, packing, cache/GC
crates/core     custom executor: unified engine, CPU oracle, loader, tokenizer
crates/metal        typed Rust handles over the Metal C ABI
crates/llamaref     pinned llama.cpp reference backend
crates/supervisor   queue, governor, worker protocol, API, bench, verify
crates/cli          the chine binary
native/executor     executor C ABI header and worker protocol spec
native/metal        Metal C ABI (ch_metal.h) and MSL kernels (ADR-007)
native/llamaref     narrow C shim over pinned llama.cpp
kernels/            kernel work-package registry
adapters/           model-family adapter specs
benchmarks/         standard-v1 workload and per-tier results
docs/adr/           architecture decisions (ADR-001..013)
scripts/            build, test, demo, fetch-llama, bench-baseline
```

## Verification

`scripts/test.sh` is the gate (511 tests at this revision) and the same
gate CI runs on a macOS runner. It covers formatting,
`clippy -D warnings`, and:

- adversarial GGUF and Safetensors fixtures;
- planner math, governor semantics and import fault injection;
- worker protocol and HTTP/SSE;
- CLI end-to-end runs with an isolated worker, crash injection and
  cancellation;
- Metal kernels and engine blocks qualified against the CPU oracle, with
  bit-identity gates between chunk and per-token execution paths. Those
  gates assert exact bits on physical Apple GPUs; a CI runner's
  paravirtualized GPU compiles fast-math differently, so there the
  recorded contract is the drift-bounded form with top-1 equality.

GPU and isolated-worker tests are mandatory: a missing device or worker
binary fails the gate instead of skipping. Real-weight comparisons against
pinned llama.cpp are machine-local gates. `rust-toolchain.toml` pins the
toolchain that CI also uses.

Comparisons follow the four-way identity rules in
[benchmarks/README.md](benchmarks/README.md): identical artifacts, token
IDs, numerical profiles and declared cache state. Nothing is claimed
until the measurement is in the record.

## License

Apache-2.0 (see [LICENSE](LICENSE)). Importing a model is not a license
review: model weights carry their own licenses.
