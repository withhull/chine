# ADR-011: One unified layer engine for every model family

Status: **accepted** (2026-10-26). Extends ADR-005 (ChineCore behind
`InferenceBackend`) and ADR-007 (Metal ABI). Supersedes the per-family
forward passes (`forward.rs`/`gpu.rs` for dense Qwen3, `qwen35_forward.rs`
/`qwen35_gpu.rs` for Qwen3.5) once their migration gates pass.

## Context

Each model family has its own hand-written forward pass, on CPU and on
Metal, about 8,000 lines across the two families. Every cross-cutting
improvement (one command buffer per step, batched prefill, attention
rewrites, matrix-unit GEMM) had to be built per family. They drifted: the
Qwen3.5 attention and prefill work never reached dense Qwen3. The target
list (Llama 3.x, Gemma, Mistral, DeepSeek, gpt-oss, MoE) makes that cost
multiply. The product owner decided: one unified engine only.

## Decision

1. **Adapters describe, one engine executes.** An adapter resolves an
   artifact's metadata and tensor index into a typed `ModelGraph`:
   - the embedding, a list of layers, the final norm and the output head;
   - per layer: a pre-norm, a token **mixer** and an **FFN**, with residual
     wiring;
   - each tensor binding checked once for dtype and shape.
   An unknown architecture or block is a typed refusal that names the
   missing block. It is never approximated.
2. **Blocks are the only place model math lives.** Each block is
   implemented once, on the CPU oracle and on Metal:
   - mixers: `Attention` (GQA; optional Q/K norm; full or partial RoPE,
     NeoX or standard; optional sigmoid output gate) and `GatedDeltaNet`;
   - FFNs: `SwiGLU` now, `GeGLU` and `MoE` (routed + shared experts) as
     families need them;
   - norms: `RMSNorm` with the weight variants families use.
   Each block offers `decode` (one token) and `prefill` (a chunk) and
   declares its session state (KV planes, recurrent/convolution state)
   and scratch for the planner. A new family adds an adapter, and a new
   block only for genuinely new math.
3. **The engine owns everything cross-cutting, once:**
   - one command buffer per decode step and per prefill chunk;
   - chunked prefill, with logits for the last token only;
   - device-side greedy sampling;
   - session reset and cancellation at chunk/token boundaries;
   - preflight allocation from the graph (one plan function, ledger ==
     plan), checked per class against the admitted plan;
   - per-operator profiling;
   - weight access through one `WeightStore`: resident now, and the place
     Phase 2 streaming plugs in.
4. **One oracle per block, plus the old oracles as references.** The CPU
   engine executes the same `ModelGraph` with scalar f32 blocks. The
   existing per-family CPU oracles stay as independent references until
   the graph oracle reproduces them bit-for-bit, then retire.
5. **Same-profile execution is unchanged.** The graph *is* the artifact's
   computation: no block is substituted, dropped or approximated, and
   precision profiles stay declared and recorded (ADR-010).

## Migration (each step gated, nothing deleted before its gate)

1. Engine core plus the blocks Qwen3.5 needs. Qwen3.5 is served through
   the engine. Gate: identical greedy output and teacher-forced logits
   matching the current Qwen3.5 GPU path (same kernels, so bit-identical
   decode is expected) and the CPU oracle within registered tolerances.
   Then retire `qwen35_gpu.rs`'s model code.
2. Dense Qwen3 as a graph on the same blocks. Gate: its CPU oracle and
   registered llama.cpp envelopes, plus the recorded batch-size claims
   re-proven or explicitly superseded in `docs/claims.md`. Then retire
   `gpu.rs`/`forward.rs` model code.
3. New families are adapters (+ blocks): Llama 3.x, Gemma, Mistral, then
   MoE.

Performance work from now on (chunk-wide DeltaNet, batched attention,
fusions, GPU sampling, streaming) lands in engine blocks or the engine,
never in a family-specific path.

## Consequences

- One place to optimize: every gain reaches every family.
- The migration costs some performance work now (rebuilding today's
  Qwen3.5 speed on blocks). The bit-identical gate makes that safe.
- The CLI, worker protocol and planner contracts do not change; ChineCore
  still implements `InferenceBackend`.

## Step 1 record (2026-10-26): Qwen3.5 served through the engine

- `crate::engine`: `ModelGraph` (`graph.rs`), the `WeightSource` seam
  (`mod.rs`), the Qwen3.5 adapter (`adapters.rs`), and on Metal the
  blocks `Matrix`, `SwiGluBlock`, `GdnBlock` and `AttentionBlock` plus
  `EngineModel` (`metal.rs`): one plan function, one command buffer per
  step and per chunk, sessions and profiling. Each block's `decode` and
  `prefill` share one `core`.
- **Gate passed before anything was deleted.** Run side by side, one model
  resident at a time, on the fixture, the pinned 0.8B and the 2B (sha256
  `1b04acba…21f2c1`): equal session state and plan total, equal device
  ledger, bit-identical per-layer hidden states and logits at every decode
  step, bit-identical batched-prefill logits and the decode step after it.
- The executor serves `qwen35` through `EngineModel`; the load note names
  the graph. Worker greedy output still equals the CPU oracle.
- **Retired:** the Qwen3.5 model code in `qwen35_gpu.rs` (layers, model,
  plan; 1,207 lines). Only the device operators remain there. Its
  qualification suite moved to `engine::metal::oracle_tests` unchanged,
  except that it targets the engine.
- **Found on the way:**
  - The engine makes the head explicit (`output` if present, else the
    tied embedding). The retired model always used the embedding; the
    CPU oracle still does and is fixed when the graph oracle lands.
  - The derived RoPE tables are now priced as scratch, not weights. The
    planner reserves exactly the artifact's bytes, and on the 2B (no MTP
    block, so no slack) the old classing was rightly refused by 768 B.
- **2B evidence:** `chine run` (unified engine) and pinned llama.cpp
  Metal produce the same 16 greedy tokens for "The capital of France
  is".
- Next: step 2 (dense Qwen3 on the same blocks), and the graph-driven CPU
  oracle.

## Engine note (2026-10-26): greedy decoding on the device, pipelined

- `ch_metal_argmax_f32` chooses the greedy token on the GPU with exactly
  the CPU rule (`sampling::argmax`: NaN skipped, ties to the lowest index;
  the order is associative, so the two-pass reduction gives the
  sequential answer). It writes one of two alternating token slots.
  `ch_metal_row_indirect_f32` embeds a token id read from such a slot; the
  all-NaN marker writes zeros and the host fails the request when it
  reads the marker.
- `EngineModel::submit_greedy` / `collect_greedy`: the executor queues step
  i+1 (reading step i's choice on the device) before waiting for step i,
  and reads each token with `download_after`, which waits only for that
  step's command buffer. The worker's per-token work (output frames,
  stop/EOS handling) now overlaps the GPU. Temperature > 0 keeps the CPU
  sampler. `reset()` drains first, so a step left in flight at EOS or a
  stop sequence never races the next request's state reset.
- Evidence (pinned 0.8B, M3 8 GB): device greedy and pipelined greedy
  produce the CPU rule's tokens at every step (fixture and pinned, 40
  steps). In-process the step is GPU-bound (14.2–15.0 ms for all three
  paths: the logits download was not the cost). Served decode via
  `chine bench` (128 in / 256 out, ABAB): 56.5 → 64.0 and 51.0 → 59.0
  tok/s (+13–16%), locally-calibrated.

## Engine note (2026-10-26): chunk-wide mixer cores in prefill

- Prefill used to run each layer's sequential core once per token of a
  16-token chunk (about 14 launches plus 5 slice copies per token per
  layer). It now runs both mixer cores chunk-wide:
  - **Gated DeltaNet:** convolution over the chunk with the carried
    history (`conv_chunk` + history refresh), row-batched SiLU, split,
    L2 norms, beta/decay and gated norm, and one recurrence launch that
    walks the chunk's tokens inside each (head, column) thread.
  - **Gated attention:** row-batched gate split and Q/K norms, then RoPE,
    packed-f16 KV store and causal attention for all chunk queries.
    Attention uses one shared inline body (`ch_q35_attend`) for the
    per-token and chunk kernels.
- Every chunk kernel keeps its per-token counterpart's per-element
  arithmetic. Gate: with `set_chunk_core(false)` (the per-token cores) as
  reference, last-token logits and the next decode step are
  **bit-identical** on the fixture (prompts 1/5/16/37) and the pinned 0.8B
  (1/5/16/37/512). The per-token path stays as that reference.
- Effect (pinned 0.8B, M3 8 GB, in-process): 512-token prefill
  1.73 s → 1.29 s (DeltaNet core) → 0.84 s (plus attention). Profile now:
  the matrix-unit GEMM 58%, the DeltaNet chunk recurrence 27%.

## Engine note (2026-10-26): register-resident DeltaNet recurrence; fast-math caveat

- The chunk recurrence was memory-traffic bound: it re-read each 128-float
  state column from device memory twice per token and wrote it once
  (~48 MB per 16-token launch, ~27 GB per 512-token prompt).
  `ch_qwen35_gdn_chunk_k128` (a template instance for key dim 128) keeps
  the column in registers for the whole chunk: read once, written once.
  Other key dims keep the generic kernel.
- **Fast-math caveat.** The kernel library is compiled with Metal's
  default options, i.e. fast math, which may reassociate sums per
  kernel. Source-identical arithmetic in two kernels is therefore not
  guaranteed to round identically. The register kernel differs from the
  generic one by 5.0e-4 / 7.2e-4 on 512-token logits / next step (within
  the registered 2e-3 GPU bound). Earlier "bit-identical" results held
  because those kernels compiled identically, not by guarantee. Rule from
  here on: paths that must agree bit-for-bit share one compiled kernel.
  Decode's single-token step now runs the same chunk kernel with n = 1,
  so chunked prefill and decode stay bit-identical to each other (gated).
- Qualification against the CPU oracle (registered tolerances unchanged):
  pinned delta operator 4.27e-4 out / 1.37e-4 state, real artifact hidden
  7.0e-5 / logits 4.7e-4, fixture model 6.6e-7 / 1.4e-6, batched vs
  sequential 4.6e-4 / 4.0e-4 with identical top-1; worker greedy output
  equals the CPU oracle.
- Effect (pinned 0.8B, M3 8 GB, in-process): 512-token prefill 879 →
  646 ms; recurrence per decode token 1.25 → 0.47 ms; greedy decode
  ~73–77 tok/s in-process.

## Engine note (2026-10-26): 32-token matrix-unit GEMM tile

- `ch_q8_0_gemm_mma` now owns a 32-row x 32-token tile (4 accumulators per
  SIMD group instead of 2), so each dequantized weight tile serves twice as
  many tokens; prefill chunks are 32 tokens (`PREFILL_BATCH`). Per-token
  summation order over K is unchanged.
- Gate: GEMM tolerance test (adds batch 32 and a 27-token partial tile;
  batch 33 refused); engine and chunk-core tests; 512-token batched vs
  sequential logits 4.55e-4 / next step 3.98e-4, identical to the 16-token
  tile, same top-1.
- Effect (pinned 0.8B, M3 8 GB, in-process, alternating A/B x3 from two
  prebuilt binaries): 512-token prefill 624-625 ms -> 516-518 ms (-17%),
  locally-calibrated.

## Engine note (2026-10-26): one GEMM launch per 128-token prefill chunk

- Profile at `45b8386` (512 tokens, ungrouped): matrix-unit GEMM 69%,
  DeltaNet recurrence 10%, attention 9%. A 64-row tile (16 rows x 32
  tokens per SIMD group) was slower (520 -> 568 ms, A/B x3) and was not
  kept: the GEMM was short of threadgroups, not of math per tile. With a
  32-token chunk a 1,024-row matrix gets 32 threadgroups, and each chunk
  re-reads every weight.
- `ch_q8_0_gemm_mma` now covers a whole batch in one launch: a 1-D grid
  of 32-row x 32-token tiles, with neighbouring threadgroups sharing a
  weight tile (batch up to `CH_METAL_GEMM_MMA_MAX_BATCH` = 512). Per-tile
  arithmetic is unchanged. `Matrix::gemm` slices F16 and batch-1 GEMMs
  into <= 32-token pieces through the existing offsets. Prefill chunks
  are 128 tokens.
- Chunk size measured, not assumed (pinned 0.8B, M3 8 GB, in-process,
  alternating prebuilt binaries, two passes; planned scratch at capacity
  640):

  | chunk | 512-token prefill | scratch |
  | --- | --- | --- |
  | 32 | 493-522 ms | 9.7 MiB |
  | 64 | 413-438 ms | 15.4 MiB |
  | 128 | 393-414 ms | 26.8 MiB |
  | 256 | 401-409 ms | 49.6 MiB |

  128 is kept: the gains flatten after it while scratch keeps doubling.
  The attention-score scratch is chunk x heads x context, so long
  admitted contexts pay more. The planner prices it, so admission stays
  exact; a tiled attention without the score buffer is the follow-up.
- Gate: GEMM tolerance at batches 64, 77, 512 (513 refused); fixture
  batched prefill still bit-identical to sequential decode (1, 3, 128,
  132 tokens); 512-token logits 4.55e-4 / next step 3.98e-4, the same as
  before, top-1 equal. Profile after: GEMM 338 ms (was 422 ms ungrouped),
  still 68%.

## Engine note (2026-10-26): GEMM kernel variants measured, none kept

Isolated timing (`q8_0_gemm_batch_timing` with `CHINE_PROFILE=1
CHINE_GEMM_MMA=1`, now batches 32 and 128 after a warm-up, 100
iterations; M3 8 GB, alternating prebuilt binaries, two passes). Baseline
at 128 tokens: 3584x1024 470-500 us, 6144x1024 787-830 us, 1024x3584
500-556 us, about 2 TFLOP/s f32. Variants, all tolerance-passing:

| variant | 3584x1024 | 6144x1024 | 1024x3584 | verdict |
| --- | --- | --- | --- | --- |
| 2x2 SIMD-group layout, vector loads, 32x32 | 461-467 | 756-780 | 480-521 | noise |
| same, 64x32 / 32x64 | 441-483 | 745-803 | 543-595 | no gain |
| same, 64x64 (16 accumulators) | 2294-2416 | 3728-3746 | 2377-2484 | register spill |
| double-buffered stages, one barrier per block | 496-598 | 832-990 | 524-744 | slower |
| K = 64 per stage (two Q8_0 blocks) | 521-599 | 876-991 | 540-687 | slower |
| f16 operand tiles (prototype, not a profile change) | 476-492 | 784-822 | 534-553 | no gain |

- The pinned llama.cpp on this M3 does not use the Metal tensor API
  ("disabled for pre-M5"). Its `kernel_mul_mm` stages dequantized
  weights and activations as half. The f16 prototype shows that operand
  precision is not what separates the two engines, so the oracle-f32
  profile stays.
- From the isolated rates, the GEMMs are about 265 ms of the ~400 ms
  512-token prefill; the rest (DeltaNet recurrence, attention, norms and
  small operators) is about 135 ms. llama.cpp's whole pp512 is about
  258 ms, so closing the gap needs both less non-GEMM time and a faster
  GEMM. Why llama.cpp's GEMM is faster is not yet identified: it reads
  the same 34-byte Q8_0 blocks, a 64x32 tile and K = 32 stages, all of
  which were tried above. It is not timed in isolation here (the pinned
  tools have no matmul benchmark).

## Engine note (2026-10-26): DeltaNet recurrence split across SIMD lanes

- Non-GEMM profile at `00fd250` (512 tokens, ungrouped): DeltaNet
  recurrence 61 ms, attention 39 ms, RMS norm 16 ms, then small
  operators. The recurrence ran one thread per (head, state column):
  16 x 128 = 2,048 threads walking every token with two 128-long dot
  products, so it was latency-bound.
- `ch_qwen35_gdn_chunk_k128` now splits each column across P = 8 lanes
  of one SIMD group (16 rows each, in registers). Lane partial dot
  products are combined with XOR shuffles, so every lane of a column sees
  the same total. A SIMD group covers 4 adjacent columns, and lane l works
  on column l % 4, so state loads and stores stay coalesced. Decode's
  single-token step runs the same kernel (n = 1), so chunked prefill and
  decode remain bit-identical to each other.
- Measured (recurrence GPU time, profile, two passes each; decode is the
  per-token recurrence in `profile_pinned_decode`):

  | layout | prefill recurrence | decode recurrence |
  | --- | --- | --- |
  | one lane per column (before) | 61-62 ms | 0.445-0.468 ms |
  | 4 lanes, blocked | 65 ms | - |
  | 8 lanes, blocked | 35-41 ms | - |
  | 16 lanes, blocked | 26-28 ms | 0.617-0.619 ms (slower) |
  | 32 lanes, blocked | 44-49 ms | - |
  | 16 lanes, interleaved | 25 ms | 0.609-0.638 ms (slower) |
  | **8 lanes, interleaved (kept)** | **19.5 ms** | **0.404-0.424 ms** |

- Accuracy (registered tolerances unchanged): real-artifact GPU vs CPU
  oracle, hidden 7.0e-5 -> 6.5e-5 and logits 4.7e-4 -> 2.8e-4; 512-token
  batched vs sequential 4.27e-4 / 3.52e-4, top-1 equal; lane-split vs
  generic recurrence within 2e-3.
- Whole 512-token prefill, in-process, five alternating passes
  (the Claude app's renderer was using up to 65% CPU, so passes are noisy):
  best 407 -> 368 ms, median 422 -> 399 ms. Locally-calibrated.

## Engine note (2026-10-26): grouped-query attention kernel

- The attention kernel ran one threadgroup per (query head, row). The
  four query heads sharing a KV head each re-read the K/V prefix, each
  thread walked every key serially for its two output dims, and both
  softmax reductions used threadgroup barriers.
- `ch_qwen35_attention_gqa<HD, G>` runs one threadgroup per (KV head,
  row) for all G heads:
  - each K and V row is read once;
  - lanes hold contiguous head dims of every head's query in registers;
  - SIMD group h normalizes head h with SIMD-only reductions, using the
    same per-element softmax expression;
  - the V pass splits keys across SIMD groups and sums the four partials
    in a fixed order.
- Instantiated for (256, 4), which covers the 0.8B and the 2B, and for
  (16, 2), the fixture, so its bit-identity gate exercises it. Other
  geometries keep the old kernel. Decode's single-token call routes
  through the same kernel (n = 1), so prefill and decode stay
  bit-identical.
- Measured (pinned 0.8B, M3 8 GB, alternating prebuilt binaries):
  - attention GPU time, 512-token prefill: 39.5-40.4 -> 8.8-9.0 ms;
  - decode, per token at a short context: 0.333-0.340 -> 0.113-0.117 ms
    (this grows with context);
  - whole 512-token prefill, five passes each: 362-365 -> 329-333 ms.
- Accuracy (registered tolerances unchanged):
  - fixture prefill is still bit-identical to sequential decode;
  - operator tests vs the CPU packed-cache reference pass (4x2x16 and
    8x2x256);
  - real-artifact GPU vs CPU oracle: hidden 6.5e-5, logits 2.78e-4;
  - 512-token batched vs sequential: 3.36e-4 / 3.36e-4, top-1 equal.

## Engine note (2026-10-27): where 512-token prefill time goes now

Measured on the pinned 0.8B (M3 8 GB) at `171e53c`, in the grouped run,
by skipping operators in a throwaway build (outputs wrong, timing only;
three passes each; full prefill ~330 ms):

| skipped | prefill | implied cost |
| --- | --- | --- |
| nothing | ~330 ms | - |
| matrix-unit GEMMs | 71-84 ms | GEMMs ~255-265 ms |
| GEMMs + all small operators | 42-45 ms | small operators ~35 ms together |
| GEMMs + DeltaNet recurrence | 51-64 ms | ~20 ms |
| GEMMs + attention | 59-68 ms | ~14 ms |
| everything | 8-10 ms | host encode, embedding, head |

- No small operator costs more than about 3-10 ms on its own (conv and
  shift, `silu_mul`, `f32_axpy`, RMS norm, L2 norm, gated norm, SiLU,
  embedding rows); there is no single kernel left to fix.
- Tried and not kept: fusing the mixer residual add with the post-norm
  (`add_rmsnorm`, both decode and prefill). It passed its gates, but
  prefill was 326-332 vs 322-336 ms and decode 12.59 vs 12.50 ms per
  token, i.e. no measurable gain.
- The GEMMs are about 80% of prefill. Prefill parity therefore depends
  on the GEMM, where the f32 kernel variants measured earlier all
  plateau at ~2 TFLOP/s.

## Engine note (2026-10-27): why llama.cpp's GEMM was faster; fragment-layout kernel

- **Measured in isolation, same method on both sides:** 16 serial
  dispatches, wall time per dispatch.
  - llama.cpp: `benchmarks/tools/ggml-mul-mat.c`, which runs the pinned
    ggml's `mul_mat` (it selects `kernel_mul_mm_q8_0_f32`).
  - ChineCore: `q8_0_gemm_mma_grouped_wall_timing_when_requested`.
  - Cross-check: the ggml per-node cost from 2-node minus 1-node graphs
    matches its serial 16-node figure, so concurrency does not explain it.
- **llama.cpp was 1.3-1.4x faster.** At 128 tokens: 3584x1024 391-420 vs
  504-559 us; 6144x1024 652-750 vs 868-996 us.
- **Cause: kernel structure, not precision.**
  - Its threadgroup tiles hold each 8x8 fragment contiguously, so loads
    are stride-8 with no transpose. Ours were row-major with stride 32,
    so every row of a fragment load hit the same banks.
  - It loads the next block into registers before the barrier, overlapping
    device latency with other SIMD groups' multiplies.
  - Full tiles store straight from the accumulators.
  - Tiles are 64 x 32.
  - Not the cause: half-precision tiles (llama.cpp uses them) gave nothing
    in either layout, so the oracle-f32 profile stays. Removing bounds
    branches, forced unrolling and aligned activation loads also changed
    nothing measurable.
- **`ch_q8_0_gemm_mma` now uses that structure in f32.** Isolated time at
  128 tokens: 3584x1024 462-490 us, 6144x1024 792-853 us. About half the
  gap is closed; the remaining ~15% is unexplained and close to run-to-run
  noise.
- **In the model** (pinned 0.8B, M3 8 GB idle, two prebuilt binaries
  alternated, six passes each):
  - full 512-token prefill: 311 -> 297 ms;
  - GEMM-only (other operators skipped in a throwaway build): 260 ->
    251-253 ms;
  - per-operator profile: GEMM GPU time 340-368 -> 270-276 ms.
  - Accuracy: GEMM tolerance tests at every shape; 512-token batched vs
    sequential 3.36e-4, top-1 equal.

## Engine note (2026-10-27): GEMM gap to llama.cpp resolved; code shape kept, precision not

- **The harness comparison is fair.** In
  `benchmarks/tools/ggml-mul-mat.c`, `chain` mode makes each ggml node
  consume the previous output. Dependent and independent nodes time the
  same with concurrency disabled (2048x2048 x 128: 432.7 vs 428.8 us).
  Concurrent encoding gives ggml a further ~5% on independent nodes,
  which our serial encoder cannot get (a separate lever).
- **Code shape, measured on a quiet Mac at 128 tokens** (3584x1024 /
  6144x1024 / 1024x3584, in us):

  | kernel | 3584x1024 | 6144x1024 | 1024x3584 |
  | --- | --- | --- | --- |
  | fragment layout (`b5899a1`) | 450-453 | 736-742 | 543-550 |
  | + clamped loads, full unrolling | 432-439 | 707-715 | 508-517 |
  | + pointer increments, 16-bit indices, SIMD barriers (kept) | 410-415 | 677-691 | 461-471 |
  | llama.cpp `kernel_mul_mm_q8_0_f32` | 385-387 | 631-636 | 429-437 |

  Not kept, because no change was measurable: aligned activation loads,
  byte-wise quant loads, and an 8 KB threadgroup footprint.
- **The remaining ~6% is precision.** The same kernel with f16 operand
  tiles (information only) matches llama.cpp exactly: 388-392 / 639-674 /
  441 us. llama.cpp rounds dequantized weights and activations to f16
  before multiplying; ChineCore keeps f32 (the oracle-f32 profile).
  Adopting f16 operands would be a declared profile change (ADR-010) and
  a product decision, not an optimization.
- **In the model** (pinned 0.8B, M3 8 GB idle, prebuilt binaries
  alternated, six passes each): 512-token prefill 319-321 -> 286-292 ms.
  Accuracy is unchanged (512-token batched vs sequential 3.36e-4, top-1
  equal; GEMM tolerance tests at every shape).

## Step 2 plan (2026-10-27): dense Qwen3 on the engine

- **Scope.** Dense Qwen3 on Metal (`--kernels metal`) moves to the engine,
  and `gpu.rs` (its Metal model code) is retired after the gate.
  `forward.rs` stays for now. It is the served CPU executor (scalar and
  NEON, mapped/owned storage, bounded CPU block streaming for the 8 GB
  tier) and the arithmetic oracle, and the graph-driven CPU engine that
  would replace it does not exist yet. It retires in a later step, with
  its own gate.
- **Blocks needed.**
  - `AttentionSpec` with `output_gate: false` (plain Q projection) and
    `rope_dims == head_dim` (full NeoX rotation).
  - KV cache precision is an engine load option, not a graph property.
    Packed f16 serves declared `f16`: the same values the old path stored
    as f16-rounded f32. A new f32 plane serves declared `f32`, the dense
    Qwen3 default. Neither is ever substituted for the other.
- **Adapter and weights.** `adapters::qwen3` maps the resolved Qwen3
  contract to a graph. A family-independent GGUF weight source reads
  through the verified handle.
- **Serving.** Qwen3 with `--kernels metal` loads through the same engine
  load and serve path as Qwen3.5 (plan checks per class, ledger == plan,
  pipelined greedy decoding). `prefill_batch` is refused, as for Qwen3.5,
  because the engine prefills in fixed chunks.
- **Superseded claim.** The old path's "teacher-forced logits bit-identical
  across prefill batch sizes" relied on per-row GEMMs. The engine uses
  the matrix-unit GEMM for batches, so chunked prefill is
  tolerance-qualified against sequential decode on real weights. The
  fixture stays bit-identical where the GEMM is per-row. `docs/claims.md`
  is updated accordingly.
- **Gate before `gpu.rs` is deleted:**
  - fixture: engine vs the CPU oracle (f32 and f16 KV) within the
    registered GPU tolerances, and greedy output equal to the oracle;
  - the pinned real Qwen3-0.6B Q8_0 (sha256 `9465e63a…bb031`,
    re-downloaded and verified): engine vs CPU oracle teacher-forced
    logits within the registered envelope, `verify --reference` against
    pinned llama.cpp within its registered envelopes, and the CLI
    completion matching the recorded real-model check.

## Step 2 record (2026-10-27): dense Qwen3 served through the engine

- **Engine.**
  - `AttentionBlock` supports attention without an output gate; full NeoX
    RoPE is the partial-RoPE kernel with `rope_dims == head_dim`.
  - KV cache precision is a load option (`KvCache::{F16, F32}`):
    `alloc_plan_with` / `EngineModel::load_with`.
  - f32 KV uses new chunk kernels (`ch_qwen35_kv_store_f32_chunk`,
    `ch_qwen35_attention_f32_chunk`, GQA instances for float caches).
    Decode runs them with n = 1, so prefill and decode stay
    bit-identical.
  - GQA instances added for head_dim 128 with group 2 and 4 (Qwen3
    geometries), for both cache types.
- **Adapter and weights.** `adapters::qwen3` maps the resolved dense
  contract. `GgufWeights` is the family-independent weight source, and
  `ArtifactTensorSource` delegates to it.
- **Serving.**
  - `--kernels metal` loads dense Qwen3 through the shared `load_engine`
    (the qwen35 path uses it too). Dense Qwen3 has kernel identity
    `ADR011:qwen3-q8_0+f16:metal`.
  - `verify` no longer sends CPU cache geometry to a Metal side and
    records the cache actually applied.
  - The supervisor's warm-reuse workspace test now pins the engine's
    behavior: reuse is kept while the plan covers the retained scratch,
    and a reload under a reservation one byte short is refused.
- **Gate passed before anything was deleted.**
  - Fixture: engine vs CPU oracle (f32 and f16 KV) within the registered
    policy with equal greedy text; chunked prefill bit-identical to
    decode for both precisions.
  - Pinned Qwen3-0.6B Q8_0 (re-downloaded; sha256 `9465e63a…bb031`
    verified): teacher-forced logits vs the CPU oracle 4.196e-5 (retired
    path 4.274e-5), top-1 equal.
  - `verify --reference` vs pinned llama.cpp (oracle-f32): PASS for f32
    and f16 KV (max 0.352 / 0.350 under tolerance 0.5, top-1 100%,
    generated texts equal).
  - The CLI completion is unchanged.
  - Qwen3.5 numbers are unchanged.
- **Retired:**
  - `crates/core/src/gpu.rs`;
  - the operators only it used: f32 per-token and batched `rope`,
    `kv_store` and `attention`, and `rmsnorm_f16w` (kernels, C ABI,
    pipelines, Rust wrappers and their operator tests);
  - the `--prefill-batch` option (CLI and execution options; a requested
    batch is now an unknown option). The engine reports its effective
    chunk in `prefill_batch` stats and bench records.
- **Superseded claim.** "Teacher-forced logits bit-identical across
  prefill batch sizes 1-8" (`docs/claims.md`) is replaced by the evidence
  above.
- **Remaining.** `forward.rs` stays as the CPU executor and oracle until
  the graph-driven CPU engine exists (a later step with its own gate).
  Dense Qwen3 performance vs llama.cpp is not measured yet.

## Engine note (2026-10-27): dense Qwen3 prefill — attention scores without reductions

- **Profile.** Pinned Qwen3-0.6B Q8_0, 512 tokens, f16 KV, ungrouped,
  from the new opt-in `real_qwen3_prefill_timing_and_profile_when_available`
  test:
  - GEMM 207 ms (62%), at llama.cpp's speed except the documented f16
    operand precision;
  - attention 77 ms (23%): every one of the 28 layers is full attention
    (head_dim 128, group 2);
  - RMS norm 23 ms, mostly per-buffer overhead.
- **Attention split** (throwaway diagnostic builds): the value pass was
  ~30 ms, the per-key SIMD reductions of the scores ~15 ms, and score
  loads and softmax ~31 ms. At head_dim 128 each lane held 4 dims, so
  every score paid a 5-step reduction for 4 FMAs.
- **Tried and not kept:** R = 4 query rows per threadgroup sharing K/V
  loads. It was bit-identical but slower (attention 77 -> 131 ms): 4x
  fewer threadgroups while the per-key reductions stayed serial.
- **Kept, both in `ch_qwen35_attention_gqa`:**
  - The G queries are staged in threadgroup memory, and each thread owns
    whole keys: one sequential dot product per (key, head), with no
    cross-lane reduction.
  - The value pass reads probabilities from 256-key tiles in threadgroup
    memory instead of device scratch, in the same per-SIMD-group key
    order.
  - Decode runs the same kernel (n = 1), so prefill and decode stay
    bit-identical; the fixture gate passes unchanged.
- **Measured** (prebuilt binaries alternated):
  - Qwen3-0.6B 512-token prefill: 298-299 -> 261-262 ms; attention GPU
    time 76-78 -> 40-41 ms; batched vs sequential 1.89e-3 at logit scale
    11.5, top-1 equal.
  - Qwen3.5-0.8B prefill unchanged (287-290 ms); drift 3.20e-4 /
    3.93e-4, within its bound.
  - Decode attention per token at a short context unchanged (0.126 ms).
  - Real-model accuracy got slightly better (sequential dot products):
    - Qwen3 vs the CPU oracle: 4.196e-5 -> 3.600e-5;
    - Qwen3.5 logits vs the oracle: 2.78e-4 -> 2.71e-4;
    - `verify --reference` vs pinned llama.cpp: PASS (max 0.349, top-1
      100%, generated texts equal).

## Engine note (2026-11-22): the metal-f16-act GEMM operand profile is the engine default

- **Decision (product, recorded per ADR-010).** Batched prefill Q8_0 GEMMs
  on the engine now stage dequantized weights and activations as half and
  multiply in half with f32 accumulation — llama.cpp `kernel_mul_mm`
  numerics, the profile ADR-010 pre-declared as `metal-f16-act`. ADR-011's
  own measurement had attributed the last ~6% of the GEMM gap to exactly
  this operand precision (an f16-operand build of the kernel matched
  llama.cpp exactly at the 0.8B shapes); the product owner chose it as the
  default on 2026-11-22. Opt-out: `numerical_profile: "oracle-f32"` (f32
  operands, the prior numerics). The resolved profile is recorded in the
  kernel identity (`…:metal:f16op` / `:f32op`), in stats
  (`numerical_profile`) and in every `verify` caveat; nothing is silently
  substituted. Decode GEMVs, the exact speculative verify, F16-weight
  matrices and every CPU path are unchanged.
- **Kernel.** `ch_q8_0_gemm_mma_f16op` (`native/metal/kernels.metal`):
  same 64x32 fragment tiling, layout, summation order and edge handling as
  `ch_q8_0_gemm_mma`; sa/sb stage as half, the edge output stage keeps its
  own f32 buffer (accumulators never round through it).
- **Qualification (all on this M3 8 GB).**
  - Kernel vs the f32 CPU reference, half-domain boundary data (see
    `half_domain_weights`: the +-65504 block scales of the f32 kernel's
    boundary set dequantize past half range and are out of profile domain;
    real artifact bytes are in domain): worst rel diff 2.9e-6, registered
    bound 1e-4.
  - Whole-model drift, batched prefill vs sequential f32 decode,
    profile-registered bound 1e-2 (the f32 profile keeps its 2e-3):
    0.8B 7.4e-3 / 4.0e-3 and 2B 4.8e-3 / 2.3e-3, top-1 equal on both.
  - Register-vs-generic recurrence bound re-registered per profile: 2e-2
    under metal-f16-act (observed 6.8e-3 / 7.0e-3 on the 0.8B and
    1.07e-2 / 6.8e-3 on the 2B — operand rounding amplifies the two
    compilations' fast-math difference), 2e-3 with f32 operands
    (observed <= 2.0e-3).
  - `verify --reference` vs pinned llama.cpp (registered Q8_0 policy
    5e-1, `--kernels metal`, f16 KV): 0.8B PASS max 0.34; 2B PASS max
    0.337, top-5 overlap 1.00; Qwen3-0.6B PASS max 0.349; generated texts
    equal on all three.
  - Gemma 3 is unchanged by the profile: the comparison scores through
    per-token decode, and both profiles give byte-identical verify
    numbers (a pre-existing, recorded fact — the frozen-id CPU-side
    comparison fails Gemma's large logit scale under both profiles, which
    is why ADR-013 gates Gemma on the real-text Metal comparison
    instead). That gate is unchanged under metal-f16-act: max |diff|
    0.006-0.026 over 2-688 tokens and windowed decode, top-1 equal at
    every length, matching the ADR-013 record.
  - Fixture bit-identity gates are unaffected (fixture weights are F16,
    per-row path); MTP speculative greedy stays token-identical on the
    pinned 0.8B (the verify pass is exact per-row GEMV arithmetic).
- **Speed, measured (in-run A/B, interleaved).** Isolated GEMM at
  128-token batches: the f16-operand kernel is 2-3.5% faster than the f32
  kernel at every measured 0.8B/2B/Gemma shape, and matches llama.cpp
  exactly at the 0.8B shape (3584x1024: 452.3 vs 451.4 us, same window),
  replicating the earlier prototype finding. In the 2B model (GEMM is
  78-79% of prefill GPU time): GEMM GPU time 646-670 vs 675-796 ms
  (~4%); whole-prefill wall time is at the noise floor (754 vs 749 ms
  median interleaved) because 4% x 78% share is ~3%, inside the +-2-3%
  process noise with the desktop app resident. The request-level number is
  recorded by the standard bench rerun on a quiet machine, as before.
- **Tried and not kept:** the pipelined block-load variant (raw bytes of
  block N+1 prefetched into registers before the staging barrier, the
  other structure ADR-011 recorded in llama's kernel) — no consistent win
  at 128-token chunks under load noise; deleted.
- **Residual, honestly recorded:** at cols >= 1152 (the 2B's 2048-col and
  Gemma's 1152/6912-col shapes) our kernel at f16 operands remains 2-5%
  behind llama's in the same measurement window, while the 1024-col 0.8B
  shapes match exactly. Unexplained; llama's `kernel_mul_mm` pipeline
  carries shape-specialization constants (`bci/bco/ne12/ne13/r2/r3`) that
  our ABI does not. A 128-row tile is a known dead end (register spill,
  the variant table above). Follow-up needs a quiet machine.

- **Request-level record (2026-11-22, quiet AC, `2d03884`, all rounds
  valid, zero swap growth).** `scripts/bench-baseline.sh`, 512/256, 3
  ABBA rounds, one model at a time, Claude app closed:
  - Qwen3.5-2B (3 GiB, default KV f16, `…-qwen35-2b-q8_0-f16op-quiet`):
    warm decode 39.82 vs 40.60 tok/s, paired 0.980 (0.982/0.980/0.980);
    warm TTFT 601 vs 587 ms, paired 1.02× — **prefill 98% of llama.cpp**,
    up from 94% (two AC runs with f32 operands) and 86% (noisy battery
    run). The Phase 1 prefill bar is met on the 2B in this run;
    reproduction pending by the repo's convention. Both engines were
    steady this run (ChineCore 600-602 ms, llama.cpp 587-588 ms).
  - Gemma 3 1B (f16 KV, ignore-EOS, `…-gemma3-1b-f16op-quiet`): warm
    TTFT 330 vs 290 ms, paired 1.14× (~88%, within the recorded 85-90%
    band — the residual is the 5% kernel gap at its 6912x1152 FFN
    shapes above); warm decode 65.04 vs 75.89 tok/s, paired 0.857 —
    llama.cpp's fastest recorded decode round (75.9-76.1 tok/s; it
    swings 61-76 across runs) while ChineCore is unchanged and steady
    (64.90-65.19, median 65.04, identical to every prior run). The
    profile does not touch decode.

- **Follow-up, same day: the edge stage's occupancy tax.** The f16-operand
  kernel's static 8 KB edge stage (`ot`) was allocated by every group,
  though full tiles never touch it; at ~32 KB threadgroup memory per core
  that halved resident groups (14 KB -> 2/core; 6 KB -> 5/core). Replacing
  it with per-SIMD-group 64-float stages (256 B each, 7 KB total) and
  storing full fragments straight from the accumulators:
  - isolated, same-window, 128-token batches: 6912x1152 890.5 vs llama
    874.8 us (1.018, was 1.052), 1152x6912 972.7 vs 954.7 (1.019, was
    ~1.04), and the small attention shapes moved AHEAD of llama
    (1024x1152 163.1 vs 169.4, 256x1152 70.9 vs 74.2, 1152x1024 169.3 vs
    170.5);
  - in the Gemma model, in-process: 512-token batched prefill wall
    411 -> 371-389 ms (-9.5% median), GEMM GPU time 337 -> 308-311 ms;
    the per-layer GEMM stack is now 1.6% behind llama (was 4.6%);
  - 2B shapes gained 1-1.5% in-run; its 98% prefill stands.
  - A first rewrite of the edge tail (a whole-tile fragment loop holding
    the accumulators across threadgroup barriers) was measured 15-25%
    SLOWER on full tiles - a never-executed tail had inflated the
    kernel's register allocation and cut occupancy for every launch.
    It was replaced before commit by the per-SIMD-group structure above;
    the lesson is recorded: never let a cold path hold live
    accumulators across group-wide barriers.
  - Qualification unchanged and re-run green: kernel vs f32 reference
    worst rel 2.9e-6 (the edge-path rewrite preserves values exactly);
    0.8B and 2B engine suites 19/19 each; the Gemma real-text Metal
    gate identical (max 0.0256, top-1 equal at every length).
  - The quiet-machine bench rerun records the request-level number.

- **Acceptance record, same day (quiet battery run, clean tree
  `5f242d0`, all rounds valid, zero swap,
  `…-gemma3-1b-f16op-tuned-quiet`).** Warm TTFT 326 vs llama.cpp 289 ms
  — paired 1.13x, prefill **~88.5%** (pairs 1.128 / 1.128 / 1.062);
  paired decode 0.860 (ChineCore steady 65.0 ± 0.2 as in every recorded
  run; llama.cpp swung to 62.5 tok/s in round 3). The >= 90% prefill
  bar is **not crossed** on Gemma 3 by the median. Correction, recorded
  plainly: the -9.5% quoted above from in-process profiles compared
  different machine-load windows and was load-contaminated; the
  request-level effect of the edge-stage fix is -1.2% warm TTFT
  (330 -> 326 ms), which matches the same-window kernel arithmetic
  (stack -2.65% x 79% GEMM share). With the GEMM stack now 1.6-1.9%
  behind llama, the remaining ~11% request-level gap sits in the
  non-GEMM stack and host dispatch (~21% of prefill: gelu 5.6%,
  attention 5.6%, rmsnorm 3.8%, axpy 3.3%, recurrence/conv ~5%), where
  llama.cpp and exo's Flint run fused stacked kernels with fewer
  launches. The next lever on this model is fused GeGLU/norm kernels —
  not further GEMM tuning.

- **Follow-up (2026-11-22): the fused post-block norm + residual add, and
  the merged Q/K norms.** Two structural cuts to the non-GEMM stack
  identified as Gemma 3's remaining prefill gap:
  - `ch_norm_add_f32`: one pass for the engine's post-block pair
    (`rmsnorm(out)` then `y += normed`, where y is the residual stream).
    The pair reads and writes the row three times; the fused kernel reads
    twice and writes once. Metal fast math contracts the fused
    multiply-add (MSL has no `precise`), which cost 1 f32 rounding
    versus the pair's device-memory round-trip; routing the normed
    product through an `optnone` no-inline helper (a real call boundary)
    restores bit-identity — measured 0 ulps against the sequential pair
    on every shape including the engine's in-place (output aliases the
    residual input) wiring, and this is what the chunk-vs-per-token
    bit-identity gates rely on (either mode may run fused). Measured,
    in-run, at the engine's shapes: 52.2 -> 31.4 us per 128x1152 row-set
    (-40%), 71.0 -> 50.3 us at 128x2048 (-29%); the first (contracted)
    version was 15-25% slower at decode rows and is recorded as
    tried-not-kept before the optnone fix.
  - `ch_qk_norm_f32`: one launch for the per-head Q and K RMS norms
    (rows [0, q_rows) for q, the rest for k); per-row arithmetic is
    exactly `ch_rmsnorm_f32`'s — bit-identical, pinned by a metal-crate
    gate. 208 fewer launches per 512-token prefill, 52 fewer per decode
    token.
  - Applied at the two prefill sites only (decode keeps the pair: at
    rows=1 the fusion is a wash and the decode records stay untouched).
  - The gemma3 fixture bit-identity gate caught a real wiring bug before
    any commit: the first fusion wrote the result into the block output
    instead of the residual stream (the pair accumulates into s.x), so
    prefill diverged completely; Qwen3.5's gates missed it because its
    layers take the no-norm branch. Fixed, gate green, and the unit gate
    now also covers the engine's aliased wiring.
  - Ungrouped 512-token Gemma profile after: the norm/elementwise stack
  collapsed from ~30 ms (rmsnorm 16.1 + axpy 13.9) to ~10 ms (norm_add
  7.2, rmsnorm 3.1 — the per-head norms left the rmsnorm row entirely);
  GEMM unchanged at 79-80%. Kernel-level arithmetic projects the quiet
  bench TTFT saving at ~6-7 ms (pair cost ~14.5 ms quiet x 40%, plus
  launch overhead) — a projection; the standard quiet bench run records
  the request-level number.
  - Qualification: metal suite 32/32 (two new gates), engine suites
    19/19 on the real 0.8B and 2B, the Gemma real-text Metal gate
    unchanged (max 0.0256, top-1 equal), full gate 510 tests.

- **Acceptance record for the fused norms (quiet battery run, clean tree
  `6abf2a8`, all rounds valid, zero swap,
  `…-gemma3-1b-fusednorm-quiet`).** Warm TTFT 326/326/325 ms —
  **identical to the run before the fusion** (paired 1.13x, prefill
  ~88.5%; pairs 1.128/1.128/1.126). Paired decode 0.862, ChineCore
  steady 65.2-65.4, llama.cpp in its fast mode (75.7-76.0). The bar is
  not crossed, and the projection (~6-7 ms off the TTFT) was wrong.
  **Recorded plainly: this is the second projection from ungrouped
  profile deltas to fail at the bench level.** Deltas measured in the
  ungrouped profiling regime are dominated by per-dispatch overhead
  (~25 us/op) that the grouped production encoder never pays; the
  fused norms' real production saving is 208 fewer launches/encodes per
  prefill (~1 ms), not the ungrouped stack's ~20 ms. The kernels are
  kept — bit-identical, strictly less memory traffic per row, fewer
  launches — but they are not the request-level lever. The remaining
  ~11.5% TTFT gap (326 vs 289 ms) lives in the grouped regime's other
  operators (chunk attention and GeGLU, each ~24 ms ungrouped) and
  host dispatch. The recorded next step is a grouped-regime profile
  before any further kernel work: no more projections from ungrouped
  deltas.

- **The grouped-regime instrument (2026-11-22): counter samples where the
  GPU allows them, ablation where it does not.** `set_profiling` now
  attributes per-operator GPU time inside grouped (shared-command-buffer)
  dispatches when the device supports dispatch-boundary counter sampling:
  a `MTLCounterSampleBuffer` (timestamp set, 16384 slots) takes a sample
  after every encoded dispatch, the deltas fold into the same profile
  slots at the fence, `abort_group` drops the discarded group's samples,
  and `grouped_profiling_supported()` reports the capability. Measured
  probe on this M3 (Apple G15 family): `AtDispatchBoundary` sampling is
  **not supported** — stage-boundary is, but the compute encoder's
  sample method asserts; the gate test therefore skips here with the
  capability stated, and the scaffolding stays for GPUs that do support
  it. The CI runner's paravirtualized GPU reports the same absence.
- **The measurement the bench needed: grouped-regime ablation** (temporary
  hooks, never committed; each class stubbed out of the production
  encoder, 5 runs per phase, best wall, group-fenced). Gemma 3 1B,
  512-token grouped prefill, two passes:
  - GEMM (all Q8_0 matrix-unit projections): **303.0 ms of 357.9 (84.7%)
    and 314.7 of 370.7 (84.9%)** — stable across passes;
  - attention chunk: 9.3-44.1 ms (2.6-11.9%, the wide band is the
    session's GPU-contention noise);
  - GeGLU: 7.2-12.6 ms (2.0-3.4%);
  - norms (pre/post RMSNorm + both norm-adds): <= 3.7%;
  - Q/K norms + RoPE + KV store: <= 0.5%;
  - embedding/head/copies/host-encode skeleton: 4.9-5.2 ms (~1.4%).
- **What this settles.** The premise that led to the fused-norm work —
  "the remaining ~21% of prefill is the non-GEMM stack" — is refuted in
  the production regime: it was the ungrouped profiler's per-dispatch
  overhead (~25 us per command buffer). The grouped prefill is ~85% GEMM,
  every other operator class is single-digit, and the norms the fusion
  improved are <= 3.7% of it — which is exactly why the fusion could not
  move the bench TTFT. The Gemma prefill bar is therefore a GEMM-side
  problem end to end: the isolated FFN kernels are 1.6-1.9% behind
  llama's (attention shapes ahead), which bounds per-shape kernel tuning
  at ~1.5% of prefill. The remaining headroom is the one ADR-011 recorded
  early and the ablation now confirms as the only lever left with real
  size: **llama.cpp's encoder overlaps independent nodes** (q/k/v and
  gate/up are independent GEMMs per layer; ggml dispatches them
  concurrently for a further ~5%+, while the unified engine's encoder
  is strictly serial). With GEMM at ~85% of the prefill, node-concurrent
  encoding — not further kernel work — is the recorded next lever.

- **Research correction to the "next lever" (2026-11-22).** The note above
  records node-concurrent encoding as the remaining prefill lever, from
  ADR-011's early observation that "concurrent encoding gives ggml a
  further ~5% on independent nodes". llama.cpp's own merged
  concurrency work (PR #19555, Feb 2026: graph-optimize reorder with
  memory-range tracking and encoder barriers, continuing #15929/#2309)
  measures what that lever is actually worth on dense models:
  gemma3 1B pp512 **0.98x** (slightly worse), tg32 1.04x; qwen3 0.6B
  pp512 1.02x; deepseek2 30B.A3B pp512 1.03x — 0-3% on dense prefill,
  meaningful only on MoE (qwen3next tg32 1.11x) and decode. That matches
  our own ablation arithmetic: the Gemma prefill is ~85% GEMM and the
  isolated GEMMs already run near the f16 matrix ceiling (~2.28
  TFLOP/s measured), so overlapping independent GEMMs has little idle
  capacity to fill. The recorded lever list reorders accordingly: the
  headroom other engines measurably exploit on Apple Silicon in 2026 is
  **decode-side** (Rec's fused whole-decode-pass Metal kernel claims
  1.4-2.6x vs llama.cpp; MetalRT claims 1.35-2.14x decode; MLX leads
  llama.cpp decode by ~9% and 27B prefill by ~21% via grouped GEMM
  tiling) and **workload-level** (prefix caching for shared-preamble
  agents — built here, bit-identical; KV pruning up to 1.75x at long
  contexts per arXiv 2610.31479, which also warns that speculative
  decoding can *reduce* Apple Silicon throughput under chat workloads —
  consistent with keeping ours opt-in). One concrete measured target on
  our side: Gemma decode is steady at 65.0 tok/s while llama.cpp's fast
  mode reaches ~76 tok/s (76 -> ~81 GB/s effective on a 100 GB/s part);
  profiling and closing that per-row GEMV gap at the kernel level is the
  next concrete step, ahead of any prefill concurrency work.

- **The Gemma decode-gap investigation (2026-11-22): the GEMV kernels are
  at parity; the gap is the decode attention; two attempts, one anomaly,
  honestly recorded.**
- **Isolated GEMV parity (the question as posed).** A wall-timing harness
  (CHINE_PROFILE=1, `q8_0_gemv_timing_when_requested`) measured our
  per-row Q8_0 GEMV against pinned llama.cpp `mul_mv` at the Gemma decode
  shapes, same window: 6912x1152 152 vs 156 us, 1152x6912 132 vs 135,
  1024x1152 33.5 vs 28.8, 256x1152 23.6 vs 22.9, head 262144x1152 4445 vs
  4554, and the 0.8B shapes likewise — **0.98-1.03x, parity**: the decode
  GEMV is not the 65-vs-76 gap.
- **The gap is the decode attention.** The ungrouped decode profile:
  `qwen35_attention_f16_chunk` 2.77-2.89 ms/token (26 calls, ~107 us per
  layer call) — one 128-thread group per (KV head, row) pulling ~5.4 GB/s
  on the KV range (per-thread whole-key loads, no cross-key latency
  hiding). A coalesced single-group rewrite measured worse (135 us: one
  key in flight); a split-K design (16 slice groups + merge kernel)
  measured its partial at 37.5 us/call and, after three real bugs were
  found and fixed by new CPU gates (a dropped small-head_dim guard, a
  per-thread partials fold, a stale head-grid merge), passed an
  11-geometry CPU reference at <= 1.5e-7 and cut the in-engine decode
  attention to 1.12-1.16 ms/token. It was **not shipped**: five
  cross-mode bit-identity gates fail with it, and one (the gemma3
  fixture) diverges by 2.67 logits in a way not reproduced at the op
  level (the exact fixture geometry passes the unit gate at 1.2e-7) —
  unresolved after extensive bisection. Pulled rather than shipped
  unexplained.
- **The anomaly that stopped the simpler fix.** While profiling, the
  attention/rope chunk dispatches were found to overdispatch 128x their
  apparent mapping (`kv_heads x 128` / `heads x 128` threadgroups where
  the kernels read head = tg.x), and bounds-sentinel tests proved the
  excess groups write out of bounds. The "obvious" grid correction
  (one group per head) was attempted — and **breaks the kernels'
  outputs**: the old GQA attention at the corrected grid is 0.56 off the
  CPU reference, while the same kernel at the historical x128 grid
  matches it at 6e-8 (deterministic across runs; the engine oracle gate
  reproduces this: green at x128, Layer(3) off by 7.6e-3 at the
  corrected grid). The overdispatched groups are load-bearing in a way
  static analysis of the kernel source does not explain — recorded as
  an open anomaly. Consequence: the grid is the kernel's de-facto
  contract; any attention rewrite must treat x128 as the baseline and
  requalify against it. The sentinel bounds gate was removed with its
  disproven premise.
- **What shipped from this investigation:** the GEMV wall-timing harness
  (the parity record above), the decode-attention timing harness, and
  this diagnosis. The recorded next lever for the 65-vs-76 gap is a
  decode-attention kernel that qualifies against the x128 baseline —
  the split-K work here is the measured starting point (37.5 us partial,
  three found-and-fixed bugs, and the unresolved model-level divergence
  to chase).

- **The "x128 grid anomaly" is resolved (2026-11-22): it was never an
  anomaly.** `ch_dispatch_args` encodes with `dispatchThreads` - its
  MTLSize is TOTAL THREADS, not threadgroups - so the historical
  `kv_heads x 128` / `heads x 128` grids are the correct "N groups of
  128 threads" encoding of one-group-per-(head, row). There was no
  overdispatch and no out-of-bounds work from these grids; the previous
  note's "corrected" one-group-per-head grid passed
  `MTLSizeMake(kv_heads, n, 1)` as *total threads*, which Metal pads to
  a single partial threadgroup where only threads with
  thread_position_in_grid < gridSize execute - starving the kernel to
  kv_heads of its 128 threads (staging loops `for idx = tid; idx < ...;
  idx += 128` undone, one lane of the softmax) and producing the
  measured wrong output. Every observation fits and none other was
  needed: the historical grid matches the CPU reference at 6e-8; the
  starved grid is 0.56 off (unit) and Layer(3) 7.6e-3 off (oracle);
  the bounds-sentinel's first "OOB proof" was its own guard-length bug
  (it passed once fixed); and the gemma3 cross-mode divergence that
  stopped the split-K build was the rope grid "fix" - the same
  starvation applied to the prefill rope while the decode rope (a
  dispatchThreadgroups path) stayed correct. The previous note's claims
  that the overdispatch was load-bearing, that the grid is a "de-facto
  contract", and its sentinel-gate premise are retracted.
- **What this unlocks:** the split-K attention kernels from the
  investigation were correct (11-geometry CPU reference at <=1.5e-7
  after three found-and-fixed bugs) and were blocked only by
  grid-starvation side effects. They ship now, redesigned for
  cross-mode bit-identity: the slice count is a pure function of the
  row's absolute position (`slices_t = clamp(ceil(total_t/32), 1, 16)`
  from `seq_t` and `window`), so a given position's reduction order is
  identical in chunked prefill and per-token decode - the bit-identity
  gates keep their exact-bit assertions unchanged.
- **The shipped shape is llama.cpp's own split, and the acceptance
  number is measured (2026-11-22).** Split-K serves n = 1 (decode);
  n > 1 keeps the one-group-per-row kernel, so prefill cost and its
  record are unchanged (the isolated n=128-at-full-slices case was
  1686 us vs ~230 us - a bench prefill regression that this split
  avoids entirely). Where the two kernels' reduction orders meet in a
  comparison, the gates stay exact-bit; where they cross modes, they
  are drift-registered per the metal-f16-act precedent (1e-2 x
  max|logits|, top-1 equal, enforced on every GPU, drift printed:
  5.96e-8 unit cross-kernel, <= 7.6e-4 gemma3 fixture, MTP state
  token-identity exact). Acceptance, quiet machine, app closed, zero
  swap, clean tree `118a711`
  (`20261122T022842Z-…-gemma3-1b-splitk-quiet3`): decode
  **74.13/74.21/74.22 tok/s** - the 65.0 ± 0.2 steady of every prior
  run moved +14.2%, exceeding the ~71.5 projection from the -60%
  in-engine attention cost - paired median **1.067** (0.977 / 1.197 /
  1.067; >= 0.9 against llama.cpp's fastest round of the three). For
  the first time on Gemma 3 the decode pair is at-or-ahead; prefill
  unchanged at 324 vs 290 ms (paired 1.12x, ~89%, the >= 90% bar
  still open and bounded at ~1.5% by the grouped ablation). Two
  same-code runs under active-session load
  (`…-splitk-quiet`, `…-splitk-quiet2`) are recorded as evidence of the
  load's effect, not acceptance: both engines ~20-25% below quiet,
  llama.cpp hit hardest, paired decode 1.033/1.031.

- **Rust-engine landscape check (2026-11-22): candle and mistral.rs,
  from their own sources.** What the two big Rust projects are, and
  what the comparison means for this repo's positioning.
- **candle** (huggingface, 21.4k stars): a general minimalist ML
  *framework* — PyTorch-like tensors, training, a broad model zoo
  (vision/audio/diffusion/LLM), safetensors/npz/pytorch/ggml formats,
  CPU/CUDA/WASM backends plus `candle-metal-kernels`. Not a product:
  no server, scheduler, admission, or exactness layer. Its Apple
  Silicon quantized-inference story validates this repo's ADR-010:
  candle's Metal decode was far behind llama.cpp until PR #2615
  (2025-07) adopted ggml's matrix-matrix kernel designs — "within 10%
  of llama.cpp" on prefill by their own report, a user-measured 100.6
  vs 82.8 tok/s avg on M3 Max after the PR. Their speed on this
  hardware came from sourcing kernels from the reference — the exact
  decision ADR-010 records here — and their recent Metal work (fused
  MoE kernels, encoder reuse, PR #3444 2026-04) mirrors our grouped
  encoder. Nothing here changes our kernel plan; it is external
  confirmation of it.
- **mistral.rs** (7.9k stars, built on candle): the full *product*
  surface — OpenAI + Anthropic APIs, web UI, Python/Rust SDKs, 45+
  architectures, multimodal (vision/video/audio/speech/image-gen/
  embeddings), GGUF + ISQ/UQFF + GPTQ/AWQ/HQQ/FP8/BNB, LoRA/X-LoRA,
  AnyMoE, per-layer quantization topology, MCP/agents/tool-calling,
  multi-GPU/distributed, continuous batching, and PagedAttention
  (v0.9.5: **default on supported Metal models** with automatic
  memory planning). On CUDA/GB10 their UQFF-vs-GGUF numbers lead
  llama.cpp prefill 1.86x (Gemma 4 E4B) — but those artifacts are not
  identity-matched (different formats claimed comparable; our
  four-way identity rules would not allow that pairing). On **Apple
  Silicon their own tracking issue #903** (open 2024-11, updated
  2026-04) records mistral.rs behind llama.cpp at short/medium
  contexts (Llama-3.2-3B: TG 63.3 vs 75.3, PP 434 vs 533; 8B-class:
  TG 37.4 vs 38.9) and **ahead only at long-context decode** (TG5000
  35.4 vs 33.1) after adopting MLX-derived KV optimizations
  (#933 from mlx#1597). The third-party SiliconBench audit
  (2026-08, unified-memory desktops) shows mistral.rs's Metal
  concurrency fragile: c=1 fine, c=8 partial (40/100 requests),
  c=16 crashed on chat; several configs never start.
- **What that means here.** Where they are ahead of us: model breadth
  (45+ vs our 3 families), multimodality, quantization breadth
  (K-quants, FP8 — we ship Q8_0 + the ABI for Q4_0/Q4_1/Q8_0 only),
  LoRA, agents/MCP, multi-GPU, and — most relevant to our tier —
  **paged KV + long-context decode** (their TG5000 lead is the
  measured proof that KV paging pays at long contexts; our KV is a
  capacity-fixed resident plane) and **continuous batching** (we run
  one generation per model with a queue; our concurrency story is
  unmeasured). Where we hold measured ground none of them claims:
  identity-matched parity-or-ahead of llama.cpp on 1-2B decode at the
  8 GB tier (74.2 vs 69.6 medians, 2026-11-22), bit-exactness
  discipline (token-identical MTP, bit-identical prefix cache,
  reference-identity gates), typed admission before allocation, the
  macOS memory-pressure governor, and the honesty contract itself —
  no claims ledger or certified/estimated labels exist in either
  project. Two design imports to consider when the time comes, now
  recorded with sources: mistral.rs's MLX-derived long-context KV
  optimization (for our >2048-context decode) and v0.9.5's
  paged-KV-with-automatic-memory-planning (the ADR-006 constrained
  tier's streaming strategy); mistral.rs's ISQ/UQFF/AnyMoE features
  are out of contract here (requantization and model mutation are
  refused by ADR-001, never traded for speed).

- **Prefill M-scaling, shipped and accepted (2026-11-22).** The Gemma
  prefill gap decomposed by measurement: isolated GEMM parity at
  128-row batches was an artifact of the comparison shape - llama.cpp's
  request runs its prefill GEMMs at the full 512-row batch, and the
  matrix-unit kernel's M-scaling is where the gap lives (isolated,
  same window, ours f16op vs pinned ggml mul_mat: 6912x1152 ahead at
  128 rows, behind 1.028x at 512; 1152x6912 behind 1.199x at 512,
  llama +25% per token from 128 to 512 rows, ours +11%; small
  attention shapes llama +29-77%). An interleaved in-model A/B (one
  build, caller-side chunking) measured the structural change alone:
  512-token prefill 375.6 vs 384.4 ms on the loaded machine. Shipped
  as the one-line `PREFILL_BATCH` 128 -> 512 (every dependent derives
  from the constant; scratch 17.6 -> 56.1 MiB for Gemma, inside the
  planner's 192 MiB workspaces reservation, typed refusal as the
  safety net for a future model that outgrows it). Qualified at 512:
  the full gate (514), the Gemma real-text Metal reference gate
  (top-1 equal, drift unchanged), the pinned 0.8B gates (MTP token
  identity, prefix-cache bit identity, oracle). Acceptance, quiet
  machine, app closed, zero swap, clean tree
  (`20261122T030606Z-…-gemma3-1b-pb512-quiet`): warm TTFT
  **303/301/301 vs 306/289/289 ms, paired median 1.04x - prefill
  ~96%, the >= 90% bar met on Gemma 3 for the first time**, decode
  held (74.09 median, 0.977-0.979 against llama.cpp's fast rounds) -
  both Phase 1 bars met in one run. The request-level move (-23 ms
  from the 324 ms record) exceeds the loaded in-process A/B's -8.9 ms:
  the quiet machine pays the M-scaling gain in full. The remaining ~4%
  prefill gap is the at-512 kernel deficit (FFN down 1152x6912: ours
  3717 us / 2193 GFLOP/s vs llama 3101 / 2629), recorded as the
  follow-up lever: llama.cpp pipelines its K-loop (load-next-block
  while the mma runs on current) where our 64x32 tile re-reads W per
  token-tile behind barriers - kernel work with the register-pressure
  trap recorded below, to be measured isolated first.

- **The pipelining experiment, measured and reverted (2026-11-22).** The
  K-loop software pipeline shipped at `22f8c90` on isolated evidence:
  same-window, -7-11% at every deficit shape (FFN down 3717.4 ->
  3353.7 us at 512 rows, 1.199x -> 1.082x vs llama; k/v 1.062x; q
  1.091x; o 1.083x), both reference gates green, no occupancy-visible
  regression at the 0.8B shapes. The quiet acceptance run
  (`20261122T035251Z-…-gemma3-1b-pipe-quiet`, zero swap, clean tree)
  **reversed it**: warm TTFT 319/317/317 vs the pre-pipeline kernels'
  303/301/301 — **+16 ms, paired 1.10x vs 1.04x** — with the llama.cpp
  control reproducing its previous run exactly (289/290/289), so the
  regression is the kernel change alone. Decode unchanged (74.06).
  Reverted at `52e269c`; the standing acceptance records stay the pb512
  run (TTFT 301 ms, prefill ~96%; decode 74.09). Mechanism, recorded
  as hypothesis only: in the model the GEMMs interleave with
  attention/norm kernels in one grouped command buffer, where the
  prefetch registers held live across the mma section cost issue
  slots or occupancy; the isolated 16-identical-dispatch harness -
  on a loaded machine - rewards the same pattern. **Rule recorded:
  this is the third isolated/ungrouped projection to fail at the
  request level (fused norms, edge stage, now this) and the first to
  actively regress; kernel changes do not ship on isolated evidence
  alone - they need an in-model interleaved A/B (as the batch-512
  change had) or the quiet acceptance run before the commit counts
   as accepted.** The at-512 deficit (~4%) stays open with the llama
   `bci/bco` deeper-pipeline variants as the recorded follow-up.

- **Q4_0/Q4_1 Metal kernels (2026-11-22).** The matrix-unit GEMM set
  extended to the 4-bit pair — Q4_0 (18-byte block: `[f16 d][16 B
  nibbles]`, value `(nibble-8)*d`, low nibble -> j, high -> j+16) and
  Q4_1 (20-byte block: `[f16 d][f16 m][16 B]`, value `nibble*d + m`) —
  gemv, row, row-indirect, mid-batch gemm and both MMA profiles on the
  Q8_0 dispatch and admission surface; Q4_1 joins the MMA fast path
  (f16op under metal-f16-act). Eleven kernel gates against the CPU
  reference at real shapes (6912x1152, 1152x6912, 151936 rows;
  including the mid-batch gemm the routing audit found ungated —
  every Q4 kernel family is now gated); worst f16op relative 3.06e-6
  (Q4_0) / 1.221e-4 (Q4_1), bound 1e-3. Real-model proof: the
  ADR-013 real-text Metal gate parameterized over a second pinned
  artifact — unsloth gemma-3-1b-it Q4_0 (sha256 `27ee88e0…94b0276e`;
  361.5 MiB Q4_0 + 14.2 MiB Q4_1 + Q8_0 embeddings, the unsloth
  convention): ours vs llama.cpp Metal max |Δ| **0.0114** mean 0.0019
  at 2-688 tokens, top-1 equal, windowed decode max 0.0097 — tighter
  than the Q8_0 pin's own 0.0256/0.0044 in the same run (identical
  artifact bytes on both sides, so the 4-bit weight error is common
  to both engines and cancels; the residual isolates activation
  staging, unchanged). Re-derived and consistent with the standing
  matrix note: `verify --reference` against llama's CPU path cannot
  qualify gemma at any quant — HEAD's Q8_0 control fails it
  identically (max ~9.9, llama's own CPU-vs-Metal divergence) — so
  no registered CPU-reference tolerance is claimed for Q4-dominant
  artifacts; the typed refusal stands, exploratory runs only.

- **Q4_0 acceptance run and the decode gap's location (2026-11-22).**
  Quiet machine (battery, zero swap, all 3 ABBA rounds valid, clean
  tree `6ac210d`), gemma-3-1b-it Q4_0, 512/256, f16 KV, ignore-EOS:
  warm TTFT 312/312/313 vs 286/286/287 ms — **paired 1.09x, prefill
  ~92%, the >= 90% prefill bar met on the first Q4 artifact**; warm
  decode 86.21/86.39/86.43 (steady +/- 0.11) vs 102.11/102.53/102.15
  — **paired 0.844 (0.843-0.846), the >= 0.90 decode bar NOT met**.
  Footprint peak 938 MiB resident vs llama's 199 MiB mmap. Both
  engines decode the halved weights faster than their Q8_0 standing
  (ours 74.2 -> 86.4; llama ~75 -> 102), but llama gains 1.36x where
  we gain 1.16x. Correction of this note's first draft, which closed
  the model arithmetic on "388 MB/token" and mis-read the Q4_1 hint:
  the per-operator decode profile (the same CHINE_PROFILE instrument
  as the Qwen record, `profile_gemma3_decode_when_requested`) shows
  per token **Q4_0 gemv 9.56 ms = 56.6%** (gate/up 6912x1152 5.98 ms
  across 52 calls = 115 us = **38.9 GB/s**, down 1152x6912 2.01 ms =
  51.1 GB/s, k/v/q/o 1.57 ms), the tied Q8_0 head 262144x1152
  4.37 ms = **73.5 GB/s**, split-K attention 0.66 ms, norms/other
  ~2.9 ms — GPU busy 13.55 ms/token under session load (the quiet
  11.6 ms wall is the same distribution in a faster machine state).
  The Q4_0 gemv runs at half the head's streaming rate on the same
  memory system: the nibble unpack makes the kernel issue-bound on
  half the bytes. The earlier "Q4_1 gemv 59.9 GB/s, 24-byte blocks
  align" line was wrong twice over: the Q4_1 block is **20 bytes**
  (2+2+16, not 24 — the first timing harness over-allocated and
  over-counted it, inflating the print by 24/20; corrected, the Q4_1
  gemv sustains ~49.9 GB/s), and its kernel used the same scalar
  byte loads — alignment never explained the difference. llama's
  same-artifact decode (measured 102.15 tok/s = 9.8 ms/token) implies
  its Q4 dot stack sustains ~70 GB/s effective on the same traffic
  (derived, not isolated-measured). The concrete follow-up: the
  Q4_0 gemv load strategy (kernel session, same rules: no ship on
  isolated evidence, quiet acceptance decides).

- **The Q4_0 gemv load fix, measured in-model (2026-11-22).** The
  kernel's eight scalar byte loads per lane-row-block became two
  `packed_uchar4` loads (alignment 1, so the 18-byte block stride
  needs no special case) with the nibble math on uchar vectors —
  `(float4)(c & 0x0f) - 8.0f` / `(float4)(c >> 4) - 8.0f`; values and
  per-lane reduction order are identical to the scalar form (the
  real-text Metal gate is unchanged: max |d| 0.0149, top-1 equal).
  Two variants were tried and **rejected on the same instrument**:
  a quad-local simd_shuffle reusing the low lanes' bytes (gate/up
  5.05 -> 6.30 ms/token: the high lanes' independent loads overlap
  and coalesce, the shuffle serializes behind them) and a two-
  strided-blocks-per-iteration form (down 1.77 -> 1.95 ms/token, the
  most stable instrument in the table). The shipped single-block
  form, A/B'd in one session with the rmsnorm control stable
  (0.985 vs 0.986 ms/token): **gate/up 5.98 -> 5.05 ms/token
  (-15.5%), down 2.01 -> 1.77 (-12.1%)**, k/v 0.70 -> 0.66, GPU busy
  13.55 -> 13.01, wall 14.40 -> 14.03 (session load; quiet wall
  projects ~10.7 ms/token = ~94 tok/s = paired ~0.92 — a projection,
  the quiet acceptance run decides). Recorded measurement caveats:
  the isolated back-to-back gemv harness is unusable for cross-time
  A/B under session load (it showed the *unchanged* Q4_1 kernel a
  third slower forty minutes apart), and the small q/o shapes plus
  attention move +/- 25-50% between runs (contention band) — only
  the 52- and 23-call gemv entries and the rmsnorm control are
  stable enough to gate a decision. The timing harness's Q4_1 byte
  count is corrected to 20 (its GB/s print is now honest).

- **The gemv fix's quiet acceptance: the decode bar is met — at the
  bar, the thinnest margin recorded (2026-11-22).** Clean tree
  `12e52d4`, battery, zero swap, all 3 ABBA rounds valid
  (`20261122T085725Z-…-gemma3-1b-q4_0-gemvfix`): warm decode
  **91.48/91.79/91.73 (steady ± 0.15)** vs llama.cpp
  101.90/101.77/**86.93** — paired **0.898 / 0.902 / 1.055, median
  0.902 ≥ 0.90**. Honest reading: the two clean rounds (llama in its
  reproduced fast band, 101.77-101.90, identical to the pre-fix
  run's 102.11-102.53 control) pair at **0.898-0.902 — at the bar,
  not over it**; the median's third pair comes from llama's
  recorded round-to-round bimodality (its round 3 swung to 86.93
  while ChineCore held 91.73). The fix's request-level effect: warm
  decode 86.39 → **91.73 (+6.2%)**, within the in-model projection's
  reach (+8% projected, the quiet machine pays most of it). Warm
  TTFT 311/311/312 vs 286/286/288 — paired 1.09×, prefill ~92%,
  unchanged as designed (the gemv serves decode only). Footprint
  938 MiB resident vs llama's 199 MiB mmap peak. Standing: both
  Phase 1 bars are met on the Q4_0 artifact — prefill 92%, decode
  0.902 median with the caveat named; the margin is the thinnest in
  the matrix (Qwen3-1.7B's decode band was 0.989-0.991, Gemma Q8_0's
  worst pair 0.977), so it is recorded as "at the bar", never as a
  lead. The recorded next lever, if the bar is to be held with
  margin: the tied Q8_0 head (4.37 ms/token, 25.8% of decode GPU
  time, 73.5 GB/s) and the rmsnorm stack (~1 ms/token) — both
  shared with the Q8_0 artifact, so any gain pays on both pins.

- **The two levers, taken and measured — one closed, one small
  (2026-11-22).**
- **The head lever is closed by measurement, not by rewrite.** The
  73.5 GB/s of the acceptance-session profile was load tax: in a
  quieter machine state the same in-model head call runs 3.43 ms for
  320.9 MB = **93.6% of the M3's 100 GB/s DRAM spec**, and the
  isolated harness reproduces the standing parity number exactly
  (72.2 GB/s, the same window in which pinned llama.cpp measured
  0.98x ours). At the streaming ceiling with the reference at
  parity there is no demonstrated gap, so a rewrite is forbidden by
  the profiling rule; the head's 4.3 ms/token is DRAM physics (306
  MiB of tied Q8_0 head read every token by design). Recorded as
  closed.
- **The rmsnorm stack: the decode path now runs the fused
  post-block norm + residual add.** Decode was the last unfused
  consumer: prefill has run `norm_add` since the fusion work, and
  the cross-mode gates (prefill == decode bits) are themselves the
  pairing proof — wiring `Layer::decode` and `Ffn::decode` through
  the same kernel changes no bits and drops 52 dispatches per token
  (157 norm-class dispatches -> 105). The site is family-shared
  (GDN and attention mixers both flow through it), so every pinned
  model pays the saving. Interleaved A/B, same machine state:
  wall 14.09/13.85 vs 14.11/13.87 ms — **equal-or-better in every
  pair, ~0.1-0.15 ms/token** — honest size: the ops were already
  launch-bound-cheap, so this is a dispatch-count and encoding win,
  not the 0.3-0.5 ms the profile's raw norm-class total suggested;
  the small-op rows move +/- 2x between profile runs (contention),
  which is why the paired walls, not the per-op rows, gate the
  verdict. Verified: full gate, the 9 pinned 0.8B/MTP core gates,
  and the 25 e2e gates (both gemma Metal pins unchanged — the
  logits equalities re-prove bit-identity on real weights; the MTP
  accept rates unchanged). Projected quiet effect: 91.7 -> ~92.3
  tok/s, clean-round pairs ~0.91 — a modest margin over the 0.90
  bar, honestly sized; the quiet acceptance run decides.

- **The norm fusion's quiet verdict: a ~0.5% regression — reverted;
  the margin levers are closed (2026-11-22).** The quiet acceptance
  run (`20261122T092018Z-…-gemma3-1b-q4_0-normfuse`, battery, zero
  swap, clean tree `483c6a1`) returned warm decode
  **91.28/91.29/91.33 (± 0.03)** against the gemvfix run's
  91.48/91.79/91.73 — the two runs' bands do not overlap, while
  llama's control reproduced its fast mode exactly (102.45 vs
  101.77-102.53): the pipelining-run signature (control identical,
  ours moved). The loaded in-model A/B that shipped the fusion read
  "equal-or-better in every pair" — inside its +/-2% noise floor;
  **rule recorded: a loaded in-model A/B cannot resolve effects
  under ~2%, only the quiet acceptance can.** This is the fourth
  projection to fail at the quiet level (prefill fused norms, edge
  stage, pipelining, now this). The fusion's mechanism, recorded
  as the working hypothesis: at rows=1 the fused kernel is not
  strictly less work — the residual read and the no-inline
  bit-identity helper cost more GPU time than the launch-bound
  rmsnorm + axpy pair they replaced, and the 52 saved dispatches
  did not pay on the quiet machine. Reverted; the decode sites keep
  a comment naming the run. The run's paired median 1.041 is
  llama's bimodality (rounds 2-3 in its 87.4-87.7 slow mode), not a
  gain on ours — recorded plainly. **Final standing of the two
  levers:** the head is DRAM-closed (93.6% of spec, reference at
  parity), the norm stack is measured at its floor — the Q4_0
  decode stands at-the-bar (clean fast-mode pairs 0.891-0.902
  across the three quiet runs, ours steady at 91.3-91.8) and the
  recorded boundary is now honest: holding the bar *with margin*
  requires a Q4_0 gemv redesign that changes the per-lane reduction
  order (the cross-mode bit-gate risk chain, rejected twice this
  arc on the instrument) or a lever outside the kernel set; no
  further in-place margin work is available at these shapes.

- **The Q4_0 gemv reduction-order redesign, measured and shipped
  shape-adaptive (2026-11-22).** The recorded boundary named this
  work; taking it measured two things straight. First, the
  recorded "cross-mode bit-gate risk chain" was overcautious: **no
  gate pins Q4_0 gemv bits** — the batch-1-vs-gemv bit gate is
  Q8_0-only, the engine fixtures route F16/Q8_0, and every Q4_0 bar
  is tolerance-based (the 1e-4 CPU reference, the llama.cpp Metal
  envelope). The earlier "rejected twice on the instrument" were
  performance rejections (simd_shuffle, pair-step), not gate
  blockers. Second, the llama.cpp kernel_mul_mat_vec structure —
  one row per SIMD group, lanes carrying whole 32-value blocks
  strided by 32 — measured **split by shape**: on the low-row decode
  shapes it wins (down 1152x6912 1.70-1.78 -> **1.35-1.65
  ms/token**, o 1152x1024 0.37-0.49 -> 0.31) because the multi-row
  form's rows/8 groups (144 for down) cannot hide DRAM latency; on
  the high-row shapes it loses badly (gate/up 6912x1152 4.75-5.74
  -> 7.3-8.1) because 864 groups already saturate DRAM and the
  one-row form throws away the x-load amortization across rows.
  **The ship is both kernels with wrapper routing** (rows >= 2048
  multi-row, below one-row; the measured shapes bracket the
  threshold), no Rust API change — the routing lives in the native
  wrapper. Requalified: all 12 gemv gates green across both routes
  (the reduction reorder lands ~1e-6 relative, well inside the
  1e-4 CPU bound); the real-text Metal gate on the Q4_0 pin
  unchanged in envelope (max |d| 0.0149, top-1 equal, windowed
  decode 0.0102); full gate + the 9 pinned core gates + 25 e2e
  gates green. In-model net: ~-0.3 ms/token concentrated on the
  stable down instrument — honest size, and unlike the norm-fusion
  projection its mechanism is measured (144 -> 576 groups), not
  assumed. Quiet projection: 91.7 -> ~94 tok/s, fast-mode pairs
  ~0.92 — the quiet acceptance run decides, per the rule the norm
  fusion taught: loaded in-model walls cannot resolve <2% effects.

- **The shape-adaptive gemv's quiet acceptance: the decode bar met
  with margin in every pair (2026-11-22).** The first attempt
  (`20261122T144922Z-…-gemv2`) was **thermal-invalid and discarded**:
  both engines uniformly ~6x below their standing bands (llama's
  control 18.77 vs 101.8-102.5 warm tok/s — a 5.4x deviation no code
  change could produce), zero swap, healthy reclaimable — the
  machine-state signature; no verdict drawn, no record written.
  The re-run on the standing machine (`20261122T150115Z-…-gemv2`,
  AC, zero swap, clean tree `9aff576`, llama reproducing its fast
  band exactly, all 3 ABBA rounds in that mode): warm decode
  **94.22/95.26/95.65 (steady +/- 0.7)** vs 101.17/102.19/102.01 —
  **paired 0.931 / 0.932 / 0.938, median 0.932** — every clean
  fast-mode pair above the >= 0.90 bar with ~3-4 points of margin,
  the first time on the Q4_0 artifact (the pre-redesign standing
  straddled it: 0.891-0.902). The redesign's request-level effect:
  91.73 -> 95.26 warm median (**+3.9%**, vs the in-model
  projection's +2.9%). Warm TTFT 313/312/312 vs 287/286/287 —
  paired 1.09x, prefill ~92%, unchanged as designed (the gemv
  serves decode only). Footprint 938 MiB resident vs llama's
  199-200 MiB mmap peak. Standing, honestly stated: the Q4_0
  decode bar is now held with margin by measurement — the margin's
  mechanism is the 144 -> 576-group parallelism on the low-row
  shapes, not a llama-side swing (llama never left its fast mode
  this run). The recorded residual, if more is ever wanted: the
  Q4_0 gemv class still streams below the head's 93.6% DRAM
  ceiling on the high-row shapes where the multi-row form stays
  (gate/up 5.09-5.62 ms/token), and every further in-place
  variant measured this arc (shuffle, pair-step, one-row at high
  rows) lost on the instrument.

- **K-quants execute: Q4_K, Q5_K, Q6_K through the full Metal stack
  (2026-11-22).** The dequant semantics are pinned to the llama.cpp
  b11695 source (`ggml-common.h` block structs, `ggml-quants.c`
  dequantize_row_q4_K/q5_K/q6_K and get_scale_min_k4 — fetched from
  the pinned revision, never hand-derived): 256-value super-blocks
  with 6-bit packed (scale, min) pairs (Q4_K/Q5_K) or int8 per-16
  scales (Q6_K). Eight kernels per family (gemv, row, row-indirect,
  mid-batch gemm, MMA f32-operand, MMA f16op) on the Q4_0 arc's
  dispatch surface; the Q5_K set was pulled in because the target
  artifact carries it (see below), and one more format came with it:
  unsloth's K-quant convention stores the small GDN projections
  (ssm_alpha/beta, ssm_conv1d) as **F32 matrices**, which the
  engine's Matrix path had never carried — a minimal f32 gemv/gemm
  pair (one row per threadgroup) completes the artifact's type set.
  Two bugs found by the gates and fixed before any real artifact
  ran: the Q4_K kernels read `qs` at block offset 4 (the scales
  region — the CPU gate caught the permuted values) and the Q5_K MMA
  staged `qh[i]` where the element's byte is `qh[w_k + i]`.
  **Gates:** 16 kernel gates (CPU references transcribing the pinned
  formulas; the Qwen3.5-0.8B Q4_K_M geometry plus vocabulary and
  tail shapes), worst f16op relative 7.08e-4 (Q4_K) / 7.52e-4
  (Q5_K) / 2.94e-4 (Q6_K), bound 1e-3; the half-domain generators
  keep staged values finite (Q5_K's bound was tightened twice —
  448*63*31 and 16*128*32 both overflow half before the fix).
  **Real-artifact qualification:** the pinned unsloth
  Qwen3.5-0.8B-Q4_K_M (sha256 pinned at import; the Q8_0 pin's own
  source repo carries no Q4_K_M) — type mix F32 4.2 + Q4_K 196.0 +
  Q5_K 24.8 + Q6_K 286.7 MiB + Q8_0 2.1, all executed, nothing
  refused: `verify --reference` **PASS (calibration)** with texts
  equal, top-5 overlap 0.90, max |d| 1.045 — the K-quant envelope is
  ~3x Q8_0's 0.34 (the 6-bit scale arithmetic amplifies the
  CPU-vs-Metal activation gap); the registered policy for
  K-dominant artifacts is 1.5 (the same margin convention as Q8_0's
  0.5 over 0.34), registered from this measurement. The CLI runs
  the artifact resident on Metal; speculative decoding is
  token-identical to plain greedy on a 96-token continuation. The
  honest record tag derives `q6_k` (the dominant weight bytes) —
  Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K now all label correctly.
