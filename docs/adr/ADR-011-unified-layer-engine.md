# ADR-011: One unified layer engine for every model family

Status: **accepted** (2026-10-26). Extends ADR-005 (TuckCore behind
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
- The CLI, worker protocol and planner contracts do not change; TuckCore
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
- **2B evidence:** `tuckmill run` (unified engine) and pinned llama.cpp
  Metal produce the same 16 greedy tokens for "The capital of France
  is".
- Next: step 2 (dense Qwen3 on the same blocks), and the graph-driven CPU
  oracle.

## Engine note (2026-10-26): greedy decoding on the device, pipelined

- `tm_metal_argmax_f32` chooses the greedy token on the GPU with exactly
  the CPU rule (`sampling::argmax`: NaN skipped, ties to the lowest index;
  the order is associative, so the two-pass reduction gives the
  sequential answer). It writes one of two alternating token slots.
  `tm_metal_row_indirect_f32` embeds a token id read from such a slot; the
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
  `tuckmill bench` (128 in / 256 out, ABAB): 56.5 → 64.0 and 51.0 → 59.0
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
    Attention uses one shared inline body (`tm_q35_attend`) for the
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
  `tm_qwen35_gdn_chunk_k128` (a template instance for key dim 128) keeps
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

- `tm_q8_0_gemm_mma` now owns a 32-row x 32-token tile (4 accumulators per
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
- `tm_q8_0_gemm_mma` now covers a whole batch in one launch: a 1-D grid
  of 32-row x 32-token tiles, with neighbouring threadgroups sharing a
  weight tile (batch up to `TM_METAL_GEMM_MMA_MAX_BATCH` = 512). Per-tile
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

Isolated timing (`q8_0_gemm_batch_timing` with `TUCKMILL_PROFILE=1
TUCKMILL_GEMM_MMA=1`, now batches 32 and 128 after a warm-up, 100
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
- `tm_qwen35_gdn_chunk_k128` now splits each column across P = 8 lanes
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
- `tm_qwen35_attention_gqa<HD, G>` runs one threadgroup per (KV head,
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
  - TuckCore: `q8_0_gemm_mma_grouped_wall_timing_when_requested`.
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
- **`tm_q8_0_gemm_mma` now uses that structure in f32.** Isolated time at
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
  before multiplying; TuckCore keeps f32 (the oracle-f32 profile).
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
  - f32 KV uses new chunk kernels (`tm_qwen35_kv_store_f32_chunk`,
    `tm_qwen35_attention_f32_chunk`, GQA instances for float caches).
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
  - `crates/tuckcore/src/gpu.rs`;
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
- **Kept, both in `tm_qwen35_attention_gqa`:**
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
- **Kernel.** `tm_q8_0_gemm_mma_f16op` (`native/metal/kernels.metal`):
  same 64x32 fragment tiling, layout, summation order and edge handling as
  `tm_q8_0_gemm_mma`; sa/sb stage as half, the edge output stage keeps its
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
    steady this run (TuckCore 600-602 ms, llama.cpp 587-588 ms).
  - Gemma 3 1B (f16 KV, ignore-EOS, `…-gemma3-1b-f16op-quiet`): warm
    TTFT 330 vs 290 ms, paired 1.14× (~88%, within the recorded 85-90%
    band — the residual is the 5% kernel gap at its 6912x1152 FFN
    shapes above); warm decode 65.04 vs 75.89 tok/s, paired 0.857 —
    llama.cpp's fastest recorded decode round (75.9-76.1 tok/s; it
    swings 61-76 across runs) while TuckCore is unchanged and steady
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
  paired decode 0.860 (TuckCore steady 65.0 ± 0.2 as in every recorded
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
  - `tm_norm_add_f32`: one pass for the engine's post-block pair
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
  - `tm_qk_norm_f32`: one launch for the per-head Q and K RMS norms
    (rows [0, q_rows) for q, the rest for k); per-row arithmetic is
    exactly `tm_rmsnorm_f32`'s — bit-identical, pinned by a metal-crate
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
  ~88.5%; pairs 1.128/1.128/1.126). Paired decode 0.862, TuckCore
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
  (TUCKMILL_PROFILE=1, `q8_0_gemv_timing_when_requested`) measured our
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
