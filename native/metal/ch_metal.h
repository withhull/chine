/*
 * ch_metal.h — ChineCore <-> Metal narrow C ABI (ADR-007).
 *
 * One context owns one MTLDevice, one MTLCommandQueue, and the compiled
 * compute pipelines for the operations declared here. Buffers are opaque
 * resource handles with explicit lifetime; dispatch is asynchronous and
 * completion-fenced through ch_metal_fence(). Measurements (committed and
 * completed dispatch counts, last GPU time, live allocated bytes) are
 * exposed so the caller can account for real GPU resources.
 *
 * All functions are thread-safe in the sense that a context may be used
 * from one thread at a time; completion counters are atomic and may be
 * observed from any thread. Return 0 on success, a CH_METAL_ERR_* code on
 * failure, with a human-readable message in `err` when provided.
 *
 * This is NOT the worker ABI (native/executor/chine_worker_abi.h); only
 * ChineCore uses this interface.
 */
#ifndef CH_METAL_H
#define CH_METAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ch_metal_ctx ch_metal_ctx;
typedef struct ch_metal_buffer ch_metal_buffer;

enum {
    CH_METAL_OK = 0,
    CH_METAL_ERR_NO_DEVICE = -1,   /* no Metal device is available */
    CH_METAL_ERR_SOURCE = -2,      /* MSL compilation failed */
    CH_METAL_ERR_ALLOC = -3,       /* buffer allocation failed */
    CH_METAL_ERR_BOUNDS = -4,      /* size/range mismatch, zero length */
    CH_METAL_ERR_UNSUPPORTED = -5, /* invalid arguments / mode */
    CH_METAL_ERR_INTERNAL = -6     /* Metal/Objective-C runtime failure */
};

/* 1 when a Metal device exists (cheap; no context required). */
int ch_metal_available(void);

/* One-line runtime description (device name, unified memory size). Always
 * returns a valid static string. */
const char *ch_metal_runtime_info(void);

/* Create a context: device, queue, library compiled from `msl_source`, and
 * the pipeline states for every operation below. On failure returns NULL
 * and fills `err`. */
ch_metal_ctx *ch_metal_create(const char *msl_source, char *err, size_t err_cap);

/* Destroy the context. All buffers created from it must have been freed
 * first; the caller (Rust RAII handles) guarantees this. */
void ch_metal_destroy(ch_metal_ctx *ctx);

/* Allocate a buffer of `bytes` (> 0). `shared` buffers support CPU access
 * and host<->device copies; private buffers are GPU-only. Returns NULL on
 * failure. */
ch_metal_buffer *ch_metal_buffer_new(ch_metal_ctx *ctx, size_t bytes, int shared,
                                     char *err, size_t err_cap);

/* Release a buffer. Passing a buffer not owned by the context is a
 * programming error and reports CH_METAL_ERR_UNSUPPORTED. */
void ch_metal_buffer_free(ch_metal_ctx *ctx, ch_metal_buffer *buffer);

/* CPU pointer into a shared buffer (NULL for private buffers or if the
 * context does not own the buffer). Valid until the buffer is freed. */
void *ch_metal_buffer_contents(ch_metal_ctx *ctx, ch_metal_buffer *buffer);

/* Length of the buffer in bytes (0 when the handle is invalid). */
size_t ch_metal_buffer_len(const ch_metal_ctx *ctx, const ch_metal_buffer *buffer);

/* Zero `len` bytes of a buffer at byte offset `off`, without a host
 * staging allocation: shared storage is zeroed in place, private storage
 * uses a GPU fill. */
int ch_metal_buffer_zero(ch_metal_ctx *ctx, ch_metal_buffer *buffer, size_t off, size_t len,
                         char *err, size_t err_cap);

/* Host -> device and device -> host copies through shared storage. */
int ch_metal_upload(ch_metal_ctx *ctx, ch_metal_buffer *dst, size_t off,
                    const void *src, size_t len, char *err, size_t err_cap);
int ch_metal_download(ch_metal_ctx *ctx, ch_metal_buffer *src, size_t off,
                      void *dst, size_t len, char *err, size_t err_cap);

/* y[i] = x[i] * factor, i < n. n may be any length (tail-safe). */
int ch_metal_f32_scale(ch_metal_ctx *ctx, ch_metal_buffer *x, ch_metal_buffer *y,
                       float factor, uint32_t n, char *err, size_t err_cap);

/* y[i] = y[i] + alpha * a[i], i < n. */
/* Greedy argmax of x[0..n) into out[0] (u32), exactly the CPU rule: NaN
 * skipped, largest value wins, ties to the lowest index; 0xFFFFFFFF when
 * every entry is NaN, written to out[out_index]. `partial` is 512 bytes of
 * scratch. Encodable inside a dispatch group. */
int ch_metal_argmax_f32(ch_metal_ctx *ctx, ch_metal_buffer *x, uint32_t n,
                        ch_metal_buffer *partial, ch_metal_buffer *out, uint32_t out_index,
                        char *err, size_t err_cap);

/* out[i] = row ids[slot] of matrix `w` (type 8 = Q8_0, 1 = F16; `rows` x
 * `cols`), dequantized to f32: an embedding lookup whose token id was
 * produced on the device. An id >= rows (the argmax all-NaN marker) writes
 * zeros. Encodable inside a dispatch group. */
int ch_metal_row_indirect_f32(ch_metal_ctx *ctx, ch_metal_buffer *w, uint32_t type_id,
                              uint32_t rows, uint32_t cols, ch_metal_buffer *ids, uint32_t slot,
                              ch_metal_buffer *out, char *err, size_t err_cap);

/* dst[dst_off + i] = src[src_off + i] for i < n (f32 elements, bit-exact),
 * encodable inside a dispatch group. Offsets and n are checked against
 * both buffers. */
int ch_metal_copy_f32(ch_metal_ctx *ctx, ch_metal_buffer *src, uint32_t src_off,
                      ch_metal_buffer *dst, uint32_t dst_off, uint32_t n, char *err,
                      size_t err_cap);

int ch_metal_f32_axpy(ch_metal_ctx *ctx, ch_metal_buffer *a, ch_metal_buffer *y,
                      float alpha, uint32_t n, char *err, size_t err_cap);

/* y[i] = (float) x[i], i < n, where x holds IEEE binary16 values. */
int ch_metal_cast_f16_f32(ch_metal_ctx *ctx, ch_metal_buffer *x, ch_metal_buffer *y,
                          uint32_t n, char *err, size_t err_cap);

/* y[i] = (half) x[i], i < n, round-to-nearest-even. */
int ch_metal_cast_f32_f16(ch_metal_ctx *ctx, ch_metal_buffer *x, ch_metal_buffer *y,
                          uint32_t n, char *err, size_t err_cap);

/*
 * Grouped encoding (C2): operations between begin_group/end_group encode
 * into ONE command buffer committed at end_group, instead of one commit per
 * operator. abort_group discards the open group's work without committing
 * (cancellation at a step boundary; no partial GPU work is submitted).
 * Copies and fence require no open group.
 */
int ch_metal_begin_group(ch_metal_ctx *ctx, char *err, size_t err_cap);
int ch_metal_end_group(ch_metal_ctx *ctx, char *err, size_t err_cap);
int ch_metal_abort_group(ch_metal_ctx *ctx, char *err, size_t err_cap);

/* Dequantize row `row` of a packed Q8_0 matrix to `cols` f32 values. */
int ch_metal_q8_0_row_f32(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *out,
                          uint32_t row, uint32_t cols, uint32_t out_off, char *err, size_t err_cap);
int ch_metal_q4_0_row_f32(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *out,
                          uint32_t row, uint32_t cols, uint32_t out_off, char *err, size_t err_cap);
int ch_metal_q4_1_row_f32(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *out,
                          uint32_t row, uint32_t cols, uint32_t out_off, char *err, size_t err_cap);

/* Q4_K / Q6_K (K-quants, pinned to llama.cpp b11695): 256-value
 * super-blocks, cols a positive multiple of 256. Q4_K bytes per block
 * 144, Q6_K 210. */
int ch_metal_q4_k_gemv(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                       ch_metal_buffer *y, uint32_t rows, uint32_t cols,
                       char *err, size_t err_cap);
int ch_metal_q6_k_gemv(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                       ch_metal_buffer *y, uint32_t rows, uint32_t cols,
                       char *err, size_t err_cap);
int ch_metal_q4_k_row_f32(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *out,
                          uint32_t row, uint32_t cols, uint32_t out_off, char *err, size_t err_cap);
int ch_metal_q6_k_row_f32(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *out,
                          uint32_t row, uint32_t cols, uint32_t out_off, char *err, size_t err_cap);
int ch_metal_q4_k_gemm(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                       ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                       uint32_t x_off, uint32_t y_off, char *err, size_t err_cap);
int ch_metal_q6_k_gemm(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                       ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                       uint32_t x_off, uint32_t y_off, char *err, size_t err_cap);
int ch_metal_q4_k_gemm_mma(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                           ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                           uint32_t x_off, uint32_t y_off, char *err, size_t err_cap);
int ch_metal_q4_k_gemm_mma_f16op(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                                 ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                                 uint32_t x_off, uint32_t y_off, char *err, size_t err_cap);
int ch_metal_q6_k_gemm_mma(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                           ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                           uint32_t x_off, uint32_t y_off, char *err, size_t err_cap);
int ch_metal_q6_k_gemm_mma_f16op(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                                 ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                                 uint32_t x_off, uint32_t y_off, char *err, size_t err_cap);
int ch_metal_q5_k_gemv(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                       ch_metal_buffer *y, uint32_t rows, uint32_t cols,
                       char *err, size_t err_cap);
int ch_metal_q5_k_row_f32(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *out,
                          uint32_t row, uint32_t cols, uint32_t out_off, char *err, size_t err_cap);
int ch_metal_q5_k_gemm(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                       ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                       uint32_t x_off, uint32_t y_off, char *err, size_t err_cap);
int ch_metal_q5_k_gemm_mma(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                           ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                           uint32_t x_off, uint32_t y_off, char *err, size_t err_cap);
int ch_metal_q5_k_gemm_mma_f16op(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                                 ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                                 uint32_t x_off, uint32_t y_off, char *err, size_t err_cap);

/* F16 matvec: y[r] = dot(f32(w_row_r), x); `w` holds rows*cols f16 values. */
int ch_metal_f16_gemv(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                      ch_metal_buffer *y, uint32_t rows, uint32_t cols,
                      char *err, size_t err_cap);
/* F32 matvec: y[r] = dot(w_row_r, x); `w` holds rows*cols f32 values
 * (the small GDN projections of unsloth K-quant artifacts). */
int ch_metal_f32_gemv(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                      ch_metal_buffer *y, uint32_t rows, uint32_t cols,
                      char *err, size_t err_cap);
int ch_metal_f32_gemm(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                      ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                      uint32_t x_off, uint32_t y_off, char *err, size_t err_cap);

/* y[row] = rmsnorm(x[row], weight, eps) for `rows` rows of `cols`. */
int ch_metal_rmsnorm_f32(ch_metal_ctx *ctx, ch_metal_buffer *x, ch_metal_buffer *weight,
                         ch_metal_buffer *y, uint32_t rows, uint32_t cols, float eps,
                         char *err, size_t err_cap);

/* y[row] = rmsnorm(x[row], weight) + res[row] (the fused post-block
 * norm + residual add; one threadgroup of 128 per row, mirroring
 * ch_metal_rmsnorm_f32's arithmetic). */
int ch_metal_norm_add_f32(ch_metal_ctx *ctx, ch_metal_buffer *x, ch_metal_buffer *weight,
                           ch_metal_buffer *res, ch_metal_buffer *y, uint32_t rows,
                           uint32_t cols, float eps, char *err, size_t err_cap);

/* One launch for the per-head Q/K RMS norms: rows [0, q_rows) normalize q
 * with q_weight, rows [q_rows, q_rows + k_rows) normalize k with
 * k_weight; bit-identical to the two rmsnorm launches it replaces. */
int ch_metal_qk_norm_f32(ch_metal_ctx *ctx, ch_metal_buffer *q, ch_metal_buffer *q_weight,
                          ch_metal_buffer *q_out, ch_metal_buffer *k, ch_metal_buffer *k_weight,
                          ch_metal_buffer *k_out, uint32_t q_rows, uint32_t k_rows,
                          uint32_t cols, float eps, char *err, size_t err_cap);

/* NeoX RoPE in place over n_q_heads of q and n_kv_heads of k at `pos`,
 * using host-precomputed inv_freq[head_dim/2]. */
/* k_cache[h][pos][:] = k_src[h][:], same for v; `round_f16` stores
 * f16-representable f32 values (the declared f16 cache contract). */
/* Dequantize row `row` of an F16 matrix to `cols` f32 values. */
int ch_metal_f16_row_f32(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *out,
                         uint32_t row, uint32_t cols, uint32_t out_off, char *err, size_t err_cap);

/* ---------------- C3: batched prefill operations ---------------- */

/* Y[b, :] = W . X[b, :] for b < batch (1..=32); X is [batch, cols] at f32
 * element offset `x_off`, Y is [batch, rows] at `y_off`. Per-token
 * arithmetic order matches the single-token GEMV, so batch=1 is
 * bit-identical to it. */
int ch_metal_q8_0_gemm(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                       ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                       uint32_t x_off, uint32_t y_off, char *err, size_t err_cap);
int ch_metal_q4_0_gemm(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                       ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                       uint32_t x_off, uint32_t y_off, char *err, size_t err_cap);
int ch_metal_q4_1_gemm(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                       ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                       uint32_t x_off, uint32_t y_off, char *err, size_t err_cap);
/* Largest batch ch_metal_q8_0_gemm_mma accepts (one launch, a 2-D grid of
 * 32-row x 32-token tiles). */
#define CH_METAL_GEMM_MMA_MAX_BATCH 512u

/* Same contract as ch_metal_q8_0_gemm for batch 1..=CH_METAL_GEMM_MMA_MAX_BATCH,
 * on the SIMD-group matrix units (f32 tiles). Summation order differs from the GEMV, so it
 * is tolerance-qualified, not bit-identical; dense paths that claim
 * batch-size bit-identity keep ch_metal_q8_0_gemm. */
int ch_metal_q8_0_gemm_mma(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                           ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                           uint32_t x_off, uint32_t y_off, char *err, size_t err_cap);
int ch_metal_q4_0_gemm_mma(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                           ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                           uint32_t x_off, uint32_t y_off, char *err, size_t err_cap);
int ch_metal_q4_1_gemm_mma(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                           ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                           uint32_t x_off, uint32_t y_off, char *err, size_t err_cap);
/* Same contract and tiling as ch_metal_q8_0_gemm_mma, with f16 operand
 * tiles (the declared `metal-f16-act` profile, ADR-010): dequantized
 * weights and activations are rounded to half when staged, the multiply
 * runs in half, accumulation stays f32. llama.cpp's kernel_mul_mm
 * numerics; qualified separately against the f32 kernel and the
 * reference. */
int ch_metal_q8_0_gemm_mma_f16op(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                                 ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                                 uint32_t x_off, uint32_t y_off, char *err, size_t err_cap);
int ch_metal_q4_0_gemm_mma_f16op(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                                 ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                                 uint32_t x_off, uint32_t y_off, char *err, size_t err_cap);
int ch_metal_q4_1_gemm_mma_f16op(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                                 ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                                 uint32_t x_off, uint32_t y_off, char *err, size_t err_cap);
int ch_metal_f16_gemm(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                      ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                      uint32_t x_off, uint32_t y_off, char *err, size_t err_cap);

/* Batched RoPE: token t rotates at start_pos + t (q [n_tokens, n_q_heads,
 * dim], k [n_tokens, n_kv_heads, dim]). */
/* Batched causal attention: one threadgroup per (token, query head); query
 * token t attends keys 0..=(start_pos + t). Scratch is
 * n_tokens * heads * cap f32. */
/* out[i] = silu(gate[i]) * up[i]. */
int ch_metal_silu_mul_f32(ch_metal_ctx *ctx, ch_metal_buffer *gate, ch_metal_buffer *up,
                          ch_metal_buffer *out, uint32_t n, char *err, size_t err_cap);

/* GeGLU gate: out[i] = gelu_tanh(gate[i]) * up[i] (ggml's tanh form). */
int ch_metal_gelu_tanh_mul_f32(ch_metal_ctx *ctx, ch_metal_buffer *gate, ch_metal_buffer *up,
                               ch_metal_buffer *out, uint32_t n, char *err, size_t err_cap);

/* Packed Q8_0 matrix-vector decode: y[r] = dot(dequant_q8_0(w_row_r), x).
 * `w` holds `rows * cols / 32` blocks of 34 bytes ([f16 d][i8 qs; 32]).
 * `cols` must be a positive multiple of 32; `x` is f32 of length `cols`;
 * `y` is f32 of length `rows`. */
int ch_metal_q8_0_gemv(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                       ch_metal_buffer *y, uint32_t rows, uint32_t cols,
                       char *err, size_t err_cap);
int ch_metal_q4_0_gemv(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                        ch_metal_buffer *y, uint32_t rows, uint32_t cols, char *err, size_t err_cap);
int ch_metal_q4_1_gemv(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                        ch_metal_buffer *y, uint32_t rows, uint32_t cols, char *err, size_t err_cap);

/* ---------------- Q35-2A: Gated DeltaNet operators ---------------- */

/* One decode step of the depthwise causal convolution: `hist` holds the
 * previous (kernel - 1) rows (channel-major, oldest first), `x` the new
 * row, `w` the ssm_conv1d weights in GGUF memory order (channel-major,
 * taps fastest), `y` the output row. */
int ch_metal_qwen35_conv_step(ch_metal_ctx *ctx, ch_metal_buffer *hist, ch_metal_buffer *x,
                              ch_metal_buffer *w, ch_metal_buffer *y, uint32_t channels,
                              uint32_t kernel, char *err, size_t err_cap);

/* Advance the convolution history after ch_metal_qwen35_conv_step: shift
 * down one row and append the new input row (same group). */
int ch_metal_qwen35_conv_shift(ch_metal_ctx *ctx, ch_metal_buffer *hist, ch_metal_buffer *x,
                               uint32_t channels, uint32_t kernel, char *err, size_t err_cap);

/* Per-head L2 norm with the epsilon inside the root:
 * y[r][i] = x[r][i] / sqrt(sum_i x^2 + eps). */
int ch_metal_qwen35_l2norm_heads(ch_metal_ctx *ctx, ch_metal_buffer *x, ch_metal_buffer *y,
                                 uint32_t rows, uint32_t cols, float eps, char *err,
                                 size_t err_cap);

/* Per-head gates: beta_out = sigmoid(raw_beta);
 * gate_out = ssm_a * softplus(alpha + dt_bias) with ggml's thresholded
 * softplus (x > 20 ? x : log(1 + e^x)). */
int ch_metal_qwen35_beta_decay(ch_metal_ctx *ctx, ch_metal_buffer *raw_beta, ch_metal_buffer *alpha,
                               ch_metal_buffer *dt_bias, ch_metal_buffer *ssm_a,
                               ch_metal_buffer *beta_out, ch_metal_buffer *gate_out, uint32_t n,
                               char *err, size_t err_cap);

/* One token of the autoregressive delta rule for every value head; the
 * f32 state buffer [heads, k_dim, v_dim] (i-major) is updated in place. */
int ch_metal_qwen35_gdn_step(ch_metal_ctx *ctx, ch_metal_buffer *q, ch_metal_buffer *k,
                             ch_metal_buffer *v, ch_metal_buffer *gate, ch_metal_buffer *beta,
                             ch_metal_buffer *state, ch_metal_buffer *out, uint32_t heads,
                             uint32_t k_dim, uint32_t v_dim, float q_scale, char *err,
                             size_t err_cap);

/* Gated output normalization: per row, RMS norm over `cols` with the
 * ssm_norm weights, times SiLU of the z row. */
int ch_metal_qwen35_gated_norm(ch_metal_ctx *ctx, ch_metal_buffer *o, ch_metal_buffer *z,
                               ch_metal_buffer *weight, ch_metal_buffer *out, uint32_t rows,
                               uint32_t cols, float eps, char *err, size_t err_cap);

/* ---------------- Q35-2B: full-attention operators ---------------- */

/* Split the joint Q projection [heads, 2*head_dim] (query then gate per
 * head) into q and gate, both [heads, head_dim]. */
int ch_metal_qwen35_q_gate_split(ch_metal_ctx *ctx, ch_metal_buffer *qg, ch_metal_buffer *q_out,
                                 ch_metal_buffer *gate_out, uint32_t heads, uint32_t head_dim,
                                 char *err, size_t err_cap);

/* Partial split-half RoPE: rotate the first n_rot dims of every q/k head
 * row, pairs (i, i + n_rot/2), at `pos`; inv_freq holds n_rot/2 entries. */
int ch_metal_qwen35_rope_partial(ch_metal_ctx *ctx, ch_metal_buffer *q, ch_metal_buffer *k,
                                 ch_metal_buffer *inv_freq, uint32_t pos, uint32_t n_q_heads,
                                 uint32_t n_kv_heads, uint32_t head_dim, uint32_t n_rot,
                                 char *err, size_t err_cap);

/* Packed f16 KV store: k_cache[h][pos][d] = (half) k_src[h][d], two-byte
 * entries (same for v). */
int ch_metal_qwen35_kv_store_f16(ch_metal_ctx *ctx, ch_metal_buffer *k_src, ch_metal_buffer *v_src,
                                 ch_metal_buffer *k_cache, ch_metal_buffer *v_cache, uint32_t pos,
                                 uint32_t cap, uint32_t head_dim, uint32_t kv_heads, char *err,
                                 size_t err_cap);

/* Causal GQA attention over a packed f16 cache; scratch is heads*cap f32. */
int ch_metal_qwen35_attention_f16(ch_metal_ctx *ctx, ch_metal_buffer *q, ch_metal_buffer *k_cache,
                                  ch_metal_buffer *v_cache, ch_metal_buffer *out,
                                  ch_metal_buffer *scratch, uint32_t heads, uint32_t pos,
                                  uint32_t cap, uint32_t head_dim, uint32_t group, float scale,
                                  char *err, size_t err_cap);

/* x[i] = silu(x[i]) in place (conv output activation). */
int ch_metal_qwen35_silu(ch_metal_ctx *ctx, ch_metal_buffer *x, uint32_t n, char *err,
                         size_t err_cap);

/* Split the conv output [3*width] into q, k, v [width] each. */
int ch_metal_qwen35_split_qkv(ch_metal_ctx *ctx, ch_metal_buffer *y, ch_metal_buffer *q,
                              ch_metal_buffer *k, ch_metal_buffer *v, uint32_t width, char *err,
                              size_t err_cap);

/* attn[i] *= sigmoid(gate[i]). */
int ch_metal_qwen35_attn_gate(ch_metal_ctx *ctx, ch_metal_buffer *attn, ch_metal_buffer *gate,
                              uint32_t n, char *err, size_t err_cap);

/* Block until every committed command buffer has completed. Safe to call
 * with no pending work. */
int ch_metal_fence(ch_metal_ctx *ctx, char *err, size_t err_cap);

/* Measurement surface. */
uint64_t ch_metal_committed_dispatches(const ch_metal_ctx *ctx);
/* Copy a shared buffer range once the first `completed_target` committed
 * command buffers have completed, without waiting for later work (which
 * must not touch the range). Used to read a pipelined decode step's token
 * while the next step runs. */
int ch_metal_download_after(ch_metal_ctx *ctx, ch_metal_buffer *src, size_t off, void *dst,
                            size_t len, uint64_t completed_target, char *err, size_t err_cap);
uint64_t ch_metal_completed_dispatches(const ch_metal_ctx *ctx);
/* GPU time of the most recently completed command buffer, milliseconds
 * (0.0 before the first completion). */
double ch_metal_last_gpu_ms(const ch_metal_ctx *ctx);
/* Cumulative GPU nanoseconds over every completed command buffer. Wall
 * time minus this, over the same span, is host/dispatch time. */
uint64_t ch_metal_gpu_nanos_total(const ch_metal_ctx *ctx);
/* Opt-in per-operator GPU time attribution (profiling only). When
 * enabled, each committed command buffer carries its operator label and
 * its GPU time accumulates per label; buffers of an open dispatch group
 * are attributed to "group". Setting either value clears the table. */
int ch_metal_profile_set(ch_metal_ctx *ctx, int enabled);

/* 1 when the device supports dispatch-boundary counter sampling, so the
 * grouped (shared-command-buffer) encoder can attribute per-operator GPU
 * time; 0 means grouped mode has no per-operator data. */
int ch_metal_counter_sampling_supported(ch_metal_ctx *ctx);
/* Reads table entry `index`: returns 1 and fills name/nanos/count while
 * index is below the number of labels seen, 0 past the end. */
int ch_metal_profile_read(ch_metal_ctx *ctx, uint32_t index, char *name, size_t name_cap,
                          uint64_t *nanos, uint64_t *count);
/* Live bytes currently allocated in buffers from this context. */
uint64_t ch_metal_allocated_bytes(const ch_metal_ctx *ctx);

/* Bytes leased by pending command buffers (physically alive until their
 * completion, even after the owning handles were freed). gpu_allocated +
 * in_flight is a conservative upper bound on live GPU bytes: a buffer that
 * is both handle-live and awaiting completion is counted twice, so the
 * bound never understates. */
uint64_t ch_metal_in_flight_bytes(const ch_metal_ctx *ctx);

/* Chunk-wide Gated DeltaNet core (engine prefill): each processes `n`
 * tokens with exactly the per-element arithmetic of its per-token
 * counterpart (ch_metal_qwen35_conv_step + conv_shift, split_qkv,
 * beta_decay, gdn_step). Rows are token-major. */
int ch_metal_qwen35_conv_chunk(ch_metal_ctx *ctx, ch_metal_buffer *hist, ch_metal_buffer *x,
                               ch_metal_buffer *w, ch_metal_buffer *y, uint32_t channels,
                               uint32_t kern, uint32_t n, char *err, size_t err_cap);
int ch_metal_qwen35_split_qkv_chunk(ch_metal_ctx *ctx, ch_metal_buffer *y, ch_metal_buffer *q,
                                    ch_metal_buffer *k, ch_metal_buffer *v, uint32_t width,
                                    uint32_t n, char *err, size_t err_cap);
int ch_metal_qwen35_beta_decay_chunk(ch_metal_ctx *ctx, ch_metal_buffer *raw_beta,
                                     ch_metal_buffer *alpha, ch_metal_buffer *dt_bias,
                                     ch_metal_buffer *ssm_a, ch_metal_buffer *beta_out,
                                     ch_metal_buffer *gate_out, uint32_t per, uint32_t n,
                                     char *err, size_t err_cap);
int ch_metal_qwen35_gdn_chunk(ch_metal_ctx *ctx, ch_metal_buffer *q, ch_metal_buffer *k,
                              ch_metal_buffer *v, ch_metal_buffer *gate, ch_metal_buffer *beta,
                              ch_metal_buffer *state, ch_metal_buffer *out, uint32_t heads,
                              uint32_t k_dim, uint32_t v_dim, float q_scale, uint32_t n,
                              uint32_t qk_stride, uint32_t v_stride, uint32_t g_stride,
                              char *err, size_t err_cap);
/* ch_metal_qwen35_gdn_chunk that also writes the state after token `snap_t`
 * (0 <= snap_t < n) to `snap` (same shape as `state`): the rollback point of
 * a speculative verify. Arithmetic and outputs are those of the plain call. */
int ch_metal_qwen35_gdn_chunk_snap(ch_metal_ctx *ctx, ch_metal_buffer *q, ch_metal_buffer *k,
                                   ch_metal_buffer *v, ch_metal_buffer *gate,
                                   ch_metal_buffer *beta, ch_metal_buffer *state,
                                   ch_metal_buffer *out, ch_metal_buffer *snap, uint32_t snap_t,
                                   uint32_t snap_stride, uint32_t heads, uint32_t k_dim,
                                   uint32_t v_dim, float q_scale, uint32_t n, uint32_t qk_stride,
                                   uint32_t v_stride,
                                   uint32_t g_stride, char *err, size_t err_cap);
/* Shift a convolution history [kern-1, channels] by the n rows of `x` (the
 * history refresh of ch_metal_qwen35_conv_chunk on its own). */
int ch_metal_qwen35_conv_chunk_shift(ch_metal_ctx *ctx, ch_metal_buffer *hist,
                                     ch_metal_buffer *x, uint32_t channels, uint32_t kern,
                                     uint32_t n, char *err, size_t err_cap);

/* Chunk-wide gated attention core (engine prefill): RoPE, packed f16 KV
 * store and causal attention for `n` tokens at positions pos0..pos0+n-1,
 * each with exactly its per-token counterpart's arithmetic. Rows are
 * token-major; attention scratch is [n, heads, cap]. */
int ch_metal_qwen35_rope_partial_chunk(ch_metal_ctx *ctx, ch_metal_buffer *q, ch_metal_buffer *k,
                                       ch_metal_buffer *inv_freq, uint32_t pos0,
                                       uint32_t n_q_heads, uint32_t n_kv_heads,
                                       uint32_t head_dim, uint32_t n_rot, uint32_t n, char *err,
                                       size_t err_cap);
int ch_metal_qwen35_kv_store_f16_chunk(ch_metal_ctx *ctx, ch_metal_buffer *k_src,
                                       ch_metal_buffer *v_src, ch_metal_buffer *k_cache,
                                       ch_metal_buffer *v_cache, uint32_t pos0, uint32_t cap,
                                       uint32_t head_dim, uint32_t kv_heads, uint32_t n, char *err,
                                       size_t err_cap);
int ch_metal_qwen35_attention_f16_chunk(ch_metal_ctx *ctx, ch_metal_buffer *q,
                                        ch_metal_buffer *k_cache, ch_metal_buffer *v_cache,
                                        ch_metal_buffer *out, ch_metal_buffer *scratch,
                                        uint32_t heads, uint32_t pos0, uint32_t cap,
                                        uint32_t head_dim, uint32_t group, float scale, uint32_t n,
                                        uint32_t window, char *err, size_t err_cap);

/* The same KV store and chunk attention over an f32 cache (declared f32
 * KV: entries are exact copies, never rounded). Plane layout and geometry
 * are those of the f16 functions, at 4 bytes per entry. */
int ch_metal_qwen35_kv_store_f32_chunk(ch_metal_ctx *ctx, ch_metal_buffer *k_src,
                                       ch_metal_buffer *v_src, ch_metal_buffer *k_cache,
                                       ch_metal_buffer *v_cache, uint32_t pos0, uint32_t cap,
                                       uint32_t head_dim, uint32_t kv_heads, uint32_t n, char *err,
                                       size_t err_cap);
int ch_metal_qwen35_attention_f32_chunk(ch_metal_ctx *ctx, ch_metal_buffer *q,
                                        ch_metal_buffer *k_cache, ch_metal_buffer *v_cache,
                                        ch_metal_buffer *out, ch_metal_buffer *scratch,
                                        uint32_t heads, uint32_t pos0, uint32_t cap,
                                        uint32_t head_dim, uint32_t group, float scale, uint32_t n,
                                        uint32_t window, char *err, size_t err_cap);

/* Testing switch: use the generic device-memory recurrence kernel instead
 * of the register-resident specialization (results are bit-identical). */
int ch_metal_set_gdn_chunk_generic(ch_metal_ctx *ctx, int generic);

#ifdef __cplusplus
}
#endif

#endif /* CH_METAL_H */

