/*
 * ChineCore Metal kernels (ADR-007), compiled at runtime from this exact
 * repository revision. Correctness-first: every kernel is validated against
 * the CPU scalar reference before any tuning; see crates/metal tests.
 *
 * Numerics: f32 accumulation with the same dequantization the CPU reference
 * uses (Q8_0 dequant = d * q, f16 scale). GPU reduction order differs from
 * the CPU sequence, so comparisons are tolerance-registered, never exact.
 */
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

/* y[i] = x[i] * factor; tail-safe for any n. */
kernel void ch_scale_f32(device const float *x [[buffer(0)]],
                         device float *y [[buffer(1)]],
                         constant float &factor [[buffer(2)]],
                         constant uint &n [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    y[gid] = x[gid] * factor;
}

/* y[i] = y[i] + alpha * a[i]. */
kernel void ch_axpy_f32(device const float *a [[buffer(0)]],
                        device float *y [[buffer(1)]],
                        constant float &alpha [[buffer(2)]],
                        constant uint &n [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    y[gid] += alpha * a[gid];
}

/* y[i] = (float) x[i]; x holds IEEE binary16. */
kernel void ch_cast_f16_f32(device const half *x [[buffer(0)]],
                            device float *y [[buffer(1)]],
                            constant uint &n [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    y[gid] = (float)x[gid];
}

/* y[i] = (half) x[i]; round-to-nearest-even, overflow to infinity. */
kernel void ch_cast_f32_f16(device const float *x [[buffer(0)]],
                            device half *y [[buffer(1)]],
                            constant uint &n [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    y[gid] = (half)x[gid];
}

/*
 * Packed Q8_0 matrix-vector decode: y[r] = dot(dequant_q8_0(w_row_r), x).
 *
 * Q8_0 blocks are 34 bytes: [f16 d][i8 qs; 32]; dequant(w) = d * q.
 * Multi-row form (Phase 1C): each SIMD group computes CH_Q8_ROWS rows, and
 * a threadgroup holds CH_Q8_SG SIMD groups. Lane L handles quants
 * (L % 4) * 8 .. +7 of block (L / 4), so one SIMD-group step covers 8
 * blocks; the 8 matching x values are loaded once into registers and
 * reused for every row. Per-lane partials reduce with simd_sum. Reduction
 * order differs from the CPU sequence (tolerance-registered, ADR-007).
 * `cols` must be a positive multiple of 32 (validated C-side).
 */
#define CH_Q8_ROWS 4u
#define CH_Q8_SG 2u

kernel void ch_q8_0_gemv(device const uchar *w [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device float *y [[buffer(2)]],
                         constant uint &rows [[buffer(3)]],
                         constant uint &cols [[buffer(4)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    const uint blocks = cols / 32u;
    const size_t row_bytes = (size_t)blocks * 34u;
    const uint first = (tg * CH_Q8_SG + sg) * CH_Q8_ROWS;
    if (first >= rows) {
        return;
    }
    const uint sub = lane % 4u;   /* which 8-quant slice of the block */
    const uint lane_blk = lane / 4u; /* which of the 8 blocks this step */

    float acc[CH_Q8_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint b = lane_blk; b < blocks; b += 8u) {
        const device packed_float4 *xv =
            (const device packed_float4 *)(x + (size_t)b * 32u + sub * 8u);
        const float4 x0 = float4(xv[0]);
        const float4 x1 = float4(xv[1]);
        for (uint r = 0; r < CH_Q8_ROWS; ++r) {
            const uint row = first + r;
            if (row >= rows) {
                break;
            }
            const device uchar *blk = w + (size_t)row * row_bytes + (size_t)b * 34u;
            const float d = (float)(*(const device half *)blk);
            const device packed_char4 *qv = (const device packed_char4 *)(blk + 2u + sub * 8u);
            const float4 q0 = float4(char4(qv[0]));
            const float4 q1 = float4(char4(qv[1]));
            acc[r] += d * (dot(q0, x0) + dot(q1, x1));
        }
    }
    for (uint r = 0; r < CH_Q8_ROWS; ++r) {
        const float total = simd_sum(acc[r]);
        const uint row = first + r;
        if (lane == 0u && row < rows) {
            y[row] = total;
        }
    }
}

/* dst[dst_off + i] = src[src_off + i] for i < n: a bit-exact f32 copy
 * between buffers, encodable inside a compute dispatch group (batched
 * prefill moves per-token slices with it). */
kernel void ch_copy_f32(device const float *src [[buffer(0)]],
                        device float *dst [[buffer(1)]],
                        constant uint &src_off [[buffer(2)]],
                        constant uint &dst_off [[buffer(3)]],
                        constant uint &n [[buffer(4)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    dst[dst_off + gid] = src[src_off + gid];
}

/*
 * Greedy argmax over f32 logits, exactly the CPU rule
 * (`chine_backend::sampling::argmax`): NaN entries are skipped, the
 * largest value wins, and ties go to the lowest index. That order is
 * associative, so the two-pass parallel reduction returns the sequential
 * answer. CH_ARGMAX_NONE means every entry was NaN.
 */
#define CH_ARGMAX_NONE 0xFFFFFFFFu
#define CH_ARGMAX_GROUPS 64u
#define CH_ARGMAX_THREADS 256u

inline bool ch_argmax_better(float av, uint ai, float bv, uint bi) {
    if (ai == CH_ARGMAX_NONE) {
        return false;
    }
    if (bi == CH_ARGMAX_NONE) {
        return true;
    }
    return av > bv || (av == bv && ai < bi);
}

inline void ch_argmax_simd(thread float &v, thread uint &i) {
    for (ushort off = 16; off > 0; off >>= 1) {
        const float ov = simd_shuffle_down(v, off);
        const uint oi = simd_shuffle_down(i, off);
        if (ch_argmax_better(ov, oi, v, i)) {
            v = ov;
            i = oi;
        }
    }
}

inline void ch_argmax_threadgroup(thread float &v, thread uint &i,
                                  threadgroup float *tv, threadgroup uint *ti,
                                  uint lane, uint sg) {
    ch_argmax_simd(v, i);
    if (lane == 0u) {
        tv[sg] = v;
        ti[sg] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        v = lane < CH_ARGMAX_THREADS / 32u ? tv[lane] : -INFINITY;
        i = lane < CH_ARGMAX_THREADS / 32u ? ti[lane] : CH_ARGMAX_NONE;
        ch_argmax_simd(v, i);
    }
}

/* Pass 1: each of CH_ARGMAX_GROUPS threadgroups reduces a strided slice. */
kernel void ch_argmax_partial(device const float *x [[buffer(0)]],
                              device float *part_v [[buffer(1)]],
                              device uint *part_i [[buffer(2)]],
                              constant uint &n [[buffer(3)]],
                              uint tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tv[CH_ARGMAX_THREADS / 32u];
    threadgroup uint ti[CH_ARGMAX_THREADS / 32u];
    float v = -INFINITY;
    uint i = CH_ARGMAX_NONE;
    for (uint k = tg * CH_ARGMAX_THREADS + tid; k < n; k += CH_ARGMAX_GROUPS * CH_ARGMAX_THREADS) {
        const float xv = x[k];
        if (!isnan(xv) && ch_argmax_better(xv, k, v, i)) {
            v = xv;
            i = k;
        }
    }
    ch_argmax_threadgroup(v, i, tv, ti, lane, sg);
    if (tid == 0u) {
        part_v[tg] = v;
        part_i[tg] = i;
    }
}

/* Pass 2: one threadgroup reduces the partials into out[out_index]. */
kernel void ch_argmax_final(device const float *part_v [[buffer(0)]],
                            device const uint *part_i [[buffer(1)]],
                            device uint *out [[buffer(2)]],
                            constant uint &out_index [[buffer(3)]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tv[CH_ARGMAX_THREADS / 32u];
    threadgroup uint ti[CH_ARGMAX_THREADS / 32u];
    float v = tid < CH_ARGMAX_GROUPS ? part_v[tid] : -INFINITY;
    uint i = tid < CH_ARGMAX_GROUPS ? part_i[tid] : CH_ARGMAX_NONE;
    ch_argmax_threadgroup(v, i, tv, ti, lane, sg);
    if (tid == 0u) {
        out[out_index] = i;
    }
}

/*
 * Embedding lookup whose row is a token id produced on the device
 * (ids[slot], written by ch_argmax_final), so the host can encode the next
 * decode step before it has read the token. An id outside the table (the
 * all-NaN marker) writes zeros; the host fails that request when it reads
 * the marker, so the row is never used.
 */
kernel void ch_q8_0_row_indirect_f32(device const uchar *w [[buffer(0)]],
                                     device float *out [[buffer(1)]],
                                     device const uint *ids [[buffer(2)]],
                                     constant uint &slot [[buffer(3)]],
                                     constant uint &rows [[buffer(4)]],
                                     constant uint &cols [[buffer(5)]],
                                     uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) {
        return;
    }
    const uint row = ids[slot];
    if (row >= rows) {
        out[gid] = 0.0f;
        return;
    }
    const uint blocks = cols / 32u;
    const device uchar *blk = w + ((size_t)row * (size_t)blocks + gid / 32u) * 34u;
    const float d = (float)(*(const device half *)blk);
    const float q = (float)((const device char *)(blk + 2))[gid % 32u];
    out[gid] = d * q;
}

kernel void ch_f16_row_indirect_f32(device const half *w [[buffer(0)]],
                                    device float *out [[buffer(1)]],
                                    device const uint *ids [[buffer(2)]],
                                    constant uint &slot [[buffer(3)]],
                                    constant uint &rows [[buffer(4)]],
                                    constant uint &cols [[buffer(5)]],
                                    uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) {
        return;
    }
    const uint row = ids[slot];
    out[gid] = row < rows ? (float)w[(size_t)row * (size_t)cols + gid] : 0.0f;
}

/* ---------------- C2: transformer operation kernels ---------------- */

/* Dequantize one Q8_0 matrix row (or embedding row) to f32:
 * out[d] = d_scale(d/32) * q(d). One thread per output element. */
kernel void ch_q8_0_row_f32(device const uchar *w [[buffer(0)]],
                            device float *out [[buffer(1)]],
                            constant uint &row [[buffer(2)]],
                            constant uint &cols [[buffer(3)]],
                            constant uint &out_off [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) {
        return;
    }
    const uint blocks = cols / 32u;
    const device uchar *wrow = w + (size_t)row * (size_t)blocks * 34u;
    const uint b = gid / 32u;
    const uint lane = gid % 32u;
    const device uchar *blk = wrow + (size_t)b * 34u;
    const float d = (float)(*(const device half *)blk);
    const float q = (float)((const device char *)(blk + 2))[lane];
    out[out_off + gid] = d * q;
}

/* F16 matvec: y[r] = dot(f32(w_row_r), x). Lane-strided within one SIMD
 * group per row; used for F16-weight artifacts (fixtures, F16 twins). */
kernel void ch_f16_gemv(device const half *w [[buffer(0)]],
                        device const float *x [[buffer(1)]],
                        device float *y [[buffer(2)]],
                        constant uint &rows [[buffer(3)]],
                        constant uint &cols [[buffer(4)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint lane [[thread_position_in_threadgroup]]) {
    if (row >= rows) {
        return;
    }
    const device half *wrow = w + (size_t)row * (size_t)cols;
    float acc = 0.0f;
    for (uint i = lane; i < cols; i += 32u) {
        acc += (float)wrow[i] * x[i];
    }
    const float total = simd_sum(acc);
    if (lane == 0u) {
        y[row] = total;
    }
}

/* Threadgroup-wide reduction of `value` over 128 threads: 4 SIMD partials
 * are combined through shared memory. Returns the sum to every lane. */
inline float ch_reduce_sum_128(float value, threadgroup float *scratch,
                               uint simd_lane, uint simd_group) {
    value = simd_sum(value);
    if (simd_lane == 0u) {
        scratch[simd_group] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    if (simd_group == 0u) {
        float v = simd_lane < 4u ? scratch[simd_lane] : 0.0f;
        total = simd_sum(v);
        if (simd_lane == 0u) {
            scratch[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

/* RMSNorm over one row of `cols` f32 values with a per-column weight and
 * epsilon. One threadgroup of 128 threads per row; in-place (x == y) is
 * safe because each thread only reads its own elements after the barrier. */
kernel void ch_rmsnorm_f32(device const float *x [[buffer(0)]],
                           device const float *weight [[buffer(1)]],
                           device float *y [[buffer(2)]],
                           constant uint &cols [[buffer(3)]],
                           constant float &eps [[buffer(4)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[4];
    const device float *xrow = x + (size_t)row * (size_t)cols;
    device float *yrow = y + (size_t)row * (size_t)cols;
    float acc = 0.0f;
    for (uint i = tid; i < cols; i += 128u) {
        acc += xrow[i] * xrow[i];
    }
    const float sumsq = ch_reduce_sum_128(acc, scratch, lane, simd_group);
    const float rms = 1.0f / sqrt(sumsq / (float)cols + eps);
    for (uint i = tid; i < cols; i += 128u) {
        yrow[i] = xrow[i] * rms * weight[i];
    }
}

/* One launch for the per-head Q/K RMS norms of an attention core: rows
 * [0, q_rows) normalize q with q_weight, rows [q_rows, q_rows + k_rows)
 * normalize k with k_weight. Per-row arithmetic is exactly
 * ch_rmsnorm_f32's (same stride-128 accumulate, same ch_reduce_sum_128,
 * same x * rms * w — no add, so no contraction), making the merge
 * bit-identical to the two rmsnorm launches it replaces. */
kernel void ch_qk_norm_f32(device const float *q [[buffer(0)]],
                           device const float *q_weight [[buffer(1)]],
                           device float *q_out [[buffer(2)]],
                           device const float *k [[buffer(3)]],
                           device const float *k_weight [[buffer(4)]],
                           device float *k_out [[buffer(5)]],
                           constant uint &q_rows [[buffer(6)]],
                           constant uint &k_rows [[buffer(7)]],
                           constant uint &cols [[buffer(8)]],
                           constant float &eps [[buffer(9)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[4];
    if (row >= q_rows + k_rows) {
        return;
    }
    const bool is_q = row < q_rows;
    const size_t r = is_q ? row : row - q_rows;
    const device float *xrow = (is_q ? q : k) + r * (size_t)cols;
    device float *yrow = (is_q ? q_out : k_out) + r * (size_t)cols;
    const device float *weight = is_q ? q_weight : k_weight;
    float acc = 0.0f;
    for (uint i = tid; i < cols; i += 128u) {
        acc += xrow[i] * xrow[i];
    }
    const float sumsq = ch_reduce_sum_128(acc, scratch, lane, simd_group);
    const float rms = 1.0f / sqrt(sumsq / (float)cols + eps);
    for (uint i = tid; i < cols; i += 128u) {
        yrow[i] = xrow[i] * rms * weight[i];
    }
}

/* y[row] = rmsnorm(x[row]) + res[row]: the fused post-block norm +
 * residual add. The rmsnorm + axpy pair it replaces reads and writes the
 * row three times (norm reads x writes y; axpy reads y and res, writes
 * y); this reads x and res and writes y once. Per-row arithmetic mirrors
 * ch_rmsnorm_f32 exactly (same stride-128 accumulate, same
 * ch_reduce_sum_128, same x * rms * w); the final multiply-add contracts
 * to one fma (Metal fast math; MSL has no `precise`), which is one
 * rounding less than the pair's memory round-trip and is qualified by
 * the metal-crate gate within 4 eps x max(|norm|,|res|) (cancellation-
 * aware), not by bit equality. Both engine paths that the bit-identity
 * gates compare use this kernel, so the pair's numerics stay stable
 * across chunk and per-token execution. */
/* The normed product as a real call boundary: `optnone` disables
 * optimization inside (including multiply-add contraction) and implies
 * no-inline, so the returned value is rounded to f32 exactly like the
 * rmsnorm + axpy pair's device-memory round-trip. This keeps the fused
 * kernel bit-identical to the pair it replaces, which the chunk-vs-
 * per-token bit-identity gates depend on (both engine modes must agree
 * bit for bit even when one runs fused and the other does not). */
__attribute__((optnone)) static inline float ch_normed_uncontracted(float x, float rms,
                                                                   float w) {
    return x * rms * w;
}

kernel void ch_norm_add_f32(device const float *x [[buffer(0)]],
                            device const float *weight [[buffer(1)]],
                            device const float *res [[buffer(2)]],
                            device float *y [[buffer(3)]],
                            constant uint &cols [[buffer(4)]],
                            constant float &eps [[buffer(5)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[4];
    const device float *xrow = x + (size_t)row * (size_t)cols;
    const device float *resrow = res + (size_t)row * (size_t)cols;
    device float *yrow = y + (size_t)row * (size_t)cols;
    float acc = 0.0f;
    for (uint i = tid; i < cols; i += 128u) {
        acc += xrow[i] * xrow[i];
    }
    const float sumsq = ch_reduce_sum_128(acc, scratch, lane, simd_group);
    const float rms = 1.0f / sqrt(sumsq / (float)cols + eps);
    for (uint i = tid; i < cols; i += 128u) {
        yrow[i] = ch_normed_uncontracted(xrow[i], rms, weight[i]) + resrow[i];
    }
}

/*
 * NeoX-style RoPE applied in place to q (n_q heads) and k (n_kv heads) at
 * position `pos`, with inv_freq[i] = theta^(-2i/head_dim) for i < head_dim/2
 * precomputed on the host (matching the scalar executor's f32 table).
 * Layout per head: [.., dim] rotations pair (i, i + head_dim/2). One
 * threadgroup per head of head_dim/2 lanes.
/* k_cache[h][pos][d] = k_src[h][d] (same for v). One threadgroup per kv
 * head over head_dim lanes. */
/*
 * Causal GQA attention for one query position. One threadgroup per query
 * head; 128 threads. Scores for keys 0..=pos land in scratch (per-head
 * region of `heads * cap` floats), softmax in registers/shared, weighted
 * value accumulation parallel over head_dim lanes. `scale` is
 * 1/sqrt(head_dim). No mask beyond causality (Qwen3 dense here has no
 * sliding window).
/* out[i] = silu(gate[i]) * up[i], i < n. */
/* GeGLU gate: out[i] = gelu_tanh(gate[i]) * up[i], with ggml's tanh
 * approximation 0.5 x (1 + tanh(sqrt(2/pi) x (1 + 0.044715 x^2))). */
kernel void ch_gelu_tanh_mul_f32(device const float *gate [[buffer(0)]],
                                 device const float *up [[buffer(1)]],
                                 device float *out [[buffer(2)]],
                                 constant uint &n [[buffer(3)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= n) {
        return;
    }
    const float x = gate[i];
    const float inner = 0.7978845608028654f * x * (1.0f + 0.044715f * x * x);
    out[i] = 0.5f * x * (1.0f + precise::tanh(inner)) * up[i];
}

kernel void ch_silu_mul_f32(device const float *gate [[buffer(0)]],
                            device const float *up [[buffer(1)]],
                            device float *out [[buffer(2)]],
                            constant uint &n [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    const float g = gate[gid];
    const float s = g / (1.0f + exp(-g));
    out[gid] = s * up[gid];
}

/* Dequantize one F16 matrix row to f32 (embedding rows for F16 artifacts). */
kernel void ch_f16_row_f32(device const half *w [[buffer(0)]],
                           device float *out [[buffer(1)]],
                           constant uint &row [[buffer(2)]],
                           constant uint &cols [[buffer(3)]],
                           constant uint &out_off [[buffer(4)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) {
        return;
    }
    out[out_off + gid] = (float)w[(size_t)row * (size_t)cols + gid];
}

/* ---------------- C3: batched prefill kernels ---------------- */

/*
 * Batched packed Q8_0 GEMM: Y[b, :] = W · X[b, :] for b < batch. One SIMD
 * group per output row; each weight slice is loaded once and applied to
 * the whole batch, so weight traffic does not multiply with batch size.
 * Per-(row, b) arithmetic is exactly the single-token GEMV's: the same
 * lane-to-quant mapping (lane L: block L/4 of each 8-block step, quants
 * (L%4)*8..+7), the same block order, the same per-block expression and
 * the same simd_sum, so batch=1 is bit-identical to ch_q8_0_gemv.
 * X is [batch, cols] at element offset `x_off`; Y is [batch, rows] at
 * `y_off`; batch must be 1..=32 (validated C-side).
 */
kernel void ch_q8_0_gemm(device const uchar *w [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device float *y [[buffer(2)]],
                         constant uint &rows [[buffer(3)]],
                         constant uint &cols [[buffer(4)]],
                         constant uint &batch [[buffer(5)]],
                         constant uint &x_off [[buffer(6)]],
                         constant uint &y_off [[buffer(7)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint lane [[thread_position_in_threadgroup]]) {
    if (row >= rows || batch == 0u || batch > 32u) {
        return;
    }
    const uint blocks = cols / 32u;
    const device uchar *wrow = w + (size_t)row * (size_t)blocks * 34u;
    const uint sub = lane % 4u;
    const uint lane_blk = lane / 4u;
    float acc[32];
    for (uint b = 0; b < batch; ++b) {
        acc[b] = 0.0f;
    }
    for (uint blk = lane_blk; blk < blocks; blk += 8u) {
        const device uchar *block = wrow + (size_t)blk * 34u;
        const float d = (float)(*(const device half *)block);
        const device packed_char4 *qv = (const device packed_char4 *)(block + 2u + sub * 8u);
        const float4 q0 = float4(char4(qv[0]));
        const float4 q1 = float4(char4(qv[1]));
        const uint col = blk * 32u + sub * 8u;
        for (uint b = 0; b < batch; ++b) {
            const device packed_float4 *xv =
                (const device packed_float4 *)(x + x_off + b * cols + col);
            const float4 x0 = float4(xv[0]);
            const float4 x1 = float4(xv[1]);
            acc[b] += d * (dot(q0, x0) + dot(q1, x1));
        }
    }
    for (uint b = 0; b < batch; ++b) {
        const float total = simd_sum(acc[b]);
        if (lane == 0u) {
            y[y_off + b * rows + row] = total;
        }
    }
}

/*
 * Batched Q8_0 GEMM on the SIMD-group matrix units (prefill):
 * Y[b, :] = W · X[b, :] for b < batch, as C (tokens x rows) = X · W^T.
 * One threadgroup (4 SIMD groups) owns a 64-row x 32-token output tile;
 * threadgroup tg covers row tile tg / token_tiles and token tile
 * tg % token_tiles, so neighbours share a weight tile and one launch covers
 * a whole prefill chunk. Per 32-wide K block (one Q8_0 block per row):
 * - every thread first loads its share of the next W block (dequantized,
 *   d * q, exactly the CPU dequant) and X block into registers, so device
 *   latency overlaps other SIMD groups' multiplies;
 * - it then stores them into threadgroup memory as contiguous 8x8
 *   fragments (stride-8, conflict-free simdgroup loads, no transpose):
 *   X fragments [token][k], W^T fragments [k][row];
 * - SIMD group sg computes rows (sg % 2) * 32 .. +32 and tokens
 *   (sg / 2) * 16 .. +16 with 8 f32 accumulators.
 * Full tiles store straight from the accumulators to device memory;
 * edge tiles stage through threadgroup memory and store in bounds only.
 * Arithmetic is f32 throughout (oracle-f32 profile); summation order
 * differs from the GEMV, so results are tolerance-registered, not
 * bit-identical. Layout and code shape follow the pinned llama.cpp
 * kernel_mul_mm, kept in f32: out-of-range rows and tokens load a clamped
 * in-range row instead of branching, loops unroll fully, block pointers
 * advance by increment, fragment indices are 16-bit, and SIMD barriers
 * separate the fragment loads from the multiplies (ADR-011).
 */
#define CH_MMA_ROWS 64u
#define CH_MMA_TOK 32u
#define CH_UNROLL _Pragma("clang loop unroll(full)")

kernel void ch_q8_0_gemm_mma(device const uchar *w [[buffer(0)]],
                             device const float *x [[buffer(1)]],
                             device float *y [[buffer(2)]],
                             constant uint &rows [[buffer(3)]],
                             constant uint &cols [[buffer(4)]],
                             constant uint &batch [[buffer(5)]],
                             constant uint &x_off [[buffer(6)]],
                             constant uint &y_off [[buffer(7)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
    /* sa: 32 W^T fragments (4 k-blocks x 8 row-blocks), sb: 16 X fragments
     * (4 k-blocks x 4 token-blocks); 64 floats each. The edge-tile output
     * stage (64 x 32) reuses sa. */
    threadgroup float sa[CH_MMA_ROWS * 32u];
    threadgroup float sb[CH_MMA_TOK * 32u];

    const uint token_tiles = (batch + CH_MMA_TOK - 1u) / CH_MMA_TOK;
    const uint tok0 = (tg % token_tiles) * CH_MMA_TOK;
    if (tok0 >= batch) {
        return;
    }
    const uint tile_batch = min(batch - tok0, CH_MMA_TOK);
    const uint row0 = (tg / token_tiles) * CH_MMA_ROWS;
    const uint blocks = cols / 32u;
    const size_t row_bytes = (size_t)blocks * 34u;

    /* W loader: thread t owns 16 quants (half a Q8_0 block) of row t / 2. */
    const uint w_r = tid / 2u;
    const uint w_k = (tid % 2u) * 16u;
    const device uchar *wrow = w + (size_t)min(row0 + w_r, rows - 1u) * row_bytes;
    /* X loader: thread t owns 8 values of token t / 4. */
    const uint x_t = tid / 4u;
    const uint x_k = (tid % 4u) * 8u;
    const device float *xrow = x + x_off + (size_t)(tok0 + min(x_t, tile_batch - 1u)) * cols + x_k;

    /* Fragment slots this thread writes. */
    threadgroup float *sa_dst = sa + 64u * ((w_k / 8u) * 8u + w_r / 8u) + (w_r % 8u);
    threadgroup float *sb_dst = sb + 64u * ((x_k / 8u) * 4u + x_t / 8u) + 8u * (x_t % 8u);

    const uint sg_r = (sg % 2u) * 4u; /* first row-block of this SIMD group */
    const uint sg_t = (sg / 2u) * 2u; /* first token-block */
    simdgroup_float8x8 mc[2][4];
    for (uint j = 0; j < 2u; ++j) {
        for (uint i = 0; i < 4u; ++i) {
            mc[j][i] = simdgroup_float8x8(0.0f);
        }
    }

    /* Out-of-range rows and tokens load a clamped in-range row; their
     * results land only in output positions that are never stored. */
    const device uchar *block = wrow;
    const device float *xblk = xrow;
    /* This SIMD group's fragment bases; each k-block of 8 advances them. */
    const threadgroup float *lsa = sa + 64u * sg_r;
    const threadgroup float *lsb = sb + 64u * sg_t;
    for (uint blk = 0; blk < blocks; ++blk, block += 34, xblk += 32) {
        float wv[16];
        const float d = (float)(*(const device half *)block);
        const device packed_char4 *qv = (const device packed_char4 *)(block + 2u + w_k);
        CH_UNROLL for (uint i = 0; i < 4u; ++i) {
            const float4 q = float4(char4(qv[i]));
            wv[4u * i + 0u] = d * q.x;
            wv[4u * i + 1u] = d * q.y;
            wv[4u * i + 2u] = d * q.z;
            wv[4u * i + 3u] = d * q.w;
        }
        const device packed_float4 *xp = (const device packed_float4 *)xblk;
        const float4 xv0 = float4(xp[0]);
        const float4 xv1 = float4(xp[1]);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        /* W^T fragment element (k % 8, row % 8); the 16 k values span two
         * k-blocks of 8. */
        CH_UNROLL for (ushort i = 0; i < 16; ++i) {
            sa_dst[64 * 8 * (i / 8) + 8 * (i % 8)] = wv[i];
        }
        /* X fragment row (token % 8), 8 contiguous k values. */
        *(threadgroup float4 *)(sb_dst) = xv0;
        *(threadgroup float4 *)(sb_dst + 4u) = xv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        CH_UNROLL for (ushort kb = 0; kb < 4; ++kb) {
            simdgroup_float8x8 ma[4];
            simdgroup_float8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsa + 64 * (8 * kb + i), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2; ++j) {
                simdgroup_load(mb[j], lsb + 64 * (4 * kb + j), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2; ++j) {
                CH_UNROLL for (ushort i = 0; i < 4; ++i) {
                    simdgroup_multiply_accumulate(mc[j][i], mb[j], ma[i], mc[j][i]);
                }
            }
        }
    }

    if (row0 + CH_MMA_ROWS <= rows && tile_batch == CH_MMA_TOK) {
        device float *c = y + y_off + (size_t)(tok0 + sg_t * 8u) * rows + row0 + sg_r * 8u;
        for (uint j = 0; j < 2u; ++j) {
            for (uint i = 0; i < 4u; ++i) {
                simdgroup_store(mc[j][i], c + (size_t)j * 8u * rows + i * 8u, rows);
            }
        }
        return;
    }
    /* Edge tile: stage [32 tokens x 64 rows], store in bounds only. */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *ot = sa;
    for (uint j = 0; j < 2u; ++j) {
        for (uint i = 0; i < 4u; ++i) {
            simdgroup_store(mc[j][i], ot + (sg_t + j) * 8u * CH_MMA_ROWS + (sg_r + i) * 8u,
                            CH_MMA_ROWS);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < CH_MMA_ROWS * CH_MMA_TOK; idx += 128u) {
        const uint r = idx % CH_MMA_ROWS;
        const uint b = idx / CH_MMA_ROWS;
        if (row0 + r < rows && b < tile_batch) {
            y[y_off + (size_t)(tok0 + b) * rows + row0 + r] = ot[b * CH_MMA_ROWS + r];
        }
    }
}

/* Batched Q8_0 GEMM with f16 operand tiles: the declared `metal-f16-act`
 * profile (ADR-010), numerics of llama.cpp's kernel_mul_mm — dequantized
 * weights and activations are rounded to half when staged, the matrix
 * multiply runs in half, accumulation stays f32. Everything else (tiling,
 * fragment layout, summation order, edge handling) is identical to
 * ch_q8_0_gemm_mma. */
kernel void ch_q8_0_gemm_mma_f16op(device const uchar *w [[buffer(0)]],
                                   device const float *x [[buffer(1)]],
                                   device float *y [[buffer(2)]],
                                   constant uint &rows [[buffer(3)]],
                                   constant uint &cols [[buffer(4)]],
                                   constant uint &batch [[buffer(5)]],
                                   constant uint &x_off [[buffer(6)]],
                                   constant uint &y_off [[buffer(7)]],
                                   uint tg [[threadgroup_position_in_grid]],
                                   uint tid [[thread_position_in_threadgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]]) {
    /* sa: 32 W^T fragments (4 k-blocks x 8 row-blocks), sb: 16 X fragments
     * (4 k-blocks x 4 token-blocks); 32 halves each. edge_frag: one
     * 64-float stage per SIMD group for partial edge fragments (see the
     * edge comment at the store). 7 KB total keeps resident groups per
     * core high. */
    threadgroup half sa[CH_MMA_ROWS * 32u];
    threadgroup half sb[CH_MMA_TOK * 32u];
    threadgroup float edge_frag[4][64];

    const uint token_tiles = (batch + CH_MMA_TOK - 1u) / CH_MMA_TOK;
    const uint tok0 = (tg % token_tiles) * CH_MMA_TOK;
    if (tok0 >= batch) {
        return;
    }
    const uint tile_batch = min(batch - tok0, CH_MMA_TOK);
    const uint row0 = (tg / token_tiles) * CH_MMA_ROWS;
    const uint blocks = cols / 32u;
    const size_t row_bytes = (size_t)blocks * 34u;

    /* W loader: thread t owns 16 quants (half a Q8_0 block) of row t / 2. */
    const uint w_r = tid / 2u;
    const uint w_k = (tid % 2u) * 16u;
    const device uchar *wrow = w + (size_t)min(row0 + w_r, rows - 1u) * row_bytes;
    /* X loader: thread t owns 8 values of token t / 4. */
    const uint x_t = tid / 4u;
    const uint x_k = (tid % 4u) * 8u;
    const device float *xrow = x + x_off + (size_t)(tok0 + min(x_t, tile_batch - 1u)) * cols + x_k;

    /* Fragment slots this thread writes. */
    threadgroup half *sa_dst = sa + 64u * ((w_k / 8u) * 8u + w_r / 8u) + (w_r % 8u);
    threadgroup half *sb_dst = sb + 64u * ((x_k / 8u) * 4u + x_t / 8u) + 8u * (x_t % 8u);

    const uint sg_r = (sg % 2u) * 4u; /* first row-block of this SIMD group */
    const uint sg_t = (sg / 2u) * 2u; /* first token-block */
    simdgroup_float8x8 mc[2][4];
    for (uint j = 0; j < 2u; ++j) {
        for (uint i = 0; i < 4u; ++i) {
            mc[j][i] = simdgroup_float8x8(0.0f);
        }
    }

    /* Out-of-range rows and tokens load a clamped in-range row; their
     * results land only in output positions that are never stored. */
    const device uchar *block = wrow;
    const device float *xblk = xrow;
    /* This SIMD group's fragment bases; each k-block of 8 advances them. */
    const threadgroup half *lsa = sa + 64u * sg_r;
    const threadgroup half *lsb = sb + 64u * sg_t;
    for (uint blk = 0; blk < blocks; ++blk, block += 34, xblk += 32) {
        float wv[16];
        const float d = (float)(*(const device half *)block);
        const device packed_char4 *qv = (const device packed_char4 *)(block + 2u + w_k);
        CH_UNROLL for (uint i = 0; i < 4u; ++i) {
            const float4 q = float4(char4(qv[i]));
            wv[4u * i + 0u] = d * q.x;
            wv[4u * i + 1u] = d * q.y;
            wv[4u * i + 2u] = d * q.z;
            wv[4u * i + 3u] = d * q.w;
        }
        const device packed_float4 *xp = (const device packed_float4 *)xblk;
        const float4 xv0 = float4(xp[0]);
        const float4 xv1 = float4(xp[1]);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        /* W^T fragment element (k % 8, row % 8); the 16 k values span two
         * k-blocks of 8. Rounding to half happens here, once per staged
         * element. */
        CH_UNROLL for (ushort i = 0; i < 16u; ++i) {
            sa_dst[64 * 8 * (i / 8) + 8 * (i % 8)] = (half)wv[i];
        }
        /* X fragment row (token % 8), 8 contiguous k values. */
        *(threadgroup half4 *)(sb_dst) = half4(xv0);
        *(threadgroup half4 *)(sb_dst + 4u) = half4(xv1);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        CH_UNROLL for (ushort kb = 0; kb < 4u; ++kb) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort i = 0; i < 4u; ++i) {
                simdgroup_load(ma[i], lsa + 64 * (8 * kb + i), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2u; ++j) {
                simdgroup_load(mb[j], lsb + 64 * (4 * kb + j), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2u; ++j) {
                CH_UNROLL for (ushort i = 0; i < 4u; ++i) {
                    simdgroup_multiply_accumulate(mc[j][i], mb[j], ma[i], mc[j][i]);
                }
            }
        }
    }

    if (row0 + CH_MMA_ROWS <= rows && tile_batch == CH_MMA_TOK) {
        device float *c = y + y_off + (size_t)(tok0 + sg_t * 8u) * rows + row0 + sg_r * 8u;
        for (uint j = 0; j < 2u; ++j) {
            for (uint i = 0; i < 4u; ++i) {
                simdgroup_store(mc[j][i], c + (size_t)j * 8u * rows + i * 8u, rows);
            }
        }
        return;
    }
    /* Edge tile: every SIMD group consumes all of its own accumulators
     * first (full 8x8 fragments store straight from the registers, no
     * synchronization), and only the partial fragments then round-trip
     * through the shared 64-float stage. Keeping the accumulators'
     * live range inside one barrier-free pass is what keeps the kernel's
     * register allocation (and with it full-tile occupancy) unchanged —
     * a tail that holds them across threadgroup barriers inflated the
     * allocation and cost every full tile 15-25% (measured, ADR-011
     * engine note 2026-11-22). The 256 B stage replaces the whole-tile
     * 8 KB buffer this kernel used to carry statically, which cost every
     * full-tile group resident slots (3.4% at 6912x1152). */
    const uint tok_end = tok0 + tile_batch;
    {
        threadgroup float * const edge_frag_mine = edge_frag[sg];
        const uint rb_base = row0 + sg_r * 8u;
        const uint tb_base = tok0 + sg_t * 8u;
        for (uint j = 0; j < 2u; ++j) {
            for (uint i = 0; i < 4u; ++i) {
                const uint rb = rb_base + i * 8u;
                const uint tb = tb_base + j * 8u;
                if (rb >= rows || tb >= tok_end) {
                    continue; /* nothing of this fragment is valid */
                }
                if (rb + 8u <= rows && tb + 8u <= tok_end) {
                    simdgroup_store(mc[j][i], y + y_off + (size_t)tb * rows + rb, rows);
                } else {
                    simdgroup_store(mc[j][i], edge_frag_mine, 8);
                    /* sg-scoped: orders the stage write against the reads
                     * below without a threadgroup-wide barrier (the tail
                     * holds no cross-SIMD-group synchronization at all). */
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint lane = tid % 32u; lane < 64u; lane += 32u) {
                        const uint fr = lane / 8u; /* token within fragment */
                        const uint ft = lane % 8u; /* row within fragment */
                        if (tb + fr < tok_end && rb + ft < rows) {
                            y[y_off + (size_t)(tb + fr) * rows + rb + ft] =
                                edge_frag_mine[fr * 8u + ft];
                        }
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
    }
}

/* Batched F16 GEMM with the same contract as ch_q8_0_gemm. */
kernel void ch_f16_gemm(device const half *w [[buffer(0)]],
                        device const float *x [[buffer(1)]],
                        device float *y [[buffer(2)]],
                        constant uint &rows [[buffer(3)]],
                        constant uint &cols [[buffer(4)]],
                        constant uint &batch [[buffer(5)]],
                        constant uint &x_off [[buffer(6)]],
                        constant uint &y_off [[buffer(7)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint lane [[thread_position_in_threadgroup]]) {
    if (row >= rows || batch == 0u || batch > 32u) {
        return;
    }
    const device half *wrow = w + (size_t)row * (size_t)cols;
    float acc[32];
    for (uint b = 0; b < batch; ++b) {
        acc[b] = 0.0f;
    }
    for (uint i = lane; i < cols; i += 32u) {
        const float wv = (float)wrow[i];
        for (uint b = 0; b < batch; ++b) {
            acc[b] += wv * x[x_off + b * cols + i];
        }
    }
    for (uint b = 0; b < batch; ++b) {
        const float total = simd_sum(acc[b]);
        if (lane == 0u) {
            y[y_off + b * rows + row] = total;
        }
    }
}

/*
 * Batched NeoX RoPE: rows = n_tokens * (n_q_heads + n_kv_heads); the first
 * n_tokens*n_q_heads rows rotate q, the rest rotate k. Token t is rotated
 * at position start_pos + t, so a prefilled chunk matches single-token
 * semantics exactly (same angles, same pair order).
/* Batched KV writes: rows = n_tokens * kv_heads; token t lands at
 * start_pos + t. Same f16 rounding contract as the single-token kernel. */
/*
 * Batched causal GQA attention: one threadgroup per (token, query head);
 * query token t attends keys 0..=(start_pos + t) — exactly the masking and
 * arithmetic of the single-token kernel per (t, head). Scratch region is
 * (t * heads + head) * cap floats.
/* ---------------- Q35-2A: Gated DeltaNet operators ---------------- */

/*
 * One decode step of the depthwise causal convolution. `hist` holds the
 * previous (kernel - 1) rows of the QKV stream (channel-major, oldest
 * first), `x` the new row, `w` the `ssm_conv1d` weights in GGUF memory
 * order (channel-major, taps fastest: w[c * kernel + tap]). The window is
 * hist followed by x, so the output row is the graph's `ggml_ssm_conv`
 * result for this token. One thread per channel.
 */
kernel void ch_qwen35_conv_step(device const float *hist [[buffer(0)]],
                                device const float *x [[buffer(1)]],
                                device const float *w [[buffer(2)]],
                                device float *y [[buffer(3)]],
                                constant uint &channels [[buffer(4)]],
                                constant uint &kern [[buffer(5)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= channels) {
        return;
    }
    const uint k = kern;
    float acc = 0.0f;
    for (uint j = 0; j + 1u < k; ++j) {
        acc += w[gid * k + j] * hist[j * channels + gid];
    }
    acc += w[gid * k + (k - 1u)] * x[gid];
    y[gid] = acc;
}

/*
 * Advance the convolution history after a conv_step: shift down by one
 * row and append the new input row. Runs after ch_qwen35_conv_step inside
 * the same group, so the next token sees exactly the last (kernel - 1)
 * rows of the window.
 */
kernel void ch_qwen35_conv_shift(device float *hist [[buffer(0)]],
                                 device const float *x [[buffer(1)]],
                                 constant uint &channels [[buffer(2)]],
                                 constant uint &kern [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= channels || kern < 2u) {
        return;
    }
    for (uint j = 0; j + 2u < kern; ++j) {
        hist[j * channels + gid] = hist[(j + 1u) * channels + gid];
    }
    hist[(kern - 2u) * channels + gid] = x[gid];
}

/*
 * Per-head L2 normalization with the tolerance epsilon inside the root:
 * x / sqrt(sum(x^2) + eps). One threadgroup of 128 threads per row
 * (head); shared with the full-attention Q/K head norms.
 */
kernel void ch_qwen35_l2norm_heads(device const float *x [[buffer(0)]],
                                   device float *y [[buffer(1)]],
                                   constant uint &cols [[buffer(2)]],
                                   constant float &eps [[buffer(3)]],
                                   uint row [[threadgroup_position_in_grid]],
                                   uint tid [[thread_position_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[4];
    const device float *xrow = x + (size_t)row * (size_t)cols;
    device float *yrow = y + (size_t)row * (size_t)cols;
    float acc = 0.0f;
    for (uint i = tid; i < cols; i += 128u) {
        acc += xrow[i] * xrow[i];
    }
    const float sumsq = ch_reduce_sum_128(acc, scratch, lane, simd_group);
    const float inv = 1.0f / sqrt(sumsq + eps);
    for (uint i = tid; i < cols; i += 128u) {
        yrow[i] = xrow[i] * inv;
    }
}

/*
 * Per-head gate preparation: beta = sigmoid(raw_beta) and
 * gate = ssm_a * softplus(alpha + dt_bias), with ggml's thresholded
 * softplus (x > 20 ? x : log(1 + exp(x))). One thread per head.
 */
kernel void ch_qwen35_beta_decay(device const float *raw_beta [[buffer(0)]],
                                 device const float *alpha [[buffer(1)]],
                                 device const float *dt_bias [[buffer(2)]],
                                 device const float *ssm_a [[buffer(3)]],
                                 device float *beta_out [[buffer(4)]],
                                 device float *gate_out [[buffer(5)]],
                                 constant uint &n [[buffer(6)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    const float b = raw_beta[gid];
    beta_out[gid] = 1.0f / (1.0f + exp(-b));
    const float t = alpha[gid] + dt_bias[gid];
    const float sp = t > 20.0f ? t : log(1.0f + exp(t));
    gate_out[gid] = ssm_a[gid] * sp;
}

/*
 * One token of the autoregressive delta rule, matching
 * `build_delta_net_autoregressive` in per-element arithmetic:
 *   decay = exp(gate); s~ = s * decay; sk = s~^T k; d = beta (v - sk);
 *   s = s~ + k d^T; o = s^T (q * scale).
 * State layout: [head][i][j] with i over the key dim (contraction) and j
 * over the value dim, i-major f32.
 *
 * Every value column j is independent: its decay, its sk_j, d_j, column
 * update and o_j touch only column j. One thread per (head, column), so
 * the recurrence stays sequential where it must (over i, ascending, as in
 * the CPU rule) and parallel where it can. Adjacent threads read adjacent
 * j at the same i (coalesced). s~ is recomputed in the second pass
 * instead of stored: the same product, so the same rounding.
 * (Phase 1B: the earlier one-thread-per-head form was 90% of decode time.)
 */
kernel void ch_qwen35_gdn_step(device const float *q [[buffer(0)]],
                               device const float *k [[buffer(1)]],
                               device const float *v [[buffer(2)]],
                               device const float *gate [[buffer(3)]],
                               device const float *beta [[buffer(4)]],
                               device float *state [[buffer(5)]],
                               device float *out [[buffer(6)]],
                               constant uint &k_dim [[buffer(7)]],
                               constant uint &v_dim [[buffer(8)]],
                               constant float &q_scale [[buffer(9)]],
                               uint2 gid [[thread_position_in_grid]]) {
    const uint j = gid.x;
    const uint head = gid.y;
    const uint sk = k_dim;
    const uint sv = v_dim;
    if (j >= sv) {
        return;
    }
    device float *s = state + (size_t)head * (size_t)sk * (size_t)sv;
    const device float *kk = k + (size_t)head * (size_t)sk;
    const device float *qq = q + (size_t)head * (size_t)sk;
    const float decay = exp(gate[head]);
    const float b = beta[head];

    float sk_acc = 0.0f;
    for (uint i = 0; i < sk; ++i) {
        sk_acc += (s[i * sv + j] * decay) * kk[i];
    }
    const float d = b * (v[(size_t)head * (size_t)sv + j] - sk_acc);
    float acc = 0.0f;
    for (uint i = 0; i < sk; ++i) {
        const float updated = s[i * sv + j] * decay + kk[i] * d;
        s[i * sv + j] = updated;
        acc += updated * (qq[i] * q_scale);
    }
    out[(size_t)head * (size_t)sv + j] = acc;
}

/*
 * Gated output normalization: per value head, RMS norm over the head's
 * value dim with the `ssm_norm` weights, times SiLU of the gate row.
 * One thread per head, same order as the CPU operator.
 */
kernel void ch_qwen35_gated_norm(device const float *o [[buffer(0)]],
                                 device const float *z [[buffer(1)]],
                                 device const float *weight [[buffer(2)]],
                                 device float *out [[buffer(3)]],
                                 constant uint &cols [[buffer(4)]],
                                 constant float &eps [[buffer(5)]],
                                 uint head [[thread_position_in_grid]]) {
    const device float *row = o + (size_t)head * (size_t)cols;
    const device float *zrow = z + (size_t)head * (size_t)cols;
    device float *orow = out + (size_t)head * (size_t)cols;
    float acc = 0.0f;
    for (uint i = 0; i < cols; ++i) {
        acc += row[i] * row[i];
    }
    const float inv = 1.0f / sqrt(acc / (float)cols + eps);
    for (uint i = 0; i < cols; ++i) {
        const float zz = zrow[i];
        const float silu = zz / (1.0f + exp(-zz));
        orow[i] = row[i] * inv * weight[i] * silu;
    }
}

/* ---------------- Q35-2B: full-attention operators ---------------- */

/*
 * Split the joint Q projection `qg` [heads, 2*head_dim] (query first,
 * gate second, per head) into q and gate, both [heads, head_dim].
 */
kernel void ch_qwen35_q_gate_split(device const float *qg [[buffer(0)]],
                                   device float *q_out [[buffer(1)]],
                                   device float *gate_out [[buffer(2)]],
                                   constant uint &head_dim [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]]) {
    const uint hd = head_dim;
    const uint head = gid / hd;
    const uint i = gid % hd;
    q_out[gid] = qg[(size_t)head * (size_t)(2u * hd) + i];
    gate_out[gid] = qg[(size_t)head * (size_t)(2u * hd) + hd + i];
}

/*
 * Partial split-half RoPE (IMRoPE reduced to text positions): rotate only
 * the first `n_rot` dims of each head row, pairs (i, i + n_rot/2), angle
 * pos * inv_freq[i] (inv_freq holds n_rot/2 entries); dims at or beyond
 * n_rot pass through untouched; with n_rot == head_dim it is full NeoX
 * RoPE (dense Qwen3). One threadgroup per head over q then k.
 */
kernel void ch_qwen35_rope_partial_f32(device float *q [[buffer(0)]],
                                       device float *k [[buffer(1)]],
                                       device const float *inv_freq [[buffer(2)]],
                                       constant uint &pos [[buffer(3)]],
                                       constant uint &n_q_heads [[buffer(4)]],
                                       constant uint &n_kv_heads [[buffer(5)]],
                                       constant uint &head_dim [[buffer(6)]],
                                       constant uint &n_rot [[buffer(7)]],
                                       uint head [[threadgroup_position_in_grid]],
                                       uint lane [[thread_position_in_threadgroup]]) {
    const uint half_rot = n_rot / 2u;
    if (lane >= half_rot) {
        return;
    }
    const bool is_q = head < n_q_heads;
    const uint local_head = is_q ? head : head - n_q_heads;
    device float *base = is_q ? q : k;
    device float *row = base + (size_t)local_head * (size_t)head_dim;
    const float angle = (float)pos * inv_freq[lane];
    const float c = cos(angle);
    const float s = sin(angle);
    const float a = row[lane];
    const float b = row[lane + half_rot];
    row[lane] = a * c - b * s;
    row[lane + half_rot] = a * s + b * c;
}

/* Packed f16 KV store: k_cache[h][pos][d] = (half) k_src[h][d], true
 * two-byte entries (the declared packed cache contract), same for v. */
kernel void ch_qwen35_kv_store_f16(device const float *k_src [[buffer(0)]],
                                   device const float *v_src [[buffer(1)]],
                                   device half *k_cache [[buffer(2)]],
                                   device half *v_cache [[buffer(3)]],
                                   constant uint &pos [[buffer(4)]],
                                   constant uint &cap [[buffer(5)]],
                                   constant uint &head_dim [[buffer(6)]],
                                   uint gid [[thread_position_in_grid]]) {
    const size_t head = (size_t)gid / (size_t)head_dim;
    const size_t lane = (size_t)gid % (size_t)head_dim;
    const size_t src = head * (size_t)head_dim + lane;
    const size_t dst = (head * (size_t)cap + (size_t)pos) * (size_t)head_dim + lane;
    k_cache[dst] = (half)k_src[src];
    v_cache[dst] = (half)v_src[src];
}

/*
 * Causal GQA attention over a packed f16 cache (values decode to f32 for
 * the arithmetic, mirroring the CPU reference's packed-cache path).
 * One threadgroup (4 SIMD groups) per head. Scores: each SIMD group takes
 * positions sg, sg+4, ... and reduces its dot product with simd_sum (no
 * threadgroup barrier per position). Softmax weights are normalized once
 * per position into the scratch row, then every value dim accumulates
 * them (Phase 1C: the per-position barrier and the per-dim recomputation
 * of exp() dominated prefill attention).
 */
/* One head's causal attention over a packed f16 (or f32) cache, shared by
 * the per-token and chunk kernels, so both run exactly this code. */
template <typename C>
inline void ch_q35_attend(const device float *qh, const device C *kc, const device C *vc,
                          device float *oh, device float *head_scratch, uint start, uint seq,
                          uint head_dim, float scale, uint tid, uint lane, uint simd_group,
                          threadgroup float *reduce_scratch, threadgroup float *row_stats) {
    for (uint j = start + simd_group; j < seq; j += 4u) {
        const device C *kr = kc + (size_t)j * (size_t)head_dim;
        float part = 0.0f;
        for (uint d = lane; d < head_dim; d += 32u) {
            part += qh[d] * (float)kr[d];
        }
        const float s = simd_sum(part) * scale;
        if (lane == 0u) {
            head_scratch[j] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    float local_max = -INFINITY;
    for (uint j = start + tid; j < seq; j += 128u) {
        local_max = max(local_max, head_scratch[j]);
    }
    local_max = simd_max(local_max);
    if (lane == 0u) {
        reduce_scratch[simd_group] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float m = reduce_scratch[0];
        for (uint g = 1u; g < 4u; ++g) {
            m = max(m, reduce_scratch[g]);
        }
        row_stats[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float row_max = row_stats[0];

    float local_sum = 0.0f;
    for (uint j = start + tid; j < seq; j += 128u) {
        local_sum += exp(head_scratch[j] - row_max);
    }
    const float denom = ch_reduce_sum_128(local_sum, reduce_scratch, lane, simd_group);
    if (tid == 0u) {
        row_stats[1] = denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float norm = row_stats[1];
    for (uint j = start + tid; j < seq; j += 128u) {
        head_scratch[j] = exp(head_scratch[j] - row_max) / norm;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    for (uint d = tid; d < head_dim; d += 128u) {
        float accv = 0.0f;
        for (uint j = start; j < seq; ++j) {
            accv += head_scratch[j] * (float)vc[(size_t)j * (size_t)head_dim + d];
        }
        oh[d] = accv;
    }
}

kernel void ch_qwen35_attention_f16(device const float *q [[buffer(0)]],
                                    device const half *k_cache [[buffer(1)]],
                                    device const half *v_cache [[buffer(2)]],
                                    device float *out [[buffer(3)]],
                                    device float *scratch [[buffer(4)]],
                                    constant uint &pos [[buffer(5)]],
                                    constant uint &cap [[buffer(6)]],
                                    constant uint &head_dim [[buffer(7)]],
                                    constant uint &group [[buffer(8)]],
                                    constant float &scale [[buffer(9)]],
                                    uint head [[threadgroup_position_in_grid]],
                                    uint tid [[thread_position_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float reduce_scratch[4];
    threadgroup float row_stats[2];
    const uint kv_head = head / group;
    ch_q35_attend(q + (size_t)head * (size_t)head_dim,
                  k_cache + ((size_t)kv_head * (size_t)cap) * (size_t)head_dim,
                  v_cache + ((size_t)kv_head * (size_t)cap) * (size_t)head_dim,
                  out + (size_t)head * (size_t)head_dim, scratch + (size_t)head * (size_t)cap,
                  0u, pos + 1u, head_dim, scale, tid, lane, simd_group, reduce_scratch, row_stats);
}

/* First key a query row attends: the last `window` positions up to and
 * including its own (sliding-window attention), or position 0 when window
 * is 0 (full causal attention). */
inline uint ch_attn_start(uint seq, uint window) {
    return (window != 0u && seq > window) ? seq - window : 0u;
}

/* Chunk-wide causal attention: threadgroup (head, t) attends query row t of
 * a chunk starting at `pos0` over positions 0..pos0+t, with the same code as
 * the per-token kernel. Every chunk token's K/V must already be stored.
 * Scratch rows are [n, heads, cap]. */
template <typename C>
kernel void ch_qwen35_attention_chunk(device const float *q [[buffer(0)]],
                                      device const C *k_cache [[buffer(1)]],
                                      device const C *v_cache [[buffer(2)]],
                                          device float *out [[buffer(3)]],
                                          device float *scratch [[buffer(4)]],
                                          constant uint &pos0 [[buffer(5)]],
                                          constant uint &cap [[buffer(6)]],
                                          constant uint &head_dim [[buffer(7)]],
                                          constant uint &group [[buffer(8)]],
                                          constant float &scale [[buffer(9)]],
                                          constant uint &heads [[buffer(10)]],
                                          constant uint &window [[buffer(11)]],
                                          uint2 tg [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]],
                                          uint lane [[thread_index_in_simdgroup]],
                                          uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float reduce_scratch[4];
    threadgroup float row_stats[2];
    const uint head = tg.x;
    const uint t = tg.y;
    const uint kv_head = head / group;
    const size_t row = (size_t)t * heads + head;
    const uint seq = pos0 + t + 1u;
    ch_q35_attend(q + row * (size_t)head_dim,
                  k_cache + ((size_t)kv_head * (size_t)cap) * (size_t)head_dim,
                  v_cache + ((size_t)kv_head * (size_t)cap) * (size_t)head_dim,
                  out + row * (size_t)head_dim, scratch + row * (size_t)cap,
                  ch_attn_start(seq, window), seq, head_dim, scale, tid, lane, simd_group,
                  reduce_scratch, row_stats);
}

#define CH_ATTN_CHUNK_ARGS(C)                                                               \
    device const float *q [[buffer(0)]], device const C *k_cache [[buffer(1)]],             \
        device const C *v_cache [[buffer(2)]], device float *out [[buffer(3)]],             \
        device float *scratch [[buffer(4)]], constant uint &pos0 [[buffer(5)]],             \
        constant uint &cap [[buffer(6)]], constant uint &head_dim [[buffer(7)]],            \
        constant uint &group [[buffer(8)]], constant float &scale [[buffer(9)]],            \
        constant uint &heads [[buffer(10)]], constant uint &window [[buffer(11)]],          \
        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
        uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]

template [[host_name("ch_qwen35_attention_f16_chunk")]] kernel void
ch_qwen35_attention_chunk<half>(CH_ATTN_CHUNK_ARGS(half));
template [[host_name("ch_qwen35_attention_f32_chunk")]] kernel void
ch_qwen35_attention_chunk<float>(CH_ATTN_CHUNK_ARGS(float));

/*
 * Chunk attention over a packed f16 (f32_cache 0) or f32 (1) cache:
 * one threadgroup per (KV head, row) - the prefill path. The G query
 * vectors are staged in threadgroup memory; each thread owns whole keys
 * (one sequential dot product per (key, head), no cross-lane
 * reduction); SIMD group h normalizes head h; SIMD groups split the
 * keys for the value phase and the four partials sum in a fixed order.
 * Scratch rows are [n, heads, cap].
 * Summation order differs from ch_q35_attend (tolerance-qualified).
 */
template <uint HD, uint G, typename C>
kernel void ch_qwen35_attention_gqa(device const float *q [[buffer(0)]],
                                    device const C *k_cache [[buffer(1)]],
                                    device const C *v_cache [[buffer(2)]],
                                    device float *out [[buffer(3)]],
                                    device float *scratch [[buffer(4)]],
                                    constant uint &pos0 [[buffer(5)]],
                                    constant uint &cap [[buffer(6)]],
                                    constant uint &head_dim [[buffer(7)]],
                                    constant uint &group [[buffer(8)]],
                                    constant float &scale [[buffer(9)]],
                                    constant uint &heads [[buffer(10)]],
                                    constant uint &window [[buffer(11)]],
                                    uint2 tg [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint EPL = (HD + 31u) / 32u;
    threadgroup float partial[4u * G * HD];
    threadgroup float qt[G * HD];
    constexpr uint PT = 256u; /* probability tile (keys) */
    threadgroup float pt[G * PT];
    if (head_dim != HD || group != G) {
        return;
    }
    const uint kv_head = tg.x;
    const uint t = tg.y;
    const uint seq = pos0 + t + 1u;
    const uint start = ch_attn_start(seq, window);
    const uint d0 = lane * EPL;
    const bool active = d0 < HD;
    const size_t row0 = (size_t)t * heads + (size_t)kv_head * G;
    const device C *kc = k_cache + (size_t)kv_head * (size_t)cap * HD;
    const device C *vc = v_cache + (size_t)kv_head * (size_t)cap * HD;

    for (uint idx = tid; idx < G * HD; idx += 128u) {
        qt[idx] = q[row0 * HD + idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = start + tid; j < seq; j += 128u) {
        const device C *kr = kc + (size_t)j * HD;
        float part[G];
        for (uint h = 0; h < G; ++h) {
            part[h] = 0.0f;
        }
        for (uint d = 0; d < HD; d += 4u) {
            const float4 k4 = float4(*(const device vec<C, 4> *)(kr + d));
            for (uint h = 0; h < G; ++h) {
                part[h] += dot(*(const threadgroup float4 *)(qt + h * HD + d), k4);
            }
        }
        for (uint h = 0; h < G; ++h) {
            scratch[(row0 + h) * cap + j] = part[h] * scale;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    for (uint h = sg; h < G; h += 4u) {
        device float *sh = scratch + (row0 + h) * cap;
        float m = -INFINITY;
        for (uint j = start + lane; j < seq; j += 32u) {
            m = max(m, sh[j]);
        }
        m = simd_max(m);
        float sum = 0.0f;
        for (uint j = start + lane; j < seq; j += 32u) {
            sum += exp(sh[j] - m);
        }
        const float norm = simd_sum(sum);
        for (uint j = start + lane; j < seq; j += 32u) {
            sh[j] = exp(sh[j] - m) / norm;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    float acc[G][EPL];
    for (uint h = 0; h < G; ++h) {
        for (uint e = 0; e < EPL; ++e) {
            acc[h][e] = 0.0f;
        }
    }
    /* Values: probabilities are staged per tile of PT keys in threadgroup
     * memory; each SIMD group still takes keys j = sg mod 4 in ascending
     * order, so the accumulation order is unchanged. */
    for (uint base = start; base < seq; base += PT) {
        const uint tile = min(PT, seq - base);
        for (uint idx = tid; idx < G * PT; idx += 128u) {
            const uint h = idx / PT;
            const uint jj = idx % PT;
            if (jj < tile) {
                pt[idx] = scratch[(row0 + h) * cap + base + jj];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint jj = sg; jj < tile; jj += 4u) {
            const uint j = base + jj;
            float vv[EPL];
            for (uint e = 0; e < EPL; ++e) {
                vv[e] = active ? (float)vc[(size_t)j * HD + d0 + e] : 0.0f;
            }
            for (uint h = 0; h < G; ++h) {
                const float p = pt[h * PT + jj];
                for (uint e = 0; e < EPL; ++e) {
                    acc[h][e] += p * vv[e];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        for (uint h = 0; h < G; ++h) {
            for (uint e = 0; e < EPL; ++e) {
                partial[(sg * G + h) * HD + d0 + e] = acc[h][e];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < G * HD; idx += 128u) {
        const float total = ((partial[idx] + partial[G * HD + idx]) + partial[2u * G * HD + idx]) +
                            partial[3u * G * HD + idx];
        out[row0 * HD + idx] = total;
    }
}

#define CH_ATTN_GQA_ARGS(C)                                                               \
    device const float *q [[buffer(0)]], device const C *k_cache [[buffer(1)]],             \
        device const C *v_cache [[buffer(2)]], device float *out [[buffer(3)]],             \
        device float *scratch [[buffer(4)]], constant uint &pos0 [[buffer(5)]],              \
        constant uint &cap [[buffer(6)]], constant uint &head_dim [[buffer(7)]],             \
        constant uint &group [[buffer(8)]], constant float &scale [[buffer(9)]],             \
        constant uint &heads [[buffer(10)]], constant uint &window [[buffer(11)]],           \
        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],  \
        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]

template [[host_name("ch_qwen35_attention_gqa_256x4")]] kernel void
ch_qwen35_attention_gqa<256u, 4u, half>(CH_ATTN_GQA_ARGS(half));
template [[host_name("ch_qwen35_attention_gqa_16x2")]] kernel void
ch_qwen35_attention_gqa<16u, 2u, half>(CH_ATTN_GQA_ARGS(half));
template [[host_name("ch_qwen35_attention_gqa_128x2")]] kernel void
ch_qwen35_attention_gqa<128u, 2u, half>(CH_ATTN_GQA_ARGS(half));
template [[host_name("ch_qwen35_attention_gqa_128x4")]] kernel void
ch_qwen35_attention_gqa<128u, 4u, half>(CH_ATTN_GQA_ARGS(half));
template [[host_name("ch_qwen35_attention_gqa_f32_16x2")]] kernel void
ch_qwen35_attention_gqa<16u, 2u, float>(CH_ATTN_GQA_ARGS(float));
template [[host_name("ch_qwen35_attention_gqa_f32_128x2")]] kernel void
ch_qwen35_attention_gqa<128u, 2u, float>(CH_ATTN_GQA_ARGS(float));
template [[host_name("ch_qwen35_attention_gqa_f32_128x4")]] kernel void
ch_qwen35_attention_gqa<128u, 4u, float>(CH_ATTN_GQA_ARGS(float));

/*
 * Split-K chunk attention (ADR-011 engine note, 2026-11-22). One
 * threadgroup per (KV head, row) could not hide the KV range's latency
 * (~5.4 GB/s at decode - the measured Gemma decode gap); the partial
 * kernel below gives every (KV head, row) up to CH_ATTN_SK_SLICES
 * threadgroups, each covering a slice of the row's key range, writing
 * per-slice online-softmax partials [m, l, acc dims] to scratch; the
 * merge kernel folds them (max-rescaled) into the output rows.
 *
 * The slice count is a pure function of the row's absolute position -
 * slices = clamp(ceil(total/32), 1, 16) from (seq, window) - so a given
 * position's reduction order is IDENTICAL in chunked prefill and
 * per-token decode, and the cross-mode bit-identity gates keep their
 * exact-bit assertions. Idle slice groups (slice >= slices) return
 * before any load and write nothing; the merge reads only s < slices.
 * Scratch partials are [(n, heads, SLICES, head_dim + 2)].
 */
#define CH_ATTN_SK_SLICES 16u
#define CH_ATTN_SK_KT 128u

static inline uint ch_attn_sk_slices(uint total) {
    const uint sk = (total + 31u) / 32u; /* ~32 keys per slice group */
    return sk < 1u ? 1u : (sk > CH_ATTN_SK_SLICES ? CH_ATTN_SK_SLICES : sk);
}

template <typename C, uint HD, uint G>
kernel void ch_qwen35_attn_sk_partial(device const float *q [[buffer(0)]],
                                      device const C *k_cache [[buffer(1)]],
                                      device const C *v_cache [[buffer(2)]],
                                      device float *scratch [[buffer(3)]],
                                      constant uint &pos0 [[buffer(4)]],
                                      constant uint &cap [[buffer(5)]],
                                      constant uint &head_dim [[buffer(6)]],
                                      constant uint &group [[buffer(7)]],
                                      constant float &scale [[buffer(8)]],
                                      constant uint &heads [[buffer(9)]],
                                      constant uint &window [[buffer(10)]],
                                      uint2 tg [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint EPL = (HD + 31u) / 32u;
    threadgroup float sc[G * CH_ATTN_SK_KT];
    threadgroup float tgs[G * 2u];
    threadgroup float qt[G * HD];
    /* The value loop splits keys across the four SIMD groups, so each
     * group's lanes hold a partial accumulator for the same dims; the
     * partials fold across groups before any write. Layout per thread:
     * (sg, head, lane, dim-within-lane). */
    threadgroup float part_acc[4u * G * 32u * EPL];
    if (head_dim != HD || group != G) {
        return;
    }
    const uint kv_head = tg.x / CH_ATTN_SK_SLICES;
    const uint slice = tg.x % CH_ATTN_SK_SLICES;
    const uint t = tg.y;
    const uint seq = pos0 + t + 1u;
    const uint start = ch_attn_start(seq, window);
    const uint total = seq - start;
    const uint slices = ch_attn_sk_slices(total);
    if (slice >= slices) {
        return; /* idle: the merge reads only s < slices */
    }
    const uint per = (total + slices - 1u) / slices;
    const uint jb = start + slice * per;
    const uint je = jb + per < seq ? jb + per : seq;
    const uint d0 = lane * EPL;
    const bool active = d0 < HD;
    const size_t row0 = (size_t)t * heads + (size_t)kv_head * G;
    const device C *kc = k_cache + (size_t)kv_head * (size_t)cap * HD;
    const device C *vc = v_cache + (size_t)kv_head * (size_t)cap * HD;

    for (uint idx = tid; idx < G * HD; idx += 128u) {
        qt[idx] = q[row0 * HD + idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mrun[G];
    float lrun[G];
    float acc[G][EPL];
    for (uint h = 0; h < G; ++h) {
        mrun[h] = -INFINITY;
        lrun[h] = 0.0f;
        for (uint e = 0; e < EPL; ++e) {
            acc[h][e] = 0.0f;
        }
    }
    for (uint base = jb; base < je; base += CH_ATTN_SK_KT) {
        const uint tile = je - base < CH_ATTN_SK_KT ? je - base : CH_ATTN_SK_KT;
        /* Scores: per-thread key ownership (deep in-flight loads), staged
         * per head into threadgroup memory. */
        if (tid < tile) {
            const device C *kr = kc + (size_t)(base + tid) * HD;
            float part[G];
            for (uint h = 0; h < G; ++h) {
                part[h] = 0.0f;
            }
            for (uint d = 0; d < HD; d += 4u) {
                const float4 k4 = float4(*(const device vec<C, 4> *)(kr + d));
                for (uint h = 0; h < G; ++h) {
                    part[h] += dot(*(const threadgroup float4 *)(qt + h * HD + d), k4);
                }
            }
            for (uint h = 0; h < G; ++h) {
                sc[h * CH_ATTN_SK_KT + tid] = part[h] * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        /* Tile statistics per head, one SIMD group per head. */
        for (uint h = sg; h < G; h += 4u) {
            float mt = -INFINITY;
            for (uint jj = lane; jj < tile; jj += 32u) {
                mt = max(mt, sc[h * CH_ATTN_SK_KT + jj]);
            }
            mt = simd_max(mt);
            float lt = 0.0f;
            for (uint jj = lane; jj < tile; jj += 32u) {
                lt += exp(sc[h * CH_ATTN_SK_KT + jj] - mt);
            }
            lt = simd_sum(lt);
            if (lane == 0u) {
                tgs[h * 2u] = mt;
                tgs[h * 2u + 1u] = lt;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        /* Fold the tile into the running partial: rescale by the max
         * delta (uniform per head), then accumulate this tile's values
         * with lane-dim ownership over sg-interleaved ascending keys. */
        for (uint h = 0; h < G; ++h) {
            const float mt = tgs[h * 2u];
            const float lt = tgs[h * 2u + 1u];
            const float r = mrun[h] == -INFINITY ? 0.0f : exp(mrun[h] - mt);
            lrun[h] = lrun[h] * r + lt;
            mrun[h] = mt;
            for (uint e = 0; e < EPL; ++e) {
                acc[h][e] *= r;
            }
        }
        for (uint jj = sg; jj < tile; jj += 4u) {
            const uint j = base + jj;
            const device C *vr = vc + (size_t)j * HD;
            float vv[EPL];
            for (uint e = 0; e < EPL; ++e) {
                /* Lanes whose dims end past HD (small head_dim: EPL is 1)
                 * must contribute zero, not the next row's bytes. */
                vv[e] = active ? (float)vr[d0 + e] : 0.0f;
            }
            for (uint h = 0; h < G; ++h) {
                const float p = exp(sc[h * CH_ATTN_SK_KT + jj] - tgs[h * 2u]);
                for (uint e = 0; e < EPL; ++e) {
                    acc[h][e] += p * vv[e];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint h = 0; h < G; ++h) {
        for (uint e = 0; e < EPL; ++e) {
            part_acc[((sg * G + h) * 32u + lane) * EPL + e] = acc[h][e];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 0; h < G; ++h) {
        device float *pmh = scratch + ((row0 + h) * CH_ATTN_SK_SLICES + slice) * (HD + 2u);
        if (tid == 0u) {
            pmh[0] = mrun[h];
            pmh[1] = lrun[h];
        }
        for (uint e = 0; e < EPL; ++e) {
            if (sg == 0u && active) {
                const float total_acc = part_acc[((h) * 32u + lane) * EPL + e] +
                                       part_acc[((G + h) * 32u + lane) * EPL + e] +
                                       part_acc[((2u * G + h) * 32u + lane) * EPL + e] +
                                       part_acc[((3u * G + h) * 32u + lane) * EPL + e];
                pmh[2u + d0 + e] = total_acc;
            }
        }
    }
}

/* Fold the per-slice partials into the output rows: m* = max over this
 * row's active slices, l* = sum l_s e^(m_s - m*), out = sum acc_s
 * e^(m_s - m*) / l*. One head per SIMD group; per-slice merge weights
 * are computed once (simd-scoped sync only) and the dim fold runs four
 * independent accumulators (a single runtime-bounded loop left a
 * 128-deep dependent load chain per lane in an earlier version). */
template <uint HD, uint G>
kernel void ch_qwen35_attn_sk_merge(device const float *scratch [[buffer(0)]],
                                    device float *out [[buffer(1)]],
                                    constant uint &pos0 [[buffer(2)]],
                                    constant uint &head_dim [[buffer(3)]],
                                    constant uint &group [[buffer(4)]],
                                    constant uint &heads [[buffer(5)]],
                                    constant uint &window [[buffer(6)]],
                                    uint2 tg [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
    (void)tid;
    if (head_dim != HD || group != G) {
        return;
    }
    threadgroup float wg[G][CH_ATTN_SK_SLICES];
    const uint kv_head = tg.x;
    const uint t = tg.y;
    const uint seq = pos0 + t + 1u;
    const uint start = ch_attn_start(seq, window);
    const uint total = seq - start;
    const uint slices = ch_attn_sk_slices(total);
    const size_t row0 = (size_t)t * heads + (size_t)kv_head * G;
    for (uint h = sg; h < G; h += 4u) {
        const device float *base = scratch + ((row0 + h) * CH_ATTN_SK_SLICES) * (HD + 2u);
        float m = -INFINITY;
        for (uint s2 = lane; s2 < slices; s2 += 32u) {
            m = max(m, base[s2 * (HD + 2u)]);
        }
        m = simd_max(m);
        float l = 0.0f;
        for (uint s2 = lane; s2 < slices; s2 += 32u) {
            const float ls = base[s2 * (HD + 2u) + 1u];
            if (ls > 0.0f) {
                l += ls * exp(base[s2 * (HD + 2u)] - m);
            }
        }
        l = simd_sum(l);
        for (uint s2 = lane; s2 < slices; s2 += 32u) {
            const float ms = base[s2 * (HD + 2u)];
            const float ls = base[s2 * (HD + 2u) + 1u];
            wg[h][s2] = ls > 0.0f ? exp(ms - m) : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = lane; d < HD; d += 32u) {
            float a0 = 0.0f;
            float a1 = 0.0f;
            float a2 = 0.0f;
            float a3 = 0.0f;
            uint s2 = 0u;
            for (; s2 + 3u < slices; s2 += 4u) {
                a0 += base[(s2 + 0u) * (HD + 2u) + 2u + d] * wg[h][s2 + 0u];
                a1 += base[(s2 + 1u) * (HD + 2u) + 2u + d] * wg[h][s2 + 1u];
                a2 += base[(s2 + 2u) * (HD + 2u) + 2u + d] * wg[h][s2 + 2u];
                a3 += base[(s2 + 3u) * (HD + 2u) + 2u + d] * wg[h][s2 + 3u];
            }
            for (; s2 < slices; ++s2) {
                a0 += base[s2 * (HD + 2u) + 2u + d] * wg[h][s2];
            }
            const float a = (a0 + a1) + (a2 + a3);
            out[(row0 + h) * HD + d] = a / l;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define CH_ATTN_SK_ARGS(C)                                                                  \
    device const float *q [[buffer(0)]], device const C *k_cache [[buffer(1)]],             \
        device const C *v_cache [[buffer(2)]], device float *scratch [[buffer(3)]],          \
        constant uint &pos0 [[buffer(4)]], constant uint &cap [[buffer(5)]],                \
        constant uint &head_dim [[buffer(6)]], constant uint &group [[buffer(7)]],           \
        constant float &scale [[buffer(8)]], constant uint &heads [[buffer(9)]],             \
        constant uint &window [[buffer(10)]],                                                \
        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]

template [[host_name("ch_qwen35_attn_sk_256x4")]] kernel void
ch_qwen35_attn_sk_partial<half, 256u, 4u>(CH_ATTN_SK_ARGS(half));
template [[host_name("ch_qwen35_attn_sk_16x2")]] kernel void
ch_qwen35_attn_sk_partial<half, 16u, 2u>(CH_ATTN_SK_ARGS(half));
template [[host_name("ch_qwen35_attn_sk_128x2")]] kernel void
ch_qwen35_attn_sk_partial<half, 128u, 2u>(CH_ATTN_SK_ARGS(half));
template [[host_name("ch_qwen35_attn_sk_128x4")]] kernel void
ch_qwen35_attn_sk_partial<half, 128u, 4u>(CH_ATTN_SK_ARGS(half));
template [[host_name("ch_qwen35_attn_sk_f32_16x2")]] kernel void
ch_qwen35_attn_sk_partial<float, 16u, 2u>(CH_ATTN_SK_ARGS(float));
template [[host_name("ch_qwen35_attn_sk_f32_128x2")]] kernel void
ch_qwen35_attn_sk_partial<float, 128u, 2u>(CH_ATTN_SK_ARGS(float));
template [[host_name("ch_qwen35_attn_sk_f32_128x4")]] kernel void
ch_qwen35_attn_sk_partial<float, 128u, 4u>(CH_ATTN_SK_ARGS(float));

#define CH_ATTN_SK_MERGE_ARGS                                                              \
    device const float *scratch [[buffer(0)]], device float *out [[buffer(1)]],             \
        constant uint &pos0 [[buffer(2)]], constant uint &head_dim [[buffer(3)]],            \
        constant uint &group [[buffer(4)]], constant uint &heads [[buffer(5)]],             \
        constant uint &window [[buffer(6)]],                                                 \
        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]

template [[host_name("ch_qwen35_attn_sk_merge_256x4")]] kernel void
ch_qwen35_attn_sk_merge<256u, 4u>(CH_ATTN_SK_MERGE_ARGS);
template [[host_name("ch_qwen35_attn_sk_merge_16x2")]] kernel void
ch_qwen35_attn_sk_merge<16u, 2u>(CH_ATTN_SK_MERGE_ARGS);
template [[host_name("ch_qwen35_attn_sk_merge_128x2")]] kernel void
ch_qwen35_attn_sk_merge<128u, 2u>(CH_ATTN_SK_MERGE_ARGS);
template [[host_name("ch_qwen35_attn_sk_merge_128x4")]] kernel void
ch_qwen35_attn_sk_merge<128u, 4u>(CH_ATTN_SK_MERGE_ARGS);

/* attn[i] *= sigmoid(gate[i]) — the joint Q-projection output gate. */
kernel void ch_qwen35_attn_gate(device float *attn [[buffer(0)]],
                                device const float *gate [[buffer(1)]],
                                constant uint &n [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    attn[gid] *= 1.0f / (1.0f + exp(-gate[gid]));
}

/* x[i] = silu(x[i]) in place (conv output activation, graph order). */
kernel void ch_qwen35_silu(device float *x [[buffer(0)]],
                           constant uint &n [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    const float v = x[gid];
    x[gid] = v / (1.0f + exp(-v));
}

/* Split the conv output [3*width] into q, k, v [width] each (graph order:
 * q first, then k, then v). */
kernel void ch_qwen35_split_qkv(device const float *y [[buffer(0)]],
                                device float *q [[buffer(1)]],
                                device float *k [[buffer(2)]],
                                device float *v [[buffer(3)]],
                                constant uint &width [[buffer(4)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= width) {
        return;
    }
    q[gid] = y[gid];
    k[gid] = y[width + gid];
    v[gid] = y[2u * width + gid];
}

/* ---------------- Chunk-wide Gated DeltaNet core (engine prefill) ----------
 * Each kernel processes all `n` tokens of a prefill chunk in one launch with
 * exactly the per-element arithmetic of its per-token counterpart above,
 * so chunked prefill equals running the per-token core n times. */

/* Causal convolution over a chunk. Token t's tap j (j < k-1) reads the
 * input at time t-(k-1)+j: from the chunk when that is >= 0, else from the
 * carried history (oldest first), exactly as ch_qwen35_conv_step sees it
 * after t shifts. One thread per (channel, token). */
kernel void ch_qwen35_conv_chunk(device const float *hist [[buffer(0)]],
                                 device const float *x [[buffer(1)]],
                                 device const float *w [[buffer(2)]],
                                 device float *y [[buffer(3)]],
                                 constant uint &channels [[buffer(4)]],
                                 constant uint &kern [[buffer(5)]],
                                 constant uint &n [[buffer(6)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    const uint t = gid.y;
    if (c >= channels || t >= n) {
        return;
    }
    const uint k = kern;
    float acc = 0.0f;
    for (uint j = 0; j + 1u < k; ++j) {
        const int src = (int)t + (int)j - (int)(k - 1u);
        const float v = src >= 0 ? x[(size_t)src * channels + c]
                                 : hist[(size_t)(t + j) * channels + c];
        acc += w[c * k + j] * v;
    }
    acc += w[c * k + (k - 1u)] * x[(size_t)t * channels + c];
    y[(size_t)t * channels + c] = acc;
}

/* History after a chunk: the last k-1 inputs (oldest first), in place.
 * Reads index n+j > j, so ascending j never reads an overwritten slot. */
 kernel void ch_qwen35_conv_chunk_shift(device float *hist [[buffer(0)]],
                                        device const float *x [[buffer(1)]],
                                        constant uint &channels [[buffer(2)]],
                                        constant uint &kern [[buffer(3)]],
                                        constant uint &n [[buffer(4)]],
                                        uint c [[thread_position_in_grid]]) {
     if (c >= channels || kern < 2u) {
         return;
     }
     const uint h = kern - 1u;
     for (uint j = 0; j < h; ++j) {
         const uint src = n + j;
         hist[(size_t)j * channels + c] =
             src < h ? hist[(size_t)src * channels + c] : x[(size_t)(src - h) * channels + c];
     }
 }

/* Per-token q/k/v split of conv rows [n, 3*width] into [n, width] each. */
kernel void ch_qwen35_split_qkv_chunk(device const float *y [[buffer(0)]],
                                      device float *q [[buffer(1)]],
                                      device float *k [[buffer(2)]],
                                      device float *v [[buffer(3)]],
                                      constant uint &width [[buffer(4)]],
                                      constant uint &n [[buffer(5)]],
                                      uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x;
    const uint t = gid.y;
    if (i >= width || t >= n) {
        return;
    }
    const size_t row = (size_t)t * 3u * width;
    q[(size_t)t * width + i] = y[row + i];
    k[(size_t)t * width + i] = y[row + width + i];
    v[(size_t)t * width + i] = y[row + 2u * width + i];
}

/* ch_qwen35_beta_decay over [n, per] rows; the per-head biases index by
 * gid % per. */
kernel void ch_qwen35_beta_decay_chunk(device const float *raw_beta [[buffer(0)]],
                                       device const float *alpha [[buffer(1)]],
                                       device const float *dt_bias [[buffer(2)]],
                                       device const float *ssm_a [[buffer(3)]],
                                       device float *beta_out [[buffer(4)]],
                                       device float *gate_out [[buffer(5)]],
                                       constant uint &total [[buffer(6)]],
                                       constant uint &per [[buffer(7)]],
                                       uint gid [[thread_position_in_grid]]) {
    if (gid >= total) {
        return;
    }
    const uint h = gid % per;
    const float b = raw_beta[gid];
    beta_out[gid] = 1.0f / (1.0f + exp(-b));
    const float t = alpha[gid] + dt_bias[h];
    const float sp = t > 20.0f ? t : log(1.0f + exp(t));
    gate_out[gid] = ssm_a[h] * sp;
}

/* The delta rule over a chunk: thread (column j, head) walks tokens
 * t = 0..n-1 in order with exactly ch_qwen35_gdn_step's per-element
 * arithmetic, so the state after the chunk equals n single steps. Per
 * token, rows are `qk_stride` (q, k), `v_stride` (v, out) and `g_stride`
 * (gate, beta) wide; head h reads offset h*k_dim / h*v_dim / h. */
kernel void ch_qwen35_gdn_chunk(device const float *q [[buffer(0)]],
                                device const float *k [[buffer(1)]],
                                device const float *v [[buffer(2)]],
                                device const float *gate [[buffer(3)]],
                                device const float *beta [[buffer(4)]],
                                device float *state [[buffer(5)]],
                                device float *out [[buffer(6)]],
                                    device float *snap [[buffer(7)]],
                                constant uint &k_dim [[buffer(8)]],
                                constant uint &v_dim [[buffer(9)]],
                                constant float &q_scale [[buffer(10)]],
                                constant uint &heads [[buffer(11)]],
                                constant uint &n [[buffer(12)]],
                                constant uint &qk_stride [[buffer(13)]],
                                constant uint &v_stride [[buffer(14)]],
                                constant uint &g_stride [[buffer(15)]],
                                     constant uint &snap_t [[buffer(16)]],
                                     constant uint &snap_stride [[buffer(17)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    const uint j = gid.x;
    const uint head = gid.y;
    const uint sk = k_dim;
    const uint sv = v_dim;
    if (j >= sv || head >= heads) {
        return;
    }
    device float *s = state + (size_t)head * (size_t)sk * (size_t)sv;
    for (uint t = 0; t < n; ++t) {
        const size_t qk_row = (size_t)t * qk_stride + (size_t)head * sk;
        const size_t v_row = (size_t)t * v_stride + (size_t)head * sv;
        const device float *kk = k + qk_row;
        const device float *qq = q + qk_row;
        const float decay = exp(gate[(size_t)t * g_stride + head]);
        const float b = beta[(size_t)t * g_stride + head];
        float sk_acc = 0.0f;
        for (uint i = 0; i < sk; ++i) {
            sk_acc += (s[i * sv + j] * decay) * kk[i];
        }
        const float d = b * (v[v_row + j] - sk_acc);
        float acc = 0.0f;
        for (uint i = 0; i < sk; ++i) {
            const float updated = s[i * sv + j] * decay + kk[i] * d;
            s[i * sv + j] = updated;
            acc += updated * (qq[i] * q_scale);
        }
        out[v_row + j] = acc;
        if (t <= snap_t) {
            /* Rollback points (speculative verify): the state after every
             * row t' <= snap_t lands in snapshot slot t'. snap_stride = 0
             * is the single-slot ABI (row 0 only, offset 0); UINT32_MAX
             * never matches. */
            device float *snap_s =
                snap + (size_t)t * (size_t)snap_stride + (size_t)head * (size_t)sk * (size_t)sv;
            for (uint i = 0; i < sk; ++i) {
                snap_s[i * sv + j] = s[i * sv + j];
            }
        }
    }
}



/* Partial split-half RoPE over a chunk: threadgroup (head, t) rotates q or
 * k row t at position pos0+t, exactly ch_qwen35_rope_partial_f32's math.
 * q rows are [n, n_q_heads, head_dim], k rows [n, n_kv_heads, head_dim]. */
kernel void ch_qwen35_rope_partial_chunk(device float *q [[buffer(0)]],
                                         device float *k [[buffer(1)]],
                                         device const float *inv_freq [[buffer(2)]],
                                         constant uint &pos0 [[buffer(3)]],
                                         constant uint &n_q_heads [[buffer(4)]],
                                         constant uint &n_kv_heads [[buffer(5)]],
                                         constant uint &head_dim [[buffer(6)]],
                                         constant uint &n_rot [[buffer(7)]],
                                         uint2 tg [[threadgroup_position_in_grid]],
                                         uint lane [[thread_index_in_threadgroup]]) {
    const uint half_rot = n_rot / 2u;
    if (lane >= half_rot) {
        return;
    }
    const uint head = tg.x;
    const uint t = tg.y;
    const bool is_q = head < n_q_heads;
    const uint local_head = is_q ? head : head - n_q_heads;
    device float *row = is_q
        ? q + ((size_t)t * n_q_heads + local_head) * (size_t)head_dim
        : k + ((size_t)t * n_kv_heads + local_head) * (size_t)head_dim;
    const float angle = (float)(pos0 + t) * inv_freq[lane];
    const float c = cos(angle);
    const float s = sin(angle);
    const float a = row[lane];
    const float b = row[lane + half_rot];
    row[lane] = a * c - b * s;
    row[lane + half_rot] = a * s + b * c;
}

/* KV store of a chunk's rows [n, kv_heads, head_dim] at positions
 * pos0+t: packed f16 (same rounding as ch_qwen35_kv_store_f16) or f32
 * (an exact copy). */
template <typename C>
kernel void ch_qwen35_kv_store_chunk(device const float *k_src [[buffer(0)]],
                                     device const float *v_src [[buffer(1)]],
                                     device C *k_cache [[buffer(2)]],
                                     device C *v_cache [[buffer(3)]],
                                         constant uint &pos0 [[buffer(4)]],
                                         constant uint &cap [[buffer(5)]],
                                         constant uint &head_dim [[buffer(6)]],
                                         constant uint &kv_heads [[buffer(7)]],
                                         constant uint &n [[buffer(8)]],
                                         uint gid [[thread_position_in_grid]]) {
    const uint per = kv_heads * head_dim;
    if (gid >= per * n) {
        return;
    }
    const size_t t = gid / per;
    const size_t rem = gid % per;
    const size_t head = rem / head_dim;
    const size_t lane = rem % head_dim;
    const size_t dst = (head * (size_t)cap + (size_t)pos0 + t) * (size_t)head_dim + lane;
    k_cache[dst] = (C)k_src[gid];
    v_cache[dst] = (C)v_src[gid];
}

#define CH_KV_CHUNK_ARGS(C)                                                                  \
    device const float *k_src [[buffer(0)]], device const float *v_src [[buffer(1)]],        \
        device C *k_cache [[buffer(2)]], device C *v_cache [[buffer(3)]],                     \
        constant uint &pos0 [[buffer(4)]], constant uint &cap [[buffer(5)]],                  \
        constant uint &head_dim [[buffer(6)]], constant uint &kv_heads [[buffer(7)]],         \
        constant uint &n [[buffer(8)]], uint gid [[thread_position_in_grid]]

template [[host_name("ch_qwen35_kv_store_f16_chunk")]] kernel void
ch_qwen35_kv_store_chunk<half>(CH_KV_CHUNK_ARGS(half));
template [[host_name("ch_qwen35_kv_store_f32_chunk")]] kernel void
ch_qwen35_kv_store_chunk<float>(CH_KV_CHUNK_ARGS(float));

/* ch_qwen35_gdn_chunk with the state column held in registers for the
 * whole chunk (read once, written once) instead of re-read from device
 * memory twice per token. Each column is split across P lanes of one
 * SIMD group, each holding SK / P rows; the two per-token dot products
 * are lane partial sums combined with XOR shuffles, so every lane of a
 * column sees the same total. A SIMD group covers C = 32 / P adjacent
 * columns and lane l works on column l % C, so neighbouring lanes touch
 * neighbouring state columns (coalesced state loads and stores). Same per-element expressions as the generic
 * kernel; the dot-product summation order differs (tolerance-qualified).
 * Decode's single-token step runs this kernel with n = 1, so chunked
 * prefill and decode stay bit-identical to each other. Instantiated for
 * the key dimensions in use (the array size must be a compile-time
 * constant); the grid is P * v_dim lanes per head in threadgroups of 32,
 * and v_dim must be a multiple of C. */
template <uint SK, uint P>
kernel void ch_qwen35_gdn_chunk_reg(device const float *q [[buffer(0)]],
                                    device const float *k [[buffer(1)]],
                                    device const float *v [[buffer(2)]],
                                    device const float *gate [[buffer(3)]],
                                    device const float *beta [[buffer(4)]],
                                    device float *state [[buffer(5)]],
                                    device float *out [[buffer(6)]],
                                    device float *snap [[buffer(7)]],
                                    constant uint &k_dim [[buffer(8)]],
                                    constant uint &v_dim [[buffer(9)]],
                                    constant float &q_scale [[buffer(10)]],
                                    constant uint &heads [[buffer(11)]],
                                    constant uint &n [[buffer(12)]],
                                    constant uint &qk_stride [[buffer(13)]],
                                    constant uint &v_stride [[buffer(14)]],
                                    constant uint &g_stride [[buffer(15)]],
                                     constant uint &snap_t [[buffer(16)]],
                                     constant uint &snap_stride [[buffer(17)]],
                                     uint2 gid [[thread_position_in_grid]]) {
     constexpr uint RPL = SK / P;
    constexpr uint C = 32u / P;
    const uint lane = gid.x % 32u;
    const uint j = (gid.x / 32u) * C + lane % C;
    const uint part = lane / C;
    const uint head = gid.y;
    const uint sv = v_dim;
    /* Whole columns only: a column's P lanes all pass or all return, so
     * the shuffles below always see their partners. */
    if (j >= sv || head >= heads || k_dim != SK) {
        return;
    }
    const uint r0 = part * RPL;
    device float *s = state + (size_t)head * (size_t)SK * (size_t)sv;
    float col[RPL];
    for (uint i = 0; i < RPL; ++i) {
        col[i] = s[(r0 + i) * sv + j];
    }
    for (uint t = 0; t < n; ++t) {
        const size_t qk_row = (size_t)t * qk_stride + (size_t)head * SK + r0;
        const size_t v_row = (size_t)t * v_stride + (size_t)head * sv;
        const device float *kk = k + qk_row;
        const device float *qq = q + qk_row;
        const float decay = exp(gate[(size_t)t * g_stride + head]);
        const float b = beta[(size_t)t * g_stride + head];
        float sk_acc = 0.0f;
        for (uint i = 0; i < RPL; ++i) {
            sk_acc += (col[i] * decay) * kk[i];
        }
        for (uint m = C; m < 32u; m <<= 1) {
            sk_acc += simd_shuffle_xor(sk_acc, (ushort)m);
        }
        const float d = b * (v[v_row + j] - sk_acc);
        float acc = 0.0f;
        for (uint i = 0; i < RPL; ++i) {
            const float updated = col[i] * decay + kk[i] * d;
            col[i] = updated;
            acc += updated * (qq[i] * q_scale);
        }
        for (uint m = C; m < 32u; m <<= 1) {
            acc += simd_shuffle_xor(acc, (ushort)m);
        }
        if (part == 0u) {
            out[v_row + j] = acc;
        }
        if (t <= snap_t) {
            /* Rollback points (speculative verify): the state after every
             * row t' <= snap_t lands in snapshot slot t'. snap_stride = 0
             * is the single-slot ABI (row 0 only, offset 0); UINT32_MAX
             * never matches. */
            device float *snap_s =
                snap + (size_t)t * (size_t)snap_stride + (size_t)head * (size_t)SK * (size_t)sv;
            for (uint i = 0; i < RPL; ++i) {
                snap_s[(r0 + i) * sv + j] = col[i];
            }
        }
    }
    for (uint i = 0; i < RPL; ++i) {
        s[(r0 + i) * sv + j] = col[i];
    }
}

/* Lanes per state column (must match the host's CH_GDN_LANES). */
#define CH_GDN_LANES 8u

#define CH_GDN_REG_ARGS                                                          \
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],      \
        device const float *v [[buffer(2)]], device const float *gate [[buffer(3)]], \
        device const float *beta [[buffer(4)]], device float *state [[buffer(5)]],  \
        device float *out [[buffer(6)]], device float *snap [[buffer(7)]],          \
        constant uint &k_dim [[buffer(8)]], constant uint &v_dim [[buffer(9)]],     \
        constant float &q_scale [[buffer(10)]], constant uint &heads [[buffer(11)]], \
        constant uint &n [[buffer(12)]], constant uint &qk_stride [[buffer(13)]],   \
        constant uint &v_stride [[buffer(14)]], constant uint &g_stride [[buffer(15)]], \
        constant uint &snap_t [[buffer(16)]],                                      \
        constant uint &snap_stride [[buffer(17)]],                                \
        uint2 gid [[thread_position_in_grid]]

template [[host_name("ch_qwen35_gdn_chunk_k128")]] kernel void
ch_qwen35_gdn_chunk_reg<128u, CH_GDN_LANES>(CH_GDN_REG_ARGS);


/* ---------------- Q4_0 kernels (2026-11-22) ----------------
 * Block layout (llama.cpp): [f16 d][u8 qs; 16]; low nibble of byte j is
 * element j, high nibble is element j+16; value = d * (nibble - 8).
 * Every structure mirrors the Q8_0 counterpart exactly — only the
 * dequant changes. */

/* Multi-row slice form (4 rows per group, lanes covering 8-value
 * slices, x amortized across the rows): serves high-row shapes where
 * the group count alone saturates DRAM and the x-load amortization
 * wins (measured: 6912x1152 gate/up 4.75-5.74 ms/token across machine
 * states vs the one-row form's 7.3-8.1). Selected by the wrapper for
 * rows >= 2048. */
kernel void ch_q4_0_gemv(device const uchar *w [[buffer(0)]],
                          device const float *x [[buffer(1)]],
                          device float *y [[buffer(2)]],
                          constant uint &rows [[buffer(3)]],
                          constant uint &cols [[buffer(4)]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    const uint blocks = cols / 32u;
    const size_t row_bytes = (size_t)blocks * 18u;
    const uint first = (tg * CH_Q8_SG + sg) * CH_Q8_ROWS;
    if (first >= rows) {
        return;
    }
    const uint sub = lane % 4u;        /* 8-value slice of the block */
    const uint lane_blk = lane / 4u;   /* which of the 8 blocks this step */
    const uint byte0 = (sub % 2u) * 8u;
    const bool hi = sub >= 2u;

    float acc[CH_Q8_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint b = lane_blk; b < blocks; b += 8u) {
        const device packed_float4 *xv =
            (const device packed_float4 *)(x + (size_t)b * 32u + sub * 8u);
        const float4 x0 = float4(xv[0]);
        const float4 x1 = float4(xv[1]);
        for (uint r = 0; r < CH_Q8_ROWS; ++r) {
            const uint row = first + r;
            if (row >= rows) {
                break;
            }
            const device uchar *blk = w + (size_t)row * row_bytes + (size_t)b * 18u;
            const float d = (float)(*(const device half *)blk);
            /* The 8 data bytes this lane covers, as two packed 4-byte
             * loads (alignment 1, so the 18-byte block stride needs no
             * special case) instead of eight scalar byte loads; the
             * nibble math runs on uchar vectors (unsigned, so >> 4 is
             * the logical shift the scalar form's int promotion gave).
             * Two alternatives were measured and rejected on the same
             * in-model A/B (gemma3 Q4_0 decode, per token): a
             * quad-local simd_shuffle reuse of the low lanes' bytes
             * (6.30 vs 5.05 ms on the gate/up shape - the high lanes'
             * independent loads overlap and coalesce, the shuffle
             * serializes behind them) and a two-strided-blocks-per-
             * iteration form (down shape 1.95 vs 1.77 ms, the most
             * stable instrument). Values are identical to the scalar
             * form. */
            const device packed_uchar4 *qv =
                (const device packed_uchar4 *)(blk + 2u + byte0);
            const uchar4 c0 = uchar4(qv[0]);
            const uchar4 c1 = uchar4(qv[1]);
            float4 q0, q1;
            if (!hi) {
                q0 = float4(c0 & (uchar4)0x0fu) - 8.0f;
                q1 = float4(c1 & (uchar4)0x0fu) - 8.0f;
            } else {
                q0 = float4(c0 >> 4) - 8.0f;
                q1 = float4(c1 >> 4) - 8.0f;
            }
            acc[r] += d * (dot(q0, x0) + dot(q1, x1));
        }
    }
    for (uint r = 0; r < CH_Q8_ROWS; ++r) {
        const float total = simd_sum(acc[r]);
        const uint row = first + r;
        if (lane == 0u && row < rows) {
            y[row] = total;
        }
    }
}

kernel void ch_q4_0_gemv_one(device const uchar *w [[buffer(0)]],
                          device const float *x [[buffer(1)]],
                          device float *y [[buffer(2)]],
                          constant uint &rows [[buffer(3)]],
                          constant uint &cols [[buffer(4)]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    /* One row per SIMD group (the llama.cpp kernel_mul_mat_vec
     * structure): each lane carries whole 32-value blocks strided by
     * 32, so the 36-block gate/up loop is a single pass for 32 of 36
     * lanes and a row is one load stream plus one simd_sum. Selected
     * by the wrapper for rows < CH_Q4_GEMV_ONE_MIN: on the low-row
     * decode shapes the multi-row form's group count (rows/8) could
     * not hide DRAM latency and this form wins (measured in-model,
     * per token: down 1152x6912 1.70-1.78 -> 1.35-1.65 ms, o
     * 1152x1024 0.37-0.49 -> 0.31); on high-row shapes it loses to
     * the x-amortization (gate/up 6912x1152 4.75-5.74 vs 7.3-8.1),
     * so both kernels stay and the wrapper routes. The per-row lane
     * partials differ from the slice form's, so results move within
     * the registered CPU-reference tolerance; no bit-gate pins this
     * kernel (the Q4_0 family is tolerance-qualified against the
     * scalar reference and, on real artifacts, against llama.cpp
     * Metal envelopes). */
    const uint blocks = cols / 32u;
    const size_t row_bytes = (size_t)blocks * 18u;
    const uint row = tg * CH_Q8_SG + sg;
    if (row >= rows) {
        return;
    }
    const device uchar *base = w + (size_t)row * row_bytes;
    float acc = 0.0f;
    for (uint b = lane; b < blocks; b += 32u) {
        const device uchar *blk = base + (size_t)b * 18u;
        const float d = (float)(*(const device half *)blk);
        /* 16 data bytes as four packed 4-byte loads (alignment 1, so
         * the 18-byte block stride needs no special case); nibble math
         * on uchar vectors (unsigned, so >> 4 is the logical shift the
         * scalar form's int promotion gave). Low plane = values
         * 0..15, high plane = 16..31. */
        const device packed_uchar4 *qv =
            (const device packed_uchar4 *)(blk + 2u);
        const uchar4 c0 = uchar4(qv[0]);
        const uchar4 c1 = uchar4(qv[1]);
        const uchar4 c2 = uchar4(qv[2]);
        const uchar4 c3 = uchar4(qv[3]);
        const device packed_float4 *xv =
            (const device packed_float4 *)(x + (size_t)b * 32u);
        const float4 x0 = float4(xv[0]);
        const float4 x1 = float4(xv[1]);
        const float4 x2 = float4(xv[2]);
        const float4 x3 = float4(xv[3]);
        const float4 x4 = float4(xv[4]);
        const float4 x5 = float4(xv[5]);
        const float4 x6 = float4(xv[6]);
        const float4 x7 = float4(xv[7]);
        acc += d * (dot(float4(c0 & (uchar4)0x0fu) - 8.0f, x0)
                  + dot(float4(c1 & (uchar4)0x0fu) - 8.0f, x1)
                  + dot(float4(c2 & (uchar4)0x0fu) - 8.0f, x2)
                  + dot(float4(c3 & (uchar4)0x0fu) - 8.0f, x3)
                  + dot(float4(c0 >> 4) - 8.0f, x4)
                  + dot(float4(c1 >> 4) - 8.0f, x5)
                  + dot(float4(c2 >> 4) - 8.0f, x6)
                  + dot(float4(c3 >> 4) - 8.0f, x7));
    }
    const float total = simd_sum(acc);
    if (lane == 0u) {
        y[row] = total;
    }
}

/* Dequantize one Q4_0 matrix row (or embedding row) to f32: one thread
 * per output element. */
kernel void ch_q4_0_row_f32(device const uchar *w [[buffer(0)]],
                             device float *out [[buffer(1)]],
                             constant uint &row [[buffer(2)]],
                             constant uint &cols [[buffer(3)]],
                             constant uint &out_off [[buffer(4)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) {
        return;
    }
    const uint blocks = cols / 32u;
    const device uchar *wrow = w + (size_t)row * (size_t)blocks * 18u;
    const device uchar *blk = wrow + (size_t)(gid / 32u) * 18u;
    const float d = (float)(*(const device half *)blk);
    const uint j = gid % 32u;
    const uint nib = j < 16u ? (blk[2u + j] & 0x0fu) : (blk[2u + j - 16u] >> 4);
    out[out_off + gid] = d * ((float)nib - 8.0f);
}

/* Embedding lookup whose row is a device-side token id (ids[slot]). */
kernel void ch_q4_0_row_indirect_f32(device const uchar *w [[buffer(0)]],
                                     device float *out [[buffer(1)]],
                                     device const uint *ids [[buffer(2)]],
                                     constant uint &slot [[buffer(3)]],
                                     constant uint &rows [[buffer(4)]],
                                     constant uint &cols [[buffer(5)]],
                                     uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) {
        return;
    }
    const uint row = ids[slot];
    if (row >= rows) {
        out[gid] = 0.0f;
        return;
    }
    const uint blocks = cols / 32u;
    const device uchar *blk = w + ((size_t)row * (size_t)blocks + gid / 32u) * 18u;
    const float d = (float)(*(const device half *)blk);
    const uint j = gid % 32u;
    const uint nib = j < 16u ? (blk[2u + j] & 0x0fu) : (blk[2u + j - 16u] >> 4);
    out[gid] = d * ((float)nib - 8.0f);
}

/* Per-row Q4_0 GEMM in slices of at most 32 tokens, the Q8_0 gemm's
 * structure. */
kernel void ch_q4_0_gemm(device const uchar *w [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device float *y [[buffer(2)]],
                         constant uint &rows [[buffer(3)]],
                         constant uint &cols [[buffer(4)]],
                         constant uint &batch [[buffer(5)]],
                         constant uint &x_off [[buffer(6)]],
                         constant uint &y_off [[buffer(7)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint lane [[thread_position_in_threadgroup]]) {
    if (row >= rows || batch == 0u || batch > 32u) {
        return;
    }
    const uint blocks = cols / 32u;
    const device uchar *wrow = w + (size_t)row * (size_t)blocks * 18u;
    const uint sub = lane % 4u;
    const uint lane_blk = lane / 4u;
    const uint byte0 = (sub % 2u) * 8u;
    const bool hi = sub >= 2u;
    float acc[32];
    for (uint b = 0; b < batch; ++b) {
        acc[b] = 0.0f;
    }
    for (uint blk = lane_blk; blk < blocks; blk += 8u) {
        const device uchar *block = wrow + (size_t)blk * 18u;
        const float d = (float)(*(const device half *)block);
        const device uchar *qs = block + 2u + byte0;
        float4 q0, q1;
        if (!hi) {
            q0 = float4((float)(qs[0] & 0x0fu) - 8.0f, (float)(qs[1] & 0x0fu) - 8.0f,
                        (float)(qs[2] & 0x0fu) - 8.0f, (float)(qs[3] & 0x0fu) - 8.0f);
            q1 = float4((float)(qs[4] & 0x0fu) - 8.0f, (float)(qs[5] & 0x0fu) - 8.0f,
                        (float)(qs[6] & 0x0fu) - 8.0f, (float)(qs[7] & 0x0fu) - 8.0f);
        } else {
            q0 = float4((float)(qs[0] >> 4) - 8.0f, (float)(qs[1] >> 4) - 8.0f,
                        (float)(qs[2] >> 4) - 8.0f, (float)(qs[3] >> 4) - 8.0f);
            q1 = float4((float)(qs[4] >> 4) - 8.0f, (float)(qs[5] >> 4) - 8.0f,
                        (float)(qs[6] >> 4) - 8.0f, (float)(qs[7] >> 4) - 8.0f);
        }
        const uint col = blk * 32u + sub * 8u;
        for (uint b = 0; b < batch; ++b) {
            const device packed_float4 *xv =
                (const device packed_float4 *)(x + x_off + b * cols + col);
            const float4 x0 = float4(xv[0]);
            const float4 x1 = float4(xv[1]);
            acc[b] += d * (dot(q0, x0) + dot(q1, x1));
        }
    }
    for (uint b = 0; b < batch; ++b) {
        const float total = simd_sum(acc[b]);
        if (lane == 0u) {
            y[y_off + b * rows + row] = total;
        }
    }
}

/* Batched Q4_0 GEMM on the SIMD-group matrix units: the Q8_0 f32-operand
 * kernel's tiling, fragments, edge handling and summation order, with the
 * Q4_0 dequant. Thread t owns 16 values (half a block) of row t/2: the
 * low nibbles of the block's 16 payload bytes when w_k is 0, the high
 * nibbles when w_k is 16. */
kernel void ch_q4_0_gemm_mma(device const uchar *w [[buffer(0)]],
                              device const float *x [[buffer(1)]],
                              device float *y [[buffer(2)]],
                              constant uint &rows [[buffer(3)]],
                              constant uint &cols [[buffer(4)]],
                              constant uint &batch [[buffer(5)]],
                              constant uint &x_off [[buffer(6)]],
                              constant uint &y_off [[buffer(7)]],
                              uint tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[CH_MMA_ROWS * 32u];
    threadgroup float sb[CH_MMA_TOK * 32u];

    const uint token_tiles = (batch + CH_MMA_TOK - 1u) / CH_MMA_TOK;
    const uint tok0 = (tg % token_tiles) * CH_MMA_TOK;
    if (tok0 >= batch) {
        return;
    }
    const uint tile_batch = min(batch - tok0, CH_MMA_TOK);
    const uint row0 = (tg / token_tiles) * CH_MMA_ROWS;
    const uint blocks = cols / 32u;
    const size_t row_bytes = (size_t)blocks * 18u;

    const uint w_r = tid / 2u;
    const uint w_k = (tid % 2u) * 16u;
    const device uchar *wrow = w + (size_t)min(row0 + w_r, rows - 1u) * row_bytes;
    const uint x_t = tid / 4u;
    const uint x_k = (tid % 4u) * 8u;
    const device float *xrow = x + x_off + (size_t)(tok0 + min(x_t, tile_batch - 1u)) * cols + x_k;

    threadgroup float *sa_dst = sa + 64u * ((w_k / 8u) * 8u + w_r / 8u) + (w_r % 8u);
    threadgroup float *sb_dst = sb + 64u * ((x_k / 8u) * 4u + x_t / 8u) + 8u * (x_t % 8u);

    const uint sg_r = (sg % 2u) * 4u;
    const uint sg_t = (sg / 2u) * 2u;
    simdgroup_float8x8 mc[2][4];
    for (uint j = 0; j < 2u; ++j) {
        for (uint i = 0; i < 4u; ++i) {
            mc[j][i] = simdgroup_float8x8(0.0f);
        }
    }

    const device uchar *block = wrow;
    const device float *xblk = xrow;
    const threadgroup float *lsa = sa + 64u * sg_r;
    const threadgroup float *lsb = sb + 64u * sg_t;
    for (uint blk = 0; blk < blocks; ++blk, block += 18, xblk += 32) {
        float wv[16];
        const float d = (float)(*(const device half *)block);
        const device uchar *qs = block + 2u;
        CH_UNROLL for (uint i = 0; i < 16u; ++i) {
            const uint nib = w_k == 0u ? (qs[i] & 0x0fu) : (qs[i] >> 4);
            wv[i] = d * ((float)nib - 8.0f);
        }
        const device packed_float4 *xp = (const device packed_float4 *)xblk;
        const float4 xv0 = float4(xp[0]);
        const float4 xv1 = float4(xp[1]);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        CH_UNROLL for (ushort i = 0; i < 16u; ++i) {
            sa_dst[64 * 8 * (i / 8) + 8 * (i % 8)] = wv[i];
        }
        *(threadgroup float4 *)(sb_dst) = xv0;
        *(threadgroup float4 *)(sb_dst + 4u) = xv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        CH_UNROLL for (ushort kb = 0; kb < 4u; ++kb) {
            simdgroup_float8x8 ma[4];
            simdgroup_float8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort i = 0; i < 4u; ++i) {
                simdgroup_load(ma[i], lsa + 64 * (8 * kb + i), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2u; ++j) {
                simdgroup_load(mb[j], lsb + 64 * (4 * kb + j), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2u; ++j) {
                CH_UNROLL for (ushort i = 0; i < 4u; ++i) {
                    simdgroup_multiply_accumulate(mc[j][i], mb[j], ma[i], mc[j][i]);
                }
            }
        }
    }

    if (row0 + CH_MMA_ROWS <= rows && tile_batch == CH_MMA_TOK) {
        device float *c = y + y_off + (size_t)(tok0 + sg_t * 8u) * rows + row0 + sg_r * 8u;
        for (uint j = 0; j < 2u; ++j) {
            for (uint i = 0; i < 4u; ++i) {
                simdgroup_store(mc[j][i], c + (size_t)j * 8u * rows + i * 8u, rows);
            }
        }
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *ot = sa;
    for (uint j = 0; j < 2u; ++j) {
        for (uint i = 0; i < 4u; ++i) {
            simdgroup_store(mc[j][i], ot + (sg_t + j) * 8u * CH_MMA_ROWS + (sg_r + i) * 8u,
                            CH_MMA_ROWS);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < CH_MMA_ROWS * CH_MMA_TOK; idx += 128u) {
        const uint r = idx % CH_MMA_ROWS;
        const uint b = idx / CH_MMA_ROWS;
        if (row0 + r < rows && b < tile_batch) {
            y[y_off + (size_t)(tok0 + b) * rows + row0 + r] = ot[b * CH_MMA_ROWS + r];
        }
    }
}

/* Batched Q4_0 GEMM with f16 operand tiles: the declared metal-f16-act
 * profile, the Q8_0 f16op kernel's structure with the Q4_0 dequant. */
kernel void ch_q4_0_gemm_mma_f16op(device const uchar *w [[buffer(0)]],
                                   device const float *x [[buffer(1)]],
                                   device float *y [[buffer(2)]],
                                   constant uint &rows [[buffer(3)]],
                                   constant uint &cols [[buffer(4)]],
                                   constant uint &batch [[buffer(5)]],
                                   constant uint &x_off [[buffer(6)]],
                                   constant uint &y_off [[buffer(7)]],
                                   uint tg [[threadgroup_position_in_grid]],
                                   uint tid [[thread_position_in_threadgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[CH_MMA_ROWS * 32u];
    threadgroup half sb[CH_MMA_TOK * 32u];
    threadgroup float edge_frag[4][64];

    const uint token_tiles = (batch + CH_MMA_TOK - 1u) / CH_MMA_TOK;
    const uint tok0 = (tg % token_tiles) * CH_MMA_TOK;
    if (tok0 >= batch) {
        return;
    }
    const uint tile_batch = min(batch - tok0, CH_MMA_TOK);
    const uint row0 = (tg / token_tiles) * CH_MMA_ROWS;
    const uint blocks = cols / 32u;
    const size_t row_bytes = (size_t)blocks * 18u;

    const uint w_r = tid / 2u;
    const uint w_k = (tid % 2u) * 16u;
    const device uchar *wrow = w + (size_t)min(row0 + w_r, rows - 1u) * row_bytes;
    const uint x_t = tid / 4u;
    const uint x_k = (tid % 4u) * 8u;
    const device float *xrow = x + x_off + (size_t)(tok0 + min(x_t, tile_batch - 1u)) * cols + x_k;

    threadgroup half *sa_dst = sa + 64u * ((w_k / 8u) * 8u + w_r / 8u) + (w_r % 8u);
    threadgroup half *sb_dst = sb + 64u * ((x_k / 8u) * 4u + x_t / 8u) + 8u * (x_t % 8u);

    const uint sg_r = (sg % 2u) * 4u;
    const uint sg_t = (sg / 2u) * 2u;
    simdgroup_float8x8 mc[2][4];
    for (uint j = 0; j < 2u; ++j) {
        for (uint i = 0; i < 4u; ++i) {
            mc[j][i] = simdgroup_float8x8(0.0f);
        }
    }

    const device uchar *block = wrow;
    const device float *xblk = xrow;
    const threadgroup half *lsa = sa + 64u * sg_r;
    const threadgroup half *lsb = sb + 64u * sg_t;
    for (uint blk = 0; blk < blocks; ++blk, block += 18, xblk += 32) {
        float wv[16];
        const float d = (float)(*(const device half *)block);
        const device uchar *qs = block + 2u;
        CH_UNROLL for (uint i = 0; i < 16u; ++i) {
            const uint nib = w_k == 0u ? (qs[i] & 0x0fu) : (qs[i] >> 4);
            wv[i] = d * ((float)nib - 8.0f);
        }
        const device packed_float4 *xp = (const device packed_float4 *)xblk;
        const float4 xv0 = float4(xp[0]);
        const float4 xv1 = float4(xp[1]);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        CH_UNROLL for (ushort i = 0; i < 16u; ++i) {
            sa_dst[64 * 8 * (i / 8) + 8 * (i % 8)] = (half)wv[i];
        }
        *(threadgroup half4 *)(sb_dst) = half4(xv0);
        *(threadgroup half4 *)(sb_dst + 4u) = half4(xv1);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        CH_UNROLL for (ushort kb = 0; kb < 4u; ++kb) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort i = 0; i < 4u; ++i) {
                simdgroup_load(ma[i], lsa + 64 * (8 * kb + i), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2u; ++j) {
                simdgroup_load(mb[j], lsb + 64 * (4 * kb + j), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2u; ++j) {
                CH_UNROLL for (ushort i = 0; i < 4u; ++i) {
                    simdgroup_multiply_accumulate(mc[j][i], mb[j], ma[i], mc[j][i]);
                }
            }
        }
    }

    if (row0 + CH_MMA_ROWS <= rows && tile_batch == CH_MMA_TOK) {
        device float *c = y + y_off + (size_t)(tok0 + sg_t * 8u) * rows + row0 + sg_r * 8u;
        for (uint j = 0; j < 2u; ++j) {
            for (uint i = 0; i < 4u; ++i) {
                simdgroup_store(mc[j][i], c + (size_t)j * 8u * rows + i * 8u, rows);
            }
        }
        return;
    }
    const uint tok_end = tok0 + tile_batch;
    {
        threadgroup float * const edge_frag_mine = edge_frag[sg];
        const uint rb_base = row0 + sg_r * 8u;
        const uint tb_base = tok0 + sg_t * 8u;
        for (uint j = 0; j < 2u; ++j) {
            for (uint i = 0; i < 4u; ++i) {
                const uint rb = rb_base + i * 8u;
                const uint tb = tb_base + j * 8u;
                if (rb >= rows || tb >= tok_end) {
                    continue;
                }
                if (rb + 8u <= rows && tb + 8u <= tok_end) {
                    simdgroup_store(mc[j][i], y + y_off + (size_t)tb * rows + rb, rows);
                } else {
                    simdgroup_store(mc[j][i], edge_frag_mine, 8);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint lane = tid % 32u; lane < 64u; lane += 32u) {
                        const uint fr = lane / 8u;
                        const uint ft = lane % 8u;
                        if (tb + fr < tok_end && rb + ft < rows) {
                            y[y_off + (size_t)(tb + fr) * rows + rb + ft] =
                                edge_frag_mine[fr * 8u + ft];
                        }
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
    }
}

/* ---------------- Q4_1 kernels (2026-11-22) ----------------
 * Block layout (llama.cpp): [f16 d][f16 m][u8 qs; 16]; low nibble of
 * byte j is element j, high nibble is element j+16; value = d * nibble
 * + m. The Q4_0 kernels' structures with the two-constant dequant. */

kernel void ch_q4_1_gemv(device const uchar *w [[buffer(0)]],
                          device const float *x [[buffer(1)]],
                          device float *y [[buffer(2)]],
                          constant uint &rows [[buffer(3)]],
                          constant uint &cols [[buffer(4)]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    const uint blocks = cols / 32u;
    const size_t row_bytes = (size_t)blocks * 20u;
    const uint first = (tg * CH_Q8_SG + sg) * CH_Q8_ROWS;
    if (first >= rows) {
        return;
    }
    const uint sub = lane % 4u;
    const uint lane_blk = lane / 4u;
    const uint byte0 = (sub % 2u) * 8u;
    const bool hi = sub >= 2u;

    float acc[CH_Q8_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint b = lane_blk; b < blocks; b += 8u) {
        const device packed_float4 *xv =
            (const device packed_float4 *)(x + (size_t)b * 32u + sub * 8u);
        const float4 x0 = float4(xv[0]);
        const float4 x1 = float4(xv[1]);
        for (uint r = 0; r < CH_Q8_ROWS; ++r) {
            const uint row = first + r;
            if (row >= rows) {
                break;
            }
            const device uchar *blk = w + (size_t)row * row_bytes + (size_t)b * 20u;
            const float d = (float)(*(const device half *)blk);
            const float m = (float)(*(const device half *)(blk + 2u));
            const device uchar *qs = blk + 4u + byte0;
            float4 q0, q1;
            if (!hi) {
                q0 = float4((float)(qs[0] & 0x0fu) * d + m, (float)(qs[1] & 0x0fu) * d + m,
                            (float)(qs[2] & 0x0fu) * d + m, (float)(qs[3] & 0x0fu) * d + m);
                q1 = float4((float)(qs[4] & 0x0fu) * d + m, (float)(qs[5] & 0x0fu) * d + m,
                            (float)(qs[6] & 0x0fu) * d + m, (float)(qs[7] & 0x0fu) * d + m);
            } else {
                q0 = float4((float)(qs[0] >> 4) * d + m, (float)(qs[1] >> 4) * d + m,
                            (float)(qs[2] >> 4) * d + m, (float)(qs[3] >> 4) * d + m);
                q1 = float4((float)(qs[4] >> 4) * d + m, (float)(qs[5] >> 4) * d + m,
                            (float)(qs[6] >> 4) * d + m, (float)(qs[7] >> 4) * d + m);
            }
            acc[r] += dot(q0, x0) + dot(q1, x1);
        }
    }
    for (uint r = 0; r < CH_Q8_ROWS; ++r) {
        const float total = simd_sum(acc[r]);
        const uint row = first + r;
        if (lane == 0u && row < rows) {
            y[row] = total;
        }
    }
}

kernel void ch_q4_1_row_f32(device const uchar *w [[buffer(0)]],
                             device float *out [[buffer(1)]],
                             constant uint &row [[buffer(2)]],
                             constant uint &cols [[buffer(3)]],
                             constant uint &out_off [[buffer(4)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) {
        return;
    }
    const uint blocks = cols / 32u;
    const device uchar *wrow = w + (size_t)row * (size_t)blocks * 20u;
    const device uchar *blk = wrow + (size_t)(gid / 32u) * 20u;
    const float d = (float)(*(const device half *)blk);
    const float m = (float)(*(const device half *)(blk + 2u));
    const uint j = gid % 32u;
    const uint nib = j < 16u ? (blk[4u + j] & 0x0fu) : (blk[4u + j - 16u] >> 4);
    out[out_off + gid] = (float)nib * d + m;
}

kernel void ch_q4_1_row_indirect_f32(device const uchar *w [[buffer(0)]],
                                     device float *out [[buffer(1)]],
                                     device const uint *ids [[buffer(2)]],
                                     constant uint &slot [[buffer(3)]],
                                     constant uint &rows [[buffer(4)]],
                                     constant uint &cols [[buffer(5)]],
                                     uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) {
        return;
    }
    const uint row = ids[slot];
    if (row >= rows) {
        out[gid] = 0.0f;
        return;
    }
    const uint blocks = cols / 32u;
    const device uchar *blk = w + ((size_t)row * (size_t)blocks + gid / 32u) * 20u;
    const float d = (float)(*(const device half *)blk);
    const float m = (float)(*(const device half *)(blk + 2u));
    const uint j = gid % 32u;
    const uint nib = j < 16u ? (blk[4u + j] & 0x0fu) : (blk[4u + j - 16u] >> 4);
    out[gid] = (float)nib * d + m;
}

kernel void ch_q4_1_gemm(device const uchar *w [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device float *y [[buffer(2)]],
                         constant uint &rows [[buffer(3)]],
                         constant uint &cols [[buffer(4)]],
                         constant uint &batch [[buffer(5)]],
                         constant uint &x_off [[buffer(6)]],
                         constant uint &y_off [[buffer(7)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint lane [[thread_position_in_threadgroup]]) {
    if (row >= rows || batch == 0u || batch > 32u) {
        return;
    }
    const uint blocks = cols / 32u;
    const device uchar *wrow = w + (size_t)row * (size_t)blocks * 20u;
    const uint sub = lane % 4u;
    const uint lane_blk = lane / 4u;
    const uint byte0 = (sub % 2u) * 8u;
    const bool hi = sub >= 2u;
    float acc[32];
    for (uint b = 0; b < batch; ++b) {
        acc[b] = 0.0f;
    }
    for (uint blk = lane_blk; blk < blocks; blk += 8u) {
        const device uchar *block = wrow + (size_t)blk * 20u;
        const float d = (float)(*(const device half *)block);
        const float m = (float)(*(const device half *)(block + 2u));
        const device uchar *qs = block + 4u + byte0;
        float4 q0, q1;
        if (!hi) {
            q0 = float4((float)(qs[0] & 0x0fu) * d + m, (float)(qs[1] & 0x0fu) * d + m,
                        (float)(qs[2] & 0x0fu) * d + m, (float)(qs[3] & 0x0fu) * d + m);
            q1 = float4((float)(qs[4] & 0x0fu) * d + m, (float)(qs[5] & 0x0fu) * d + m,
                        (float)(qs[6] & 0x0fu) * d + m, (float)(qs[7] & 0x0fu) * d + m);
        } else {
            q0 = float4((float)(qs[0] >> 4) * d + m, (float)(qs[1] >> 4) * d + m,
                        (float)(qs[2] >> 4) * d + m, (float)(qs[3] >> 4) * d + m);
            q1 = float4((float)(qs[4] >> 4) * d + m, (float)(qs[5] >> 4) * d + m,
                        (float)(qs[6] >> 4) * d + m, (float)(qs[7] >> 4) * d + m);
        }
        const uint col = blk * 32u + sub * 8u;
        for (uint b = 0; b < batch; ++b) {
            const device packed_float4 *xv =
                (const device packed_float4 *)(x + x_off + b * cols + col);
            const float4 x0 = float4(xv[0]);
            const float4 x1 = float4(xv[1]);
            acc[b] += dot(q0, x0) + dot(q1, x1);
        }
    }
    for (uint b = 0; b < batch; ++b) {
        const float total = simd_sum(acc[b]);
        if (lane == 0u) {
            y[y_off + b * rows + row] = total;
        }
    }
}

/* Batched Q4_1 GEMM on the SIMD-group matrix units (f32 operands). */
kernel void ch_q4_1_gemm_mma(device const uchar *w [[buffer(0)]],
                              device const float *x [[buffer(1)]],
                              device float *y [[buffer(2)]],
                              constant uint &rows [[buffer(3)]],
                              constant uint &cols [[buffer(4)]],
                              constant uint &batch [[buffer(5)]],
                              constant uint &x_off [[buffer(6)]],
                              constant uint &y_off [[buffer(7)]],
                              uint tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[CH_MMA_ROWS * 32u];
    threadgroup float sb[CH_MMA_TOK * 32u];

    const uint token_tiles = (batch + CH_MMA_TOK - 1u) / CH_MMA_TOK;
    const uint tok0 = (tg % token_tiles) * CH_MMA_TOK;
    if (tok0 >= batch) {
        return;
    }
    const uint tile_batch = min(batch - tok0, CH_MMA_TOK);
    const uint row0 = (tg / token_tiles) * CH_MMA_ROWS;
    const uint blocks = cols / 32u;
    const size_t row_bytes = (size_t)blocks * 20u;

    const uint w_r = tid / 2u;
    const uint w_k = (tid % 2u) * 16u;
    const device uchar *wrow = w + (size_t)min(row0 + w_r, rows - 1u) * row_bytes;
    const uint x_t = tid / 4u;
    const uint x_k = (tid % 4u) * 8u;
    const device float *xrow = x + x_off + (size_t)(tok0 + min(x_t, tile_batch - 1u)) * cols + x_k;

    threadgroup float *sa_dst = sa + 64u * ((w_k / 8u) * 8u + w_r / 8u) + (w_r % 8u);
    threadgroup float *sb_dst = sb + 64u * ((x_k / 8u) * 4u + x_t / 8u) + 8u * (x_t % 8u);

    const uint sg_r = (sg % 2u) * 4u;
    const uint sg_t = (sg / 2u) * 2u;
    simdgroup_float8x8 mc[2][4];
    for (uint j = 0; j < 2u; ++j) {
        for (uint i = 0; i < 4u; ++i) {
            mc[j][i] = simdgroup_float8x8(0.0f);
        }
    }

    const device uchar *block = wrow;
    const device float *xblk = xrow;
    const threadgroup float *lsa = sa + 64u * sg_r;
    const threadgroup float *lsb = sb + 64u * sg_t;
    for (uint blk = 0; blk < blocks; ++blk, block += 20, xblk += 32) {
        float wv[16];
        const float d = (float)(*(const device half *)block);
        const float m = (float)(*(const device half *)(block + 2u));
        const device uchar *qs = block + 4u;
        CH_UNROLL for (uint i = 0; i < 16u; ++i) {
            const uint nib = w_k == 0u ? (qs[i] & 0x0fu) : (qs[i] >> 4);
            wv[i] = (float)nib * d + m;
        }
        const device packed_float4 *xp = (const device packed_float4 *)xblk;
        const float4 xv0 = float4(xp[0]);
        const float4 xv1 = float4(xp[1]);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        CH_UNROLL for (ushort i = 0; i < 16u; ++i) {
            sa_dst[64 * 8 * (i / 8) + 8 * (i % 8)] = wv[i];
        }
        *(threadgroup float4 *)(sb_dst) = xv0;
        *(threadgroup float4 *)(sb_dst + 4u) = xv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        CH_UNROLL for (ushort kb = 0; kb < 4u; ++kb) {
            simdgroup_float8x8 ma[4];
            simdgroup_float8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort i = 0; i < 4u; ++i) {
                simdgroup_load(ma[i], lsa + 64 * (8 * kb + i), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2u; ++j) {
                simdgroup_load(mb[j], lsb + 64 * (4 * kb + j), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2u; ++j) {
                CH_UNROLL for (ushort i = 0; i < 4u; ++i) {
                    simdgroup_multiply_accumulate(mc[j][i], mb[j], ma[i], mc[j][i]);
                }
            }
        }
    }

    if (row0 + CH_MMA_ROWS <= rows && tile_batch == CH_MMA_TOK) {
        device float *c = y + y_off + (size_t)(tok0 + sg_t * 8u) * rows + row0 + sg_r * 8u;
        for (uint j = 0; j < 2u; ++j) {
            for (uint i = 0; i < 4u; ++i) {
                simdgroup_store(mc[j][i], c + (size_t)j * 8u * rows + i * 8u, rows);
            }
        }
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *ot = sa;
    for (uint j = 0; j < 2u; ++j) {
        for (uint i = 0; i < 4u; ++i) {
            simdgroup_store(mc[j][i], ot + (sg_t + j) * 8u * CH_MMA_ROWS + (sg_r + i) * 8u,
                            CH_MMA_ROWS);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < CH_MMA_ROWS * CH_MMA_TOK; idx += 128u) {
        const uint r = idx % CH_MMA_ROWS;
        const uint b = idx / CH_MMA_ROWS;
        if (row0 + r < rows && b < tile_batch) {
            y[y_off + (size_t)(tok0 + b) * rows + row0 + r] = ot[b * CH_MMA_ROWS + r];
        }
    }
}

/* Batched Q4_1 GEMM with f16 operand tiles (metal-f16-act). */
kernel void ch_q4_1_gemm_mma_f16op(device const uchar *w [[buffer(0)]],
                                   device const float *x [[buffer(1)]],
                                   device float *y [[buffer(2)]],
                                   constant uint &rows [[buffer(3)]],
                                   constant uint &cols [[buffer(4)]],
                                   constant uint &batch [[buffer(5)]],
                                   constant uint &x_off [[buffer(6)]],
                                   constant uint &y_off [[buffer(7)]],
                                   uint tg [[threadgroup_position_in_grid]],
                                   uint tid [[thread_position_in_threadgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[CH_MMA_ROWS * 32u];
    threadgroup half sb[CH_MMA_TOK * 32u];
    threadgroup float edge_frag[4][64];

    const uint token_tiles = (batch + CH_MMA_TOK - 1u) / CH_MMA_TOK;
    const uint tok0 = (tg % token_tiles) * CH_MMA_TOK;
    if (tok0 >= batch) {
        return;
    }
    const uint tile_batch = min(batch - tok0, CH_MMA_TOK);
    const uint row0 = (tg / token_tiles) * CH_MMA_ROWS;
    const uint blocks = cols / 32u;
    const size_t row_bytes = (size_t)blocks * 20u;

    const uint w_r = tid / 2u;
    const uint w_k = (tid % 2u) * 16u;
    const device uchar *wrow = w + (size_t)min(row0 + w_r, rows - 1u) * row_bytes;
    const uint x_t = tid / 4u;
    const uint x_k = (tid % 4u) * 8u;
    const device float *xrow = x + x_off + (size_t)(tok0 + min(x_t, tile_batch - 1u)) * cols + x_k;

    threadgroup half *sa_dst = sa + 64u * ((w_k / 8u) * 8u + w_r / 8u) + (w_r % 8u);
    threadgroup half *sb_dst = sb + 64u * ((x_k / 8u) * 4u + x_t / 8u) + 8u * (x_t % 8u);

    const uint sg_r = (sg % 2u) * 4u;
    const uint sg_t = (sg / 2u) * 2u;
    simdgroup_float8x8 mc[2][4];
    for (uint j = 0; j < 2u; ++j) {
        for (uint i = 0; i < 4u; ++i) {
            mc[j][i] = simdgroup_float8x8(0.0f);
        }
    }

    const device uchar *block = wrow;
    const device float *xblk = xrow;
    const threadgroup half *lsa = sa + 64u * sg_r;
    const threadgroup half *lsb = sb + 64u * sg_t;
    for (uint blk = 0; blk < blocks; ++blk, block += 20, xblk += 32) {
        float wv[16];
        const float d = (float)(*(const device half *)block);
        const float m = (float)(*(const device half *)(block + 2u));
        const device uchar *qs = block + 4u;
        CH_UNROLL for (uint i = 0; i < 16u; ++i) {
            const uint nib = w_k == 0u ? (qs[i] & 0x0fu) : (qs[i] >> 4);
            wv[i] = (float)nib * d + m;
        }
        const device packed_float4 *xp = (const device packed_float4 *)xblk;
        const float4 xv0 = float4(xp[0]);
        const float4 xv1 = float4(xp[1]);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        CH_UNROLL for (ushort i = 0; i < 16u; ++i) {
            sa_dst[64 * 8 * (i / 8) + 8 * (i % 8)] = (half)wv[i];
        }
        *(threadgroup half4 *)(sb_dst) = half4(xv0);
        *(threadgroup half4 *)(sb_dst + 4u) = half4(xv1);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        CH_UNROLL for (ushort kb = 0; kb < 4u; ++kb) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort i = 0; i < 4u; ++i) {
                simdgroup_load(ma[i], lsa + 64 * (8 * kb + i), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2u; ++j) {
                simdgroup_load(mb[j], lsb + 64 * (4 * kb + j), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2u; ++j) {
                CH_UNROLL for (ushort i = 0; i < 4u; ++i) {
                    simdgroup_multiply_accumulate(mc[j][i], mb[j], ma[i], mc[j][i]);
                }
            }
        }
    }

    if (row0 + CH_MMA_ROWS <= rows && tile_batch == CH_MMA_TOK) {
        device float *c = y + y_off + (size_t)(tok0 + sg_t * 8u) * rows + row0 + sg_r * 8u;
        for (uint j = 0; j < 2u; ++j) {
            for (uint i = 0; i < 4u; ++i) {
                simdgroup_store(mc[j][i], c + (size_t)j * 8u * rows + i * 8u, rows);
            }
        }
        return;
    }
    const uint tok_end = tok0 + tile_batch;
    {
        threadgroup float * const edge_frag_mine = edge_frag[sg];
        const uint rb_base = row0 + sg_r * 8u;
        const uint tb_base = tok0 + sg_t * 8u;
        for (uint j = 0; j < 2u; ++j) {
            for (uint i = 0; i < 4u; ++i) {
                const uint rb = rb_base + i * 8u;
                const uint tb = tb_base + j * 8u;
                if (rb >= rows || tb >= tok_end) {
                    continue;
                }
                if (rb + 8u <= rows && tb + 8u <= tok_end) {
                    simdgroup_store(mc[j][i], y + y_off + (size_t)tb * rows + rb, rows);
                } else {
                    simdgroup_store(mc[j][i], edge_frag_mine, 8);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint lane = tid % 32u; lane < 64u; lane += 32u) {
                        const uint fr = lane / 8u;
                        const uint ft = lane % 8u;
                        if (tb + fr < tok_end && rb + ft < rows) {
                            y[y_off + (size_t)(tb + fr) * rows + rb + ft] =
                                edge_frag_mine[fr * 8u + ft];
                        }
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
    }
}


/* ---------------- Q4_K kernels (2026-11-22) ----------------
 * Pinned to llama.cpp b11695 (ggml-common.h block_q4_K, ggml-quants.c
 * dequantize_row_q4_K / get_scale_min_k4): 256-value super-blocks,
 * bytes [f16 d][f16 dmin][u8 scales; 12][u8 qs; 128]. Eight 32-value
 * slices; slice k reads bytes [32*(k/2), +32) of qs (low nibbles for
 * even k, high for odd) with its own (scale, min) pair decoded from
 * the 12-byte packed form: for k < 4, sc = scales[k] & 63 and
 * m = scales[k+4] & 63; for k >= 4, sc = (scales[k+4] & 0xF) |
 * ((scales[k-4] >> 6) << 4) and m = (scales[k+4] >> 4) |
 * ((scales[k] >> 6) << 4). Value = d*sc*nib - dmin*m. */

kernel void ch_q4_k_gemv(device const uchar *w [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device float *y [[buffer(2)]],
                         constant uint &rows [[buffer(3)]],
                         constant uint &cols [[buffer(4)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    /* Verbatim port of llama.cpp b11695 kernel_mul_mv_q4_K_f32_impl
     * (ggml/src/ggml-metal/kernels/mul_mv.metal:1511), pinned by revision.
     * Lane decomposition: ix = tiisg/8 (super-block stride 4), it =
     * tiisg%8 -> iq = it/4, ir = it%4; each lane holds 32 x values in
     * 16+16 SCALAR registers (yl/yh), re-read per block iteration; the
     * W side loads qs as uint16 pairs so one AND serves two values (the
     * nibble stays in its bit position; 1/256 and 1/16 exact
     * power-of-two corrections recover it), and the affine min term
     * subtracts once via the per-block sumy. Scales decode through the
     * reference's u16 kmask construction, byte-for-byte their
     * expressions. N_R0_Q4_K = 2 rows per simdgroup (impl.h), NSG = 2
     * (64-thread groups). ABI-level adaptations only: our buffers and
     * row-tail guards on the loads (the reference guards only the
     * writes; our buffers are bounds-checked, so tail rows break out
     * instead of reading past the end). */
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const int nb = (int)(cols / 256u);
    const size_t row_bytes = (size_t)nb * 144u;

    const short ix = (short)(lane / 8u);  /* 0..3 */
    const short it = (short)(lane % 8u);  /* 0..7 */
    const short iq = it / 4;              /* 0 or 1 */
    const short ir = it % 4;              /* 0..3 */

    const int first_row = (int)((tg * 2u + sg) * 2u);
    if (first_row >= (int)rows) {
        return;
    }

    device const uchar *base = w + (size_t)first_row * row_bytes;

    float yl[16];
    float yh[16];

    float sumf[2] = {0.f, 0.f};

    device const float * y4 = x + (size_t)ix * 256u + 64u * (uint)iq + 8u * (uint)ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i + 0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i + 32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        device const uchar * blk = base + (size_t)ib * 144u;
        device const uint16_t * sc = (device const uint16_t *)(blk + 4u) + iq;
        device const uint16_t * q1 = (device const uint16_t *)(blk + 16u) + 16 * iq + 4 * ir;
        device const half * dh = (device const half *)blk;

        for (short row = 0; row < 2; ++row) {
            if (first_row + row >= (int)rows) {
                break;
            }

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            CH_UNROLL for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                  (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += row_bytes/2;
            sc += row_bytes/2;
            dh += row_bytes/2;
        }

        y4 += 4 * 256;
    }

    for (int row = 0; row < 2 && first_row + row < (int)rows; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (lane == 0u) {
            y[first_row + row] = sum_all;
        }
    }
}

/* Dequantize one Q4_K matrix row (or embedding row) to f32: one thread
 * per output element. */
kernel void ch_q4_k_row_f32(device const uchar *w [[buffer(0)]],
                            device float *out [[buffer(1)]],
                            constant uint &row [[buffer(2)]],
                            constant uint &cols [[buffer(3)]],
                            constant uint &out_off [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) {
        return;
    }
    const uint blocks = cols / 256u;
    const device uchar *blk =
        w + (size_t)row * (size_t)blocks * 144u + (size_t)(gid / 256u) * 144u;
    const uint k = (gid % 256u) / 32u;
    const float d = (float)(*(const device half *)blk);
    const float dmin = (float)(*(const device half *)(blk + 2u));
    const device uchar *sc = blk + 4u;
    float a, mn;
    if (k < 4u) {
        a = (float)(sc[k] & 63u);
        mn = (float)(sc[k + 4u] & 63u);
    } else {
        a = (float)((sc[k + 4u] & 0xFu) | ((sc[k - 4u] >> 6) << 4));
        mn = (float)((sc[k + 4u] >> 4) | ((sc[k] >> 6) << 4));
    }
    const uint j = gid % 32u;
    const device uchar *qs = blk + 16u + 32u * (k / 2u) + j;
    const uint nib = k % 2u == 0u ? (qs[0] & 0x0fu) : (qs[0] >> 4);
    out[out_off + gid] = d * a * (float)nib - dmin * mn;
}

kernel void ch_q4_k_row_indirect_f32(device const uchar *w [[buffer(0)]],
                                     device float *out [[buffer(1)]],
                                     device const uint *ids [[buffer(2)]],
                                     constant uint &slot [[buffer(3)]],
                                     constant uint &rows [[buffer(4)]],
                                     constant uint &cols [[buffer(5)]],
                                     uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) {
        return;
    }
    const uint row = ids[slot];
    if (row >= rows) {
        out[gid] = 0.0f;
        return;
    }
    const uint blocks = cols / 256u;
    const device uchar *blk =
        w + ((size_t)row * (size_t)blocks + gid / 256u) * 144u;
    const uint k = (gid % 256u) / 32u;
    const float d = (float)(*(const device half *)blk);
    const float dmin = (float)(*(const device half *)(blk + 2u));
    const device uchar *sc = blk + 4u;
    float a, mn;
    if (k < 4u) {
        a = (float)(sc[k] & 63u);
        mn = (float)(sc[k + 4u] & 63u);
    } else {
        a = (float)((sc[k + 4u] & 0xFu) | ((sc[k - 4u] >> 6) << 4));
        mn = (float)((sc[k + 4u] >> 4) | ((sc[k] >> 6) << 4));
    }
    const uint j = gid % 32u;
    const device uchar *qs = blk + 16u + 32u * (k / 2u) + j;
    const uint nib = k % 2u == 0u ? (qs[0] & 0x0fu) : (qs[0] >> 4);
    out[gid] = d * a * (float)nib - dmin * mn;
}

/* Mid-batch Q4_K GEMV projection: one threadgroup per output row. */
kernel void ch_q4_k_gemm(device const uchar *w [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device float *y [[buffer(2)]],
                         constant uint &rows [[buffer(3)]],
                         constant uint &cols [[buffer(4)]],
                         constant uint &batch [[buffer(5)]],
                         constant uint &x_off [[buffer(6)]],
                         constant uint &y_off [[buffer(7)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint lane [[thread_position_in_threadgroup]]) {
    if (row >= rows || batch == 0u || batch > 32u) {
        return;
    }
    const uint blocks = cols / 256u;
    const device uchar *wrow = w + (size_t)row * (size_t)blocks * 144u;
    const uint k = lane % 8u;
    const uint lane_sb = lane / 8u;
    const bool hi = k % 2u == 1u;
    float acc[32];
    for (uint b = 0; b < batch; ++b) {
        acc[b] = 0.0f;
    }
    for (uint blk = lane_sb; blk < blocks; blk += 4u) {
        const device uchar *block = wrow + (size_t)blk * 144u;
        const float d = (float)(*(const device half *)block);
        const float dmin = (float)(*(const device half *)(block + 2u));
        const device uchar *sc = block + 4u;
        float a, mn;
        if (k < 4u) {
            a = (float)(sc[k] & 63u);
            mn = (float)(sc[k + 4u] & 63u);
        } else {
            a = (float)((sc[k + 4u] & 0xFu) | ((sc[k - 4u] >> 6) << 4));
            mn = (float)((sc[k + 4u] >> 4) | ((sc[k] >> 6) << 4));
        }
        const float ds = d * a;
        const float dm = dmin * mn;
        const device uchar *qs = block + 16u + 32u * (k / 2u);
        const device packed_uchar4 *qv = (const device packed_uchar4 *)qs;
        float4 q0, q1, q2, q3, q4, q5, q6, q7;
        if (!hi) {
            q0 = float4(uchar4(qv[0]) & (uchar4)0x0fu) * ds - dm;
            q1 = float4(uchar4(qv[1]) & (uchar4)0x0fu) * ds - dm;
            q2 = float4(uchar4(qv[2]) & (uchar4)0x0fu) * ds - dm;
            q3 = float4(uchar4(qv[3]) & (uchar4)0x0fu) * ds - dm;
            q4 = float4(uchar4(qv[4]) & (uchar4)0x0fu) * ds - dm;
            q5 = float4(uchar4(qv[5]) & (uchar4)0x0fu) * ds - dm;
            q6 = float4(uchar4(qv[6]) & (uchar4)0x0fu) * ds - dm;
            q7 = float4(uchar4(qv[7]) & (uchar4)0x0fu) * ds - dm;
        } else {
            q0 = float4(uchar4(qv[0]) >> 4) * ds - dm;
            q1 = float4(uchar4(qv[1]) >> 4) * ds - dm;
            q2 = float4(uchar4(qv[2]) >> 4) * ds - dm;
            q3 = float4(uchar4(qv[3]) >> 4) * ds - dm;
            q4 = float4(uchar4(qv[4]) >> 4) * ds - dm;
            q5 = float4(uchar4(qv[5]) >> 4) * ds - dm;
            q6 = float4(uchar4(qv[6]) >> 4) * ds - dm;
            q7 = float4(uchar4(qv[7]) >> 4) * ds - dm;
        }
        const uint col = blk * 256u + k * 32u;
        for (uint b = 0; b < batch; ++b) {
            const device packed_float4 *xv =
                (const device packed_float4 *)(x + x_off + b * cols + col);
            acc[b] += dot(q0, float4(xv[0])) + dot(q1, float4(xv[1]))
                    + dot(q2, float4(xv[2])) + dot(q3, float4(xv[3]))
                    + dot(q4, float4(xv[4])) + dot(q5, float4(xv[5]))
                    + dot(q6, float4(xv[6])) + dot(q7, float4(xv[7]));
        }
    }
    for (uint b = 0; b < batch; ++b) {
        const float total = simd_sum(acc[b]);
        if (lane == 0u) {
            y[y_off + b * rows + row] = total;
        }
    }
}

/* Batched Q4_K GEMM on the SIMD-group matrix units (f32 operands). The
 * 32-value K-step tiles the super-block's eight slices exactly
 * (slice k = blk & 7), so each thread stages one (scale, min) pair. */
kernel void ch_q4_k_gemm_mma(device const uchar *w [[buffer(0)]],
                             device const float *x [[buffer(1)]],
                             device float *y [[buffer(2)]],
                             constant uint &rows [[buffer(3)]],
                             constant uint &cols [[buffer(4)]],
                             constant uint &batch [[buffer(5)]],
                             constant uint &x_off [[buffer(6)]],
                             constant uint &y_off [[buffer(7)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[CH_MMA_ROWS * 32u];
    threadgroup float sb[CH_MMA_TOK * 32u];

    const uint token_tiles = (batch + CH_MMA_TOK - 1u) / CH_MMA_TOK;
    const uint tok0 = (tg % token_tiles) * CH_MMA_TOK;
    if (tok0 >= batch) {
        return;
    }
    const uint tile_batch = min(batch - tok0, CH_MMA_TOK);
    const uint row0 = (tg / token_tiles) * CH_MMA_ROWS;
    const uint blocks = cols / 32u;
    const size_t row_bytes = (size_t)(cols / 256u) * 144u;

    const uint w_r = tid / 2u;
    const uint w_k = (tid % 2u) * 16u;
    const device uchar *wrow = w + (size_t)min(row0 + w_r, rows - 1u) * row_bytes;
    const uint x_t = tid / 4u;
    const uint x_k = (tid % 4u) * 8u;
    const device float *xrow = x + x_off + (size_t)(tok0 + min(x_t, tile_batch - 1u)) * cols + x_k;

    threadgroup float *sa_dst = sa + 64u * ((w_k / 8u) * 8u + w_r / 8u) + (w_r % 8u);
    threadgroup float *sb_dst = sb + 64u * ((x_k / 8u) * 4u + x_t / 8u) + 8u * (x_t % 8u);

    const uint sg_r = (sg % 2u) * 4u;
    const uint sg_t = (sg / 2u) * 2u;
    simdgroup_float8x8 mc[2][4];
    for (uint j = 0; j < 2u; ++j) {
        for (uint i = 0; i < 4u; ++i) {
            mc[j][i] = simdgroup_float8x8(0.0f);
        }
    }

    const device float *xblk = xrow;
    const threadgroup float *lsa = sa + 64u * sg_r;
    const threadgroup float *lsb = sb + 64u * sg_t;
    for (uint blk = 0; blk < blocks; ++blk, xblk += 32) {
        const uint k = blk & 7u;
        const device uchar *block = wrow + (size_t)(blk >> 3) * 144u;
        const float d = (float)(*(const device half *)block);
        const float dmin = (float)(*(const device half *)(block + 2u));
        const device uchar *sc = block + 4u;
        float a, mn;
        if (k < 4u) {
            a = (float)(sc[k] & 63u);
            mn = (float)(sc[k + 4u] & 63u);
        } else {
            a = (float)((sc[k + 4u] & 0xFu) | ((sc[k - 4u] >> 6) << 4));
            mn = (float)((sc[k + 4u] >> 4) | ((sc[k] >> 6) << 4));
        }
        const float ds = d * a;
        const float dm = dmin * mn;
        const device uchar *qs = block + 16u + 32u * (k / 2u) + w_k;
        float wv[16];
        if (k % 2u == 0u) {
            CH_UNROLL for (uint i = 0; i < 16u; ++i) {
                wv[i] = (float)(qs[i] & 0x0fu) * ds - dm;
            }
        } else {
            CH_UNROLL for (uint i = 0; i < 16u; ++i) {
                wv[i] = (float)(qs[i] >> 4) * ds - dm;
            }
        }
        const device packed_float4 *xp = (const device packed_float4 *)xblk;
        const float4 xv0 = float4(xp[0]);
        const float4 xv1 = float4(xp[1]);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        CH_UNROLL for (ushort i = 0; i < 16u; ++i) {
            sa_dst[64 * 8 * (i / 8) + 8 * (i % 8)] = wv[i];
        }
        *(threadgroup float4 *)(sb_dst) = xv0;
        *(threadgroup float4 *)(sb_dst + 4u) = xv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        CH_UNROLL for (ushort kb = 0; kb < 4u; ++kb) {
            simdgroup_float8x8 ma[4];
            simdgroup_float8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort i = 0; i < 4u; ++i) {
                simdgroup_load(ma[i], lsa + 64 * (8 * kb + i), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2u; ++j) {
                simdgroup_load(mb[j], lsb + 64 * (4 * kb + j), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2u; ++j) {
                CH_UNROLL for (ushort i = 0; i < 4u; ++i) {
                    simdgroup_multiply_accumulate(mc[j][i], mb[j], ma[i], mc[j][i]);
                }
            }
        }
    }

    if (row0 + CH_MMA_ROWS <= rows && tile_batch == CH_MMA_TOK) {
        device float *c = y + y_off + (size_t)(tok0 + sg_t * 8u) * rows + row0 + sg_r * 8u;
        for (uint j = 0; j < 2u; ++j) {
            for (uint i = 0; i < 4u; ++i) {
                simdgroup_store(mc[j][i], c + (size_t)j * 8u * rows + i * 8u, rows);
            }
        }
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *ot = sa;
    for (uint j = 0; j < 2u; ++j) {
        for (uint i = 0; i < 4u; ++i) {
            simdgroup_store(mc[j][i], ot + (sg_t + j) * 8u * CH_MMA_ROWS + (sg_r + i) * 8u,
                            CH_MMA_ROWS);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < CH_MMA_ROWS * CH_MMA_TOK; idx += 128u) {
        const uint r = idx % CH_MMA_ROWS;
        const uint b = idx / CH_MMA_ROWS;
        if (row0 + r < rows && b < tile_batch) {
            y[y_off + (size_t)(tok0 + b) * rows + row0 + r] = ot[b * CH_MMA_ROWS + r];
        }
    }
}

/* Batched Q4_K GEMM with f16 operand tiles (metal-f16-act). */
kernel void ch_q4_k_gemm_mma_f16op(device const uchar *w [[buffer(0)]],
                                   device const float *x [[buffer(1)]],
                                   device float *y [[buffer(2)]],
                                   constant uint &rows [[buffer(3)]],
                                   constant uint &cols [[buffer(4)]],
                                   constant uint &batch [[buffer(5)]],
                                   constant uint &x_off [[buffer(6)]],
                                   constant uint &y_off [[buffer(7)]],
                                   uint tg [[threadgroup_position_in_grid]],
                                   uint tid [[thread_position_in_threadgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[CH_MMA_ROWS * 32u];
    threadgroup half sb[CH_MMA_TOK * 32u];
    threadgroup float edge_frag[4][64];

    const uint token_tiles = (batch + CH_MMA_TOK - 1u) / CH_MMA_TOK;
    const uint tok0 = (tg % token_tiles) * CH_MMA_TOK;
    if (tok0 >= batch) {
        return;
    }
    const uint tile_batch = min(batch - tok0, CH_MMA_TOK);
    const uint row0 = (tg / token_tiles) * CH_MMA_ROWS;
    const uint blocks = cols / 32u;
    const size_t row_bytes = (size_t)(cols / 256u) * 144u;

    const uint w_r = tid / 2u;
    const uint w_k = (tid % 2u) * 16u;
    const device uchar *wrow = w + (size_t)min(row0 + w_r, rows - 1u) * row_bytes;
    const uint x_t = tid / 4u;
    const uint x_k = (tid % 4u) * 8u;
    const device float *xrow = x + x_off + (size_t)(tok0 + min(x_t, tile_batch - 1u)) * cols + x_k;

    threadgroup half *sa_dst = sa + 64u * ((w_k / 8u) * 8u + w_r / 8u) + (w_r % 8u);
    threadgroup half *sb_dst = sb + 64u * ((x_k / 8u) * 4u + x_t / 8u) + 8u * (x_t % 8u);

    const uint sg_r = (sg % 2u) * 4u;
    const uint sg_t = (sg / 2u) * 2u;
    simdgroup_float8x8 mc[2][4];
    for (uint j = 0; j < 2u; ++j) {
        for (uint i = 0; i < 4u; ++i) {
            mc[j][i] = simdgroup_float8x8(0.0f);
        }
    }

    const device float *xblk = xrow;
    const threadgroup half *lsa = sa + 64u * sg_r;
    const threadgroup half *lsb = sb + 64u * sg_t;
    for (uint blk = 0; blk < blocks; ++blk, xblk += 32) {
        const uint k = blk & 7u;
        const device uchar *block = wrow + (size_t)(blk >> 3) * 144u;
        const float d = (float)(*(const device half *)block);
        const float dmin = (float)(*(const device half *)(block + 2u));
        const device uchar *sc = block + 4u;
        float a, mn;
        if (k < 4u) {
            a = (float)(sc[k] & 63u);
            mn = (float)(sc[k + 4u] & 63u);
        } else {
            a = (float)((sc[k + 4u] & 0xFu) | ((sc[k - 4u] >> 6) << 4));
            mn = (float)((sc[k + 4u] >> 4) | ((sc[k] >> 6) << 4));
        }
        const float ds = d * a;
        const float dm = dmin * mn;
        const device uchar *qs = block + 16u + 32u * (k / 2u) + w_k;
        float wv[16];
        if (k % 2u == 0u) {
            CH_UNROLL for (uint i = 0; i < 16u; ++i) {
                wv[i] = (float)(qs[i] & 0x0fu) * ds - dm;
            }
        } else {
            CH_UNROLL for (uint i = 0; i < 16u; ++i) {
                wv[i] = (float)(qs[i] >> 4) * ds - dm;
            }
        }
        const device packed_float4 *xp = (const device packed_float4 *)xblk;
        const float4 xv0 = float4(xp[0]);
        const float4 xv1 = float4(xp[1]);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        CH_UNROLL for (ushort i = 0; i < 16u; ++i) {
            sa_dst[64 * 8 * (i / 8) + 8 * (i % 8)] = (half)wv[i];
        }
        *(threadgroup half4 *)(sb_dst) = half4(xv0);
        *(threadgroup half4 *)(sb_dst + 4u) = half4(xv1);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        CH_UNROLL for (ushort kb = 0; kb < 4u; ++kb) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort i = 0; i < 4u; ++i) {
                simdgroup_load(ma[i], lsa + 64 * (8 * kb + i), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2u; ++j) {
                simdgroup_load(mb[j], lsb + 64 * (4 * kb + j), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2u; ++j) {
                CH_UNROLL for (ushort i = 0; i < 4u; ++i) {
                    simdgroup_multiply_accumulate(mc[j][i], mb[j], ma[i], mc[j][i]);
                }
            }
        }
    }

    if (row0 + CH_MMA_ROWS <= rows && tile_batch == CH_MMA_TOK) {
        device float *c = y + y_off + (size_t)(tok0 + sg_t * 8u) * rows + row0 + sg_r * 8u;
        for (uint j = 0; j < 2u; ++j) {
            for (uint i = 0; i < 4u; ++i) {
                simdgroup_store(mc[j][i], c + (size_t)j * 8u * rows + i * 8u, rows);
            }
        }
        return;
    }
    const uint tok_end = tok0 + tile_batch;
    {
        threadgroup float * const edge_frag_mine = edge_frag[sg];
        const uint rb_base = row0 + sg_r * 8u;
        const uint tb_base = tok0 + sg_t * 8u;
        for (uint j = 0; j < 2u; ++j) {
            for (uint i = 0; i < 4u; ++i) {
                const uint rb = rb_base + i * 8u;
                const uint tb = tb_base + j * 8u;
                if (rb >= rows || tb >= tok_end) {
                    continue;
                }
                if (rb + 8u <= rows && tb + 8u <= tok_end) {
                    simdgroup_store(mc[j][i], y + y_off + (size_t)tb * rows + rb, rows);
                } else {
                    simdgroup_store(mc[j][i], edge_frag_mine, 8);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint lane = tid % 32u; lane < 64u; lane += 32u) {
                        const uint fr = lane / 8u;
                        const uint ft = lane % 8u;
                        if (tb + fr < tok_end && rb + ft < rows) {
                            y[y_off + (size_t)(tb + fr) * rows + rb + ft] =
                                edge_frag_mine[fr * 8u + ft];
                        }
                    }
                }
            }
        }
    }
}


/* ---------------- Q6_K kernels (2026-11-22) ----------------
 * Pinned to llama.cpp b11695 (ggml-common.h block_q6_K, ggml-quants.c
 * dequantize_row_q6_K): 256-value super-blocks, bytes [u8 ql; 128][u8 qh;
 * 64][i8 scales; 16][f16 d]. Sixteen 16-value slices; slice k (value
 * range [16k, 16k+16)) reads ql bytes [64*(k/8) + 32*((k%8)/2 % 2) +
 * 16*(k%2), +16) — low nibble when (k%8)/2 < 2, high when >= 2 — and qh
 * bytes [32*(k/8) + 16*(k%2), +16) at bit shift 2*((k%8)/2), with the
 * int8 scale scales[k]. Value = d * scales[k] * (q - 32), where
 * q = nibble | (((qh >> shift) & 3) << 4). */

kernel void ch_q6_k_gemv(device const uchar *w [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device float *y [[buffer(2)]],
                         constant uint &rows [[buffer(3)]],
                         constant uint &cols [[buffer(4)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    /* Verbatim port of llama.cpp b11695 kernel_mul_mv_q6_K_f32_impl
     * (ggml/src/ggml-metal/kernels/mul_mv.metal:1763), pinned by
     * revision. Lane decomposition: tid = tiisg/2, ix = tiisg%2
     * (super-block stride 2); ip = tid/8, il = tid%8, l0 = 4*il,
     * is = 8*ip + l0/16; each lane holds 16 x values in SCALAR yl
     * registers, re-read per block iteration; the W side extracts the
     * 6-bit codes inline in the multiply with the reference's kmask
     * bit patterns and applies the int8 scale per 16-value run at the
     * end. N_R0_Q6_K = 2 (impl.h), NSG = 2 (64-thread groups).
     * ABI-level adaptations only: our buffers and row-tail guards on
     * the loads (the reference guards only the writes). */
    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = (int)(cols / 256u);
    const size_t row_bytes = (size_t)nb * 210u;

    const int first_row = (int)((tg * 2u + sg) * 2u);
    if (first_row >= (int)rows) {
        return;
    }

    device const uchar *base = w + (size_t)first_row * row_bytes;

    float sumf[2] = {0.f, 0.f};

    float yl[16];

    const short tid = (short)(lane / 2u);
    const short ix  = (short)(lane % 2u);
    const short ip  = tid/8;         /* 0 or 1 */
    const short il  = tid%8;
    const short l0  = 4*il;
    const short is  = 8*ip + l0/16;  /* 0..15 */

    const short y_offset   = 128*ip + l0;
    const short q_offset_l =  64*ip + l0;
    const short q_offset_h =  32*ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const uchar * blk = base + (size_t)i * 210u;
        device const uint8_t * q1 = blk + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = blk + 128u + q_offset_h;
        device const int8_t  * sc = (device const int8_t *)(blk + 192u) + is;
        device const half    * dh = (device const half *)(blk + 208u);

        device const float * yy = x + (size_t)i * 256u + (size_t)y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = yy[l +  0];
            yl[4*l + 1] = yy[l + 32];
            yl[4*l + 2] = yy[l + 64];
            yl[4*l + 3] = yy[l + 96];
        }

        for (short row = 0; row < 2; ++row) {
            if (first_row + row >= (int)rows) {
                break;
            }

            float4 sums = {0.f, 0.f, 0.f, 0.f};

            CH_UNROLL for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

            q1 += row_bytes;
            q2 += row_bytes;
            qh += row_bytes;
            sc += row_bytes;
            dh += row_bytes/2;
        }
    }

    for (int row = 0; row < 2 && first_row + row < (int)rows; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (lane == 0u) {
            y[first_row + row] = sum_all;
        }
    }
}

/* Dequantize one Q6_K matrix row (or embedding row) to f32: one thread
 * per output element. */
kernel void ch_q6_k_row_f32(device const uchar *w [[buffer(0)]],
                            device float *out [[buffer(1)]],
                            constant uint &row [[buffer(2)]],
                            constant uint &cols [[buffer(3)]],
                            constant uint &out_off [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) {
        return;
    }
    const uint blocks = cols / 256u;
    const device uchar *blk =
        w + (size_t)row * (size_t)blocks * 210u + (size_t)(gid / 256u) * 210u;
    const uint r = gid % 256u;
    const uint c = r / 128u;
    const uint jj = (r % 128u) / 32u;
    const uint l = r % 32u;
    const float d = (float)(*(const device half *)(blk + 208u));
    const float s = (float)((const device int8_t *)(blk + 192u))[r / 16u];
    const uint nib = jj < 2u ? (blk[64u * c + 32u * (jj % 2u) + l] & 0x0fu)
                             : (blk[64u * c + 32u * (jj % 2u) + l] >> 4);
    const uint qh = (blk[128u + 32u * c + l] >> (2u * jj)) & 3u;
    out[out_off + gid] = d * s * ((float)(nib | (qh << 4)) - 32.0f);
}

kernel void ch_q6_k_row_indirect_f32(device const uchar *w [[buffer(0)]],
                                     device float *out [[buffer(1)]],
                                     device const uint *ids [[buffer(2)]],
                                     constant uint &slot [[buffer(3)]],
                                     constant uint &rows [[buffer(4)]],
                                     constant uint &cols [[buffer(5)]],
                                     uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) {
        return;
    }
    const uint row = ids[slot];
    if (row >= rows) {
        out[gid] = 0.0f;
        return;
    }
    const uint blocks = cols / 256u;
    const device uchar *blk =
        w + ((size_t)row * (size_t)blocks + gid / 256u) * 210u;
    const uint r = gid % 256u;
    const uint c = r / 128u;
    const uint jj = (r % 128u) / 32u;
    const uint l = r % 32u;
    const float d = (float)(*(const device half *)(blk + 208u));
    const float s = (float)((const device int8_t *)(blk + 192u))[r / 16u];
    const uint nib = jj < 2u ? (blk[64u * c + 32u * (jj % 2u) + l] & 0x0fu)
                             : (blk[64u * c + 32u * (jj % 2u) + l] >> 4);
    const uint qh = (blk[128u + 32u * c + l] >> (2u * jj)) & 3u;
    out[gid] = d * s * ((float)(nib | (qh << 4)) - 32.0f);
}

/* Mid-batch Q6_K GEMV projection: one threadgroup per output row. */
kernel void ch_q6_k_gemm(device const uchar *w [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device float *y [[buffer(2)]],
                         constant uint &rows [[buffer(3)]],
                         constant uint &cols [[buffer(4)]],
                         constant uint &batch [[buffer(5)]],
                         constant uint &x_off [[buffer(6)]],
                         constant uint &y_off [[buffer(7)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint lane [[thread_position_in_threadgroup]]) {
    if (row >= rows || batch == 0u || batch > 32u) {
        return;
    }
    const uint blocks = cols / 256u;
    const device uchar *wrow = w + (size_t)row * (size_t)blocks * 210u;
    const uint k = lane % 16u;
    const uint lane_sb = lane / 16u;
    const uint c = k / 8u;
    const uint jj = (k % 8u) / 2u;
    const bool hi = jj >= 2u;
    const uint l0 = 16u * (k % 2u);
    const uint ql0 = 64u * c + 32u * (jj % 2u) + l0;
    const uint qh0 = 32u * c + l0;
    const uint shift = 2u * jj;
    float acc[32];
    for (uint b = 0; b < batch; ++b) {
        acc[b] = 0.0f;
    }
    for (uint blk = lane_sb; blk < blocks; blk += 2u) {
        const device uchar *block = wrow + (size_t)blk * 210u;
        const float d = (float)(*(const device half *)(block + 208u));
        const float s = (float)((const device int8_t *)(block + 192u))[k];
        const float ds = d * s;
        const device packed_uchar4 *qlv = (const device packed_uchar4 *)(block + ql0);
        const device packed_uchar4 *qhv = (const device packed_uchar4 *)(block + 128u + qh0);
        float4 q0, q1, q2, q3;
        if (!hi) {
            q0 = float4(uchar4(qlv[0]) & (uchar4)0x0fu);
            q1 = float4(uchar4(qlv[1]) & (uchar4)0x0fu);
            q2 = float4(uchar4(qlv[2]) & (uchar4)0x0fu);
            q3 = float4(uchar4(qlv[3]) & (uchar4)0x0fu);
        } else {
            q0 = float4(uchar4(qlv[0]) >> 4);
            q1 = float4(uchar4(qlv[1]) >> 4);
            q2 = float4(uchar4(qlv[2]) >> 4);
            q3 = float4(uchar4(qlv[3]) >> 4);
        }
        const uchar4 h0 = uchar4(qhv[0]);
        const uchar4 h1 = uchar4(qhv[1]);
        const uchar4 h2 = uchar4(qhv[2]);
        const uchar4 h3 = uchar4(qhv[3]);
        q0 += float4(h0 >> shift & (uchar4)3u) * 16.0f;
        q1 += float4(h1 >> shift & (uchar4)3u) * 16.0f;
        q2 += float4(h2 >> shift & (uchar4)3u) * 16.0f;
        q3 += float4(h3 >> shift & (uchar4)3u) * 16.0f;
        q0 -= 32.0f;
        q1 -= 32.0f;
        q2 -= 32.0f;
        q3 -= 32.0f;
        const uint col = blk * 256u + k * 16u;
        for (uint b = 0; b < batch; ++b) {
            const device packed_float4 *xv =
                (const device packed_float4 *)(x + x_off + b * cols + col);
            acc[b] += ds * (dot(q0, float4(xv[0])) + dot(q1, float4(xv[1]))
                          + dot(q2, float4(xv[2])) + dot(q3, float4(xv[3])));
        }
    }
    for (uint b = 0; b < batch; ++b) {
        const float total = simd_sum(acc[b]);
        if (lane == 0u) {
            y[y_off + b * rows + row] = total;
        }
    }
}

/* Batched Q6_K GEMM on the SIMD-group matrix units (f32 operands). The
 * 32-value K-step spans two 16-value slices; each staging thread's 16
 * values sit inside one slice, so one int8 scale per thread per step. */
kernel void ch_q6_k_gemm_mma(device const uchar *w [[buffer(0)]],
                             device const float *x [[buffer(1)]],
                             device float *y [[buffer(2)]],
                             constant uint &rows [[buffer(3)]],
                             constant uint &cols [[buffer(4)]],
                             constant uint &batch [[buffer(5)]],
                             constant uint &x_off [[buffer(6)]],
                             constant uint &y_off [[buffer(7)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[CH_MMA_ROWS * 32u];
    threadgroup float sb[CH_MMA_TOK * 32u];

    const uint token_tiles = (batch + CH_MMA_TOK - 1u) / CH_MMA_TOK;
    const uint tok0 = (tg % token_tiles) * CH_MMA_TOK;
    if (tok0 >= batch) {
        return;
    }
    const uint tile_batch = min(batch - tok0, CH_MMA_TOK);
    const uint row0 = (tg / token_tiles) * CH_MMA_ROWS;
    const uint blocks = cols / 32u;
    const size_t row_bytes = (size_t)(cols / 256u) * 210u;

    const uint w_r = tid / 2u;
    const uint w_k = (tid % 2u) * 16u;
    const device uchar *wrow = w + (size_t)min(row0 + w_r, rows - 1u) * row_bytes;
    const uint x_t = tid / 4u;
    const uint x_k = (tid % 4u) * 8u;
    const device float *xrow = x + x_off + (size_t)(tok0 + min(x_t, tile_batch - 1u)) * cols + x_k;

    threadgroup float *sa_dst = sa + 64u * ((w_k / 8u) * 8u + w_r / 8u) + (w_r % 8u);
    threadgroup float *sb_dst = sb + 64u * ((x_k / 8u) * 4u + x_t / 8u) + 8u * (x_t % 8u);

    const uint sg_r = (sg % 2u) * 4u;
    const uint sg_t = (sg / 2u) * 2u;
    simdgroup_float8x8 mc[2][4];
    for (uint j = 0; j < 2u; ++j) {
        for (uint i = 0; i < 4u; ++i) {
            mc[j][i] = simdgroup_float8x8(0.0f);
        }
    }

    const device float *xblk = xrow;
    const threadgroup float *lsa = sa + 64u * sg_r;
    const threadgroup float *lsb = sb + 64u * sg_t;
    for (uint blk = 0; blk < blocks; ++blk, xblk += 32) {
        const device uchar *block = wrow + (size_t)(blk >> 3) * 210u;
        const float d = (float)(*(const device half *)(block + 208u));
        const uint v0 = blk * 32u + w_k;
        const uint r0 = v0 & 255u;
        const float s = (float)((const device int8_t *)(block + 192u))[(r0 / 16u) & 15u];
        const float ds = d * s;
        const device uchar *qlb = block;
        const device uchar *qhb = block + 128u;
        float wv[16];
        CH_UNROLL for (uint i = 0; i < 16u; ++i) {
            const uint r = r0 + i;
            const uint c = r / 128u;
            const uint jj = (r % 128u) / 32u;
            const uint l = r % 32u;
            const uint nib = jj < 2u ? (qlb[64u * c + 32u * (jj % 2u) + l] & 0x0fu)
                                     : (qlb[64u * c + 32u * (jj % 2u) + l] >> 4);
            const uint qh = (qhb[32u * c + l] >> (2u * jj)) & 3u;
            wv[i] = ds * ((float)(nib | (qh << 4)) - 32.0f);
        }
        const device packed_float4 *xp = (const device packed_float4 *)xblk;
        const float4 xv0 = float4(xp[0]);
        const float4 xv1 = float4(xp[1]);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        CH_UNROLL for (ushort i = 0; i < 16u; ++i) {
            sa_dst[64 * 8 * (i / 8) + 8 * (i % 8)] = wv[i];
        }
        *(threadgroup float4 *)(sb_dst) = xv0;
        *(threadgroup float4 *)(sb_dst + 4u) = xv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        CH_UNROLL for (ushort kb = 0; kb < 4u; ++kb) {
            simdgroup_float8x8 ma[4];
            simdgroup_float8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort i = 0; i < 4u; ++i) {
                simdgroup_load(ma[i], lsa + 64 * (8 * kb + i), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2u; ++j) {
                simdgroup_load(mb[j], lsb + 64 * (4 * kb + j), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2u; ++j) {
                CH_UNROLL for (ushort i = 0; i < 4u; ++i) {
                    simdgroup_multiply_accumulate(mc[j][i], mb[j], ma[i], mc[j][i]);
                }
            }
        }
    }

    if (row0 + CH_MMA_ROWS <= rows && tile_batch == CH_MMA_TOK) {
        device float *c = y + y_off + (size_t)(tok0 + sg_t * 8u) * rows + row0 + sg_r * 8u;
        for (uint j = 0; j < 2u; ++j) {
            for (uint i = 0; i < 4u; ++i) {
                simdgroup_store(mc[j][i], c + (size_t)j * 8u * rows + i * 8u, rows);
            }
        }
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *ot = sa;
    for (uint j = 0; j < 2u; ++j) {
        for (uint i = 0; i < 4u; ++i) {
            simdgroup_store(mc[j][i], ot + (sg_t + j) * 8u * CH_MMA_ROWS + (sg_r + i) * 8u,
                            CH_MMA_ROWS);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < CH_MMA_ROWS * CH_MMA_TOK; idx += 128u) {
        const uint r = idx % CH_MMA_ROWS;
        const uint b = idx / CH_MMA_ROWS;
        if (row0 + r < rows && b < tile_batch) {
            y[y_off + (size_t)(tok0 + b) * rows + row0 + r] = ot[b * CH_MMA_ROWS + r];
        }
    }
}

/* Batched Q6_K GEMM with f16 operand tiles (metal-f16-act). */
kernel void ch_q6_k_gemm_mma_f16op(device const uchar *w [[buffer(0)]],
                                   device const float *x [[buffer(1)]],
                                   device float *y [[buffer(2)]],
                                   constant uint &rows [[buffer(3)]],
                                   constant uint &cols [[buffer(4)]],
                                   constant uint &batch [[buffer(5)]],
                                   constant uint &x_off [[buffer(6)]],
                                   constant uint &y_off [[buffer(7)]],
                                   uint tg [[threadgroup_position_in_grid]],
                                   uint tid [[thread_position_in_threadgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[CH_MMA_ROWS * 32u];
    threadgroup half sb[CH_MMA_TOK * 32u];
    threadgroup float edge_frag[4][64];

    const uint token_tiles = (batch + CH_MMA_TOK - 1u) / CH_MMA_TOK;
    const uint tok0 = (tg % token_tiles) * CH_MMA_TOK;
    if (tok0 >= batch) {
        return;
    }
    const uint tile_batch = min(batch - tok0, CH_MMA_TOK);
    const uint row0 = (tg / token_tiles) * CH_MMA_ROWS;
    const uint blocks = cols / 32u;
    const size_t row_bytes = (size_t)(cols / 256u) * 210u;

    const uint w_r = tid / 2u;
    const uint w_k = (tid % 2u) * 16u;
    const device uchar *wrow = w + (size_t)min(row0 + w_r, rows - 1u) * row_bytes;
    const uint x_t = tid / 4u;
    const uint x_k = (tid % 4u) * 8u;
    const device float *xrow = x + x_off + (size_t)(tok0 + min(x_t, tile_batch - 1u)) * cols + x_k;

    threadgroup half *sa_dst = sa + 64u * ((w_k / 8u) * 8u + w_r / 8u) + (w_r % 8u);
    threadgroup half *sb_dst = sb + 64u * ((x_k / 8u) * 4u + x_t / 8u) + 8u * (x_t % 8u);

    const uint sg_r = (sg % 2u) * 4u;
    const uint sg_t = (sg / 2u) * 2u;
    simdgroup_float8x8 mc[2][4];
    for (uint j = 0; j < 2u; ++j) {
        for (uint i = 0; i < 4u; ++i) {
            mc[j][i] = simdgroup_float8x8(0.0f);
        }
    }

    const device float *xblk = xrow;
    const threadgroup half *lsa = sa + 64u * sg_r;
    const threadgroup half *lsb = sb + 64u * sg_t;
    for (uint blk = 0; blk < blocks; ++blk, xblk += 32) {
        const device uchar *block = wrow + (size_t)(blk >> 3) * 210u;
        const float d = (float)(*(const device half *)(block + 208u));
        const uint v0 = blk * 32u + w_k;
        const uint r0 = v0 & 255u;
        const float s = (float)((const device int8_t *)(block + 192u))[(r0 / 16u) & 15u];
        const float ds = d * s;
        const device uchar *qlb = block;
        const device uchar *qhb = block + 128u;
        float wv[16];
        CH_UNROLL for (uint i = 0; i < 16u; ++i) {
            const uint r = r0 + i;
            const uint c = r / 128u;
            const uint jj = (r % 128u) / 32u;
            const uint l = r % 32u;
            const uint nib = jj < 2u ? (qlb[64u * c + 32u * (jj % 2u) + l] & 0x0fu)
                                     : (qlb[64u * c + 32u * (jj % 2u) + l] >> 4);
            const uint qh = (qhb[32u * c + l] >> (2u * jj)) & 3u;
            wv[i] = ds * ((float)(nib | (qh << 4)) - 32.0f);
        }
        const device packed_float4 *xp = (const device packed_float4 *)xblk;
        const float4 xv0 = float4(xp[0]);
        const float4 xv1 = float4(xp[1]);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        CH_UNROLL for (ushort i = 0; i < 16u; ++i) {
            sa_dst[64 * 8 * (i / 8) + 8 * (i % 8)] = (half)wv[i];
        }
        *(threadgroup half4 *)(sb_dst) = half4(xv0);
        *(threadgroup half4 *)(sb_dst + 4u) = half4(xv1);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        CH_UNROLL for (ushort kb = 0; kb < 4u; ++kb) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort i = 0; i < 4u; ++i) {
                simdgroup_load(ma[i], lsa + 64 * (8 * kb + i), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2u; ++j) {
                simdgroup_load(mb[j], lsb + 64 * (4 * kb + j), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2u; ++j) {
                CH_UNROLL for (ushort i = 0; i < 4u; ++i) {
                    simdgroup_multiply_accumulate(mc[j][i], mb[j], ma[i], mc[j][i]);
                }
            }
        }
    }

    if (row0 + CH_MMA_ROWS <= rows && tile_batch == CH_MMA_TOK) {
        device float *c = y + y_off + (size_t)(tok0 + sg_t * 8u) * rows + row0 + sg_r * 8u;
        for (uint j = 0; j < 2u; ++j) {
            for (uint i = 0; i < 4u; ++i) {
                simdgroup_store(mc[j][i], c + (size_t)j * 8u * rows + i * 8u, rows);
            }
        }
        return;
    }
    const uint tok_end = tok0 + tile_batch;
    {
        threadgroup float * const edge_frag_mine = edge_frag[sg];
        const uint rb_base = row0 + sg_r * 8u;
        const uint tb_base = tok0 + sg_t * 8u;
        for (uint j = 0; j < 2u; ++j) {
            for (uint i = 0; i < 4u; ++i) {
                const uint rb = rb_base + i * 8u;
                const uint tb = tb_base + j * 8u;
                if (rb >= rows || tb >= tok_end) {
                    continue;
                }
                if (rb + 8u <= rows && tb + 8u <= tok_end) {
                    simdgroup_store(mc[j][i], y + y_off + (size_t)tb * rows + rb, rows);
                } else {
                    simdgroup_store(mc[j][i], edge_frag_mine, 8);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint lane = tid % 32u; lane < 64u; lane += 32u) {
                        const uint fr = lane / 8u;
                        const uint ft = lane % 8u;
                        if (tb + fr < tok_end && rb + ft < rows) {
                            y[y_off + (size_t)(tb + fr) * rows + rb + ft] =
                                edge_frag_mine[fr * 8u + ft];
                        }
                    }
                }
            }
        }
    }
}


/* ---------------- Q5_K kernels (2026-11-22) ----------------
 * Pinned to llama.cpp b11695 (block_q5_K, dequantize_row_q5_K):
 * 256-value super-blocks, bytes [f16 d][f16 dmin][u8 scales; 12][u8
 * qh; 32][u8 qs; 128]. Same eight-slice layout and 6-bit scale packing
 * as Q4_K; slice k reads ql bytes [48 + 32*(k/2), +32) (low nibble
 * for even k, high for odd) with its high bit in qh[l] at shift
 * 2*(k/2) + (k%2), and value = d*sc*(nib + (hbit << 4)) - dmin*m. */

kernel void ch_q5_k_gemv(device const uchar *w [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device float *y [[buffer(2)]],
                         constant uint &rows [[buffer(3)]],
                         constant uint &cols [[buffer(4)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    const uint blocks = cols / 256u;
    const size_t row_bytes = (size_t)blocks * 176u;
    const uint first = (tg * CH_Q8_SG + sg) * CH_Q8_ROWS;
    if (first >= rows) {
        return;
    }
    const uint k = lane % 8u;
    const uint lane_sb = lane / 8u;
    const bool hi = k % 2u == 1u;
    const uint shift = 2u * (k / 2u) + (k % 2u);

    float acc[CH_Q8_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint b = lane_sb; b < blocks; b += 4u) {
        const device packed_float4 *xv =
            (const device packed_float4 *)(x + (size_t)b * 256u + k * 32u);
        float4 x0 = float4(xv[0]);
        float4 x1 = float4(xv[1]);
        float4 x2 = float4(xv[2]);
        float4 x3 = float4(xv[3]);
        float4 x4 = float4(xv[4]);
        float4 x5 = float4(xv[5]);
        float4 x6 = float4(xv[6]);
        float4 x7 = float4(xv[7]);
        for (uint r = 0; r < CH_Q8_ROWS; ++r) {
            const uint row = first + r;
            if (row >= rows) {
                break;
            }
            const device uchar *blk = w + (size_t)row * row_bytes + (size_t)b * 176u;
            const float d = (float)(*(const device half *)blk);
            const float dmin = (float)(*(const device half *)(blk + 2u));
            const device uchar *sc = blk + 4u;
            float a, mn;
            if (k < 4u) {
                a = (float)(sc[k] & 63u);
                mn = (float)(sc[k + 4u] & 63u);
            } else {
                a = (float)((sc[k + 4u] & 0xFu) | ((sc[k - 4u] >> 6) << 4));
                mn = (float)((sc[k + 4u] >> 4) | ((sc[k] >> 6) << 4));
            }
            const float ds = d * a;
            const float dm = dmin * mn;
            const device packed_uchar4 *qv =
                (const device packed_uchar4 *)(blk + 48u + 32u * (k / 2u));
            const device packed_uchar4 *hv = (const device packed_uchar4 *)(blk + 16u);
            const uchar4 h0 = uchar4(hv[0]);
            const uchar4 h1 = uchar4(hv[1]);
            const uchar4 h2 = uchar4(hv[2]);
            const uchar4 h3 = uchar4(hv[3]);
            const uchar4 h4 = uchar4(hv[4]);
            const uchar4 h5 = uchar4(hv[5]);
            const uchar4 h6 = uchar4(hv[6]);
            const uchar4 h7 = uchar4(hv[7]);
            const uchar4 hb0 = (h0 >> shift) & (uchar4)1u;
            const uchar4 hb1 = (h1 >> shift) & (uchar4)1u;
            const uchar4 hb2 = (h2 >> shift) & (uchar4)1u;
            const uchar4 hb3 = (h3 >> shift) & (uchar4)1u;
            const uchar4 hb4 = (h4 >> shift) & (uchar4)1u;
            const uchar4 hb5 = (h5 >> shift) & (uchar4)1u;
            const uchar4 hb6 = (h6 >> shift) & (uchar4)1u;
            const uchar4 hb7 = (h7 >> shift) & (uchar4)1u;
            float4 q0, q1, q2, q3, q4, q5, q6, q7;
            if (!hi) {
                q0 = float4(uchar4(qv[0]) & (uchar4)0x0fu) + float4(hb0) * 16.0f;
                q1 = float4(uchar4(qv[1]) & (uchar4)0x0fu) + float4(hb1) * 16.0f;
                q2 = float4(uchar4(qv[2]) & (uchar4)0x0fu) + float4(hb2) * 16.0f;
                q3 = float4(uchar4(qv[3]) & (uchar4)0x0fu) + float4(hb3) * 16.0f;
                q4 = float4(uchar4(qv[4]) & (uchar4)0x0fu) + float4(hb4) * 16.0f;
                q5 = float4(uchar4(qv[5]) & (uchar4)0x0fu) + float4(hb5) * 16.0f;
                q6 = float4(uchar4(qv[6]) & (uchar4)0x0fu) + float4(hb6) * 16.0f;
                q7 = float4(uchar4(qv[7]) & (uchar4)0x0fu) + float4(hb7) * 16.0f;
            } else {
                q0 = float4(uchar4(qv[0]) >> 4) + float4(hb0) * 16.0f;
                q1 = float4(uchar4(qv[1]) >> 4) + float4(hb1) * 16.0f;
                q2 = float4(uchar4(qv[2]) >> 4) + float4(hb2) * 16.0f;
                q3 = float4(uchar4(qv[3]) >> 4) + float4(hb3) * 16.0f;
                q4 = float4(uchar4(qv[4]) >> 4) + float4(hb4) * 16.0f;
                q5 = float4(uchar4(qv[5]) >> 4) + float4(hb5) * 16.0f;
                q6 = float4(uchar4(qv[6]) >> 4) + float4(hb6) * 16.0f;
                q7 = float4(uchar4(qv[7]) >> 4) + float4(hb7) * 16.0f;
            }
            acc[r] += (dot(q0 * ds - dm, x0) + dot(q1 * ds - dm, x1)
                     + dot(q2 * ds - dm, x2) + dot(q3 * ds - dm, x3)
                     + dot(q4 * ds - dm, x4) + dot(q5 * ds - dm, x5)
                     + dot(q6 * ds - dm, x6) + dot(q7 * ds - dm, x7));
        }
    }
    for (uint r = 0; r < CH_Q8_ROWS; ++r) {
        const float total = simd_sum(acc[r]);
        const uint row = first + r;
        if (lane == 0u && row < rows) {
            y[row] = total;
        }
    }
}

/* Dequantize one Q5_K matrix row (or embedding row) to f32: one thread
 * per output element. */
kernel void ch_q5_k_row_f32(device const uchar *w [[buffer(0)]],
                            device float *out [[buffer(1)]],
                            constant uint &row [[buffer(2)]],
                            constant uint &cols [[buffer(3)]],
                            constant uint &out_off [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) {
        return;
    }
    const uint blocks = cols / 256u;
    const device uchar *blk =
        w + (size_t)row * (size_t)blocks * 176u + (size_t)(gid / 256u) * 176u;
    const uint k = (gid % 256u) / 32u;
    const float d = (float)(*(const device half *)blk);
    const float dmin = (float)(*(const device half *)(blk + 2u));
    const device uchar *sc = blk + 4u;
    float a, mn;
    if (k < 4u) {
        a = (float)(sc[k] & 63u);
        mn = (float)(sc[k + 4u] & 63u);
    } else {
        a = (float)((sc[k + 4u] & 0xFu) | ((sc[k - 4u] >> 6) << 4));
        mn = (float)((sc[k + 4u] >> 4) | ((sc[k] >> 6) << 4));
    }
    const uint j = gid % 32u;
    const device uchar *qs = blk + 48u + 32u * (k / 2u) + j;
    const uint nib = k % 2u == 0u ? (qs[0] & 0x0fu) : (qs[0] >> 4);
    const uint hb = (blk[16u + j] >> (2u * (k / 2u) + (k % 2u))) & 1u;
    out[out_off + gid] = d * a * ((float)(nib | (hb << 4)) - 0.0f) - dmin * mn;
}

kernel void ch_q5_k_row_indirect_f32(device const uchar *w [[buffer(0)]],
                                     device float *out [[buffer(1)]],
                                     device const uint *ids [[buffer(2)]],
                                     constant uint &slot [[buffer(3)]],
                                     constant uint &rows [[buffer(4)]],
                                     constant uint &cols [[buffer(5)]],
                                     uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) {
        return;
    }
    const uint row = ids[slot];
    if (row >= rows) {
        out[gid] = 0.0f;
        return;
    }
    const uint blocks = cols / 256u;
    const device uchar *blk =
        w + ((size_t)row * (size_t)blocks + gid / 256u) * 176u;
    const uint k = (gid % 256u) / 32u;
    const float d = (float)(*(const device half *)blk);
    const float dmin = (float)(*(const device half *)(blk + 2u));
    const device uchar *sc = blk + 4u;
    float a, mn;
    if (k < 4u) {
        a = (float)(sc[k] & 63u);
        mn = (float)(sc[k + 4u] & 63u);
    } else {
        a = (float)((sc[k + 4u] & 0xFu) | ((sc[k - 4u] >> 6) << 4));
        mn = (float)((sc[k + 4u] >> 4) | ((sc[k] >> 6) << 4));
    }
    const uint j = gid % 32u;
    const device uchar *qs = blk + 48u + 32u * (k / 2u) + j;
    const uint nib = k % 2u == 0u ? (qs[0] & 0x0fu) : (qs[0] >> 4);
    const uint hb = (blk[16u + j] >> (2u * (k / 2u) + (k % 2u))) & 1u;
    out[gid] = d * a * (float)(nib | (hb << 4)) - dmin * mn;
}

/* Mid-batch Q5_K GEMV projection: one threadgroup per output row. */
kernel void ch_q5_k_gemm(device const uchar *w [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device float *y [[buffer(2)]],
                         constant uint &rows [[buffer(3)]],
                         constant uint &cols [[buffer(4)]],
                         constant uint &batch [[buffer(5)]],
                         constant uint &x_off [[buffer(6)]],
                         constant uint &y_off [[buffer(7)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint lane [[thread_position_in_threadgroup]]) {
    if (row >= rows || batch == 0u || batch > 32u) {
        return;
    }
    const uint blocks = cols / 256u;
    const device uchar *wrow = w + (size_t)row * (size_t)blocks * 176u;
    const uint k = lane % 8u;
    const uint lane_sb = lane / 8u;
    const bool hi = k % 2u == 1u;
    const uint shift = 2u * (k / 2u) + (k % 2u);
    float acc[32];
    for (uint b = 0; b < batch; ++b) {
        acc[b] = 0.0f;
    }
    for (uint blk = lane_sb; blk < blocks; blk += 4u) {
        const device uchar *block = wrow + (size_t)blk * 176u;
        const float d = (float)(*(const device half *)block);
        const float dmin = (float)(*(const device half *)(block + 2u));
        const device uchar *sc = block + 4u;
        float a, mn;
        if (k < 4u) {
            a = (float)(sc[k] & 63u);
            mn = (float)(sc[k + 4u] & 63u);
        } else {
            a = (float)((sc[k + 4u] & 0xFu) | ((sc[k - 4u] >> 6) << 4));
            mn = (float)((sc[k + 4u] >> 4) | ((sc[k] >> 6) << 4));
        }
        const float ds = d * a;
        const float dm = dmin * mn;
        const device packed_uchar4 *qv =
            (const device packed_uchar4 *)(block + 48u + 32u * (k / 2u));
        const device packed_uchar4 *hv = (const device packed_uchar4 *)(block + 16u);
        const uchar4 h0 = (uchar4(hv[0]) >> shift) & (uchar4)1u;
        const uchar4 h1 = (uchar4(hv[1]) >> shift) & (uchar4)1u;
        const uchar4 h2 = (uchar4(hv[2]) >> shift) & (uchar4)1u;
        const uchar4 h3 = (uchar4(hv[3]) >> shift) & (uchar4)1u;
        const uchar4 h4 = (uchar4(hv[4]) >> shift) & (uchar4)1u;
        const uchar4 h5 = (uchar4(hv[5]) >> shift) & (uchar4)1u;
        const uchar4 h6 = (uchar4(hv[6]) >> shift) & (uchar4)1u;
        const uchar4 h7 = (uchar4(hv[7]) >> shift) & (uchar4)1u;
        float4 q0, q1, q2, q3, q4, q5, q6, q7;
        if (!hi) {
            q0 = float4(uchar4(qv[0]) & (uchar4)0x0fu) + float4(h0) * 16.0f;
            q1 = float4(uchar4(qv[1]) & (uchar4)0x0fu) + float4(h1) * 16.0f;
            q2 = float4(uchar4(qv[2]) & (uchar4)0x0fu) + float4(h2) * 16.0f;
            q3 = float4(uchar4(qv[3]) & (uchar4)0x0fu) + float4(h3) * 16.0f;
            q4 = float4(uchar4(qv[4]) & (uchar4)0x0fu) + float4(h4) * 16.0f;
            q5 = float4(uchar4(qv[5]) & (uchar4)0x0fu) + float4(h5) * 16.0f;
            q6 = float4(uchar4(qv[6]) & (uchar4)0x0fu) + float4(h6) * 16.0f;
            q7 = float4(uchar4(qv[7]) & (uchar4)0x0fu) + float4(h7) * 16.0f;
        } else {
            q0 = float4(uchar4(qv[0]) >> 4) + float4(h0) * 16.0f;
            q1 = float4(uchar4(qv[1]) >> 4) + float4(h1) * 16.0f;
            q2 = float4(uchar4(qv[2]) >> 4) + float4(h2) * 16.0f;
            q3 = float4(uchar4(qv[3]) >> 4) + float4(h3) * 16.0f;
            q4 = float4(uchar4(qv[4]) >> 4) + float4(h4) * 16.0f;
            q5 = float4(uchar4(qv[5]) >> 4) + float4(h5) * 16.0f;
            q6 = float4(uchar4(qv[6]) >> 4) + float4(h6) * 16.0f;
            q7 = float4(uchar4(qv[7]) >> 4) + float4(h7) * 16.0f;
        }
        q0 = q0 * ds - dm;
        q1 = q1 * ds - dm;
        q2 = q2 * ds - dm;
        q3 = q3 * ds - dm;
        q4 = q4 * ds - dm;
        q5 = q5 * ds - dm;
        q6 = q6 * ds - dm;
        q7 = q7 * ds - dm;
        const uint col = blk * 256u + k * 32u;
        for (uint b = 0; b < batch; ++b) {
            const device packed_float4 *xv =
                (const device packed_float4 *)(x + x_off + b * cols + col);
            acc[b] += dot(q0, float4(xv[0])) + dot(q1, float4(xv[1]))
                    + dot(q2, float4(xv[2])) + dot(q3, float4(xv[3]))
                    + dot(q4, float4(xv[4])) + dot(q5, float4(xv[5]))
                    + dot(q6, float4(xv[6])) + dot(q7, float4(xv[7]));
        }
    }
    for (uint b = 0; b < batch; ++b) {
        const float total = simd_sum(acc[b]);
        if (lane == 0u) {
            y[y_off + b * rows + row] = total;
        }
    }
}

/* Batched Q5_K GEMM on the SIMD-group matrix units (f32 operands). The
 * 32-value K-step tiles the super-block's eight slices exactly
 * (slice k = blk & 7), so each thread stages one (scale, min) pair. */
kernel void ch_q5_k_gemm_mma(device const uchar *w [[buffer(0)]],
                             device const float *x [[buffer(1)]],
                             device float *y [[buffer(2)]],
                             constant uint &rows [[buffer(3)]],
                             constant uint &cols [[buffer(4)]],
                             constant uint &batch [[buffer(5)]],
                             constant uint &x_off [[buffer(6)]],
                             constant uint &y_off [[buffer(7)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[CH_MMA_ROWS * 32u];
    threadgroup float sb[CH_MMA_TOK * 32u];

    const uint token_tiles = (batch + CH_MMA_TOK - 1u) / CH_MMA_TOK;
    const uint tok0 = (tg % token_tiles) * CH_MMA_TOK;
    if (tok0 >= batch) {
        return;
    }
    const uint tile_batch = min(batch - tok0, CH_MMA_TOK);
    const uint row0 = (tg / token_tiles) * CH_MMA_ROWS;
    const uint blocks = cols / 32u;
    const size_t row_bytes = (size_t)(cols / 256u) * 176u;

    const uint w_r = tid / 2u;
    const uint w_k = (tid % 2u) * 16u;
    const device uchar *wrow = w + (size_t)min(row0 + w_r, rows - 1u) * row_bytes;
    const uint x_t = tid / 4u;
    const uint x_k = (tid % 4u) * 8u;
    const device float *xrow = x + x_off + (size_t)(tok0 + min(x_t, tile_batch - 1u)) * cols + x_k;

    threadgroup float *sa_dst = sa + 64u * ((w_k / 8u) * 8u + w_r / 8u) + (w_r % 8u);
    threadgroup float *sb_dst = sb + 64u * ((x_k / 8u) * 4u + x_t / 8u) + 8u * (x_t % 8u);

    const uint sg_r = (sg % 2u) * 4u;
    const uint sg_t = (sg / 2u) * 2u;
    simdgroup_float8x8 mc[2][4];
    for (uint j = 0; j < 2u; ++j) {
        for (uint i = 0; i < 4u; ++i) {
            mc[j][i] = simdgroup_float8x8(0.0f);
        }
    }

    const device float *xblk = xrow;
    const threadgroup float *lsa = sa + 64u * sg_r;
    const threadgroup float *lsb = sb + 64u * sg_t;
    for (uint blk = 0; blk < blocks; ++blk, xblk += 32) {
        const uint k = blk & 7u;
        const device uchar *block = wrow + (size_t)(blk >> 3) * 176u;
        const float d = (float)(*(const device half *)block);
        const float dmin = (float)(*(const device half *)(block + 2u));
        const device uchar *sc = block + 4u;
        float a, mn;
        if (k < 4u) {
            a = (float)(sc[k] & 63u);
            mn = (float)(sc[k + 4u] & 63u);
        } else {
            a = (float)((sc[k + 4u] & 0xFu) | ((sc[k - 4u] >> 6) << 4));
            mn = (float)((sc[k + 4u] >> 4) | ((sc[k] >> 6) << 4));
        }
        const float ds = d * a;
        const float dm = dmin * mn;
        const device uchar *qs = block + 48u + 32u * (k / 2u) + w_k;
        const device uchar *qh = block + 16u + w_k;
        const uint shift = 2u * (k / 2u) + (k % 2u);
        float wv[16];
        if (k % 2u == 0u) {
            CH_UNROLL for (uint i = 0; i < 16u; ++i) {
                const uint nib = qs[i] & 0x0fu;
                const uint hb = (qh[i] >> shift) & 1u;
                wv[i] = (float)(nib | (hb << 4)) * ds - dm;
            }
        } else {
            CH_UNROLL for (uint i = 0; i < 16u; ++i) {
                const uint nib = qs[i] >> 4;
                const uint hb = (qh[i] >> shift) & 1u;
                wv[i] = (float)(nib | (hb << 4)) * ds - dm;
            }
        }
        const device packed_float4 *xp = (const device packed_float4 *)xblk;
        const float4 xv0 = float4(xp[0]);
        const float4 xv1 = float4(xp[1]);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        CH_UNROLL for (ushort i = 0; i < 16u; ++i) {
            sa_dst[64 * 8 * (i / 8) + 8 * (i % 8)] = wv[i];
        }
        *(threadgroup float4 *)(sb_dst) = xv0;
        *(threadgroup float4 *)(sb_dst + 4u) = xv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        CH_UNROLL for (ushort kb = 0; kb < 4u; ++kb) {
            simdgroup_float8x8 ma[4];
            simdgroup_float8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort i = 0; i < 4u; ++i) {
                simdgroup_load(ma[i], lsa + 64 * (8 * kb + i), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2u; ++j) {
                simdgroup_load(mb[j], lsb + 64 * (4 * kb + j), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2u; ++j) {
                CH_UNROLL for (ushort i = 0; i < 4u; ++i) {
                    simdgroup_multiply_accumulate(mc[j][i], mb[j], ma[i], mc[j][i]);
                }
            }
        }
    }

    if (row0 + CH_MMA_ROWS <= rows && tile_batch == CH_MMA_TOK) {
        device float *c = y + y_off + (size_t)(tok0 + sg_t * 8u) * rows + row0 + sg_r * 8u;
        for (uint j = 0; j < 2u; ++j) {
            for (uint i = 0; i < 4u; ++i) {
                simdgroup_store(mc[j][i], c + (size_t)j * 8u * rows + i * 8u, rows);
            }
        }
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *ot = sa;
    for (uint j = 0; j < 2u; ++j) {
        for (uint i = 0; i < 4u; ++i) {
            simdgroup_store(mc[j][i], ot + (sg_t + j) * 8u * CH_MMA_ROWS + (sg_r + i) * 8u,
                            CH_MMA_ROWS);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < CH_MMA_ROWS * CH_MMA_TOK; idx += 128u) {
        const uint r = idx % CH_MMA_ROWS;
        const uint b = idx / CH_MMA_ROWS;
        if (row0 + r < rows && b < tile_batch) {
            y[y_off + (size_t)(tok0 + b) * rows + row0 + r] = ot[b * CH_MMA_ROWS + r];
        }
    }
}

/* Batched Q5_K GEMM with f16 operand tiles (metal-f16-act). */
kernel void ch_q5_k_gemm_mma_f16op(device const uchar *w [[buffer(0)]],
                                   device const float *x [[buffer(1)]],
                                   device float *y [[buffer(2)]],
                                   constant uint &rows [[buffer(3)]],
                                   constant uint &cols [[buffer(4)]],
                                   constant uint &batch [[buffer(5)]],
                                   constant uint &x_off [[buffer(6)]],
                                   constant uint &y_off [[buffer(7)]],
                                   uint tg [[threadgroup_position_in_grid]],
                                   uint tid [[thread_position_in_threadgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[CH_MMA_ROWS * 32u];
    threadgroup half sb[CH_MMA_TOK * 32u];
    threadgroup float edge_frag[4][64];

    const uint token_tiles = (batch + CH_MMA_TOK - 1u) / CH_MMA_TOK;
    const uint tok0 = (tg % token_tiles) * CH_MMA_TOK;
    if (tok0 >= batch) {
        return;
    }
    const uint tile_batch = min(batch - tok0, CH_MMA_TOK);
    const uint row0 = (tg / token_tiles) * CH_MMA_ROWS;
    const uint blocks = cols / 32u;
    const size_t row_bytes = (size_t)(cols / 256u) * 176u;

    const uint w_r = tid / 2u;
    const uint w_k = (tid % 2u) * 16u;
    const device uchar *wrow = w + (size_t)min(row0 + w_r, rows - 1u) * row_bytes;
    const uint x_t = tid / 4u;
    const uint x_k = (tid % 4u) * 8u;
    const device float *xrow = x + x_off + (size_t)(tok0 + min(x_t, tile_batch - 1u)) * cols + x_k;

    threadgroup half *sa_dst = sa + 64u * ((w_k / 8u) * 8u + w_r / 8u) + (w_r % 8u);
    threadgroup half *sb_dst = sb + 64u * ((x_k / 8u) * 4u + x_t / 8u) + 8u * (x_t % 8u);

    const uint sg_r = (sg % 2u) * 4u;
    const uint sg_t = (sg / 2u) * 2u;
    simdgroup_float8x8 mc[2][4];
    for (uint j = 0; j < 2u; ++j) {
        for (uint i = 0; i < 4u; ++i) {
            mc[j][i] = simdgroup_float8x8(0.0f);
        }
    }

    const device float *xblk = xrow;
    const threadgroup half *lsa = sa + 64u * sg_r;
    const threadgroup half *lsb = sb + 64u * sg_t;
    for (uint blk = 0; blk < blocks; ++blk, xblk += 32) {
        const uint k = blk & 7u;
        const device uchar *block = wrow + (size_t)(blk >> 3) * 176u;
        const float d = (float)(*(const device half *)block);
        const float dmin = (float)(*(const device half *)(block + 2u));
        const device uchar *sc = block + 4u;
        float a, mn;
        if (k < 4u) {
            a = (float)(sc[k] & 63u);
            mn = (float)(sc[k + 4u] & 63u);
        } else {
            a = (float)((sc[k + 4u] & 0xFu) | ((sc[k - 4u] >> 6) << 4));
            mn = (float)((sc[k + 4u] >> 4) | ((sc[k] >> 6) << 4));
        }
        const float ds = d * a;
        const float dm = dmin * mn;
        const device uchar *qs = block + 48u + 32u * (k / 2u) + w_k;
        const device uchar *qh = block + 16u + w_k;
        const uint shift = 2u * (k / 2u) + (k % 2u);
        float wv[16];
        if (k % 2u == 0u) {
            CH_UNROLL for (uint i = 0; i < 16u; ++i) {
                const uint nib = qs[i] & 0x0fu;
                const uint hb = (qh[i] >> shift) & 1u;
                wv[i] = (float)(nib | (hb << 4)) * ds - dm;
            }
        } else {
            CH_UNROLL for (uint i = 0; i < 16u; ++i) {
                const uint nib = qs[i] >> 4;
                const uint hb = (qh[i] >> shift) & 1u;
                wv[i] = (float)(nib | (hb << 4)) * ds - dm;
            }
        }
        const device packed_float4 *xp = (const device packed_float4 *)xblk;
        const float4 xv0 = float4(xp[0]);
        const float4 xv1 = float4(xp[1]);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        CH_UNROLL for (ushort i = 0; i < 16u; ++i) {
            sa_dst[64 * 8 * (i / 8) + 8 * (i % 8)] = (half)wv[i];
        }
        *(threadgroup half4 *)(sb_dst) = half4(xv0);
        *(threadgroup half4 *)(sb_dst + 4u) = half4(xv1);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        CH_UNROLL for (ushort kb = 0; kb < 4u; ++kb) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort i = 0; i < 4u; ++i) {
                simdgroup_load(ma[i], lsa + 64 * (8 * kb + i), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2u; ++j) {
                simdgroup_load(mb[j], lsb + 64 * (4 * kb + j), 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            CH_UNROLL for (ushort j = 0; j < 2u; ++j) {
                CH_UNROLL for (ushort i = 0; i < 4u; ++i) {
                    simdgroup_multiply_accumulate(mc[j][i], mb[j], ma[i], mc[j][i]);
                }
            }
        }
    }

    if (row0 + CH_MMA_ROWS <= rows && tile_batch == CH_MMA_TOK) {
        device float *c = y + y_off + (size_t)(tok0 + sg_t * 8u) * rows + row0 + sg_r * 8u;
        for (uint j = 0; j < 2u; ++j) {
            for (uint i = 0; i < 4u; ++i) {
                simdgroup_store(mc[j][i], c + (size_t)j * 8u * rows + i * 8u, rows);
            }
        }
        return;
    }
    const uint tok_end = tok0 + tile_batch;
    {
        threadgroup float * const edge_frag_mine = edge_frag[sg];
        const uint rb_base = row0 + sg_r * 8u;
        const uint tb_base = tok0 + sg_t * 8u;
        for (uint j = 0; j < 2u; ++j) {
            for (uint i = 0; i < 4u; ++i) {
                const uint rb = rb_base + i * 8u;
                const uint tb = tb_base + j * 8u;
                if (rb >= rows || tb >= tok_end) {
                    continue;
                }
                if (rb + 8u <= rows && tb + 8u <= tok_end) {
                    simdgroup_store(mc[j][i], y + y_off + (size_t)tb * rows + rb, rows);
                } else {
                    simdgroup_store(mc[j][i], edge_frag_mine, 8);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint lane = tid % 32u; lane < 64u; lane += 32u) {
                        const uint fr = lane / 8u;
                        const uint ft = lane % 8u;
                        if (tb + fr < tok_end && rb + ft < rows) {
                            y[y_off + (size_t)(tb + fr) * rows + rb + ft] =
                                edge_frag_mine[fr * 8u + ft];
                        }
                    }
                }
            }
        }
    }
}


/* ---------------- F32 matrix kernels (2026-11-22) ----------------
 * Unsloth's K-quant artifacts store the small GDN projections
 * (ssm_alpha/beta, ssm_conv1d) as F32 matrices; the engine's Matrix
 * path needs a type-0 gemv/gemm. One row per threadgroup, 32 lanes. */

kernel void ch_f32_gemv(device const float *w [[buffer(0)]],
                        device const float *x [[buffer(1)]],
                        device float *y [[buffer(2)]],
                        constant uint &rows [[buffer(3)]],
                        constant uint &cols [[buffer(4)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint lane [[thread_position_in_threadgroup]]) {
    if (row >= rows) {
        return;
    }
    const device float *wrow = w + (size_t)row * (size_t)cols;
    float acc = 0.0f;
    for (uint i = lane; i < cols; i += 32u) {
        acc += wrow[i] * x[i];
    }
    const float total = simd_sum(acc);
    if (lane == 0u) {
        y[row] = total;
    }
}

/* Mid-batch F32 GEMV projection: one threadgroup per output row. */
kernel void ch_f32_gemm(device const float *w [[buffer(0)]],
                        device const float *x [[buffer(1)]],
                        device float *y [[buffer(2)]],
                        constant uint &rows [[buffer(3)]],
                        constant uint &cols [[buffer(4)]],
                        constant uint &batch [[buffer(5)]],
                        constant uint &x_off [[buffer(6)]],
                        constant uint &y_off [[buffer(7)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint lane [[thread_position_in_threadgroup]]) {
    if (row >= rows || batch == 0u || batch > 32u) {
        return;
    }
    const device float *wrow = w + (size_t)row * (size_t)cols;
    float acc[32];
    for (uint b = 0; b < batch; ++b) {
        acc[b] = 0.0f;
    }
    for (uint i = lane; i < cols; i += 32u) {
        const float wv = wrow[i];
        for (uint b = 0; b < batch; ++b) {
            acc[b] += wv * x[x_off + b * cols + i];
        }
    }
    for (uint b = 0; b < batch; ++b) {
        const float total = simd_sum(acc[b]);
        if (lane == 0u) {
            y[y_off + b * rows + row] = total;
        }
    }
}
