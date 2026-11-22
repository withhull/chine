/*
 * ch_metal.m — Objective-C implementation of the ChineCore <-> Metal ABI
 * (ADR-007). Compiled without ARC: this is a C-style FFI layer with
 * explicit retain/release ownership; completion counters and the live
 * allocation figure are C11 atomics so completion handlers can update them
 * from Metal's thread without locks.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <os/lock.h>
#include <stdatomic.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ch_metal.h"

/* 64 partial (f32 value, u32 index) pairs for ch_metal_argmax_f32. */
#define CH_ARGMAX_PARTIAL_BYTES 512u

#define CH_PROFILE_SLOTS 64
#define CH_PROFILE_NAME 40
/* Grouped per-operator attribution: one label record per dispatch inside
 * the shared command buffer; deltas between consecutive counter samples
 * fold into the profile slots at the next fence. */
#define CH_GROUP_LABELS 2048

typedef struct {
    char name[CH_PROFILE_NAME];
    uint32_t idx;
} ch_group_label;

/* One operator label's accumulated GPU time (profiling only). */
typedef struct {
    char name[CH_PROFILE_NAME];
    uint64_t nanos;
    uint64_t count;
} ch_profile_slot;

/* A label copied by value into completion handlers, so no pointer has to
 * outlive the dispatch call that produced it. */
typedef struct {
    char s[CH_PROFILE_NAME];
} ch_label;

struct ch_metal_ctx {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> pso_scale;
    id<MTLComputePipelineState> pso_axpy;
    id<MTLComputePipelineState> pso_copy;
    id<MTLComputePipelineState> pso_argmax_partial;
    id<MTLComputePipelineState> pso_argmax_final;
    id<MTLComputePipelineState> pso_q8row_indirect;
    id<MTLComputePipelineState> pso_q35_conv_chunk;
    id<MTLComputePipelineState> pso_q35_conv_chunk_shift;
    id<MTLComputePipelineState> pso_q35_split_qkv_chunk;
    id<MTLComputePipelineState> pso_q35_beta_decay_chunk;
    id<MTLComputePipelineState> pso_q35_gdn_chunk;
    id<MTLComputePipelineState> pso_q35_gdn_chunk_k128;
    id<MTLComputePipelineState> pso_q35_rope_chunk;
    id<MTLComputePipelineState> pso_q35_kv_store_chunk;
    id<MTLComputePipelineState> pso_q35_attention_chunk;
    id<MTLComputePipelineState> pso_q35_attn_sk_256x4;
    id<MTLComputePipelineState> pso_q35_attn_sk_16x2;
    id<MTLComputePipelineState> pso_q35_attn_sk_128x2;
    id<MTLComputePipelineState> pso_q35_attn_sk_128x4;
    id<MTLComputePipelineState> pso_q35_attn_sk_f32_16x2;
    id<MTLComputePipelineState> pso_q35_attn_sk_f32_128x2;
    id<MTLComputePipelineState> pso_q35_attn_sk_f32_128x4;
    id<MTLComputePipelineState> pso_q35_attn_sk_merge_256x4;
    id<MTLComputePipelineState> pso_q35_attn_sk_merge_16x2;
    id<MTLComputePipelineState> pso_q35_attn_sk_merge_128x2;
    id<MTLComputePipelineState> pso_q35_attn_sk_merge_128x4;
    id<MTLComputePipelineState> pso_q35_attention_gqa_256x4;
    id<MTLComputePipelineState> pso_q35_attention_gqa_16x2;
    id<MTLComputePipelineState> pso_q35_attention_gqa_128x2;
    id<MTLComputePipelineState> pso_q35_attention_gqa_128x4;
    id<MTLComputePipelineState> pso_q35_attention_gqa_f32_16x2;
    id<MTLComputePipelineState> pso_q35_attention_gqa_f32_128x2;
    id<MTLComputePipelineState> pso_q35_attention_gqa_f32_128x4;
    id<MTLComputePipelineState> pso_q35_attention_chunk_f32;
    id<MTLComputePipelineState> pso_q35_kv_store_chunk_f32;
    id<MTLComputePipelineState> pso_f16row_indirect;
    id<MTLComputePipelineState> pso_f16f32;
    id<MTLComputePipelineState> pso_f32f16;
    id<MTLComputePipelineState> pso_q8gemv;
    id<MTLComputePipelineState> pso_f16gemv;
    id<MTLComputePipelineState> pso_q8row;
    id<MTLComputePipelineState> pso_rmsnorm;
    id<MTLComputePipelineState> pso_norm_add;
    id<MTLComputePipelineState> pso_qk_norm;
    id<MTLComputePipelineState> pso_silu;
    id<MTLComputePipelineState> pso_gelu_tanh_mul;
    id<MTLComputePipelineState> pso_f16row;
    id<MTLComputePipelineState> pso_q8gemm;
    id<MTLComputePipelineState> pso_q8gemm_mma;
    id<MTLComputePipelineState> pso_q8gemm_mma_f16op;
    id<MTLComputePipelineState> pso_f16gemm;
    id<MTLComputePipelineState> pso_q35_conv;
    id<MTLComputePipelineState> pso_q35_conv_shift;
    id<MTLComputePipelineState> pso_q35_l2norm;
    id<MTLComputePipelineState> pso_q35_beta_decay;
    id<MTLComputePipelineState> pso_q35_gdn_step;
    id<MTLComputePipelineState> pso_q35_gated_norm;
    id<MTLComputePipelineState> pso_q35_qg_split;
    id<MTLComputePipelineState> pso_q35_rope_partial;
    id<MTLComputePipelineState> pso_q35_kv_store_f16;
    id<MTLComputePipelineState> pso_q35_attention_f16;
    id<MTLComputePipelineState> pso_q35_attn_gate;
    id<MTLComputePipelineState> pso_q35_silu;
    id<MTLComputePipelineState> pso_q35_split_qkv;
    /* Last committed command buffer (retained); the queue executes in
     * order, so waiting on it fences everything committed before it. */
    id<MTLCommandBuffer> pending;
    /* Open dispatch group (nil when none): operations encode into one
     * command buffer instead of committing per operator. */
    id<MTLCommandBuffer> group_cb;
    id<MTLComputeCommandEncoder> group_enc;
    NSMutableArray *group_keep; /* in-flight resource leases for the group */
    int group_active;
    _Atomic(uint64_t) committed;
    _Atomic(uint64_t) completed;
    _Atomic(uint64_t) last_gpu_nanos;
    /* Cumulative GPU time over every completed command buffer. */
    _Atomic(uint64_t) gpu_nanos_total;
    /* Opt-in per-operator attribution (profiling only; see the header). */
    _Atomic(int) profiling;
    /* Force the generic (device-memory) recurrence kernel; the gate tests
     * compare it against the register-resident specialization. */
    int gdn_chunk_generic;
    os_unfair_lock profile_lock;
    uint32_t profile_used;
    ch_profile_slot profile[CH_PROFILE_SLOTS];
    /* Grouped per-operator attribution (profiling in grouped mode): GPU
     * timestamps sampled at dispatch boundaries inside the shared
     * command buffer, resolved host-side at the fence (Shared storage)
     * and folded into the profile slots. counter_ready is probed at
     * create; paravirtualized GPUs may not expose counter sampling, in
     * which case grouped mode simply has no per-operator data (as
     * before). */
    id<MTLCounterSampleBuffer> counter_buf;
    uint32_t counter_cap;
    uint32_t counter_used;
    uint32_t counter_label_used;
    uint32_t counter_group_base; /* counter_used at begin_group (abort) */
    uint32_t counter_label_base; /* counter_label_used at begin_group */
    int counter_ready;
    int counter_overflow;
    ch_group_label counter_labels[CH_GROUP_LABELS];
    _Atomic(uint64_t) allocated;
    /* Bytes leased by pending command buffers: physically alive until the
     * completion handler runs, even if their handles were already freed.
     * Combined with `allocated` this bounds live GPU bytes from above (a
     * buffer that is both handle-live and in flight is counted twice; the
     * bound is conservative, never understated). */
    _Atomic(uint64_t) leased;
};

struct ch_metal_buffer {
    id<MTLBuffer> buffer; /* retained */
    size_t len;
    int shared;
    ch_metal_ctx *owner; /* not retained; RAII handles order lifetime */
};

static void ch_set_err(char *err, size_t cap, NSString *msg) {
    if (err == NULL || cap == 0) {
        return;
    }
    const char *utf8 = [msg UTF8String];
    snprintf(err, cap, "%s", utf8 != NULL ? utf8 : "unknown metal error");
}

static int ch_fail(char *err, size_t cap, int code, NSString *msg) {
    ch_set_err(err, cap, msg);
    return code;
}

static int ch_check_ctx(ch_metal_ctx *ctx, char *err, size_t cap, const char *what) {
    if (ctx == NULL) {
        return ch_fail(err, cap, CH_METAL_ERR_UNSUPPORTED,
                       [NSString stringWithFormat:@"%s: null context", what]);
    }
    return CH_METAL_OK;
}

/* Validates a buffer handle: owned by this context, non-null, long enough. */
static int ch_check_buffer(ch_metal_ctx *ctx, ch_metal_buffer *b, size_t need,
                           char *err, size_t cap, const char *name) {
    if (b == NULL || b->owner != ctx) {
        return ch_fail(err, cap, CH_METAL_ERR_UNSUPPORTED,
                       [NSString stringWithFormat:@"%s: buffer does not belong to this context", name]);
    }
    if (need == 0 || b->len < need) {
        return ch_fail(err, cap, CH_METAL_ERR_BOUNDS,
                       [NSString stringWithFormat:@"%s: buffer holds %zu bytes but %zu are required",
                                                  name, b->len, need]);
    }
    return CH_METAL_OK;
}

/* Commits a command buffer, retaining `in_flight` (the buffers its work
 * reads or writes) until the command buffer completes. A buffer handle may
 * therefore be freed while its GPU work is still pending: the device
 * resource stays leased until the completion handler runs. Without this,
 * dropping a buffer before an explicit fence could free storage the GPU is
 * still using. */
/* Fold `nanos` into the named profile slot (single context, serialized
 * by the caller). */
static void ch_profile_add(ch_metal_ctx *ctx, const char *name, uint64_t nanos) {
    if (name == NULL || name[0] == '\0') {
        return;
    }
    os_unfair_lock_lock(&ctx->profile_lock);
    uint32_t slot = 0;
    while (slot < ctx->profile_used && strcmp(ctx->profile[slot].name, name) != 0) {
        slot++;
    }
    if (slot == ctx->profile_used && slot < CH_PROFILE_SLOTS) {
        strlcpy(ctx->profile[slot].name, name, CH_PROFILE_NAME);
        ctx->profile_used++;
    }
    if (slot < CH_PROFILE_SLOTS) {
        ctx->profile[slot].nanos += nanos;
        ctx->profile[slot].count += 1;
    }
    os_unfair_lock_unlock(&ctx->profile_lock);
}

/* Encode a counter sample at the next dispatch-boundary slot; returns the
 * slot index, or counter_cap when sampling is off or exhausted. */
static uint32_t ch_counter_sample(ch_metal_ctx *ctx, id<MTLComputeCommandEncoder> enc) {
    if (!atomic_load_explicit(&ctx->profiling, memory_order_relaxed) || !ctx->counter_ready ||
        ctx->counter_used >= ctx->counter_cap) {
        return ctx->counter_cap;
    }
    [enc sampleCountersInBuffer:ctx->counter_buf
                      atSampleIndex:(NSUInteger)ctx->counter_used
                         withBarrier:NO];
    uint32_t idx = ctx->counter_used;
    ctx->counter_used++;
    return idx;
}

static int ch_commit(ch_metal_ctx *ctx, id<MTLCommandBuffer> cb, NSArray *in_flight,
                     const char *what, char *err, size_t cap) {
    (void)err;
    (void)cap;
    ch_label label = {{0}};
    if (atomic_load_explicit(&ctx->profiling, memory_order_relaxed) && what != NULL) {
        strlcpy(label.s, what, sizeof label.s);
    }
    uint64_t leased = 0;
    for (id<MTLBuffer> buffer in in_flight) {
        leased += (uint64_t)buffer.length;
    }
    atomic_fetch_add_explicit(&ctx->leased, leased, memory_order_relaxed);
    atomic_fetch_add_explicit(&ctx->committed, 1, memory_order_relaxed);
    ctx->pending = [cb retain];
    ch_metal_ctx *c = ctx;
    [cb addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        /* Touch the lease array so the block capture keeps it (and every
         * leased MTLBuffer) alive until completion; the array is released
         * with the handler afterwards. */
        (void)in_flight.count;
        atomic_fetch_sub_explicit(&c->leased, leased, memory_order_relaxed);
        double nanos = (buffer.GPUEndTime - buffer.GPUStartTime) * 1e9;
        if (nanos < 0.0) {
            nanos = 0.0;
        }
        atomic_store_explicit(&c->last_gpu_nanos, (uint64_t)nanos, memory_order_relaxed);
        atomic_fetch_add_explicit(&c->gpu_nanos_total, (uint64_t)nanos, memory_order_relaxed);
        if (label.s[0] != '\0') {
            ch_profile_add(c, label.s, (uint64_t)nanos);
        }
        atomic_fetch_add_explicit(&c->completed, 1, memory_order_release);
    }];
    [cb commit];
    return CH_METAL_OK;
}

/* Waits for every committed command buffer. Shared by fence and the copy
 * contract: host copies are ordered after all committed GPU work. */
static int ch_wait_pending(ch_metal_ctx *ctx, char *err, size_t cap) {
    if (ctx->pending == nil) {
        return CH_METAL_OK;
    }
    [ctx->pending waitUntilCompleted];
    if (ctx->pending.status == MTLCommandBufferStatusError) {
        return ch_fail(err, cap, CH_METAL_ERR_INTERNAL,
                       [NSString stringWithFormat:@"command buffer failed: %@",
                                                  ctx->pending.error != nil
                                                      ? [ctx->pending.error localizedDescription]
                                                      : @"unknown"]);
    }
    /* Fold the grouped per-operator deltas: resolved samples hold the GPU
     * clock at each dispatch boundary, and compute dispatches inside one
     * encoder are ordered, so delta = t[s] - t[s-1] is the operator's GPU
     * time. Baseline samples carry no label and are simply skipped. */
    if (ctx->counter_ready && ctx->counter_label_used > 0) {
        @autoreleasepool {
            NSData *resolved = [ctx->counter_buf resolveCounterRange:NSMakeRange(0, ctx->counter_used)];
            if (resolved != nil && resolved.length >= (NSUInteger)ctx->counter_used * sizeof(uint64_t)) {
                const uint64_t *ts = (const uint64_t *)resolved.bytes;
                for (uint32_t i = 0; i < ctx->counter_label_used; i++) {
                    uint32_t s_idx = ctx->counter_labels[i].idx;
                    if (s_idx == 0 || s_idx >= ctx->counter_used) {
                        continue; /* baseline or out of range */
                    }
                    uint64_t end = ts[s_idx];
                    uint64_t begin = ts[s_idx - 1];
                    if (end == MTLCounterErrorValue || begin == MTLCounterErrorValue || end <= begin) {
                        continue; /* unresolved or non-monotonic sample */
                    }
                    ch_profile_add(ctx, ctx->counter_labels[i].name, end - begin);
                }
            }
        }
        ctx->counter_used = 0;
        ctx->counter_label_used = 0;
    }
    /* waitUntilCompleted signals execution completion; addCompletedHandler
     * blocks are scheduled separately and may still be pending. Settlement
     * of the counter/lease accounting is part of the fence contract (the
     * in-flight byte figure must be zero when the fence returns), so wait
     * for the handlers to run. Execution is already complete, so the wait
     * only covers scheduling latency (2 s budget). If it still does not
     * settle, report a typed failure: claiming success here would weaken
     * the acknowledged-fence invariant. */
    for (int spin = 0; spin < 40000; ++spin) {
        if (atomic_load_explicit(&ctx->completed, memory_order_relaxed) >=
            atomic_load_explicit(&ctx->committed, memory_order_relaxed)) {
            return CH_METAL_OK;
        }
        usleep(50);
    }
    return ch_fail(err, cap, CH_METAL_ERR_INTERNAL,
                   @"fence: command buffers completed but completion-handler accounting did not settle");
}

int ch_metal_available(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil ? 1 : 0;
    }
}

const char *ch_metal_runtime_info(void) {
    static char info[256];
    static int filled = 0;
    if (!filled) {
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil) {
                snprintf(info, sizeof(info), "no Metal device available");
            } else {
                snprintf(info, sizeof(info), "%s (recommended GPU working set %llu MiB)",
                         [[device name] UTF8String],
                         (unsigned long long)([device recommendedMaxWorkingSetSize] / (1024ull * 1024ull)));
            }
            filled = 1;
        }
    }
    return info;
}

static id<MTLComputePipelineState> ch_pipeline(ch_metal_ctx *ctx, NSString *name,
                                               NSError **out_error) {
    id<MTLFunction> fn = [ctx->library newFunctionWithName:name];
    if (fn == nil) {
        if (out_error != NULL) {
            *out_error = [NSError errorWithDomain:@"ch_metal"
                                             code:CH_METAL_ERR_SOURCE
                                         userInfo:@{NSLocalizedDescriptionKey:
                                                        [NSString stringWithFormat:@"kernel %@ not found in the MSL source", name]}];
        }
        return nil;
    }
    id<MTLComputePipelineState> pso = [ctx->device newComputePipelineStateWithFunction:fn error:out_error];
    [fn release];
    return pso;
}

ch_metal_ctx *ch_metal_create(const char *msl_source, char *err, size_t err_cap) {
    if (msl_source == NULL) {
        ch_set_err(err, err_cap, @"null MSL source");
        return NULL;
    }
    @autoreleasepool {
        ch_metal_ctx *ctx = calloc(1, sizeof(ch_metal_ctx));
        if (ctx == NULL) {
            ch_set_err(err, err_cap, @"context allocation failed");
            return NULL;
        }
        /* Create-rule: MTLCreateSystemDefaultDevice returns +1. */
        ctx->device = MTLCreateSystemDefaultDevice();
        if (ctx->device == nil) {
            ch_set_err(err, err_cap, @"no Metal device available");
            free(ctx);
            return NULL;
        }
        /* New-rule: +1. */
        ctx->queue = [ctx->device newCommandQueue];
        ctx->counter_ready = 0;
        @autoreleasepool {
            id<MTLCounterSet> ts_set = nil;
            for (id<MTLCounterSet> set in ctx->device.counterSets) {
                if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
                    ts_set = set;
                    break;
                }
            }
            if (ts_set != nil && [ctx->device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]) {
                MTLCounterSampleBufferDescriptor *desc = [[MTLCounterSampleBufferDescriptor alloc] init];
                desc.counterSet = ts_set;
                desc.sampleCount = 16384;
                desc.storageMode = MTLStorageModeShared;
                desc.label = @"tm grouped per-operator profiling";
                NSError *cerr = nil;
                ctx->counter_buf = [ctx->device newCounterSampleBufferWithDescriptor:desc error:&cerr];
                [desc release];
                if (ctx->counter_buf != nil) {
                    ctx->counter_cap = (uint32_t)ctx->counter_buf.sampleCount;
                    ctx->counter_ready = 1;
                }
            }
        }
        NSError *error = nil;
        NSString *source = [NSString stringWithUTF8String:msl_source];
        /* New-rule: +1. */
        ctx->library = [ctx->device newLibraryWithSource:source options:nil error:&error];
        if (ctx->library == nil) {
            ch_set_err(err, err_cap,
                       [NSString stringWithFormat:@"MSL compilation failed: %@",
                                                  error != nil ? [error localizedDescription] : @"unknown"]);
            ch_metal_destroy(ctx);
            return NULL;
        }
        /* ch_pipeline returns +1 (New-rule inside). */
        ctx->pso_scale = ch_pipeline(ctx, @"ch_scale_f32", &error);
        ctx->pso_axpy = ch_pipeline(ctx, @"ch_axpy_f32", &error);
        ctx->pso_copy = ch_pipeline(ctx, @"ch_copy_f32", &error);
        ctx->pso_argmax_partial = ch_pipeline(ctx, @"ch_argmax_partial", &error);
        ctx->pso_argmax_final = ch_pipeline(ctx, @"ch_argmax_final", &error);
        ctx->pso_q8row_indirect = ch_pipeline(ctx, @"ch_q8_0_row_indirect_f32", &error);
        ctx->pso_q35_conv_chunk = ch_pipeline(ctx, @"ch_qwen35_conv_chunk", &error);
        ctx->pso_q35_conv_chunk_shift = ch_pipeline(ctx, @"ch_qwen35_conv_chunk_shift", &error);
        ctx->pso_q35_split_qkv_chunk = ch_pipeline(ctx, @"ch_qwen35_split_qkv_chunk", &error);
        ctx->pso_q35_beta_decay_chunk = ch_pipeline(ctx, @"ch_qwen35_beta_decay_chunk", &error);
        ctx->pso_q35_gdn_chunk = ch_pipeline(ctx, @"ch_qwen35_gdn_chunk", &error);
        ctx->pso_q35_gdn_chunk_k128 = ch_pipeline(ctx, @"ch_qwen35_gdn_chunk_k128", &error);
        ctx->pso_q35_rope_chunk = ch_pipeline(ctx, @"ch_qwen35_rope_partial_chunk", &error);
        ctx->pso_q35_kv_store_chunk = ch_pipeline(ctx, @"ch_qwen35_kv_store_f16_chunk", &error);
        ctx->pso_q35_attention_chunk = ch_pipeline(ctx, @"ch_qwen35_attention_f16_chunk", &error);
        ctx->pso_q35_attn_sk_256x4 = ch_pipeline(ctx, @"ch_qwen35_attn_sk_256x4", &error);
        ctx->pso_q35_attn_sk_16x2 = ch_pipeline(ctx, @"ch_qwen35_attn_sk_16x2", &error);
        ctx->pso_q35_attn_sk_128x2 = ch_pipeline(ctx, @"ch_qwen35_attn_sk_128x2", &error);
        ctx->pso_q35_attn_sk_128x4 = ch_pipeline(ctx, @"ch_qwen35_attn_sk_128x4", &error);
        ctx->pso_q35_attn_sk_f32_16x2 =
            ch_pipeline(ctx, @"ch_qwen35_attn_sk_f32_16x2", &error);
        ctx->pso_q35_attn_sk_f32_128x2 =
            ch_pipeline(ctx, @"ch_qwen35_attn_sk_f32_128x2", &error);
        ctx->pso_q35_attn_sk_f32_128x4 =
            ch_pipeline(ctx, @"ch_qwen35_attn_sk_f32_128x4", &error);
        ctx->pso_q35_attn_sk_merge_256x4 =
            ch_pipeline(ctx, @"ch_qwen35_attn_sk_merge_256x4", &error);
        ctx->pso_q35_attn_sk_merge_16x2 =
            ch_pipeline(ctx, @"ch_qwen35_attn_sk_merge_16x2", &error);
        ctx->pso_q35_attn_sk_merge_128x2 =
            ch_pipeline(ctx, @"ch_qwen35_attn_sk_merge_128x2", &error);
        ctx->pso_q35_attn_sk_merge_128x4 =
            ch_pipeline(ctx, @"ch_qwen35_attn_sk_merge_128x4", &error);
        ctx->pso_q35_attention_gqa_256x4 =
            ch_pipeline(ctx, @"ch_qwen35_attention_gqa_256x4", &error);
        ctx->pso_q35_attention_gqa_16x2 =
            ch_pipeline(ctx, @"ch_qwen35_attention_gqa_16x2", &error);
        ctx->pso_q35_attention_gqa_128x2 =
            ch_pipeline(ctx, @"ch_qwen35_attention_gqa_128x2", &error);
        ctx->pso_q35_attention_gqa_128x4 =
            ch_pipeline(ctx, @"ch_qwen35_attention_gqa_128x4", &error);
        ctx->pso_q35_attention_gqa_f32_16x2 =
            ch_pipeline(ctx, @"ch_qwen35_attention_gqa_f32_16x2", &error);
        ctx->pso_q35_attention_gqa_f32_128x2 =
            ch_pipeline(ctx, @"ch_qwen35_attention_gqa_f32_128x2", &error);
        ctx->pso_q35_attention_gqa_f32_128x4 =
            ch_pipeline(ctx, @"ch_qwen35_attention_gqa_f32_128x4", &error);
        ctx->pso_q35_attention_chunk_f32 =
            ch_pipeline(ctx, @"ch_qwen35_attention_f32_chunk", &error);
        ctx->pso_q35_kv_store_chunk_f32 =
            ch_pipeline(ctx, @"ch_qwen35_kv_store_f32_chunk", &error);
        ctx->pso_f16row_indirect = ch_pipeline(ctx, @"ch_f16_row_indirect_f32", &error);
        ctx->pso_f16f32 = ch_pipeline(ctx, @"ch_cast_f16_f32", &error);
        ctx->pso_f32f16 = ch_pipeline(ctx, @"ch_cast_f32_f16", &error);
        ctx->pso_q8gemv = ch_pipeline(ctx, @"ch_q8_0_gemv", &error);
        ctx->pso_f16gemv = ch_pipeline(ctx, @"ch_f16_gemv", &error);
        ctx->pso_q8row = ch_pipeline(ctx, @"ch_q8_0_row_f32", &error);
        ctx->pso_rmsnorm = ch_pipeline(ctx, @"ch_rmsnorm_f32", &error);
        ctx->pso_norm_add = ch_pipeline(ctx, @"ch_norm_add_f32", &error);
        ctx->pso_qk_norm = ch_pipeline(ctx, @"ch_qk_norm_f32", &error);
        ctx->pso_silu = ch_pipeline(ctx, @"ch_silu_mul_f32", &error);
        ctx->pso_gelu_tanh_mul = ch_pipeline(ctx, @"ch_gelu_tanh_mul_f32", &error);
        ctx->pso_f16row = ch_pipeline(ctx, @"ch_f16_row_f32", &error);
        ctx->pso_q8gemm = ch_pipeline(ctx, @"ch_q8_0_gemm", &error);
        ctx->pso_q8gemm_mma = ch_pipeline(ctx, @"ch_q8_0_gemm_mma", &error);
        ctx->pso_q8gemm_mma_f16op = ch_pipeline(ctx, @"ch_q8_0_gemm_mma_f16op", &error);
        ctx->pso_f16gemm = ch_pipeline(ctx, @"ch_f16_gemm", &error);
        ctx->pso_q35_conv = ch_pipeline(ctx, @"ch_qwen35_conv_step", &error);
        ctx->pso_q35_conv_shift = ch_pipeline(ctx, @"ch_qwen35_conv_shift", &error);
        ctx->pso_q35_l2norm = ch_pipeline(ctx, @"ch_qwen35_l2norm_heads", &error);
        ctx->pso_q35_beta_decay = ch_pipeline(ctx, @"ch_qwen35_beta_decay", &error);
        ctx->pso_q35_gdn_step = ch_pipeline(ctx, @"ch_qwen35_gdn_step", &error);
        ctx->pso_q35_gated_norm = ch_pipeline(ctx, @"ch_qwen35_gated_norm", &error);
        ctx->pso_q35_qg_split = ch_pipeline(ctx, @"ch_qwen35_q_gate_split", &error);
        ctx->pso_q35_rope_partial = ch_pipeline(ctx, @"ch_qwen35_rope_partial_f32", &error);
        ctx->pso_q35_kv_store_f16 = ch_pipeline(ctx, @"ch_qwen35_kv_store_f16", &error);
        ctx->pso_q35_attention_f16 = ch_pipeline(ctx, @"ch_qwen35_attention_f16", &error);
        ctx->pso_q35_attn_gate = ch_pipeline(ctx, @"ch_qwen35_attn_gate", &error);
        ctx->pso_q35_silu = ch_pipeline(ctx, @"ch_qwen35_silu", &error);
        ctx->pso_q35_split_qkv = ch_pipeline(ctx, @"ch_qwen35_split_qkv", &error);
        if (ctx->pso_scale == nil || ctx->pso_axpy == nil || ctx->pso_copy == nil ||
            ctx->pso_argmax_partial == nil || ctx->pso_argmax_final == nil ||
            ctx->pso_q8row_indirect == nil || ctx->pso_f16row_indirect == nil ||
            ctx->pso_q35_conv_chunk == nil || ctx->pso_q35_conv_chunk_shift == nil ||
            ctx->pso_q35_split_qkv_chunk == nil || ctx->pso_q35_beta_decay_chunk == nil ||
            ctx->pso_q35_gdn_chunk == nil || ctx->pso_q35_gdn_chunk_k128 == nil ||
            ctx->pso_q35_rope_chunk == nil ||
            ctx->pso_q35_kv_store_chunk == nil || ctx->pso_q35_attention_chunk == nil ||
            ctx->pso_q35_attn_sk_256x4 == nil || ctx->pso_q35_attn_sk_16x2 == nil ||
            ctx->pso_q35_attn_sk_128x2 == nil || ctx->pso_q35_attn_sk_128x4 == nil ||
            ctx->pso_q35_attn_sk_f32_16x2 == nil || ctx->pso_q35_attn_sk_f32_128x2 == nil ||
            ctx->pso_q35_attn_sk_f32_128x4 == nil || ctx->pso_q35_attn_sk_merge_256x4 == nil ||
            ctx->pso_q35_attn_sk_merge_16x2 == nil ||
            ctx->pso_q35_attn_sk_merge_128x2 == nil ||
            ctx->pso_q35_attn_sk_merge_128x4 == nil ||
            ctx->pso_q35_attention_gqa_256x4 == nil || ctx->pso_q35_attention_gqa_16x2 == nil ||
            ctx->pso_q35_attention_gqa_128x2 == nil ||
            ctx->pso_q35_attention_gqa_128x4 == nil ||
            ctx->pso_q35_attention_gqa_f32_16x2 == nil ||
            ctx->pso_q35_attention_gqa_f32_128x2 == nil ||
            ctx->pso_q35_attention_gqa_f32_128x4 == nil ||
            ctx->pso_q35_attention_chunk_f32 == nil || ctx->pso_q35_kv_store_chunk_f32 == nil ||
            ctx->pso_f16f32 == nil ||
            ctx->pso_f32f16 == nil || ctx->pso_q8gemv == nil || ctx->pso_f16gemv == nil ||
            ctx->pso_q8row == nil || ctx->pso_rmsnorm == nil || ctx->pso_norm_add == nil ||
            ctx->pso_qk_norm == nil || ctx->pso_silu == nil ||
            ctx->pso_f16row == nil || ctx->pso_q8gemm == nil || ctx->pso_q8gemm_mma == nil ||
            ctx->pso_q8gemm_mma_f16op == nil || ctx->pso_f16gemm == nil ||
            ctx->pso_q35_conv == nil || ctx->pso_q35_conv_shift == nil ||
            ctx->pso_q35_l2norm == nil || ctx->pso_q35_beta_decay == nil ||
            ctx->pso_q35_gdn_step == nil || ctx->pso_q35_gated_norm == nil ||
            ctx->pso_q35_qg_split == nil || ctx->pso_q35_rope_partial == nil ||
            ctx->pso_q35_kv_store_f16 == nil || ctx->pso_q35_attention_f16 == nil ||
            ctx->pso_q35_attn_gate == nil || ctx->pso_q35_silu == nil ||
            ctx->pso_q35_split_qkv == nil) {
            ch_set_err(err, err_cap,
                       [NSString stringWithFormat:@"pipeline creation failed: %@",
                                                  error != nil ? [error localizedDescription] : @"unknown"]);
            ch_metal_destroy(ctx);
            return NULL;
        }
        atomic_init(&ctx->committed, 0);
        atomic_init(&ctx->completed, 0);
        atomic_init(&ctx->last_gpu_nanos, 0);
        atomic_init(&ctx->gpu_nanos_total, 0);
        atomic_init(&ctx->profiling, 0);
        ctx->profile_lock = OS_UNFAIR_LOCK_INIT;
        ctx->profile_used = 0;
        atomic_init(&ctx->allocated, 0);
        atomic_init(&ctx->leased, 0);
        return ctx;
    }
}

void ch_metal_destroy(ch_metal_ctx *ctx) {
    if (ctx == NULL) {
        return;
    }
    @autoreleasepool {
        if (ctx->group_active) {
            /* Uncommitted group: discard its work, never abandon silently. */
            [ctx->group_enc endEncoding];
            [ctx->group_enc release];
            [ctx->group_cb release];
            [ctx->group_keep release];
            ctx->group_enc = nil;
            ctx->group_cb = nil;
            ctx->group_keep = nil;
            ctx->group_active = 0;
        }
        if (ctx->pending != nil) {
            [ctx->pending waitUntilCompleted];
            [ctx->pending release];
            ctx->pending = nil;
        }
        uint64_t still_allocated = atomic_load_explicit(&ctx->allocated, memory_order_relaxed);
        if (still_allocated != 0) {
            fprintf(stderr, "ch_metal: %llu bytes still allocated at destroy; buffer lifetime bug\n",
                    (unsigned long long)still_allocated);
        }
        [ctx->counter_buf release];
        [ctx->pso_f16gemm release];
        [ctx->pso_q8gemm release];
        [ctx->pso_q8gemm_mma release];
        [ctx->pso_q8gemm_mma_f16op release];
        [ctx->pso_f16row release];
        [ctx->pso_silu release];
        [ctx->pso_rmsnorm release];
        [ctx->pso_norm_add release];
        [ctx->pso_qk_norm release];
        [ctx->pso_q8row release];
        [ctx->pso_f16gemv release];
        [ctx->pso_q8gemv release];
        [ctx->pso_f32f16 release];
        [ctx->pso_f16f32 release];
        [ctx->pso_axpy release];
        [ctx->pso_copy release];
        [ctx->pso_argmax_partial release];
        [ctx->pso_argmax_final release];
        [ctx->pso_q8row_indirect release];
        [ctx->pso_q35_conv_chunk release];
        [ctx->pso_q35_conv_chunk_shift release];
        [ctx->pso_q35_split_qkv_chunk release];
        [ctx->pso_q35_beta_decay_chunk release];
        [ctx->pso_q35_gdn_chunk release];
        [ctx->pso_q35_gdn_chunk_k128 release];
        [ctx->pso_q35_rope_chunk release];
        [ctx->pso_q35_kv_store_chunk release];
        [ctx->pso_q35_attention_chunk release];
        [ctx->pso_q35_attn_sk_256x4 release];
        [ctx->pso_q35_attn_sk_16x2 release];
        [ctx->pso_q35_attn_sk_128x2 release];
        [ctx->pso_q35_attn_sk_128x4 release];
        [ctx->pso_q35_attn_sk_f32_16x2 release];
        [ctx->pso_q35_attn_sk_f32_128x2 release];
        [ctx->pso_q35_attn_sk_f32_128x4 release];
        [ctx->pso_q35_attn_sk_merge_256x4 release];
        [ctx->pso_q35_attn_sk_merge_16x2 release];
        [ctx->pso_q35_attn_sk_merge_128x2 release];
        [ctx->pso_q35_attn_sk_merge_128x4 release];
        [ctx->pso_q35_attention_gqa_256x4 release];
        [ctx->pso_q35_attention_gqa_16x2 release];
        [ctx->pso_q35_attention_gqa_128x2 release];
        [ctx->pso_q35_attention_gqa_128x4 release];
        [ctx->pso_q35_attention_gqa_f32_16x2 release];
        [ctx->pso_q35_attention_gqa_f32_128x2 release];
        [ctx->pso_q35_attention_gqa_f32_128x4 release];
        [ctx->pso_q35_attention_chunk_f32 release];
        [ctx->pso_q35_kv_store_chunk_f32 release];
        [ctx->pso_f16row_indirect release];
        [ctx->pso_scale release];
        [ctx->library release];
        [ctx->queue release];
        [ctx->device release];
        free(ctx);
    }
}

ch_metal_buffer *ch_metal_buffer_new(ch_metal_ctx *ctx, size_t bytes, int shared,
                                     char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "buffer_new") != CH_METAL_OK) {
        return NULL;
    }
    if (bytes == 0) {
        ch_set_err(err, err_cap, @"buffer_new: zero-length allocation");
        return NULL;
    }
    @autoreleasepool {
        MTLResourceOptions options = shared ? MTLResourceStorageModeShared
                                            : MTLResourceStorageModePrivate;
        id<MTLBuffer> buffer = [ctx->device newBufferWithLength:bytes options:options];
        if (buffer == nil) {
            ch_set_err(err, err_cap,
                       [NSString stringWithFormat:@"buffer_new: allocation of %zu bytes failed", bytes]);
            return NULL;
        }
        ch_metal_buffer *handle = calloc(1, sizeof(ch_metal_buffer));
        if (handle == NULL) {
            [buffer release];
            ch_set_err(err, err_cap, @"buffer_new: handle allocation failed");
            return NULL;
        }
        handle->buffer = buffer; /* +1 from newBufferWithLength */
        handle->len = bytes;
        handle->shared = shared ? 1 : 0;
        handle->owner = ctx;
        atomic_fetch_add_explicit(&ctx->allocated, bytes, memory_order_relaxed);
        return handle;
    }
}

void ch_metal_buffer_free(ch_metal_ctx *ctx, ch_metal_buffer *buffer) {
    if (ctx == NULL || buffer == NULL) {
        return;
    }
    if (buffer->owner != ctx) {
        fprintf(stderr, "ch_metal: buffer_free from a different context refused\n");
        return;
    }
    @autoreleasepool {
        atomic_fetch_sub_explicit(&ctx->allocated, buffer->len, memory_order_relaxed);
        [buffer->buffer release];
        buffer->buffer = nil;
        free(buffer);
    }
}

void *ch_metal_buffer_contents(ch_metal_ctx *ctx, ch_metal_buffer *buffer) {
    if (ctx == NULL || buffer == NULL || buffer->owner != ctx || !buffer->shared) {
        return NULL;
    }
    return [buffer->buffer contents];
}

size_t ch_metal_buffer_len(const ch_metal_ctx *ctx, const ch_metal_buffer *buffer) {
    if (ctx == NULL || buffer == NULL || buffer->owner != (ch_metal_ctx *)ctx) {
        return 0;
    }
    return buffer->len;
}

/*
 * Copy contract: host<->device copies on shared buffers are ordered after
 * every committed GPU command buffer (the call waits for completion first).
 * Reads therefore observe finished compute, and writes cannot race
 * in-flight work. Explicit ch_metal_fence() checkpoints remain available
 * for callers that want to pay the wait once instead of per copy.
 */
static int ch_copy(ch_metal_ctx *ctx, ch_metal_buffer *buffer, size_t off,
                   void *host, size_t len, int to_device, char *err, size_t err_cap,
                   const char *what) {
    if (ch_check_ctx(ctx, err, err_cap, what) != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    /* Validate the handle (null, ownership) before touching its fields. */
    if (buffer == NULL || buffer->owner != ctx) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       [NSString stringWithFormat:@"%s: buffer does not belong to this context", what]);
    }
    if (len == 0 || off > buffer->len || len > buffer->len - off) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS,
                       [NSString stringWithFormat:@"%s: range %zu..%zu outside buffer", what, off, off + len]);
    }
    if (!buffer->shared) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       [NSString stringWithFormat:@"%s: private buffers have no host copy", what]);
    }
    if (ctx->group_active) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       [NSString stringWithFormat:@"%s: close the open dispatch group first", what]);
    }
    if (ch_wait_pending(ctx, err, err_cap) != CH_METAL_OK) {
        return CH_METAL_ERR_INTERNAL;
    }
    void *contents = [buffer->buffer contents];
    if (to_device) {
        memcpy((uint8_t *)contents + off, host, len);
    } else {
        memcpy(host, (const uint8_t *)contents + off, len);
    }
    return CH_METAL_OK;
}

int ch_metal_upload(ch_metal_ctx *ctx, ch_metal_buffer *dst, size_t off,
                    const void *src, size_t len, char *err, size_t err_cap) {
    return ch_copy(ctx, dst, off, (void *)src, len, 1, err, err_cap, "upload");
}

int ch_metal_download(ch_metal_ctx *ctx, ch_metal_buffer *src, size_t off,
                      void *dst, size_t len, char *err, size_t err_cap) {
    return ch_copy(ctx, src, off, dst, len, 0, err, err_cap, "download");
}

int ch_metal_download_after(ch_metal_ctx *ctx, ch_metal_buffer *src, size_t off, void *dst,
                            size_t len, uint64_t completed_target, char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "download_after") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (src == NULL || src->owner != ctx) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       @"download_after: buffer does not belong to this context");
    }
    if (len == 0 || off > src->len || len > src->len - off) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS,
                       [NSString stringWithFormat:@"download_after: range %zu..%zu outside buffer", off, off + len]);
    }
    if (!src->shared) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       @"download_after: private buffers have no host copy");
    }
    if (completed_target > atomic_load_explicit(&ctx->committed, memory_order_relaxed)) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       @"download_after: target is beyond the committed command buffers");
    }
    /* Command buffers complete in queue order; wait for the target only,
     * not for later work (the caller guarantees later work does not touch
     * this range). */
    while (atomic_load_explicit(&ctx->completed, memory_order_acquire) < completed_target) {
        usleep(50);
    }
    memcpy(dst, (const uint8_t *)[src->buffer contents] + off, len);
    return CH_METAL_OK;
}

/* ---------------------------------------------------------------- ops -- */

/*
 * Dispatch frames. With no open group, an operation creates and commits its
 * own command buffer (correctness-first; C0 behavior). With an open group
 * (ch_metal_begin_group), operations append to that one command buffer, so
 * a whole transformer step becomes one committed unit with one completion —
 * no per-operator commit overhead.
 */
typedef struct {
    id<MTLCommandBuffer> cb;          /* nil inside an open group */
    id<MTLComputeCommandEncoder> enc;
    NSMutableArray *keep;             /* in-flight resource leases */
    int implicit;
} ch_frame;

static id<MTLComputeCommandEncoder> ch_frame_open(ch_metal_ctx *ctx, ch_frame *f) {
    if (ctx->group_active) {
        f->implicit = 0;
        f->cb = nil;
        f->enc = ctx->group_enc;
        f->keep = ctx->group_keep;
        return f->enc;
    }
    f->implicit = 1;
    f->cb = [ctx->queue commandBuffer];
    f->enc = [f->cb computeCommandEncoder];
    f->keep = [NSMutableArray array];
    return f->enc;
}

static int ch_frame_commit(ch_metal_ctx *ctx, ch_frame *f, char *err, size_t cap,
                           const char *what) {
    if (f->enc == nil) {
        return ch_fail(err, cap, CH_METAL_ERR_INTERNAL,
                       [NSString stringWithFormat:@"%s: command encoder unavailable", what]);
    }
    if (!f->implicit) {
        /* Grouped: no per-buffer commit exists to time, so per-operator
         * attribution samples the GPU clock at this dispatch boundary
         * and folds the delta at the fence. */
        if (atomic_load_explicit(&ctx->profiling, memory_order_relaxed) && ctx->counter_ready &&
            ctx->counter_label_used < CH_GROUP_LABELS) {
            uint32_t idx = ch_counter_sample(ctx, ctx->group_enc);
            if (idx < ctx->counter_cap) {
                strlcpy(ctx->counter_labels[ctx->counter_label_used].name,
                        what != NULL ? what : "op", CH_PROFILE_NAME);
                ctx->counter_labels[ctx->counter_label_used].idx = idx;
                ctx->counter_label_used++;
            } else {
                ctx->counter_overflow = 1;
            }
        }
        return CH_METAL_OK; /* the open group commits once at end_group */
    }
    [f->enc endEncoding];
    return ch_commit(ctx, f->cb, f->keep, what, err, cap);
}

int ch_metal_begin_group(ch_metal_ctx *ctx, char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "begin_group") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (ctx->group_active) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       @"begin_group: a dispatch group is already open");
    }
    @autoreleasepool {
        ctx->group_cb = [[ctx->queue commandBuffer] retain];
        ctx->group_enc = [[ctx->group_cb computeCommandEncoder] retain];
        ctx->group_keep = [[NSMutableArray alloc] init];
        if (ctx->group_cb == nil || ctx->group_enc == nil || ctx->group_keep == nil) {
            if (ctx->group_enc != nil) { [ctx->group_enc endEncoding]; }
            [ctx->group_enc release];
            [ctx->group_cb release];
            [ctx->group_keep release];
            ctx->group_enc = nil;
            ctx->group_cb = nil;
            ctx->group_keep = nil;
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL,
                           @"begin_group: command buffering unavailable");
        }
        ctx->group_active = 1;
        /* Profiling baseline: the first operator's delta starts here, so
         * host time between groups never lands in any label. */
        ctx->counter_group_base = ctx->counter_used;
        ctx->counter_label_base = ctx->counter_label_used;
        ch_counter_sample(ctx, ctx->group_enc);
        return CH_METAL_OK;
    }
}

int ch_metal_end_group(ch_metal_ctx *ctx, char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "end_group") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (!ctx->group_active) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       @"end_group: no dispatch group is open");
    }
    @autoreleasepool {
        [ctx->group_enc endEncoding];
        id<MTLCommandBuffer> cb = ctx->group_cb;
        NSMutableArray *keep = ctx->group_keep;
        id<MTLComputeCommandEncoder> enc = ctx->group_enc;
        ctx->group_cb = nil;
        ctx->group_enc = nil;
        ctx->group_keep = nil;
        ctx->group_active = 0;
        int rc = ch_commit(ctx, cb, keep, "group", err, err_cap);
        [enc release];
        [cb release];
        [keep release];
        return rc;
    }
}

/* Discards the open group's work without committing it (cancellation at a
 * step boundary: no partial GPU work is ever submitted). */
int ch_metal_abort_group(ch_metal_ctx *ctx, char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "abort_group") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (!ctx->group_active) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       @"abort_group: no dispatch group is open");
    }
    [ctx->group_enc endEncoding];
    [ctx->group_enc release];
    [ctx->group_cb release];
    [ctx->group_keep release];
    ctx->group_enc = nil;
    ctx->group_cb = nil;
    ctx->group_keep = nil;
    ctx->group_active = 0;
    /* Samples encoded into the discarded buffer never execute; drop their
     * bookkeeping so the next resolve stays consistent. */
    ctx->counter_used = ctx->counter_group_base;
    ctx->counter_label_used = ctx->counter_label_base;
    return CH_METAL_OK;
}

static int ch_dispatch_1d(ch_metal_ctx *ctx, id<MTLComputePipelineState> pso,
                          ch_metal_buffer *a, ch_metal_buffer *b, float scalar,
                          int has_scalar, uint32_t n, int cast_f16_in, int cast_f16_out,
                          char *err, size_t err_cap, const char *what) {
    if (ch_check_ctx(ctx, err, err_cap, what) != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (n == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS,
                       [NSString stringWithFormat:@"%s: zero length", what]);
    }
    size_t a_need = (size_t)n * (cast_f16_in ? 2 : 4);
    size_t b_need = (size_t)n * (cast_f16_out ? 2 : 4);
    if (ch_check_buffer(ctx, a, a_need, err, err_cap, what) != CH_METAL_OK ||
        ch_check_buffer(ctx, b, b_need, err, err_cap, what) != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL,
                           [NSString stringWithFormat:@"%s: command encoder unavailable", what]);
        }
        [enc setComputePipelineState:pso];
        [enc setBuffer:a->buffer offset:0 atIndex:0];
        [enc setBuffer:b->buffer offset:0 atIndex:1];
        [f.keep addObject:a->buffer];
        [f.keep addObject:b->buffer];
        uint32_t index = 2;
        if (has_scalar) {
            [enc setBytes:&scalar length:sizeof(scalar) atIndex:index++];
        }
        [enc setBytes:&n length:sizeof(n) atIndex:index];
        NSUInteger width = pso.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width : 32;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, what);
    }
}

int ch_metal_f32_scale(ch_metal_ctx *ctx, ch_metal_buffer *x, ch_metal_buffer *y,
                       float factor, uint32_t n, char *err, size_t err_cap) {
    return ch_dispatch_1d(ctx, ctx->pso_scale, x, y, factor, 1, n, 0, 0, err, err_cap,
                          "f32_scale");
}

int ch_metal_f32_axpy(ch_metal_ctx *ctx, ch_metal_buffer *a, ch_metal_buffer *y,
                      float alpha, uint32_t n, char *err, size_t err_cap) {
    return ch_dispatch_1d(ctx, ctx->pso_axpy, a, y, alpha, 1, n, 0, 0, err, err_cap,
                          "f32_axpy");
}

int ch_metal_argmax_f32(ch_metal_ctx *ctx, ch_metal_buffer *x, uint32_t n,
                        ch_metal_buffer *partial, ch_metal_buffer *out, uint32_t out_index,
                        char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "argmax_f32") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (n == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS, @"argmax_f32: empty input");
    }
    /* partial holds 64 (value, index) pairs: 64 f32 then 64 u32. */
    if (ch_check_buffer(ctx, x, (size_t)n * 4u, err, err_cap, "argmax_f32.x") != CH_METAL_OK ||
        ch_check_buffer(ctx, partial, CH_ARGMAX_PARTIAL_BYTES, err, err_cap,
                        "argmax_f32.partial") != CH_METAL_OK ||
        ch_check_buffer(ctx, out, ((size_t)out_index + 1u) * 4u, err, err_cap,
                        "argmax_f32.out") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"argmax_f32: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_argmax_partial];
        [enc setBuffer:x->buffer offset:0 atIndex:0];
        [enc setBuffer:partial->buffer offset:0 atIndex:1];
        [enc setBuffer:partial->buffer offset:256 atIndex:2];
        [enc setBytes:&n length:sizeof(n) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(64, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc setComputePipelineState:ctx->pso_argmax_final];
        [enc setBuffer:partial->buffer offset:0 atIndex:0];
        [enc setBuffer:partial->buffer offset:256 atIndex:1];
        [enc setBuffer:out->buffer offset:0 atIndex:2];
        [enc setBytes:&out_index length:sizeof(out_index) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [f.keep addObject:x->buffer];
        [f.keep addObject:partial->buffer];
        [f.keep addObject:out->buffer];
        return ch_frame_commit(ctx, &f, err, err_cap, "argmax_f32");
    }
}

int ch_metal_row_indirect_f32(ch_metal_ctx *ctx, ch_metal_buffer *w, uint32_t type_id,
                              uint32_t rows, uint32_t cols, ch_metal_buffer *ids, uint32_t slot,
                              ch_metal_buffer *out, char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "row_indirect") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    id<MTLComputePipelineState> pso = nil;
    size_t w_need = 0;
    if (type_id == 8u && cols > 0 && (cols % 32u) == 0u) {
        pso = ctx->pso_q8row_indirect;
        w_need = (size_t)rows * ((size_t)cols / 32u) * 34u;
    } else if (type_id == 1u && cols > 0) {
        pso = ctx->pso_f16row_indirect;
        w_need = (size_t)rows * (size_t)cols * 2u;
    } else {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       [NSString stringWithFormat:@"row_indirect: type %u with cols %u is unsupported",
                                                  type_id, cols]);
    }
    if (rows == 0 ||
        ch_check_buffer(ctx, w, w_need, err, err_cap, "row_indirect.weights") != CH_METAL_OK ||
        ch_check_buffer(ctx, ids, ((size_t)slot + 1u) * 4u, err, err_cap, "row_indirect.ids") != CH_METAL_OK ||
        ch_check_buffer(ctx, out, (size_t)cols * 4u, err, err_cap, "row_indirect.out") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"row_indirect: no encoder");
        }
        [enc setComputePipelineState:pso];
        [enc setBuffer:w->buffer offset:0 atIndex:0];
        [enc setBuffer:out->buffer offset:0 atIndex:1];
        [enc setBuffer:ids->buffer offset:0 atIndex:2];
        [enc setBytes:&slot length:sizeof(slot) atIndex:3];
        [enc setBytes:&rows length:sizeof(rows) atIndex:4];
        [enc setBytes:&cols length:sizeof(cols) atIndex:5];
        [f.keep addObject:w->buffer];
        [f.keep addObject:out->buffer];
        [f.keep addObject:ids->buffer];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)cols, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, "row_indirect");
    }
}

int ch_metal_copy_f32(ch_metal_ctx *ctx, ch_metal_buffer *src, uint32_t src_off,
                      ch_metal_buffer *dst, uint32_t dst_off, uint32_t n, char *err,
                      size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "copy_f32") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (n == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS, @"copy_f32: zero-length copy");
    }
    size_t src_need = ((size_t)src_off + (size_t)n) * 4u;
    size_t dst_need = ((size_t)dst_off + (size_t)n) * 4u;
    if (ch_check_buffer(ctx, src, src_need, err, err_cap, "copy_f32.src") != CH_METAL_OK ||
        ch_check_buffer(ctx, dst, dst_need, err, err_cap, "copy_f32.dst") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"copy_f32: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_copy];
        [enc setBuffer:src->buffer offset:0 atIndex:0];
        [enc setBuffer:dst->buffer offset:0 atIndex:1];
        [f.keep addObject:src->buffer];
        [f.keep addObject:dst->buffer];
        [enc setBytes:&src_off length:sizeof(src_off) atIndex:2];
        [enc setBytes:&dst_off length:sizeof(dst_off) atIndex:3];
        [enc setBytes:&n length:sizeof(n) atIndex:4];
        NSUInteger width = ctx->pso_copy.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width * 4 : 128;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, "copy_f32");
    }
}

int ch_metal_cast_f16_f32(ch_metal_ctx *ctx, ch_metal_buffer *x, ch_metal_buffer *y,
                          uint32_t n, char *err, size_t err_cap) {
    return ch_dispatch_1d(ctx, ctx->pso_f16f32, x, y, 0.0f, 0, n, 1, 0, err, err_cap,
                          "cast_f16_f32");
}

int ch_metal_cast_f32_f16(ch_metal_ctx *ctx, ch_metal_buffer *x, ch_metal_buffer *y,
                          uint32_t n, char *err, size_t err_cap) {
    return ch_dispatch_1d(ctx, ctx->pso_f32f16, x, y, 0.0f, 0, n, 0, 1, err, err_cap,
                          "cast_f32_f16");
}

int ch_metal_q8_0_gemv(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                       ch_metal_buffer *y, uint32_t rows, uint32_t cols,
                       char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "q8_0_gemv") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (rows == 0 || cols == 0 || (cols % 32u) != 0u) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       [NSString stringWithFormat:@"q8_0_gemv: rows %u, cols %u (cols must be a positive multiple of 32)",
                                                  rows, cols]);
    }
    size_t w_need = (size_t)rows * ((size_t)cols / 32u) * 34u;
    size_t x_need = (size_t)cols * 4u;
    size_t y_need = (size_t)rows * 4u;
    if (ch_check_buffer(ctx, w, w_need, err, err_cap, "q8_0_gemv.weights") != CH_METAL_OK ||
        ch_check_buffer(ctx, x, x_need, err, err_cap, "q8_0_gemv.x") != CH_METAL_OK ||
        ch_check_buffer(ctx, y, y_need, err, err_cap, "q8_0_gemv.y") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL,
                           @"q8_0_gemv: command encoder unavailable");
        }
        [enc setComputePipelineState:ctx->pso_q8gemv];
        [enc setBuffer:w->buffer offset:0 atIndex:0];
        [enc setBuffer:x->buffer offset:0 atIndex:1];
        [enc setBuffer:y->buffer offset:0 atIndex:2];
        [f.keep addObject:w->buffer];
        [f.keep addObject:x->buffer];
        [f.keep addObject:y->buffer];
        [enc setBytes:&rows length:sizeof(rows) atIndex:3];
        [enc setBytes:&cols length:sizeof(cols) atIndex:4];
        NSUInteger lanes = ctx->pso_q8gemv.threadExecutionWidth;
        if (lanes != 32) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL,
                           [NSString stringWithFormat:@"q8_0_gemv: device simd width %lu != 32", (unsigned long)lanes]);
        }
        /* Multi-row kernel: 2 SIMD groups x 4 rows per threadgroup (the
         * CH_Q8_SG / CH_Q8_ROWS constants in kernels.metal). */
        const NSUInteger rows_per_tg = 8;
        NSUInteger groups = ((NSUInteger)rows + rows_per_tg - 1) / rows_per_tg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        /* Profiling attributes GEMV time per matrix shape. */
        char label[CH_PROFILE_NAME];
        if (atomic_load_explicit(&ctx->profiling, memory_order_relaxed)) {
            snprintf(label, sizeof label, "q8_0_gemv:%ux%u", rows, cols);
        } else {
            strlcpy(label, "q8_0_gemv", sizeof label);
        }
        return ch_frame_commit(ctx, &f, err, err_cap, label);
    }
}

int ch_metal_f16_gemv(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                      ch_metal_buffer *y, uint32_t rows, uint32_t cols,
                      char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "f16_gemv") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (rows == 0 || cols == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED, @"f16_gemv: zero shape");
    }
    size_t w_need = (size_t)rows * (size_t)cols * 2u;
    size_t x_need = (size_t)cols * 4u;
    size_t y_need = (size_t)rows * 4u;
    if (ch_check_buffer(ctx, w, w_need, err, err_cap, "f16_gemv.weights") != CH_METAL_OK ||
        ch_check_buffer(ctx, x, x_need, err, err_cap, "f16_gemv.x") != CH_METAL_OK ||
        ch_check_buffer(ctx, y, y_need, err, err_cap, "f16_gemv.y") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL,
                           @"f16_gemv: command encoder unavailable");
        }
        [enc setComputePipelineState:ctx->pso_f16gemv];
        [enc setBuffer:w->buffer offset:0 atIndex:0];
        [enc setBuffer:x->buffer offset:0 atIndex:1];
        [enc setBuffer:y->buffer offset:0 atIndex:2];
        [f.keep addObject:w->buffer];
        [f.keep addObject:x->buffer];
        [f.keep addObject:y->buffer];
        [enc setBytes:&rows length:sizeof(rows) atIndex:3];
        [enc setBytes:&cols length:sizeof(cols) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, "f16_gemv");
    }
}

int ch_metal_q8_0_row_f32(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *out,
                          uint32_t row, uint32_t cols, uint32_t out_off, char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "q8_0_row") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (cols == 0 || (cols % 32u) != 0u) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       @"q8_0_row: cols must be a positive multiple of 32");
    }
    size_t w_need = ((size_t)row + 1u) * ((size_t)cols / 32u) * 34u;
    if (ch_check_buffer(ctx, w, w_need, err, err_cap, "q8_0_row.weights") != CH_METAL_OK ||
        ch_check_buffer(ctx, out, ((size_t)out_off + cols) * 4u, err, err_cap, "q8_0_row.out") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL,
                           @"q8_0_row: command encoder unavailable");
        }
        [enc setComputePipelineState:ctx->pso_q8row];
        [enc setBuffer:w->buffer offset:0 atIndex:0];
        [enc setBuffer:out->buffer offset:0 atIndex:1];
        [f.keep addObject:w->buffer];
        [f.keep addObject:out->buffer];
        [enc setBytes:&row length:sizeof(row) atIndex:2];
        [enc setBytes:&cols length:sizeof(cols) atIndex:3];
        [enc setBytes:&out_off length:sizeof(out_off) atIndex:4];
        NSUInteger tg = 256;
        NSUInteger width = ctx->pso_q8row.threadExecutionWidth;
        if (width > 0 && width < tg) { tg = width * 8; }
        [enc dispatchThreads:MTLSizeMake((NSUInteger)cols, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, "q8_0_row");
    }
}

static int ch_rmsnorm_dispatch(ch_metal_ctx *ctx, id<MTLComputePipelineState> pso,
                               ch_metal_buffer *x, ch_metal_buffer *weight,
                               ch_metal_buffer *y, uint32_t rows, uint32_t cols, float eps,
                               size_t weight_window, char *err, size_t err_cap,
                               const char *what) {
    if (ch_check_ctx(ctx, err, err_cap, what) != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (rows == 0 || cols == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       [NSString stringWithFormat:@"%s: zero shape", what]);
    }
    size_t x_need = (size_t)rows * (size_t)cols * 4u;
    if (ch_check_buffer(ctx, x, x_need, err, err_cap, "rmsnorm.x") != CH_METAL_OK ||
        ch_check_buffer(ctx, weight, (size_t)cols * weight_window, err, err_cap, "rmsnorm.weight") != CH_METAL_OK ||
        ch_check_buffer(ctx, y, x_need, err, err_cap, "rmsnorm.y") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL,
                           [NSString stringWithFormat:@"%s: no encoder", what]);
        }
        [enc setComputePipelineState:pso];
        [enc setBuffer:x->buffer offset:0 atIndex:0];
        [enc setBuffer:weight->buffer offset:0 atIndex:1];
        [enc setBuffer:y->buffer offset:0 atIndex:2];
        [f.keep addObject:x->buffer];
        [f.keep addObject:weight->buffer];
        [f.keep addObject:y->buffer];
        [enc setBytes:&cols length:sizeof(cols) atIndex:3];
        [enc setBytes:&eps length:sizeof(eps) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, what);
    }
}

int ch_metal_rmsnorm_f32(ch_metal_ctx *ctx, ch_metal_buffer *x, ch_metal_buffer *weight,
                         ch_metal_buffer *y, uint32_t rows, uint32_t cols, float eps,
                         char *err, size_t err_cap) {
    return ch_rmsnorm_dispatch(ctx, ctx->pso_rmsnorm, x, weight, y, rows, cols, eps, 4,
                               err, err_cap, "rmsnorm");
}

int ch_metal_norm_add_f32(ch_metal_ctx *ctx, ch_metal_buffer *x, ch_metal_buffer *weight,
                           ch_metal_buffer *res, ch_metal_buffer *y, uint32_t rows,
                           uint32_t cols, float eps, char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "norm_add") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (rows == 0 || cols == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED, @"norm_add: zero shape");
    }
    size_t need = (size_t)rows * (size_t)cols * 4u;
    if (ch_check_buffer(ctx, x, need, err, err_cap, "norm_add.x") != CH_METAL_OK ||
        ch_check_buffer(ctx, weight, (size_t)cols * 4u, err, err_cap, "norm_add.weight") != CH_METAL_OK ||
        ch_check_buffer(ctx, res, need, err, err_cap, "norm_add.res") != CH_METAL_OK ||
        ch_check_buffer(ctx, y, need, err, err_cap, "norm_add.y") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"norm_add: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_norm_add];
        [enc setBuffer:x->buffer offset:0 atIndex:0];
        [enc setBuffer:weight->buffer offset:0 atIndex:1];
        [enc setBuffer:res->buffer offset:0 atIndex:2];
        [enc setBuffer:y->buffer offset:0 atIndex:3];
        [f.keep addObject:x->buffer];
        [f.keep addObject:weight->buffer];
        [f.keep addObject:res->buffer];
        [f.keep addObject:y->buffer];
        [enc setBytes:&cols length:sizeof(cols) atIndex:4];
        [enc setBytes:&eps length:sizeof(eps) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, "norm_add");
    }
}

int ch_metal_qk_norm_f32(ch_metal_ctx *ctx, ch_metal_buffer *q, ch_metal_buffer *q_weight,
                          ch_metal_buffer *q_out, ch_metal_buffer *k, ch_metal_buffer *k_weight,
                          ch_metal_buffer *k_out, uint32_t q_rows, uint32_t k_rows, uint32_t cols,
                          float eps, char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "qk_norm") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (q_rows == 0 || k_rows == 0 || cols == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED, @"qk_norm: zero shape");
    }
    size_t q_need = (size_t)q_rows * (size_t)cols * 4u;
    size_t k_need = (size_t)k_rows * (size_t)cols * 4u;
    if (ch_check_buffer(ctx, q, q_need, err, err_cap, "qk_norm.q") != CH_METAL_OK ||
        ch_check_buffer(ctx, q_weight, (size_t)cols * 4u, err, err_cap, "qk_norm.q_weight") != CH_METAL_OK ||
        ch_check_buffer(ctx, q_out, q_need, err, err_cap, "qk_norm.q_out") != CH_METAL_OK ||
        ch_check_buffer(ctx, k, k_need, err, err_cap, "qk_norm.k") != CH_METAL_OK ||
        ch_check_buffer(ctx, k_weight, (size_t)cols * 4u, err, err_cap, "qk_norm.k_weight") != CH_METAL_OK ||
        ch_check_buffer(ctx, k_out, k_need, err, err_cap, "qk_norm.k_out") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"qk_norm: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_qk_norm];
        [enc setBuffer:q->buffer offset:0 atIndex:0];
        [enc setBuffer:q_weight->buffer offset:0 atIndex:1];
        [enc setBuffer:q_out->buffer offset:0 atIndex:2];
        [enc setBuffer:k->buffer offset:0 atIndex:3];
        [enc setBuffer:k_weight->buffer offset:0 atIndex:4];
        [enc setBuffer:k_out->buffer offset:0 atIndex:5];
        [f.keep addObject:q->buffer];
        [f.keep addObject:q_weight->buffer];
        [f.keep addObject:q_out->buffer];
        [f.keep addObject:k->buffer];
        [f.keep addObject:k_weight->buffer];
        [f.keep addObject:k_out->buffer];
        [enc setBytes:&q_rows length:sizeof(q_rows) atIndex:6];
        [enc setBytes:&k_rows length:sizeof(k_rows) atIndex:7];
        [enc setBytes:&cols length:sizeof(cols) atIndex:8];
        [enc setBytes:&eps length:sizeof(eps) atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(q_rows + k_rows), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, "qk_norm");
    }
}

int ch_metal_silu_mul_f32(ch_metal_ctx *ctx, ch_metal_buffer *gate, ch_metal_buffer *up,
                          ch_metal_buffer *out, uint32_t n, char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "silu_mul") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (n == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS, @"silu_mul: zero length");
    }
    size_t need = (size_t)n * 4u;
    if (ch_check_buffer(ctx, gate, need, err, err_cap, "silu_mul.gate") != CH_METAL_OK ||
        ch_check_buffer(ctx, up, need, err, err_cap, "silu_mul.up") != CH_METAL_OK ||
        ch_check_buffer(ctx, out, need, err, err_cap, "silu_mul.out") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"silu_mul: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_silu];
        [enc setBuffer:gate->buffer offset:0 atIndex:0];
        [enc setBuffer:up->buffer offset:0 atIndex:1];
        [enc setBuffer:out->buffer offset:0 atIndex:2];
        [f.keep addObject:gate->buffer];
        [f.keep addObject:up->buffer];
        [f.keep addObject:out->buffer];
        [enc setBytes:&n length:sizeof(n) atIndex:3];
        NSUInteger width = ctx->pso_silu.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width * 8 : 256;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, "silu_mul");
    }
}

int ch_metal_gelu_tanh_mul_f32(ch_metal_ctx *ctx, ch_metal_buffer *gate, ch_metal_buffer *up,
                          ch_metal_buffer *out, uint32_t n, char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "gelu_tanh_mul") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (n == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS, @"gelu_tanh_mul: zero length");
    }
    size_t need = (size_t)n * 4u;
    if (ch_check_buffer(ctx, gate, need, err, err_cap, "gelu_tanh_mul.gate") != CH_METAL_OK ||
        ch_check_buffer(ctx, up, need, err, err_cap, "gelu_tanh_mul.up") != CH_METAL_OK ||
        ch_check_buffer(ctx, out, need, err, err_cap, "gelu_tanh_mul.out") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"gelu_tanh_mul: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_gelu_tanh_mul];
        [enc setBuffer:gate->buffer offset:0 atIndex:0];
        [enc setBuffer:up->buffer offset:0 atIndex:1];
        [enc setBuffer:out->buffer offset:0 atIndex:2];
        [f.keep addObject:gate->buffer];
        [f.keep addObject:up->buffer];
        [f.keep addObject:out->buffer];
        [enc setBytes:&n length:sizeof(n) atIndex:3];
        NSUInteger width = ctx->pso_gelu_tanh_mul.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width * 8 : 256;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, "gelu_tanh_mul");
    }
}

int ch_metal_f16_row_f32(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *out,
                         uint32_t row, uint32_t cols, uint32_t out_off, char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "f16_row") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (cols == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED, @"f16_row: zero cols");
    }
    size_t w_need = ((size_t)row + 1u) * (size_t)cols * 2u;
    if (ch_check_buffer(ctx, w, w_need, err, err_cap, "f16_row.weights") != CH_METAL_OK ||
        ch_check_buffer(ctx, out, ((size_t)out_off + cols) * 4u, err, err_cap, "f16_row.out") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"f16_row: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_f16row];
        [enc setBuffer:w->buffer offset:0 atIndex:0];
        [enc setBuffer:out->buffer offset:0 atIndex:1];
        [f.keep addObject:w->buffer];
        [f.keep addObject:out->buffer];
        [enc setBytes:&row length:sizeof(row) atIndex:2];
        [enc setBytes:&cols length:sizeof(cols) atIndex:3];
        [enc setBytes:&out_off length:sizeof(out_off) atIndex:4];
        NSUInteger tg = 256;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)cols, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, "f16_row");
    }
}

static int ch_gemm_dispatch_rows(ch_metal_ctx *ctx, id<MTLComputePipelineState> pso,
                                 ch_metal_buffer *w, size_t w_need, ch_metal_buffer *x,
                                 ch_metal_buffer *y, uint32_t rows, uint32_t cols,
                                 uint32_t batch, uint32_t x_off, uint32_t y_off,
                                 NSUInteger rows_per_tg, NSUInteger threads,
                                 uint32_t max_batch, NSUInteger tokens_per_tg, char *err,
                                 size_t err_cap, const char *what);

static int ch_gemm_dispatch(ch_metal_ctx *ctx, id<MTLComputePipelineState> pso,
                            ch_metal_buffer *w, size_t w_need, ch_metal_buffer *x,
                            ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                            uint32_t x_off, uint32_t y_off, char *err, size_t err_cap,
                            const char *what) {
    return ch_gemm_dispatch_rows(ctx, pso, w, w_need, x, y, rows, cols, batch, x_off, y_off, 1,
                                 32, 32, 0, err, err_cap, what);
}

/* One threadgroup of `threads` lanes per `rows_per_tg` output rows and,
 * when `tokens_per_tg` is nonzero, per `tokens_per_tg` tokens (a 1-D grid
 * of row tiles x token tiles; the kernel decodes the pair). */
static int ch_gemm_dispatch_rows(ch_metal_ctx *ctx, id<MTLComputePipelineState> pso,
                                 ch_metal_buffer *w, size_t w_need, ch_metal_buffer *x,
                                 ch_metal_buffer *y, uint32_t rows, uint32_t cols,
                                 uint32_t batch, uint32_t x_off, uint32_t y_off,
                                 NSUInteger rows_per_tg, NSUInteger threads,
                                 uint32_t max_batch, NSUInteger tokens_per_tg, char *err,
                                 size_t err_cap, const char *what) {
    if (ch_check_ctx(ctx, err, err_cap, what) != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (rows == 0 || cols == 0 || batch == 0 || batch > max_batch) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       [NSString stringWithFormat:@"%s: rows %u, cols %u, batch %u (1..=%u)",
                                                  what, rows, cols, batch, max_batch]);
    }
    size_t x_need = ((size_t)x_off + (size_t)batch * (size_t)cols) * 4u;
    size_t y_need = ((size_t)y_off + (size_t)batch * (size_t)rows) * 4u;
    if (ch_check_buffer(ctx, w, w_need, err, err_cap, "gemm.weights") != CH_METAL_OK ||
        ch_check_buffer(ctx, x, x_need, err, err_cap, "gemm.x") != CH_METAL_OK ||
        ch_check_buffer(ctx, y, y_need, err, err_cap, "gemm.y") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL,
                           [NSString stringWithFormat:@"%s: no encoder", what]);
        }
        [enc setComputePipelineState:pso];
        [enc setBuffer:w->buffer offset:0 atIndex:0];
        [enc setBuffer:x->buffer offset:0 atIndex:1];
        [enc setBuffer:y->buffer offset:0 atIndex:2];
        [f.keep addObject:w->buffer];
        [f.keep addObject:x->buffer];
        [f.keep addObject:y->buffer];
        [enc setBytes:&rows length:sizeof(rows) atIndex:3];
        [enc setBytes:&cols length:sizeof(cols) atIndex:4];
        [enc setBytes:&batch length:sizeof(batch) atIndex:5];
        [enc setBytes:&x_off length:sizeof(x_off) atIndex:6];
        [enc setBytes:&y_off length:sizeof(y_off) atIndex:7];
        NSUInteger groups = ((NSUInteger)rows + rows_per_tg - 1) / rows_per_tg;
        NSUInteger token_groups =
            tokens_per_tg == 0 ? 1 : ((NSUInteger)batch + tokens_per_tg - 1) / tokens_per_tg;
        [enc dispatchThreadgroups:MTLSizeMake(groups * token_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, what);
    }
}

int ch_metal_q8_0_gemm_mma(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                           ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                           uint32_t x_off, uint32_t y_off, char *err, size_t err_cap) {
    if (cols == 0 || (cols % 32u) != 0u || batch == 0 || batch > CH_METAL_GEMM_MMA_MAX_BATCH) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       [NSString stringWithFormat:@"q8_0_gemm_mma: cols %u (positive multiple of 32), batch %u (1..=%u)",
                                                  cols, batch, CH_METAL_GEMM_MMA_MAX_BATCH]);
    }
    size_t w_need = (size_t)rows * ((size_t)cols / 32u) * 34u;
    /* 4 SIMD groups own a 64-row x 32-token tile (CH_MMA_* in kernels.metal). */
    return ch_gemm_dispatch_rows(ctx, ctx->pso_q8gemm_mma, w, w_need, x, y, rows, cols, batch,
                                 x_off, y_off, 64, 128, CH_METAL_GEMM_MMA_MAX_BATCH, 32, err,
                                 err_cap, "q8_0_gemm_mma");
}

int ch_metal_q8_0_gemm_mma_f16op(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                                 ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                                 uint32_t x_off, uint32_t y_off, char *err, size_t err_cap) {
    if (cols == 0 || (cols % 32u) != 0u || batch == 0 || batch > CH_METAL_GEMM_MMA_MAX_BATCH) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       [NSString stringWithFormat:@"q8_0_gemm_mma_f16op: cols %u (positive multiple of 32), batch %u (1..=%u)",
                                                   cols, batch, CH_METAL_GEMM_MMA_MAX_BATCH]);
    }
    size_t w_need = (size_t)rows * ((size_t)cols / 32u) * 34u;
    /* Same 64-row x 32-token tiles as ch_q8_0_gemm_mma; only the staged
     * operand precision differs (metal-f16-act, ADR-010). */
    return ch_gemm_dispatch_rows(ctx, ctx->pso_q8gemm_mma_f16op, w, w_need, x, y, rows, cols,
                                 batch, x_off, y_off, 64, 128, CH_METAL_GEMM_MMA_MAX_BATCH, 32,
                                 err, err_cap, "q8_0_gemm_mma_f16op");
}

int ch_metal_q8_0_gemm(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                       ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                       uint32_t x_off, uint32_t y_off, char *err, size_t err_cap) {
    if (cols == 0 || (cols % 32u) != 0u) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       @"q8_0_gemm: cols must be a positive multiple of 32");
    }
    size_t w_need = (size_t)rows * ((size_t)cols / 32u) * 34u;
    return ch_gemm_dispatch(ctx, ctx->pso_q8gemm, w, w_need, x, y, rows, cols, batch, x_off,
                            y_off, err, err_cap, "q8_0_gemm");
}

int ch_metal_f16_gemm(ch_metal_ctx *ctx, ch_metal_buffer *w, ch_metal_buffer *x,
                      ch_metal_buffer *y, uint32_t rows, uint32_t cols, uint32_t batch,
                      uint32_t x_off, uint32_t y_off, char *err, size_t err_cap) {
    size_t w_need = (size_t)rows * (size_t)cols * 2u;
    return ch_gemm_dispatch(ctx, ctx->pso_f16gemm, w, w_need, x, y, rows, cols, batch, x_off,
                            y_off, err, err_cap, "f16_gemm");
}

int ch_metal_fence(ch_metal_ctx *ctx, char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "fence") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (ctx->group_active) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       @"fence: a dispatch group is open; end or abort it first");
    }
    return ch_wait_pending(ctx, err, err_cap);
}

uint64_t ch_metal_committed_dispatches(const ch_metal_ctx *ctx) {
    return ctx == NULL ? 0 : atomic_load_explicit(&ctx->committed, memory_order_relaxed);
}

uint64_t ch_metal_completed_dispatches(const ch_metal_ctx *ctx) {
    return ctx == NULL ? 0 : atomic_load_explicit(&ctx->completed, memory_order_relaxed);
}

double ch_metal_last_gpu_ms(const ch_metal_ctx *ctx) {
    if (ctx == NULL) {
        return 0.0;
    }
    return (double)atomic_load_explicit(&ctx->last_gpu_nanos, memory_order_relaxed) / 1e6;
}

uint64_t ch_metal_gpu_nanos_total(const ch_metal_ctx *ctx) {
    return ctx == NULL ? 0 : atomic_load_explicit(&ctx->gpu_nanos_total, memory_order_relaxed);
}

int ch_metal_profile_set(ch_metal_ctx *ctx, int enabled) {
    if (ctx == NULL) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    os_unfair_lock_lock(&ctx->profile_lock);
    ctx->profile_used = 0;
    memset(ctx->profile, 0, sizeof ctx->profile);
    os_unfair_lock_unlock(&ctx->profile_lock);
    atomic_store_explicit(&ctx->profiling, enabled ? 1 : 0, memory_order_relaxed);
    return CH_METAL_OK;
}

int ch_metal_counter_sampling_supported(ch_metal_ctx *ctx) {
    return ctx != NULL && ctx->counter_ready;
}

int ch_metal_profile_read(ch_metal_ctx *ctx, uint32_t index, char *name, size_t name_cap,
                          uint64_t *nanos, uint64_t *count) {
    if (ctx == NULL || name == NULL || name_cap == 0 || nanos == NULL || count == NULL) {
        return 0;
    }
    int found = 0;
    os_unfair_lock_lock(&ctx->profile_lock);
    if (index < ctx->profile_used) {
        strlcpy(name, ctx->profile[index].name, name_cap);
        *nanos = ctx->profile[index].nanos;
        *count = ctx->profile[index].count;
        found = 1;
    }
    os_unfair_lock_unlock(&ctx->profile_lock);
    return found;
}

uint64_t ch_metal_allocated_bytes(const ch_metal_ctx *ctx) {
    return ctx == NULL ? 0 : atomic_load_explicit(&ctx->allocated, memory_order_relaxed);
}

uint64_t ch_metal_in_flight_bytes(const ch_metal_ctx *ctx) {
    return ctx == NULL ? 0 : atomic_load_explicit(&ctx->leased, memory_order_relaxed);
}

/* ---------------- Q35-2A: Gated DeltaNet operator dispatches ---------------- */

int ch_metal_qwen35_conv_step(ch_metal_ctx *ctx, ch_metal_buffer *hist, ch_metal_buffer *x,
                              ch_metal_buffer *w, ch_metal_buffer *y, uint32_t channels,
                              uint32_t kernel, char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "qwen35_conv") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (channels == 0 || kernel == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS, @"qwen35_conv: zero geometry");
    }
    if (ch_check_buffer(ctx, hist, (size_t)(kernel - 1) * channels * 4u, err, err_cap,
                        "qwen35_conv.hist") != CH_METAL_OK ||
        ch_check_buffer(ctx, x, (size_t)channels * 4u, err, err_cap, "qwen35_conv.x") != CH_METAL_OK ||
        ch_check_buffer(ctx, w, (size_t)kernel * channels * 4u, err, err_cap,
                        "qwen35_conv.w") != CH_METAL_OK ||
        ch_check_buffer(ctx, y, (size_t)channels * 4u, err, err_cap, "qwen35_conv.y") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"qwen35_conv: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_conv];
        [enc setBuffer:hist->buffer offset:0 atIndex:0];
        [enc setBuffer:x->buffer offset:0 atIndex:1];
        [enc setBuffer:w->buffer offset:0 atIndex:2];
        [enc setBuffer:y->buffer offset:0 atIndex:3];
        [f.keep addObject:hist->buffer];
        [f.keep addObject:x->buffer];
        [f.keep addObject:w->buffer];
        [f.keep addObject:y->buffer];
        [enc setBytes:&channels length:sizeof(channels) atIndex:4];
        [enc setBytes:&kernel length:sizeof(kernel) atIndex:5];
        NSUInteger width = ctx->pso_q35_conv.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width * 8 : 256;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)channels, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, "qwen35_conv");
    }
}

int ch_metal_qwen35_conv_shift(ch_metal_ctx *ctx, ch_metal_buffer *hist, ch_metal_buffer *x,
                               uint32_t channels, uint32_t kernel, char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "qwen35_conv_shift") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (channels == 0 || kernel == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS, @"qwen35_conv_shift: zero geometry");
    }
    if (ch_check_buffer(ctx, hist, (size_t)(kernel - 1) * channels * 4u, err, err_cap,
                        "qwen35_conv_shift.hist") != CH_METAL_OK ||
        ch_check_buffer(ctx, x, (size_t)channels * 4u, err, err_cap,
                        "qwen35_conv_shift.x") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"qwen35_conv_shift: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_conv_shift];
        [enc setBuffer:hist->buffer offset:0 atIndex:0];
        [enc setBuffer:x->buffer offset:0 atIndex:1];
        [f.keep addObject:hist->buffer];
        [f.keep addObject:x->buffer];
        [enc setBytes:&channels length:sizeof(channels) atIndex:2];
        [enc setBytes:&kernel length:sizeof(kernel) atIndex:3];
        NSUInteger width = ctx->pso_q35_conv_shift.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width * 8 : 256;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)channels, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, "qwen35_conv_shift");
    }
}

int ch_metal_qwen35_l2norm_heads(ch_metal_ctx *ctx, ch_metal_buffer *x, ch_metal_buffer *y,
                                 uint32_t rows, uint32_t cols, float eps, char *err,
                                 size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "qwen35_l2norm") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (rows == 0 || cols == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS, @"qwen35_l2norm: zero geometry");
    }
    if (ch_check_buffer(ctx, x, (size_t)rows * cols * 4u, err, err_cap, "qwen35_l2norm.x") != CH_METAL_OK ||
        ch_check_buffer(ctx, y, (size_t)rows * cols * 4u, err, err_cap, "qwen35_l2norm.y") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"qwen35_l2norm: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_l2norm];
        [enc setBuffer:x->buffer offset:0 atIndex:0];
        [enc setBuffer:y->buffer offset:0 atIndex:1];
        [f.keep addObject:x->buffer];
        [f.keep addObject:y->buffer];
        [enc setBytes:&cols length:sizeof(cols) atIndex:2];
        [enc setBytes:&eps length:sizeof(eps) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, "qwen35_l2norm");
    }
}

int ch_metal_qwen35_beta_decay(ch_metal_ctx *ctx, ch_metal_buffer *raw_beta, ch_metal_buffer *alpha,
                               ch_metal_buffer *dt_bias, ch_metal_buffer *ssm_a,
                               ch_metal_buffer *beta_out, ch_metal_buffer *gate_out, uint32_t n,
                               char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "qwen35_beta_decay") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (n == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS, @"qwen35_beta_decay: zero length");
    }
    size_t need = (size_t)n * 4u;
    if (ch_check_buffer(ctx, raw_beta, need, err, err_cap, "qwen35_beta_decay.beta") != CH_METAL_OK ||
        ch_check_buffer(ctx, alpha, need, err, err_cap, "qwen35_beta_decay.alpha") != CH_METAL_OK ||
        ch_check_buffer(ctx, dt_bias, need, err, err_cap, "qwen35_beta_decay.dt") != CH_METAL_OK ||
        ch_check_buffer(ctx, ssm_a, need, err, err_cap, "qwen35_beta_decay.a") != CH_METAL_OK ||
        ch_check_buffer(ctx, beta_out, need, err, err_cap, "qwen35_beta_decay.beta_out") != CH_METAL_OK ||
        ch_check_buffer(ctx, gate_out, need, err, err_cap, "qwen35_beta_decay.gate_out") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"qwen35_beta_decay: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_beta_decay];
        [enc setBuffer:raw_beta->buffer offset:0 atIndex:0];
        [enc setBuffer:alpha->buffer offset:0 atIndex:1];
        [enc setBuffer:dt_bias->buffer offset:0 atIndex:2];
        [enc setBuffer:ssm_a->buffer offset:0 atIndex:3];
        [enc setBuffer:beta_out->buffer offset:0 atIndex:4];
        [enc setBuffer:gate_out->buffer offset:0 atIndex:5];
        [f.keep addObject:raw_beta->buffer];
        [f.keep addObject:alpha->buffer];
        [f.keep addObject:dt_bias->buffer];
        [f.keep addObject:ssm_a->buffer];
        [f.keep addObject:beta_out->buffer];
        [f.keep addObject:gate_out->buffer];
        [enc setBytes:&n length:sizeof(n) atIndex:6];
        NSUInteger width = ctx->pso_q35_beta_decay.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width * 8 : 256;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, "qwen35_beta_decay");
    }
}

int ch_metal_qwen35_gdn_step(ch_metal_ctx *ctx, ch_metal_buffer *q, ch_metal_buffer *k,
                             ch_metal_buffer *v, ch_metal_buffer *gate, ch_metal_buffer *beta,
                             ch_metal_buffer *state, ch_metal_buffer *out, uint32_t heads,
                             uint32_t k_dim, uint32_t v_dim, float q_scale, char *err,
                             size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "qwen35_gdn_step") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (heads == 0 || k_dim == 0 || v_dim == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS, @"qwen35_gdn_step: zero geometry");
    }
    size_t qk = (size_t)heads * k_dim * 4u;
    size_t vv = (size_t)heads * v_dim * 4u;
    if (ch_check_buffer(ctx, q, qk, err, err_cap, "gdn_step.q") != CH_METAL_OK ||
        ch_check_buffer(ctx, k, qk, err, err_cap, "gdn_step.k") != CH_METAL_OK ||
        ch_check_buffer(ctx, v, vv, err, err_cap, "gdn_step.v") != CH_METAL_OK ||
        ch_check_buffer(ctx, gate, (size_t)heads * 4u, err, err_cap, "gdn_step.gate") != CH_METAL_OK ||
        ch_check_buffer(ctx, beta, (size_t)heads * 4u, err, err_cap, "gdn_step.beta") != CH_METAL_OK ||
        ch_check_buffer(ctx, state, (size_t)heads * k_dim * v_dim * 4u, err, err_cap,
                        "gdn_step.state") != CH_METAL_OK ||
        ch_check_buffer(ctx, out, vv, err, err_cap, "gdn_step.out") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"gdn_step: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_gdn_step];
        [enc setBuffer:q->buffer offset:0 atIndex:0];
        [enc setBuffer:k->buffer offset:0 atIndex:1];
        [enc setBuffer:v->buffer offset:0 atIndex:2];
        [enc setBuffer:gate->buffer offset:0 atIndex:3];
        [enc setBuffer:beta->buffer offset:0 atIndex:4];
        [enc setBuffer:state->buffer offset:0 atIndex:5];
        [enc setBuffer:out->buffer offset:0 atIndex:6];
        [f.keep addObject:q->buffer];
        [f.keep addObject:k->buffer];
        [f.keep addObject:v->buffer];
        [f.keep addObject:gate->buffer];
        [f.keep addObject:beta->buffer];
        [f.keep addObject:state->buffer];
        [f.keep addObject:out->buffer];
        uint32_t kd = k_dim;
        uint32_t vd = v_dim;
        [enc setBytes:&kd length:sizeof(kd) atIndex:7];
        [enc setBytes:&vd length:sizeof(vd) atIndex:8];
        [enc setBytes:&q_scale length:sizeof(q_scale) atIndex:9];
        /* One thread per (value column, head); columns of one head share
         * a threadgroup row so a SIMD group reads contiguous state. */
        NSUInteger width = ctx->pso_q35_gdn_step.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width : 32;
        if (tg > (NSUInteger)v_dim) {
            tg = (NSUInteger)v_dim;
        }
        [enc dispatchThreads:MTLSizeMake((NSUInteger)v_dim, (NSUInteger)heads, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, "qwen35_gdn_step");
    }
}

int ch_metal_qwen35_gated_norm(ch_metal_ctx *ctx, ch_metal_buffer *o, ch_metal_buffer *z,
                               ch_metal_buffer *weight, ch_metal_buffer *out, uint32_t rows,
                               uint32_t cols, float eps, char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "qwen35_gated_norm") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (rows == 0 || cols == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS, @"qwen35_gated_norm: zero geometry");
    }
    size_t need = (size_t)rows * cols * 4u;
    if (ch_check_buffer(ctx, o, need, err, err_cap, "gated_norm.o") != CH_METAL_OK ||
        ch_check_buffer(ctx, z, need, err, err_cap, "gated_norm.z") != CH_METAL_OK ||
        ch_check_buffer(ctx, weight, (size_t)cols * 4u, err, err_cap, "gated_norm.w") != CH_METAL_OK ||
        ch_check_buffer(ctx, out, need, err, err_cap, "gated_norm.out") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"gated_norm: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_gated_norm];
        [enc setBuffer:o->buffer offset:0 atIndex:0];
        [enc setBuffer:z->buffer offset:0 atIndex:1];
        [enc setBuffer:weight->buffer offset:0 atIndex:2];
        [enc setBuffer:out->buffer offset:0 atIndex:3];
        [f.keep addObject:o->buffer];
        [f.keep addObject:z->buffer];
        [f.keep addObject:weight->buffer];
        [f.keep addObject:out->buffer];
        [enc setBytes:&cols length:sizeof(cols) atIndex:4];
        [enc setBytes:&eps length:sizeof(eps) atIndex:5];
        NSUInteger width = ctx->pso_q35_gated_norm.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width : 32;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, "qwen35_gated_norm");
    }
}

/* ---------------- Q35-2B: full-attention operator dispatches ---------------- */

int ch_metal_qwen35_q_gate_split(ch_metal_ctx *ctx, ch_metal_buffer *qg, ch_metal_buffer *q_out,
                                 ch_metal_buffer *gate_out, uint32_t heads, uint32_t head_dim,
                                 char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "qg_split") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (heads == 0 || head_dim == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS, @"qg_split: zero geometry");
    }
    size_t row = (size_t)heads * head_dim * 4u;
    if (ch_check_buffer(ctx, qg, row * 2u, err, err_cap, "qg_split.qg") != CH_METAL_OK ||
        ch_check_buffer(ctx, q_out, row, err, err_cap, "qg_split.q") != CH_METAL_OK ||
        ch_check_buffer(ctx, gate_out, row, err, err_cap, "qg_split.gate") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"qg_split: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_qg_split];
        [enc setBuffer:qg->buffer offset:0 atIndex:0];
        [enc setBuffer:q_out->buffer offset:0 atIndex:1];
        [enc setBuffer:gate_out->buffer offset:0 atIndex:2];
        [f.keep addObject:qg->buffer];
        [f.keep addObject:q_out->buffer];
        [f.keep addObject:gate_out->buffer];
        uint32_t hd = head_dim;
        [enc setBytes:&hd length:sizeof(hd) atIndex:3];
        NSUInteger width = ctx->pso_q35_qg_split.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width * 8 : 256;
        NSUInteger total = (NSUInteger)heads * head_dim;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, "qwen35_qg_split");
    }
}

int ch_metal_qwen35_rope_partial(ch_metal_ctx *ctx, ch_metal_buffer *q, ch_metal_buffer *k,
                                 ch_metal_buffer *inv_freq, uint32_t pos, uint32_t n_q_heads,
                                 uint32_t n_kv_heads, uint32_t head_dim, uint32_t n_rot,
                                 char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "q35_rope_partial") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (head_dim == 0 || n_rot == 0 || n_rot > head_dim || n_rot % 2u != 0u) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS,
                       @"q35_rope_partial: n_rot must be an even prefix of head_dim");
    }
    if (ch_check_buffer(ctx, q, (size_t)n_q_heads * head_dim * 4u, err, err_cap,
                        "rope.q") != CH_METAL_OK ||
        ch_check_buffer(ctx, k, (size_t)n_kv_heads * head_dim * 4u, err, err_cap,
                        "rope.k") != CH_METAL_OK ||
        ch_check_buffer(ctx, inv_freq, (size_t)(n_rot / 2u) * 4u, err, err_cap,
                        "rope.inv_freq") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"q35_rope_partial: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_rope_partial];
        [enc setBuffer:q->buffer offset:0 atIndex:0];
        [enc setBuffer:k->buffer offset:0 atIndex:1];
        [enc setBuffer:inv_freq->buffer offset:0 atIndex:2];
        [f.keep addObject:q->buffer];
        [f.keep addObject:k->buffer];
        [f.keep addObject:inv_freq->buffer];
        [enc setBytes:&pos length:sizeof(pos) atIndex:3];
        [enc setBytes:&n_q_heads length:sizeof(n_q_heads) atIndex:4];
        [enc setBytes:&n_kv_heads length:sizeof(n_kv_heads) atIndex:5];
        [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:6];
        [enc setBytes:&n_rot length:sizeof(n_rot) atIndex:7];
        NSUInteger total = (NSUInteger)(n_q_heads + n_kv_heads);
        [enc dispatchThreadgroups:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, "qwen35_rope_partial");
    }
}

int ch_metal_qwen35_kv_store_f16(ch_metal_ctx *ctx, ch_metal_buffer *k_src, ch_metal_buffer *v_src,
                                 ch_metal_buffer *k_cache, ch_metal_buffer *v_cache, uint32_t pos,
                                 uint32_t cap, uint32_t head_dim, uint32_t kv_heads, char *err,
                                 size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "kv_store_f16") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (head_dim == 0 || kv_heads == 0 || pos >= cap) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS, @"kv_store_f16: geometry out of range");
    }
    size_t src = (size_t)kv_heads * head_dim * 4u;
    size_t dst = (size_t)kv_heads * cap * head_dim * 2u;
    if (ch_check_buffer(ctx, k_src, src, err, err_cap, "kv_store_f16.k_src") != CH_METAL_OK ||
        ch_check_buffer(ctx, v_src, src, err, err_cap, "kv_store_f16.v_src") != CH_METAL_OK ||
        ch_check_buffer(ctx, k_cache, dst, err, err_cap, "kv_store_f16.k_cache") != CH_METAL_OK ||
        ch_check_buffer(ctx, v_cache, dst, err, err_cap, "kv_store_f16.v_cache") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"kv_store_f16: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_kv_store_f16];
        [enc setBuffer:k_src->buffer offset:0 atIndex:0];
        [enc setBuffer:v_src->buffer offset:0 atIndex:1];
        [enc setBuffer:k_cache->buffer offset:0 atIndex:2];
        [enc setBuffer:v_cache->buffer offset:0 atIndex:3];
        [f.keep addObject:k_src->buffer];
        [f.keep addObject:v_src->buffer];
        [f.keep addObject:k_cache->buffer];
        [f.keep addObject:v_cache->buffer];
        [enc setBytes:&pos length:sizeof(pos) atIndex:4];
        [enc setBytes:&cap length:sizeof(cap) atIndex:5];
        [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:6];
        /* A full row per head: head_dim can exceed the 128-thread group. */
        [enc dispatchThreads:MTLSizeMake((NSUInteger)kv_heads * head_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, "qwen35_kv_store_f16");
    }
}

static id<MTLComputePipelineState> ch_attn_sk_partial_pso(ch_metal_ctx *ctx, uint32_t head_dim,
                                                          uint32_t group, int f32_cache);

int ch_metal_qwen35_attention_f16(ch_metal_ctx *ctx, ch_metal_buffer *q, ch_metal_buffer *k_cache,
                                  ch_metal_buffer *v_cache, ch_metal_buffer *out,
                                  ch_metal_buffer *scratch, uint32_t heads, uint32_t pos,
                                  uint32_t cap, uint32_t head_dim, uint32_t group, float scale,
                                  char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "attn_f16") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (heads == 0 || head_dim == 0 || group == 0 || pos >= cap || heads % group != 0u) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS, @"attn_f16: geometry out of range");
    }
    /* One query row through the split-K chunk kernels (the decode path;
     * chunked prefill keeps the one-group-per-row kernel - see
     * ch_attention_chunk). */
    if (ch_attn_sk_partial_pso(ctx, head_dim, group, 0) != nil) {
        return ch_metal_qwen35_attention_f16_chunk(ctx, q, k_cache, v_cache, out, scratch, heads,
                                                   pos, cap, head_dim, group, scale, 1u, 0u,
                                                   err, err_cap);
    }
    size_t qb = (size_t)heads * head_dim * 4u;
    size_t cache = (size_t)(heads / group) * cap * head_dim * 2u;
    if (ch_check_buffer(ctx, q, qb, err, err_cap, "attn_f16.q") != CH_METAL_OK ||
        ch_check_buffer(ctx, k_cache, cache, err, err_cap, "attn_f16.k") != CH_METAL_OK ||
        ch_check_buffer(ctx, v_cache, cache, err, err_cap, "attn_f16.v") != CH_METAL_OK ||
        ch_check_buffer(ctx, out, qb, err, err_cap, "attn_f16.out") != CH_METAL_OK ||
        ch_check_buffer(ctx, scratch, (size_t)heads * cap * 4u, err, err_cap,
                        "attn_f16.scratch") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"attn_f16: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_attention_f16];
        [enc setBuffer:q->buffer offset:0 atIndex:0];
        [enc setBuffer:k_cache->buffer offset:0 atIndex:1];
        [enc setBuffer:v_cache->buffer offset:0 atIndex:2];
        [enc setBuffer:out->buffer offset:0 atIndex:3];
        [enc setBuffer:scratch->buffer offset:0 atIndex:4];
        [f.keep addObject:q->buffer];
        [f.keep addObject:k_cache->buffer];
        [f.keep addObject:v_cache->buffer];
        [f.keep addObject:out->buffer];
        [f.keep addObject:scratch->buffer];
        uint32_t capv = cap;
        uint32_t hdv = head_dim;
        uint32_t gv = group;
        [enc setBytes:&pos length:sizeof(pos) atIndex:5];
        [enc setBytes:&capv length:sizeof(capv) atIndex:6];
        [enc setBytes:&hdv length:sizeof(hdv) atIndex:7];
        [enc setBytes:&gv length:sizeof(gv) atIndex:8];
        [enc setBytes:&scale length:sizeof(scale) atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, "qwen35_attention_f16");
    }
}

int ch_metal_qwen35_attn_gate(ch_metal_ctx *ctx, ch_metal_buffer *attn, ch_metal_buffer *gate,
                              uint32_t n, char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "attn_gate") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (n == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS, @"attn_gate: zero length");
    }
    size_t need = (size_t)n * 4u;
    if (ch_check_buffer(ctx, attn, need, err, err_cap, "attn_gate.attn") != CH_METAL_OK ||
        ch_check_buffer(ctx, gate, need, err, err_cap, "attn_gate.gate") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"attn_gate: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_attn_gate];
        [enc setBuffer:attn->buffer offset:0 atIndex:0];
        [enc setBuffer:gate->buffer offset:0 atIndex:1];
        [f.keep addObject:attn->buffer];
        [f.keep addObject:gate->buffer];
        [enc setBytes:&n length:sizeof(n) atIndex:2];
        NSUInteger width = ctx->pso_q35_attn_gate.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width * 8 : 256;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, "qwen35_attn_gate");
    }
}

int ch_metal_qwen35_silu(ch_metal_ctx *ctx, ch_metal_buffer *x, uint32_t n, char *err,
                         size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "q35_silu") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (n == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS, @"q35_silu: zero length");
    }
    if (ch_check_buffer(ctx, x, (size_t)n * 4u, err, err_cap, "q35_silu.x") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"q35_silu: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_silu];
        [enc setBuffer:x->buffer offset:0 atIndex:0];
        [f.keep addObject:x->buffer];
        [enc setBytes:&n length:sizeof(n) atIndex:1];
        NSUInteger width = ctx->pso_q35_silu.threadExecutionWidth;
        NSUInteger tg = width > 0 ? width * 8 : 256;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, "qwen35_silu");
    }
}

int ch_metal_qwen35_split_qkv(ch_metal_ctx *ctx, ch_metal_buffer *y, ch_metal_buffer *q,
                              ch_metal_buffer *k, ch_metal_buffer *v, uint32_t width, char *err,
                              size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "q35_split_qkv") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (width == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS, @"q35_split_qkv: zero width");
    }
    size_t qk = (size_t)width * 4u;
    if (ch_check_buffer(ctx, y, qk * 3u, err, err_cap, "split_qkv.y") != CH_METAL_OK ||
        ch_check_buffer(ctx, q, qk, err, err_cap, "split_qkv.q") != CH_METAL_OK ||
        ch_check_buffer(ctx, k, qk, err, err_cap, "split_qkv.k") != CH_METAL_OK ||
        ch_check_buffer(ctx, v, qk, err, err_cap, "split_qkv.v") != CH_METAL_OK) {
        return CH_METAL_ERR_BOUNDS;
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"q35_split_qkv: no encoder");
        }
        [enc setComputePipelineState:ctx->pso_q35_split_qkv];
        [enc setBuffer:y->buffer offset:0 atIndex:0];
        [enc setBuffer:q->buffer offset:0 atIndex:1];
        [enc setBuffer:k->buffer offset:0 atIndex:2];
        [enc setBuffer:v->buffer offset:0 atIndex:3];
        [f.keep addObject:y->buffer];
        [f.keep addObject:q->buffer];
        [f.keep addObject:k->buffer];
        [f.keep addObject:v->buffer];
        [enc setBytes:&width length:sizeof(width) atIndex:4];
        NSUInteger tw = ctx->pso_q35_split_qkv.threadExecutionWidth;
        NSUInteger tg = tw > 0 ? tw * 8 : 256;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)width, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return ch_frame_commit(ctx, &f, err, err_cap, "qwen35_split_qkv");
    }
}

int ch_metal_buffer_zero(ch_metal_ctx *ctx, ch_metal_buffer *buffer, size_t off, size_t len,
                         char *err, size_t err_cap) {
    if (ch_check_ctx(ctx, err, err_cap, "buffer_zero") != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    if (buffer == NULL || buffer->owner != ctx) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS, @"buffer_zero: foreign buffer");
    }
    if (len == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS, @"buffer_zero: zero length");
    }
    if (off + len > buffer->len) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS, @"buffer_zero: range outside the buffer");
    }
    @autoreleasepool {
        if (buffer->shared) {
            /* Shared storage is host-coherent: zero in place, no staging. */
            memset((char *)buffer->buffer.contents + off, 0, len);
            return CH_METAL_OK;
        }
        /* Private storage: GPU-side fill (cold path, synchronous). */
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"buffer_zero: no command buffer");
        }
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit fillBuffer:buffer->buffer range:NSMakeRange(off, len) value:0];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL, @"buffer_zero: GPU fill failed");
        }
        return CH_METAL_OK;
    }
}

/* ---- chunk-wide Gated DeltaNet core (engine prefill) ---- */

typedef struct {
    ch_metal_buffer *buf;
    size_t need; /* bytes the kernel may touch */
    const char *name;
} ch_arg_buffer;

/* Encode one compute dispatch over checked buffers and 32-bit constants. */
static int ch_dispatch_args(ch_metal_ctx *ctx, id<MTLComputePipelineState> pso,
                            const ch_arg_buffer *bufs, int nbufs, const uint32_t *consts,
                            int nconsts, MTLSize grid, MTLSize group, char *err,
                            size_t err_cap, const char *what) {
    if (ch_check_ctx(ctx, err, err_cap, what) != CH_METAL_OK) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    for (int i = 0; i < nbufs; ++i) {
        if (ch_check_buffer(ctx, bufs[i].buf, bufs[i].need, err, err_cap, bufs[i].name) !=
            CH_METAL_OK) {
            return CH_METAL_ERR_BOUNDS;
        }
    }
    @autoreleasepool {
        ch_frame f;
        id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
        if (enc == nil) {
            return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL,
                           [NSString stringWithFormat:@"%s: no encoder", what]);
        }
        [enc setComputePipelineState:pso];
        for (int i = 0; i < nbufs; ++i) {
            [enc setBuffer:bufs[i].buf->buffer offset:0 atIndex:(NSUInteger)i];
            [f.keep addObject:bufs[i].buf->buffer];
        }
        for (int i = 0; i < nconsts; ++i) {
            [enc setBytes:&consts[i] length:sizeof(uint32_t) atIndex:(NSUInteger)(nbufs + i)];
        }
        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        return ch_frame_commit(ctx, &f, err, err_cap, what);
    }
}

int ch_metal_qwen35_conv_chunk(ch_metal_ctx *ctx, ch_metal_buffer *hist, ch_metal_buffer *x,
                               ch_metal_buffer *w, ch_metal_buffer *y, uint32_t channels,
                               uint32_t kern, uint32_t n, char *err, size_t err_cap) {
    if (channels == 0 || kern < 2 || n == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       @"qwen35_conv_chunk: channels, n must be positive and kernel >= 2");
    }
    size_t row = (size_t)channels * 4u;
    ch_arg_buffer bufs[4] = {
        {hist, (size_t)(kern - 1u) * row, "conv_chunk.hist"},
        {x, (size_t)n * row, "conv_chunk.x"},
        {w, (size_t)channels * kern * 4u, "conv_chunk.w"},
        {y, (size_t)n * row, "conv_chunk.y"},
    };
    uint32_t consts[3] = {channels, kern, n};
    int rc = ch_dispatch_args(ctx, ctx->pso_q35_conv_chunk, bufs, 4, consts, 3,
                              MTLSizeMake(channels, n, 1), MTLSizeMake(64, 1, 1), err, err_cap,
                              "qwen35_conv_chunk");
    if (rc != CH_METAL_OK) {
        return rc;
    }
    /* History refresh after every output of the chunk was computed (the
     * queue/encoder orders the two dispatches). */
    ch_arg_buffer sbufs[2] = {
        {hist, (size_t)(kern - 1u) * row, "conv_chunk.hist"},
        {x, (size_t)n * row, "conv_chunk.x"},
    };
    return ch_dispatch_args(ctx, ctx->pso_q35_conv_chunk_shift, sbufs, 2, consts, 3,
                            MTLSizeMake(channels, 1, 1), MTLSizeMake(64, 1, 1), err, err_cap,
                            "qwen35_conv_chunk_shift");
}

int ch_metal_qwen35_conv_chunk_shift(ch_metal_ctx *ctx, ch_metal_buffer *hist,
                                     ch_metal_buffer *x, uint32_t channels, uint32_t kern,
                                     uint32_t n, char *err, size_t err_cap) {
    if (channels == 0 || kern < 2 || n == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       @"qwen35_conv_chunk_shift: channels, n must be positive and kernel >= 2");
    }
    size_t row = (size_t)channels * 4u;
    ch_arg_buffer bufs[2] = {
        {hist, (size_t)(kern - 1u) * row, "conv_chunk_shift.hist"},
        {x, (size_t)n * row, "conv_chunk_shift.x"},
    };
    uint32_t consts[3] = {channels, kern, n};
    return ch_dispatch_args(ctx, ctx->pso_q35_conv_chunk_shift, bufs, 2, consts, 3,
                            MTLSizeMake(channels, 1, 1), MTLSizeMake(64, 1, 1), err, err_cap,
                            "qwen35_conv_chunk_shift");
}

int ch_metal_qwen35_split_qkv_chunk(ch_metal_ctx *ctx, ch_metal_buffer *y, ch_metal_buffer *q,
                                    ch_metal_buffer *k, ch_metal_buffer *v, uint32_t width,
                                    uint32_t n, char *err, size_t err_cap) {
    if (width == 0 || n == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       @"qwen35_split_qkv_chunk: width and n must be positive");
    }
    size_t out = (size_t)n * width * 4u;
    ch_arg_buffer bufs[4] = {
        {y, 3u * out, "split_chunk.y"},
        {q, out, "split_chunk.q"},
        {k, out, "split_chunk.k"},
        {v, out, "split_chunk.v"},
    };
    uint32_t consts[2] = {width, n};
    return ch_dispatch_args(ctx, ctx->pso_q35_split_qkv_chunk, bufs, 4, consts, 2,
                            MTLSizeMake(width, n, 1), MTLSizeMake(64, 1, 1), err, err_cap,
                            "qwen35_split_qkv_chunk");
}

int ch_metal_qwen35_beta_decay_chunk(ch_metal_ctx *ctx, ch_metal_buffer *raw_beta,
                                     ch_metal_buffer *alpha, ch_metal_buffer *dt_bias,
                                     ch_metal_buffer *ssm_a, ch_metal_buffer *beta_out,
                                     ch_metal_buffer *gate_out, uint32_t per, uint32_t n,
                                     char *err, size_t err_cap) {
    if (per == 0 || n == 0) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       @"qwen35_beta_decay_chunk: per and n must be positive");
    }
    uint32_t total = per * n;
    size_t rows = (size_t)total * 4u;
    ch_arg_buffer bufs[6] = {
        {raw_beta, rows, "beta_chunk.raw_beta"}, {alpha, rows, "beta_chunk.alpha"},
        {dt_bias, (size_t)per * 4u, "beta_chunk.dt_bias"}, {ssm_a, (size_t)per * 4u, "beta_chunk.ssm_a"},
        {beta_out, rows, "beta_chunk.beta_out"}, {gate_out, rows, "beta_chunk.gate_out"},
    };
    uint32_t consts[2] = {total, per};
    return ch_dispatch_args(ctx, ctx->pso_q35_beta_decay_chunk, bufs, 6, consts, 2,
                            MTLSizeMake(total, 1, 1), MTLSizeMake(64, 1, 1), err, err_cap,
                            "qwen35_beta_decay_chunk");
}

/* Lanes per state column in ch_qwen35_gdn_chunk_k128 (kernels.metal). */
#define CH_GDN_LANES 8u

int ch_metal_qwen35_gdn_chunk(ch_metal_ctx *ctx, ch_metal_buffer *q, ch_metal_buffer *k,
                              ch_metal_buffer *v, ch_metal_buffer *gate, ch_metal_buffer *beta,
                              ch_metal_buffer *state, ch_metal_buffer *out, uint32_t heads,
                              uint32_t k_dim, uint32_t v_dim, float q_scale, uint32_t n,
                              uint32_t qk_stride, uint32_t v_stride, uint32_t g_stride,
                              char *err, size_t err_cap) {
    /* No rollback point: the state buffer stands in for the snapshot and
     * snap_t never matches. */
    return ch_metal_qwen35_gdn_chunk_snap(ctx, q, k, v, gate, beta, state, out, state, UINT32_MAX,
                                          heads, k_dim, v_dim, q_scale, n, qk_stride, v_stride,
                                          g_stride, err, err_cap);
}

int ch_metal_qwen35_gdn_chunk_snap(ch_metal_ctx *ctx, ch_metal_buffer *q, ch_metal_buffer *k,
                                   ch_metal_buffer *v, ch_metal_buffer *gate,
                                   ch_metal_buffer *beta, ch_metal_buffer *state,
                                   ch_metal_buffer *out, ch_metal_buffer *snap, uint32_t snap_t,
                                   uint32_t heads, uint32_t k_dim, uint32_t v_dim, float q_scale,
                                   uint32_t n, uint32_t qk_stride, uint32_t v_stride,
                                   uint32_t g_stride, char *err, size_t err_cap) {
    if (heads == 0 || k_dim == 0 || v_dim == 0 || n == 0 || qk_stride < heads * k_dim ||
        v_stride < heads * v_dim || g_stride < heads ||
        (snap_t != UINT32_MAX && snap_t >= n)) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       @"qwen35_gdn_chunk: invalid geometry, strides or rollback token");
    }
    ch_arg_buffer bufs[8] = {
        {q, (size_t)n * qk_stride * 4u, "gdn_chunk.q"},
        {k, (size_t)n * qk_stride * 4u, "gdn_chunk.k"},
        {v, (size_t)n * v_stride * 4u, "gdn_chunk.v"},
        {gate, (size_t)n * g_stride * 4u, "gdn_chunk.gate"},
        {beta, (size_t)n * g_stride * 4u, "gdn_chunk.beta"},
        {state, (size_t)heads * k_dim * v_dim * 4u, "gdn_chunk.state"},
        {out, (size_t)n * v_stride * 4u, "gdn_chunk.out"},
        {snap, (size_t)heads * k_dim * v_dim * 4u, "gdn_chunk.snap"},
    };
    /* Constants by buffer index: 8 k_dim, 9 v_dim, 10 q_scale (f32 bits),
     * 11 heads, 12 n, 13-15 strides, 16 snap_t (the token after which the
     * state is also written to `snap`; UINT32_MAX: never). */
    uint32_t scale_bits;
    memcpy(&scale_bits, &q_scale, sizeof scale_bits);
    uint32_t consts[9] = {k_dim, v_dim, scale_bits, heads, n, qk_stride, v_stride, g_stride, snap_t};
    /* Register-resident state column, split across CH_GDN_LANES lanes,
     * when the key dimension has an instantiation (see kernels.metal). */
    const bool reg = k_dim == 128u && (v_dim % (32u / CH_GDN_LANES)) == 0u &&
                     !ctx->gdn_chunk_generic;
    id<MTLComputePipelineState> pso = reg ? ctx->pso_q35_gdn_chunk_k128 : ctx->pso_q35_gdn_chunk;
    NSUInteger lanes = reg ? (NSUInteger)v_dim * CH_GDN_LANES : (NSUInteger)v_dim;
    /* The lane-split kernel maps lanes in whole SIMD groups of 32. */
    NSUInteger tg = lanes < 32u ? lanes : 32u;
    return ch_dispatch_args(ctx, pso, bufs, 8, consts, 9, MTLSizeMake(lanes, heads, 1),
                            MTLSizeMake(tg, 1, 1), err, err_cap, "qwen35_gdn_chunk");
}

/* ---- chunk-wide gated attention core (engine prefill) ---- */

int ch_metal_qwen35_rope_partial_chunk(ch_metal_ctx *ctx, ch_metal_buffer *q, ch_metal_buffer *k,
                                       ch_metal_buffer *inv_freq, uint32_t pos0,
                                       uint32_t n_q_heads, uint32_t n_kv_heads,
                                       uint32_t head_dim, uint32_t n_rot, uint32_t n, char *err,
                                       size_t err_cap) {
    if (n == 0 || head_dim == 0 || n_rot == 0 || n_rot > head_dim || (n_rot % 2u) != 0u ||
        n_rot / 2u > 128u) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       @"qwen35_rope_partial_chunk: invalid geometry");
    }
    ch_arg_buffer bufs[3] = {
        {q, (size_t)n * n_q_heads * head_dim * 4u, "rope_chunk.q"},
        {k, (size_t)n * n_kv_heads * head_dim * 4u, "rope_chunk.k"},
        {inv_freq, (size_t)(n_rot / 2u) * 4u, "rope_chunk.inv_freq"},
    };
    uint32_t consts[5] = {pos0, n_q_heads, n_kv_heads, head_dim, n_rot};
    return ch_dispatch_args(ctx, ctx->pso_q35_rope_chunk, bufs, 3, consts, 5,
                            MTLSizeMake((NSUInteger)(n_q_heads + n_kv_heads) * 128u, n, 1),
                            MTLSizeMake(128, 1, 1), err, err_cap, "qwen35_rope_partial_chunk");
}

/* KV store of a chunk into a packed f16 (f32_cache 0) or f32 (1) cache. */
static int ch_kv_store_chunk(ch_metal_ctx *ctx, ch_metal_buffer *k_src, ch_metal_buffer *v_src,
                             ch_metal_buffer *k_cache, ch_metal_buffer *v_cache, uint32_t pos0,
                             uint32_t cap, uint32_t head_dim, uint32_t kv_heads, uint32_t n,
                             int f32_cache, char *err, size_t err_cap, const char *what) {
    if (n == 0 || head_dim == 0 || kv_heads == 0 || (uint64_t)pos0 + n > cap) {
        return ch_fail(err, err_cap, CH_METAL_ERR_BOUNDS,
                       [NSString stringWithFormat:@"%s: positions beyond the cache capacity", what]);
    }
    size_t rows = (size_t)n * kv_heads * head_dim * 4u;
    size_t plane = (size_t)kv_heads * cap * head_dim * (f32_cache ? 4u : 2u);
    ch_arg_buffer bufs[4] = {
        {k_src, rows, "kv_chunk.k"}, {v_src, rows, "kv_chunk.v"},
        {k_cache, plane, "kv_chunk.k_cache"}, {v_cache, plane, "kv_chunk.v_cache"},
    };
    uint32_t consts[5] = {pos0, cap, head_dim, kv_heads, n};
    return ch_dispatch_args(ctx,
                            f32_cache ? ctx->pso_q35_kv_store_chunk_f32
                                      : ctx->pso_q35_kv_store_chunk,
                            bufs, 4, consts, 5,
                            MTLSizeMake((NSUInteger)n * kv_heads * head_dim, 1, 1),
                            MTLSizeMake(128, 1, 1), err, err_cap, what);
}

int ch_metal_qwen35_kv_store_f16_chunk(ch_metal_ctx *ctx, ch_metal_buffer *k_src,
                                       ch_metal_buffer *v_src, ch_metal_buffer *k_cache,
                                       ch_metal_buffer *v_cache, uint32_t pos0, uint32_t cap,
                                       uint32_t head_dim, uint32_t kv_heads, uint32_t n, char *err,
                                       size_t err_cap) {
    return ch_kv_store_chunk(ctx, k_src, v_src, k_cache, v_cache, pos0, cap, head_dim, kv_heads,
                             n, 0, err, err_cap, "qwen35_kv_store_f16_chunk");
}

int ch_metal_qwen35_kv_store_f32_chunk(ch_metal_ctx *ctx, ch_metal_buffer *k_src,
                                       ch_metal_buffer *v_src, ch_metal_buffer *k_cache,
                                       ch_metal_buffer *v_cache, uint32_t pos0, uint32_t cap,
                                       uint32_t head_dim, uint32_t kv_heads, uint32_t n, char *err,
                                       size_t err_cap) {
    return ch_kv_store_chunk(ctx, k_src, v_src, k_cache, v_cache, pos0, cap, head_dim, kv_heads,
                             n, 1, err, err_cap, "qwen35_kv_store_f32_chunk");
}

/* The grouped attention instance for this geometry and cache type, or
 * nil (the generic chunk kernel then runs). */
#define CH_ATTN_SK_SLICES 16u

static id<MTLComputePipelineState> ch_attn_sk_partial_pso(ch_metal_ctx *ctx, uint32_t head_dim,
                                                          uint32_t group, int f32_cache) {
    if (f32_cache) {
        if (head_dim == 16u && group == 2u) {
            return ctx->pso_q35_attn_sk_f32_16x2;
        }
        if (head_dim == 128u && group == 2u) {
            return ctx->pso_q35_attn_sk_f32_128x2;
        }
        if (head_dim == 128u && group == 4u) {
            return ctx->pso_q35_attn_sk_f32_128x4;
        }
        return nil;
    }
    if (head_dim == 256u && group == 4u) {
        return ctx->pso_q35_attn_sk_256x4;
    }
    if (head_dim == 16u && group == 2u) {
        return ctx->pso_q35_attn_sk_16x2;
    }
    if (head_dim == 128u && group == 2u) {
        return ctx->pso_q35_attn_sk_128x2;
    }
    if (head_dim == 128u && group == 4u) {
        return ctx->pso_q35_attn_sk_128x4;
    }
    return nil;
}

static id<MTLComputePipelineState> ch_attention_gqa_pso(ch_metal_ctx *ctx, uint32_t head_dim,
                                                       uint32_t group, int f32_cache) {
    if (f32_cache) {
        if (head_dim == 16u && group == 2u) {
            return ctx->pso_q35_attention_gqa_f32_16x2;
        }
        if (head_dim == 128u && group == 2u) {
            return ctx->pso_q35_attention_gqa_f32_128x2;
        }
        if (head_dim == 128u && group == 4u) {
            return ctx->pso_q35_attention_gqa_f32_128x4;
        }
        return nil;
    }
    if (head_dim == 256u && group == 4u) {
        return ctx->pso_q35_attention_gqa_256x4;
    }
    if (head_dim == 16u && group == 2u) {
        return ctx->pso_q35_attention_gqa_16x2;
    }
    if (head_dim == 128u && group == 2u) {
        return ctx->pso_q35_attention_gqa_128x2;
    }
    if (head_dim == 128u && group == 4u) {
        return ctx->pso_q35_attention_gqa_128x4;
    }
    return nil;
}

static id<MTLComputePipelineState> ch_attn_sk_merge_pso(ch_metal_ctx *ctx, uint32_t head_dim,
                                                       uint32_t group) {
    if (head_dim == 256u && group == 4u) {
        return ctx->pso_q35_attn_sk_merge_256x4;
    }
    if (head_dim == 16u && group == 2u) {
        return ctx->pso_q35_attn_sk_merge_16x2;
    }
    if (head_dim == 128u && group == 2u) {
        return ctx->pso_q35_attn_sk_merge_128x2;
    }
    if (head_dim == 128u && group == 4u) {
        return ctx->pso_q35_attn_sk_merge_128x4;
    }
    return nil;
}

/* Chunk attention over a packed f16 (f32_cache 0) or f32 (1) cache. */
static int ch_attention_chunk(ch_metal_ctx *ctx, ch_metal_buffer *q, ch_metal_buffer *k_cache,
                              ch_metal_buffer *v_cache, ch_metal_buffer *out,
                              ch_metal_buffer *scratch, uint32_t heads, uint32_t pos0,
                              uint32_t cap, uint32_t head_dim, uint32_t group, float scale,
                              uint32_t n, uint32_t window, int f32_cache, char *err,
                              size_t err_cap, const char *what) {
    if (n == 0 || heads == 0 || head_dim == 0 || group == 0 || (heads % group) != 0u ||
        (uint64_t)pos0 + n > cap) {
        return ch_fail(err, err_cap, CH_METAL_ERR_UNSUPPORTED,
                       [NSString stringWithFormat:@"%s: invalid geometry or positions", what]);
    }
    uint32_t kv_heads = heads / group;
    size_t rows = (size_t)n * heads * head_dim * 4u;
    size_t plane = (size_t)kv_heads * cap * head_dim * (f32_cache ? 4u : 2u);
    ch_arg_buffer bufs[5] = {
        {q, rows, "attn_chunk.q"}, {k_cache, plane, "attn_chunk.k_cache"},
        {v_cache, plane, "attn_chunk.v_cache"}, {out, rows, "attn_chunk.out"},
        {scratch, (size_t)n * heads * cap * 4u, "attn_chunk.scratch"},
    };
    uint32_t scale_bits;
    memcpy(&scale_bits, &scale, sizeof scale_bits);
    /* window: attend the last `window` positions (sliding window); 0 is
     * full causal attention. */
    uint32_t consts[7] = {pos0, cap, head_dim, group, scale_bits, heads, window};
    /* Split-K kernels when the geometry has an instantiation; decode
     * routes here too (n = 1). One frame, two raw dispatchThreadgroups
     * calls (this direct encoder API is groups - NOT the total-threads
     * encoding of ch_dispatch_args): up to SLICES groups per (KV head,
     * row) write [m, l, acc] partials to scratch (head_dim + 2 floats
     * per (row head, slice); layout [(n, heads, SLICES, head_dim+2)]),
     * then the merge folds them. The kernels derive each row's slice
     * count from its absolute position, so a position's reduction order
     * is identical in chunked prefill and per-token decode. */
    id<MTLComputePipelineState> part = ch_attn_sk_partial_pso(ctx, head_dim, group, f32_cache);
    id<MTLComputePipelineState> merge = ch_attn_sk_merge_pso(ctx, head_dim, group);
    if (part != nil && merge != nil && n == 1u) {
        size_t partial_need = (size_t)n * heads * CH_ATTN_SK_SLICES * ((size_t)head_dim + 2u) * 4u;
        if (ch_check_buffer(ctx, scratch, partial_need, err, err_cap, "attn_sk.partials") !=
            CH_METAL_OK) {
            return CH_METAL_ERR_BOUNDS;
        }
        @autoreleasepool {
            ch_frame f;
            id<MTLComputeCommandEncoder> enc = ch_frame_open(ctx, &f);
            if (enc == nil) {
                return ch_fail(err, err_cap, CH_METAL_ERR_INTERNAL,
                               [NSString stringWithFormat:@"%s: no encoder", what]);
            }
            [f.keep addObject:q->buffer];
            [f.keep addObject:k_cache->buffer];
            [f.keep addObject:v_cache->buffer];
            [f.keep addObject:out->buffer];
            [f.keep addObject:scratch->buffer];
            [enc setComputePipelineState:part];
            [enc setBuffer:q->buffer offset:0 atIndex:0];
            [enc setBuffer:k_cache->buffer offset:0 atIndex:1];
            [enc setBuffer:v_cache->buffer offset:0 atIndex:2];
            [enc setBuffer:scratch->buffer offset:0 atIndex:3];
            [enc setBytes:&pos0 length:sizeof pos0 atIndex:4];
            [enc setBytes:&cap length:sizeof cap atIndex:5];
            [enc setBytes:&head_dim length:sizeof head_dim atIndex:6];
            [enc setBytes:&group length:sizeof group atIndex:7];
            [enc setBytes:&consts[4] length:sizeof consts[4] atIndex:8];
            [enc setBytes:&heads length:sizeof heads atIndex:9];
            [enc setBytes:&window length:sizeof window atIndex:10];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)kv_heads * CH_ATTN_SK_SLICES, n, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc setComputePipelineState:merge];
            [enc setBuffer:scratch->buffer offset:0 atIndex:0];
            [enc setBuffer:out->buffer offset:0 atIndex:1];
            [enc setBytes:&pos0 length:sizeof pos0 atIndex:2];
            [enc setBytes:&head_dim length:sizeof head_dim atIndex:3];
            [enc setBytes:&group length:sizeof group atIndex:4];
            [enc setBytes:&heads length:sizeof heads atIndex:5];
            [enc setBytes:&window length:sizeof window atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)kv_heads, n, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            return ch_frame_commit(ctx, &f, err, err_cap, what);
        }
    }
    /* Prefill (n > 1): one threadgroup per (KV head, row) - the split-K
     * partials round trip costs ~4x a chunk call at full slices, so the
     * chunked path keeps this kernel (llama.cpp makes the same split:
     * flash-decode partials for single rows, one-group-per-row for
     * chunks). The two reduction orders differ, so cross-mode logits are
     * drift-registered (the metal-f16-act precedent); the decode path
     * itself is CPU-reference-qualified at 1e-6. */
    id<MTLComputePipelineState> gqa = ch_attention_gqa_pso(ctx, head_dim, group, f32_cache);
    if (gqa != nil) {
        return ch_dispatch_args(ctx, gqa, bufs, 5, consts, 7,
                                MTLSizeMake((NSUInteger)kv_heads * 128u, n, 1),
                                MTLSizeMake(128, 1, 1), err, err_cap, what);
    }
    return ch_dispatch_args(ctx,
                            f32_cache ? ctx->pso_q35_attention_chunk_f32
                                      : ctx->pso_q35_attention_chunk,
                            bufs, 5, consts, 7, MTLSizeMake((NSUInteger)heads * 128u, n, 1),
                            MTLSizeMake(128, 1, 1), err, err_cap, what);
}

int ch_metal_qwen35_attention_f16_chunk(ch_metal_ctx *ctx, ch_metal_buffer *q,
                                        ch_metal_buffer *k_cache, ch_metal_buffer *v_cache,
                                        ch_metal_buffer *out, ch_metal_buffer *scratch,
                                        uint32_t heads, uint32_t pos0, uint32_t cap,
                                        uint32_t head_dim, uint32_t group, float scale, uint32_t n,
                                        uint32_t window, char *err, size_t err_cap) {
    return ch_attention_chunk(ctx, q, k_cache, v_cache, out, scratch, heads, pos0, cap,
                              head_dim, group, scale, n, window, 0, err, err_cap,
                              "qwen35_attention_f16_chunk");
}

int ch_metal_qwen35_attention_f32_chunk(ch_metal_ctx *ctx, ch_metal_buffer *q,
                                        ch_metal_buffer *k_cache, ch_metal_buffer *v_cache,
                                        ch_metal_buffer *out, ch_metal_buffer *scratch,
                                        uint32_t heads, uint32_t pos0, uint32_t cap,
                                        uint32_t head_dim, uint32_t group, float scale, uint32_t n,
                                        uint32_t window, char *err, size_t err_cap) {
    return ch_attention_chunk(ctx, q, k_cache, v_cache, out, scratch, heads, pos0, cap,
                              head_dim, group, scale, n, window, 1, err, err_cap,
                              "qwen35_attention_f32_chunk");
}

int ch_metal_set_gdn_chunk_generic(ch_metal_ctx *ctx, int generic) {
    if (ctx == NULL) {
        return CH_METAL_ERR_UNSUPPORTED;
    }
    ctx->gdn_chunk_generic = generic ? 1 : 0;
    return CH_METAL_OK;
}

