/*
 * chine_worker_abi.h — narrow C ABI for the Chine execution worker.
 *
 * Version: ABI 1 (must match chine_manifest::BACKEND_ABI_VERSION and the
 * `abi` field of the Hello/HelloAck protocol frames).
 *
 * Design rules (PRD sections 6, 11.3, 15):
 *  - The supervisor owns admission, queues, cancellation policy, and
 *    lifecycle; this ABI only exposes execution.
 *  - No model state or tensor computation crosses the ABI boundary: callbacks
 *    deliver committed tokens and state transitions, nothing else.
 *  - Every function returns a chine_status_t; CH_OK is zero. No function
 *    aborts, and no function may retain pointers after it returns unless the
 *    contract below says so explicitly.
 *  - The worker must never stream draft/speculative text as committed output.
 *
 * The current build's worker implements this ABI in Rust behind the framed
 * protocol (see PROTOCOL.md); native/executor/reference_worker_stub.cpp is a
 * compilable reference of the coupling a real C++/Objective-C++ executor
 * must provide.
 */

#ifndef CHINE_WORKER_ABI_H
#define CHINE_WORKER_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CHINE_ABI_VERSION 1u

typedef enum chine_status {
    CH_OK = 0,
    /* The requested operation is outside the admitted plan. */
    CH_ERR_PLAN = 1,
    /* Context or output overflow; nothing was executed. */
    CH_ERR_CONTEXT = 2,
    /* Designated execution kernels are not available in this build. */
    CH_ERR_KERNELS = 3,
    /* A required transform (e.g. activation rotation) is unimplemented. */
    CH_ERR_TRANSFORM = 4,
    /* Artifact integrity failure discovered during load. */
    CH_ERR_INTEGRITY = 5,
    /* Memory admission failed inside the worker. */
    CH_ERR_MEMORY = 6,
    /* Storage backing the artifact was lost. */
    CH_ERR_STORAGE = 7,
    /* Cancellation was requested and honored at a safe boundary. */
    CH_ERR_CANCELLED = 8,
    /* Internal invariant violation. */
    CH_ERR_INTERNAL = 9,
} chine_status_t;

typedef enum chine_run_state {
    CH_STATE_WARMING = 0,
    CH_STATE_PREFILLING = 1,
    CH_STATE_GENERATING = 2,
    CH_STATE_THROTTLING = 3,
    CH_STATE_PAUSED = 4,
} chine_run_state_t;

/*
 * Engine descriptor. `certified` must be 0 unless the named tests passed for
 * exactly this artifact/profile/device configuration; a worker presenting
 * non-model output as model output is a release-blocking defect.
 */
typedef struct chine_engine_info {
    const char *engine_name;      /* e.g. "reference-test" or "native"      */
    const char *build_id;         /* version + build identity               */
    uint32_t abi_version;         /* must equal CHINE_ABI_VERSION        */
    uint32_t certified;           /* 0/1                                    */
} chine_engine_info_t;

/*
 * Execution callbacks. All callbacks are invoked from the generation thread.
 * Returning CH_OK from on_token acknowledges the committed token; any other
 * value requests a stop at that boundary (the worker reports CH_ERR_CANCELLED
 * if stop_requested is the reason).
 */
typedef chine_status_t (*chine_on_token_fn)(void *ctx,
                                                  uint64_t index,
                                                  const char *utf8_text,
                                                  size_t utf8_len);

typedef void (*chine_on_state_fn)(void *ctx, chine_run_state_t state);

typedef struct chine_callbacks {
    void *ctx;
    chine_on_token_fn on_token;
    chine_on_state_fn on_state;
} chine_callbacks_t;

typedef struct chine_generate_params {
    const uint32_t *prompt_tokens; /* exact token IDs, not text              */
    size_t prompt_len;
    uint64_t max_output_tokens;
    const char *const *stop;       /* NULL-terminated array, may be NULL     */
    uint64_t seed;
} chine_generate_params_t;

/* Opaque worker handle. */
typedef struct chine_worker chine_worker_t;

/* ---- lifecycle ---- */

/* Must be the first call. Returns NULL with *status set on failure. */
chine_worker_t *chine_worker_create(const char *plan_json,
                                          size_t plan_json_len,
                                          chine_status_t *status);

/*
 * Load the plan's artifact. Executors that stream from storage must verify
 * artifact identity hashes here (tensor index + metadata) before executing;
 * a mismatch is CH_ERR_INTEGRITY. `reason_out` receives a static string.
 */
chine_status_t chine_worker_load(chine_worker_t *w,
                                       const char **reason_out);

/* ---- generation ---- */

/*
 * Generate. Honors cancellation (chine_worker_cancel) at committed-token
 * boundaries and reports state transitions through callbacks. Committed
 * tokens only: never draft text.
 */
chine_status_t chine_worker_generate(chine_worker_t *w,
                                           const chine_generate_params_t *params,
                                           const chine_callbacks_t *callbacks,
                                           uint64_t *committed_out);

/* Cooperative cancellation; safe to call from another thread. */
void chine_worker_cancel(chine_worker_t *w);

/* Safe-boundary pause control (governor). Safe to call from another thread. */
void chine_worker_set_paused(chine_worker_t *w, uint32_t paused);

/* ---- introspection / teardown ---- */

chine_status_t chine_worker_info(const chine_worker_t *w,
                                       chine_engine_info_t *info_out);

/* Process footprint attributable to the worker, when measurable. */
chine_status_t chine_worker_footprint(const chine_worker_t *w,
                                            uint64_t *bytes_out);

/*
 * Destroy the worker. Blocks until in-flight work has drained or was safely
 * cancelled; buffers are released only after completion fences. Never returns
 * with work still executing in another thread.
 */
void chine_worker_destroy(chine_worker_t *w);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CHINE_WORKER_ABI_H */
