/*
 * Native CUDA Driver API checkpoint/restore probe.
 *
 * This deliberately does not use Python, PyTorch, ctypes, RTP-LLM, or the
 * cuda-checkpoint command-line utility.  A child process creates a CUDA
 * context, writes a deterministic pattern to device memory, and waits.  The
 * parent calls the checkpoint Driver APIs directly and, after a successful
 * restore/unlock, asks the child to verify that its device memory survived.
 *
 * Build:
 *   gcc -std=c11 -O2 -Wall -Wextra \
 *     -I/usr/local/cuda/include \
 *     sleep_mode_integration/cuda_checkpoint_native_probe.c \
 *     -L/usr/lib64 -Wl,-rpath,/usr/lib64 -lcuda \
 *     -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart -ldl \
 *     -o /tmp/cuda_checkpoint_native_probe
 *
 * Run:
 *   CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 90 \
 *     /tmp/cuda_checkpoint_native_probe
 *
 * Run with the target controlling its own checkpoint, matching NVIDIA's R580
 * migration API sample:
 *   CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 90 \
 *     /tmp/cuda_checkpoint_native_probe --self
 *
 * Retain the primary context for every visible GPU before running the same
 * self-checkpoint sequence. This isolates multi-GPU process support without
 * involving PyTorch, NCCL, symmetric memory, or multicast:
 *   CUDA_VISIBLE_DEVICES=0,1,2,3 timeout --signal=KILL 90 \
 *     /tmp/cuda_checkpoint_native_probe --self-all-devices
 *
 * Keep a live Driver-API pinned-host allocation across checkpoint/restore.
 * This is the native equivalent of a live PyTorch pin_memory() tensor:
 *   RTP_CKPT_PINNED_BYTES=67108864 \
 *   CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 90 \
 *     /tmp/cuda_checkpoint_native_probe --self-pinned-host-live
 *
 * Allocate and use the same pinned-host memory, but release it before the
 * checkpoint. This is the direct control for the live-allocation case:
 *   RTP_CKPT_PINNED_BYTES=4096 \
 *   CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 90 \
 *     /tmp/cuda_checkpoint_native_probe --self-pinned-host-release
 *
 * Keep an ordinary pageable host allocation alive as a negative control:
 *   RTP_CKPT_PINNED_BYTES=67108864 \
 *   CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 90 \
 *     /tmp/cuda_checkpoint_native_probe --self-pageable-host-live
 *
 * Keep the same pinned-host allocation plus a CUDA stream and completed event
 * alive. This isolates the stream/event metadata used by asynchronous copies:
 *   CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 90 \
 *     /tmp/cuda_checkpoint_native_probe --self-pinned-host-event-live
 *
 * Create, use, and destroy 448 streams and events before checkpointing. This
 * mirrors the per-rank channel count observed while the production collective
 * stack was being torn down:
 *   CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 90 \
 *     /tmp/cuda_checkpoint_native_probe --self-channel-churn
 *
 * Keep a FABRIC-exportable VMM allocation alive across the checkpoint:
 *   CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 90 \
 *     /tmp/cuda_checkpoint_native_probe --self-fabric-vmm
 *
 * Keep an import of the allocation alive in a separate holder process:
 *   CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 90 \
 *     /tmp/cuda_checkpoint_native_probe --fabric-external-holder
 *
 * Import a two-GPU FABRIC multicast object from a separate creator, attach the
 * second GPU, release the local multicast handle, and then checkpoint:
 *   CUDA_VISIBLE_DEVICES=0,1 timeout --signal=KILL 90 \
 *     /tmp/cuda_checkpoint_native_probe --multicast-import-release
 *
 * Keep the peer's imported multicast handle live during checkpoint:
 *   CUDA_VISIBLE_DEVICES=0,1 timeout --signal=KILL 90 \
 *     /tmp/cuda_checkpoint_native_probe --multicast-import-live
 *
 * Match the RTP-LLM FABRIC promotion path: retain both the original raw FABRIC
 * import and a second node-local POSIX import, bind/map through the local
 * handle, tear both imports down, and checkpoint while only an external fd
 * keeps the multicast object alive:
 *   CUDA_VISIBLE_DEVICES=0,1 timeout --signal=KILL 90 \
 *     /tmp/cuda_checkpoint_native_probe --multicast-dual-import-bind-release
 *
 * Run the same lifecycle without the peer's raw FABRIC import. This matches
 * the creator rank, which binds a node-local POSIX import of a FABRIC-created
 * object:
 *   CUDA_VISIBLE_DEVICES=0,1 timeout --signal=KILL 90 \
 *     /tmp/cuda_checkpoint_native_probe --multicast-posix-bind-release
 *
 * With torch_memory_saver preloaded, pause a tagged cudaMalloc VMM allocation
 * before checkpointing. This matches RTP-LLM's level-3 weight/KV state:
 *   LD_PRELOAD=/path/to/torch_memory_saver_hook_mode_preload.so \
 *   CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 90 \
 *     /tmp/cuda_checkpoint_native_probe --self-tms-paused-vmm
 */

#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>
#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

enum {
    PATTERN_WORDS = 1024,
    LOCK_TIMEOUT_MS = 10000,
};

static void print_result(const char* operation, CUresult result) {
    const char* name = NULL;
    const char* description = NULL;
    (void)cuGetErrorName(result, &name);
    (void)cuGetErrorString(result, &description);
    printf(
        "%-32s -> %d (%s: %s)\n",
        operation,
        (int)result,
        name != NULL ? name : "unknown",
        description != NULL ? description : "unknown");
}

static int require_success(const char* operation, CUresult result) {
    print_result(operation, result);
    return result == CUDA_SUCCESS;
}

static int require_runtime_success(
    const char* operation,
    cudaError_t result) {
    printf(
        "%-32s -> %d (%s)\n",
        operation,
        (int)result,
        cudaGetErrorString(result));
    return result == cudaSuccess;
}

static int print_state(pid_t pid, const char* label, CUprocessState* state_out) {
    CUprocessState state = CU_PROCESS_STATE_FAILED;
    CUresult result = cuCheckpointProcessGetState((int)pid, &state);
    print_result(label, result);
    if (result != CUDA_SUCCESS) {
        return 0;
    }

    static const char* const state_names[] = {
        "RUNNING",
        "LOCKED",
        "CHECKPOINTED",
        "FAILED",
    };
    const char* state_name = "UNKNOWN";
    if ((unsigned int)state <
        (sizeof(state_names) / sizeof(state_names[0]))) {
        state_name = state_names[state];
    }
    printf("%-32s    state=%d (%s)\n", "", (int)state, state_name);
    if (state_out != NULL) {
        *state_out = state;
    }
    return 1;
}

static int write_all(int fd, const void* buffer, size_t size) {
    const unsigned char* cursor = buffer;
    while (size != 0) {
        ssize_t written = write(fd, cursor, size);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return 0;
        }
        cursor += written;
        size -= (size_t)written;
    }
    return 1;
}

static int read_all(int fd, void* buffer, size_t size) {
    unsigned char* cursor = buffer;
    while (size != 0) {
        ssize_t count = read(fd, cursor, size);
        if (count == 0) {
            return 0;
        }
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            return 0;
        }
        cursor += count;
        size -= (size_t)count;
    }
    return 1;
}

static uint32_t pattern_at(size_t index) {
    return UINT32_C(0xc0da0000) ^ (uint32_t)index;
}

static int run_cuda_child(int ready_fd, int resume_fd) {
    CUdevice device;
    CUcontext context = NULL;
    CUdeviceptr allocation = 0;
    uint32_t expected[PATTERN_WORDS];
    uint32_t observed[PATTERN_WORDS];
    unsigned char signal_byte = 0;

    for (size_t index = 0; index < PATTERN_WORDS; ++index) {
        expected[index] = pattern_at(index);
    }

    if (!require_success("child cuInit", cuInit(0)) ||
        !require_success("child cuDeviceGet", cuDeviceGet(&device, 0)) ||
        !require_success(
            "child cuCtxCreate", cuCtxCreate(&context, NULL, 0, device)) ||
        !require_success(
            "child cuMemAlloc",
            cuMemAlloc(&allocation, sizeof(expected))) ||
        !require_success(
            "child cuMemcpyHtoD",
            cuMemcpyHtoD(allocation, expected, sizeof(expected))) ||
        !require_success("child cuCtxSynchronize", cuCtxSynchronize())) {
        signal_byte = 1;
        (void)write_all(ready_fd, &signal_byte, sizeof(signal_byte));
        return 2;
    }

    printf(
        "child ready: pid=%ld allocation=0x%llx bytes=%zu\n",
        (long)getpid(),
        (unsigned long long)allocation,
        sizeof(expected));
    signal_byte = 0;
    if (!write_all(ready_fd, &signal_byte, sizeof(signal_byte))) {
        perror("child write ready");
        return 3;
    }

    if (!read_all(resume_fd, &signal_byte, sizeof(signal_byte))) {
        perror("child read resume");
        return 4;
    }

    if (!require_success(
            "child cuMemcpyDtoH",
            cuMemcpyDtoH(observed, allocation, sizeof(observed))) ||
        !require_success(
            "child post-restore sync", cuCtxSynchronize())) {
        return 5;
    }

    for (size_t index = 0; index < PATTERN_WORDS; ++index) {
        if (observed[index] != expected[index]) {
            fprintf(
                stderr,
                "device data mismatch at word %zu: expected=0x%08x "
                "observed=0x%08x\n",
                index,
                expected[index],
                observed[index]);
            return 6;
        }
    }

    printf("child verification: PASS (%d words preserved)\n", PATTERN_WORDS);
    (void)cuMemFree(allocation);
    (void)cuCtxDestroy(context);
    return 0;
}

static void kill_and_reap(pid_t child) {
    if (kill(child, SIGKILL) != 0 && errno != ESRCH) {
        perror("kill child");
    }
    while (waitpid(child, NULL, 0) < 0 && errno == EINTR) {
    }
}

static int run_self_probe(int retain_all_devices) {
    CUcontext context = NULL;
    CUdevice* retained_devices = NULL;
    int retained_device_count = 0;
    int visible_device_count = 0;
    CUdeviceptr allocation = 0;
    uint32_t expected[PATTERN_WORDS];
    uint32_t observed[PATTERN_WORDS];
    CUcheckpointLockArgs lock_args = {0};
    CUcheckpointCheckpointArgs checkpoint_args = {0};
    CUcheckpointRestoreArgs restore_args = {0};
    CUcheckpointUnlockArgs unlock_args = {0};
    CUresult lock_result;
    CUresult checkpoint_result = CUDA_ERROR_UNKNOWN;
    CUresult restore_result = CUDA_ERROR_UNKNOWN;
    CUresult unlock_result = CUDA_ERROR_UNKNOWN;

    for (size_t index = 0; index < PATTERN_WORDS; ++index) {
        expected[index] = pattern_at(index);
    }

    if (!require_success("self cuInit", cuInit(0)) ||
        !require_success(
            "self cuDeviceGetCount",
            cuDeviceGetCount(&visible_device_count)) ||
        visible_device_count <= 0) {
        return 2;
    }
    retained_device_count = retain_all_devices ? visible_device_count : 1;
    retained_devices = calloc(
        (size_t)retained_device_count, sizeof(*retained_devices));
    if (retained_devices == NULL) {
        perror("calloc retained devices");
        return 2;
    }
    for (int ordinal = 0; ordinal < retained_device_count; ++ordinal) {
        CUcontext retained_context = NULL;
        char operation[64];
        if (!require_success(
                "self cuDeviceGet",
                cuDeviceGet(&retained_devices[ordinal], ordinal))) {
            free(retained_devices);
            return 2;
        }
        (void)snprintf(
            operation,
            sizeof(operation),
            "self primary context gpu %d",
            ordinal);
        if (!require_success(
                operation,
                cuDevicePrimaryCtxRetain(
                    &retained_context, retained_devices[ordinal]))) {
            free(retained_devices);
            return 2;
        }
        if (ordinal == 0) {
            context = retained_context;
        }
    }

    if (!require_success(
            "self cuCtxSetCurrent", cuCtxSetCurrent(context)) ||
        !require_success(
            "self cuMemAlloc",
            cuMemAlloc(&allocation, sizeof(expected))) ||
        !require_success(
            "self cuMemcpyHtoD",
            cuMemcpyHtoD(allocation, expected, sizeof(expected))) ||
        !require_success(
            "self cuCtxSynchronize", cuCtxSynchronize())) {
        return 2;
    }

    printf(
        "self ready: pid=%ld allocation=0x%llx bytes=%zu retained_devices=%d\n",
        (long)getpid(),
        (unsigned long long)allocation,
        sizeof(expected),
        retained_device_count);

    /*
     * Once this process locks itself, do not call ordinary CUDA APIs such as
     * cuGetErrorName until after Unlock.  Print raw CUresult values while the
     * API lock is held.
     */
    lock_args.timeoutMs = LOCK_TIMEOUT_MS;
    lock_result =
        cuCheckpointProcessLock((int)getpid(), &lock_args);
    printf("self Lock raw result             -> %d\n", (int)lock_result);
    if (lock_result == CUDA_SUCCESS) {
        checkpoint_result = cuCheckpointProcessCheckpoint(
            (int)getpid(), &checkpoint_args);
        printf(
            "self Checkpoint raw result       -> %d\n",
            (int)checkpoint_result);
    }
    if (checkpoint_result == CUDA_SUCCESS) {
        restore_result = cuCheckpointProcessRestore(
            (int)getpid(), &restore_args);
        printf(
            "self Restore raw result          -> %d\n",
            (int)restore_result);
    }
    if (restore_result == CUDA_SUCCESS) {
        unlock_result =
            cuCheckpointProcessUnlock((int)getpid(), &unlock_args);
        printf(
            "self Unlock raw result           -> %d\n",
            (int)unlock_result);
    }
    if (unlock_result != CUDA_SUCCESS) {
        fprintf(
            stderr,
            "RESULT: FAIL - self sequence results=%d/%d/%d/%d\n",
            (int)lock_result,
            (int)checkpoint_result,
            (int)restore_result,
            (int)unlock_result);
        return 3;
    }

    if (!require_success(
            "self cuMemcpyDtoH",
            cuMemcpyDtoH(observed, allocation, sizeof(observed))) ||
        !require_success(
            "self post-restore sync", cuCtxSynchronize())) {
        return 4;
    }
    for (size_t index = 0; index < PATTERN_WORDS; ++index) {
        if (observed[index] != expected[index]) {
            fprintf(
                stderr,
                "self data mismatch at word %zu: expected=0x%08x "
                "observed=0x%08x\n",
                index,
                expected[index],
                observed[index]);
            return 5;
        }
    }

    printf(
        "RESULT: PASS - self checkpoint/restore preserved %d words\n",
        PATTERN_WORDS);
    (void)cuMemFree(allocation);
    for (int ordinal = retained_device_count - 1; ordinal >= 0; --ordinal) {
        (void)cuDevicePrimaryCtxRelease(retained_devices[ordinal]);
    }
    free(retained_devices);
    return 0;
}

static int run_checkpoint_sequence_and_verify(
    const char* label,
    CUdeviceptr allocation,
    const uint32_t* expected,
    uint32_t* observed) {
    CUcheckpointLockArgs lock_args = {0};
    CUcheckpointCheckpointArgs checkpoint_args = {0};
    CUcheckpointRestoreArgs restore_args = {0};
    CUcheckpointUnlockArgs unlock_args = {0};
    CUresult lock_result;
    CUresult checkpoint_result = CUDA_ERROR_UNKNOWN;
    CUresult restore_result = CUDA_ERROR_UNKNOWN;
    CUresult unlock_result = CUDA_ERROR_UNKNOWN;

    lock_args.timeoutMs = LOCK_TIMEOUT_MS;
    lock_result = cuCheckpointProcessLock((int)getpid(), &lock_args);
    printf("%s Lock raw result -> %d\n", label, (int)lock_result);
    if (lock_result == CUDA_SUCCESS) {
        checkpoint_result = cuCheckpointProcessCheckpoint(
            (int)getpid(), &checkpoint_args);
        printf(
            "%s Checkpoint raw result -> %d\n",
            label,
            (int)checkpoint_result);
    }
    if (checkpoint_result == CUDA_SUCCESS) {
        restore_result = cuCheckpointProcessRestore(
            (int)getpid(), &restore_args);
        printf(
            "%s Restore raw result -> %d\n",
            label,
            (int)restore_result);
    }
    if (restore_result == CUDA_SUCCESS) {
        unlock_result =
            cuCheckpointProcessUnlock((int)getpid(), &unlock_args);
        printf(
            "%s Unlock raw result -> %d\n",
            label,
            (int)unlock_result);
    }
    if (unlock_result != CUDA_SUCCESS) {
        fprintf(
            stderr,
            "RESULT: FAIL - %s sequence=%d/%d/%d/%d\n",
            label,
            (int)lock_result,
            (int)checkpoint_result,
            (int)restore_result,
            (int)unlock_result);
        return 3;
    }

    if (!require_success(
            "hypothesis cuMemcpyDtoH",
            cuMemcpyDtoH(
                observed,
                allocation,
                PATTERN_WORDS * sizeof(*observed))) ||
        !require_success(
            "hypothesis post-restore sync", cuCtxSynchronize())) {
        return 4;
    }
    for (size_t index = 0; index < PATTERN_WORDS; ++index) {
        if (observed[index] != expected[index]) {
            fprintf(
                stderr,
                "%s data mismatch at word %zu: expected=0x%08x "
                "observed=0x%08x\n",
                label,
                index,
                expected[index],
                observed[index]);
            return 5;
        }
    }
    printf(
        "RESULT: PASS - %s checkpoint/restore preserved %d words\n",
        label,
        PATTERN_WORDS);
    return 0;
}

static int run_self_pinned_host_probe(
    int keep_stream_and_event,
    int release_before_checkpoint,
    int use_pageable_host_memory) {
    size_t pinned_bytes = 64 * 1024 * 1024;
    const char* pinned_bytes_value = getenv("RTP_CKPT_PINNED_BYTES");
    CUdevice device = -1;
    CUcontext context = NULL;
    CUdeviceptr allocation = 0;
    CUstream stream = NULL;
    CUevent event = NULL;
    void* pinned = NULL;
    uint32_t expected[PATTERN_WORDS];
    uint32_t observed[PATTERN_WORDS];
    const char* label;
    int result = 2;

    if (pinned_bytes_value != NULL && pinned_bytes_value[0] != '\0') {
        char* end = NULL;
        unsigned long long parsed;

        errno = 0;
        parsed = strtoull(pinned_bytes_value, &end, 10);
        if (errno != 0 || end == pinned_bytes_value || *end != '\0' ||
            parsed > SIZE_MAX || parsed < sizeof(expected)) {
            fprintf(
                stderr,
                "RTP_CKPT_PINNED_BYTES must be an integer in [%zu, %zu]\n",
                sizeof(expected),
                (size_t)SIZE_MAX);
            return 2;
        }
        pinned_bytes = (size_t)parsed;
    }

    for (size_t index = 0; index < PATTERN_WORDS; ++index) {
        expected[index] = pattern_at(index);
    }

    if (!require_success("pinned cuInit", cuInit(0)) ||
        !require_success("pinned cuDeviceGet", cuDeviceGet(&device, 0)) ||
        !require_success(
            "pinned primary context",
            cuDevicePrimaryCtxRetain(&context, device)) ||
        !require_success(
            "pinned cuCtxSetCurrent", cuCtxSetCurrent(context)) ||
        !require_success(
            "pinned cuMemAlloc",
            cuMemAlloc(&allocation, sizeof(expected)))) {
        goto cleanup;
    }
    if (use_pageable_host_memory) {
        pinned = malloc(pinned_bytes);
        if (pinned == NULL) {
            perror("pageable host malloc");
            goto cleanup;
        }
        printf("%-32s -> 0\n", "pageable host malloc");
    } else if (!require_success(
                   "pinned cuMemHostAlloc",
                   cuMemHostAlloc(&pinned, pinned_bytes, 0))) {
        goto cleanup;
    }
    memcpy(pinned, expected, sizeof(expected));

    if (keep_stream_and_event) {
        if (!require_success(
                "pinned cuStreamCreate",
                cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING)) ||
            !require_success(
                "pinned cuEventCreate",
                cuEventCreate(&event, CU_EVENT_DISABLE_TIMING)) ||
            !require_success(
                "pinned async HtoD",
                cuMemcpyHtoDAsync(
                    allocation,
                    pinned,
                    sizeof(expected),
                    stream)) ||
            !require_success(
                "pinned cuEventRecord", cuEventRecord(event, stream)) ||
            !require_success(
                "pinned cuEventSynchronize", cuEventSynchronize(event))) {
            goto cleanup;
        }
    } else if (!require_success(
                   "pinned cuMemcpyHtoD",
                   cuMemcpyHtoD(
                       allocation,
                       pinned,
                       sizeof(expected))) ||
               !require_success(
                   "pinned cuCtxSynchronize", cuCtxSynchronize())) {
        goto cleanup;
    }

    if (release_before_checkpoint) {
        if (event != NULL) {
            if (!require_success(
                    "pinned cuEventDestroy", cuEventDestroy(event))) {
                goto cleanup;
            }
            event = NULL;
        }
        if (stream != NULL) {
            if (!require_success(
                    "pinned cuStreamDestroy", cuStreamDestroy(stream))) {
                goto cleanup;
            }
            stream = NULL;
        }
        if (use_pageable_host_memory) {
            free(pinned);
        } else {
            if (!require_success(
                    "pinned cuMemFreeHost", cuMemFreeHost(pinned))) {
                goto cleanup;
            }
        }
        pinned = NULL;
    }

    printf(
        "pinned ready: pid=%ld host_live=%d bytes=%zu stream_event=%d\n",
        (long)getpid(),
        pinned != NULL,
        pinned_bytes,
        keep_stream_and_event);
    if (use_pageable_host_memory) {
        label = "pageable-host-live";
    } else if (release_before_checkpoint) {
        label = "pinned-host-released";
    } else if (keep_stream_and_event) {
        label = "pinned-event-live";
    } else {
        label = "pinned-host-live";
    }
    result = run_checkpoint_sequence_and_verify(
        label, allocation, expected, observed);
    if (result != 0) {
        /*
         * A failed restore leaves the process CHECKPOINTED. CUDA teardown APIs
         * can block forever in that state; process exit is the only safe
         * cleanup path for this negative test.
         */
        return result;
    }

cleanup:
    if (event != NULL) {
        (void)cuEventDestroy(event);
    }
    if (stream != NULL) {
        (void)cuStreamDestroy(stream);
    }
    if (pinned != NULL) {
        if (use_pageable_host_memory) {
            free(pinned);
        } else {
            (void)cuMemFreeHost(pinned);
        }
    }
    if (allocation != 0) {
        (void)cuMemFree(allocation);
    }
    if (device >= 0) {
        (void)cuDevicePrimaryCtxRelease(device);
    }
    return result;
}

static int run_self_channel_churn_probe(void) {
    enum { CHANNEL_OBJECT_COUNT = 448 };
    CUdevice device = -1;
    CUcontext context = NULL;
    CUdeviceptr allocation = 0;
    CUstream* streams = NULL;
    CUevent* events = NULL;
    int created = 0;
    uint32_t expected[PATTERN_WORDS];
    uint32_t observed[PATTERN_WORDS];
    int result = 2;

    for (size_t index = 0; index < PATTERN_WORDS; ++index) {
        expected[index] = pattern_at(index);
    }
    streams = calloc(CHANNEL_OBJECT_COUNT, sizeof(*streams));
    events = calloc(CHANNEL_OBJECT_COUNT, sizeof(*events));
    if (streams == NULL || events == NULL) {
        perror("channel churn calloc");
        goto cleanup;
    }
    if (!require_success("churn cuInit", cuInit(0)) ||
        !require_success("churn cuDeviceGet", cuDeviceGet(&device, 0)) ||
        !require_success(
            "churn primary context",
            cuDevicePrimaryCtxRetain(&context, device)) ||
        !require_success(
            "churn cuCtxSetCurrent", cuCtxSetCurrent(context)) ||
        !require_success(
            "churn cuMemAlloc",
            cuMemAlloc(&allocation, sizeof(expected))) ||
        !require_success(
            "churn cuMemcpyHtoD",
            cuMemcpyHtoD(allocation, expected, sizeof(expected)))) {
        goto cleanup;
    }

    for (; created < CHANNEL_OBJECT_COUNT; ++created) {
        if (!require_success(
                "churn cuStreamCreate",
                cuStreamCreate(
                    &streams[created],
                    CU_STREAM_NON_BLOCKING)) ||
            !require_success(
                "churn cuEventCreate",
                cuEventCreate(
                    &events[created],
                    CU_EVENT_DISABLE_TIMING)) ||
            !require_success(
                "churn cuMemsetD8Async",
                cuMemsetD8Async(
                    allocation,
                    (unsigned char)created,
                    sizeof(expected),
                    streams[created])) ||
            !require_success(
                "churn cuEventRecord",
                cuEventRecord(
                    events[created],
                    streams[created]))) {
            goto cleanup;
        }
    }
    if (!require_success("churn pre-destroy sync", cuCtxSynchronize())) {
        goto cleanup;
    }
    for (int index = created - 1; index >= 0; --index) {
        if (!require_success(
                "churn cuEventDestroy",
                cuEventDestroy(events[index])) ||
            !require_success(
                "churn cuStreamDestroy",
                cuStreamDestroy(streams[index]))) {
            goto cleanup;
        }
        events[index] = NULL;
        streams[index] = NULL;
    }
    created = 0;
    if (!require_success(
            "churn restore pattern",
            cuMemcpyHtoD(allocation, expected, sizeof(expected))) ||
        !require_success(
            "churn post-destroy sync", cuCtxSynchronize())) {
        goto cleanup;
    }

    printf(
        "channel churn ready: pid=%ld destroyed_stream_events=%d\n",
        (long)getpid(),
        CHANNEL_OBJECT_COUNT);
    result = run_checkpoint_sequence_and_verify(
        "channel-churn", allocation, expected, observed);
    if (result != 0) {
        return result;
    }

cleanup:
    for (int index = created - 1; index >= 0; --index) {
        if (events != NULL && events[index] != NULL) {
            (void)cuEventDestroy(events[index]);
        }
        if (streams != NULL && streams[index] != NULL) {
            (void)cuStreamDestroy(streams[index]);
        }
    }
    free(events);
    free(streams);
    if (allocation != 0) {
        (void)cuMemFree(allocation);
    }
    if (device >= 0) {
        (void)cuDevicePrimaryCtxRelease(device);
    }
    return result;
}

static int run_self_fabric_vmm_probe(void) {
    CUdevice device;
    CUcontext context = NULL;
    CUdeviceptr address = 0;
    CUmemGenericAllocationHandle allocation = 0;
    CUmemAllocationProp properties = {0};
    CUmemAccessDesc access = {0};
    CUmemFabricHandle fabric = {0};
    size_t granularity = 0;
    size_t allocation_size = 0;
    uint32_t expected[PATTERN_WORDS];
    uint32_t observed[PATTERN_WORDS];
    CUcheckpointLockArgs lock_args = {0};
    CUcheckpointCheckpointArgs checkpoint_args = {0};
    CUcheckpointRestoreArgs restore_args = {0};
    CUcheckpointUnlockArgs unlock_args = {0};
    CUresult lock_result;
    CUresult checkpoint_result = CUDA_ERROR_UNKNOWN;
    CUresult restore_result = CUDA_ERROR_UNKNOWN;
    CUresult unlock_result = CUDA_ERROR_UNKNOWN;

    for (size_t index = 0; index < PATTERN_WORDS; ++index) {
        expected[index] = pattern_at(index);
    }

    if (!require_success("fabric cuInit", cuInit(0)) ||
        !require_success("fabric cuDeviceGet", cuDeviceGet(&device, 0)) ||
        !require_success(
            "fabric primary context",
            cuDevicePrimaryCtxRetain(&context, device)) ||
        !require_success(
            "fabric cuCtxSetCurrent", cuCtxSetCurrent(context))) {
        return 2;
    }

    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = device;
    properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    if (!require_success(
            "fabric get granularity",
            cuMemGetAllocationGranularity(
                &granularity,
                &properties,
                CU_MEM_ALLOC_GRANULARITY_MINIMUM)) ||
        granularity == 0) {
        return 2;
    }
    allocation_size =
        ((sizeof(expected) + granularity - 1) / granularity) * granularity;
    if (!require_success(
            "fabric address reserve",
            cuMemAddressReserve(
                &address, allocation_size, granularity, 0, 0)) ||
        !require_success(
            "fabric allocation create",
            cuMemCreate(
                &allocation, allocation_size, &properties, 0)) ||
        !require_success(
            "fabric memory map",
            cuMemMap(address, allocation_size, 0, allocation, 0))) {
        return 2;
    }

    access.location = properties.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (!require_success(
            "fabric set access",
            cuMemSetAccess(address, allocation_size, &access, 1)) ||
        !require_success(
            "fabric cuMemcpyHtoD",
            cuMemcpyHtoD(address, expected, sizeof(expected))) ||
        !require_success(
            "fabric cuCtxSynchronize", cuCtxSynchronize()) ||
        !require_success(
            "fabric export handle",
            cuMemExportToShareableHandle(
                &fabric,
                allocation,
                CU_MEM_HANDLE_TYPE_FABRIC,
                0))) {
        return 2;
    }

    printf(
        "fabric ready: pid=%ld address=0x%llx allocation_size=%zu "
        "fabric_bytes=%zu\n",
        (long)getpid(),
        (unsigned long long)address,
        allocation_size,
        sizeof(fabric));

    lock_args.timeoutMs = LOCK_TIMEOUT_MS;
    lock_result = cuCheckpointProcessLock((int)getpid(), &lock_args);
    printf("fabric Lock raw result           -> %d\n", (int)lock_result);
    if (lock_result == CUDA_SUCCESS) {
        checkpoint_result = cuCheckpointProcessCheckpoint(
            (int)getpid(), &checkpoint_args);
        printf(
            "fabric Checkpoint raw result     -> %d\n",
            (int)checkpoint_result);
    }
    if (checkpoint_result == CUDA_SUCCESS) {
        restore_result = cuCheckpointProcessRestore(
            (int)getpid(), &restore_args);
        printf(
            "fabric Restore raw result        -> %d\n",
            (int)restore_result);
    }
    if (restore_result == CUDA_SUCCESS) {
        unlock_result =
            cuCheckpointProcessUnlock((int)getpid(), &unlock_args);
        printf(
            "fabric Unlock raw result         -> %d\n",
            (int)unlock_result);
    }
    if (unlock_result != CUDA_SUCCESS) {
        fprintf(
            stderr,
            "RESULT: FAIL - FABRIC VMM sequence results=%d/%d/%d/%d\n",
            (int)lock_result,
            (int)checkpoint_result,
            (int)restore_result,
            (int)unlock_result);
        return 3;
    }

    if (!require_success(
            "fabric cuMemcpyDtoH",
            cuMemcpyDtoH(observed, address, sizeof(observed))) ||
        !require_success(
            "fabric post-restore sync", cuCtxSynchronize())) {
        return 4;
    }
    for (size_t index = 0; index < PATTERN_WORDS; ++index) {
        if (observed[index] != expected[index]) {
            fprintf(
                stderr,
                "FABRIC VMM data mismatch at word %zu: expected=0x%08x "
                "observed=0x%08x\n",
                index,
                expected[index],
                observed[index]);
            return 5;
        }
    }

    printf(
        "RESULT: PASS - FABRIC VMM checkpoint/restore preserved %d words\n",
        PATTERN_WORDS);
    (void)cuMemUnmap(address, allocation_size);
    (void)cuMemAddressFree(address, allocation_size);
    (void)cuMemRelease(allocation);
    (void)cuDevicePrimaryCtxRelease(device);
    return 0;
}

typedef struct {
    int status;
    CUmemFabricHandle fabric;
} fabric_owner_message;

static int run_fabric_owner_child(int message_fd, int command_fd) {
    CUdevice device;
    CUcontext context = NULL;
    CUdeviceptr address = 0;
    CUmemGenericAllocationHandle allocation = 0;
    CUmemAllocationProp properties = {0};
    CUmemAccessDesc access = {0};
    size_t granularity = 0;
    size_t allocation_size = 0;
    uint32_t expected[PATTERN_WORDS];
    uint32_t observed[PATTERN_WORDS];
    fabric_owner_message message = {.status = 1};
    unsigned char command = 1;
    CUcheckpointLockArgs lock_args = {0};
    CUcheckpointCheckpointArgs checkpoint_args = {0};
    CUcheckpointRestoreArgs restore_args = {0};
    CUcheckpointUnlockArgs unlock_args = {0};
    CUresult lock_result;
    CUresult checkpoint_result = CUDA_ERROR_UNKNOWN;
    CUresult restore_result = CUDA_ERROR_UNKNOWN;
    CUresult unlock_result = CUDA_ERROR_UNKNOWN;

    for (size_t index = 0; index < PATTERN_WORDS; ++index) {
        expected[index] = pattern_at(index);
    }

    if (!require_success("owner cuInit", cuInit(0)) ||
        !require_success("owner cuDeviceGet", cuDeviceGet(&device, 0)) ||
        !require_success(
            "owner primary context",
            cuDevicePrimaryCtxRetain(&context, device)) ||
        !require_success(
            "owner cuCtxSetCurrent", cuCtxSetCurrent(context))) {
        (void)write_all(message_fd, &message, sizeof(message));
        return 2;
    }
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = device;
    properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    if (!require_success(
            "owner get granularity",
            cuMemGetAllocationGranularity(
                &granularity,
                &properties,
                CU_MEM_ALLOC_GRANULARITY_MINIMUM)) ||
        granularity == 0) {
        (void)write_all(message_fd, &message, sizeof(message));
        return 2;
    }
    allocation_size =
        ((sizeof(expected) + granularity - 1) / granularity) * granularity;
    if (!require_success(
            "owner address reserve",
            cuMemAddressReserve(
                &address, allocation_size, granularity, 0, 0)) ||
        !require_success(
            "owner allocation create",
            cuMemCreate(
                &allocation, allocation_size, &properties, 0)) ||
        !require_success(
            "owner memory map",
            cuMemMap(address, allocation_size, 0, allocation, 0))) {
        (void)write_all(message_fd, &message, sizeof(message));
        return 2;
    }
    access.location = properties.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (!require_success(
            "owner set access",
            cuMemSetAccess(address, allocation_size, &access, 1)) ||
        !require_success(
            "owner cuMemcpyHtoD",
            cuMemcpyHtoD(address, expected, sizeof(expected))) ||
        !require_success(
            "owner cuCtxSynchronize", cuCtxSynchronize()) ||
        !require_success(
            "owner export handle",
            cuMemExportToShareableHandle(
                &message.fabric,
                allocation,
                CU_MEM_HANDLE_TYPE_FABRIC,
                0))) {
        (void)write_all(message_fd, &message, sizeof(message));
        return 2;
    }

    message.status = 0;
    printf(
        "owner ready: pid=%ld address=0x%llx allocation_size=%zu\n",
        (long)getpid(),
        (unsigned long long)address,
        allocation_size);
    if (!write_all(message_fd, &message, sizeof(message)) ||
        !read_all(command_fd, &command, sizeof(command)) ||
        command != 0) {
        fprintf(stderr, "owner/holder handshake failed\n");
        return 3;
    }

    lock_args.timeoutMs = LOCK_TIMEOUT_MS;
    lock_result = cuCheckpointProcessLock((int)getpid(), &lock_args);
    printf("owner Lock raw result            -> %d\n", (int)lock_result);
    if (lock_result == CUDA_SUCCESS) {
        checkpoint_result = cuCheckpointProcessCheckpoint(
            (int)getpid(), &checkpoint_args);
        printf(
            "owner Checkpoint raw result      -> %d\n",
            (int)checkpoint_result);
    }
    if (checkpoint_result == CUDA_SUCCESS) {
        restore_result = cuCheckpointProcessRestore(
            (int)getpid(), &restore_args);
        printf(
            "owner Restore raw result         -> %d\n",
            (int)restore_result);
    }
    if (restore_result == CUDA_SUCCESS) {
        unlock_result =
            cuCheckpointProcessUnlock((int)getpid(), &unlock_args);
        printf(
            "owner Unlock raw result          -> %d\n",
            (int)unlock_result);
    }
    if (unlock_result != CUDA_SUCCESS) {
        fprintf(
            stderr,
            "RESULT: FAIL - external-holder owner sequence=%d/%d/%d/%d\n",
            (int)lock_result,
            (int)checkpoint_result,
            (int)restore_result,
            (int)unlock_result);
        return 4;
    }

    if (!require_success(
            "owner cuMemcpyDtoH",
            cuMemcpyDtoH(observed, address, sizeof(observed))) ||
        !require_success(
            "owner post-restore sync", cuCtxSynchronize())) {
        return 5;
    }
    for (size_t index = 0; index < PATTERN_WORDS; ++index) {
        if (observed[index] != expected[index]) {
            fprintf(
                stderr,
                "external-holder data mismatch at word %zu: expected=0x%08x "
                "observed=0x%08x\n",
                index,
                expected[index],
                observed[index]);
            return 6;
        }
    }

    (void)cuMemUnmap(address, allocation_size);
    (void)cuMemAddressFree(address, allocation_size);
    (void)cuMemRelease(allocation);
    (void)cuDevicePrimaryCtxRelease(device);
    return 0;
}

static int run_fabric_external_holder_probe(void) {
    int message_pipe[2];
    int command_pipe[2];
    pid_t owner;
    fabric_owner_message message = {.status = 1};
    unsigned char command = 1;
    CUdevice device;
    CUcontext context = NULL;
    CUmemGenericAllocationHandle imported = 0;

    if (pipe(message_pipe) != 0 || pipe(command_pipe) != 0) {
        perror("fabric holder pipe");
        return 2;
    }
    owner = fork();
    if (owner < 0) {
        perror("fabric holder fork");
        return 2;
    }
    if (owner == 0) {
        int child_result;
        close(message_pipe[0]);
        close(command_pipe[1]);
        child_result =
            run_fabric_owner_child(message_pipe[1], command_pipe[0]);
        close(message_pipe[1]);
        close(command_pipe[0]);
        fflush(NULL);
        _exit(child_result);
    }

    close(message_pipe[1]);
    close(command_pipe[0]);
    if (!read_all(message_pipe[0], &message, sizeof(message)) ||
        message.status != 0) {
        fprintf(stderr, "owner failed before exporting FABRIC handle\n");
        close(message_pipe[0]);
        close(command_pipe[1]);
        kill_and_reap(owner);
        return 3;
    }
    close(message_pipe[0]);

    if (!require_success("holder cuInit", cuInit(0)) ||
        !require_success("holder cuDeviceGet", cuDeviceGet(&device, 0)) ||
        !require_success(
            "holder primary context",
            cuDevicePrimaryCtxRetain(&context, device)) ||
        !require_success(
            "holder cuCtxSetCurrent", cuCtxSetCurrent(context)) ||
        !require_success(
            "holder import FABRIC",
            cuMemImportFromShareableHandle(
                &imported,
                &message.fabric,
                CU_MEM_HANDLE_TYPE_FABRIC))) {
        close(command_pipe[1]);
        kill_and_reap(owner);
        return 4;
    }
    printf(
        "holder imported: pid=%ld allocation=0x%llx owner=%ld\n",
        (long)getpid(),
        (unsigned long long)imported,
        (long)owner);
    command = 0;
    if (!write_all(command_pipe[1], &command, sizeof(command))) {
        perror("holder signal owner");
        close(command_pipe[1]);
        kill_and_reap(owner);
        return 5;
    }
    close(command_pipe[1]);

    int wait_status = 0;
    while (waitpid(owner, &wait_status, 0) < 0) {
        if (errno != EINTR) {
            perror("holder waitpid");
            return 6;
        }
    }
    (void)cuMemRelease(imported);
    (void)cuDevicePrimaryCtxRelease(device);
    if (!WIFEXITED(wait_status) || WEXITSTATUS(wait_status) != 0) {
        fprintf(
            stderr,
            "RESULT: FAIL - FABRIC owner status=0x%x\n",
            wait_status);
        return 7;
    }

    printf(
        "RESULT: PASS - external FABRIC holder survived owner "
        "checkpoint/restore\n");
    return 0;
}

typedef struct {
    int status;
    size_t size;
    CUmemFabricHandle fabric;
} multicast_creator_message;

static int send_multicast_message_with_fd(
    int socket_fd,
    const multicast_creator_message* message,
    int shareable_fd) {
    struct iovec payload = {
        .iov_base = (void*)message,
        .iov_len = sizeof(*message),
    };
    unsigned char control[CMSG_SPACE(sizeof(shareable_fd))] = {0};
    struct msghdr header = {
        .msg_iov = &payload,
        .msg_iovlen = 1,
        .msg_control = control,
        .msg_controllen = sizeof(control),
    };
    struct cmsghdr* control_header = CMSG_FIRSTHDR(&header);
    control_header->cmsg_level = SOL_SOCKET;
    control_header->cmsg_type = SCM_RIGHTS;
    control_header->cmsg_len = CMSG_LEN(sizeof(shareable_fd));
    memcpy(CMSG_DATA(control_header), &shareable_fd, sizeof(shareable_fd));
    return sendmsg(socket_fd, &header, 0) == (ssize_t)sizeof(*message);
}

static int receive_multicast_message_with_fd(
    int socket_fd,
    multicast_creator_message* message,
    int* shareable_fd) {
    struct iovec payload = {
        .iov_base = message,
        .iov_len = sizeof(*message),
    };
    unsigned char control[CMSG_SPACE(sizeof(*shareable_fd))] = {0};
    struct msghdr header = {
        .msg_iov = &payload,
        .msg_iovlen = 1,
        .msg_control = control,
        .msg_controllen = sizeof(control),
    };
    ssize_t received = recvmsg(socket_fd, &header, 0);
    struct cmsghdr* control_header = CMSG_FIRSTHDR(&header);
    if (received != (ssize_t)sizeof(*message) ||
        control_header == NULL ||
        control_header->cmsg_level != SOL_SOCKET ||
        control_header->cmsg_type != SCM_RIGHTS ||
        control_header->cmsg_len != CMSG_LEN(sizeof(*shareable_fd))) {
        return 0;
    }
    memcpy(
        shareable_fd,
        CMSG_DATA(control_header),
        sizeof(*shareable_fd));
    return *shareable_fd >= 0;
}

static int run_multicast_import_child(
    int message_fd,
    int release_before_checkpoint) {
    multicast_creator_message message = {.status = 1};
    CUdevice device;
    CUcontext context = NULL;
    CUmemGenericAllocationHandle multicast = 0;
    CUdeviceptr allocation = 0;
    uint32_t expected[PATTERN_WORDS];
    uint32_t observed[PATTERN_WORDS];
    CUcheckpointLockArgs lock_args = {0};
    CUcheckpointCheckpointArgs checkpoint_args = {0};
    CUcheckpointRestoreArgs restore_args = {0};
    CUcheckpointUnlockArgs unlock_args = {0};
    CUresult lock_result;
    CUresult checkpoint_result = CUDA_ERROR_UNKNOWN;
    CUresult restore_result = CUDA_ERROR_UNKNOWN;
    CUresult unlock_result = CUDA_ERROR_UNKNOWN;

    if (!read_all(message_fd, &message, sizeof(message)) ||
        message.status != 0) {
        fprintf(stderr, "multicast creator failed before export\n");
        return 2;
    }
    for (size_t index = 0; index < PATTERN_WORDS; ++index) {
        expected[index] = pattern_at(index);
    }
    if (!require_success("mc peer cuInit", cuInit(0)) ||
        !require_success("mc peer cuDeviceGet", cuDeviceGet(&device, 1)) ||
        !require_success(
            "mc peer primary context",
            cuDevicePrimaryCtxRetain(&context, device)) ||
        !require_success(
            "mc peer cuCtxSetCurrent", cuCtxSetCurrent(context)) ||
        !require_success(
            "mc peer cuMemAlloc",
            cuMemAlloc(&allocation, sizeof(expected))) ||
        !require_success(
            "mc peer cuMemcpyHtoD",
            cuMemcpyHtoD(allocation, expected, sizeof(expected))) ||
        !require_success(
            "mc peer import FABRIC",
            cuMemImportFromShareableHandle(
                &multicast,
                &message.fabric,
                CU_MEM_HANDLE_TYPE_FABRIC)) ||
        !require_success(
            "mc peer add device",
            cuMulticastAddDevice(multicast, device)) ||
        !require_success(
            "mc peer pre-checkpoint sync", cuCtxSynchronize())) {
        return 3;
    }
    if (release_before_checkpoint) {
        if (!require_success(
                "mc peer release import",
                cuMemRelease(multicast))) {
            return 3;
        }
        multicast = 0;
    }
    printf(
        "mc peer import state: pid=%ld device=%d multicast_size=%zu live=%d\n",
        (long)getpid(),
        (int)device,
        message.size,
        multicast != 0);

    lock_args.timeoutMs = LOCK_TIMEOUT_MS;
    lock_result = cuCheckpointProcessLock((int)getpid(), &lock_args);
    printf("mc peer Lock raw result          -> %d\n", (int)lock_result);
    if (lock_result == CUDA_SUCCESS) {
        checkpoint_result = cuCheckpointProcessCheckpoint(
            (int)getpid(), &checkpoint_args);
        printf(
            "mc peer Checkpoint raw result    -> %d\n",
            (int)checkpoint_result);
    }
    if (checkpoint_result == CUDA_SUCCESS) {
        restore_result = cuCheckpointProcessRestore(
            (int)getpid(), &restore_args);
        printf(
            "mc peer Restore raw result       -> %d\n",
            (int)restore_result);
    }
    if (restore_result == CUDA_SUCCESS) {
        unlock_result =
            cuCheckpointProcessUnlock((int)getpid(), &unlock_args);
        printf(
            "mc peer Unlock raw result        -> %d\n",
            (int)unlock_result);
    }
    if (unlock_result != CUDA_SUCCESS) {
        fprintf(
            stderr,
            "RESULT: FAIL - multicast import/release sequence=%d/%d/%d/%d\n",
            (int)lock_result,
            (int)checkpoint_result,
            (int)restore_result,
            (int)unlock_result);
        return 4;
    }

    if (!require_success(
            "mc peer cuMemcpyDtoH",
            cuMemcpyDtoH(observed, allocation, sizeof(observed))) ||
        !require_success(
            "mc peer post-restore sync", cuCtxSynchronize())) {
        return 5;
    }
    for (size_t index = 0; index < PATTERN_WORDS; ++index) {
        if (observed[index] != expected[index]) {
            fprintf(
                stderr,
                "multicast peer data mismatch at word %zu: "
                "expected=0x%08x observed=0x%08x\n",
                index,
                expected[index],
                observed[index]);
            return 6;
        }
    }
    (void)cuMemFree(allocation);
    if (multicast != 0) {
        (void)cuMemRelease(multicast);
    }
    (void)cuDevicePrimaryCtxRelease(device);
    return 0;
}

static int run_multicast_import_probe(int release_before_checkpoint) {
    int message_pipe[2];
    pid_t peer;
    multicast_creator_message message = {.status = 1};
    CUdevice device;
    CUcontext context = NULL;
    CUmemGenericAllocationHandle multicast = 0;
    CUmulticastObjectProp properties = {0};
    size_t granularity = 0;

    if (pipe(message_pipe) != 0) {
        perror("multicast creator pipe");
        return 2;
    }
    peer = fork();
    if (peer < 0) {
        perror("multicast creator fork");
        return 2;
    }
    if (peer == 0) {
        int child_result;
        close(message_pipe[1]);
        child_result = run_multicast_import_child(
            message_pipe[0], release_before_checkpoint);
        close(message_pipe[0]);
        fflush(NULL);
        _exit(child_result);
    }
    close(message_pipe[0]);

    if (!require_success("mc creator cuInit", cuInit(0)) ||
        !require_success(
            "mc creator cuDeviceGet", cuDeviceGet(&device, 0)) ||
        !require_success(
            "mc creator primary context",
            cuDevicePrimaryCtxRetain(&context, device)) ||
        !require_success(
            "mc creator cuCtxSetCurrent", cuCtxSetCurrent(context))) {
        (void)write_all(message_pipe[1], &message, sizeof(message));
        close(message_pipe[1]);
        kill_and_reap(peer);
        return 3;
    }

    properties.numDevices = 2;
    properties.size = 2 * 1024 * 1024;
    properties.handleTypes =
        CU_MEM_HANDLE_TYPE_FABRIC |
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    if (!require_success(
            "mc creator get granularity",
            cuMulticastGetGranularity(
                &granularity,
                &properties,
                CU_MULTICAST_GRANULARITY_RECOMMENDED)) ||
        granularity == 0) {
        (void)write_all(message_pipe[1], &message, sizeof(message));
        close(message_pipe[1]);
        kill_and_reap(peer);
        return 3;
    }
    properties.size =
        ((properties.size + granularity - 1) / granularity) * granularity;
    if (!require_success(
            "mc creator create",
            cuMulticastCreate(&multicast, &properties)) ||
        !require_success(
            "mc creator add device",
            cuMulticastAddDevice(multicast, device)) ||
        !require_success(
            "mc creator export FABRIC",
            cuMemExportToShareableHandle(
                &message.fabric,
                multicast,
                CU_MEM_HANDLE_TYPE_FABRIC,
                0))) {
        (void)write_all(message_pipe[1], &message, sizeof(message));
        close(message_pipe[1]);
        kill_and_reap(peer);
        return 3;
    }
    message.status = 0;
    message.size = properties.size;
    printf(
        "mc creator ready: pid=%ld peer=%ld device=%d size=%zu "
        "granularity=%zu\n",
        (long)getpid(),
        (long)peer,
        (int)device,
        properties.size,
        granularity);
    if (!write_all(message_pipe[1], &message, sizeof(message))) {
        perror("multicast creator send");
        close(message_pipe[1]);
        kill_and_reap(peer);
        return 4;
    }
    close(message_pipe[1]);

    int wait_status = 0;
    while (waitpid(peer, &wait_status, 0) < 0) {
        if (errno != EINTR) {
            perror("multicast creator waitpid");
            return 5;
        }
    }
    (void)cuMemRelease(multicast);
    (void)cuDevicePrimaryCtxRelease(device);
    if (!WIFEXITED(wait_status) || WEXITSTATUS(wait_status) != 0) {
        fprintf(
            stderr,
            "RESULT: FAIL - multicast peer status=0x%x\n",
            wait_status);
        return 6;
    }

    printf(
        "RESULT: PASS - %s FABRIC multicast import survived "
        "checkpoint/restore\n",
        release_before_checkpoint ? "released" : "live");
    return 0;
}

static int run_multicast_dual_import_child(
    int socket_fd,
    int import_raw_fabric) {
    multicast_creator_message message = {.status = 1};
    int multicast_fd = -1;
    CUdevice devices[2] = {-1, -1};
    CUcontext contexts[2] = {NULL, NULL};
    CUmemGenericAllocationHandle raw_multicast = 0;
    CUmemGenericAllocationHandle local_multicast = 0;
    CUmemGenericAllocationHandle backing = 0;
    CUdeviceptr multicast_address = 0;
    CUdeviceptr ordinary_allocation = 0;
    CUmemAllocationProp backing_properties = {0};
    CUmemAccessDesc access = {0};
    uint32_t expected[PATTERN_WORDS];
    uint32_t observed[PATTERN_WORDS];
    unsigned char signal = 1;
    CUcheckpointLockArgs lock_args = {0};
    CUcheckpointCheckpointArgs checkpoint_args = {0};
    CUcheckpointRestoreArgs restore_args = {0};
    CUcheckpointUnlockArgs unlock_args = {0};
    CUresult lock_result;
    CUresult checkpoint_result = CUDA_ERROR_UNKNOWN;
    CUresult restore_result = CUDA_ERROR_UNKNOWN;
    CUresult unlock_result = CUDA_ERROR_UNKNOWN;

    if (!receive_multicast_message_with_fd(
            socket_fd, &message, &multicast_fd) ||
        message.status != 0 || message.size == 0) {
        fprintf(stderr, "dual-import peer failed to receive holder handles\n");
        return 2;
    }
    for (size_t index = 0; index < PATTERN_WORDS; ++index) {
        expected[index] = pattern_at(index);
    }

    if (!require_success("dual peer cuInit", cuInit(0))) {
        return 3;
    }
    for (int ordinal = 0; ordinal < 2; ++ordinal) {
        char operation[64];
        if (!require_success(
                "dual peer cuDeviceGet",
                cuDeviceGet(&devices[ordinal], ordinal))) {
            return 3;
        }
        (void)snprintf(
            operation,
            sizeof(operation),
            "dual peer primary context gpu %d",
            ordinal);
        if (!require_success(
                operation,
                cuDevicePrimaryCtxRetain(
                    &contexts[ordinal], devices[ordinal]))) {
            return 3;
        }
    }
    if (!require_success(
            "dual peer cuCtxSetCurrent", cuCtxSetCurrent(contexts[1])) ||
        (import_raw_fabric &&
         !require_success(
             "dual peer import raw FABRIC",
             cuMemImportFromShareableHandle(
                 &raw_multicast,
                 &message.fabric,
                 CU_MEM_HANDLE_TYPE_FABRIC))) ||
        !require_success(
            "dual peer import local POSIX",
            cuMemImportFromShareableHandle(
                &local_multicast,
                (void*)(intptr_t)multicast_fd,
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR))) {
        return 3;
    }

    backing_properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    backing_properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    backing_properties.location.id = devices[1];
    backing_properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    if (!require_success(
            "dual peer backing create",
            cuMemCreate(
                &backing,
                message.size,
                &backing_properties,
                0)) ||
        !require_success(
            "dual peer multicast bind",
            cuMulticastBindMem(
                local_multicast,
                0,
                backing,
                0,
                message.size,
                0)) ||
        !require_success(
            "dual peer address reserve",
            cuMemAddressReserve(
                &multicast_address,
                message.size,
                0,
                0,
                0)) ||
        !require_success(
            "dual peer multicast map",
            cuMemMap(
                multicast_address,
                message.size,
                0,
                local_multicast,
                0))) {
        return 3;
    }
    access.location = backing_properties.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (!require_success(
            "dual peer multicast access",
            cuMemSetAccess(
                multicast_address,
                message.size,
                &access,
                1)) ||
        !require_success(
            "dual peer multicast memset",
            cuMemsetD8(multicast_address, 0x5a, message.size)) ||
        !require_success(
            "dual peer multicast sync", cuCtxSynchronize()) ||
        !require_success(
            "dual peer ordinary allocation",
            cuMemAlloc(&ordinary_allocation, sizeof(expected))) ||
        !require_success(
            "dual peer ordinary write",
            cuMemcpyHtoD(
                ordinary_allocation, expected, sizeof(expected)))) {
        return 3;
    }

    if (!require_success(
            "dual peer multicast unmap",
            cuMemUnmap(multicast_address, message.size)) ||
        !require_success(
            "dual peer address free",
            cuMemAddressFree(multicast_address, message.size)) ||
        !require_success(
            "dual peer multicast unbind",
            cuMulticastUnbind(
                local_multicast,
                devices[1],
                0,
                message.size)) ||
        !require_success(
            "dual peer backing release",
            cuMemRelease(backing)) ||
        !require_success(
            "dual peer local release",
            cuMemRelease(local_multicast)) ||
        (import_raw_fabric &&
         !require_success(
             "dual peer raw release",
             cuMemRelease(raw_multicast))) ||
        !require_success(
            "dual peer teardown sync", cuCtxSynchronize())) {
        return 3;
    }
    multicast_address = 0;
    backing = 0;
    local_multicast = 0;
    raw_multicast = 0;
    close(multicast_fd);
    multicast_fd = -1;

    signal = 0;
    if (!write_all(socket_fd, &signal, sizeof(signal)) ||
        !read_all(socket_fd, &signal, sizeof(signal)) ||
        signal != 0) {
        fprintf(stderr, "dual-import holder teardown handshake failed\n");
        return 4;
    }
    printf(
        "%s peer torn down: pid=%ld size=%zu; external fd only\n",
        import_raw_fabric ? "dual-import" : "POSIX-import",
        (long)getpid(),
        message.size);

    lock_args.timeoutMs = LOCK_TIMEOUT_MS;
    lock_result = cuCheckpointProcessLock((int)getpid(), &lock_args);
    printf("dual peer Lock raw result        -> %d\n", (int)lock_result);
    if (lock_result == CUDA_SUCCESS) {
        checkpoint_result = cuCheckpointProcessCheckpoint(
            (int)getpid(), &checkpoint_args);
        printf(
            "dual peer Checkpoint raw result  -> %d\n",
            (int)checkpoint_result);
    }
    if (checkpoint_result == CUDA_SUCCESS) {
        restore_result = cuCheckpointProcessRestore(
            (int)getpid(), &restore_args);
        printf(
            "dual peer Restore raw result     -> %d\n",
            (int)restore_result);
    }
    if (restore_result == CUDA_SUCCESS) {
        unlock_result =
            cuCheckpointProcessUnlock((int)getpid(), &unlock_args);
        printf(
            "dual peer Unlock raw result      -> %d\n",
            (int)unlock_result);
    }
    if (unlock_result != CUDA_SUCCESS) {
        fprintf(
            stderr,
            "RESULT: FAIL - %s multicast sequence=%d/%d/%d/%d\n",
            import_raw_fabric ? "dual-import" : "POSIX-import",
            (int)lock_result,
            (int)checkpoint_result,
            (int)restore_result,
            (int)unlock_result);
        return 5;
    }

    if (!require_success(
            "dual peer ordinary read",
            cuMemcpyDtoH(
                observed, ordinary_allocation, sizeof(observed))) ||
        !require_success(
            "dual peer post-restore sync", cuCtxSynchronize())) {
        return 6;
    }
    for (size_t index = 0; index < PATTERN_WORDS; ++index) {
        if (observed[index] != expected[index]) {
            fprintf(
                stderr,
                "dual-import data mismatch at word %zu: "
                "expected=0x%08x observed=0x%08x\n",
                index,
                expected[index],
                observed[index]);
            return 7;
        }
    }
    (void)cuMemFree(ordinary_allocation);
    (void)cuDevicePrimaryCtxRelease(devices[1]);
    (void)cuDevicePrimaryCtxRelease(devices[0]);
    printf(
        "RESULT: PASS - released %s multicast import path survived "
        "checkpoint/restore\n",
        import_raw_fabric ? "dual FABRIC/POSIX" : "POSIX-only FABRIC-object");
    return 0;
}

static int run_multicast_dual_import_bind_release_probe(
    int import_raw_fabric) {
    int peer_socket[2];
    pid_t peer;
    multicast_creator_message message = {.status = 1};
    CUdevice devices[2] = {-1, -1};
    CUcontext contexts[2] = {NULL, NULL};
    CUmemGenericAllocationHandle multicast = 0;
    CUmemGenericAllocationHandle backing = 0;
    CUmulticastObjectProp properties = {0};
    CUmemAllocationProp backing_properties = {0};
    CUmemFabricHandle fabric = {0};
    int multicast_fd = -1;
    size_t granularity = 0;
    unsigned char signal = 1;

    if (socketpair(AF_UNIX, SOCK_STREAM, 0, peer_socket) != 0) {
        perror("dual-import socketpair");
        return 2;
    }
    peer = fork();
    if (peer < 0) {
        perror("dual-import fork");
        return 2;
    }
    if (peer == 0) {
        int child_result;
        close(peer_socket[0]);
        child_result =
            run_multicast_dual_import_child(
                peer_socket[1], import_raw_fabric);
        close(peer_socket[1]);
        fflush(NULL);
        _exit(child_result);
    }
    close(peer_socket[1]);

    if (!require_success("dual holder cuInit", cuInit(0))) {
        kill_and_reap(peer);
        return 3;
    }
    for (int ordinal = 0; ordinal < 2; ++ordinal) {
        char operation[64];
        if (!require_success(
                "dual holder cuDeviceGet",
                cuDeviceGet(&devices[ordinal], ordinal))) {
            kill_and_reap(peer);
            return 3;
        }
        (void)snprintf(
            operation,
            sizeof(operation),
            "dual holder primary context gpu %d",
            ordinal);
        if (!require_success(
                operation,
                cuDevicePrimaryCtxRetain(
                    &contexts[ordinal], devices[ordinal]))) {
            kill_and_reap(peer);
            return 3;
        }
    }
    if (!require_success(
            "dual holder cuCtxSetCurrent", cuCtxSetCurrent(contexts[0]))) {
        kill_and_reap(peer);
        return 3;
    }

    properties.numDevices = 2;
    properties.size = 2 * 1024 * 1024;
    properties.handleTypes =
        CU_MEM_HANDLE_TYPE_FABRIC |
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    if (!require_success(
            "dual holder get granularity",
            cuMulticastGetGranularity(
                &granularity,
                &properties,
                CU_MULTICAST_GRANULARITY_RECOMMENDED)) ||
        granularity == 0) {
        kill_and_reap(peer);
        return 3;
    }
    properties.size =
        ((properties.size + granularity - 1) / granularity) * granularity;
    if (!require_success(
            "dual holder multicast create",
            cuMulticastCreate(&multicast, &properties)) ||
        !require_success(
            "dual holder add gpu 0",
            cuMulticastAddDevice(multicast, devices[0])) ||
        !require_success(
            "dual holder add gpu 1",
            cuMulticastAddDevice(multicast, devices[1])) ||
        !require_success(
            "dual holder export FABRIC",
            cuMemExportToShareableHandle(
                &fabric,
                multicast,
                CU_MEM_HANDLE_TYPE_FABRIC,
                0)) ||
        !require_success(
            "dual holder export POSIX",
            cuMemExportToShareableHandle(
                &multicast_fd,
                multicast,
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0))) {
        kill_and_reap(peer);
        return 3;
    }

    backing_properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    backing_properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    backing_properties.location.id = devices[0];
    backing_properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    if (!require_success(
            "dual holder backing create",
            cuMemCreate(
                &backing,
                properties.size,
                &backing_properties,
                0)) ||
        !require_success(
            "dual holder multicast bind",
            cuMulticastBindMem(
                multicast,
                0,
                backing,
                0,
                properties.size,
                0))) {
        kill_and_reap(peer);
        return 3;
    }

    message.status = 0;
    message.size = properties.size;
    message.fabric = fabric;
    if (!send_multicast_message_with_fd(
            peer_socket[0], &message, multicast_fd) ||
        !read_all(peer_socket[0], &signal, sizeof(signal)) ||
        signal != 0) {
        fprintf(stderr, "dual-import child initialization failed\n");
        kill_and_reap(peer);
        return 4;
    }

    if (!require_success(
            "dual holder multicast unbind",
            cuMulticastUnbind(
                multicast,
                devices[0],
                0,
                properties.size)) ||
        !require_success(
            "dual holder backing release",
            cuMemRelease(backing)) ||
        !require_success(
            "dual holder multicast release",
            cuMemRelease(multicast)) ||
        !require_success(
            "dual holder clear context", cuCtxSetCurrent(NULL)) ||
        !require_success(
            "dual holder release gpu 1",
            cuDevicePrimaryCtxRelease(devices[1])) ||
        !require_success(
            "dual holder release gpu 0",
            cuDevicePrimaryCtxRelease(devices[0]))) {
        kill_and_reap(peer);
        return 5;
    }
    backing = 0;
    multicast = 0;
    printf(
        "dual-import external holder: pid=%ld fd=%d size=%zu "
        "CUDA handles released\n",
        (long)getpid(),
        multicast_fd,
        properties.size);

    signal = 0;
    if (!write_all(peer_socket[0], &signal, sizeof(signal))) {
        perror("dual-import signal checkpoint");
        kill_and_reap(peer);
        return 5;
    }
    close(peer_socket[0]);

    int wait_status = 0;
    while (waitpid(peer, &wait_status, 0) < 0) {
        if (errno != EINTR) {
            perror("dual-import waitpid");
            close(multicast_fd);
            return 6;
        }
    }
    close(multicast_fd);
    if (!WIFEXITED(wait_status) || WEXITSTATUS(wait_status) != 0) {
        fprintf(
            stderr,
            "RESULT: FAIL - dual-import peer status=0x%x\n",
            wait_status);
        return 7;
    }
    return 0;
}

typedef void (*tms_tag_fn)(const char*);
typedef void (*tms_bool_fn)(bool);

static int run_self_tms_paused_vmm_probe(void) {
    const size_t allocation_size = 64 * 1024 * 1024;
    void* allocation = NULL;
    size_t free_before = 0;
    size_t total = 0;
    size_t free_after_pause = 0;
    tms_tag_fn pause_fn =
        (tms_tag_fn)dlsym(RTLD_DEFAULT, "tms_pause");
    tms_tag_fn resume_fn =
        (tms_tag_fn)dlsym(RTLD_DEFAULT, "tms_resume");
    tms_tag_fn set_tag_fn =
        (tms_tag_fn)dlsym(RTLD_DEFAULT, "tms_set_current_tag");
    tms_bool_fn set_region_fn =
        (tms_bool_fn)dlsym(RTLD_DEFAULT, "tms_set_interesting_region");
    tms_bool_fn set_backup_fn =
        (tms_bool_fn)dlsym(RTLD_DEFAULT, "tms_set_enable_cpu_backup");
    CUcheckpointLockArgs lock_args = {0};
    CUcheckpointCheckpointArgs checkpoint_args = {0};
    CUcheckpointRestoreArgs restore_args = {0};
    CUcheckpointUnlockArgs unlock_args = {0};
    CUresult lock_result;
    CUresult checkpoint_result = CUDA_ERROR_UNKNOWN;
    CUresult restore_result = CUDA_ERROR_UNKNOWN;
    CUresult unlock_result = CUDA_ERROR_UNKNOWN;

    if (pause_fn == NULL || resume_fn == NULL || set_tag_fn == NULL ||
        set_region_fn == NULL || set_backup_fn == NULL) {
        fprintf(
            stderr,
            "RESULT: FAIL - torch_memory_saver C API is not preloaded\n");
        return 2;
    }
    if (!require_runtime_success("tms cudaSetDevice", cudaSetDevice(0)) ||
        !require_runtime_success(
            "tms mem info before",
            cudaMemGetInfo(&free_before, &total))) {
        return 2;
    }

    set_tag_fn("checkpoint_probe");
    set_backup_fn(false);
    set_region_fn(true);
    cudaError_t allocation_result =
        cudaMalloc(&allocation, allocation_size);
    set_region_fn(false);
    if (!require_runtime_success("tms tagged cudaMalloc", allocation_result) ||
        !require_runtime_success(
            "tms cudaMemset",
            cudaMemset(allocation, 0x5a, allocation_size)) ||
        !require_runtime_success(
            "tms pre-pause sync", cudaDeviceSynchronize())) {
        return 3;
    }

    pause_fn("checkpoint_probe");
    if (!require_runtime_success(
            "tms pause error state", cudaGetLastError()) ||
        !require_runtime_success(
            "tms mem info after pause",
            cudaMemGetInfo(&free_after_pause, &total))) {
        return 3;
    }
    printf(
        "tms paused: pid=%ld ptr=%p bytes=%zu free_before=%zu "
        "free_after_pause=%zu reclaimed=%zu\n",
        (long)getpid(),
        allocation,
        allocation_size,
        free_before,
        free_after_pause,
        free_after_pause >= free_before ? free_after_pause - free_before : 0);

    lock_args.timeoutMs = LOCK_TIMEOUT_MS;
    lock_result = cuCheckpointProcessLock((int)getpid(), &lock_args);
    printf("tms Lock raw result              -> %d\n", (int)lock_result);
    if (lock_result == CUDA_SUCCESS) {
        checkpoint_result = cuCheckpointProcessCheckpoint(
            (int)getpid(), &checkpoint_args);
        printf(
            "tms Checkpoint raw result        -> %d\n",
            (int)checkpoint_result);
    }
    if (checkpoint_result == CUDA_SUCCESS) {
        restore_result = cuCheckpointProcessRestore(
            (int)getpid(), &restore_args);
        printf(
            "tms Restore raw result           -> %d\n",
            (int)restore_result);
    }
    if (restore_result == CUDA_SUCCESS) {
        unlock_result =
            cuCheckpointProcessUnlock((int)getpid(), &unlock_args);
        printf(
            "tms Unlock raw result            -> %d\n",
            (int)unlock_result);
    }
    if (unlock_result != CUDA_SUCCESS) {
        fprintf(
            stderr,
            "RESULT: FAIL - paused TMS VMM sequence=%d/%d/%d/%d\n",
            (int)lock_result,
            (int)checkpoint_result,
            (int)restore_result,
            (int)unlock_result);
        return 4;
    }

    resume_fn("checkpoint_probe");
    if (!require_runtime_success(
            "tms resume error state", cudaGetLastError()) ||
        !require_runtime_success(
            "tms post-resume memset",
            cudaMemset(allocation, 0xa5, allocation_size)) ||
        !require_runtime_success(
            "tms post-resume sync", cudaDeviceSynchronize()) ||
        !require_runtime_success("tms cudaFree", cudaFree(allocation))) {
        return 5;
    }

    printf(
        "RESULT: PASS - paused TMS VMM allocation survived "
        "checkpoint/restore\n");
    return 0;
}

int main(int argc, char** argv) {
    int ready_pipe[2];
    int resume_pipe[2];
    pid_t child;
    unsigned char child_status = 1;
    int driver_version = 0;
    int restore_tid = -1;
    CUprocessState state;
    CUcheckpointLockArgs lock_args = {0};
    CUcheckpointCheckpointArgs checkpoint_args = {0};
    CUcheckpointRestoreArgs restore_args = {0};
    CUcheckpointUnlockArgs unlock_args = {0};
    CUresult result;

    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IOLBF, 0);

    if (argc == 2 && strcmp(argv[1], "--self") == 0) {
        return run_self_probe(0);
    }
    if (argc == 2 && strcmp(argv[1], "--self-all-devices") == 0) {
        return run_self_probe(1);
    }
    if (argc == 2 && strcmp(argv[1], "--self-pinned-host-live") == 0) {
        return run_self_pinned_host_probe(0, 0, 0);
    }
    if (argc == 2 &&
        strcmp(argv[1], "--self-pinned-host-release") == 0) {
        return run_self_pinned_host_probe(0, 1, 0);
    }
    if (argc == 2 &&
        strcmp(argv[1], "--self-pageable-host-live") == 0) {
        return run_self_pinned_host_probe(0, 0, 1);
    }
    if (argc == 2 &&
        strcmp(argv[1], "--self-pinned-host-event-live") == 0) {
        return run_self_pinned_host_probe(1, 0, 0);
    }
    if (argc == 2 && strcmp(argv[1], "--self-channel-churn") == 0) {
        return run_self_channel_churn_probe();
    }
    if (argc == 2 && strcmp(argv[1], "--self-fabric-vmm") == 0) {
        return run_self_fabric_vmm_probe();
    }
    if (argc == 2 && strcmp(argv[1], "--fabric-external-holder") == 0) {
        return run_fabric_external_holder_probe();
    }
    if (argc == 2 && strcmp(argv[1], "--multicast-import-release") == 0) {
        return run_multicast_import_probe(1);
    }
    if (argc == 2 && strcmp(argv[1], "--multicast-import-live") == 0) {
        return run_multicast_import_probe(0);
    }
    if (argc == 2 &&
        strcmp(
            argv[1],
            "--multicast-dual-import-bind-release") == 0) {
        return run_multicast_dual_import_bind_release_probe(1);
    }
    if (argc == 2 &&
        strcmp(argv[1], "--multicast-posix-bind-release") == 0) {
        return run_multicast_dual_import_bind_release_probe(0);
    }
    if (argc == 2 && strcmp(argv[1], "--self-tms-paused-vmm") == 0) {
        return run_self_tms_paused_vmm_probe();
    }

    if (argc == 4 && strcmp(argv[1], "--target") == 0) {
        char* ready_end = NULL;
        char* resume_end = NULL;
        long ready_fd = strtol(argv[2], &ready_end, 10);
        long resume_fd = strtol(argv[3], &resume_end, 10);
        if (ready_end == argv[2] || *ready_end != '\0' ||
            resume_end == argv[3] || *resume_end != '\0' ||
            ready_fd < 0 || resume_fd < 0) {
            fprintf(stderr, "invalid target pipe descriptors\n");
            return 2;
        }
        return run_cuda_child((int)ready_fd, (int)resume_fd);
    }

    if (pipe(ready_pipe) != 0 || pipe(resume_pipe) != 0) {
        perror("pipe");
        return 2;
    }

    child = fork();
    if (child < 0) {
        perror("fork");
        return 2;
    }
    if (child == 0) {
        char ready_fd_text[32];
        char resume_fd_text[32];
        close(ready_pipe[0]);
        close(resume_pipe[1]);
        (void)snprintf(
            ready_fd_text,
            sizeof(ready_fd_text),
            "%d",
            ready_pipe[1]);
        (void)snprintf(
            resume_fd_text,
            sizeof(resume_fd_text),
            "%d",
            resume_pipe[0]);
        execl(
            "/proc/self/exe",
            "cuda_checkpoint_native_probe",
            "--target",
            ready_fd_text,
            resume_fd_text,
            (char*)NULL);
        perror("exec /proc/self/exe");
        _exit(127);
    }

    close(ready_pipe[1]);
    close(resume_pipe[0]);
    if (!read_all(ready_pipe[0], &child_status, sizeof(child_status)) ||
        child_status != 0) {
        fprintf(stderr, "child failed during CUDA initialization\n");
        kill_and_reap(child);
        return 3;
    }
    close(ready_pipe[0]);

    if (!require_success("parent cuInit", cuInit(0)) ||
        !require_success(
            "parent cuDriverGetVersion",
            cuDriverGetVersion(&driver_version))) {
        kill_and_reap(child);
        return 4;
    }
    printf("CUDA Driver API version: %d\n", driver_version);

    result = cuCheckpointProcessGetRestoreThreadId((int)child, &restore_tid);
    print_result("GetRestoreThreadId", result);
    if (result == CUDA_SUCCESS) {
        printf("%-32s    tid=%d\n", "", restore_tid);
    }
    if (!print_state(child, "GetState before lock", &state)) {
        kill_and_reap(child);
        return 5;
    }

    lock_args.timeoutMs = LOCK_TIMEOUT_MS;
    result = cuCheckpointProcessLock((int)child, &lock_args);
    print_result("Lock", result);
    if (result != CUDA_SUCCESS ||
        !print_state(child, "GetState after lock", &state)) {
        kill_and_reap(child);
        return 6;
    }

    result =
        cuCheckpointProcessCheckpoint((int)child, &checkpoint_args);
    print_result("Checkpoint(zero args)", result);
    if (result != CUDA_SUCCESS ||
        !print_state(child, "GetState after checkpoint", &state)) {
        kill_and_reap(child);
        return 7;
    }

    result = cuCheckpointProcessRestore((int)child, &restore_args);
    print_result("Restore(zero args)", result);
    (void)print_state(child, "GetState after restore", &state);
    if (result != CUDA_SUCCESS) {
        printf(
            "RESULT: FAIL - native cuCheckpointProcessRestore returned %d\n",
            (int)result);
        close(resume_pipe[1]);
        kill_and_reap(child);
        return 8;
    }

    /*
     * Let the target enter its post-restore CUDA verification call while API
     * entry is still locked.  The call is expected to block until Unlock.
     */
    child_status = 0;
    if (!write_all(
            resume_pipe[1], &child_status, sizeof(child_status))) {
        perror("parent signal child");
        close(resume_pipe[1]);
        kill_and_reap(child);
        return 9;
    }

    result = cuCheckpointProcessUnlock((int)child, &unlock_args);
    print_result("Unlock(zero args)", result);
    (void)print_state(child, "GetState after unlock", &state);
    if (result != CUDA_SUCCESS) {
        close(resume_pipe[1]);
        kill_and_reap(child);
        return 10;
    }
    close(resume_pipe[1]);

    int wait_status = 0;
    while (waitpid(child, &wait_status, 0) < 0) {
        if (errno != EINTR) {
            perror("waitpid");
            return 11;
        }
    }
    if (!WIFEXITED(wait_status) || WEXITSTATUS(wait_status) != 0) {
        fprintf(stderr, "child verification failed: status=0x%x\n", wait_status);
        return 12;
    }

    printf("RESULT: PASS - native checkpoint/restore preserved CUDA state\n");
    return 0;
}
