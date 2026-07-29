load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "sleep_memory_policy",
    hdrs = ["SleepMemoryPolicy.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "pinned_host_memory_stats",
    srcs = ["PinnedHostMemoryStats.cc"],
    hdrs = ["PinnedHostMemoryVerifier.h"],
    copts = copts(),
    deps = [
        "@com_google_absl//absl/status",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "pinned_host_memory_verifier",
    srcs = ["PinnedHostMemoryVerifier.cc"],
    copts = copts(),
    deps = [
        ":pinned_host_memory_stats",
        "//rtp_llm/cpp/utils:core_utils",
        "@com_google_absl//absl/status",
    ] + torch_deps() + select({
        "//:using_cuda": [
            "@local_config_cuda//cuda:cuda_headers",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "sleep",
    hdrs = [
        "SleepLifecycleController.h",
    ],
    srcs = [
        "SleepLifecycleController.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "drain_manager",
    hdrs = [
        "DrainManager.h",
    ],
    srcs = [
        "DrainManager.cc",
    ],
    copts = copts(),
    deps = [
        ":sleep",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "admission_gate",
    hdrs = [
        "AdmissionGate.h",
    ],
    srcs = [
        "AdmissionGate.cc",
    ],
    copts = copts(),
    deps = [
        ":sleep",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
    ],
    visibility = ["//visibility:public"],
)
