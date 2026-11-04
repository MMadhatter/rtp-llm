load("@arch_config//:arch_select.bzl", "torch_deps")
load("//:def.bzl", "copts")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "engram_native_lib",
    srcs = ["engram_native.cc"],
    copts = copts(),
    local_defines = ["TORCH_EXTENSION_NAME=libengram_native"],
    alwayslink = True,
    deps = [
        "@local_config_cuda//cuda:cuda_driver",
        "@local_config_cuda//cuda:cuda_headers",
    ] + torch_deps(),
    target_compatible_with = select({
        "//:using_cuda13_arm": [],
        "//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "test_packaged_native",
    size = "small",
    srcs = ["test_packaged_native.py"],
    data = [
        "//rtp_llm/libs:engram_libs",
        "//rtp_llm/libs:libth_transformer_config_so",
    ],
    deps = [
        "//rtp_llm:ops",
        "//rtp_llm:torch",
        "//rtp_llm/models_py/modules/dsv41/engram_distributed:core",
    ],
    env = {"CUDA_VISIBLE_DEVICES": "", "GPU_COUNT": "0"},
    tags = ["engram_distributed_cpu"],
    target_compatible_with = select({
        "//:using_cuda13_arm": [],
        "//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

filegroup(
    name = "source",
    srcs = ["engram_native.cc"],
)

# Development/CI builds only; serving imports the prebuilt ABI-compatible bridge.
py_library(
    name = "build_support",
    testonly = True,
    srcs = ["__init__.py", "load_native.py"],
    data = [":source"],
    deps = ["//rtp_llm:torch"],
)
