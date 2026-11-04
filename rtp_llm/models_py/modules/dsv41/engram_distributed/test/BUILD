package(default_visibility = ["//visibility:public"])

py_library(
    name = "exchange_test_certs",
    testonly = True,
    srcs = ["make_exchange_test_certs.py"],
)

py_test(
    name = "test_engram_exchange",
    size = "small",
    srcs = ["test_engram_exchange.py"],
    main = "test_engram_exchange.py",
    deps = ["//rtp_llm/models_py/modules/dsv41/engram_distributed:core", ":exchange_test_certs"],
    env = {"GPU_COUNT": "0", "CUDA_VISIBLE_DEVICES": ""},
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
)

py_test(
    name = "test_engram_publish",
    size = "small",
    srcs = ["test_engram_publish.py"],
    main = "test_engram_publish.py",
    deps = ["//rtp_llm/models_py/modules/dsv41/engram_distributed:core"],
    env = {"GPU_COUNT": "0", "CUDA_VISIBLE_DEVICES": ""},
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
)

py_test(
    name = "test_engram_manifest",
    size = "small",
    srcs = ["test_engram_manifest.py"],
    main = "test_engram_manifest.py",
    deps = ["//rtp_llm/models_py/modules/dsv41/engram_distributed:core"],
    env = {"GPU_COUNT": "0", "CUDA_VISIBLE_DEVICES": ""},
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
)

py_test(
    name = "test_engram_config",
    size = "small",
    srcs = ["test_engram_config.py"],
    main = "test_engram_config.py",
    deps = ["//rtp_llm/models_py/modules/dsv41/engram_distributed:core"],
    env = {"GPU_COUNT": "0", "CUDA_VISIBLE_DEVICES": ""},
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
)

py_test(
    name = "test_engram_sharding",
    size = "small",
    srcs = ["test_engram_sharding.py"],
    main = "test_engram_sharding.py",
    deps = ["//rtp_llm/models_py/modules/dsv41/engram_distributed:core"],
    env = {"GPU_COUNT": "0", "CUDA_VISIBLE_DEVICES": ""},
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
)

py_test(
    name = "test_engram_storage",
    size = "small",
    srcs = ["test_engram_storage.py"],
    main = "test_engram_storage.py",
    deps = ["//rtp_llm/models_py/modules/dsv41/engram_distributed:core"],
    env = {"GPU_COUNT": "0", "CUDA_VISIBLE_DEVICES": ""},
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
)

py_test(
    name = "test_engram_model_attach",
    size = "small",
    srcs = ["test_engram_model_attach.py"],
    main = "test_engram_model_attach.py",
    deps = ["//rtp_llm/models_py/modules/dsv41/engram_distributed:core"],
    env = {"GPU_COUNT": "0", "CUDA_VISIBLE_DEVICES": ""},
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
)

py_test(
    name = "test_engram_checkpoint",
    size = "small",
    srcs = ["test_engram_checkpoint.py"],
    main = "test_engram_checkpoint.py",
    deps = ["//rtp_llm/models_py/modules/dsv41/engram_distributed:runtime"],
    env = {"GPU_COUNT": "0", "CUDA_VISIBLE_DEVICES": ""},
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
    target_compatible_with = select({
        "@//:using_cuda13_arm": [],
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "test_engram_bootstrap",
    size = "small",
    srcs = ["test_engram_bootstrap.py", "test_engram_manifest.py"],
    main = "test_engram_bootstrap.py",
    deps = ["//rtp_llm/models_py/modules/dsv41/engram_distributed:runtime", "//rtp_llm/models_py:engram_model_cpu"],
    env = {"GPU_COUNT": "0", "CUDA_VISIBLE_DEVICES": ""},
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
    target_compatible_with = select({
        "@//:using_cuda13_arm": [],
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "test_engram_model_loading",
    size = "small",
    srcs = ["test_engram_model_loading.py"],
    main = "test_engram_model_loading.py",
    deps = [
        "//rtp_llm/models_py/modules/dsv41/engram_distributed:runtime",
        "//rtp_llm/models_py:engram_model_cpu",
        "//rtp_llm/models_py:engram_warmup_cpu",
        "//rtp_llm/models_py:engram_scale_source_cpu",
    ],
    env = {"GPU_COUNT": "0", "CUDA_VISIBLE_DEVICES": ""},
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
    target_compatible_with = select({
        "@//:using_cuda13_arm": [],
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

# Real TCPStore clients and HTTP exchange; allocations are injected, no CUDA.
# The default CUDA13 x86 unit lane supplies Torch on its CPU worker pool.
py_test(
    name = "test_startup_store",
    size = "small",
    srcs = [
        "test_startup_store.py",
        ":test_engram_exchange.py",
    ],
    main = "test_startup_store.py",
    deps = [
        "//rtp_llm/models_py/modules/dsv41/engram_distributed:core",
        "//rtp_llm:torch",
        ":exchange_test_certs",
    ],
    env = {"GPU_COUNT": "0", "CUDA_VISIBLE_DEVICES": ""},
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
)

py_test(
    name = "test_fabric_ci",
    size = "small",
    srcs = [
        "run_fabric_ci.py",
        "test_fabric_ci.py",
        "test_native.py",
        "test_fabric_lookup_cuda.py",
    ],
    main = "test_fabric_ci.py",
    env = {"GPU_COUNT": "0"},
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
)

py_test(
    name = "test_embedding_contract",
    size = "small",
    srcs = ["test_embedding_contract.py"],
    main = "test_embedding_contract.py",
    deps = ["//rtp_llm/models_py:engram_model_cpu"],
    env = {"GPU_COUNT": "0"},
    legacy_create_init = 0,
    target_compatible_with = select({
        "@//:using_cuda13_arm": [],
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

# Opt-in only. The runner verifies Fabric capability and two idle GPUs and
# owns their locks. Explicit invocations clear run_under; the default wrapper
# hides CUDA (GPU_COUNT=0), so it cannot enter DeviceResource PID cleanup.
py_test(
    name = "test_fabric_integration",
    size = "large",
    srcs = [
        "run_fabric_ci.py",
        "test_native.py",
        "test_fabric_lookup_cuda.py",
    ],
    main = "run_fabric_ci.py",
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py/modules/dsv41/engram_distributed/native:build_support",
        "//rtp_llm:numpy",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
    ],
    tags = ["manual", "engram_fabric"],
    # A direct invocation with the repository's default run_under must fail
    # closed (CUDA hidden), never enter DeviceResource's PID cleanup path.
    # Explicit invocations clear run_under and let run_fabric_ci own the locks.
    env = {"GPU_COUNT": "0"},
    exec_properties = {"gpu": "SM100_ARM_CU13", "gpu_count": "2"},
    legacy_create_init = 0,
    target_compatible_with = select({
        "@//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "test_engram_placement",
    size = "small",
    srcs = ["test_engram_placement.py", "test_engram_manifest.py", "test_engram_sharding.py"],
    main = "test_engram_placement.py",
    deps = ["//rtp_llm/models_py/modules/dsv41/engram_distributed:core"],
    env = {"GPU_COUNT": "0", "CUDA_VISIBLE_DEVICES": ""},
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
)
