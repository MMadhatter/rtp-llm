test_envs = {
    "DEVICE_RESERVE_MEMORY_BYTES": "512000000",  # 512MB
}

py_test(
    name = "mhc_test",
    srcs = ["mhc_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:testlib",
    ],
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "v4_moe_test",
    srcs = ["v4_moe_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:testlib",
    ],
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "v4_layer_test",
    srcs = ["v4_layer_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:testlib",
    ],
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "batched_experts_test",
    srcs = ["batched_experts_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:testlib",
    ],
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "noaux_tc_fused_test",
    srcs = ["noaux_tc_fused_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:testlib",
    ],
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "mhc_fused_test",
    srcs = ["mhc_fused_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:testlib",
    ],
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

# Blackwell-only — selector falls back gracefully on non-sm_100, but
# the test exercises the FP4 path so we tag it sm_100.
py_test(
    name = "megamoe_fp4_sm100_test",
    srcs = ["megamoe_fp4_sm100_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:testlib",
    ],
    env = test_envs,
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {'gpu':'SM100_ARM'},
)

# DeepGEMM MegaMoE — runs on Hopper (H20) and Blackwell (sm_100). Kernel
# is FP8 grouped; falls back to bf16 batched_experts when deep_gemm
# isn't importable.
py_test(
    name = "deepgemm_megamoe_test",
    srcs = ["deepgemm_megamoe_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:testlib",
    ],
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)
