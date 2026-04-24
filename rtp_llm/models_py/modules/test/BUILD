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
