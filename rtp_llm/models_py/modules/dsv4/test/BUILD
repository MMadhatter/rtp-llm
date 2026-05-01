py_test(
    name = "mhc_test",
    srcs = ["mhc_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm/models_py:tile_kernels_mhc",
        "//rtp_llm:testlib",
    ],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "indexer_decode_fast_test",
    srcs = ["indexer_decode_fast_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "compressor_decode_fast_test",
    srcs = ["compressor_decode_fast_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "test_dsv4_kv_cache",
    srcs = ["test_dsv4_kv_cache.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)
