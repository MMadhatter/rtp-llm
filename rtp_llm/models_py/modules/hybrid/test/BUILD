
test_envs = {
    "DEVICE_RESERVE_MEMORY_BYTES": "512000000",  # 512MB
}

flashinfer_test_envs = {
    "DEVICE_RESERVE_MEMORY_BYTES": "512000000",  # 512MB
    "FLASHINFER_JIT_VERBOSE": "1",  # Enable verbose JIT compilation logs
}

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

# FlashInfer dependencies (包括 apache-tvm-ffi 用于 JIT 编译)
flashinfer_deps = [
    "//rtp_llm:apache-tvm-ffi",
    "//rtp_llm:flashinfer-python",
]

py_test (
    name = "mla_attention_test",
    srcs = ["mla_attention_test.py", "mla_attention_ref.py"],
    deps = py_test_deps + [
        "//rtp_llm:testlib",
    ] + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": []
    }),
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "mla_reuse_cache_test",
    srcs = ["mla_reuse_cache_test.py", "mla_attention_ref.py"],
    deps = py_test_deps + [
        "//rtp_llm:testlib",
    ] + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": []
    }),
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "mla_perf_test",
    srcs = ["mla_perf_test.py", "mla_attention_ref.py"],
    deps = py_test_deps + [
        "//rtp_llm:testlib",
    ],
    env = test_envs,
    tags = ["open_skip", "H20", "manual"],
    exec_properties = {'gpu':'H20'},
)

py_test (
   name = "mlp_test",
   srcs = ["mlp_test.py", "dense_mlp_ref.py"],
   deps = [
       "//rtp_llm/models_py:models",
       "//rtp_llm:config",
       "//rtp_llm:utils",
      "//rtp_llm:testlib",
       "//rtp_llm/test/model_test/test_util:test_util"
   ],
   env = test_envs,
   tags = ["open_skip", "H20"],
   exec_properties = {'gpu':'H20'},
)

py_test (
   name = "flashinfer_jit_test",
   srcs = ["flashinfer_jit_test.py"],
   deps = [
       "//rtp_llm/models_py:models",
       "//rtp_llm:config",
       "//rtp_llm:utils",
      "//rtp_llm:testlib",
       "//rtp_llm/test/model_test/test_util:test_util"
   ],
   env = flashinfer_test_envs,
   tags = ["open_skip", "H20"],
   exec_properties = {'gpu':'H20'},
)

py_test (
   name = "indexer_test",
   srcs = ["indexer_test.py", "indexer_ref.py"],
   deps = py_test_deps + [
        "//rtp_llm:testlib",
   ],
   env = test_envs,
   tags = ["open_skip", "H20"],
   exec_properties = {'gpu':'H20'},
)

py_test (
   name = "compressor_test",
   srcs = ["compressor_test.py"],
   deps = [
       "//rtp_llm/models_py:models",
       "//rtp_llm:testlib",
   ],
   env = test_envs,
   tags = ["open_skip", "H20"],
   exec_properties = {'gpu':'H20'},
)

py_test (
   name = "hca_attention_test",
   srcs = ["hca_attention_test.py"],
   deps = [
       "//rtp_llm/models_py:models",
       "//rtp_llm:testlib",
   ],
   env = test_envs,
   tags = ["open_skip", "H20"],
   exec_properties = {'gpu':'H20'},
)

py_test (
   name = "csa_attention_test",
   srcs = ["csa_attention_test.py"],
   deps = [
       "//rtp_llm/models_py:models",
       "//rtp_llm:testlib",
   ],
   env = test_envs,
   tags = ["open_skip", "H20"],
   exec_properties = {'gpu':'H20'},
)

py_test (
   name = "cache_topology_test",
   srcs = ["cache_topology_test.py"],
   deps = [
       "//rtp_llm/models_py:models",
       "//rtp_llm:testlib",
   ],
   env = test_envs,
   tags = ["open_skip", "H20"],
   exec_properties = {'gpu':'H20'},
)

py_test (
   name = "fused_qk_norm_rope_test",
   srcs = ["fused_qk_norm_rope_test.py"],
   deps = [
       "//rtp_llm/models_py:models",
       "//rtp_llm:testlib",
   ],
   env = test_envs,
   tags = ["open_skip", "H20"],
   exec_properties = {'gpu':'H20'},
)

py_test (
   name = "deepseek_v4_rope_test",
   srcs = ["deepseek_v4_rope_test.py"],
   deps = [
       "//rtp_llm/models_py:models",
       "//rtp_llm:testlib",
   ],
   env = test_envs,
   tags = ["open_skip", "H20"],
   exec_properties = {'gpu':'H20'},
)

py_test (
   name = "fused_indexer_test",
   srcs = ["fused_indexer_test.py"],
   deps = [
       "//rtp_llm/models_py:models",
       "//rtp_llm:testlib",
   ],
   env = test_envs,
   tags = ["open_skip", "H20"],
   exec_properties = {'gpu':'H20'},
)

py_test (
   name = "online_sink_attention_test",
   srcs = ["online_sink_attention_test.py"],
   deps = [
       "//rtp_llm/models_py:models",
       "//rtp_llm:testlib",
   ],
   env = test_envs,
   tags = ["open_skip", "H20"],
   exec_properties = {'gpu':'H20'},
)

py_test (
   name = "fused_compressor_test",
   srcs = ["fused_compressor_test.py"],
   deps = [
       "//rtp_llm/models_py:models",
       "//rtp_llm:testlib",
   ],
   env = test_envs,
   tags = ["open_skip", "H20"],
   exec_properties = {'gpu':'H20'},
)

# Blackwell-only — kernel skipped on non-sm_100 hosts via runtime decorator.
py_test (
   name = "fp4_indexer_sm100_test",
   srcs = ["fp4_indexer_sm100_test.py"],
   deps = [
       "//rtp_llm/models_py:models",
       "//rtp_llm:testlib",
   ],
   env = test_envs,
   tags = ["open_skip", "SM100_ARM"],
   exec_properties = {'gpu':'SM100_ARM'},
)

# Mixed: ReferenceSparseCsaTest runs on CPU; FlashMlaCsaSm100Test +
# FlashMlaCsaBenchmark are skipped at runtime when sm_100 / flash_mla absent.
py_test (
   name = "flashmla_csa_test",
   srcs = ["flashmla_csa_test.py"],
   deps = [
       "//rtp_llm/models_py:models",
       "//rtp_llm:testlib",
   ],
   env = test_envs,
   tags = ["open_skip", "SM100_ARM"],
   exec_properties = {'gpu':'SM100_ARM'},
)

# Same shape contract as fp4_indexer_sm100_test — DeepGEMM FP8 indexer.
py_test (
   name = "deepgemm_indexer_test",
   srcs = ["deepgemm_indexer_test.py"],
   deps = [
       "//rtp_llm/models_py:models",
       "//rtp_llm:testlib",
   ],
   env = test_envs,
   tags = ["open_skip", "H20"],
   exec_properties = {'gpu':'H20'},
)