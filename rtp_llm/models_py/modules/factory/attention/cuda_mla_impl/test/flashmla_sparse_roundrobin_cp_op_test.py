"""
Unit tests for RoundRobinSparseMlaFp8CPOp (Context Parallel prefill for Sparse MLA FP8).

Tests the CP path with tp_size=1 (single rank): all_gather is identity,
so output should match non-CP SparseMlaFp8Op on the same inputs.

Usage:
    python -m pytest rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test/flashmla_sparse_roundrobin_cp_op_test.py -v
    python -m unittest rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.test.flashmla_sparse_roundrobin_cp_op_test
"""

import math
from unittest import SkipTest, TestCase, main, skipIf
from unittest.mock import patch

import torch

from rtp_llm.ops import KvCacheDataType
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyAttentionInputs,
    PyContextParallelParams,
    rtp_llm_ops,
)


def _check_cuda_flashmla():
    """Require CUDA >= 12.9 and flash_mla import."""
    try:
        if not torch.version.cuda:
            return False
        major, minor = map(int, torch.version.cuda.split(".")[:2])
        if (major, minor) < (12, 9):
            return False
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata  # noqa: F401

        return True
    except (ImportError, AttributeError, ValueError):
        return False


CUDA_FLASHMLA_OK = _check_cuda_flashmla()
SKIP_REASON = "Requires CUDA >= 12.9 and flash_mla"


def _set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_block_table(
    batch_size: int, seq_len: int, page_size: int, device: torch.device
) -> torch.Tensor:
    num_blocks_per_seq = math.ceil(seq_len / page_size)
    block_table = torch.zeros(
        [batch_size, num_blocks_per_seq], dtype=torch.int32, device=torch.device("cpu")
    )
    bias = 0
    for i in range(batch_size):
        block_table[i, :] = torch.arange(
            bias,
            bias + num_blocks_per_seq,
            dtype=torch.int32,
            device=torch.device("cpu"),
        )
        bias += num_blocks_per_seq
    return block_table.to(device)


@skipIf(not CUDA_FLASHMLA_OK, SKIP_REASON)
class RoundRobinSparseMlaFp8CPOpTest(TestCase):
    """Test RoundRobinSparseMlaFp8CPOp with single rank (tp_size=1)."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise SkipTest("CUDA not available")
        cls.device = torch.device("cuda:0")
        torch.cuda.set_device(cls.device)

    def setUp(self):
        torch.set_default_device(self.device)
        torch.set_default_dtype(torch.bfloat16)
        torch.cuda.empty_cache()

    def tearDown(self):
        torch.cuda.empty_cache()

    def _build_common_params(
        self,
        total_q_len: int,
        chunk_lengths: list,
        prefix_len: int = 0,
    ):
        """Build common attn_inputs, mla_params, parallelism_config, and tensors."""
        device = self.device
        num_heads = 64
        kv_lora_rank = 512
        qk_rope_head_dim = 64
        qk_nope_head_dim = 512
        page_size = 64
        softmax_extra_scale = 1.0
        top_k = 128
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        fp8_bytes_per_token = 656

        n_restore = sum(chunk_lengths)
        total_kv_len = prefix_len + n_restore

        cp_params = PyContextParallelParams()
        cp_params.prefill_cp_chunk_lengths = torch.tensor(
            chunk_lengths, dtype=torch.int32, device=device
        )
        # tp_size=1 → no padding needed
        cp_params.prefill_cp_padding_lengths = torch.zeros(
            len(chunk_lengths), dtype=torch.int32, device=device
        )
        cp_params.prefill_qkv_restore_indice = torch.arange(
            n_restore, dtype=torch.long, device=device
        )
        cp_params.prefill_qkv_padding_mask = torch.ones(
            n_restore, dtype=torch.int32, device=device
        )
        cp_params.prefill_actual_input_lengths_cpu = torch.tensor(
            [n_restore], dtype=torch.int32, device=torch.device("cpu")
        )

        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.input_lengths = torch.tensor(
            [n_restore], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.sequence_lengths = torch.tensor(
            [total_kv_len], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.prefix_lengths = torch.tensor(
            [prefix_len], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.context_parallel_info = cp_params

        block_table_host = _make_block_table(
            1, total_kv_len, page_size, torch.device("cpu")
        )
        block_table_device = block_table_host.to(device)
        attn_inputs.kv_cache_block_id_host = block_table_host
        attn_inputs.kv_cache_block_id_device = block_table_device

        mla_params = rtp_llm_ops.SparseMlaParams()
        mla_params.fill_params(attn_inputs, page_size)

        from rtp_llm.ops import CPRotateMethod, ParallelismConfig, PrefillCPConfig

        parallelism_config = ParallelismConfig()
        parallelism_config.tp_rank = 0
        parallelism_config.tp_size = 1
        parallelism_config.prefill_cp_config = PrefillCPConfig()
        parallelism_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        parallelism_config.prefill_cp_config.comm_buffer_size = 0
        parallelism_config.prefill_cp_config.kv_cache_sharded = True

        q = (
            torch.randn(
                total_q_len, num_heads, qk_head_dim, dtype=torch.bfloat16, device=device
            )
            * 0.1
        )
        compressed_kv = (
            torch.randn(total_q_len, kv_lora_rank, dtype=torch.bfloat16, device=device)
            * 0.1
        )
        k_pe = (
            torch.randn(
                total_q_len, qk_rope_head_dim, dtype=torch.bfloat16, device=device
            )
            * 0.1
        )
        topk_indices = torch.randint(
            0, total_kv_len, (total_q_len, 1, top_k), dtype=torch.int32, device=device
        )
        batch_indice_d = torch.zeros(total_q_len, dtype=torch.int32, device=device)

        num_blocks = block_table_host.shape[1]
        kv_cache_base = torch.empty(
            num_blocks, page_size, fp8_bytes_per_token, dtype=torch.uint8, device=device
        )
        kv_cache = KVCache()
        kv_cache.kv_cache_base = kv_cache_base

        return dict(
            attn_inputs=attn_inputs,
            mla_params=mla_params,
            parallelism_config=parallelism_config,
            block_table_device=block_table_device,
            q=q,
            compressed_kv=compressed_kv,
            k_pe=k_pe,
            topk_indices=topk_indices,
            batch_indice_d=batch_indice_d,
            kv_cache=kv_cache,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            page_size=page_size,
            softmax_extra_scale=softmax_extra_scale,
            top_k=top_k,
        )

    def _make_roundrobin_op(self, params):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
            RoundRobinSparseMlaFp8CPOp,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
            MlaKVCacheWriteOp,
        )

        op = RoundRobinSparseMlaFp8CPOp(
            num_heads=params["num_heads"],
            kv_lora_rank=params["kv_lora_rank"],
            qk_rope_head_dim=params["qk_rope_head_dim"],
            qk_nope_head_dim=params["qk_nope_head_dim"],
            page_size=params["page_size"],
            softmax_extra_scale=params["softmax_extra_scale"],
            top_k=params["top_k"],
            attn_inputs=params["attn_inputs"],
            parallelism_config=params["parallelism_config"],
        )
        op.kv_cache_write_op = MlaKVCacheWriteOp(kv_cache_dtype=KvCacheDataType.FP8)
        op.write_cache_store_impl = None
        op.attn_inputs = params["attn_inputs"]
        op.plan(params["mla_params"], params["block_table_device"])
        return op

    def test_roundrobin_forward_output_shape(self):
        """RoundRobin CP op forward returns correct shape [total_q_len, num_heads, kv_lora_rank]."""
        _set_seed(123)
        total_q_len = 8
        chunk_lengths = [8]
        params = self._build_common_params(total_q_len, chunk_lengths)
        op = self._make_roundrobin_op(params)

        def _identity_all_gather(tensor, group=None):
            return tensor

        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=_identity_all_gather,
        ):
            out = op.forward(
                params["q"],
                params["compressed_kv"],
                params["k_pe"],
                params["topk_indices"],
                params["batch_indice_d"],
                params["kv_cache"],
                layer_id=0,
            )
        torch.cuda.synchronize()
        self.assertEqual(
            out.shape,
            (total_q_len, params["num_heads"], params["kv_lora_rank"]),
        )

    def test_roundrobin_forward_match_non_cp(self):
        """
        With tp_size=1, RoundRobin CP all_gather is identity.
        Output should match non-CP SparseMlaFp8Op on the same inputs.
        """
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
            SparseMlaFp8Op,
        )

        _set_seed(42)
        total_q_len = 8
        chunk_lengths = [8]
        params = self._build_common_params(total_q_len, chunk_lengths)
        op = self._make_roundrobin_op(params)

        def _identity_all_gather(tensor, group=None):
            return tensor

        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=_identity_all_gather,
        ):
            out_cp = op.forward(
                params["q"],
                params["compressed_kv"],
                params["k_pe"],
                params["topk_indices"],
                params["batch_indice_d"],
                params["kv_cache"],
                layer_id=0,
            )
        torch.cuda.synchronize()

        self.assertEqual(
            out_cp.shape,
            (total_q_len, params["num_heads"], params["kv_lora_rank"]),
        )

        # Non-CP reference
        kv_cache_base = params["kv_cache"].kv_cache_base
        kv_cache_flat = kv_cache_base.view(-1, 1, kv_cache_base.size(-1))
        if kv_cache_flat.ndim == 3:
            kv_cache_flat = kv_cache_flat.unsqueeze(-2)
        non_cp_op = SparseMlaFp8Op(
            num_heads=params["num_heads"],
            kv_lora_rank=params["kv_lora_rank"],
            qk_rope_head_dim=params["qk_rope_head_dim"],
            qk_nope_head_dim=params["qk_nope_head_dim"],
            page_size=params["page_size"],
            softmax_extra_scale=params["softmax_extra_scale"],
            top_k=params["top_k"],
        )
        non_cp_op.plan(params["mla_params"], params["block_table_device"])
        out_non_cp = non_cp_op.forward(
            params["q"], kv_cache_flat, params["topk_indices"], layer_id=0
        )
        torch.cuda.synchronize()

        self.assertTrue(
            torch.allclose(out_cp, out_non_cp, atol=1e-2, rtol=1e-2),
            "RoundRobin CP output should match non-CP when tp_size=1 (all_gather identity)",
        )

    def test_roundrobin_forward_multi_chunk(self):
        """RoundRobin CP op works correctly with multiple chunks."""
        _set_seed(77)
        total_q_len = 8
        chunk_lengths = [4, 4]
        params = self._build_common_params(total_q_len, chunk_lengths)
        op = self._make_roundrobin_op(params)

        def _identity_all_gather(tensor, group=None):
            return tensor

        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=_identity_all_gather,
        ):
            out = op.forward(
                params["q"],
                params["compressed_kv"],
                params["k_pe"],
                params["topk_indices"],
                params["batch_indice_d"],
                params["kv_cache"],
                layer_id=0,
            )
        torch.cuda.synchronize()
        self.assertEqual(
            out.shape,
            (total_q_len, params["num_heads"], params["kv_lora_rank"]),
        )

    def test_roundrobin_no_topk_returns_none(self):
        """When topk is None, forward should return None (write-only path)."""
        _set_seed(99)
        total_q_len = 8
        chunk_lengths = [8]
        params = self._build_common_params(total_q_len, chunk_lengths)
        op = self._make_roundrobin_op(params)

        def _identity_all_gather(tensor, group=None):
            return tensor

        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=_identity_all_gather,
        ):
            out = op.forward(
                params["q"],
                params["compressed_kv"],
                params["k_pe"],
                None,  # topk=None
                params["batch_indice_d"],
                params["kv_cache"],
                layer_id=0,
            )
        self.assertIsNone(out)

    def test_roundrobin_prefix_cache_not_supported(self):
        """RoundRobin with prefix_lengths > 0 should raise AssertionError."""
        _set_seed(55)
        total_q_len = 8
        chunk_lengths = [8]
        # prefix_len > 0 triggers the assertion in plan()
        with self.assertRaises(AssertionError):
            params = self._build_common_params(
                total_q_len, chunk_lengths, prefix_len=64
            )
            self._make_roundrobin_op(params)

    def test_roundrobin_cu_kv_seqlens_global(self):
        """Verify cu_kv_seqlens_global is correctly computed."""
        _set_seed(33)
        total_q_len = 8
        chunk_lengths = [8]
        params = self._build_common_params(total_q_len, chunk_lengths)
        op = self._make_roundrobin_op(params)

        self.assertEqual(
            int(op.cu_kv_seqlens_global[-1].item()),
            sum(chunk_lengths),
            "cu_kv_seqlens_global[-1] should equal total KV length",
        )


if __name__ == "__main__":
    main()
