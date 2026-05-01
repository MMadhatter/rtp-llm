"""Equivalence test: Indexer.forward_decode_vectorized FAST (Triton
v4_indexer_score) vs REF (torch einsum + relu + sum + mask chain).
"""

import os
import sys
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _make_indexer(
    ratio,
    dim,
    q_lora,
    n_heads,
    head_dim,
    rope_dim,
    topk,
    max_b,
    max_seq,
    device,
    seed=0,
):
    from rtp_llm.models_py.modules.dsv4.indexer import Indexer

    torch.manual_seed(seed)
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    idx = Indexer(
        dim=dim,
        q_lora_rank=q_lora,
        index_n_heads=n_heads,
        index_head_dim=head_dim,
        rope_head_dim=rope_dim,
        index_topk=topk,
        compress_ratio=ratio,
        max_batch_size=max_b,
        max_seq_len=max_seq,
    ).to(device)
    torch.set_default_dtype(prev_dtype)
    # ape inside the nested compressor is torch.empty(...) — initialize.
    with torch.no_grad():
        idx.compressor.ape.normal_(mean=0.0, std=0.1)
    theta = torch.randn(max_seq, rope_dim // 2, device=device)
    idx.freqs_cis = torch.polar(torch.ones_like(theta), theta)
    idx.compressor.kv_cache = idx.kv_cache
    idx.compressor.freqs_cis = idx.freqs_cis
    return idx


class TestIndexerDecodeFast(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = "cuda:0"

    def _run_case(self, ratio, B, start_positions, topk=16):
        dim, q_lora, n_heads, head_dim, rope_dim = 256, 192, 64, 128, 64
        max_seq = ((max(start_positions) + ratio + 64) // ratio) * ratio
        max_b = max(B, 4)
        idx = _make_indexer(
            ratio,
            dim,
            q_lora,
            n_heads,
            head_dim,
            rope_dim,
            topk,
            max_b,
            max_seq,
            self.device,
            seed=7,
        )
        torch.manual_seed(11)
        idx.kv_cache.normal_()

        x = torch.randn(B, 1, dim, dtype=torch.bfloat16, device=self.device) * 0.1
        qr = torch.randn(B, 1, q_lora, dtype=torch.bfloat16, device=self.device) * 0.1
        sp = torch.tensor(start_positions[:B], dtype=torch.int32, device=self.device)

        out_fast = torch.full((B, 1, topk), -1, dtype=torch.int32, device=self.device)
        out_ref = torch.full((B, 1, topk), -1, dtype=torch.int32, device=self.device)
        snap = {
            "kv_state": idx.compressor.kv_state.detach().clone(),
            "score_state": idx.compressor.score_state.detach().clone(),
            "kv_cache": idx.kv_cache.detach().clone(),
        }

        os.environ["DSV4_INDEXER_FAST"] = "1"
        idx.forward_decode_vectorized(x, qr, sp, out_fast)

        idx.compressor.kv_state.copy_(snap["kv_state"])
        idx.compressor.score_state.copy_(snap["score_state"])
        idx.kv_cache.copy_(snap["kv_cache"])
        os.environ["DSV4_INDEXER_FAST"] = "0"
        idx.forward_decode_vectorized(x, qr, sp, out_ref)
        os.environ["DSV4_INDEXER_FAST"] = "1"

        for b in range(B):
            ref_set = {int(v) for v in out_ref[b, 0].tolist() if v >= 0}
            fast_set = {int(v) for v in out_fast[b, 0].tolist() if v >= 0}
            if not ref_set:
                continue
            inter = len(ref_set & fast_set)
            self.assertGreaterEqual(
                inter / len(ref_set),
                0.9,
                f"topk overlap too low (ratio={ratio}, B={B}, b={b}): "
                f"ref={sorted(ref_set)} fast={sorted(fast_set)}",
            )

    def test_hca_ratio128_b1(self):
        self._run_case(128, 1, [255])

    def test_hca_ratio128_b4(self):
        self._run_case(128, 4, [255, 511, 127, 383])

    def test_csa_ratio4_b4(self):
        self._run_case(4, 4, [15, 31, 63, 127])

    def test_csa_ratio4_b8(self):
        self._run_case(4, 8, [15, 31, 63, 127, 7, 11, 23, 47])


if __name__ == "__main__":
    unittest.main()
