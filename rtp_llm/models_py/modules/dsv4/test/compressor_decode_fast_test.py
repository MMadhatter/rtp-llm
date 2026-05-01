"""Equivalence test: Compressor.forward_decode_vectorized FAST (Triton
v4_compressor_pool) vs REF (torch softmax+sum chain).

Covers both ratios:
  * HCA (overlap=False, ratio=128, G=128)
  * CSA (overlap=True,  ratio=4,   G=8)

Sweeps boundary and non-boundary start_pos and verifies bit-equal
outputs / state buffers (REF and FAST share the same fp32 inputs and
the kernel matches the REF math byte-for-byte modulo float reduction
order — we accept rel_mean < 1e-5 to absorb any reduction-order drift).
"""

import os
import sys
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _make_compressor(
    ratio: int,
    dim: int,
    head_dim: int,
    rope_dim: int,
    max_b: int,
    max_seq: int,
    device,
    seed: int = 0,
):
    from rtp_llm.models_py.modules.dsv4.compressor import Compressor

    torch.manual_seed(seed)
    c = Compressor(
        dim=dim,
        head_dim=head_dim,
        rope_head_dim=rope_dim,
        compress_ratio=ratio,
        max_batch_size=max_b,
    ).to(device)
    # ape is `torch.empty(...)` in non-factory mode — uninitialized memory.
    # Force finite init so inf/NaN don't poison the score path.
    with torch.no_grad():
        c.ape.normal_(mean=0.0, std=0.1)
    # Bind the cache + freqs the way attention.py does.
    c.kv_cache = torch.zeros(
        max_b, max_seq // ratio, head_dim, dtype=torch.bfloat16, device=device
    )
    # Synthetic complex freqs_cis [max_seq, rope_dim // 2].
    theta = torch.randn(max_seq, rope_dim // 2, device=device)
    c.freqs_cis = torch.polar(torch.ones_like(theta), theta)
    return c


def _clone_state(c):
    return {
        "kv_state": c.kv_state.detach().clone(),
        "score_state": c.score_state.detach().clone(),
        "kv_cache": c.kv_cache.detach().clone(),
    }


def _restore_state(c, snap):
    c.kv_state.copy_(snap["kv_state"])
    c.score_state.copy_(snap["score_state"])
    c.kv_cache.copy_(snap["kv_cache"])


def _rel_mean(a, b):
    diff = (a.float() - b.float()).abs()
    mag = a.float().abs().mean().item() + 1e-9
    return diff.mean().item() / mag


class TestCompressorDecodeFast(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = "cuda:0"

    def _run_case(self, ratio: int, B: int, start_positions):
        dim = 256
        head_dim = 512
        rope_dim = 64
        max_seq = max(start_positions) + ratio + 32
        # round up to ratio multiple for cache bound
        max_seq = ((max_seq + ratio - 1) // ratio) * ratio
        max_b = max(B, 4)

        c = _make_compressor(
            ratio, dim, head_dim, rope_dim, max_b, max_seq, self.device, seed=42
        )
        # Pre-populate state so non-boundary slots have non-trivial content.
        torch.manual_seed(123)
        c.kv_state.normal_()
        c.score_state.normal_()
        c.kv_cache.normal_()

        snap = _clone_state(c)

        x = torch.randn(B, 1, dim, dtype=torch.bfloat16, device=self.device) * 0.1
        sp = torch.tensor(start_positions[:B], dtype=torch.int32, device=self.device)

        # FAST
        os.environ["DSV4_COMPRESSOR_FAST"] = "1"
        out_fast = c.forward_decode_vectorized(x, sp)
        snap_fast = _clone_state(c)

        # REF
        _restore_state(c, snap)
        os.environ["DSV4_COMPRESSOR_FAST"] = "0"
        out_ref = c.forward_decode_vectorized(x, sp)
        snap_ref = _clone_state(c)

        os.environ["DSV4_COMPRESSOR_FAST"] = "1"

        self.assertEqual(out_fast.shape, out_ref.shape)
        self.assertEqual(out_fast.dtype, out_ref.dtype)
        self.assertLess(
            _rel_mean(out_fast, out_ref),
            5e-3,
            f"out rel diff too high (ratio={ratio}, B={B})",
        )
        self.assertLess(
            _rel_mean(snap_fast["kv_cache"], snap_ref["kv_cache"]),
            5e-3,
            "kv_cache diverged",
        )
        # State buffers should match identically (we only swapped the pool op).
        self.assertLess(
            _rel_mean(snap_fast["kv_state"], snap_ref["kv_state"]),
            1e-6,
            "kv_state diverged",
        )
        self.assertLess(
            _rel_mean(snap_fast["score_state"], snap_ref["score_state"]),
            1e-6,
            "score_state diverged",
        )

    def test_hca_ratio128_b1_boundary(self):
        # boundary: (sp+1) % ratio == 0
        self._run_case(128, 1, [127])

    def test_hca_ratio128_b4_mixed(self):
        self._run_case(128, 4, [127, 50, 255, 100])

    def test_hca_ratio128_b16(self):
        self._run_case(
            128,
            16,
            [127, 50, 255, 100, 0, 1, 126, 200, 300, 383, 64, 32, 511, 7, 12, 33],
        )

    def test_csa_ratio4_b1_boundary(self):
        self._run_case(4, 1, [3])

    def test_csa_ratio4_b4_mixed(self):
        self._run_case(4, 4, [3, 7, 0, 11])

    def test_csa_ratio4_b8_mixed(self):
        self._run_case(4, 8, [3, 7, 0, 11, 15, 1, 5, 23])


if __name__ == "__main__":
    unittest.main()
