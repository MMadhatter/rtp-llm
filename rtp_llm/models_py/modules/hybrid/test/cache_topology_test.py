"""Tests for the per-layer KV cache topology helper (DeepSeek-V4 M4)."""

from unittest import TestCase, main

from rtp_llm.models_py.modules.hybrid.cache_topology import (
    LayerCacheKind,
    LayerCacheSpec,
    block_raw_tokens,
    derive_layer_cache_plan,
    lcm,
    state_cache_block_size,
    total_compressed_blocks,
)


# Authoritative DeepSeek-V4-Flash-Base compress_ratios (44 entries).
FLASH_BASE_COMPRESS_RATIOS = [
    0,   0,
    4, 128,   4,   4,   4, 128,   4,   4,
    4, 128,   4,   4,   4, 128,   4,   4,
    4, 128,   4,   4,   4, 128,   4,   4,
    4, 128,   4,   4,   4, 128,   4,   4,
    4, 128,   4,   4,   4, 128,   4,   4,
    4,   0,
]


class LcmTest(TestCase):
    def test_basic(self):
        self.assertEqual(lcm(4, 128), 128)
        self.assertEqual(lcm(6, 8), 24)

    def test_zero_input(self):
        self.assertEqual(lcm(0, 5), 0)
        self.assertEqual(lcm(5, 0), 0)


class BlockRawTokensTest(TestCase):
    def test_v4_flash_block_covers_128_raw_tokens(self):
        # Paper §3.6.2: cache block covers lcm(m, m') = lcm(4, 128) = 128.
        self.assertEqual(block_raw_tokens(4, 128), 128)


class DeriveLayerCachePlanTest(TestCase):
    def test_flash_base_layer_count(self):
        plan = derive_layer_cache_plan(FLASH_BASE_COMPRESS_RATIOS)
        self.assertEqual(len(plan), len(FLASH_BASE_COMPRESS_RATIOS))

    def test_flash_base_kind_distribution(self):
        plan = derive_layer_cache_plan(FLASH_BASE_COMPRESS_RATIOS)
        kinds = [spec.kind for spec in plan]
        # Three NON_CACHE entries (idx 0, 1, 43), no SWA_ONLY in Flash-Base.
        self.assertEqual(kinds.count(LayerCacheKind.NON_CACHE), 3)
        self.assertEqual(kinds.count(LayerCacheKind.SWA_ONLY), 0)
        # CSA layers: ratio == 4 entries, HCA layers: ratio == 128 entries.
        n_csa = sum(1 for r in FLASH_BASE_COMPRESS_RATIOS if r == 4)
        n_hca = sum(1 for r in FLASH_BASE_COMPRESS_RATIOS if r == 128)
        self.assertEqual(kinds.count(LayerCacheKind.CSA), n_csa)
        self.assertEqual(kinds.count(LayerCacheKind.HCA), n_hca)

    def test_csa_entries_per_block_is_32(self):
        plan = derive_layer_cache_plan(FLASH_BASE_COMPRESS_RATIOS)
        for spec in plan:
            if spec.kind == LayerCacheKind.CSA:
                self.assertEqual(spec.entries_per_block, 32)
                self.assertEqual(spec.compress_ratio, 4)

    def test_hca_entries_per_block_is_1(self):
        plan = derive_layer_cache_plan(FLASH_BASE_COMPRESS_RATIOS)
        for spec in plan:
            if spec.kind == LayerCacheKind.HCA:
                self.assertEqual(spec.entries_per_block, 1)
                self.assertEqual(spec.compress_ratio, 128)

    def test_non_cache_zero_entries(self):
        plan = derive_layer_cache_plan(FLASH_BASE_COMPRESS_RATIOS)
        for spec in plan:
            if spec.kind == LayerCacheKind.NON_CACHE:
                self.assertEqual(spec.entries_per_block, 0)
                self.assertEqual(spec.compress_ratio, 1)

    def test_layer_idx_is_sequential(self):
        plan = derive_layer_cache_plan([4, 128, 4, 0])
        self.assertEqual([s.layer_idx for s in plan], [0, 1, 2, 3])

    def test_pure_swa_route(self):
        """V4-Pro spec-style: first 2 layers are pure SWA (ratio 1)."""
        plan = derive_layer_cache_plan(
            [1, 1, 4, 128, 4],
            pure_swa_ratios=(1,),
        )
        self.assertEqual(plan[0].kind, LayerCacheKind.SWA_ONLY)
        self.assertEqual(plan[1].kind, LayerCacheKind.SWA_ONLY)
        self.assertEqual(plan[2].kind, LayerCacheKind.CSA)
        self.assertEqual(plan[3].kind, LayerCacheKind.HCA)

    def test_unknown_ratio_raises(self):
        with self.assertRaises(ValueError):
            # 7 is neither 0, 4, nor 128.
            derive_layer_cache_plan([4, 7, 128])

    def test_zero_m_raises(self):
        with self.assertRaises(ValueError):
            derive_layer_cache_plan([4], m=0, m_prime=128)
        with self.assertRaises(ValueError):
            derive_layer_cache_plan([4], m=4, m_prime=0)

    def test_returns_immutable_specs(self):
        plan = derive_layer_cache_plan([4, 128])
        with self.assertRaises(Exception):
            # frozen=True dataclass — assignment should fail.
            plan[0].layer_idx = 99   # type: ignore[misc]

    def test_custom_m_and_m_prime(self):
        # Smaller hypothetical config: m=2, m_prime=8 -> lcm=8 -> CSA k1=4, HCA k2=1
        plan = derive_layer_cache_plan([2, 8, 0], m=2, m_prime=8)
        self.assertEqual(plan[0].kind, LayerCacheKind.CSA)
        self.assertEqual(plan[0].entries_per_block, 4)
        self.assertEqual(plan[1].kind, LayerCacheKind.HCA)
        self.assertEqual(plan[1].entries_per_block, 1)


class StateCacheBlockSizeTest(TestCase):
    def test_includes_window_and_tail(self):
        # n_win=128, head_dim=512, bytes=2 (bf16):
        #   (128 + 127) * 2 (K&V) * 512 * 2 = 522,240 bytes
        size = state_cache_block_size(n_win=128, head_dim=512, m=4, m_prime=128)
        self.assertEqual(size, (128 + 127) * 2 * 512 * 2)

    def test_scales_with_bytes_per_elem(self):
        bf16 = state_cache_block_size(n_win=8, head_dim=64)
        fp32 = state_cache_block_size(n_win=8, head_dim=64, bytes_per_elem=4)
        self.assertEqual(fp32, 2 * bf16)


class TotalCompressedBlocksTest(TestCase):
    def test_excludes_non_cached_layers(self):
        plan = derive_layer_cache_plan(FLASH_BASE_COMPRESS_RATIOS)
        n_cached = sum(
            1 for s in plan
            if s.kind in (LayerCacheKind.CSA, LayerCacheKind.HCA)
        )
        # 41 cached layers (44 total - 3 NON_CACHE).
        self.assertEqual(n_cached, 41)
        self.assertEqual(total_compressed_blocks(plan, 10), 410)

    def test_zero_when_no_cached_layers(self):
        plan = [LayerCacheSpec(0, LayerCacheKind.NON_CACHE, 1, 0)]
        self.assertEqual(total_compressed_blocks(plan, 10), 0)


if __name__ == "__main__":
    main()
