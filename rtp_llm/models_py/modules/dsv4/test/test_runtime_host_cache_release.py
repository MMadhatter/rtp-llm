import unittest
from types import SimpleNamespace

from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model
from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer


class _V4RuntimeState:
    def __init__(self) -> None:
        attn = SimpleNamespace(
            _cp_ctx=object(),
            compressor=None,
            indexer=None,
        )
        attn.set_cp_ctx = lambda cp_context: setattr(attn, "_cp_ctx", cp_context)
        self.layers = [SimpleNamespace(attn=attn)]
        self.set_cp_info(object(), 4, 2)

    set_cp_info = V4Transformer.set_cp_info
    _propagate_cp_ctx = V4Transformer._propagate_cp_ctx
    release_runtime_host_caches = V4Transformer.release_runtime_host_caches


class RuntimeHostCacheReleaseTest(unittest.TestCase):
    @staticmethod
    def _make_model(params_dict, v4=None) -> DeepSeekV4Model:
        model = object.__new__(DeepSeekV4Model)
        object.__setattr__(model, "params_dict", params_dict)
        if v4 is not None:
            object.__setattr__(model, "v4", v4)
        return model

    def test_releases_attention_params_and_request_scoped_cp_metadata(self) -> None:
        v4 = _V4RuntimeState()
        model = self._make_model({1: object(), 8: object()}, v4)

        DeepSeekV4Model.release_runtime_host_caches(model)

        self.assertEqual(model.params_dict, {})
        self.assertIsNone(v4._cp_info)
        self.assertEqual(v4._cp_size, 1)
        self.assertEqual(v4._cp_rank, 0)
        self.assertFalse(v4._kv_cache_sharded)
        self.assertIsNone(v4.layers[0].attn._cp_ctx)

    def test_tolerates_partially_initialized_model(self) -> None:
        model = self._make_model({1: object()})

        DeepSeekV4Model.release_runtime_host_caches(model)

        self.assertEqual(model.params_dict, {})


if __name__ == "__main__":
    unittest.main()
