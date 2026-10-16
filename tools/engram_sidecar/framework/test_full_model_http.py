"""Preflight negative cases only; these fake files never run a model."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tools.engram_sidecar.framework.full_model_http import (
    NATIVE_LIBS, TEXT_FIELDS, chat_payload, checkpoint_preflight, launch_environment, runtime_preflight,
)


class FullModelPreflightTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        text = dict.fromkeys(TEXT_FIELDS, 1)
        text.update(num_hidden_layers=2, engram_layer_ids=[0, 1], engram_num_embeddings=[100, 101], engram_head_dim=64)
        self.config = {"text_config": text, "vision_config": {}, "quantization_config": {}, "dtype": "bfloat16",
                       "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2, "image_token_id": 3,
                       "architectures": ["DeepseekV41ForCausalLM"]}
        self.weights = {key: "weights.safetensors" for key in (
            "embed.weight", "head.weight", "norm.weight", "image_start", "image_newline", "image_end",
            "vision.fake", "aligner.fake")}
        for layer in range(2):
            for key in ("attn.fake", "ffn.fake", "engram.wkv.weight", "engram.wkv.scale", "engram.q_weight", "engram.k_weight"):
                self.weights[f"layers.{layer}.{key}"] = "weights.safetensors"
        (self.root / "weights.safetensors").write_bytes(b"metadata-only-unit-test")
        (self.root / "encoding").mkdir()
        for name in ("tokenizer.json", "tokenizer_config.json", "encoding/encoding.py"):
            (self.root / name).write_text("{}")
        self.save()

    def save(self):
        (self.root / "config.json").write_text(json.dumps(self.config))
        (self.root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": self.weights}))

    def test_metadata_screen_counts_geometry_without_weight_loading(self):
        report = checkpoint_preflight(self.root)
        self.assertEqual(report["engram_logical_bytes"], 201 * 66)
        self.assertEqual(report["full_layer_count"], 2)
        self.assertFalse(report["consumer_has_table_entries"])

    def test_component_checkpoint_rejected(self):
        self.config["fixture_component_only"] = True
        self.save()
        with self.assertRaisesRegex(ValueError, "not a full LLM"):
            checkpoint_preflight(self.root)

    def test_missing_chat_encoder_rejected(self):
        (self.root / "encoding/encoding.py").unlink()
        with self.assertRaisesRegex(ValueError, "encoding asset"):
            checkpoint_preflight(self.root)

    def test_wrong_hf_architecture_rejected(self):
        self.config["architectures"] = ["Qwen3ForCausalLM"]
        self.save()
        with self.assertRaisesRegex(ValueError, "architecture"):
            checkpoint_preflight(self.root)

    def test_consumer_full_tables_rejected(self):
        self.weights["layers.0.engram.embed.weight"] = "weights.safetensors"
        self.save()
        with self.assertRaisesRegex(ValueError, "must omit"):
            checkpoint_preflight(self.root)

    def test_incomplete_backbone_rejected(self):
        del self.weights["layers.1.ffn.fake"]
        self.save()
        with self.assertRaisesRegex(ValueError, "missing MoE"):
            checkpoint_preflight(self.root)

    def test_missing_local_engram_projection_rejected(self):
        del self.weights["layers.1.engram.k_weight"]
        self.save()
        with self.assertRaisesRegex(ValueError, "local Engram projection"):
            checkpoint_preflight(self.root)

    def test_missing_vision_weights_rejected(self):
        del self.weights["vision.fake"]
        self.save()
        with self.assertRaisesRegex(ValueError, "local ViT"):
            checkpoint_preflight(self.root)

    def test_unsafe_shard_rejected(self):
        self.weights["head.weight"] = "../outside.safetensors"
        self.save()
        with self.assertRaisesRegex(ValueError, "unsafe"):
            checkpoint_preflight(self.root)

    def test_runtime_rejects_only_config_and_compute_before_imports(self):
        libs = self.root / "rtp_llm/libs"
        libs.mkdir(parents=True)
        for name in NATIVE_LIBS[:2]:
            (libs / name).touch()
        with self.assertRaisesRegex(RuntimeError, "libth_transformer.so,libth_grammar_tokenizer_info.so"):
            runtime_preflight(self.root)

    def test_launch_enforces_full_eager_no_draft_and_trusted_capacity(self):
        args = SimpleNamespace(checkpoint=self.root, bundle=self.root / "bundle.json", port=18630,
                               max_allocation_bytes=128 << 30, budget_bytes=256 << 30)
        env = launch_environment(args, {"ENABLE_CUDA_GRAPH": "1", "HACK_LAYER_NUM": "1", "SP_TYPE": "dspark",
                                        "MODEL_SERVICE_CONFIG": "old", "CUDA_VISIBLE_DEVICES": "0,1,2,3"})
        self.assertEqual(env["ENABLE_CUDA_GRAPH"], "0")
        self.assertEqual(env["HACK_LAYER_NUM"], "0")
        self.assertEqual(env["SP_TYPE"], "")
        self.assertEqual(env["THINK_MODE"], "0")
        self.assertEqual(env["TP_SIZE"], "4")
        self.assertEqual(env["EP_SIZE"], "4")
        self.assertEqual(env["RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES"], str(256 << 30))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1,2,3")
        self.assertNotIn("MODEL_SERVICE_CONFIG", env)

    def test_local_pdfusion_clears_external_routing_residue(self):
        args = SimpleNamespace(checkpoint=self.root, bundle=self.root / "bundle.json", port=18630,
                               max_allocation_bytes=128 << 30, budget_bytes=256 << 30)
        stale_keys = ("MODEL_SERVICE_CONFIG", "REMOTE_RPC_SERVER_IP", "REMOTE_VIT_SERVER_IP",
                      "RTP_LLM_DECODE_CM2_CONFIG", "RTP_LLM_MULTIMODAL_PART_CM2_CONFIG")
        env = launch_environment(args, {**dict.fromkeys(stale_keys, "stale-endpoint"), "USE_LOCAL": "0"})
        self.assertEqual(env["USE_LOCAL"], "1")
        self.assertEqual(env["ROLE_TYPE"], "PDFUSION")
        for key in stale_keys:
            self.assertNotIn(key, env)

    def test_short_chat_uses_real_v41_thinking_disabled_schema(self):
        from rtp_llm.openai.renderers.v41_recipe import resolve_thinking

        payload = chat_payload(max_tokens=32)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["max_tokens"], 32)
        for server_default in (True, False):
            self.assertIs(resolve_thinking(payload, server_default)[0], False)


if __name__ == "__main__":
    unittest.main()
