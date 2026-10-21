#!/usr/bin/env python3
"""Explicit TP4 full-checkpoint experiment with opt-in decode Graph, separate from component tests.

runtime/preflight never start a model or allocate EGM. launch starts the actual
RTP server in this process; its lifetime and the Owner's STOP belong to the
operator. No fixture, fake compute module, synthetic projection, or HTTP mock.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
NATIVE_LIBS = ("libth_transformer_config.so", "librtp_compute_ops.so",
               "libth_transformer.so", "libth_grammar_tokenizer_info.so")
PYTHON_PACKAGES = ("torch", "transformers", "tokenizers", "safetensors", "fastsafetensors",
                   "triton", "deep_gemm", "flashinfer", "flash_mla", "tilelang", "fastapi", "uvicorn",
                   "pydantic", "grpc", "requests", "PIL", "engram_native_dev_492ecf45")
TOP_LEVEL_FIELDS = ("text_config", "vision_config", "quantization_config", "dtype",
                    "bos_token_id", "eos_token_id", "pad_token_id", "image_token_id")
TEXT_FIELDS = ("num_hidden_layers", "hidden_size", "vocab_size", "num_attention_heads",
               "head_dim", "qk_rope_head_dim", "compress_ratios", "o_groups", "o_lora_rank",
               "index_head_dim", "index_n_heads", "index_topk", "routed_scaling_factor",
               "num_experts_per_tok", "n_routed_experts", "n_shared_experts", "moe_intermediate_size",
               "max_position_embeddings", "kv_source_layer_ids", "index_source_layer_ids",
               "engram_layer_ids", "engram_num_embeddings", "engram_max_ngram_size",
               "engram_n_heads", "engram_head_dim", "engram_compressed_vocab_size",
               "engram_pad_token_id", "engram_vocab_size")


def sha256_file(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def checkpoint_preflight(checkpoint):
    """Read config, index, and file metadata only; never materialize weights.

    This is a fail-fast screen, not a proof all model descriptors are satisfiable.
    Real loading/forward/HTTP still must pass. Consumer must omit full tables.
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.is_absolute():
        raise ValueError("absolute checkpoint path required")
    raw = json.loads((checkpoint / "config.json").read_text())
    if raw.get("fixture_component_only"):
        raise ValueError("component fixture is not a full LLM checkpoint")
    missing = [key for key in TOP_LEVEL_FIELDS if key not in raw]
    if missing:
        raise ValueError("missing V4.1 top-level fields: " + ",".join(missing))
    text = raw["text_config"]
    if not isinstance(text, dict) or not isinstance(raw["vision_config"], dict):
        raise ValueError("text_config and vision_config must be objects")
    missing = [key for key in TEXT_FIELDS if key not in text]
    if missing:
        raise ValueError("missing V4.1 text fields: " + ",".join(missing))
    architectures = raw.get("architectures", [])
    if architectures and "DeepseekV41ForCausalLM" not in architectures:
        raise ValueError("checkpoint architecture does not identify DeepseekV41ForCausalLM")
    if (not text["engram_layer_ids"]
            or len(text["engram_layer_ids"]) != len(text["engram_num_embeddings"])):
        raise ValueError("full Engram layer geometry required")
    for required in ("tokenizer.json", "tokenizer_config.json", "encoding/encoding.py"):
        if not (checkpoint / required).is_file():
            raise ValueError("missing HTTP tokenizer/encoding asset: " + required)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    weights = index.get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("nonempty safetensors weight_map required")
    forbidden = [name for name in weights if ".engram.embed." in name]
    if forbidden:
        raise ValueError("consumer checkpoint must omit Engram embedding table entries")
    for layer in text["engram_layer_ids"]:
        for suffix in ("wkv.weight", "wkv.scale", "q_weight", "k_weight"):
            if f"layers.{layer}.engram.{suffix}" not in weights:
                raise ValueError(f"missing local Engram projection: layer={layer}, suffix={suffix}")
    # Genuine full backbone and local ViT loading are mandatory in this launch.
    for layer in range(text["num_hidden_layers"]):
        prefix = f"layers.{layer}."
        if not any(key.startswith(prefix + "attn.") for key in weights):
            raise ValueError(f"missing attention weights for full layer {layer}")
        if not any(key.startswith(prefix + "ffn.") for key in weights):
            raise ValueError(f"missing MoE weights for full layer {layer}")
    if not any(key.startswith("vision.") for key in weights) or not any(key.startswith("aligner.") for key in weights):
        raise ValueError("missing local ViT weights")
    for name in ("embed.weight", "head.weight", "norm.weight", "image_start", "image_newline", "image_end"):
        if name not in weights:
            raise ValueError("missing global model weight: " + name)
    files = set()
    for relative in weights.values():
        if (not isinstance(relative, str) or Path(relative).is_absolute()
                or ".." in Path(relative).parts or not relative.endswith(".safetensors")):
            raise ValueError("unsafe checkpoint index path")
        path = checkpoint / relative
        if not path.is_file():
            raise ValueError("checkpoint shard unavailable: " + relative)
        files.add(path)
    engram_bytes = sum(text["engram_num_embeddings"]) * (text["engram_head_dim"] + text["engram_head_dim"] // 32)
    return {"model_type": "deepseek_v41", "full_layer_count": text["num_hidden_layers"],
            "engram_layers": text["engram_layer_ids"], "engram_logical_bytes": engram_bytes,
            "consumer_tensor_count": len(weights), "consumer_shard_count": len(files),
            "consumer_shard_file_bytes": sum(path.stat().st_size for path in files),
            "consumer_has_table_entries": False,
            "config_sha256": sha256_file(checkpoint / "config.json"),
            "index_sha256": sha256_file(checkpoint / "model.safetensors.index.json")}



def runtime_preflight(repository=REPO, *, load_native=False):
    """Require bounded same-checkout library paths; no recursive .so fallback."""
    libs = Path(repository) / "rtp_llm" / "libs"
    missing = [name for name in NATIVE_LIBS if not (libs / name).is_file()]
    if missing:
        raise RuntimeError("missing exact-main runtime libraries: " + ",".join(missing))
    missing_packages = [name for name in PYTHON_PACKAGES if importlib.util.find_spec(name) is None]
    if missing_packages:
        raise RuntimeError("missing runtime Python packages: " + ",".join(missing_packages))
    hashes = {name: sha256_file(libs / name) for name in NATIVE_LIBS}
    if sys.platform.startswith("linux"):
        for name in NATIVE_LIBS:
            checked = subprocess.run(["ldd", str(libs / name)], text=True, capture_output=True, timeout=60)
            if checked.returncode or "not found" in checked.stdout:
                raise RuntimeError("unresolved ELF dependency: " + name + "\n" + checked.stdout + checked.stderr)
    if load_native:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(repository) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        # Respect required main initialization order. This is only a load/import
        # probe, with GPUs hidden; no model is constructed and no weights read.
        code = (
            "import torch; import rtp_llm.ops as ops; ops.ensure_engine_ops_loaded(); "
            "import libth_grammar_tokenizer_info; import rtp_llm.start_server; "
            "import rtp_llm.models.deepseek_v41; "
            "import rtp_llm.openai.renderers.deepseekv41_renderer; "
            "from rtp_llm.dash_sc.app import DashScApp; "
            "from rtp_llm.frontend.frontend_app import FrontendApp; "
            "from rtp_llm.server.backend_manager import BackendManager; "
            "import rtp_llm.start_backend_server; "
            # These are lazy in the RTP wrappers; load the actual wheel and
            # its CUDA bindings before any complete-checkpoint allocation.
            "from flash_mla import flash_mla_sparse_fwd, flash_mla_with_kvcache, get_mla_metadata; "
            "from flash_mla import cuda as flash_mla_cuda; "
            "assert all(callable(f) for f in (flash_mla_sparse_fwd, flash_mla_with_kvcache, "
            "get_mla_metadata, flash_mla_cuda.sparse_prefill_fwd, flash_mla_cuda.sparse_decode_fwd)), "
            "'missing V4.1 attention entrypoints'; "
            "assert not torch.cuda.is_initialized(), 'preflight initialized CUDA'; "
            "print('FULL_HTTP_NATIVE_IMPORT_OK')"
        )
        subprocess.run([sys.executable, "-c", code], cwd=repository, env=env, check=True, timeout=180)
    return {"native_sha256": hashes, "native_import_checked": load_native,
            "note": "Library provenance must match the recorded same-main build; names alone do not prove ABI."}


def launch_environment(args, inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    cuda_graph = getattr(args, "cuda_graph", False)
    if type(cuda_graph) is not bool:
        raise ValueError("cuda_graph must be an explicit boolean")
    # Dedicated test slice: no inherited service discovery, drafts, truncation,
    # or model overrides. Driver/library/device visibility remains operator set.
    for key in ("MODEL_SERVICE_CONFIG", "REMOTE_RPC_SERVER_IP", "REMOTE_VIT_SERVER_IP",
                "RTP_LLM_DECODE_CM2_CONFIG", "RTP_LLM_MULTIMODAL_PART_CM2_CONFIG",
                "SP_CHECKPOINT_PATH", "PTUNING_PATH", "LORA_INFO",
                "GENERATION_PREFILL_CAPTURE_CONFIG", "GENERATION_PREFILL_CUDA_GRAPH_MAX_REQUESTS",
                "PREFILL_CAPTURE_CONFIG", "DECODE_CAPTURE_CONFIG"):
        env.pop(key, None)
    env.update({
        "MODEL_TYPE": "deepseek_v41", "CHECKPOINT_PATH": str(args.checkpoint),
        "TOKENIZER_PATH": str(args.checkpoint), "RTP_ENGRAM_EXTERNAL_BUNDLE": str(args.bundle),
        "RTP_ENGRAM_MAX_ALLOCATION_BYTES": str(args.max_allocation_bytes),
        "RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES": str(args.budget_bytes),
        "LOAD_PYTHON_MODEL": "1", "ROLE_TYPE": "PDFUSION", "USE_LOCAL": "1", "THINK_MODE": "0", "ACT_TYPE": "BF16",
        "TP_SIZE": "4", "EP_SIZE": "4", "DP_SIZE": "1", "WORLD_SIZE": "4", "WORLD_RANK": "0",
        "ENABLE_CUDA_GRAPH": "1" if cuda_graph else "0",
        "RTP_ENGRAM_GRAPH_MODE": "process_lifetime" if cuda_graph else "eager",
        "ENABLE_NATIVE_CUDA_GRAPH": "0", "DECODE_CAPTURE_CONFIG": "1" if cuda_graph else "",
        "SP_TYPE": "", "SP_MODEL_TYPE": "", "HACK_LAYER_NUM": "0",
        "JSON_MODEL_OVERRIDE_ARGS": "", "LOAD_METHOD": "fastsafetensors", "FASTSAFETENSORS_NOGDS": "1",
        "VIT_SEPARATION": "0",
        "DSV4_MOE_STRATEGY": "mega", "DSV4_USE_MEGA_MOE_SE": "0", "DSV4_USE_MEGA_MOE_FUSED": "0",
        "DSV41_SWA_BOUNDED_REPLAY": "0", "DSV41_CED": "0", "FP8_KV_CACHE": "1",
        "SEQ_SIZE_PER_BLOCK": "128", "KERNEL_SEQ_SIZE_PER_BLOCK": "128",
        "MAX_SEQ_LEN": "4096", "MAX_CONTEXT_BATCH_SIZE": "1", "CONCURRENCY_LIMIT": "1",
        "REUSE_CACHE": "1", "RESERVER_RUNTIME_MEM_MB": "32768", "KV_CACHE_MEM_MB": "8192",
        "FRONTEND_SERVER_COUNT": "1", "START_PORT": str(args.port), "WARM_UP": "1", "MODEL_WARM_UP": "1",
        "PYTHONPATH": str(REPO) + os.pathsep + env.get("PYTHONPATH", ""),
    })
    return env


def verify_bundle(args):
    from rtp_llm.models_py.modules.dsv41.engram import EngramLayout, NgramHashState
    from rtp_llm.models_py.modules.dsv41.engram_bootstrap import _read_bundle, hash_state_digest
    from tools.engram_sidecar.framework.checkpoint_owner import read_source

    source = read_source(checkpoint=args.checkpoint, model_revision=args.model_revision)
    layout = EngramLayout(source.config)
    actual = hash_state_digest(NgramHashState(layout, str(args.checkpoint)))
    bundle, _, _, handles = _read_bundle(args.bundle, layout, actual)
    del handles
    if bundle["model_revision"] != args.model_revision:
        raise ValueError("operator checkpoint revision differs from Owner bundle")
    return {key: bundle[key] for key in ("dataset_id", "generation", "model_revision", "hash_abi")}


def chat_payload(*, max_tokens=32):
    # V4.1 resolve_thinking gives this explicit schema precedence over defaults.
    # A short content smoke must not spend its whole token budget on reasoning.
    return {"model": "deepseek_v41", "messages": [{"role": "user", "content": "你好，请用一句话介绍自己。"}],
            "max_tokens": max_tokens, "temperature": 0, "stream": False,
            "thinking": {"type": "disabled"}}


def chat_request(port, *, max_tokens=32):
    payload = chat_payload(max_tokens=max_tokens)
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=180) as response:
        result = json.load(response)
    if "error" in result or "error_code" in result:
        raise RuntimeError("HTTP returned an inference error: " + json.dumps(result, ensure_ascii=False))
    content = result["choices"][0]["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("chat response has no user-visible text; not a successful human-readable output")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("runtime", "preflight", "launch", "chat"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--model-revision")
    parser.add_argument("--max-allocation-bytes", type=int, default=1 << 30)
    parser.add_argument("--budget-bytes", type=int, default=8 << 30)
    parser.add_argument("--port", type=int, default=18630)
    parser.add_argument("--cuda-graph", action="store_true",
                        help="Experimental decode-only batch-1 Graph; mappings remain pinned until worker exit")
    parser.add_argument("--load-native", action="store_true", help="Probe exact native + start_server imports in a GPU-hidden child")
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65000:
        parser.error("port must be in [1024,65000]")
    if args.command == "chat":
        print(json.dumps(chat_request(args.port), ensure_ascii=False, indent=2))
        return 0
    report = {"runtime": runtime_preflight(load_native=args.load_native or args.command == "launch")}
    if args.command == "runtime":
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0
    if (args.checkpoint is None or args.bundle is None or not args.bundle.is_absolute()
            or not args.model_revision):
        parser.error("checkpoint, absolute bundle and model revision required")
    from rtp_llm.models_py.modules.dsv41.engram_capacity import validate_capacity
    validate_capacity(args.max_allocation_bytes, args.budget_bytes)
    os.environ["RTP_ENGRAM_MAX_ALLOCATION_BYTES"] = str(args.max_allocation_bytes)
    os.environ["RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES"] = str(args.budget_bytes)
    report["checkpoint"] = checkpoint_preflight(args.checkpoint)
    report["bundle"] = verify_bundle(args)
    print(json.dumps(report, sort_keys=True, indent=2), flush=True)
    if args.command == "launch":
        # Preflight made no CUDA allocation. exec starts the real main server;
        # it sets rank-local devices in its normal worker initialization.
        os.chdir(REPO)
        os.execve(sys.executable, [sys.executable, "-m", "rtp_llm.start_server"], launch_environment(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
