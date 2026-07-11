from __future__ import annotations

import glob
import logging
import os
import re
import shutil
import threading
from collections import OrderedDict
from typing import Any, Mapping

import safetensors.torch
import torch

from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import ModelWeights

# Assuming these imports are from your project and accessible
from rtp_llm.model_loader.weight_module import WeightModule

from .tipc import CudaIpcHelper, SharedMemIpcMeta, SharedMemoryIPCHelper

# Dictionary for renaming specific layer weight names from an external format
# (e.g., 'verl') to the internal 'rtp-llm' format.
RENAME_DICTIONARY = {
    # verl
    "embed_tokens.weight": "embedding",
    "norm.weight": "final_layernorm.gamma",
    "norm.bias": "final_layernorm.beta",
    "lm_head.weight": "lm_head",
    "input_layernorm.weight": "pre_layernorm_weights.gamma",
    "post_attention_layernorm.weight": "post_layernorm_weights.gamma",
    "self_attn.qkv_proj.weight": "self_attention_weights.query_weight.kernel",
    "self_attn.qkv_proj.bias": "self_attention_weights.query_weight.bias",
    "self_attn.o_proj.weight": "self_attention_weights.attention_output_weight.kernel",
    "mlp.gate_proj.weight": "ffn_weights.intermediate_weight.kernel",
    "mlp.up_proj.weight": "ffn_weights.intermediate_weight3.kernel",
    "mlp.down_proj.weight": "ffn_weights.intermediate_weight2.kernel",
    # roll - megatron
    "mbedding.word_embeddings.weight": "embedding",
    "self_attention.linear_proj.weight": "self_attention_weights.attention_output_weight.kernel",
    "self_attention.linear_proj.bias": "self_attention_weights.attention_output_weight.bias",
    "self_attention.linear_qkv.weight": "self_attention_weights.query_weight.kernel",
    "self_attention.linear_qkv.bias": "self_attention_weights.query_weight.bias",
    "mlp.linear_fc1.layer_norm_weight": "post_layernorm_weights.gamma",
    # ???
    "mlp.linear_fc1.weight": "",
}


def rename_function(layer_name: str) -> str:
    """
    Transforms a layer weight name from an external format (e.g., 'verl')
    into the format required by 'rtp-llm'.
    The input format is expected to be like 'model.layers.1.self_attn_qkv_proj.bias'.
    Args:
        layer_name: The layer weight name string from an external source.
    Returns:
        The transformed layer weight name in 'rtp-llm's internal format.
        For example, 'model.layers.1.self_attn_qkv_proj.bias' might become
        'self_attention_weights.query_weight.bias' if it matches a pattern
        and is in the RENAME_DICTIONARY.
    Error Handling:
        This function does not explicitly raise errors but performs string manipulations
        and dictionary lookups. If an unexpected `layer_name` format is provided,
        it might return a string that is not correctly transformed or recognized
        by downstream components.
    """
    # Remove the "model." prefix
    if layer_name.startswith("model."):
        name: str = layer_name[len("model.") :]
    elif layer_name.startswith("decoder."):
        name: str = layer_name[len("decoder.") :]
    else:
        name: str = layer_name
    if "layers" in layer_name:
        # Remove "layers." prefix
        name = name[len("layers.") :]
        # Remove the layer number and the dot following it (e.g., "1." from "1.self_attn...")
        # This assumes the format "layers.<number>.<rest_of_name>"
        first_dot_after_layers = name.find(".")
        if first_dot_after_layers != -1:
            name = name[first_dot_after_layers + 1 :]
        if name in RENAME_DICTIONARY:
            return RENAME_DICTIONARY[name]
        return name
    else:
        if name in RENAME_DICTIONARY:
            return RENAME_DICTIONARY[name]
        return name


class WeightManager:
    """
    Manages model weight updates, including renaming weights from an external
    source and handling inter-process communication (IPC) for tensor transfer.
    It ensures that incoming tensors are correctly processed and sharded/replicated
    as per the rtp-llm model's internal structure (e.g., for Tensor Parallelism (TP)
    or Pipeline Parallelism (PP)).
    """

    def __init__(
        self, device, weight: ModelWeights, model_weights_loader: ModelLoader
    ) -> None:
        """
        Initializes the WeightManager with a model's weights, device information, and weight loader.
        """
        self._s_helper = SharedMemoryIPCHelper()

        # Use the explicit device/weights/loader passed in by the caller (e.g. BaseModel),
        # instead of relying on any global "engine" object.
        if isinstance(device, torch.device):
            self._device = device
        else:
            self._device = torch.device(device)

        self._weights: ModelWeights = weight
        self._weights_loader: ModelLoader = model_weights_loader
        self._weight_module = self._weights_loader._model_weights_info
        self._working_stream: torch.cuda.Stream = torch.cuda.Stream(
            device=self._device,
        )
        # TODO: Consider the actual need for this lock. If updates are always
        # serialized via the server's request handling, a per-update lock might
        # be redundant or require finer-grained locking within _weights.update_...
        self._lock = threading.Lock()

    def extract_layer_number(self, s: str) -> int | None:
        """
        Extracts the layer number (an integer) from a string that follows
        the pattern 'layers.<number>'.
        Args:
            s: The input string, e.g., 'model.layers.2.mlp.gate_proj.weight'.
        Returns:
            The extracted layer number as an integer if found; otherwise, returns `None`.
        Error Handling:
            Returns `None` if the pattern 'layers.<number>' is not found,
            or if the captured group cannot be converted to an integer.
        """
        match = re.search(r"layers\.(\d+)", s)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        else:
            return None

    def update(self, req: dict[str, str]) -> None:
        """
        Receives an Inter-Process Communication (IPC) tensor description and
        updates the corresponding model weights.
        For models with Tensor Parallelism (TP) or Pipeline Parallelism (PP),
        this function expects the transmitted tensor to be a complete, unsharded tensor.
        It then handles the internal sharding or replication according to the
        rtp-llm's specific model parallelism configuration.
        Args:
            req: A dictionary containing the IPC request details. Expected keys are:
                 - "desc": A string describing the tensor's IPC metadata
                           (e.g., `CuIpcTensorMeta` or `SharedMemIpcMeta` encoded string).
                 - "name": A string representing the original name of the weight
                           (e.g., 'model.layers.1.self_attn_qkv_proj.bias').
                 - "method": A string indicating the IPC method used ("cuda_ipc" or "shm").
        Returns:
            None. The method updates internal model weights directly.
        Error Handling:
            - `KeyError`: If "desc", "name", or "method" fields are missing from `req`.
            - `ValueError`: If the "method" is invalid (not "cuda_ipc" or "shm"),
                            or if a layer weight name is invalid and its ID cannot be extracted.
            - `NotImplementedError`: If "cuda_ipc" method is attempted (currently disallowed).
            - `Exception`: If the tensor cannot be built from the IPC metadata (e.g., invalid descriptor).
                          This is a general catch-all for unexpected failures in `_t_helper.build_from_meta`.
        """
        if "desc" not in req:
            raise KeyError(
                "Update request is missing the 'desc' field. "
                "It must contain IPC tensor metadata."
            )
        if "name" not in req:
            raise KeyError(
                "Update request is missing the 'name' field. "
                "It must specify the weight name to update."
            )
        if "method" not in req:
            raise KeyError(
                "Update request is missing the 'method' field. "
                "It must specify the IPC method (e.g., 'cuda_ipc' or 'shm')."
            )
        method: str = req["method"]
        desc: str = req["desc"]
        name: str = req["name"]
        stored_name: str = name

        if method not in {"cuda_ipc", "shm"}:
            raise ValueError(
                f"Invalid IPC method '{method}' provided. Only 'cuda_ipc' and 'shm' are allowed."
            )
        tensor: torch.Tensor | None = None

        if method == "cuda_ipc":
            helper = CudaIpcHelper()
            tensor = helper.build_from_meta(bytes.fromhex(desc))
        else:  # method == "shm"
            sm_meta: SharedMemIpcMeta = SharedMemIpcMeta.decode(desc)
            tensor = self._s_helper.build_from_meta(sm_meta)

        if tensor is None:
            logging.error(
                f"Fail to build tensor from ipc description {desc}, method: {method}"
            )
            # This should ideally not be reached if build_from_meta consistently returns a tensor or raises an error.
            raise Exception(
                f"Failed to build tensor from IPC description '{desc}' using method '{method}'. Tensor is None."
            )

        logging.info(
            f"update weight request: {name}, shape: {tensor.shape}, device: {tensor.device}, dtype: {tensor.dtype}"
        )
        with torch.cuda.stream(self._working_stream):
            config = self._weights_loader.get_load_config()
            if "layers" in name:
                # This is a layer-specific weight
                layer_id: int | None = self.extract_layer_number(name)
                if layer_id is None:
                    raise ValueError(
                        f"Invalid layer weight name format: '{name}'. "
                        "Could not extract layer number. Expected format like 'model.layers.<id>...'"
                    )
                name: str = rename_function(name)
                fail: bool = True

                for receptor in self._weight_module.layer_weights[layer_id]:
                    if receptor.name == name or (
                        "ffn_weights" in name and receptor.name == "__ffn_weights__"
                    ):
                        assert isinstance(receptor, WeightModule)

                        # split tensor into shards
                        shard = receptor.update(
                            tensor=tensor,
                            device=self._device,
                            load_config=config,
                            module_name=name,
                        )
                        if isinstance(shard, dict):
                            shard = next(iter(shard.values()))

                        # update tensor weight
                        self._weights.update_layer_weight(
                            layer_id=layer_id, name=name, data=shard
                        )
                        fail = False

                if fail:
                    raise KeyError(
                        f"{stored_name} not found. wanted name list is {[w.name for w in self._weight_module.layer_weights[layer_id]]}"
                    )

            else:
                # weight is global weight

                name: str = rename_function(name)
                fail: bool = True
                for weight in self._weight_module.weights:
                    if weight.name == name:
                        shard: dict = weight.update(
                            tensor,
                            self._device,
                            load_config=self._weights_loader.get_load_config(),
                        )
                        if isinstance(shard, dict):
                            shard = next(iter(shard.values()))
                        self._weights.update_global_weight(name=name, data=shard)
                        fail = False

                if fail:
                    raise KeyError(
                        f"{stored_name} not found. wanted name list is {[w.name for w in self._weight_module.weights]}"
                    )

            self._working_stream.synchronize()

    # ------------------------------------------------------------------
    # Sleep level 2 (discard weights) raw disk backup / restore.
    #
    # In level-2 sleep the weights region is opened without torch_memory_saver
    # host cpu_backup, so ``pause("weights")`` frees GPU *and* host memory and
    # ``resume("weights")`` remaps blank pages at the same virtual address. To
    # bring the *same* weights back on wake, the C++ sleep hooks call
    # :meth:`dump_raw_backup` once before the first pause and
    # :meth:`restore_raw_backup` after every resume.
    #
    # These operate on the already-processed live GPU tensors (post dequant /
    # MoE fusion / TP split), so dump is contiguous bytes and restore is an
    # in-place ``copy_`` — no re-run of the loader pipeline, and the tensors'
    # ``data_ptr`` (aliased by the C++ engine and baked into CUDA graphs) is
    # preserved. Scope: the base ``ModelWeights`` only; LoRA adapters,
    # multimodal ViT, and C++-side dynamic EPLB expert buffers are out of scope
    # for v1 (see weight_memory_saver.py coverage checklist).
    # ------------------------------------------------------------------

    _L2_BACKUP_COMPLETE_SUFFIX = ".complete"

    def _l2_backup_dir(self) -> str:
        """Resolve the per-process local-disk directory for level-2 raw backups.

        The base comes from ``--sleep_l2_backup_dir`` / ``SLEEP_L2_BACKUP_DIR``
        (should be local NVMe/SSD; the dump can be hundreds of GB or, for very
        large models, ~1 TB). When unset it falls back to ``/tmp/rtp_llm_sleep_l2``
        with a warning. A per-``<pid>`` subdirectory is always appended so
        co-located instances (prefill + decode on one node, or multiple rank
        workers sharing one configured base) never collide — their live tensors
        differ but the rank tag can be identical. The whole per-pid dir is
        discarded after a successful wake (see :meth:`restore_raw_backup`), so no
        permanent copy is left on disk.

        Warns when the path resolves onto a tmpfs/ramfs mount, since that keeps
        the weights in host RAM and defeats the point of discarding them.
        """
        base = os.environ.get("SLEEP_L2_BACKUP_DIR", "").strip()
        if not base:
            base = "/tmp/rtp_llm_sleep_l2"
            logging.warning(
                "SLEEP_L2_BACKUP_DIR is not set; falling back to %s for level-2 "
                "weight backup. Point --sleep_l2_backup_dir at a local NVMe/SSD "
                "path with enough free space (dump size ~= weight size).",
                base,
            )
        out_dir = os.path.join(base, str(os.getpid()))
        try:
            fstype = self._filesystem_type(out_dir)
            if fstype in ("tmpfs", "ramfs"):
                logging.warning(
                    "level-2 backup dir %s resolves to a %s (RAM-backed) mount; "
                    "the weight backup will consume host RAM instead of disk, "
                    "defeating weight discard. Point it at local NVMe/SSD.",
                    out_dir,
                    fstype,
                )
        except Exception:  # pragma: no cover - best-effort diagnostics only
            pass
        return out_dir

    @staticmethod
    def _filesystem_type(path: str) -> str | None:
        """Best-effort filesystem type for ``path`` via /proc/mounts (longest prefix)."""
        abs_path = os.path.abspath(path)
        # Walk up to the nearest existing ancestor (the dir may not exist yet).
        probe = abs_path
        while probe and not os.path.exists(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        best_mount = ""
        best_type: str | None = None
        try:
            with open("/proc/mounts", "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    mount_point, fstype = parts[1], parts[2]
                    if (
                        probe == mount_point
                        or probe.startswith(mount_point.rstrip("/") + "/")
                    ) and len(mount_point) >= len(best_mount):
                        best_mount = mount_point
                        best_type = fstype
        except FileNotFoundError:
            return None
        return best_type

    def _l2_rank_tag(self) -> tuple[int, int, int]:
        config = self._weights_loader.get_load_config()
        return int(config.tp_rank), int(config.dp_rank), int(config.ep_rank)

    def _l2_marker_path(self, out_dir: str, tp: int, dp: int, ep: int) -> str:
        return os.path.join(
            out_dir,
            f"rank_{tp:02d}_{dp:02d}_{ep:02d}{self._L2_BACKUP_COMPLETE_SUFFIX}",
        )

    def _l2_file_prefix(self, out_dir: str, tp: int, dp: int, ep: int) -> str:
        return f"{out_dir}/model-{tp:02d}-{dp:02d}-{ep:02d}-part-"

    def dump_raw_backup(self) -> None:
        """Serialize the live GPU weights to a local-disk raw backup (idempotent).

        Called once before the first level-2 ``pause("weights")``. Skips if a
        rank-scoped ``.complete`` marker already exists. Weights are written as
        rank-prefixed keys (matching ``ModelWeights.layer_weight_prefix`` /
        ``global_weight_prefix``) into ~6 GB safetensors partitions, staged
        through pageable host memory one tensor at a time.
        """
        tp, dp, ep = self._l2_rank_tag()
        out_dir = self._l2_backup_dir()
        marker = self._l2_marker_path(out_dir, tp, dp, ep)
        if os.path.exists(marker):
            logging.info(
                "dump_raw_backup: backup already complete at %s, skip", out_dir
            )
            return

        os.makedirs(out_dir, exist_ok=True)
        layer_prefix = ModelWeights.layer_weight_prefix(tp, dp, ep)
        global_prefix = ModelWeights.global_weight_prefix(tp, dp, ep)
        file_prefix = self._l2_file_prefix(out_dir, tp, dp, ep)
        max_size = 6 * 1024**3  # 6GB partitions, mirroring dump_weight_as_ft_style

        part_idx = 0
        current_size = 0
        current_dict: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        total_bytes = 0

        def flush(force: bool) -> None:
            nonlocal part_idx, current_size, current_dict
            if not current_dict:
                return
            if not force and current_size < max_size:
                return
            filename = f"{file_prefix}{part_idx:05d}.safetensors"
            safetensors.torch.save_file(current_dict, filename)
            logging.info(
                "dump_raw_backup: saved partition %d (%.2f GB) -> %s",
                part_idx,
                current_size / 1024**3,
                filename,
            )
            current_dict = OrderedDict()
            part_idx += 1
            current_size = 0

        with self._lock:
            with torch.cuda.stream(self._working_stream), torch.inference_mode():
                for layer_id, layer_dict in enumerate(self._weights.weights):
                    for name, tensor in layer_dict.items():
                        key = f"{layer_prefix}{layer_id}.{name}"
                        current_dict[key] = tensor.detach().cpu().contiguous()
                        nbytes = tensor.numel() * tensor.element_size()
                        current_size += nbytes
                        total_bytes += nbytes
                        flush(force=False)
                for name, tensor in self._weights.global_weights.items():
                    key = f"{global_prefix}{name}"
                    current_dict[key] = tensor.detach().cpu().contiguous()
                    nbytes = tensor.numel() * tensor.element_size()
                    current_size += nbytes
                    total_bytes += nbytes
                    flush(force=False)
                flush(force=True)
                self._working_stream.synchronize()

        # Marker written last so a crashed/partial dump is not treated as valid.
        with open(marker, "w") as f:
            f.write(f"parts={part_idx} bytes={total_bytes}\n")
        logging.info(
            "dump_raw_backup: complete rank(%02d,%02d,%02d) %d parts, %.2f GB at %s",
            tp,
            dp,
            ep,
            part_idx,
            total_bytes / 1024**3,
            out_dir,
        )

    def restore_raw_backup(self) -> None:
        """Reload the raw disk backup in place into the existing GPU tensors.

        Called after ``resume("weights")`` (which has remapped blank pages at
        the original VA). For each stored tensor, validates shape/dtype against
        the live tensor and ``copy_`` s the bytes back — preserving ``data_ptr``
        so C++ aliases and captured CUDA graphs remain valid.
        """
        tp, dp, ep = self._l2_rank_tag()
        out_dir = self._l2_backup_dir()
        marker = self._l2_marker_path(out_dir, tp, dp, ep)
        if not os.path.exists(marker):
            raise FileNotFoundError(
                f"restore_raw_backup: no completed level-2 backup found at {marker}. "
                "Was dump_raw_backup run on sleep?"
            )
        layer_prefix = ModelWeights.layer_weight_prefix(tp, dp, ep)
        global_prefix = ModelWeights.global_weight_prefix(tp, dp, ep)
        files = sorted(
            glob.glob(f"{self._l2_file_prefix(out_dir, tp, dp, ep)}*.safetensors")
        )
        if not files:
            raise FileNotFoundError(
                f"restore_raw_backup: marker present but no partitions matched "
                f"{self._l2_file_prefix(out_dir, tp, dp, ep)}*.safetensors"
            )

        restored = 0
        with self._lock:
            with torch.cuda.stream(self._working_stream), torch.inference_mode():
                for filename in files:
                    state = safetensors.torch.load_file(filename, device="cpu")
                    for key, cpu_tensor in state.items():
                        ori = self._resolve_live_tensor(
                            key, layer_prefix, global_prefix
                        )
                        if ori is None:
                            logging.warning(
                                "restore_raw_backup: key %s has no live tensor, skip",
                                key,
                            )
                            continue
                        if (
                            ori.shape != cpu_tensor.shape
                            or ori.dtype != cpu_tensor.dtype
                        ):
                            raise ValueError(
                                f"restore_raw_backup: mismatch for {key}: live "
                                f"{tuple(ori.shape)}/{ori.dtype} vs backup "
                                f"{tuple(cpu_tensor.shape)}/{cpu_tensor.dtype}"
                            )
                        # Direct H2D copy into the existing storage (no GPU temp).
                        ori.copy_(cpu_tensor)
                        restored += 1
                    del state
                self._working_stream.synchronize()
        logging.info(
            "restore_raw_backup: restored %d tensors from %d parts at %s",
            restored,
            len(files),
            out_dir,
        )
        # Backup is use-once: discard it after a successful restore. Weights can be
        # hundreds of GB (up to ~1TB for very large models), so leaving a permanent
        # on-disk copy is not acceptable. The next /sleep re-dumps from the live GPU
        # tensors. Reached only on success (the loop raises on any shape/dtype
        # mismatch), so we never delete a backup we failed to fully apply.
        shutil.rmtree(out_dir, ignore_errors=True)
        logging.info("restore_raw_backup: discarded use-once backup dir %s", out_dir)

    def _resolve_live_tensor(
        self, key: str, layer_prefix: str, global_prefix: str
    ) -> torch.Tensor | None:
        if key.startswith(layer_prefix):
            rest = key[len(layer_prefix) :]  # "{layer_id}.{name}"
            layer_id_str, _, name = rest.partition(".")
            layer_id = int(layer_id_str)
            return self._weights.weights[layer_id].get(name)
        if key.startswith(global_prefix):
            name = key[len(global_prefix) :]
            return self._weights.global_weights.get(name)
        return None
