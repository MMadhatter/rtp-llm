# SPDX-License-Identifier: Apache-2.0
"""Transactional model binding for experimental, eager-only external Engram.

This module imports neither CUDA nor the native RTP engine at import time.
Factory injection is for control-flow unit tests; the defaults use the actual
Engram implementation. Successful external mappings live until worker exit.
There is deliberately no live detach, reload, discovery or service protocol.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


EXTERNAL_BUNDLE_ENV = "RTP_ENGRAM_EXTERNAL_BUNDLE"


class EngramModelInitializationError(RuntimeError):
    def __init__(self, code: str, *, resource=None):
        self.code = code
        self.resource = resource
        super().__init__(code)


@dataclass
class EngramModelAttachment:
    phase: str = "INITIALIZING"
    external: bool = False
    storage: Any = field(default=None, repr=False)
    hash_state: Any = field(default=None, repr=False)
    layers: tuple = field(default=(), repr=False)
    failed_resources: list = field(default_factory=list, repr=False)


def _retain_default(storage):
    from .engram_bootstrap import retain_for_process

    retain_for_process(storage)


def require_eager_engram(model, *, is_cuda_graph=False, capture_check=None):
    """Reject unmanaged graph use before submitting any external lookup work."""
    attachment = getattr(model, "_engram_attachment", None)
    if attachment is None or not attachment.external:
        return
    if attachment.phase != "READY":
        raise EngramModelInitializationError("EXTERNAL_ENGRAM_NOT_READY")
    if is_cuda_graph:
        raise EngramModelInitializationError("EXTERNAL_ENGRAM_GRAPH_UNSUPPORTED")
    if capture_check is not None and capture_check():
        raise EngramModelInitializationError("EXTERNAL_ENGRAM_GRAPH_UNSUPPORTED")


def initialize_model_engram(
    model,
    *,
    layout_factory=None,
    hash_factory=None,
    layer_factory=None,
    attach_factory=None,
    retain=None,
    environ=None,
):
    """Build every model-local Engram layer, then publish the complete binding.

    ``model`` supplies ``_v4_args.v41_config``, ``v4.layers``, and
    ``v4.embed.weight.device``. On success it receives ``_engram_storage``,
    ``_engram_hash_state`` and ``_engram_layers``. A successful repeated call
    returns the same attachment. A failed call cannot be retried in-process.

    External identity and GPU content verification belong to
    ``attach_from_bundle``. This function never reads capabilities or infers
    expected identity from an untrusted manifest.
    """
    previous = getattr(model, "_engram_attachment", None)
    if previous is not None:
        if previous.phase in ("READY", "EMPTY"):
            return previous
        raise EngramModelInitializationError("ENGRAM_INITIALIZATION_NOT_RETRYABLE")

    attachment = EngramModelAttachment()
    model._engram_attachment = attachment
    retain = retain or _retain_default
    constructed = []
    published = []
    storage = None
    try:
        if layout_factory is None:
            from .engram import EngramLayout

            layout_factory = EngramLayout
        config = model._v4_args.v41_config
        layout = layout_factory(config)
        if not layout.layer_ids:
            # Draft/no-Engram models do not inspect the external launch bundle.
            attachment.phase = "EMPTY"
            return attachment

        env = os.environ if environ is None else environ
        bundle_path = env.get(EXTERNAL_BUNDLE_ENV)
        attachment.external = bundle_path is not None
        if attachment.external:
            if not isinstance(bundle_path, str) or not bundle_path or not Path(bundle_path).is_absolute():
                raise EngramModelInitializationError("INVALID_EXTERNAL_ENGRAM_BUNDLE_PATH")
            hw = getattr(model, "py_hw_kernel_config", None)
            if hw is not None and bool(getattr(hw, "enable_cuda_graph", False)):
                raise EngramModelInitializationError("EXTERNAL_ENGRAM_GRAPH_UNSUPPORTED")

        if hash_factory is None:
            from .engram import NgramHashState

            hash_factory = NgramHashState
        if layer_factory is None:
            from .engram import Engram

            layer_factory = Engram.from_checkpoint
        device = model.v4.embed.weight.device
        checkpoint_path = config["checkpoint_path"]
        hash_state = hash_factory(layout, checkpoint_path, device=device)
        if attachment.external:
            if attach_factory is None:
                from .engram_bootstrap import attach_from_bundle

                attach_factory = attach_from_bundle
            storage = attach_factory(
                bundle_path, layout=layout, hash_state=hash_state, device=device
            )
            attachment.storage = storage

        for hash_index, layer_id in enumerate(layout.layer_ids):
            if layer_id >= len(model.v4.layers):
                continue  # Preserve main's model-local layer selection.
            if layer_id < 0:
                raise EngramModelInitializationError("INVALID_ENGRAM_LAYER_ID")
            kwargs = {"external_storage": storage} if attachment.external else {}
            module = layer_factory(config, layer_id, checkpoint_path, device, **kwargs)
            constructed.append((layer_id, hash_index, module))

        # No layer can become visible while another layer is still being loaded.
        for layer_id, _, module in constructed:
            layer = model.v4.layers[layer_id]
            old = getattr(layer, "engram", None)
            published.append((layer, old))
            layer.engram = module
        if attachment.external:
            retain(storage)
        attachment.hash_state = hash_state
        attachment.layers = tuple(constructed)
        model._engram_storage = storage
        model._engram_hash_state = hash_state
        model._engram_layers = tuple((lid, index) for lid, index, _ in constructed)
        attachment.phase = "READY"
        return attachment
    except BaseException as original:
        attachment.phase = "FAILED"
        cleanup_errors = []
        for layer, old in reversed(published):
            try:
                layer.engram = old
            except BaseException as error:
                cleanup_errors.append(error)
        if attachment.external:
            for _, _, module in reversed(constructed):
                try:
                    module.release_external_views()
                except BaseException as error:
                    # Keep the module as well as storage: it may still own views.
                    attachment.failed_resources.append(module)
                    cleanup_errors.append(error)
            resource = storage if storage is not None else getattr(original, "resource", None)
            if resource is not None:
                try:
                    resource.close()
                except BaseException as error:
                    cleanup_errors.append(error)
                if cleanup_errors:
                    attachment.failed_resources.append(resource)
                    retain(resource)
        if cleanup_errors:
            raise EngramModelInitializationError(
                "ENGRAM_CLEANUP_INCOMPLETE", resource=attachment
            ) from original
        raise


__all__ = [
    "EXTERNAL_BUNDLE_ENV",
    "EngramModelAttachment",
    "EngramModelInitializationError",
    "initialize_model_engram",
    "require_eager_engram",
]
