# SPDX-License-Identifier: Apache-2.0
"""Startup-only selection; no CUDA, RPC or container-runtime dependency."""
import logging
import os
from pathlib import Path

MODE_ENV = "RTP_ENGRAM_MODE"
BUNDLE_ENV = "RTP_ENGRAM_EXTERNAL_BUNDLE"
ENDPOINT_ENV = "RTP_ENGRAM_OWNER_ENDPOINT"


def resolve_mode(environ=None):
    """Return (local|shared, reason), preserving legacy explicit-bundle behavior.

    Auto falls back ONLY before attach, when no bundle was delivered. A stale,
    incomplete or invalid publication and every CUDA error remain fatal.
    """
    env = os.environ if environ is None else environ
    mode = env.get(MODE_ENV)
    endpoint = env.get(ENDPOINT_ENV)
    if mode == "local":
        return "local", "explicit_local"
    if endpoint is not None:
        if not endpoint or BUNDLE_ENV in env:
            raise ValueError("AMBIGUOUS_ENGRAM_SOURCE")
        if mode not in (None, "shared", "auto"):
            raise ValueError("INVALID_ENGRAM_MODE")
        return "shared", "rpc_owner"
    if mode is None:
        return ("shared", "legacy_bundle") if BUNDLE_ENV in env else ("local", "default")
    if mode not in ("local", "shared", "auto"):
        raise ValueError("INVALID_ENGRAM_MODE: expected local, shared or auto")
    if mode == "local":
        return "local", "explicit_local"
    path = env.get(BUNDLE_ENV)
    if path is None and mode == "auto":
        return "local", "bundle_not_configured"
    if not isinstance(path, str) or not path or not Path(path).is_absolute():
        raise ValueError("INVALID_EXTERNAL_ENGRAM_BUNDLE_PATH")
    if mode == "auto":
        try:
            # lstat does not treat a dangling symlink as an absent publication.
            Path(path).lstat()
        except FileNotFoundError:
            return "local", "bundle_not_delivered"
        # Permission/IO errors are not evidence that an Owner is absent.
    return "shared", "explicit_shared" if mode == "shared" else "bundle_present"


def freeze_auto_mode(environ=None):
    """Resolve once in the launcher before spawning ranks, then inherit it.

    The launcher must finish bundle delivery before starting this process.
    No live switching is attempted if the publication appears/disappears later.
    """
    env = os.environ if environ is None else environ
    if env.get(MODE_ENV) == "auto":
        mode, reason = resolve_mode(env)
        env[MODE_ENV] = mode
        logging.info("ENGRAM_STORAGE_SELECTION requested=auto selected=%s reason=%s", mode, reason)


def prepare_engram_startup(environ=None):
    """Pin one RPC generation before any ranks spawn; never fetch handles here."""
    env = os.environ if environ is None else environ
    mode, reason = resolve_mode(env)
    if mode == "shared" and reason == "rpc_owner":
        from .engram_rpc import pin_generation
        pin_generation(env)
        env[MODE_ENV] = "shared"
        logging.info("ENGRAM_STORAGE_SELECTION selected=shared source=rpc generation=%s",
                     env["RTP_ENGRAM_RPC_GENERATION"])
    else:
        freeze_auto_mode(env)
