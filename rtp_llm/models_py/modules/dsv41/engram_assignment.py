# SPDX-License-Identifier: Apache-2.0
"""Platform-assigned Owner contract; no discovery, election or GPU work."""
import re
from uuid import UUID

DISCOVERY_ENV = "RTP_ENGRAM_DISCOVERY"
FIELDS = ("authorization_group", "fabric_domain", "owner_instance_id", "owner_epoch")


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,256}", value):
        raise ValueError("PLATFORM_IDENTIFIER_INVALID")
    return value


def validate_assignment(value):
    if type(value) is not dict or set(value) != set(FIELDS):
        raise ValueError("PLATFORM_ASSIGNMENT_REQUIRED")
    for key in FIELDS:
        identifier(value[key])
    try:
        if str(UUID(value["owner_epoch"])) != value["owner_epoch"]:
            raise ValueError()
    except (TypeError, ValueError, AttributeError):
        raise ValueError("PLATFORM_OWNER_EPOCH_INVALID") from None
    return dict(value)


def configured_assignment(env):
    discovery = env.get(DISCOVERY_ENV)
    if discovery is None:
        return None  # Existing direct-RPC and Registry clients remain compatible.
    if discovery != "platform":
        raise ValueError("ENGRAM_DISCOVERY_INVALID")
    if any(
        k in env
        for k in (
            "RTP_ENGRAM_REGISTRY_ENDPOINT",
            "RTP_ENGRAM_RESOLVED_OWNER_ENDPOINT",
            "RTP_ENGRAM_EXTERNAL_BUNDLE",
        )
    ):
        raise ValueError("AMBIGUOUS_ENGRAM_SOURCE")
    if not env.get("RTP_ENGRAM_OWNER_ENDPOINT"):
        raise ValueError("PLATFORM_OWNER_ENDPOINT_REQUIRED")
    identifier(env.get("RTP_ENGRAM_INSTANCE_ID"))
    return validate_assignment({k: env.get("RTP_ENGRAM_" + k.upper()) for k in FIELDS})


def validate_imex_device(env):
    """Require exactly the platform-assigned accessible channel in this container."""
    import os
    import stat
    from pathlib import Path

    channel = env.get("RTP_ENGRAM_IMEX_CHANNEL", "")
    if not re.fullmatch(r"0|[1-9][0-9]*", channel):
        raise ValueError("PLATFORM_IMEX_CHANNEL_REQUIRED")
    root = Path("/dev/nvidia-caps-imex-channels")
    accessible = []
    for path in root.glob("channel*"):
        if stat.S_ISCHR(path.stat().st_mode) and os.access(path, os.R_OK | os.W_OK):
            accessible.append(path.name)
    if accessible != ["channel" + channel]:
        raise ValueError("PLATFORM_IMEX_CHANNEL_NOT_EXCLUSIVE")
