import json
import logging
import os
from typing import Any, Set

SLEEP_CONTROL_ADDRESSES_ENV = "RTP_LLM_SLEEP_CONTROL_ADDRESSES"


def _dedupe_addresses(addresses: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: Set[str] = set()
    for address in addresses:
        if address in seen:
            continue
        seen.add(address)
        deduped.append(address)
    return deduped


def _parse_address_list(raw_value: str) -> list[str]:
    value = raw_value.strip()
    if not value:
        return []
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("JSON value must be a list")
        addresses = [str(item).strip() for item in parsed]
    else:
        addresses = [item.strip() for item in value.replace(";", ",").split(",")]

    addresses = [address for address in addresses if address]
    for address in addresses:
        if ":" not in address:
            raise ValueError(f"invalid address [{address}], expected host:port")
    return _dedupe_addresses(addresses)


def get_control_addrs_from_env(
    env_name: str = SLEEP_CONTROL_ADDRESSES_ENV,
) -> list[str]:
    """Get lifecycle control addresses from an explicit env override.

    This is for separated frontend or multi-part deployments where local
    world_info may only contain the current node. The value accepts either a
    comma/semicolon-separated list or a JSON string list.
    """
    raw_value = os.environ.get(env_name, "")
    if not raw_value.strip():
        return []
    addresses = _parse_address_list(raw_value)
    logging.info("using control-plane addresses from %s: %s", env_name, addresses)
    return addresses


def get_dp_addrs_from_world_info(
    world_info: Any, parallelism_config: Any
) -> list[str]:
    """Get data parallel addresses from world_info."""
    ffn_disaggregate_config = parallelism_config.ffn_disaggregate_config
    logging.info(
        f"frontend worker ffn_disaggregate_config: {ffn_disaggregate_config.to_string()}"
    )
    # If FFN disaggregate is enabled, use only serving ranks; additional ranks
    # are internal to that node.
    if ffn_disaggregate_config.enable_ffn_disaggregate:
        serving_ranks = (
            ffn_disaggregate_config.attention_tp_size
            * ffn_disaggregate_config.attention_dp_size
        )
        members = world_info.members[:serving_ranks]
        logging.info(
            f"FFN disaggregate enabled, limiting addresses to {serving_ranks} serving ranks: {members}"
        )
    else:
        members = [
            member
            for member in world_info.members
            if (member.world_rank % parallelism_config.tp_size) == 0
        ]

    addresses = [f"{member.ip}:{member.rpc_server_port}" for member in members]
    logging.info(
        f"[world_rank: {parallelism_config.world_rank}] "
        f"using addresses from world_info: {addresses}"
    )
    return addresses


def get_control_addrs_from_world_info(world_info: Any) -> list[str]:
    """Get all backend RPC addresses that must receive lifecycle control."""
    addresses: list[str] = []
    for member in sorted(world_info.members, key=lambda x: x.world_rank):
        addresses.append(f"{member.ip}:{member.rpc_server_port}")
    addresses = _dedupe_addresses(addresses)
    logging.info("using control-plane addresses from world_info: %s", addresses)
    return addresses
