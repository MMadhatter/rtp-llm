from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Level2SleepCompatibility:
    """GPU weight owners that must survive a level-2 sleep/wake cycle."""

    lora_adapter_count: int = 0
    merge_lora: bool = False
    local_multimodal_vit: bool = False
    checkpoint_backed_propose_model: bool = False
    eplb_enabled: bool = False
    redundant_expert: int = 0


def validate_level2_sleep_compatibility(
    *,
    enable_sleep_mode: bool,
    sleep_mode_level: int,
    compatibility: Level2SleepCompatibility,
) -> None:
    """Reject level-2 configurations with GPU weights that cannot be reloaded."""
    if not enable_sleep_mode or sleep_mode_level != 2:
        return

    conflicts: List[str] = []
    if compatibility.lora_adapter_count > 0 and not (
        compatibility.merge_lora and compatibility.lora_adapter_count == 1
    ):
        conflicts.append(
            "unmerged or multiple LoRA adapters "
            f"(count={compatibility.lora_adapter_count}, "
            f"merge_lora={compatibility.merge_lora})"
        )
    if compatibility.local_multimodal_vit:
        conflicts.append("local multimodal ViT")
    if compatibility.checkpoint_backed_propose_model:
        conflicts.append("checkpoint-backed propose/draft model")
    if compatibility.eplb_enabled:
        conflicts.append("MoE EPLB")
    if compatibility.redundant_expert > 0:
        conflicts.append(
            f"redundant experts (redundant_expert={compatibility.redundant_expert})"
        )

    if conflicts:
        raise ValueError(
            "sleep mode level 2 is incompatible with active GPU weight owners: "
            + "; ".join(conflicts)
            + ". Use sleep mode level 1 instead."
        )


def reject_dynamic_lora_mutation(
    *, enable_sleep_mode: bool, sleep_mode_level: int
) -> None:
    """Prevent runtime LoRA uploads that level-2 wake cannot reconstruct."""
    if enable_sleep_mode and sleep_mode_level == 2:
        raise ValueError(
            "sleep mode level 2 does not support runtime LoRA add/update/load "
            "because the adapter GPU weights cannot be reconstructed after wake. "
            "Remove adapters or use sleep mode level 1 instead."
        )
