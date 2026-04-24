"""Custom HF ``PretrainedConfig`` subclasses for model_types not yet shipped
upstream (mirrors vLLM's ``vllm/transformers_utils/configs/``)."""

from rtp_llm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

__all__ = ["DeepseekV4Config"]
