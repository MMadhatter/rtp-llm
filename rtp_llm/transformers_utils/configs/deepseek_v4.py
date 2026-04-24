"""HF ``PretrainedConfig`` subclass for DeepSeek-V4.

The class is intentionally thin — every field is forwarded straight through
``PretrainedConfig.__init__(**kwargs)``, exactly like vLLM's
``vllm/transformers_utils/configs/deepseek_v4.py`` (PR #40760).

Two side-effects on import:
  1. The class is registered in ``transformers.models.auto`` so that
     ``AutoConfig.from_pretrained(...)`` resolves ``model_type == "deepseek_v4"``
     for callers that exist outside rtp_llm's own ``_from_hf`` parser.
  2. A summary of V4-specific keys is documented in the docstring below
     for reference; the actual default values live in the HF Hub config.json.

V4-specific kwargs the model.config.json carries (see
``develop_ds_v4.md`` § 2 and the Flash-Base/Pro configs):

  - ``head_dim``: 512 (single MQA head value width)
  - ``num_key_value_heads``: 1 (MQA)
  - ``q_lora_rank``: 1024 (Flash) / 1536 (Pro)
  - ``qk_rope_head_dim``: 64 (partial RoPE on trailing dims)
  - ``o_groups``: grouped output projection group count (8 / 16)
  - ``o_lora_rank``: 1024
  - ``sliding_window``: SWA bypass window (128)
  - ``compress_rope_theta``: independent RoPE base for compressed KV (160000)
  - ``compress_ratios``: per-layer ratio table; entries are
        0   = non-compressed (SWA-only / MTP placeholder),
        4   = CSA (compress every 4 raw tokens to 1 entry),
        128 = HCA (compress every 128 raw tokens to 1 entry)
  - ``hc_mult`` (4), ``hc_sinkhorn_iters`` (20), ``hc_eps`` (1e-6):
        Manifold-Constrained Hyper-Connection (mHC) hyperparams.
  - ``num_hash_layers``: number of leading MoE layers using deterministic
        hash routing (3).
  - ``index_n_heads``, ``index_head_dim``, ``index_topk``: lightning
        indexer geometry for CSA layers.
  - ``scoring_func``: ``"sqrtsoftplus"`` (V4) instead of V3's ``"sigmoid"``.
  - ``swiglu_limit``: SwiGLU clamp bound (10.0).
  - ``num_nextn_predict_layers``: 1 (MTP depth, same as V3).
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES


class DeepseekV4Config(PretrainedConfig):
    """Thin HF config wrapper for DeepSeek-V4.

    All fields are accepted via kwargs and stored as attributes by the parent
    ``PretrainedConfig.__init__``. Code that needs typed access should read
    ``getattr(config, "<field>", <default>)``.
    """

    model_type = "deepseek_v4"

    # The official model_type on HF is "deepseek_v4" (singular). We keep the
    # alias list small but explicit so static analyzers can find it.
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# Register so that ``transformers.AutoConfig`` resolves model_type lookups
# for V4 even on a stock ``transformers==4.57.1`` install (which predates V4).
# Idempotent: re-importing this module won't shadow an upstream registration.
if "deepseek_v4" not in CONFIG_MAPPING_NAMES:
    CONFIG_MAPPING_NAMES["deepseek_v4"] = "DeepseekV4Config"
