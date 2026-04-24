#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include <sstream>

namespace rtp_llm {

// Helper function to convert KvCacheDataType enum to string
static std::string kvCacheDataTypeToString(KvCacheDataType kv_cache_dtype) {
    switch (kv_cache_dtype) {
        case KvCacheDataType::BASE:
            return "BASE";
        case KvCacheDataType::INT8:
            return "INT8";
        case KvCacheDataType::FP8:
            return "FP8";
        default:
            return "UNKNOWN(" + std::to_string(static_cast<int>(kv_cache_dtype)) + ")";
    }
}

std::string AttentionConfigs::DebugAttentionConfigStr() const {
    std::ostringstream oss;
    oss << "  head_num: " << head_num << std::endl;
    oss << "  kv_head_num: " << kv_head_num << std::endl;
    oss << "  size_per_head: " << size_per_head << std::endl;
    oss << "  tokens_per_block: " << tokens_per_block << std::endl;
    oss << "  is_causal: " << is_causal << std::endl;
    oss << "  q_scaling: " << q_scaling << std::endl;
    oss << "  fuse_qkv_add_bias: " << fuse_qkv_add_bias << std::endl;
    oss << "  use_logn_attn: " << use_logn_attn << std::endl;
    oss << "  use_mla: " << use_mla << std::endl;
    oss << "  q_lora_rank: " << q_lora_rank << std::endl;
    oss << "  kv_lora_rank: " << kv_lora_rank << std::endl;
    oss << "  nope_head_dim: " << nope_head_dim << std::endl;
    oss << "  rope_head_dim: " << rope_head_dim << std::endl;
    oss << "  v_head_dim: " << v_head_dim << std::endl;
    oss << "  softmax_extra_scale: " << softmax_extra_scale << std::endl;
    oss << "  kv_cache_dtype: " << kvCacheDataTypeToString(kv_cache_dtype) << std::endl;
    oss << "  need_rope_kv_cache: " << need_rope_kv_cache << std::endl;
    oss << "  is_sparse: " << is_sparse << std::endl;
    oss << "  indexer_head_dim: " << indexer_head_dim << std::endl;
    oss << "  indexer_head_num: " << indexer_head_num << std::endl;
    oss << "  indexer_topk: " << indexer_topk << std::endl;
    oss << "  o_groups: " << o_groups << std::endl;
    oss << "  o_lora_rank: " << o_lora_rank << std::endl;
    oss << "  sliding_window: " << sliding_window << std::endl;
    oss << "  compress_rope_theta: " << compress_rope_theta << std::endl;
    oss << "  hc_mult: " << hc_mult << std::endl;
    oss << "  hc_sinkhorn_iters: " << hc_sinkhorn_iters << std::endl;
    oss << "  hc_eps: " << hc_eps << std::endl;
    oss << "  layer_compress_ratios.size: " << layer_compress_ratios.size() << std::endl;
    oss << rope_config.DebugRopeConfigStr();
    return oss.str();
}

}  // namespace rtp_llm
