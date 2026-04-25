# DeepSeek-V4 在 rtp_llm 上的支持计划

> 输入材料
> - 论文：`~/Desktop/DeepSeek_V4.pdf`（已抽取至 `.deps/ds_v4_paper.txt`）
> - 配置：`https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base/blob/main/config.json`
>   （已下载至 `/tmp/ds_v4_pro_config.json`）
> - 参考 PR：vLLM #40760 (`dsv4` 分支), SGLang #23600 (`deepseek_v4` 分支)
>
> 本地分析依赖（统一放在仓库根 `.deps/`，不污染全局环境）：
> - `pypdf`（PDF 文本抽取，已 `pip install --target .deps pypdf` 完成）
> - 后续若需要本地跑 tokenizer / 离线 sanity check，再按本计划末尾"依赖与分发"小节追加

---

## 1. 目标与范围

让 rtp_llm 能够正确加载、并以**可上线的推理性能**运行
`DeepseekV4ForCausalLM`（含 `DeepseekV4ForCausalLMNextN` MTP 模块），
覆盖 DeepSeek-V4-Pro 与 DeepSeek-V4-Flash 两档。基础功能优先级：

1. **能跑通**：BF16/FP8 权重加载、单机 TP/EP 单步推理结果与 HuggingFace 参考实现一致（误差 ≤ 1e-2）。
2. **能编排**：CSA + HCA 混合 attention layer 调度、状态缓存（state cache）和经典 KV-cache 双块管理。
3. **能服务**：OpenAI 接口、tool / reasoning parser、chat template、stream 输出。
4. **能优化**：MTP 投机解码、TP/EP/CP 并行、DeepGEMM/MegaMoE 融合 kernel，1M token 上下文可服务。

非目标（本期不做）：训练侧 Muon 优化器、FP4 QAT、anticipatory routing；
DualPipe 1F1B 调度；on-disk SWA KV cache 三策略——这些是训练 / 跨节点 KV 共享议题。

---

## 2. DeepSeek-V4 架构与 V3 系列的差异

> 来源：论文第 2 章 + `DeepSeek-V4-Pro-Base/config.json`

### 2.1 关键 hyper-parameter（DeepSeek-V4-Pro）

```
hidden_size         = 7168       num_hidden_layers      = 61
num_attention_heads = 128        num_key_value_heads    = 1   # 单 KV head, MQA
head_dim            = 512        qk_rope_head_dim       = 64  # 仅末 64 dim 加 RoPE
q_lora_rank         = 1536
o_lora_rank         = 1024       o_groups               = 16  # NEW: 分组输出投影
index_n_heads       = 64         index_head_dim         = 128
index_topk          = 1024
n_routed_experts    = 384        n_shared_experts       = 1
num_experts_per_tok = 6          moe_intermediate_size  = 3072
num_hash_layers     = 3          # NEW: 前 3 层 MoE 走 hash routing
scoring_func        = sqrtsoftplus   # NEW: 替代 V3 的 sigmoid
topk_method         = noaux_tc       # 与 V3 一致
routed_scaling_factor = 2.5
sliding_window      = 128            # NEW: SWA 旁路窗口大小
swiglu_limit        = 10.0           # NEW: SwiGLU clamp
hc_mult             = 4              # NEW: mHC 残差扩展因子 n_hc
hc_sinkhorn_iters   = 20             # NEW: B_l 投影到 doubly-stochastic 的迭代次数
hc_eps              = 1e-6
num_nextn_predict_layers = 1         # MTP 深度，与 V3 一致
max_position_embeddings  = 1048576   # 1M
rope_theta          = 10000
compress_rope_theta = 160000         # NEW: 压缩 KV 分支独立 RoPE base
rope_scaling        = yarn (factor=16, original=65536)
quantization_config = fp8 e4m3, ue8m0 scale, 128x128 block
vocab_size          = 129280
compress_ratios     = [128, 128, 4, 128, 4, ..., 4, 128, 4, 0]   # 长度=62
                      # 每层一个 entry：128=HCA(m'=128), 4=CSA(m=4), 0=非压缩(MTP/SWA-only)
                      # Pro：前 2 层 HCA，之后 CSA/HCA 交错；最后一项 0 对应 MTP 模块
```

DeepSeek-V4-Flash 的对应数值：43 层、d=4096、n_h=64、n_routed_experts=256、
moe_inter=2048、attention top-k=512；前 2 层为纯 SWA；其余 CSA/HCA 交错。

### 2.2 与 V3 / V3.2 的"显式"差异（按子系统）

| 子系统 | V3 / V3.2 | V4 |
|---|---|---|
| Attention 类型 | MLA（V3）；DSA = MLA + lightning indexer + sparse(V3.2） | **CSA + HCA 交错 + SWA 旁路**（参见 §2.3） |
| KV 表达 | (kv_lora_rank=512) + qk_rope_head_dim=64 | **MQA 单 head**（kv_head=1，head_dim=512），CSA/HCA 在序列维做 m / m' 倍压缩 |
| Q 路径 | q_lora_rank=1536（V3 Pro） | q_lora_rank=1536，**末 64 dim RoPE，partial RoPE** |
| 输出投影 | 直接 W_O ∈ R^{n_h·v_head_dim × d} | **Grouped Output Projection**：n_h 头分 g=16 组，每组先 → d_g=1024（即 o_lora_rank），再拼接 → d |
| Q/K Norm | 无 | 进入 core attention 前对每个 Q head、单 KV head 做 **RMSNorm** |
| Attention sink | 无 | 每 head 一个可学习 sink logit Exp(z′_h) 加到 softmax 分母 |
| KV 量化 | V3.2 全 fp8 | **混合**：RoPE 64 维 BF16 + 其余 FP8；indexer QK 路径 **FP4** |
| 残差 | 标准 x + F(x) | **mHC**：残差流扩展为 [n_hc=4, d]，三个映射 A/B/C 动态生成；B 投影到 Birkhoff 多面体 |
| MoE scoring | sigmoid（"sigmoid"） | **sqrt(softplus(·))**（"sqrtsoftplus"） |
| MoE 路由约束 | `n_group / topk_group` 强制路由目标节点 | **取消**节点约束，新增**前 num_hash_layers=3 层 hash routing**（按 token id 的 hash 选 expert） |
| 激活 | SwiGLU | **SwiGLU + clamp**：linear ∈ [−10,10]，gate ≤ 10 |
| MTP | 1 层 nextN | 不变，但架构串名为 `DeepseekV4ForCausalLMNextN` |
| RoPE | yarn 单 base | yarn 双 base：`rope_theta=10000`（主），`compress_rope_theta=160000`（压缩 KV 分支） |

### 2.3 CSA / HCA / SWA 的 layer-level 组合

`compress_ratios[i]` 决定第 i 层走哪条 attention 路径：

- `0`：非压缩（V4-Flash 的前 2 层是纯 SWA；Pro 配置的最后一个 entry 对应 MTP）
- `4` (== m)：**CSA**——KV 序列每 4 token 压成 1 entry，再用 lightning indexer 选 top-1024 entry 做 MQA + SWA 旁路 + sink
- `128` (== m')：**HCA**——KV 序列每 128 token 压成 1 entry，**全量** MQA + SWA 旁路 + sink，无 indexer

CSA 的 lightning indexer 与 V3.2 的 DSA indexer 结构高度相似（low-rank 多头、ReLU、top-k 选块），但 V4 在 indexer 的 Q/K 端使用 FP4，分数从 FP32 降到 BF16。

### 2.4 KV cache 拓扑

V3 的 PagedAttention 假设被打破，论文 §3.6.1 给出 V4 的设计（也是 SGLang/vLLM 实现的核心）：

- **State Cache**：每个 request 一个固定大小 block，存 SWA 最近 n_win 个 KV + CSA/HCA 还未凑齐压缩窗口的"尾巴" tokens 的隐状态。
- **Classical KV Cache**：每个 cache block 覆盖 `lcm(m, m')=128` 个原始 token，对应 `k1=128/4=32` 个 CSA 压缩 entry 和 `k2=128/128=1` 个 HCA 压缩 entry。需要按层维护异构形状。

---

## 3. 参考实现速览

### 3.1 vLLM PR #40760（`dsv4` 分支，HEAD bc34b25e）

新增的核心文件（路径相对 vllm 仓库根）：

- 模型层：
  - `vllm/model_executor/models/deepseek_v4.py` (849 行) ── 顶层 `DeepseekV4ForCausalLM`
  - `vllm/model_executor/models/deepseek_v4_mtp.py` (468 行) ── MTP nextN 模块
  - `vllm/transformers_utils/configs/deepseek_v4.py` ── HF config 类
  - `vllm/renderers/deepseek_v4.py`、`vllm/tokenizers/deepseek_v4.py`、`vllm/tokenizers/deepseek_v4_encoding.py`、`vllm/tool_parsers/deepseekv4_tool_parser.py`
- Attention / 残差：
  - `vllm/model_executor/layers/deepseek_v4_attention.py` (1065 行) ── CSA/HCA 主体
  - `vllm/model_executor/layers/deepseek_compressor.py` (436 行) ── token-level compressor (Eq. 9–12, 22–23)
  - `vllm/model_executor/layers/mhc.py` (436 行) ── Manifold-Constrained Hyper-Connections
  - `vllm/model_executor/layers/sparse_attn_indexer.py` (大改)
  - `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py` (大改) ── 双 base + 部分 RoPE
  - `vllm/v1/attention/backends/mla/sparse_swa.py` (494 行) ── SWA 旁路
  - `vllm/v1/attention/backends/mla/flashmla_sparse.py` (大改)
  - `vllm/v1/attention/backends/mla/indexer.py` (大改)
  - `vllm/v1/attention/backends/mla/compressor_utils.py`
- 融合 kernel（CUDA）：
  - `csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` (477 行)
  - `vllm/v1/attention/ops/deepseek_v4_ops/{cache_utils.py, fused_compress_quant_cache.py, fused_indexer_q.py, fused_inv_rope_fp8_quant.py, fused_qk_rmsnorm.py}`
  - `csrc/moe/topk_softplus_sqrt_kernels.cu` (715 行) ── sqrt(softplus) 路由 + topk
  - 配套 `tests/kernels/test_fused_*` 系列单测
- KV cache 调度：
  - `vllm/v1/kv_cache_interface.py`、`vllm/v1/core/kv_cache_utils.py`、`vllm/v1/core/kv_cache_coordinator.py` 大改，引入异构 cache 协调
- Mooncake / HMA disagg：`vllm/distributed/kv_transfer/.../mooncake_connector.py` 增量；新增 `tests/v1/kv_connector/unit/test_mooncake_connector_hma.py`

vLLM 的实现思路：把 MLA backend 推广为"sparse + heavy-compress + SWA"三种 head 形态，
core attention 仍走 FlashMLA（patched 版）。残差路径插入 `MhcLayer`，每层在前后都做 mixing。

### 3.2 SGLang PR #23600（`deepseek_v4` 分支，HEAD f5d03db8）

新增/大改文件（路径相对 sglang 仓库根）：

- 模型层：
  - `python/sglang/srt/models/deepseek_v4.py` (2086 行) ── 顶层模型
  - `python/sglang/srt/models/deepseek_v4_nextn.py` (248 行) ── MTP
  - `python/sglang/srt/configs/deepseek_v4.py` ── HF config 类
- Attention / 残差：
  - `python/sglang/srt/layers/attention/compressed/{compressor.py, indexer.py, metadata.py, paged_prefill.py}`
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py` (1333 行)
  - `python/sglang/srt/layers/attention/nsa/{quant_k_cache_v4.py, index_buf_accessor_v4.py, tilelang_kernel.py（改）}`
  - `python/sglang/srt/layers/attention/triton_ops/compressed_metadata.py`
  - `python/sglang/srt/layers/deepseek_v4_rope.py`
  - `python/sglang/srt/layers/mhc.py` (643 行)
- MoE / 量化：
  - `python/sglang/srt/layers/moe/deepseek_v4_topk.py` ── sqrt(softplus) topk
  - `python/sglang/srt/layers/quantization/mxfp4_deepseek.py`
  - `python/sglang/srt/layers/linear_bf16_fp32/{selector.py, tuner.py, configs/...}` ── DeepSeek-V4 用的 BF16xFP32 GEMM tuning
- KV cache：
  - `python/sglang/srt/mem_cache/deepseekv4_memory_pool.py` (810 行) ── 经典 KV pool（CSA/HCA）
  - `python/sglang/srt/mem_cache/hisparse_memory_pool.py` (340 行) ── 状态 pool
  - `python/sglang/srt/mem_cache/compress_state.py`
  - `python/sglang/srt/managers/hisparse_coordinator.py` (449 行) ── 调度器侧的混合 cache 协调
- TileLang / DeepGEMM 融合 kernel：`python/sglang/jit_kernel/csrc/deepseek_v4/*.cuh`（>15 个 .cuh：fused_norm_rope, hash_topk, paged_mqa_metadata, rmsnorm, rope, silu_and_mul_masked_post_quant, store, topk_v2, topk_1024, hisparse_transfer, mega_moe_pre_dispatch ...）
- 渲染 / 工具调用：`python/sglang/srt/entrypoints/openai/encoding_dsv4.py` (840 行)、`python/sglang/srt/function_call/deepseekv4_detector.py`
- 部署/外部依赖（PR body 给出）：
  - `tilelang==0.1.8`
  - `flashinfer-jit-cache==0.6.8` (CUDA 12.9)
  - `FlashMLA`（deepseek-ai 仓库 main，含 V4 patch）
  - `DeepGEMM`（sgl-project fork `release` 分支 commit `7f2a70`）

SGLang 的实现思路与 vLLM 类似但更彻底：直接新写一个独立的 `deepseek_v4` model + `deepseek_v4_backend_radix`
后端，把混合 cache 的协调器（`hisparse_coordinator`）放到 scheduler 层而不是 attention backend 层。

---

## 4. rtp_llm 现有相关组件

> 工程上 rtp_llm 已经把 DeepSeek 全家（V2 / V3 / V3.1 / V3.2 / Kimi-K2 / GLM-MoE-DSA）合并到一个 `DeepSeekV2` 类里，并已经吃下了 DSA 的所有概念，这是我们改造的"地基"。

| 关注点 | 已有实现位置 | 备注 |
|---|---|---|
| 模型注册 | `rtp_llm/models/deepseek_v2.py:813-819` 的 `register_model` 行 | 只需追加 `DeepseekV4ForCausalLM` 与 `DeepseekV4ForCausalLMNextN` |
| 顶层模型类 | `rtp_llm/models/deepseek_v2.py` `DeepSeekV2`、`DeepSeekV3Mtp` | V4 改动太大，单独建 `DeepSeekV4` 类，复用 weight loader 大部分逻辑 |
| HF 配置解析 | `DeepSeekV2._from_hf` (`rtp_llm/models/deepseek_v2.py:586-714`) | 需新增 V4 字段解析（hc_mult / o_groups / num_hash_layers / sliding_window / compress_ratios / compress_rope_theta / scoring_func=sqrtsoftplus） |
| Python 计算图 | `rtp_llm/models_py/model_desc/generic_moe.py` 的 `GenericMoeModel` | V4 残差/attention 太特殊，预计要新建 `DeepSeekV4Model`（参考 `qwen3_next.py` 的独立模型方式） |
| MLA attention python | `rtp_llm/models_py/modules/hybrid/mla_attention.py` | 不直接复用，新增 `csa_attention.py` / `hca_attention.py`；indexer 可以从 `hybrid/indexer.py` 派生 |
| Sparse 选择 indexer | `rtp_llm/models_py/modules/hybrid/indexer.py` | DSA indexer，CSA 复用同骨架，多一个 token-level compressor + FP4 路径 |
| FlashMLA sparse op | `rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/flashmla_sparse_impl.py` 与 `..._cp_impl.py` | 直接复用；compressed-MQA 需要新 backend impl 或在 FlashMLA patch 里增 head_dim=512 单 KV head 路径 |
| FlashMLA C++ binding | `rtp_llm/models_py/bindings/cuda/{SparseMlaParams.{h,cc}, kernels/mla_quant_kernel.{h,cu}}` | 增加 V4 形参 |
| DeepGEMM wrapper | `rtp_llm/models_py/kernels/cuda/deepgemm_wrapper.py` | 已就绪，FP8 ue8m0 block GEMM 直接用 |
| DeepEP wrapper | `rtp_llm/models_py/distributed/deepep_wrapper.py` | 已就绪，EP all-to-all dispatch/combine |
| TileLang | `rtp_llm/models_py/modules/hybrid/test/indexer_ref.py`，`open_source/deps/requirements_torch_gpu_cuda12_9.txt:26 (tilelang==0.1.6)` | 当前仅测试用；上线 V4 需要把 tilelang 升到 0.1.8 并把它从 test-only 升级为正式 runtime 依赖 |
| FP8 / FP4 | `rtp_llm/models_py/kernels/cuda/fp8_kernel/fp8_kernel.py`、`rtp_llm/models_py/kernels/cuda/fp4_kernel/flashinfer_cutedsl_moe.py` | 已有 FP8/FP4 基础设施 |
| MoE topk + scoring | `rtp_llm/models_py/triton_kernels/common/moe_gating.py`，`rtp_llm/cpp/config/ModelConfig.h:94 (scoring_func)`，`Weights.h:92 (e_score_correction_bias)` | 当前枚举：0=softmax、1=sigmoid。需要 **2=sqrt_softplus** 新增；hash routing 需新增 kernel |
| MTP 权重 | `DeepSeekV3MtpWeight` (`rtp_llm/models/deepseek_v2.py:721-795`) | 直接派生：换 ckpt key 前缀即可 |
| KV cache 形状 | `rtp_llm/cpp/cache/SingleConfigCreator.cc`、`rtp_llm/cpp/model_utils/AttentionConfig.h:57-60`（`is_sparse / indexer_head_*`）；`BlockPoolConfigHelper.h:129` | 需要扩展为"per-layer 异构"。当前是单一形状，改造成本最大 |
| AttentionConfigs pybind | `rtp_llm/cpp/pybind/ConfigInit.cc:1296-1321` | 增字段：`use_csa / use_hca / compress_ratio / o_groups / o_lora_rank / sliding_window / sink / hc_mult / hc_sinkhorn_iters / compress_rope_theta` |
| Renderer 注册 | `rtp_llm/openai/renderers/__init__.py`、`deepseekv31_renderer.py`、`deepseekv32_renderer.py` | 新增 `deepseekv4_renderer.py`；DeepSeek-V4 的 chat template 与 V3.1/V3.2 接近，可复用 |
| Tool / Reasoning parser | `rtp_llm/openai/renderers/sglang_helpers/function_call/deepseekv32_detector.py` | 新增 `deepseekv4_detector.py`，从 SGLang PR `function_call/deepseekv4_detector.py` 移植（仅 27 行） |
| Bazel 构建 | `WORKSPACE`、`open_source/bazel/arch_select.bzl`、`open_source/deps/git.bzl`、`open_source/deps/requirements_*.txt` | 需要：升 tilelang 到 0.1.8，引入 DeepGEMM `release@7f2a70`，FlashMLA 升级到含 V4 patch 的版本 |

---

## 5. Gap 分析（按"必须新增 / 可复用 / 升级"分类）

### 5.1 必须新增（V4 独有）
1. **mHC 残差层**（CUDA + Python wrapper）
2. **CSA token-level compressor**（softmax_row over 2m, weighted sum of C_a/C_b）
3. **HCA token-level compressor**（不重叠版）
4. **CSA lightning indexer (FP4 QK)**（与 V3.2 indexer 同骨架，新增 FP4 量化路径与 BF16 score）
5. **Compressed MQA core attention**：基于 FlashMLA 的 sparse + dense MQA，head_dim=512, kv_head=1
6. **Sliding-window 旁路 attention**（n_win=128）+ 与压缩 attention 的拼接
7. **Attention sink**（softmax 分母补 Exp(z'_h) 的 kernel patch）
8. **Inverse-RoPE on output**（对 o_{t,i} 做位置 -i 的 RoPE，抵消 KV entry 携带的绝对位置）
9. **Grouped output projection**：n_h=128 头 → g=16 组 → d_g=1024 → concat → 7168
10. **sqrt(softplus) scoring + 取消节点约束的 noaux_tc topk** kernel
11. **Hash routing for first 3 MoE layers**（按 token id 静态 hash → expert id）
12. **SwiGLU clamping**（[-10,10] 线性、≤10 gate）
13. **Heterogeneous KV cache 双池**：State Cache（每 request 固定大小）+ Classical KV Cache（每 block 覆盖 lcm(m,m')=128 token）
14. **MoE / attention 层的 layer-type 调度表**（compress_ratios → CSA / HCA / SWA-only）
15. **DeepseekV4 chat template + tool parser + reasoning parser**

### 5.2 可直接复用
- DeepSeek-V3 的 `DeepSeekV2Weight` 大部分 ckpt 加载逻辑（embedding / shared expert / e_score_correction_bias / lm_head）
- DSA indexer 的 Top-K kernel（`rtp_llm/models_py/triton_kernels/sparse_mla/block_index_to_global.py`）
- DeepGEMM ue8m0 FP8 GEMM���DeepEP all-to-all dispatch/combine
- FlashMLA sparse impl（CSA core attention 的 sparse 部分）
- MTP 调度（`rtp_llm/cpp/normal_engine/speculative/MtpExecutor.cc`）

### 5.3 需要升级 / 改造的现有组件
- `AttentionConfigs` 增加 V4 字段；pybind 同步
- `ModelConfig` 增加 `scoring_func=2`、`hash_routing_layers`、`compress_ratios`、`hc_mult`、`o_groups`、`sliding_window`、`compress_rope_theta`
- KV cache 的 `BlockPoolConfigHelper / SingleConfigCreator / CacheConfig` 改成支持异构 layer：每 layer 维护独立 block 形状与 hit/eviction policy；状态池单独管理
- `tilelang` 从 test 依赖升级为 runtime 依赖（0.1.6 → 0.1.8）
- `FlashMLA` 升级到含 V4 patch 的版本
- `DeepGEMM` 升级到 sgl-project `release@7f2a70`（含 MegaMoE 融合）

---

## 6. 分阶段实现计划

> 每个阶段有"完成条件"与"可验证 demo"，便于并行 / 局部回滚。

### M0｜骨架接通（1–2 天）
- [ ] `rtp_llm/transformers_utils/`（无则新增）放 `deepseek_v4_config.py`：把 HF 的 `DeepseekV4Config` 解析成内部 `ModelConfig`
- [ ] 在 `DeepSeekV2._from_hf` 边上新增 `DeepSeekV4._from_hf`，覆盖：
  - `scoring_func == "sqrtsoftplus"` → `config.scoring_func = 2`
  - `num_hash_layers` → `config.moe_hash_routing_layers`
  - `compress_ratios` → `config.attn_config.layer_compress_ratios: List[int]`
  - `hc_mult / hc_sinkhorn_iters / hc_eps` → `config.attn_config.hc_*`
  - `o_groups / o_lora_rank` → `config.attn_config.o_groups / o_lora_rank`
  - `sliding_window / swiglu_limit / compress_rope_theta`
- [ ] 新建 `rtp_llm/models/deepseek_v4.py`：
  - `class DeepSeekV4(BaseModel)`（先继承 `DeepSeekV2` 的 weight loader，加 V4 weight key 映射）
  - `class DeepSeekV4Mtp(DeepSeekV4)`
  - `register_model("deepseek4", DeepSeekV4, ["DeepseekV4ForCausalLM"])`
  - `register_model("deepseek-v4-mtp", DeepSeekV4Mtp, ["DeepseekV4ForCausalLMNextN"])`
- [ ] `rtp_llm/cpp/config/ModelConfig.{h,cc}` + `pybind/ConfigInit.cc`：新增字段；构建 OK
- ✅ 完成条件：随机权重 + V4 config.json 跑通 `from_pretrained` → `_create_config()` 不报错；引擎能起来但 forward 走 stub

### M1｜mHC 残差（3–4 天）
- [ ] Python：`rtp_llm/models_py/modules/mhc.py`，按论文 Eq. 2–8 实现：
  - 动态参数生成：`RMSNorm(vec(X_l)) → A_l, B_l, C_l` 三个线性
  - 约束：sigmoid → A/C；exp + 20 轮 Sinkhorn-Knopp → B
  - 前向：`X_{l+1} = B X_l + C · F(A X_l)`（pre-mix / post-mix / residual mix 三段）
- [ ] CUDA fused kernel：把 RMSNorm + 3 个小 GEMM + Sinkhorn + 残差更新合并（可分两步走：M1 先 PyTorch 验证正确性，M1.5 再上 fused kernel）
- [ ] Weight loader：mHC 的 W_pre/W_res/W_post 与静态 bias / gating factor 加载到 `DeepSeekV4Weight`
- ✅ 完成条件：在 dummy attention（直接恒等映射 F）下做 100-step 前向，与 HF 参考实现 bit-identical 误差 < 1e-3

### M2｜HCA + Grouped Output Projection + SWA 旁路（5–7 天）
HCA 比 CSA 简单（无 indexer、无 sparse），先打通：
- [ ] `rtp_llm/models_py/modules/hybrid/compressor.py`：CSA / HCA 共用的 token-level compressor（HCA 单分支，CSA 双分支）
- [ ] `rtp_llm/models_py/modules/hybrid/hca_attention.py`：
  - `c_Q_t = h_t · W_DQ` → `q_t = c_Q_t · W_UQ`
  - `C = H · W_KV`，`Z = H · W_Z`，按 m'=128 压缩
  - **Q/K RMSNorm**、partial RoPE（last 64 dim, base=160000）、**inverse RoPE on output (-i)**
  - **Sliding-window attention 旁路**：每 query 额外读 n_win=128 个未压缩 KV
  - **Attention sink**：每 head 一个可学习 logit，加到 softmax 分母
  - **Grouped output projection**：n_h → g=16 组 → o_lora_rank=1024 → concat → hidden_size
- [ ] FlashMLA backend 增加 `MQA + dense + sink + swa-bypass` 模式（先 PyTorch reference，再上 FlashMLA patch）
- [ ] AttentionConfigs / ModelConfig 透传 SWA / sink / o_groups
- ✅ 完成条件：把模型所有 layer 都临时设成 HCA，运行 1k token prompt，token-by-token logit 与 HF 参考差 < 5e-2；KV cache 大小≈完整 KV 的 1/m'

### M3｜CSA（lightning indexer + sparse MQA）（5–7 天）
- [ ] `rtp_llm/models_py/modules/hybrid/csa_attention.py`：
  - 复用 `compressor.py`（双分支版）
  - **Lightning indexer**：从 `models_py/modules/hybrid/indexer.py` 派生 `CsaIndexer`：
    - low-rank Q：`c_Q_t · W_IUQ` → 64 indexer heads × 128 dim
    - K：复用与主分支相同的压缩规则得到 `K_IComp`
    - score：`I_{t,s} = Σ_h w_h · ReLU(q_h · K_s)`，按 m=4 块；FP4 QK / BF16 score
    - top-k=1024
  - **Sparse MQA core attention**：把 V3.2 的 `flashmla_sparse_impl` 调用换成 V4 的（head_dim=512、kv_head=1、sink、swa-bypass）
- [ ] Indexer FP4 路径：复用 `rtp_llm/models_py/kernels/cuda/fp4_kernel`；若 FP4 一期不上，先 BF16 跑通
- ✅ 完成条件：在压缩比 m=4、top-k=1024 配置下，logit diff vs HF 参考 < 5e-2；同长度下 attention FLOPs 与论文 Fig. 1 一致

### M4｜异构 KV cache 双池（4–6 天）
最棘手的工程改造：
- [ ] C++：`rtp_llm/cpp/cache/`
  - `CacheConfig` 改造为 per-layer 数组（CSA / HCA / SWA-only / non-cache 四类）
  - `BlockPoolConfigHelper` 支持每 cache block 覆盖 `lcm(m, m')` 原始 token；CSA 段产生 `k1=32` 个 entry，HCA 段产生 `k2=1` 个 entry
  - 新增 `StateCachePool`：固定大小、按 request 分配，存 SWA 最近 n_win KV + CSA/HCA 待压缩 tail
  - `KVCacheManager` 在 prefix-cache 命中时按"压缩边界对齐"截断 prefix，剩余 tail 走重算路径（论文 §3.6.2）
- [ ] Python `rtp_llm/ops/compute_ops.py` 的 `LayerKVCache` 增加 layer-type 字段
- [ ] Indexer / compressor 在 step 边界把 tail 隐状态 flush 到压缩 entry
- ✅ 完成条件：变长 batch（混合短 / 中 / 长）多 step 推理结果与全长 forward 等价；KV cache 总占用按论文公式（论文 §2.3.4 / Fig. 1）符合预期

### M5｜MoE：sqrt(softplus) + 取消节点约束 + Hash routing + MTP（3–5 天）
- [ ] CPP `OpData.h` 的 `scoring_func` 枚举增加 `2 = sqrt_softplus`；对应 routing kernel 在 `rtp_llm/models_py/triton_kernels/common/moe_gating.py` 与 cpp 路径都实现
- [ ] `noaux_tc` 路由路径取消 `n_group / topk_group` 强制约束（V4 这两个字段不存在；走纯 top-k）
- [ ] **Hash routing**：`rtp_llm/models_py/modules/moe/hash_router.py`，按 token_id 的 deterministic hash → expert id；前 `num_hash_layers=3` 层走它，其余走 sqrt_softplus topk
- [ ] **SwiGLU clamping**：`rtp_llm/models_py/triton_kernels/common/activation.py` 增加 `clamped_swiglu(linear∈[-10,10], gate≤10)`
- [ ] MTP：`DeepSeekV4MtpWeight(DeepSeekV4Weight)`，复制 `DeepSeekV3MtpWeight` 的 enorm/hnorm/eh_proj/shared_head 路径；`MtpExecutor` 已有，验证调度 OK
- ✅ 完成条件：单 expert / 单 layer dump 的输出与 HF 参考 bit-equal；MTP nextN 模块独立加载并能给 spec-decode 提速

### M6｜性能优化（持续）
- [ ] `tilelang` 升级到 0.1.8 并接入 runtime；对 mHC、compressor、indexer、QK-RMSNorm-RoPE-FP8-cast 链做 fused kernel
- [ ] 引入 SGLang fork 的 DeepGEMM `release@7f2a70`（含 MegaMoE 融合）替换当前 deep_gemm；EP wave 调度（论文 §3.1）
- [ ] FlashMLA 升级到含 V4 patch 的版本
- [ ] FP4 indexer QK + ue8m0 FP8 weight 全链路打通；测 1M ctx 长上下文吞吐
- [ ] Inverse-RoPE 与 sink 写到 fused attention output kernel 里

### M7｜服务化（2–3 天）
- [ ] `rtp_llm/openai/renderers/deepseekv4_renderer.py`：从 `deepseekv32_renderer.py` 复制改 chat template
- [ ] `rtp_llm/openai/renderers/sglang_helpers/function_call/deepseekv4_detector.py`：移植 SGLang PR 的 `function_call/deepseekv4_detector.py`（27 行）
- [ ] `register_renderer("deepseek_v4", DeepseekV4Renderer)`，并在 `__init__.py` import
- [ ] reasoning parser：复用现有 sglang_helpers/reasoning_parser
- ✅ 完成条件：`/v1/chat/completions` 流式 + tool_calls + thinking 段 与官方 demo 输出一致

---

## 7. 测试策略

| 层级 | 测试 | 位置 |
|---|---|---|
| Unit | mHC 前向 vs HF 参考 | `rtp_llm/models_py/modules/test/mhc_test.py` |
| Unit | CSA / HCA compressor 数值正确性 | `rtp_llm/models_py/modules/hybrid/test/compressor_test.py` |
| Unit | Lightning indexer top-k 召回率 ≥ 99% vs 暴力 | `rtp_llm/models_py/modules/hybrid/test/csa_indexer_test.py` |
| Unit | sqrt_softplus + noaux_tc topk vs PyTorch | `rtp_llm/models_py/triton_kernels/common/test/test_moe_gating_v4.py` |
| Unit | Hash routing 决定性 + 期望分布 | 同上 |
| Integration | 单层（HCA only）→ HF 一致性 | `rtp_llm/test/model_test/...` |
| Integration | 全 model dummy weight 前向不崩 | 已有框架 |
| Integration | 真 ckpt（DeepSeek-V4-Pro-Base + Flash）短 prompt 与 HF transformers 实现 logit 对齐 | 新加 |
| E2E | OpenAI chat / tool / streaming | `rtp_llm/test/openai_response_test.py` 增 `DeepseekV4TestSuite` |
| Perf | 1M ctx FLOPs / KV 大小符合论文 | benchmark 新增 |

---

## 8. 风险与未决问题

1. **异构 KV cache 是改动半径最大的工程项**——rtp_llm 当前 `BlockPoolConfigHelper / SingleConfigCreator` 假设单一形状，per-layer block shape 改动会波及 prefix cache、disaggregate decode、Mooncake KV connector 等多处。建议 M4 单独立 PR，前置 review。
2. **TileLang 作为 runtime 依赖**：当前仓库只在 test 中用到，要把 tilelang 0.1.8 装进生产 image，需要确认 tilelang 与 rtp_llm cuda 版本（cu126/cu129）的 wheel 可用性，否则要本地构建放到 `open_source/deps/`。
3. **FlashMLA 版本对齐**：上游 FlashMLA 在 V4 PR 之后还在快速迭代，需要锁版本。当前 `open_source/deps/git.bzl:167` 引的是 `https://github.com/deepseek-ai/FlashMLA.git`，要么 patch 一份内部 fork，要么定期 rebase。
4. **DeepGEMM fork 选择**：vLLM 用 `vllm/utils/deep_gemm.py` 改了一份；SGLang 用 `sgl-project/DeepGEMM:release@7f2a70`。对齐之后新增的 MegaMoE 路径只在某些 sm90/sm100 架构上 build，**arm/rocm 镜像需要单独验证**。
5. **mHC 的 Sinkhorn-Knopp**：论文给 t_max=20 的迭代，推理可能可降到 10 甚至离线 fold 进权重；先按论文实现，performance pass 时再剪。
6. **Inverse RoPE on attention output** 这个 trick 容易漏：忘了之后会出现"长 ctx 准确率随距离单调下降但短 ctx 正常"的症状，单测里需专门覆盖。
7. **Pro vs Flash 的 layer 类型表不同**（Pro 前 2 层 HCA、Flash 前 2 层纯 SWA），需要从 `compress_ratios` 数组逐层读取，不要硬编码。
8. **vocab_size = 129280** 与 V3 的 128K 不同，渲染器与 special token 表注意同步。
9. **hash routing 的 hash 函数** 论文未给出具体形式，需对照 HuggingFace 仓库下的 `modeling_deepseek_v4.py`（在 `https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference`）取与官方一致的实现。
10. **vLLM PR 与 SGLang PR 都是 open**，最终接口可能微调，发版前需要 cross-check 一次最新 commit。

---

## 9. 依赖与本地化分发

### 9.1 本地分析依赖（已就位）
- `pypdf` → `.deps/pypdf/`：用 `python3 -m pip install --target .deps pypdf` 已装好；后续读 PDF 用 `PYTHONPATH=.deps python3 ...`

### 9.2 后续按需追加到 `.deps/` 的本地 lib（不改全局环境）
```bash
# 离线读 HF tokenizer / config 类（移植 vLLM/SGLang 配置时要参考）
PIP_TARGET=.deps python3 -m pip install transformers==4.57.1
# 算子 ground truth 验证（与 SGLang 对齐时可能用）
PIP_TARGET=.deps python3 -m pip install tilelang==0.1.8
# DeepSeek 官方 inference 实现（HF 上的 inference/ 目录），手工 git clone 到 .deps/ds_v4_official_inference/ 取参
```

### 9.3 生产构建依赖变更（在 `open_source/deps/` 内做）
- `requirements_torch_gpu_cuda12_9.txt`：`tilelang==0.1.6` → `0.1.8`；新增 `flashinfer-jit-cache==0.6.8`
- `git.bzl`：FlashMLA pin 到含 V4 patch 的 commit；新增 sgl-project `DeepGEMM:release@7f2a70` 或对内部 fork rebase
- `bazel/arch_select.bzl`：MegaMoE / DeepGEMM 新 target 暴露给 cpp build

### 9.4 Docker
- 参考 SGLang PR 给的 image 名（`lmsysorg/sglang:deepseek-v4-{blackwell,hopper,grace-blackwell}`），在 `open_source/nvidia_docker/` 增一个 `Dockerfile.deepseek_v4`，明确 cuda12.9 + tilelang 0.1.8 + FlashMLA(v4) + DeepGEMM(7f2a70)

---

## 10. 工时估计（人天，单人）

| 阶段 | 估时 |
|---|---|
| M0 骨架接通 | 1–2 |
| M1 mHC | 3–4 |
| M2 HCA + GroupedOProj + SWA | 5–7 |
| M3 CSA + Lightning indexer | 5–7 |
| M4 异构 KV cache | 4–6 |
| M5 MoE + MTP | 3–5 |
| M6 perf 优化 | 7–10（持续） |
| M7 服务化 | 2–3 |
| **合计** | **约 30–45 人天**，建议 attention/cache/MoE 三条线并行做到 25 天内可灰度 |

---

## 11. 第一周可立即开工的 PR 切分建议

1. PR-A（infra）：`AttentionConfigs / ModelConfig` 加字段 + pybind + DeepseekV4 注册 + `_from_hf` 解析 → 可单独合入
2. PR-B（mHC）：`models_py/modules/mhc.py` + 单测 → 可单独合入
3. PR-C（dep）：tilelang 升 0.1.8 + FlashMLA pin + DeepGEMM fork wiring（dockerfile + bazel） → 可单独合入
4. PR-D（renderer）：`deepseekv4_renderer.py` + `deepseekv4_detector.py` → 可单独合入
5. PR-E（model）：`deepseek_v4.py` + HCA-only 路径（先把所有 layer 当作 HCA 跑） → 依赖 A、B、C
6. 之后才进入异构 KV cache / CSA / MoE / MTP。
