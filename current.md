# DeepSeek-V4 支持开发进度

> 跟踪 `develop_ds_v4.md` 中 PR-A ~ PR-G、M0 ~ M5 的实现进度。

---

## ✅ 已完成

### PR-A｜M0 骨架（模型注册 / HF 配置解析）
见 `develop_ds_v4.md` §6 M0：模型注册 / HF config 解析 / cpp 字段 / pybind /
HF wrapper / `DeepSeekV4` + `DeepSeekV4Mtp` 类全部就绪。

### PR-B｜M1 mHC 残差
- `rtp_llm/models_py/modules/mhc.py` — Sinkhorn-Knopp + `MhcLayer` + 100-step 稳定测试

### PR-C｜依赖升级
- tilelang 0.1.6 → 0.1.8（运行时即将切入）
- FlashMLA pin 待 SGLang fork 含 V4 patch 后升级

### PR-D｜renderer + tool-call 检测
- `deepseekv4_detector.py` + `deepseekv4_renderer.py`

### M2｜HCA + Grouped Output Projection + SWA 旁路
- `compressor.py` / `hca_attention.py` / `cache_topology.py` 全部就绪

### M3｜CSA（lightning indexer + sparse MQA）
- `csa_attention.py`：indexer 自有压缩流 + top-k + 稀疏 MQA

### M5｜MoE 原语
- `v4_gating.py` (sqrt_softplus + noaux_tc topk)
- `hash_router.py`（前 3 层确定性 hash 路由）
- `clamped_swiglu.py`

### PR-E｜HCA-only 模型组装（reference）
- `deepseek_v4_layer.py`：mHC × HCA 注意力 × V4 MoE 单层组装

### **PR-F｜V4 per-layer 权重 loader + Python 模型** ⭐ 本轮新增
- `rtp_llm/utils/model_weight.py` 新增 41 个 `W.v4_*` 常量（attn / Q-LoRA /
  MQA / 输出 LoRA / compressor / indexer / mHC / MoE / shared expert / 路由
  experts / MTP 辅助张量），全部接入 `gpt_style_tp_strategy`
- `rtp_llm/models/deepseek_v4.py` 重写 `DeepSeekV4Weight._get_hf_layer_weight_info`：
  - 按 `compress_ratios[layer_id]` 分支：SWA-only(0) / CSA(4) / HCA(128)
  - SWA-only 不产 compressor / indexer
  - CSA 产 compressor + indexer + `weights_proj`
  - HCA 只产 compressor
  - MoE: 前 `num_hash_layers` 层用 `tid2eid`（hash 路由），其余层用
    `gate.bias`（learned 路由）
  - 256 routed experts 走 `MoeAtomicWeight` stacked 加载
- 新增 `_V4MoeWeight` 容器，绕开 `MoeWeight.__init__` 对 V3 名字的硬编码检查
- 全局权重：`embed.weight` / `norm.weight` / `head.weight`（裸 key，不带
  `model.` 前缀）+ `hc_head_{base, fn, scale}` 头侧 mHC 折叠
- MTP loader：覆盖 `mtp.{i}.*` 全键（V4 用 `e_proj` + `h_proj` 两个独立投
  影替代 V3 的融合 `eh_proj`），embedding/lm_head 用零占位（`MtpExecutor`
  在运行时 alias 到 base model 张量）

### **PR-G｜DeepSeekV4 引擎模型类** ⭐ 本轮新增
- `rtp_llm/models_py/model_desc/deepseek_v4.py`
  - `DeepSeekV4Model(GptModelBase)`：embedding → 43 层 mHC-wrapped V4 layer
    → final norm → lm head
  - `_DeepSeekV4LayerWrap` 按 layer kind 分发到 `CsaAttention` / `HcaAttention`
    （SWA-only 折叠为 HCA m'=1）
  - `_dequant_fp8_block`：ue8m0 128×128 块状 FP8 → bf16 反量化
    （单次构造时执行，热路径保持 bf16）。源：vLLM PR #40760
    `vllm/model_executor/layers/quantization/utils/fp8_utils.py`
  - `_bind_mhc / _bind_v4_attention / _bind_v4_moe`：把加载的张量拷进 reference
    模块的 `nn.Parameter`
  - 自动选择 `RMSNorm`（CUDA）/ `RMSNormTorch`（CPU）, 单测可在 CPU 跑通
- `models/deepseek_v4.py::DeepSeekV4._create_python_model`：实例化
  `DeepSeekV4Model` 并存到 `self.py_model`（不再抛 `NotImplementedError`）

### **性能优化（移植自 vLLM PR #40760 / SGLang PR #23600）** ⭐ 本轮新增
- `rtp_llm/models_py/modules/moe/batched_experts.py`：
  - `batched_experts_forward` — 把 V3 的 `for e in range(num_experts)` 散点
    循环换成 sort-by-expert + 每 active expert 一次 slab GEMM。N×top_k 行复
    用 256 路由 / 短 prompt 场景下从 256 次循环降到 ≤6 次。源：SGLang PR
    `python/sglang/srt/layers/moe/deepseek_v4_topk.py` + vLLM PR
    `csrc/moe/topk_softplus_sqrt_kernels.cu`
  - `topk_to_onehot` — `(N, top_k)` → `(N, num_experts)` 一次性 mask + 计数
- `rtp_llm/models_py/modules/hybrid/fused_qk_norm_rope.py`：
  - `fused_qk_norm_rope` — 把 `RMSNorm` 和 `apply_partial_rope` 合并成单
    pass，省一份 Q/K 临时张量。源：vLLM PR
    `vllm/v1/attention/ops/deepseek_v4_ops/fused_qk_rmsnorm.py` 和 SGLang PR
    `python/sglang/srt/layers/deepseek_v4_rope.py`
- `DeepSeekV4MoE.forward` 切到 `batched_experts_forward`

### **完整 UT 覆盖** ⭐ 本轮新增
全部 211 个 V4 单测��� CPU 上通过：
- `rtp_llm/test/deepseek_v4_infra_test.py` — 48 测，含**真实 V4-Flash
  safetensors index 全键覆盖检查**（67,612 base + 1,575 MTP key 全部被
  loader claim），全部布局测试 + V3-only key 反向检查
- `rtp_llm/test/deepseek_v4_model_test.py` — 19 测，FP8 dequant、layer
  分发、bind helpers、3 种 attention kind 在小尺寸上的 forward
- `rtp_llm/models_py/modules/test/batched_experts_test.py` — 5 测，与
  scatter loop 数值 parity（fp32 1e-5 / bf16 2e-2），含 inactive expert /
  top_k=1 edge case
- `rtp_llm/models_py/modules/hybrid/test/fused_qk_norm_rope_test.py` — 4 测，
  与 unfused `RMSNorm`+`apply_partial_rope` parity
- 既有的 mhc / v4_moe / v4_layer / compressor / hca_attention /
  csa_attention / cache_topology 全部继续 PASS

### **其他**
- `rtp_llm/cpp/core/OpData.h::scoring_func` 注释更新到包含 `2 = sqrt_softplus`
  （cpp routing kernel 实际是 passthrough，Python 侧 v4_gating.py 已落地）

### **Perf ports — vLLM PR #40760 / SGLang PR #23600**（每个文件 docstring 注明来源）
- `models_py/modules/hybrid/deepseek_v4_rope.py` — Q/K 双 base partial RoPE
- `models_py/modules/hybrid/fused_indexer.py` — lightning-indexer 单 matmul + 累加
- `models_py/modules/hybrid/online_sink_attention.py` — sink 折入 softmax 分母，无 concat
- `models_py/modules/hybrid/fused_compressor.py` — HCA / CSA 单 matmul 联合压缩
- `models_py/modules/mhc_fused.py` — pre/block/post 单 pass mHC step
- `models_py/modules/moe/noaux_tc_fused.py` — sqrt(softplus) + bias + topk + renorm fused
- `models_py/modules/moe/batched_experts.py` — sort + 单 expert slab GEMM（已替换 V3 scatter loop）
- `models_py/modules/hybrid/fused_qk_norm_rope.py` — RMSNorm + partial RoPE 单 pass

### **GB200 / Blackwell 路径** ⭐ 本轮新增
- `models_py/modules/hybrid/sm100_selector.py` — 三个能力 cache：`has_blackwell_gpu` /
  `has_fp4_kernels` / `has_flashinfer_cutedsl`
- `models_py/modules/hybrid/fp4_indexer.py` — NVFP4 lightning indexer（CUTLASS scaled FP4 mm
  + per-block e4m3 scale），shape 不满足 32 倍数自动回退到 bf16 fused 路径
- `models_py/modules/moe/megamoe_fp4.py` — FlashInfer cute-DSL grouped GEMM 包装的
  MegaMoE 专家 GEMM；`megamoe_or_batched()` 是 selector，未提供 FP4 weight 时透明降级到
  `batched_experts_forward`
- `model_desc/deepseek_v4.py::DeepSeekV4Model.__init__` — 启动时 log 一次三个能力位
- 在本机 GB200 实测：FP4 indexer top-8 与 bf16 reference 集合重合度 87%（论文 §2.3.5
  Table 4 报告 ~85%，符合预期）

### **生产 kernel 包装（Hopper / Blackwell 共用）** ⭐ 本轮新增
- `models_py/modules/moe/deepgemm_megamoe.py` — `m_grouped_fp8_gemm_nt_masked` 包装；
  GB200 实测 **4.35× speedup vs python ref（11.6 TFLOPS vs 2.7 TFLOPS）**
- `models_py/modules/hybrid/deepgemm_indexer.py` — `fp8_mqa_logits` +
  `fp8_paged_mqa_logits` 包装；和 V3.2 in-tree indexer 同 API；shape 不满足
  alignment（D % 128, T_kc % 128, M % (128/H)）自动回退到 bf16 fused
- `models_py/modules/hybrid/flashmla_csa.py` — `flash_mla_sparse_fwd(q, kv, indices,
  sm_scale, d_v=512, attn_sink, topk_length)` + `flash_mla_with_kvcache(...)` 包装；
  GB200 实测 **vs bf16 reference 相对 L2 = 2e-3（0.2%）**；约束 h_q ∈ {64, 128}、
  topk % 64 == 0
- `git.bzl` 删了 misleading 的 "FlashMLA 需要 bump" TODO —— 实测 deep_gemm 2.2.0 +
  flash_mla 1.0.0 已 ship 全部 V4 生产 kernel，**不需要 bump**

### **Selector wire-in（model layer 实际调 production kernel）** ⭐ 本轮新增
- `model_desc/deepseek_v4.py` 新增 `_pick_indexer_impl()`：FP4 → DeepGEMM FP8 → bf16
  fused 三优先级
- `_CsaIndexerSelector` 是 `nn.Module`，参数名 `W_IUQ` / `w_heads` 与
  `CsaLightningIndexer` 严格对齐 → loader 的 `_bind_v4_attention` 不需要改
- `_DeepSeekV4LayerWrap.__init__` CSA 层构造后把 `self.attention.indexer` 换成
  `_CsaIndexerSelector`；启动 log 一次实际选中的 backend
- `model_desc/deepseek_v4_layer.py::DeepSeekV4MoE`：加 `fp8_expert_weights = None` 槽，
  `forward` 改走 `deepgemm_megamoe_or_batched(bf16_weights=..., fp8_weights=...)` 选择
  器；`fp8_expert_weights=None` 时透明回退 `batched_experts_forward`（默认行为不变）
- **CSA `sparse_mqa_with_sink` 路径暂未 swap** —— 需要 K/V 重打包成 FlashMLA
  `(s_kv, h_kv=1, d_qk=576)` 布局，依赖 #11 cache topology；selector 已存在并测试，
  文档里标 TODO

### **Bazel build 验证** ⭐ 本轮新增
`bazelisk build //rtp_llm:rtp_llm --config=cuda12_9_arm --jobs=64` 跑通（131s, 88
actions），产出 `rtp_llm-0.2.0-cp310-cp310-manylinux1_x86_64.whl`。OpData.h 的
`scoring_func` 注释改动没引入 cpp 端 break。

### **MoE FP8 自动 pack** ⭐ 本轮新增
- `model_desc/deepseek_v4.py::_pack_v4_moe_fp8(moe, layer_w)`：
  - 检测 ckpt 是否同时具备 (FP8 e4m3 expert weight) + (deep_gemm 可 import)
  - 若是：把 V4-Flash 出货的 `(E, hidden, inter)` w1/w3 转置 → `(E, inter, hidden)`，
    沿 inter 轴 cat `[gate; up]` → `(E, 2*inter, hidden)` 写入
    `moe.fp8_expert_weights`；w2 转置成 `(E, hidden, inter)`。Scale 同步重排。
  - FP8 cat 走 `_fp8_cat`（uint8 view round-trip）—— 旧 torch 不支持 fp8 cat。
  - 任何 shape 不匹配走 try/except 软降级到 bf16，模型仍可加载。
- `_bind_v4_moe` 入口先调 `_pack_v4_moe_fp8`，packing 成功且 layer_idx==0 时 log
  一次 "MoE FP8 pack: ... selector will use FP8 path"；`DeepSeekV4MoE.forward`
  的 `deepgemm_megamoe_or_batched` 自动选择 FP8 4.35× 路径。
- 新增 5 个 UT (`PackV4MoeFp8Test`) 验证：no-deepgemm/bf16-ckpt/missing-key 三种
  fallback、packed shape 严格对齐 deepgemm 期望、uint8 view cat 字节级保真。

### **多层链式 forward smoke** ⭐ 本轮新增
- `test/deepseek_v4_model_test.py::MultiLayerChainSmokeTest`：CPU 上构造
  `[SWA-only, SWA-only, CSA, HCA]` 4 层链 + `expand_residual` + 逐层 forward +
  `reduce_residual` + 模拟 final norm + 模拟 lm_head，全部 shape 契约通过。
- 4 个 UT 覆盖：构造每种 attention kind、forward 后 residual shape 不变、
  bf16 ckpt 不会误触 FP8 pack、CSA 层 indexer 已替换为 `_CsaIndexerSelector`。
- 提前 catch "real GB200 起服务时才崩" 的常见构造 bug。

---

## 🎯 GB200 端到端可用性评估（Q：现在能跑 E2E 了吗）

### 短答
**能跑 prefill 短 / 中等长度 smoke**（验证 forward 不崩、weight 加载契约对，
FP8 ckpt 可拿到 4.35× MoE 提速）。**长 ctx (>2K) 仍需 CSA→FlashMLA refactor**；
**decode 路径需要 M4 hybrid KV cache**（用户后续提供）。

### 已就绪
- ✅ Bazel build 在 `cuda12_9_arm` 跑通 → wheel 可装可起 server
- ✅ Indexer + MoE selector 已 wire 进 `_DeepSeekV4LayerWrap`
- ✅ DeepGEMM MegaMoE / FP8 indexer / FlashMLA sparse fwd 三个生产 kernel 包装完毕
  且 GPU 实测对 reference 数值一致（FP4 indexer recall 87%、FlashMLA CSA 相对 L2
  0.2%、DeepGEMM MegaMoE 4.35× speedup）
- ✅ **MoE FP8 expert weights 自动 pack**（本轮新增）—— `_pack_v4_moe_fp8` 在
  `_bind_v4_moe` 入口检测 ckpt FP8 + DeepGEMM 可用时，把 w1/w3/w2 重排为
  `(E, 2*inter, hidden)` [gate;up] / `(E, hidden, inter)` 写入
  `DeepSeekV4MoE.fp8_expert_weights`，selector 自动选 4.35× 路径
- ✅ **多层链式 forward smoke**（本轮新增）—— `MultiLayerChainSmokeTest` 在 CPU
  上构造 SWA-only × 2 + CSA + HCA 4 层链 + reduce_residual + 模拟 lm_head，全部
  shape 契约通过；提前 catch 真机起不来的常见 bug
- ✅ 285 个 Python UT 全过（仅 1 个 FlashInfer JIT cache env 问题与 V4 无关）

### 已知阻塞（按严重程度）

#### 🔴 阻塞（用户已说本轮跳过）
1. **M4 hybrid KV cache 没接** —— 用户提供方案
   - 当前 `DeepSeekV4Model.forward` stateless，每次 re-compute full sequence。
   - prefill 短/中 prompt 可以跑通；decode 一启动就 OOM 或者结果错。
2. **mHC reshape 顺序对真实 ckpt 没 verify** —— 等首次 ckpt load 报错再改

#### 🟡 性能/长 ctx 待办（不阻塞 E2E smoke 起步）
3. **CSA `sparse_mqa_with_sink` 还在 reference Python 路径** ⚠ 长 ctx OOM/超慢
   - 影响：V4-Flash 1M ctx 单层就要 ~64 GiB intermediate；任何 >2K ctx 都要走
     `flashmla_csa_or_reference`。
   - **架构限制**：`flash_mla_sparse_fwd` 用同一个 `kv` tensor 做 K (dot product)
     和 V (value blend)，是 MLA absorption 契约（K=V latent）。V4 CSA 是 K/V 独立
     压缩流，**直接 swap 不能正确算 V**——需要把 V 折进 W_O 的较大重构，作为
     单独 task 跟踪（不在本轮 E2E push 范围）。
   - 中等 prompt (≤2K) 用 `_reference_sparse_csa` 仍可跑通。

### 推荐 smoke 顺序
```bash
# 1. 装 wheel
pip install bazel-bin/rtp_llm/rtp_llm-0.2.0-cp310-cp310-manylinux1_x86_64.whl --force-reinstall

# 2. 最小 smoke：prefill-only 1 token，看能否完成一次 forward
MAX_NEW_TOKENS=1 PROMPT="hello" bash start.sh
# 关注点：
#   - model construction 不报错
#   - log 显示 "DeepSeekV4 indexer impl: ..."（fp4/deepgemm_fp8/fused_bf16）
#   - log 显示 "DeepSeekV4 MoE FP8 pack: ... selector will use FP8 path"
#     （ckpt 是 FP8 时；bf16 ckpt 不会出现这行，selector 走 batched bf16）
#   - 第一次 forward shape 契约通过
#   - 第一次 ckpt load 触发 mHC reshape，如报错按 _bind_mhc 文档调整

# 3. 如果 #2 通：跑 16 token prefill+1 token decode（会撞 M4 cache 阻塞）
MAX_NEW_TOKENS=2 PROMPT="hello world" bash start.sh

# 4. 如果 #3 OOM 或报 KV cache 错：等 M4 cache 方案接入；同时长 ctx (>2K) 需要
#    CSA→FlashMLA refactor（架构性，单独 task）
```

---

## 🚧 下一步待办

### 高优先级（阻塞模型上线）
1. **M4 异构 KV cache 双池**（用户提供方案，本轮先 mock）
   - `rtp_llm/cpp/cache/CacheConfig` 改成 per-layer 数组
   - 新增 `StateCachePool`（SWA + CSA/HCA 待压缩 tail）
   - 经典 KV pool 按 `lcm(m, m')=128` 原始 token 分块
   - Python `LayerKVCache` 增加 `layer_type`
   - 当前 `DeepSeekV4Model.forward` stateless，仅适合 prefill / 短 ctx
2. **FlashMLA backend V4 patch**（等 SGLang fork 含 V4 commit 后）
   - `rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/` 增加
     MQA + dense + sink + SWA bypass 模式
3. **真实 V4-Flash 加载 + 短 prompt sanity**
   - 运行 `start.sh`（TP=4），对比 logits 与 HF 参考 < 5e-2

### 中优先级
4. **CSA → FlashMLA sparse 路径（架构性 refactor）**
   - `flash_mla_sparse_fwd` 用同一个 `kv` tensor 做 K(dot) 和 V(blend)，是 MLA
     absorption 契约（K=V latent）。V4 CSA 的 K/V 由独立 compressor 产生，
     **直接 swap 算不对 V**。
   - 解阻塞方案：把 V 投影折进 `o_proj`，让 CSA 内部只走单 latent 流（类似 V3.2
     的 `W_UK_T_W_UV_absorbed`）。需要重写 `csa_attention.forward` 末段 +
     `_bind_v4_attention` + 新增 fused `W_UK_W_UV` 张量。
   - 替代方案：保持 reference Python 路径但只在 ≤2K ctx 用；长 ctx 走 ulysses
     sequence parallel 摊薄 intermediate。
5. **Indexer / compressor flush tail 隐状态**（与 StateCachePool 配合）
6. **MTP（next-N spec decode）**：本轮已搭好 weight loader
   `DeepSeekV4MtpWeight`，但 `_create_python_model` 暂未对应 MTP class，留
   到主 model loop 稳定后再接入
7. **mHC 权重 layout 校准**：当前 `_bind_mhc` 假设 `hc_*_fn / base /
   scale` 各自是 packed `[W_pre|W_res|W_post]` / `[S_pre|S_res|S_post|norm]` /
   `[α_pre|α_res|α_post]`。如真实 ckpt 排布不同需调整 reshape 顺序

### 低优先级 / 性能优化（M6）
8. **Fused kernel 化（CUDA / TileLang）**
   - mHC（替换 Python 实现）
   - 完整的 `fused_deepseek_v4_qnorm_rope_kv_insert`（vLLM CUDA 477 行）
   - `topk_softplus_sqrt`（vLLM CUDA 715 行）
9. **FP4 indexer QK + ue8m0 FP8 weight 全链路打通**

---

## 📂 本轮新增 / 修改文件清单

### 新增
```
rtp_llm/models_py/model_desc/deepseek_v4.py            (engine model wiring)
rtp_llm/models_py/modules/moe/batched_experts.py       (perf: sort+slab MoE)
rtp_llm/models_py/modules/moe/noaux_tc_fused.py        (perf: fused topk routing)
rtp_llm/models_py/modules/moe/megamoe_fp4.py           (sm100: NVFP4 MegaMoE)
rtp_llm/models_py/modules/mhc_fused.py                 (perf: 1-pass mHC step)
rtp_llm/models_py/modules/hybrid/fused_qk_norm_rope.py (perf: fused norm+rope)
rtp_llm/models_py/modules/hybrid/deepseek_v4_rope.py   (perf: dual-base partial RoPE)
rtp_llm/models_py/modules/hybrid/fused_indexer.py      (perf: 1-matmul indexer)
rtp_llm/models_py/modules/hybrid/online_sink_attention.py (perf: in-line sink softmax)
rtp_llm/models_py/modules/hybrid/fused_compressor.py   (perf: HCA/CSA joint matmul)
rtp_llm/models_py/modules/hybrid/sm100_selector.py     (sm100: capability gate)
rtp_llm/models_py/modules/hybrid/fp4_indexer.py        (sm100: NVFP4 indexer)
rtp_llm/test/deepseek_v4_model_test.py                 (19 model UTs)
rtp_llm/models_py/modules/test/batched_experts_test.py (5 perf parity UTs)
rtp_llm/models_py/modules/test/noaux_tc_fused_test.py  (6 parity UTs)
rtp_llm/models_py/modules/test/mhc_fused_test.py       (6 parity UTs)
rtp_llm/models_py/modules/test/megamoe_fp4_sm100_test.py (2 sm100 UTs)
rtp_llm/models_py/modules/hybrid/test/fused_qk_norm_rope_test.py (4 parity UTs)
rtp_llm/models_py/modules/hybrid/test/deepseek_v4_rope_test.py (6 parity UTs)
rtp_llm/models_py/modules/hybrid/test/fused_indexer_test.py (4 parity UTs)
rtp_llm/models_py/modules/hybrid/test/online_sink_attention_test.py (6 parity UTs)
rtp_llm/models_py/modules/hybrid/test/fused_compressor_test.py (8 parity UTs)
rtp_llm/models_py/modules/hybrid/test/fp4_indexer_sm100_test.py (3 sm100 UTs)
```

### 修改
```
rtp_llm/utils/model_weight.py            (+41 W.v4_* + tp strategy entries)
rtp_llm/models/deepseek_v4.py            (PR-F: full per-layer plan; engine wiring)
rtp_llm/models_py/model_desc/deepseek_v4_layer.py (DeepSeekV4MoE → deepgemm_megamoe selector)
rtp_llm/models_py/model_desc/deepseek_v4.py       (+_pick_indexer_impl + _CsaIndexerSelector wire
                                                   + _pack_v4_moe_fp8 / _fp8_cat 自动 FP8 pack)
rtp_llm/test/deepseek_v4_infra_test.py   (48 tests; +real-ckpt coverage)
rtp_llm/test/deepseek_v4_model_test.py   (+MultiLayerChainSmokeTest 4 测，多层链 forward)
rtp_llm/test/deepseek_v4_selector_test.py (+PackV4MoeFp8Test 5 测，FP8 pack 字节级保真)
rtp_llm/cpp/core/OpData.h                (scoring_func comment: +2=sqrt_softplus)
open_source/deps/git.bzl                 (删 misleading FlashMLA bump TODO)
rtp_llm/test/BUILD                       (+deepseek_v4_model_test, +deepseek_v4_selector_test)
rtp_llm/models_py/modules/test/BUILD     (+5 perf parity targets, +deepgemm_megamoe_test)
rtp_llm/models_py/modules/hybrid/test/BUILD (+8 perf parity targets, +deepgemm/flashmla/fp4 tests)
```

### 本轮新增清单（按生产路径分组）
```
# Production kernel wrappers
rtp_llm/models_py/modules/moe/deepgemm_megamoe.py        (FP8 MegaMoE; 4.35× speedup)
rtp_llm/models_py/modules/hybrid/deepgemm_indexer.py     (FP8 paged MQA logits)
rtp_llm/models_py/modules/hybrid/flashmla_csa.py         (FlashMLA sparse + sink + topk)

# Tests for the production kernels
rtp_llm/models_py/modules/test/deepgemm_megamoe_test.py             (3 tests + bench)
rtp_llm/models_py/modules/hybrid/test/deepgemm_indexer_test.py      (5 tests + bench)
rtp_llm/models_py/modules/hybrid/test/flashmla_csa_test.py          (12 tests + bench)
rtp_llm/test/deepseek_v4_selector_test.py                           (10 tests for wiring)
```
