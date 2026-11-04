# 分布式 Engram 验证工具

生产代码统一位于 [`engram_distributed`](../../rtp_llm/models_py/modules/dsv41/engram_distributed)：
启动与交换、存储映射、模型挂接分别由 `startup`、`storage`/`mapping`、`model_attach` 负责。
包内 `native/` 保存 CUDA Driver/ATen bridge，`test/` 保存协议、CPU 与真实 Fabric 测试。
本目录只保留测试入口、环境采集和跨 host 合成探针；没有独立 sidecar 进程。

配置与生命周期约束见 [配置契约](../../rtp_llm/models_py/modules/dsv41/engram_distributed/CONFIG.md)。

```bash
# 标准库；不会初始化 CUDA。
bash tools/engram_distributed/test.sh --cpu
# Linux Torch/safetensors、真实 TCPStore、模型挂接和 embedding 合约。
bash tools/engram_distributed/test.sh --torch
# 显式启用两张空闲 GPU；兼容文件锁，忙设备/缺能力/skip 均不算通过。
ENGRAM_RUN_CUDA_TESTS=1 ENGRAM_TEST_NUMA=0 CUDA_VISIBLE_DEVICES=0,1 \
bash tools/engram_distributed/test.sh --cuda-fabric
```

Fabric 测试要求 HOST_NUMA/FABRIC/IMEX、可用的 Torch/Triton 和预编译 native bridge，
详见 [native 说明](../../rtp_llm/models_py/modules/dsv41/engram_distributed/native/README.md)。
同一 GPU 锁范围内运行全部 native 边界/回滚测试和 lifecycle/lookup 集成测试。
lookup 同时覆盖 host/GPU Fabric、跨卡访问、Graph replay、scale 缓存关闭与别名回收。
设置 `ENGRAM_TEST_GPU_COUNT=4 CUDA_VISIBLE_DEVICES=0,1,2,3` 可覆盖每 host 两段 GPU backing，
包括 weight/scale 的物理段边界；未设置时使用两张卡。
设备可传完整 UUID；数字按 `nvidia-smi` 物理编号解释，实际传给 CUDA 的仍是对应完整 UUID。
脚本不清理其他 GPU 进程，不重启服务。跨 host 全模型测试需使用独立 P/D 开发环境。

重写前的实验资料保留在
[归档分支](https://github.com/MMadhatter/rtp-llm/tree/codex/engram-archive-20261104/tools/engram_sidecar)。
历史结果不能替代当前版本的验收。
