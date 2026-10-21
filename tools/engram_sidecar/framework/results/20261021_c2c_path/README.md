# 跨机 EGM/C2C 读取路径诊断（path-v2）

2026-10-21。**只验证路径与内容，不做性能验收。** 原始证据（含机器标识、容器/进程信息）保留在私有记录；本目录只发布去标识汇总。

## 结论

consumer GPU 通过既有 `HOST_NUMA` + `FABRIC` 映射，直接读取 owner 进程持有的两张完整 Engram 表；N=4 与 2N=8 两轮均完成并通过 BF16 oracle。owner 侧对应 C2C 端口记录到两段与逻辑读取量成比例的流量，2N/N = 2.0000004。

| 轮次 | launch | 逻辑读取 B | 输出 B | owner C2C 计数 B | 计数/逻辑 |
|---|---:|---:|---:|---:|---:|
| N | 252 | 2,179,457,280 | 4,226,826,240 | 2,377,687,040 | 1.0909537 |
| 2N | 504 | 4,358,914,560 | 8,453,652,480 | 4,755,375,104 | 1.0909537 |

264 B 行按 32 B sector 计为 `ceil(264/32)*32 = 288 B`，`288/264 = 12/11 = 1.0909091`，与实测比值相差约 0.004%。这可解释为行级 sector 计数，但本次没有独立微架构实验证明该机制。

诊断 `exact_bf16=true`，native counters（mapping/reservation/view/context/import/owner allocation/cleanup）全部为 0，`CLOSED` 6.596 s。

## 方法

- owner 与 consumer 分别启动 55 s、0.5 s interval 的 C2C PMU reader；四个事件 FD 在诊断前已 `OPENED`。
- consumer 在独立 GPU 上以直接 `GNU timeout --signal=TERM --kill-after=3s 120s` 父进程运行诊断，12 s 软窗口。
- 诊断写 `READY` 后等待外部 `GO`，确认 reader 已启动才进入 N/2N；结束后写 `CLOSED`。
- 两侧各自保留本地单调时钟，不做跨主机时钟相减；相位归属由 reader 本地时间与 N/2N 的 1 s 间隔共同定位。

## 边界

- 系统级 socket PMU，不是 per-process；reader 自述 `physical-link scope unverified`。
- 单次运行；reader 55 s；N 轮包含冷启动（2N wall 更短），**不能据此计算带宽或延迟**。
- PMU 不计 C2C latency；没有 per-access 延迟分解。
- consumer 本地 C2C 在 55 s 内没有 GB 级 bulk 流量，这只是“读取不在 consumer 本地 C2C 汇聚”的旁证，不证明每一字节的物理链路。
- 只覆盖一个 owner、一个 consumer GPU；多 consumer、多 owner、整 rack 未测。
- 历史 Graph 正向长尾（如 207.424 μs）仍然独立存在，本诊断不构成修复。

## 复现与证据

概要数据见 [path.json](path.json)。私有原始 JSONL 与归档 SHA256 记录在内部任务状态与 freeze 记录中，不随公开仓库分发。诊断脚本、CPU 单测与 freeze 记录保留在私有 operator 目录；公开仓库只发布上述去标识结果。

## 与位置矩阵的关系

本次可把 **N0（owner HOST_NUMA0 → consumer GPU0）功能格**标记为已实测通过。N1/N2/F2 与 L0/L1/C0/C1/C2 仍未执行；单 owner 是否足够、是否需要分片或副本，仍取决于后续并发与正式模型尾部。