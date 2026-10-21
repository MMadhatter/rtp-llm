# N1 / F2 去标识结果（2026-10-21）

两份独立诊断，均为**功能路径**证据，不是延迟或带宽验收。

## N1：owner 远端 backing → consumer GPU2

consumer 使用另一 Grace 组的 GPU2，经既有 `HOST_NUMA` + `FABRIC` 映射读 owner 两张完整表；consumer 不分配 202 GB 本地表。两次运行：

| run | N logical B | 2N logical B | owner c2c0_pmu_0 N | owner c2c0_pmu_0 2N | N/logical | 2N/logical |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 2,179,457,280 | 4,358,914,560 | 2,377,687,552 | 4,755,374,592 | 1.09095396 | 1.09095384 |
| v2 | 2,179,457,280 | 4,358,914,560 | 2,377,687,040 | 4,755,377,664 | 1.09095373 | 1.09095455 |

两轮 `exact_bf16=true`、native counters 全 0、CLOSED 6.391 s / 6.923 s；2N/N = 1.99999978 / 2.00000151。v2 consumer 本地 C2C：`c2c0_pmu_0=0`，其余三个 PMU 10.7/23.4/10.7 MB，无 GB 级本地 bulk。

## F2：GPU0 + GPU2 并发读同一 owner backing

两路同时 READY、同时 GO，均 exit 0、`exact_bf16=true`；CLOSED 7.407 s / 7.511 s。

| consumer | N logical B | 2N logical B |
|---|---:|---:|
| GPU0 | 2,179,457,280 | 4,358,914,560 |
| GPU2 | 2,179,457,280 | 4,358,914,560 |

两路逻辑合计 13,076,743,680 B；owner `c2c0_pmu_0` 窗口 14,307,650,560 B，比值 1.09413（单路约 1.09095，高 0.29%）。owner 增量分段 4.755/6.473/3.038 GB，符合两路窗口部分重叠与采样边界，未归给单路。

## 边界

- owner PMU 是 system-wide socket PMU，55 s 窗口，不是 per-process；N1 v2 与 F2 的另一 C2C PMU 背景计数不归因。
- 只证明内容一致、计数口径和并发完成；不证明带宽、延迟、owner 热点上限或逐 step ≤100 μs。
- N2、EP、C1、F4 仍未执行；g2 的本地 pinned 基线是另一项组件 screen，不能与 N1 远端时延混用。
