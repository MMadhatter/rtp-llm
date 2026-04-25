# External Patches Log

记录在 RTP-LLM 仓库以外（conda env、第三方库等）做的修改，方便日后复现/回滚。

## fastsafetensors 0.7.x — 新增 `F8_E8M0` DType

**文件**: `/opt/conda310/lib/python3.10/site-packages/fastsafetensors/st_types.py`

**原因**: DeepSeek-V4 / V3.2 的 FP8 权重使用 `ue8m0` 作为 scale 的 dtype（safetensors header 中标记为 `F8_E8M0`），fastsafetensors 0.7 系列的 `DType` Enum 不识别该值，加载 ckpt 时直接报 `ValueError: 'F8_E8M0' is not a valid DType`。

**修改**: 在 `class DType(Enum)` 里追加一行：

```python
F8_E8M0 = "F8_E8M0"
```

并把它映射成 8-bit 整型（与 F8_E4M3/F8_E5M2 一致）：

`/opt/conda310/lib/python3.10/site-packages/fastsafetensors/frameworks/_torch.py`

```python
DType.F8_E8M0: DType.I8,           # 加在 dtype_to_int 表里
dtype_convert[DType.F8_E8M0] = torch.uint8   # ue8m0 是 8-bit 无符号尾数
```

**影响范围**: 只让 fastsafetensors 在加载时把这块 8-bit 缓冲读出来；后续的 quant kernel 自己解释 ue8m0 的语义。等 fastsafetensors upstream 加上 F8_E8M0 后可以撤掉这个 patch。

