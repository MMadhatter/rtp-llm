# DSV4 Thinking Tool Call XGrammar Plan

## 目标

避免 DeepSeek V4 在 thinking 内容里把 DSML tool call 当作 reasoning 文本输出，同时保持当前框架改动最小。

## 当前适配点

- DSV4 renderer 已经能在 `tool_choice=required` 或 named tool 时生成 DSML tool-call `structural_tag`。
- C++ `LogitsProcessorFactory` 已经支持 `in_think_mode + structural_tag`，并创建 `ReasoningGrammarLogitsProcessor`。
- `ReasoningGrammarLogitsProcessor` 在 thinking 阶段放行，遇到 `end_think_token_ids` 后才启动 xgrammar 约束后续 token。
- DSV4 renderer 已有非流式和流式的 reasoning/tool parser 后处理链路。

## 最小方案

1. 保留 `structural_tag` 作为 forced tool call 的生成侧强约束。
   - 不新增 public API。
   - 不把 `think_exclude_token_ids` 放进 `GenerateConfig`。
   - 不做 flat token mask，因为 `<｜DSML｜tool_calls>` 是 token 序列，mask 单 token 不可靠。

2. 在 renderer 后处理层做 tool-start interruption 兜底。
   - 如果 thinking 尚未显式遇到 `</think>`，但流里出现 `<｜DSML｜tool_calls>`，则视为 implicit think end。
   - marker 前面的文本作为 `reasoning_content`。
   - marker 及后续内容留给 DSML tool parser。
   - 对 partial marker 做 buffer，避免半截 `<｜DSML｜...` 泄露到 `reasoning_content` 或 `content`。

3. 只覆盖 DSV4 DSML tool call marker。
   - 当前 marker 集中放在 `DSML_REASONING_INTERRUPT_MARKERS`。
   - 后续如果 DSV4 增加新的 tool-start marker，只扩展这个 tuple。

## 边界

- `tool_choice=required` / named tool：继续使用 xgrammar `structural_tag` 强制工具调用格式。
- `tool_choice=auto`：本次不强制加 `structural_tag`，否则会把正常回答也约束成 tool call。auto 如需生成侧强约束，需要单独设计 normal text 或 tool call 的 grammar。
- 当前方案不承诺在采样阶段禁止 thinking 内生成 tool-start；它通过 xgrammar 保证 forced tool call 的后半段格式，通过 parser interruption 防止 tool markup 泄露到 reasoning 输出。

## 验证

- Python UT 覆盖 streaming 中 `</think>` 和 DSML marker 分 chunk 的场景。
- Python UT 覆盖 unclosed thinking 中遇到 DSML tool-start 时，reasoning 被切断且 DSML 不泄露。
- C++ UT 覆盖 `structural_tag` 在 thinking 结束后才生效。
- C++ UT 覆盖 `structural_tag + in_think_mode` 创建 `ReasoningGrammarLogitsProcessor`，不会退回 `ThinkModeLogitsProcessor`。
