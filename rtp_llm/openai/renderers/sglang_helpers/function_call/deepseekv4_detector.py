"""DeepSeek-V4 function-call format detector.

Identical to V3.2's DSML format except the outer block wrapper is
``<｜DSML｜tool_calls>...</｜DSML｜tool_calls>`` instead of
``<｜DSML｜function_calls>...</｜DSML｜function_calls>``. The inner
``<｜DSML｜invoke>`` / ``<｜DSML｜parameter>`` shape is unchanged, so we
inherit all the parsing logic from :class:`DeepSeekV32Detector` and only
swap the three regex/marker strings.

Mirrors SGLang PR #23600's ``deepseekv4_detector.py``.

Example (XML parameters)::

    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_weather">
        <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
        </｜DSML｜invoke>
    </｜DSML｜tool_calls>

Example (direct JSON parameters)::

    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_weather">
        { "city": "San Francisco" }
        </｜DSML｜invoke>
    </｜DSML｜tool_calls>
"""

from rtp_llm.openai.renderers.sglang_helpers.function_call.deepseekv32_detector import (
    DeepSeekV32Detector,
)


class DeepSeekV4Detector(DeepSeekV32Detector):
    """Detector for the DeepSeek-V4 DSML tool-call format.

    Args:
        encoding_module: optional ``encoding_dsv4`` module loaded from the
            checkpoint's ``encoding/`` folder. When present, the detector
            delegates ``parse_message_from_completion_text`` to the official
            parser shipped with the checkpoint; otherwise it falls back to
            the regex parser inherited from the V3.2 detector.
        thinking_mode: passed straight through to the official parser to
            switch between chat / thinking / tool modes.
    """

    def __init__(self, encoding_module=None, thinking_mode: str = "chat"):
        super().__init__(encoding_module=encoding_module, thinking_mode=thinking_mode)
        # V4-specific outer wrapper. Inner invoke / parameter regexes are
        # inherited from V3.2 unchanged.
        self.bot_token = "<｜DSML｜tool_calls>"
        self.eot_token = "</｜DSML｜tool_calls>"
        self.function_calls_regex = (
            r"<｜DSML｜tool_calls>(.*?)</｜DSML｜tool_calls>"
        )

    def has_tool_call(self, text: str) -> bool:
        """Return True if ``text`` looks like the start of a V4 tool-call."""
        return self.bot_token in text or "<｜DSML｜invoke" in text
