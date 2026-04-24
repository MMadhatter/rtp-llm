"""Unit tests for DeepSeekV4 DSML tool-call detector.

V4 differs from V3.2 only in the outer wrapper (``<｜DSML｜tool_calls>``
vs ``<｜DSML｜function_calls>``); the inner ``<｜DSML｜invoke>`` /
``<｜DSML｜parameter>`` shape is identical. So we test:

  1. The V4 wrapper is recognised.
  2. The V3.2 wrapper is *not* recognised by the V4 detector (regression
     guard against accidental inheritance leak).
  3. Parameters in both XML and direct-JSON shapes parse correctly.
  4. Multiple consecutive ``invoke`` blocks parse correctly.
  5. Streaming incremental parsing yields the same final tool calls as
     one-shot parsing.
"""

import json
import unittest

from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import (
    Function,
    Tool,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.deepseekv4_detector import (
    DeepSeekV4Detector,
)


def _tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="Get the weather of a city",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city"],
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="get_time",
                description="Get current time",
                parameters={"type": "object", "properties": {}},
            ),
        ),
    ]


class DeepSeekV4DetectorWrapperTest(unittest.TestCase):
    """Outer wrapper recognition / non-recognition."""

    def setUp(self):
        self.detector = DeepSeekV4Detector()
        self.tools = _tools()

    def test_recognises_v4_tool_calls_wrapper(self):
        text = (
            "Here you go:\n"
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_time"></｜DSML｜invoke>\n'
            "</｜DSML｜tool_calls>"
        )
        self.assertTrue(self.detector.has_tool_call(text))
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_time")

    def test_ignores_v32_function_calls_wrapper(self):
        # V3.2 sends function_calls; V4 detector must NOT match it as a
        # full tool-call block (would produce zero calls).
        text = (
            "<｜DSML｜function_calls>\n"
            '<｜DSML｜invoke name="get_time"></｜DSML｜invoke>\n'
            "</｜DSML｜function_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        # The V3.2 wrapper isn't the V4 ``bot_token``; calls list must be
        # empty (the leading ``has_tool_call`` returns True only because
        # of the inner ``<｜DSML｜invoke`` token, but the outer regex
        # mismatch means no parsed calls).
        self.assertEqual(len(result.calls), 0)

    def test_no_tool_call_in_plain_text(self):
        result = self.detector.detect_and_parse(
            "Just a normal message.", self.tools
        )
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "Just a normal message.")


class DeepSeekV4DetectorParameterTest(unittest.TestCase):
    """Inner parameter shapes — both XML and direct JSON."""

    def setUp(self):
        self.detector = DeepSeekV4Detector()
        self.tools = _tools()

    def test_xml_parameters(self):
        text = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="city" string="true">San Francisco'
            "</｜DSML｜parameter>\n"
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "San Francisco")

    def test_direct_json_parameters(self):
        text = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '{"city": "Tokyo", "unit": "celsius"}\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "Tokyo")
        self.assertEqual(params["unit"], "celsius")

    def test_multiple_invoke_blocks(self):
        text = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">'
            '{"city":"Paris"}</｜DSML｜invoke>\n'
            '<｜DSML｜invoke name="get_time"></｜DSML｜invoke>\n'
            "</｜DSML｜tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        names = [c.name for c in result.calls]
        self.assertEqual(set(names), {"get_weather", "get_time"})


class DeepSeekV4DetectorStructureInfoTest(unittest.TestCase):
    """structure_info() should still emit the inherited V3.2 inner-block shape."""

    def test_structure_info_uses_invoke_tag(self):
        info_fn = DeepSeekV4Detector().structure_info()
        info = info_fn("get_weather")
        self.assertIn("<｜DSML｜invoke", info.begin)
        self.assertIn('name="get_weather"', info.begin)
        self.assertEqual(info.end, "</｜DSML｜invoke>")


class DeepSeekV4DetectorStreamingTest(unittest.TestCase):
    """Incremental parsing produces the same final calls as one-shot parsing."""

    def setUp(self):
        self.detector = DeepSeekV4Detector()
        self.tools = _tools()

    def test_streaming_matches_full_parse(self):
        full_text = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">'
            '{"city":"Berlin"}</｜DSML｜invoke>\n'
            "</｜DSML｜tool_calls>"
        )
        # Stream the text 4 chars at a time.
        all_chunks = []
        for i in range(0, len(full_text), 4):
            chunk = full_text[i : i + 4]
            r = self.detector.parse_streaming_increment(chunk, self.tools)
            all_chunks.extend(r.calls)
        # The streaming detector emits at least the function-name event
        # and one or more argument-diff events. Concatenating the streamed
        # argument fragments must reconstruct the final JSON.
        names = [c.name for c in all_chunks if c.name]
        self.assertEqual(names, ["get_weather"])
        params_concat = "".join(c.parameters for c in all_chunks if c.parameters)
        self.assertEqual(json.loads(params_concat), {"city": "Berlin"})


if __name__ == "__main__":
    unittest.main()
