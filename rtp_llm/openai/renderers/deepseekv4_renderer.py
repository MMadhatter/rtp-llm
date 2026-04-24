"""DeepSeek-V4 chat renderer.

Reuses the V3.2 renderer's "load encoding script from checkpoint" pattern.
The two real differences vs V3.2:

  1. The encoding script in the checkpoint is named ``encoding_dsv4.py``
     (V3.2 ships ``encoding_dsv32.py``).
  2. The function-call wire format swaps the outer wrapper from
     ``<｜DSML｜function_calls>...</｜DSML｜function_calls>`` to
     ``<｜DSML｜tool_calls>...</｜DSML｜tool_calls>``. Inner
     ``<｜DSML｜invoke>`` / ``<｜DSML｜parameter>`` shape is unchanged, so
     :class:`DeepSeekV4Detector` only swaps three regex strings.

Mirrors:
  * SGLang PR #23600: ``python/sglang/srt/entrypoints/openai/encoding_dsv4.py``
  * vLLM PR #40760: ``vllm/renderers/deepseek_v4.py``
"""

import importlib.util
import logging
import os
import sys
from typing import Optional

from typing_extensions import override

from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.custom_renderer import RendererParams
from rtp_llm.openai.renderers.deepseekv32_renderer import DeepseekV32Renderer
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.deepseekv4_detector import (
    DeepSeekV4Detector,
)
from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import ReasoningParser


class DeepseekV4Renderer(DeepseekV32Renderer):
    """Render DeepSeek-V4 chat / tool / thinking sessions.

    Inherits prompt assembly (system message, tools attachment, thinking-mode
    flag, encode_config plumbing) from :class:`DeepseekV32Renderer` and only
    overrides the two pieces that V4 actually changes:

      * the encoding script filename loaded from the checkpoint
      * the tool-call detector class

    The reasoning parser stays at the V3-family parser since V4's thinking
    syntax (``<think>...</think>``) is unchanged.
    """

    ENCODING_SCRIPT_NAME = "encoding_dsv4.py"
    ENCODING_MODULE_NAME = "encoding_dsv4"

    @override
    def _load_encoding_module(self, ckpt_path: str):
        """Load ``encoding/encoding_dsv4.py`` from the V4 checkpoint.

        The script ships in the model repo (e.g.
        ``deepseek-ai/DeepSeek-V4-Flash-Base/encoding/encoding_dsv4.py``)
        and exposes ``encode_messages`` /
        ``parse_message_from_completion_text`` — the same ABI as V3.2's
        ``encoding_dsv32``, so the rest of the rendering / parsing pipeline
        works unchanged.
        """
        encoding_folder = os.path.join(ckpt_path, "encoding")
        encoding_script_path = os.path.join(
            encoding_folder, self.ENCODING_SCRIPT_NAME
        )

        if not os.path.exists(encoding_script_path):
            raise FileNotFoundError(
                f"DeepSeek V4 encoding script not found at "
                f"{encoding_script_path}. Please ensure the checkpoint "
                f"includes the 'encoding' folder with "
                f"{self.ENCODING_SCRIPT_NAME}."
            )

        try:
            spec = importlib.util.spec_from_file_location(
                self.ENCODING_MODULE_NAME, encoding_script_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load spec from {encoding_script_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[self.ENCODING_MODULE_NAME] = module
            spec.loader.exec_module(module)

            logging.info(
                f"Successfully loaded DeepSeek V4 encoding module from "
                f"{encoding_script_path}"
            )
            return module
        except Exception as e:
            raise ImportError(
                f"Failed to load DeepSeek V4 encoding module from "
                f"{encoding_script_path}: {str(e)}"
            )

    @override
    def _create_detector(
        self, request: ChatCompletionRequest
    ) -> Optional[BaseFormatDetector]:
        """Use the V4 (tool_calls wrapper) DSML detector."""
        if not request.tools:
            return None
        thinking_mode = "thinking" if self.in_think_mode(request) else "chat"
        return DeepSeekV4Detector(
            encoding_module=self.encoding_module,
            thinking_mode=thinking_mode,
        )


register_renderer("deepseek_v4", DeepseekV4Renderer)
