from typing import Optional

from rtp_llm.model_factory_register import register_model
from rtp_llm.models.deepseek_v2 import DeepSeekV2
from rtp_llm.models_py.model_desc.deepseek_v4_mock import DeepseekV4Model
from rtp_llm.models_py.model_desc.module_base import GptModelBase


class DeepSeekV4(DeepSeekV2):
    def _create_python_model(self) -> Optional[GptModelBase]:
        model_config = self.model_config
        parallelism_config = self.parallelism_config
        fmha_config = self.fmha_config
        py_hw_kernel_config = self.hw_kernel_config
        moe_config = self.moe_config
        max_generate_batch_size = self.max_generate_batch_size

        self.py_model = DeepseekV4Model(
            model_config,
            parallelism_config,
            self.weight,
            moe_config,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )


register_model("deepseek_v4", DeepSeekV4, ["DeepseekV4ForCausalLM"])
