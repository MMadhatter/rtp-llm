import logging

import torch

from rtp_llm.config.quant_config import Fp8PerTensorQuantConfig, QuantizationConfig
from rtp_llm.model_loader.static_fp8_quant_weight import StaticPerTensorFp8Weight


class LoadQuantDynamicPerTensorFp8Weight(StaticPerTensorFp8Weight):
    fp8_attn_weights_map = {
        W.attn_qkv_w: (
            W.attn_qkv_s,
            None,
            None,
        ),
        W.attn_o_w: (
            W.attn_o_s,
            None,
            None,
        ),
        W.mla_fusedqkrope_w: (W.mla_fusedqkrope_s, None, None),
        W.mla_fusedqkrope_no_lora_w: (W.mla_fusedqkrope_no_lora_s, None, None),
        W.mla_q_b_w: (W.mla_q_b_s, None, None),
        W.mla_k_nope_w: (W.mla_k_nope_s, None, None),
        W.mla_v_w: (W.mla_v_s, None, None),
    }

    fp8_ffn_weights_maps = {
        W.ffn_w1: (W.ffn_s1, None, None),
        W.ffn_w3: (W.ffn_s3, None, None),
        W.ffn_w2: (
            W.ffn_s2,
            None,
            None,
        ),
        W.ffn_w13: (
            W.ffn_s13,
            None,
            None,
        ),
    }

    fp8_partial_moe_weights_maps = {
        W.moe_w1: (W.moe_s1, None, None),
        W.moe_w2: (W.moe_s2, None, None),
    }

    weight_scale_map = {
        **fp8_attn_weights_map,
        **fp8_ffn_weights_maps,
        **fp8_partial_moe_weights_maps,
    }

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if (
            quant_config.is_quanted()
            or not isinstance(quant_config, Fp8PerTensorQuantConfig)
            or not quant_config.is_dynamic()
        ):
            return False
        name = src_weight_info.name
        return name in cls.w8a8_weight_list and (
            src_weight_info.weight_style
            not in [WeightStyle.TRT_ENGINE, WeightStyle.RTP_SMOOTH_LLM_STYLE]
        )

    def __init__(
        self,
        src_weight_info: AtomicWeight,
        quant_config: QuantizationConfig,
        *args,
        **kwargs,
    ):
        params = src_weight_info.extract_params(
            src_weight_info.__class__, src_weight_info, quant_config
        )
        kernel: AtomicWeight = create_w8a8_fp8_weight(src_weight_info, **params)
        sub_weights = {kernel.name: kernel}

        scale_name, _, _ = self.weight_scale_map.get(src_weight_info.name)
        scale_params = copy.deepcopy(params)
        scale_params["name"] = scale_name
        scale: AtomicWeight = create_w8a8_fp8_weight(src_weight_info, **scale_params)
        sub_weights.update({scale.name: scale})

        CompositeWeight.__init__(
            self, sub_weights, quant_config=quant_config, *args, **kwargs
        )
        self.kernel = kernel
        self.scale = scale
        from rtp_llm.models_py.utils.debug import set_trace_on_tty

        set_trace_on_tty()
        self.act_scale = None
        self.act_scale_inv = None

    def _load_raw_tensor(
        self,
        database: BaseDatabase,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ):
        kernel = self.kernel._load_raw_tensor(database, layer_id, device, load_config)
        res = {}
        quant_kernel, scale = quantize_weight_to_fp8(kernel.get(self.kernel.name))
        quant_kernel = quant_kernel.T
        res = {
            self.kernel.name: quant_kernel.contiguous().to(device),
            self.scale.name: scale.contiguous().to(device),
        }
        return res

    def _postprocess(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        device: str,
        load_config: LoadConfig,
    ):
        # this func do nothing but is called here
        processed_res = super()._postprocess(tensor, device, load_config)

        kernel_weight = processed_res[self.kernel.name]
        weight_scale_name = self.FP8_SCALE_MAP.get(self.kernel.name)

        input_scale_r_str, _ = self.FP8_ACT_SCALE_MAP.get(self.kernel.name)[1]
        intput_scale_str, _ = self.FP8_ACT_SCALE_MAP.get(self.kernel.name)[0]

        kernel_scale = processed_res.get(weight_scale_name, None)
        input_scale = processed_res.get(input_scale_r_str, None)

        if isinstance(self.kernel, MoeAtomicWeight):
            if self.kernel.name is W.moe_w1:
                # handle moe w13 weight
                num_local_experts, moe_inter_padding_size, _ = kernel_weight.shape

                assert moe_inter_padding_size == (
                    load_config.moe_inter_padding_size * 2
                )
                assert kernel_scale is not None
                max_kernel_scale = kernel_scale.max(dim=1).values
                moe_inter_padding_size = moe_inter_padding_size // 2

                for expert_id in range(num_local_experts):
                    start = 0
                    for shard_id in range(2):
                        if (
                            max_kernel_scale[expert_id]
                            != kernel_scale[expert_id][shard_id]
                        ):
                            # rescale shard
                            dq_weight = (
                                kernel_weight[expert_id][
                                    start : start + moe_inter_padding_size, :
                                ].to(torch.float16)
                                * kernel_scale[expert_id][shard_id]
                            )
                            kernel_weight[expert_id][
                                start : start + moe_inter_padding_size, :
                            ] = (dq_weight / max_kernel_scale[expert_id]).to(
                                torch.float8_e4m3fn
                            )
                # w13 to w31
                kernel_weight = torch.cat(
                    [
                        kernel_weight[:, load_config.moe_inter_padding_size :, :],
                        kernel_weight[:, : load_config.moe_inter_padding_size, :],
                    ],
                    dim=1,
                )

                processed_res[self.kernel.name] = kernel_weight
                processed_res[W.moe_s1] = max_kernel_scale

            if input_scale is not None:
                input_scale = input_scale.max()
                processed_res[input_scale_r_str] = input_scale
                processed_res[intput_scale_str] = 1.0 / input_scale
            return processed_res

        # handle qkv_proj quant weight
        if self.kernel.name is W.attn_qkv_w:
            kernel_weight = processed_res[self.kernel.name]
            kernel_scale = processed_res[W.attn_qkv_s]

            head_size = load_config.size_per_head
            head_num_kv = load_config.head_num_kv
            head_num_q = load_config.head_num
            assert head_num_q + 2 * head_num_kv == kernel_weight.shape[0] // head_size
            logical_widths = [
                head_num_q * head_size,
                head_num_kv * head_size,
                head_num_kv * head_size,
            ]

            qkv_rescale, max_scale = merge_qkv_hf_fp8_with_scale(
                [
                    kernel_weight[0 : logical_widths[0], :],
                    kernel_weight[
                        logical_widths[0] : logical_widths[0] + logical_widths[1], :
                    ],
                    kernel_weight[logical_widths[0] + logical_widths[1] :, :],
                    kernel_scale[0],
                    kernel_scale[1],
                    kernel_scale[2],
                ]
            )
            processed_res[self.kernel.name] = qkv_rescale
            processed_res[W.attn_qkv_s] = max_scale

            # maybe handle qkv_proj input scale
            if processed_res.get(W.pre_ln_static_quant_reciprocal) is not None:
                assert processed_res[W.pre_ln_static_quant_reciprocal].shape[0] == 3
                processed_res[W.pre_ln_static_quant_reciprocal] = processed_res[
                    W.pre_ln_static_quant_reciprocal
                ].max()
                processed_res[W.pre_ln_static_quant] = (
                    1.0 / processed_res[W.pre_ln_static_quant_reciprocal]
                )
            return processed_res

        return processed_res
