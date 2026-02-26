import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import rtp_llm_ops


class EmbeddingTorch(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.weight)


class Embedding(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weight: torch.Tensor,
    ):
        super().__init__()
        self.weight = weight
        self.config = config
        self.parallelism_config = parallelism_config

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tokens = input.size(0)
        hidden_size = self.weight.size(-1)
        output = torch.empty(
            (tokens, hidden_size), dtype=self.weight.dtype, device=input.device
        )
        rtp_llm_ops.embedding(output, input, self.weight.data)
        if self.parallelism_config.tp_size > 1:
            m, n = output.shape
            output = all_gather(output, group=Group.TP)
            output = (
                output.reshape(self.parallelism_config.tp_size, m, n)
                .transpose(0, 1)
                .contiguous()
                .reshape(m, -1)
            )
        return output


class EmbeddingBert(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weight: torch.Tensor,
    ):
        super().__init__()
        self.weight = weight
        self.config = config
        self.parallelism_config = parallelism_config

    def forward(
        self,
        input: torch.Tensor,
        combo_position_ids: torch.Tensor,
        position_encoding: torch.Tensor,
        combo_tokens_type_ids: torch.Tensor,
        token_type_embedding: torch.Tensor,
        input_embedding_scalar: float,
    ) -> torch.Tensor:
        tokens = input.size(0)
        hidden_size = self.weight.size(-1)
        output = torch.empty(
            (tokens, hidden_size), dtype=self.weight.dtype, device=input.device
        )

        rtp_llm_ops.embedding_bert(
            output,
            input,
            self.weight.data,
            combo_position_ids,
            position_encoding,
            combo_tokens_type_ids,
            token_type_embedding,
            input_embedding_scalar,
        )

        if self.parallelism_config.tp_size > 1:
            m, n = output.shape
            output = all_gather(output, group=Group.TP)
            output = (
                output.reshape(self.parallelism_config.tp_size, m, n)
                .transpose(0, 1)
                .contiguous()
                .reshape(m, -1)
            )
        return output


class MultiHeadEmbedding(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weight: torch.Tensor,
    ):
        super().__init__()
        # 有(max_ngram_size - 1） * n_head_per_ngram个embedding在dim_0 concat，所以查询时需要每个给每个embedding加上offset
        self.list_of_N = config.engram_config.vocab_size
        self.num_heads = len(self.list_of_N)
        self.embedding_weight = weight
        self.parallelism_config = parallelism_config

        offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device="cuda"),
                torch.cumsum(
                    torch.tensor(self.list_of_N[:-1], dtype=torch.long, device="cuda"),
                    dim=0,
                ),
            ]
        )
        self.register_buffer("offsets", offsets)

        self.force_gpu = True
        if self.force_gpu:
            self.embedding_weight = self.embedding_weight.to("cuda")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:

        shifted_input_ids = input_ids + self.offsets

        if self.force_gpu:
            orig_shape = shifted_input_ids.shape
            flat_ids = shifted_input_ids.reshape(-1)
            output = torch.empty(
                (flat_ids.numel(), self.embedding_weight.shape[-1]),
                dtype=self.embedding_weight.dtype,
                device=shifted_input_ids.device,
            )
            rtp_llm_ops.embedding(output, flat_ids, self.embedding_weight)
            output = output.reshape(*orig_shape, -1)

        else:
            # this pass does not support cuda graph capture
            shifted_input_ids = shifted_input_ids.cpu()
            output = F.embedding(shifted_input_ids, self.embedding_weight)

        # 目前并行仍然在最后一个dim（dim_1）的维度做拆分，与普通的embedding逻辑一致。之后优化可以考虑在dim0上
        if self.parallelism_config.tp_size > 1:
            orig_shape = output.shape
            output = all_gather(output, group=Group.TP)
            output = (
                output.reshape(self.parallelism_config.tp_size, *orig_shape)
                .permute(*range(1, len(orig_shape)), 0, len(orig_shape))
                .contiguous()
                .reshape(*orig_shape[:-2], -1)
            )
        return output
