from concurrent.futures import ThreadPoolExecutor

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import rtp_llm_ops


class MultiHeadEmbeddingBase(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        layer_id: int = 0,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.list_of_N = config.engram_config.vocab_size
        self.num_heads = len(self.list_of_N)
        self.parallelism_config = parallelism_config

        offsets_cuda = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device="cuda"),
                torch.cumsum(
                    torch.tensor(self.list_of_N[:-1], dtype=torch.long, device="cuda"),
                    dim=0,
                ),
            ]
        )
        self.register_buffer("offsets", offsets_cuda)

    def _tp_all_gather(self, output: torch.Tensor) -> torch.Tensor:
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

    def start_async(self, hash_input_ids_host: torch.Tensor):
        pass

    def wait_async(self) -> torch.Tensor:
        raise NotImplementedError


class GpuMultiHeadEmbedding(MultiHeadEmbeddingBase):
    """Synchronous GPU embedding lookup via rtp_llm_ops.embedding."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weight: torch.Tensor,
        layer_id: int = 0,
        **kwargs,
    ):
        super().__init__(config, parallelism_config, layer_id)
        self.embedding_weight = weight.to("cuda")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        shifted_input_ids = input_ids + self.offsets
        orig_shape = shifted_input_ids.shape
        flat_ids = shifted_input_ids.reshape(-1)
        output = torch.empty(
            (flat_ids.numel(), self.embedding_weight.shape[-1]),
            dtype=self.embedding_weight.dtype,
            device=shifted_input_ids.device,
        )
        rtp_llm_ops.embedding(output, flat_ids, self.embedding_weight)
        output = output.reshape(*orig_shape, -1)
        return self._tp_all_gather(output)


class CpuMultiHeadEmbedding(MultiHeadEmbeddingBase):
    """Async CPU embedding with pinned-memory H2D to a static GPU buffer.

    Designed for CUDA-graph compatibility: the pre-allocated ``_gpu_buffer``
    has a fixed device address so that graph capture / replay always read
    from the same pointer.

    Typical call sequence
    ---------------------
    **Normal inference (non-graph)**::

        start_async(hash_ids_host)   # kick off CPU thread
        ...                          # other GPU work can overlap
        output = forward(input_ids)  # wait_async → H2D → return

    **CUDA graph capture**::

        # start_async is NOT called → _async_future stays None
        output = forward(input_ids)  # returns _gpu_buffer directly

    **CUDA graph replay**::

        start_async(hash_ids_host)
        wait_async()                 # fill _gpu_buffer before replay
        graph.replay()               # reads from fixed _gpu_buffer addr
    """

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weight: torch.Tensor,
        layer_id: int = 0,
        max_batch_size: int = 0,
    ):
        super().__init__(config, parallelism_config, layer_id)

        self.embedding_weight = weight
        self.offsets_cpu = self.offsets.cpu()
        self._h2d_stream = torch.cuda.Stream()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._async_future = None
        self._async_event = torch.cuda.Event()
        self._actual_num_tokens = 0

        embed_dim = weight.shape[-1]
        buf_tokens = max(max_batch_size, 1)
        self._cpu_buffer = torch.empty(
            (1, buf_tokens, self.num_heads, embed_dim),
            dtype=weight.dtype,
        ).pin_memory()
        self._gpu_buffer = torch.empty(
            (1, buf_tokens, self.num_heads, embed_dim),
            dtype=weight.dtype,
            device="cuda",
        )

    def start_async(self, hash_input_ids_host: torch.Tensor):
        self._async_future = self._executor.submit(
            self._async_compute, hash_input_ids_host
        )

    def _async_compute(self, input_ids_host: torch.Tensor):
        shifted = input_ids_host + self.offsets_cpu
        output = F.embedding(shifted, self.embedding_weight)
        n = output.shape[1]
        self._cpu_buffer[:, :n, :, :].copy_(output)
        self._actual_num_tokens = n

    def wait_async(self) -> torch.Tensor:
        self._async_future.result()
        n = self._actual_num_tokens
        with torch.cuda.stream(self._h2d_stream):
            self._gpu_buffer[:, :n, :, :].copy_(
                self._cpu_buffer[:, :n, :, :], non_blocking=True
            )
            self._async_event.record(self._h2d_stream)
        torch.cuda.current_stream().wait_event(self._async_event)
        self._async_future = None
        return self._tp_all_gather(self._gpu_buffer[:, :n, :, :])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._async_future is not None:
            return self.wait_async()
        n = input_ids.shape[1]
        return self._tp_all_gather(self._gpu_buffer[:, :n, :, :])


def MultiHeadEmbedding(
    config: ModelConfig,
    parallelism_config: ParallelismConfig,
    weight: torch.Tensor,
    layer_id: int = 0,
    max_batch_size: int = 0,
) -> MultiHeadEmbeddingBase:
    if config.engram_config.use_gpu_embedding:
        return GpuMultiHeadEmbedding(
            config,
            parallelism_config,
            weight,
            layer_id,
        )
    return CpuMultiHeadEmbedding(
        config,
        parallelism_config,
        weight,
        layer_id,
        max_batch_size,
    )
