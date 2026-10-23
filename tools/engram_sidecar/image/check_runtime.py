# SPDX-License-Identifier: Apache-2.0
"""Image import check without allocating GPU memory or reading a checkpoint."""
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import rtp_llm.ops as ops

ops.ensure_engine_ops_loaded()
import libth_grammar_tokenizer_info
import engram_native_dev_492ecf45 as native
import rtp_llm.start_server
from tools.engram_sidecar.framework import checkpoint_owner
from rtp_llm.models_py.modules.dsv41.engram_mode import resolve_mode

assert not torch.cuda.is_initialized()
assert all(value == 0 for value in native.resource_counters().values())
print(json.dumps({"status": "IMPORT_OK_NO_GPU_ALLOCATION", "torch": torch.__version__,
                  "engram_native": native.__file__, "mode": resolve_mode()[0]}))
