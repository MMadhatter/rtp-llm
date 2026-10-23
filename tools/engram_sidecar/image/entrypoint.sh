#!/usr/bin/env bash
set -euo pipefail
RTP_IMAGE_ROOT=${RTP_IMAGE_ROOT:-/opt/engram-runtime}
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export VIRTUAL_ENV="$RTP_IMAGE_ROOT/venv"
export PATH="$VIRTUAL_ENV/bin:/opt/rh/gcc-toolset-12/root/usr/bin:$PATH"
export PYTHONPATH="$RTP_IMAGE_ROOT/repo:$RTP_IMAGE_ROOT/native${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-$CUDA_HOME/bin/ptxas}
RTP_IMAGE_SITE="$VIRTUAL_ENV/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="$RTP_IMAGE_ROOT/runtime-libs/nvcomp:/opt/conda310/lib:/opt/rh/gcc-toolset-12/root/usr/lib64:$RTP_IMAGE_SITE/torch/lib:$RTP_IMAGE_SITE/nvidia/nccl/lib:$RTP_IMAGE_SITE/nvidia/nvshmem/lib:$RTP_IMAGE_SITE/nvidia/cudnn/lib:$RTP_IMAGE_SITE/nvidia/cusparselt/lib:$RTP_IMAGE_SITE/nvidia/cu13/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
RTP_RUNTIME_DIR=${RTP_RUNTIME_DIR:-/tmp/rtp-engram}
mkdir -p "$RTP_RUNTIME_DIR"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$RTP_RUNTIME_DIR/cache}
cd "$RTP_RUNTIME_DIR"
command=${1:-help}
if (($#)); then shift; fi
case "$command" in
  registry) exec python -m tools.engram_sidecar.framework.registry_server "$@" ;;
  sidecar) exec python -m tools.engram_sidecar.framework.sidecar "$@" ;;
  owner) exec python -m tools.engram_sidecar.framework.checkpoint_owner "$@" ;;
  serve) exec python -m rtp_llm.start_server "$@" ;;
  check) exec python "$RTP_IMAGE_ROOT/repo/tools/engram_sidecar/image/check_runtime.py" "$@" ;;
  python) exec python "$@" ;;
  shell) exec /bin/bash "$@" ;;
  help) printf '%s
' 'Engram experimental image: sidecar [role args] -- [owner args] | registry [dev coordinator args] | owner [checkpoint_owner args] | serve [start_server args] | check | python ... | shell' 'RTP_ENGRAM_MODE=local|shared|auto; shared uses RTP_ENGRAM_REGISTRY_ENDPOINT or RTP_ENGRAM_OWNER_ENDPOINT; legacy file bootstrap remains supported.' 'RTP_ENGRAM_DISCOVERY=platform uses a platform-assigned Owner endpoint/epoch and explicit producer|consumer sidecar role; no Registry or election.' 'Owner must outlive all consumers. Local/auto fallback requires complete checkpoint weights.' ;;
  *) printf 'Unknown command: %s\n' "$command" >&2; exit 2 ;;
esac
