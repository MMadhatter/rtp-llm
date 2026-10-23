#!/usr/bin/env bash
set -euo pipefail
source "${1:?usage: owner-start.sh OWNER_CONFIG.sh}"
: "${IMAGE:?}" "${NAME:?}" "${CHECKPOINT:?}" "${STATE_DIR:?}" "${TLS_DIR:?}"
: "${OWNER_IP:?}" "${OWNER_PORT:?}" "${GPU_UUID:?}" "${NUMA:?}"
: "${MODEL_REVISION:?}" "${SERVICE_ID:?}" "${MEMORY_LIMIT:?}"
CONTAINER_UID=${CONTAINER_UID:-$(id -u)}
CONTAINER_GID=${CONTAINER_GID:-$(id -g)}
[[ $CONTAINER_UID != 0 ]] || { echo 'Use a non-root container UID' >&2; exit 2; }
[[ -f $CHECKPOINT/config.json && -f $CHECKPOINT/model.safetensors.index.json ]]
for f in ca.crt owner.crt owner.key; do test -f "$TLS_DIR/$f"; done
if pouch inspect "$NAME" >/dev/null 2>&1; then
  echo 'Container name already exists; inspect it instead of replacing a live owner.' >&2; exit 2
fi
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$GPU_UUID"; then
  echo 'Selected GPU has compute clients; choose an unused reserved GPU.' >&2; exit 2
fi
umask 077
mkdir -m 700 "$STATE_DIR"
extra=()
if [[ -n ${CODE_PATCH_DIR:-} ]]; then
  # Development overlay only; the CI image already contains these files.
  while IFS= read -r -d '' file; do
    relative=${file#"$CODE_PATCH_DIR/"}
    extra+=(-v "$file:/opt/engram-runtime/repo/$relative:ro")
  done < <(find "$CODE_PATCH_DIR" -type f -name '*.py' -print0)
fi
pouch run -d --name "$NAME" --net=host \
  --user "$CONTAINER_UID:$CONTAINER_GID" --memory "$MEMORY_LIMIT" \
  --nvidia-visible-devs="$GPU_UUID" --nvidia-capabilities=all \
  --device=/dev/nvidia-caps-imex-channels/channel0 \
  -e USER=rtp-engram -e LOGNAME=rtp-engram -e HOME=/tmp/rtp-engram \
  -e PYTHONUNBUFFERED=1 -e CUDA_VISIBLE_DEVICES=0 \
  -v "$CHECKPOINT:/checkpoint:ro" -v "$STATE_DIR:/run/engram:rw" \
  -v "$TLS_DIR:/run/engram-tls:ro" "${extra[@]}" \
  "$IMAGE" owner --checkpoint /checkpoint --model-revision "$MODEL_REVISION" \
  --publication /run/engram/state --device 0 --numa "$NUMA" \
  --max-allocation-bytes "${MAX_ALLOCATION_BYTES:-137438953472}" \
  --budget-bytes "${BUDGET_BYTES:-274877906944}" \
  --discard-private-checkpoint-cache \
  --rpc-listen "$OWNER_IP:$OWNER_PORT" --service-id "$SERVICE_ID" \
  --tls-ca /run/engram-tls/ca.crt --tls-cert /run/engram-tls/owner.crt \
  --tls-key /run/engram-tls/owner.key --allowed-client engram-consumer
printf 'Started %s. Wait for READY using pouch logs. Keep Owner alive until all consumers exit.\n' "$NAME"
