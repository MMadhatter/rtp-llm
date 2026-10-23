#!/usr/bin/env bash
set -euo pipefail
source "${1:?usage: consumer-start.sh CONSUMER_CONFIG.sh local|shared|auto}"
mode=${2:?mode required}
[[ $mode == local || $mode == shared || $mode == auto ]] || exit 2
: "${IMAGE:?}" "${NAME:?}" "${CHECKPOINT:?}" "${MODEL_ENV:?}"
: "${GPU_UUIDS:?}" "${MEMORY_LIMIT:?}" "${SHM_SIZE:?}"
CONTAINER_UID=${CONTAINER_UID:-$(id -u)}
CONTAINER_GID=${CONTAINER_GID:-$(id -g)}
[[ $CONTAINER_UID != 0 ]] || exit 2
[[ -f $MODEL_ENV && -f $CHECKPOINT/config.json ]]
if pouch inspect "$NAME" >/dev/null 2>&1; then echo 'Container name exists' >&2; exit 2; fi
args=(-e USER=rtp-engram -e LOGNAME=rtp-engram -e HOME=/tmp/rtp-engram
      --net=host --user "$CONTAINER_UID:$CONTAINER_GID" --memory "$MEMORY_LIMIT"
      --shm-size "$SHM_SIZE" --nvidia-visible-devs="$GPU_UUIDS" --nvidia-capabilities=all
      --env-file "$MODEL_ENV" -e "RTP_ENGRAM_MODE=$mode" -v "$CHECKPOINT:/checkpoint:ro")
if [[ -n ${REGISTRY_ENDPOINT:-} && -n ${OWNER_ENDPOINT:-} ]]; then
  echo 'Configure only REGISTRY_ENDPOINT or OWNER_ENDPOINT' >&2; exit 2
fi
if [[ $mode != local && -n ${OWNER_ENDPOINT:-}${REGISTRY_ENDPOINT:-} ]]; then
  : "${TLS_DIR:?}" "${SERVICE_ID:?}" "${MODEL_REVISION:?}" "${DATASET_ID:?}"
  args+=(--device=/dev/nvidia-caps-imex-channels/channel0 -v "$TLS_DIR:/run/engram-tls:ro"
    -e "RTP_ENGRAM_SERVICE_ID=$SERVICE_ID"
    -e "RTP_ENGRAM_MODEL_REVISION=$MODEL_REVISION" -e "RTP_ENGRAM_DATASET_ID=$DATASET_ID"
    -e RTP_ENGRAM_TLS_CA=/run/engram-tls/ca.crt
    -e RTP_ENGRAM_TLS_CERT=/run/engram-tls/consumer.crt
    -e RTP_ENGRAM_TLS_KEY=/run/engram-tls/consumer.key
    -e RTP_ENGRAM_MAX_ALLOCATION_BYTES=137438953472
    -e RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES=274877906944
    -e RTP_ENGRAM_GRAPH_MODE=process_lifetime)
  if [[ -n ${REGISTRY_ENDPOINT:-} ]]; then
    : "${FABRIC_DOMAIN:?}" "${AUTHORIZATION_GROUP:?}" "${INSTANCE_ID:?}"
    args+=(-e "RTP_ENGRAM_REGISTRY_ENDPOINT=$REGISTRY_ENDPOINT"
      -e "RTP_ENGRAM_FABRIC_DOMAIN=$FABRIC_DOMAIN" -e "RTP_ENGRAM_AUTHORIZATION_GROUP=$AUTHORIZATION_GROUP"
      -e "RTP_ENGRAM_INSTANCE_ID=$INSTANCE_ID")
  else
    args+=(-e "RTP_ENGRAM_OWNER_ENDPOINT=$OWNER_ENDPOINT")
  fi
elif [[ $mode == shared ]]; then echo 'shared requires REGISTRY_ENDPOINT or OWNER_ENDPOINT' >&2; exit 2; fi
# A/B must use the same prefetch policy; model.env contains the role/ports/routes.
args+=(-e DSV41_CPU_ENGRAM_LOOKUP=0 -e DSV41_ASYNC_ENGRAM_LOOKUP=0)
for device in ${RDMA_DEVICES:-}; do test -c "$device"; args+=(--device="$device"); done
pouch run -d --name "$NAME" "${args[@]}" "$IMAGE" serve
