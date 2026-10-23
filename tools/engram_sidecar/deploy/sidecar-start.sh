#!/usr/bin/env bash
set -euo pipefail
config=${1:?usage: sidecar-start.sh SIDECAR_CONFIG.sh}
source "$config"
: "${REGISTRY_ENDPOINT:?}" "${SIDECAR_ROLE:?}"
if [[ $SIDECAR_ROLE == producer || $SIDECAR_ROLE == auto ]]; then
  # Platform-authorized candidates need the Owner's pre-reserved resources.
  exec bash "$(dirname "$0")/owner-start.sh" "$config"
fi
[[ $SIDECAR_ROLE == consumer ]] || exit 2
: "${IMAGE:?}" "${NAME:?}" "${TLS_DIR:?}" "${FABRIC_DOMAIN:?}" "${AUTHORIZATION_GROUP:?}"
: "${INSTANCE_ID:?}" "${SERVICE_ID:?}" "${MODEL_REVISION:?}" "${DATASET_ID:?}"
CONTAINER_UID=${CONTAINER_UID:-$(id -u)}
CONTAINER_GID=${CONTAINER_GID:-$(id -g)}
[[ $CONTAINER_UID != 0 ]] || exit 2
if pouch inspect "$NAME" >/dev/null 2>&1; then echo 'Container name exists' >&2; exit 2; fi
for f in ca.crt consumer.crt consumer.key; do test -f "$TLS_DIR/$f"; done
extra=()
if [[ -n ${CODE_PATCH_DIR:-} ]]; then
  while IFS= read -r -d '' file; do
    relative=${file#"$CODE_PATCH_DIR/"}
    if [[ $relative == rtp_llm/* ]]; then
      target=/opt/engram-runtime/venv/lib/python3.10/site-packages/$relative
    else
      target=/opt/engram-runtime/repo/$relative
    fi
    extra+=(-v "$file:$target:ro")
  done < <(find "$CODE_PATCH_DIR" -type f \( -name '*.py' -o -path '*/image/entrypoint.sh' \) -print0)
fi
# Consumer sidecar only observes metadata. RTP workers receive GPU/IMEX devices.
pouch run -d --name "$NAME" --net=host --user "$CONTAINER_UID:$CONTAINER_GID" \
  --memory "${MEMORY_LIMIT:-1g}" \
  -e USER=rtp-engram -e LOGNAME=rtp-engram -e HOME=/tmp/rtp-engram -e PYTHONUNBUFFERED=1 \
  -v "$TLS_DIR:/run/engram-tls:ro" \
  -e "RTP_ENGRAM_REGISTRY_ENDPOINT=$REGISTRY_ENDPOINT" \
  -e "RTP_ENGRAM_FABRIC_DOMAIN=$FABRIC_DOMAIN" -e "RTP_ENGRAM_AUTHORIZATION_GROUP=$AUTHORIZATION_GROUP" \
  -e "RTP_ENGRAM_INSTANCE_ID=$INSTANCE_ID" -e "RTP_ENGRAM_SERVICE_ID=$SERVICE_ID" \
  -e "RTP_ENGRAM_MODEL_REVISION=$MODEL_REVISION" -e "RTP_ENGRAM_DATASET_ID=$DATASET_ID" \
  -e RTP_ENGRAM_TLS_CA=/run/engram-tls/ca.crt -e RTP_ENGRAM_TLS_CERT=/run/engram-tls/consumer.crt \
  -e RTP_ENGRAM_TLS_KEY=/run/engram-tls/consumer.key \
  "${extra[@]}" "$IMAGE" sidecar --role consumer --required-memory-bytes "${REQUIRED_MEMORY_BYTES:-1}"
