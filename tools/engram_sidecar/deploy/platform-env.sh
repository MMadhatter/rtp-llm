#!/usr/bin/env bash
# Source after the trusted platform config. No discovery or election occurs here.
platform_env=()
IMEX_CHANNEL=${IMEX_CHANNEL:-0}
if [[ -n ${DISCOVERY:-} ]]; then
  [[ $DISCOVERY == platform ]] || { echo 'Unsupported DISCOVERY' >&2; exit 2; }
  [[ -z ${REGISTRY_ENDPOINT:-} ]] || { echo 'Platform assignment cannot use Registry' >&2; exit 2; }
  : "${OWNER_ENDPOINT:?}" "${AUTHORIZATION_GROUP:?}" "${FABRIC_DOMAIN:?}"
  : "${INSTANCE_ID:?}" "${OWNER_INSTANCE_ID:?}" "${OWNER_EPOCH:?}"
  : "${SERVICE_ID:?}" "${MODEL_REVISION:?}" "${DATASET_ID:?}"
  # An explicit channel is mandatory even when its value is zero.
  : "${PLATFORM_IMEX_CHANNEL:?}"
  [[ $PLATFORM_IMEX_CHANNEL =~ ^(0|[1-9][0-9]*)$ ]] || exit 2
  IMEX_CHANNEL=$PLATFORM_IMEX_CHANNEL
  platform_env=(-e RTP_ENGRAM_DISCOVERY=platform
    -e "RTP_ENGRAM_OWNER_ENDPOINT=$OWNER_ENDPOINT"
    -e "RTP_ENGRAM_AUTHORIZATION_GROUP=$AUTHORIZATION_GROUP"
    -e "RTP_ENGRAM_FABRIC_DOMAIN=$FABRIC_DOMAIN"
    -e "RTP_ENGRAM_INSTANCE_ID=$INSTANCE_ID"
    -e "RTP_ENGRAM_OWNER_INSTANCE_ID=$OWNER_INSTANCE_ID"
    -e "RTP_ENGRAM_OWNER_EPOCH=$OWNER_EPOCH"
    -e "RTP_ENGRAM_SERVICE_ID=$SERVICE_ID"
    -e "RTP_ENGRAM_MODEL_REVISION=$MODEL_REVISION"
    -e "RTP_ENGRAM_DATASET_ID=$DATASET_ID"
    -e "RTP_ENGRAM_IMEX_CHANNEL=$IMEX_CHANNEL"
    -e "RTP_ENGRAM_STARTUP_TIMEOUT_S=${STARTUP_TIMEOUT_S:-300}")
fi
IMEX_CHANNEL_DEVICE=/dev/nvidia-caps-imex-channels/channel$IMEX_CHANNEL
