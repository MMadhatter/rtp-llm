#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/platform-common.example.sh"
# Set to business-instance-a for the producer's accompanying RTP process.
INSTANCE_ID=business-instance-b
NAME=rtp-business-b
CHECKPOINT=/ABSOLUTE_CHECKPOINT
# Existing P/D model, routing and port settings; do not mix Engram discovery here.
MODEL_ENV=/ABSOLUTE_MODEL_ENV_FILE
GPU_UUIDS=GPU-PLATFORM_ALLOCATED_DEVICE
MEMORY_LIMIT=300g
SHM_SIZE=32g
