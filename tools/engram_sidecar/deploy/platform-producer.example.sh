#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/platform-common.example.sh"
SIDECAR_ROLE=producer
INSTANCE_ID=business-instance-a
NAME=engram-producer-a
OWNER_IP=192.0.2.10
OWNER_PORT=19780
CHECKPOINT=/ABSOLUTE_CHECKPOINT
STATE_DIR=/ABSOLUTE_NEW_STATE_DIRECTORY
GPU_UUID=GPU-PLATFORM_RESERVED_INITIALIZATION_DEVICE
NUMA=0
RESERVATION_ID=platform-reservation-a
# Illustrative limits: platform must size these using the dataset's real footprint.
RESERVED_MEMORY_BYTES=274877906944
REQUIRED_MEMORY_BYTES=137438953472
MEMORY_LIMIT=300g
ALLOWED_CLIENT_NAMES='business-instance-a business-instance-b'
