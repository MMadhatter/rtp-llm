#!/usr/bin/env bash
# Copy/edit per deployment. Values come from the trusted platform, not a rack scan.
IMAGE=YOUR_CI_IMAGE_WITH_PLATFORM_ASSIGNMENT
DISCOVERY=platform
OWNER_ENDPOINT=192.0.2.10:19780
AUTHORIZATION_GROUP=tenant-a
SERVICE_ID=engram-shared-group-a
FABRIC_DOMAIN=platform-verified-fabric-partition
MODEL_REVISION=EXACT_MODEL_REVISION
DATASET_ID=sha256:REPLACE_WITH_64_HEX_DIGEST
OWNER_INSTANCE_ID=business-instance-a
# New UUID for each platform assignment / Owner launch attempt.
OWNER_EPOCH=01234567-1234-1234-1234-0123456789ab
PLATFORM_IMEX_CHANNEL=7
STARTUP_TIMEOUT_S=300
CONTAINER_UID=1000
CONTAINER_GID=1000
# ca.crt + consumer.crt/key (CN = this instance ID); Owner also owner.crt/key.
TLS_DIR=/ABSOLUTE_PLATFORM_SECRET_DIRECTORY
