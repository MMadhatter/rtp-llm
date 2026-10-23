#!/usr/bin/env bash
set -euo pipefail
# Run inside the image after setting RPC identity/TLS env; returns no handles.
exec /opt/engram-runtime/repo/tools/engram_sidecar/image/entrypoint.sh python -c '
import json
from rtp_llm.models_py.modules.dsv41.engram_rpc import Client
c=Client()
try: print(json.dumps(c.describe(),sort_keys=True))
finally: c.close()
'
