#!/usr/bin/env bash
set -euo pipefail
url=${1:?usage: chat.sh PREFILL_HTTP_URL [MODEL]}
model=${2:-deepseek_v41}
body=$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":"你好，请用一句话介绍自己。"}],"max_tokens":32,"temperature":0,"thinking":{"type":"disabled"}}))' "$model")
curl --fail-with-body -sS -m 90 "$url/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "$body" | python3 -c '
import json,sys
r=json.load(sys.stdin)
choices=r.get("choices",[])
text=choices[0].get("message",{}).get("content","") if choices else ""
if not isinstance(text,str) or not text.strip():
    print(json.dumps(r,ensure_ascii=False)); raise SystemExit("No nonempty chat answer")
print(text)
'
