#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Runs locally on the assigned host. Does not SSH or operate C2.
set -euo pipefail
if (( $# < 3 || $# > 4 )); then
  echo 'usage: pouch-pair.sh RUN_DIR INSTANCE sidecar|rtp|status|chat|stop-rtp [MODEL]' >&2
  exit 2
fi
run_dir=$1
instance=$2
action=$3
model=${4:-deepseek_v41}
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
run_step() {
  "${PYTHON:-python3}" "$here/c2_sim.py" "$1" \
    --run-dir "$run_dir" --instance "$instance" --model "$model" --timeout 1800
}
case "$action" in
  sidecar) run_step start-sidecar; run_step wait-owner ;;
  rtp) run_step wait-owner; run_step start-rtp; run_step wait-rtp ;;
  status|chat|stop-rtp) run_step "$action" ;;
  *) echo 'Unknown action; Owner shutdown is deliberately not automated.' >&2; exit 2 ;;
esac
