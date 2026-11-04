#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
engram_python="${PYTHON:-python3}"
engram_prefix=rtp_llm.models_py.modules.dsv41.engram_distributed.test
engram_tests=(rtp_llm.models_py.modules.dsv41.engram_distributed.test.test_fabric_ci)
engram_contract=0
for name in exchange publish config manifest sharding storage model_attach placement; do
    engram_tests+=("${engram_prefix}.test_engram_${name}")
done
case "${1:---cpu}" in
    --cpu) ;;
    --torch)
        for name in checkpoint bootstrap model_loading; do
            engram_tests+=("${engram_prefix}.test_engram_${name}")
        done
        engram_tests+=(rtp_llm.models_py.modules.dsv41.engram_distributed.test.test_startup_store)
        engram_contract=1
        ;;
    --cuda-lookup|--cuda-fabric)
        # Owns safe locks, checks real capability and treats all skips as failure.
        exec "$engram_python" -m rtp_llm.models_py.modules.dsv41.engram_distributed.test.run_fabric_ci
        ;;
    *) echo 'Usage: test.sh [--cpu|--torch|--cuda-fabric]' >&2; exit 2 ;;
esac
"$engram_python" -m unittest "${engram_tests[@]}"
if [[ "$engram_contract" == "1" ]]; then
    # This runner initializes its CPU-only Torch fixture in main(); importing it
    # as another unittest module would silently miss that setup contract.
    "$engram_python" rtp_llm/models_py/modules/dsv41/engram_distributed/test/test_embedding_contract.py --repo "$PWD"
fi
