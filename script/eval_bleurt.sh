#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [[ $# -lt 1 ]]; then
    echo "Usage: bash script/eval_bleurt.sh RESULTS_JSON [BLEURT options]" >&2
    exit 2
fi
INPUT_FILE="$1"
shift
"$PYTHON" -m utils.evaluate_bleurt --input_file "$INPUT_FILE" \
    --model_path "${BLEURT_MODEL:-BLEURT-20}" "$@"
