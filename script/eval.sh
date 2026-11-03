#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [[ $# -lt 1 ]]; then
    echo "Usage: bash script/eval.sh CHECKPOINT [main.py options]" >&2
    exit 2
fi
CHECKPOINT_PATH="$1"
shift
# Prefer the configuration snapshots saved alongside this checkpoint.
"$PYTHON" main.py --test --ckpt "$CHECKPOINT_PATH" "$@"
