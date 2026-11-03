#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [[ $# -lt 1 ]]; then
    echo "Usage: bash script/resume_train.sh RUN_DIR [main.py options]" >&2
    exit 2
fi
RUN_DIR="$1"
shift
"$PYTHON" main.py -r "$RUN_DIR" "$@"
