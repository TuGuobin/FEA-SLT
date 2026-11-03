#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
"$PYTHON" main.py -c "${CONFIGS[@]}" -e bleu "$@"
