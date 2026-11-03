#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
"$PYTHON" -m utils.prepare_annotations --dataset "$DATASET" \
    --video_root "$VIDEO_ROOT" --output_dir "$ANNO_ROOT" "$@"
