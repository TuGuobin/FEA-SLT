#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
"$PYTHON" -m utils.crop_faces --dataset "$DATASET" \
    --anno_root "$ANNO_ROOT" --video_root "$VIDEO_ROOT" \
    --output_dir "$FACE_DETECTION_ROOT" --num_workers "${NUM_WORKERS:-1}" "$@"
