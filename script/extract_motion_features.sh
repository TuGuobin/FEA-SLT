#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
"$PYTHON" -m utils.extract_motion_feature --dataset "$DATASET" \
    --anno_root "$ANNO_ROOT" --video_root "$VIDEO_ROOT" \
    --save_dir "$FEATURE_ROOT" --cache_dir "$MODEL_CACHE" \
    --overlap_size 8 --batch_size "${BATCH_SIZE:-16}" --device "$DEVICE" "$@"
