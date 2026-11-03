#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
"$PYTHON" -m utils.extract_spatial_feature --dataset "$DATASET" \
    --anno_root "$ANNO_ROOT" --video_root "$VIDEO_ROOT" \
    --save_dir "$FEATURE_ROOT" --cache_dir "$MODEL_CACHE" \
    --s2_mode s2wrapping --scales 1 2 --batch_size "${BATCH_SIZE:-32}" \
    --device "$DEVICE" "$@"
