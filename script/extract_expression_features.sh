#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [[ "${EXPRESSION_POSTFIX+x}" == x ]]; then
    set -- --expression_postfix "$EXPRESSION_POSTFIX" "$@"
fi
"$PYTHON" -m utils.extract_expression_feature --dataset "$DATASET" \
    --anno_root "$ANNO_ROOT" --video_root "$VIDEO_ROOT" \
    --save_dir "$FEATURE_ROOT" --cache_dir "$MODEL_CACHE" \
    --face_detection_dir "$FACE_DETECTION_ROOT" \
    --sample_rate "${EXPRESSION_SAMPLE_RATE:-8}" \
    --batch_size "${BATCH_SIZE:-32}" --device "$DEVICE" "$@"
