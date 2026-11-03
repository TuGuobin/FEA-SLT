#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
DATASET="${DATASET:-Phoenix14T}"
case "$DATASET" in
    Phoenix14T) CONFIGS=(configs/train.yaml) ;;
    CSL-Daily) CONFIGS=(configs/train.yaml configs/csl_daily.yaml) ;;
    *) echo "Unsupported DATASET: $DATASET" >&2; exit 2 ;;
esac
PYTHON="${PYTHON:-python}"
export ANNO_ROOT="${ANNO_ROOT:-./preprocess/$DATASET}"
export VIDEO_ROOT="${VIDEO_ROOT:-./data/$DATASET}"
export FEATURE_ROOT="${FEATURE_ROOT:-./extracted_features}"
export MODEL_CACHE="${MODEL_CACHE:-./models}"
FACE_DETECTION_ROOT="${FACE_DETECTION_ROOT:-./face_detection_results}"
DEVICE="${DEVICE:-cuda:0}"
