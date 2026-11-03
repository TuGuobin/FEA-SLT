"""Extract aligned facial-expression features at the paper's fixed frame interval."""

import math
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import ViTForImageClassification, ViTImageProcessor

from utils.feature_pipeline import batch_features, feature_parser, run_extraction, validate_postfix


class ExpressionFeatureReader:
    def __init__(self, model_name="trpakov/vit-face-expression", cache_dir=None,
                 device="cuda:0", nth_layer=-1):
        self.device = device
        self.nth_layer = nth_layer
        self.model = ViTForImageClassification.from_pretrained(
            model_name, output_hidden_states=True, cache_dir=cache_dir
        ).to(device).eval()
        self.feature_dim = self.model.config.hidden_size
        self.image_processor = ViTImageProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )

    @torch.no_grad()
    def get_feats(self, frames):
        inputs = self.image_processor(frames, return_tensors="pt").to(self.device).pixel_values
        return self.model(inputs).hidden_states[self.nth_layer][:, 0]


def distance_weighted_interpolate_features(valid_feats, valid_mask):
    """Inverse-distance interpolation (epsilon=1e-6); nearest at boundaries."""
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) != len(valid_feats) or not len(valid_indices):
        raise ValueError("Expected one feature per valid frame and at least one valid frame.")
    positions = np.arange(len(valid_mask))
    right = np.searchsorted(valid_indices, positions).clip(0, len(valid_indices) - 1)
    left = (right - 1).clip(0, len(valid_indices) - 1)
    left_weights = 1 / (np.abs(positions - valid_indices[left]) + 1e-6)
    right_weights = 1 / (np.abs(positions - valid_indices[right]) + 1e-6)
    weights = right_weights / (left_weights + right_weights)
    features = valid_feats[left] * (1 - weights[:, None]) + valid_feats[right] * weights[:, None]
    features[:valid_indices[0]] = valid_feats[0]
    features[valid_indices[-1] + 1:] = valid_feats[-1]
    features[valid_indices] = valid_feats
    return features.astype(np.float32, copy=False)


def aligned_face_crop(frame, face):
    """Rotate the image using RetinaFace eyes, then project and crop the ROI."""
    x1, y1, x2, y2 = face["bbox"]
    x1, x2 = max(0, int(x1)), min(frame.width, int(x2))
    y1, y2 = max(0, int(y1)), min(frame.height, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    if "left_eye" not in face or "right_eye" not in face:
        raise ValueError("Detection lacks eye landmarks; regenerate it with script/crop_faces.sh.")
    left_eye, right_eye = face["left_eye"], face["right_eye"]
    if left_eye is None or right_eye is None:
        return frame.crop((x1, y1, x2, y2))
    angle = math.atan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])
    pad_x, pad_y = frame.width // 2, frame.height // 2
    padded = ImageOps.expand(frame, border=(pad_x, pad_y), fill=0)
    center_x, center_y = padded.width / 2, padded.height / 2
    dx = (x1 + x2) / 2 + pad_x - center_x
    dy = (y1 + y2) / 2 + pad_y - center_y
    roi_x = center_x + math.cos(angle) * dx + math.sin(angle) * dy
    roi_y = center_y - math.sin(angle) * dx + math.cos(angle) * dy
    aligned = padded.rotate(math.degrees(angle), resample=Image.Resampling.BILINEAR)
    width, height = x2 - x1, y2 - y1
    return aligned.crop((round(roi_x - width / 2), round(roi_y - height / 2),
                         round(roi_x + width / 2), round(roi_y + height / 2)))


def expression_features(reader, frames, detection_file, batch_size, fileid, sample_rate=8):
    if sample_rate < 1:
        raise ValueError("sample_rate must be positive.")
    if not detection_file.is_file():
        raise FileNotFoundError(f"{detection_file}: run bash script/crop_faces.sh first.")
    detection = np.load(detection_file, allow_pickle=True).item()
    results = detection["results"]
    if detection.get("fileid") != fileid or set(results) != set(range(len(frames))):
        raise ValueError(f"{detection_file}: detection IDs/frame count do not match the video.")
    face_frames = []
    indices = list(range(0, len(frames), sample_rate))
    mask = np.zeros(len(indices), dtype=bool)
    for sampled_index, index in enumerate(indices):
        frame = frames[index]
        result = results[index]
        if result["status"] == "error":
            raise ValueError(f"{detection_file}, frame {index}: {result.get('error')}")
        if result["status"] != "success" or not result.get("faces"):
            continue
        # The releases contain a signer in the foreground; use the largest face.
        face = max(result["faces"].values(),
                   key=lambda face: (face["bbox"][2] - face["bbox"][0])
                   * (face["bbox"][3] - face["bbox"][1]))
        roi = aligned_face_crop(frame, face)
        if roi is None:
            continue
        face_frames.append(roi)
        mask[sampled_index] = True
    if not face_frames:
        warnings.warn(f"{fileid}: no valid face detected; saving zero expression features.")
        return np.zeros((len(indices), reader.feature_dim), dtype=np.float32)
    valid_feats = batch_features(reader, face_frames, batch_size)
    return distance_weighted_interpolate_features(valid_feats, mask)


def main():
    parser = feature_parser(__doc__, "trpakov/vit-face-expression")
    parser.add_argument("--face_detection_dir", default="./face_detection_results")
    parser.add_argument("--sample_rate", type=int, default=8, help="Extract frames 0, s, 2s, ...")
    parser.add_argument("--expression_postfix", default=None,
                        help="Filename suffix; default: _stride-<sample_rate>.")
    args = parser.parse_args()
    if args.sample_rate < 1:
        parser.error("sample_rate must be positive.")
    postfix = (f"_stride-{args.sample_rate}" if args.expression_postfix is None
               else validate_postfix(args.expression_postfix))

    def process(reader, frames, record, name, split):
        path = Path(args.face_detection_dir) / name / split / f"{record['fileid']}.npy"
        return expression_features(reader, frames, path, args.batch_size, record["fileid"],
                                   args.sample_rate)

    run_extraction(
        args, "expression",
        lambda: ExpressionFeatureReader(args.model_name, args.cache_dir, args.device, args.nth_layer),
        process, postfix,
    )


if __name__ == "__main__":
    main()
