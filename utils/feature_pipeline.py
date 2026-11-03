"""Common CLI and complete-sample iteration for feature extraction."""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils.annotations import DATASETS, SPLITS, dataset_name, load_annotations, load_frames


def feature_parser(description, model_name, batch_size=32):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--anno_root", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--model_name", default=model_name)
    parser.add_argument("--cache_dir", default="./models")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--nth_layer", type=int, default=-1)
    parser.add_argument("--mode", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--start", type=int, default=0, help="Number of samples to skip per split.")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing feature files.")
    return parser


def validate_features(features, path):
    if (features.ndim != 2 or 0 in features.shape
            or not np.issubdtype(features.dtype, np.floating)
            or not np.isfinite(features).all()):
        raise ValueError(f"{path}: expected finite floating features with shape [T, D], T,D > 0.")
    return features


def validate_postfix(postfix):
    if not isinstance(postfix, str) or any(char in postfix for char in "/\\\0"):
        raise ValueError("Feature postfix must be a filename suffix without path separators.")
    return postfix


def save_features(path, features):
    validate_features(features, path)
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as output:
        temporary = Path(output.name)
        try:
            np.save(output, features, allow_pickle=False)
            output.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def run_extraction(args, stream, reader_factory, process, postfix=""):
    if args.batch_size < 1 or args.start < 0:
        raise ValueError("batch_size must be positive and start must be nonnegative.")
    validate_postfix(postfix)
    name = dataset_name(args.anno_root, args.dataset)
    reader = None
    for split in args.mode:
        data = load_annotations(Path(args.anno_root) / f"{split}_ml.npy", split)
        output_dir = Path(args.save_dir) / f"{stream}_feat_{name}" / split
        output_dir.mkdir(parents=True, exist_ok=True)
        indices = list(range(len(data)))
        if args.reverse:
            indices.reverse()
        for index in tqdm(indices[args.start:], desc=f"{name} {split} {stream}"):
            record = data[index]
            path = output_dir / f"{record['fileid']}{postfix}.npy"
            if path.exists() and not getattr(args, "overwrite", False):
                validate_features(np.load(path, allow_pickle=False), path)
                continue
            frames = load_frames(name, args.video_root, record, split)
            if reader is None:
                reader = reader_factory()
            features = process(reader, frames, record, name, split)
            save_features(path, features.astype(np.float32, copy=False))


def batch_features(reader, frames, batch_size):
    return np.concatenate([
        reader.get_feats(frames[start:start + batch_size]).cpu().numpy()
        for start in range(0, len(frames), batch_size)
    ], axis=0)
