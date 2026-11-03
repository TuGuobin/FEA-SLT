"""Shared annotation schema and media loading for locally prepared datasets."""

from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

DATASETS = ("Phoenix14T", "CSL-Daily")
SPLITS = ("train", "dev", "test")


def validate_record(record, split=None):
    if not isinstance(record, dict):
        raise ValueError("Each annotation must be a dictionary.")
    fileid = record.get("fileid")
    if (not isinstance(fileid, str) or not fileid.strip()
            or fileid in (".", "..") or any(c in fileid for c in "/\\\0")):
        raise ValueError(f"Invalid fileid (expected a single filename): {fileid!r}")
    if not isinstance(record.get("text"), str) or not record["text"].strip():
        raise ValueError(f"{fileid}: text must be a nonempty string.")
    if not isinstance(record.get("gloss", ""), str):
        raise ValueError(f"{fileid}: gloss must be a string.")
    if split is not None and record.get("split", split) != split:
        raise ValueError(f"{fileid}: annotation split does not match {split}.")
    for lang in ("en", "es", "fr", "de"):
        if f"{lang}_text" in record and not isinstance(record[f"{lang}_text"], str):
            raise ValueError(f"{fileid}: {lang}_text must be a string.")
    folder = record.get("folder")
    if folder is not None:
        path = PurePosixPath(folder)
        if path.is_absolute() or ".." in path.parts or "\\" in folder:
            raise ValueError(f"{fileid}: folder must be a relative POSIX path.")
    return record


def load_annotations(path, split=None):
    """Load our integer-keyed .npy dictionary. Only load trusted local files."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {path}. Run python -m utils.prepare_annotations."
        )
    data = np.load(path, allow_pickle=True).item()
    if not isinstance(data, dict) or set(data) != set(range(len(data))):
        raise ValueError(f"{path}: expected contiguous integer keys 0..N-1.")
    if not data:
        raise ValueError(f"{path}: annotations are empty.")
    ids = set()
    for i in range(len(data)):
        record = validate_record(data[i], split)
        if record["fileid"] in ids:
            raise ValueError(f"{path}: duplicate fileid {record['fileid']}.")
        ids.add(record["fileid"])
    return data


def dataset_name(anno_root, explicit=None):
    name = explicit or Path(anno_root).name
    if name not in DATASETS:
        raise ValueError(f"Unsupported dataset {name!r}; choose one of {DATASETS}.")
    return name


def frame_paths(video_root, record, split):
    folder = record.get("folder", f"{split}/{record['fileid']}/*.png")
    validate_record(dict(record, folder=folder), split)
    root = Path(video_root) / "PHOENIX-2014-T/features/fullFrame-210x260px"
    paths = sorted(root.glob(folder))
    if not paths:
        raise FileNotFoundError(f"{record['fileid']}: no frames matching {root / folder}")
    return paths


def read_video(path, start_time=None, end_time=None, fps=None):
    import av

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    start_time = 0.0 if start_time is None else start_time
    if start_time < 0 or (end_time is not None and end_time <= start_time):
        raise ValueError("Expected 0 <= start_time < end_time.")
    if fps is not None and fps <= 0:
        raise ValueError("fps must be positive.")
    frames = []
    timestamps = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {path}")
        stream = container.streams.video[0]
        native_fps = float(stream.average_rate or stream.guessed_rate or 30)
        for index, frame in enumerate(container.decode(stream)):
            timestamp = frame.time if frame.time is not None else index / native_fps
            if timestamp < start_time:
                continue
            if end_time is not None and timestamp >= end_time:
                break
            frames.append(frame.to_image().convert("RGB"))
            timestamps.append(timestamp)
    if not frames:
        raise ValueError(f"No decodable frames in {path}")
    if fps is not None:
        # Zero-order hold on presentation timestamps: every stream/detector sees identical frames.
        duration = timestamps[-1] - timestamps[0] + 1 / native_fps
        if end_time is not None:
            duration = min(duration, end_time - timestamps[0])
        count = max(1, int(np.ceil(duration * fps - 1e-6)))
        grid = timestamps[0] + np.arange(count) / fps
        indices = np.searchsorted(timestamps, grid, side="right") - 1
        frames = [frames[max(0, index)] for index in indices]
    return frames


def load_frames(name, video_root, record, split):
    if name == "Phoenix14T":
        frames = []
        for path in frame_paths(video_root, record, split):
            with Image.open(path) as image:
                frames.append(image.convert("RGB"))
        return frames
    if name == "CSL-Daily":
        return read_video(Path(video_root) / "sentence-crop" / f"{record['fileid']}.mp4", fps=30)
    raise ValueError(f"Unsupported dataset: {name}")
