"""Detect faces and save bounding boxes/eye landmarks for aligned expression ROIs."""

import argparse
import os
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils.annotations import DATASETS, SPLITS, dataset_name, load_annotations, load_frames


def detect_and_crop_face(image):
    # Keep TensorFlow optional and CPU-only by default in detector workers.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    from deepface import DeepFace

    # DeepFace accepts OpenCV-style BGR arrays. Shared media loading returns RGB PIL images.
    image_array = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    try:
        face_objects = DeepFace.extract_faces(
            img_path=image_array, detector_backend="retinaface", align=True,
            enforce_detection=True,
        )
    except ValueError as error:
        if "Face could not be detected" in str(error) or "face could not be detected" in str(error):
            return {"status": "no_face", "faces": None}
        raise
    faces = {}
    for index, face in enumerate(face_objects):
        area = face["facial_area"]
        faces[f"face_{index}"] = {
            "bbox": (area["x"], area["y"], area["x"] + area["w"], area["y"] + area["h"]),
            "left_eye": area.get("left_eye"),
            "right_eye": area.get("right_eye"),
            "confidence": face.get("confidence", 1.0),
        }
    return {"status": "success" if faces else "no_face", "faces": faces or None}


def process_and_save_video(task):
    record, name, split, video_root, output_dir, overwrite = task
    path = Path(output_dir) / f"{record['fileid']}.npy"
    if path.exists() and not overwrite:
        return False
    frames = load_frames(name, video_root, record, split)
    results = {index: detect_and_crop_face(frame) for index, frame in enumerate(frames)}
    result = {"fileid": record["fileid"], "split": split, "results": results}
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as output:
        temporary = Path(output.name)
        try:
            np.save(output, result, allow_pickle=True)
            output.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--anno_root", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--output_dir", default="./face_detection_results")
    parser.add_argument("--mode", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing detections.")
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("num_workers must be positive.")
    name = dataset_name(args.anno_root, args.dataset)
    for split in args.mode:
        data = load_annotations(Path(args.anno_root) / f"{split}_ml.npy", split)
        output_dir = Path(args.output_dir) / name / split
        output_dir.mkdir(parents=True, exist_ok=True)
        # Pass one record per task rather than copying the entire corpus into every worker.
        tasks = ((record, name, split, args.video_root, str(output_dir), args.overwrite)
                 for record in data.values())
        if args.num_workers == 1:
            processed = sum(tqdm(map(process_and_save_video, tasks), total=len(data), desc=f"{name} {split}"))
        else:
            with Pool(args.num_workers) as pool:
                processed = sum(tqdm(pool.imap(process_and_save_video, tasks),
                                     total=len(data), desc=f"{name} {split}"))
        print(f"Saved {processed} detection files to {output_dir}.")


if __name__ == "__main__":
    main()
