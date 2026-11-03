"""Convert official labels (or normalized JSONL) into local training annotations."""

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from utils.annotations import DATASETS, SPLITS, frame_paths, validate_record


def normalize_split(value):
    split = {"val": "dev", "valid": "dev", "validation": "dev"}.get(value, value)
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {value!r}")
    return split


def joined(value, separator=" "):
    if isinstance(value, (list, tuple)):
        return separator.join(str(word) for word in value)
    return str(value)


def phoenix_records(root):
    annotation_root = root / "PHOENIX-2014-T/annotations/manual"
    for split in SPLITS:
        path = annotation_root / f"PHOENIX-2014-T.{split}.corpus.csv"
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, delimiter="|")
            required = {"name", "video", "orth", "translation"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{path}: expected columns {sorted(required)}.")
            for row in reader:
                folder = row["video"].strip()
                # The official corpus may store a path relative to the split.
                if not folder.startswith(f"{split}/"):
                    folder = f"{split}/{folder}"
                if not any(c in folder for c in "*?["):
                    folder = folder.rstrip("/") + "/*.png"
                record = {
                    "fileid": row["name"].strip(), "split": split,
                    "folder": folder, "text": row["translation"].strip(),
                    "gloss": row["orth"].strip(), "signer": row.get("speaker", ""),
                }
                yield record


def csl_records(label_path, split_path):
    # Official labels are pickle files. Do not load untrusted pickle files.
    with label_path.open("rb") as source:
        labels = pickle.load(source)
    info = labels.get("info") if isinstance(labels, dict) else labels
    if not isinstance(info, list):
        raise ValueError("CSL labels must be a list or a dictionary with an 'info' list.")
    split_map = {}
    with split_path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            line = line.strip()
            if not line:
                continue
            # Common release format: name|split; accept tab or whitespace too.
            cells = line.split("|") if "|" in line else line.split()
            cells = [cell.strip() for cell in cells]
            if len(cells) != 2:
                raise ValueError(f"{split_path}:{line_number}: expected name and split.")
            if cells[0].lower() == "name" and cells[1].lower() == "split":
                continue
            name, split = cells
            if name in split_map:
                raise ValueError(f"Duplicate CSL split entry: {name}")
            split_map[name] = normalize_split(split)
    used = set()
    for row in info:
        name = str(row["name"])
        if name not in split_map:
            raise ValueError(f"No split assignment for CSL sample {name}.")
        used.add(name)
        yield {
            "fileid": name, "split": split_map[name],
            "text": joined(row["word"], ""), "gloss": joined(row.get("label", "")),
            "num_frames": int(row.get("length", 0)),
        }
    unknown = set(split_map) - used
    if unknown:
        raise ValueError(f"Split entries absent from labels: {sorted(unknown)[:5]}")


def jsonl_records(path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "split" not in record:
                raise ValueError(f"{path}:{line_number}: missing split.")
            record["split"] = normalize_split(record["split"])
            record.setdefault("gloss", "")
            yield record


def prepare(args):
    if args.jsonl:
        records = jsonl_records(args.jsonl)
    elif args.dataset == "Phoenix14T":
        records = phoenix_records(args.video_root)
    else:
        label_path = args.labels or args.video_root / "csl_daily_label.pkl"
        split_path = args.split_file or args.video_root / "csl_daily_split.txt"
        records = csl_records(label_path, split_path)

    translations = {}
    if args.translations:
        with args.translations.open(encoding="utf-8") as source:
            translations = json.load(source)
        if not isinstance(translations, dict):
            raise ValueError("Translations must be a JSON object keyed by fileid.")
    splits = {split: {} for split in SPLITS}
    seen = set()
    for original in records:
        record = dict(original)
        split = normalize_split(record["split"])
        validate_record(record, split)
        fileid = record["fileid"]
        if fileid in seen:
            raise ValueError(f"Duplicate fileid across annotations/splits: {fileid}")
        seen.add(fileid)
        if fileid in translations:
            extra = translations[fileid]
            if not isinstance(extra, dict) or not set(extra).issubset(
                    {"en_text", "es_text", "fr_text", "de_text"}):
                raise ValueError(f"{fileid}: translations must contain only *_text fields.")
            record.update(extra)
            validate_record(record, split)
        if args.dataset == "Phoenix14T":
            record.setdefault("folder", f"{split}/{fileid}/*.png")
            if args.check_media:
                record["num_frames"] = len(frame_paths(args.video_root, record, split))
        elif args.check_media:
            path = args.video_root / "sentence-crop" / f"{fileid}.mp4"
            if not path.is_file():
                raise FileNotFoundError(path)
        splits[split][len(splits[split])] = record
    if set(translations) - seen:
        raise ValueError("Translations contain fileids absent from the labels.")
    if any(not records for records in splits.values()):
        raise ValueError("All three official splits (train/dev/test) must be nonempty.")

    output = args.output_dir or Path("preprocess") / args.dataset
    paths = [output / f"{split}_ml.npy" for split in SPLITS]
    if not args.overwrite and any(path.exists() for path in paths):
        raise FileExistsError(f"{output}: annotations already exist; use --overwrite explicitly.")
    output.mkdir(parents=True, exist_ok=True)
    for split, path in zip(SPLITS, paths):
        np.save(path, splits[split], allow_pickle=True)
        print(f"{split}: {len(splits[split])} samples -> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--video_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--labels", type=Path, help="CSL label pickle (trusted official source).")
    parser.add_argument("--split_file", type=Path, help="CSL name|split file.")
    parser.add_argument("--jsonl", type=Path, help="Normalized records; bypass official label adapters.")
    parser.add_argument("--translations", type=Path, help="Optional fileid-keyed translations JSON.")
    parser.add_argument("--check_media", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
