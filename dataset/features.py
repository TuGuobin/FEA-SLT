"""Feature-backed dataset shared by Phoenix14T and CSL-Daily."""

from pathlib import Path

import numpy as np
import torch

from utils.annotations import load_annotations
from utils.feature_pipeline import validate_features, validate_postfix


class FeatureDataset(torch.utils.data.Dataset):
    language = ""
    dataset_name = ""

    def __init__(self, anno_root, vid_root="", feat_root=None, mae_feat_root=None,
                 exp_feat_root=None, mode="dev", spatial=False, spatiotemporal=False,
                 expression=False, spatial_postfix="", spatiotemporal_postfix="",
                 expression_postfix="", expression_sample_rate=1, spatial_global_only=False):
        if not (spatial or spatiotemporal or expression):
            raise ValueError("Enable at least one feature stream.")
        if mode not in ("train", "dev", "test"):
            raise ValueError(f"Unknown dataset split: {mode}")
        self.vid_root = Path(vid_root)
        self.mode = mode
        self.spatial = spatial
        self.spatiotemporal = spatiotemporal
        self.expression = expression
        self.spatial_global_only = spatial_global_only
        self.spatial_postfix = validate_postfix(spatial_postfix)
        self.spatiotemporal_postfix = spatiotemporal_postfix
        self.expression_postfix = validate_postfix(expression_postfix)
        if expression_sample_rate < 1:
            raise ValueError("expression_sample_rate must be positive.")
        self.expression_sample_rate = expression_sample_rate
        self.data = load_annotations(Path(anno_root) / f"{mode}_ml.npy", mode)
        self.spatial_dir = self._feature_dir(feat_root, spatial, "spatial")
        self.spatiotemporal_dir = self._feature_dir(mae_feat_root, spatiotemporal, "motion")
        self.exp_feat_dir = self._feature_dir(exp_feat_root, expression, "expression")
        if spatiotemporal and not isinstance(spatiotemporal_postfix, str) and not spatiotemporal_postfix:
            raise ValueError("spatiotemporal_postfix list must not be empty.")
        for suffix in ([spatiotemporal_postfix] if isinstance(spatiotemporal_postfix, str)
                       else spatiotemporal_postfix):
            validate_postfix(suffix)

    def _feature_dir(self, root, enabled, name):
        if not enabled:
            return None
        if root is None:
            raise ValueError(f"Set the {name} feature root for the enabled stream.")
        path = Path(root) / self.mode
        if not path.is_dir():
            raise FileNotFoundError(f"{name} feature directory not found: {path}. Run feature extraction first.")
        return path

    @staticmethod
    def _load(directory, fileid, postfix):
        path = directory / f"{fileid}{postfix}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Feature file not found: {path}. Complete feature extraction first.")
        array = validate_features(np.load(path, allow_pickle=False), path)
        return torch.from_numpy(array.astype(np.float32, copy=False))

    def __getitem__(self, index):
        record = self.data[index]
        fileid = record["fileid"]
        spatial = self._load(self.spatial_dir, fileid, self.spatial_postfix) if self.spatial else None
        expression = self._load(self.exp_feat_dir, fileid, self.expression_postfix) if self.expression else None
        motion = None
        if self.spatiotemporal:
            if isinstance(self.spatiotemporal_postfix, str):
                motion = self._load(self.spatiotemporal_dir, fileid, self.spatiotemporal_postfix)
            else:
                motion = [self._load(self.spatiotemporal_dir, fileid, suffix)
                          for suffix in self.spatiotemporal_postfix]
        if spatial is not None and expression is not None:
            expected = (len(spatial) + self.expression_sample_rate - 1) // self.expression_sample_rate
            if len(expression) != expected:
                raise ValueError(f"{fileid}: spatial/expression counts do not match the extraction "
                                 f"interval {self.expression_sample_rate}; regenerate features.")
        if spatial is not None and self.spatial_global_only:
            if spatial.shape[1] < 1024:
                raise ValueError(f"{fileid}: global CLIP features require at least 1024 dimensions.")
            spatial = spatial[:, :1024]
        num_frames = (len(spatial) if spatial is not None else
                      (len(expression) - 1) * self.expression_sample_rate + 1
                      if expression is not None else record.get("num_frames", 0))
        result = {
            "pixel_value": spatial, "glor_value": motion, "exp_value": expression,
            "bool_mask_pos": None, "text": self._normalize_text(record["text"]),
            "gloss": record.get("gloss", ""), "id": fileid,
            "num_frames": num_frames,
            "expression_sample_rate": self.expression_sample_rate,
            "lang": self.language, "original_info": record,
        }
        if self.dataset_name == "Phoenix14T":
            result["vid_path"] = str(self.vid_root / "PHOENIX-2014-T/features/fullFrame-210x260px"
                                     / self.mode / fileid)
        else:
            result["vid_path"] = str(self.vid_root / "sentence-crop" / f"{fileid}.mp4")
        return result

    def _normalize_text(self, text):
        return text.strip()

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        return batch
