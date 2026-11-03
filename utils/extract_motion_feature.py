"""Extract VideoMAE features from overlapping 16-frame windows."""

import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel

from utils.feature_pipeline import batch_features, feature_parser, run_extraction
from utils.helpers import sliding_window_for_list


class VideoMAEFeatureReader:
    def __init__(self, model_name="MCG-NJU/videomae-large", cache_dir=None,
                 device="cuda:0", nth_layer=-1):
        self.device = device
        self.nth_layer = nth_layer
        self.image_processor = VideoMAEImageProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = VideoMAEModel.from_pretrained(model_name, cache_dir=cache_dir).to(device).eval()

    @torch.no_grad()
    def get_feats(self, videos):
        inputs = self.image_processor(images=videos, return_tensors="pt").to(self.device)
        return self.model(**inputs, output_hidden_states=True).hidden_states[self.nth_layer][:, 0]


def main():
    parser = feature_parser(__doc__, "MCG-NJU/videomae-large", batch_size=16)
    parser.add_argument("--overlap_size", type=int, default=8)
    args = parser.parse_args()
    if not 0 <= args.overlap_size < 16:
        parser.error("overlap_size must be in [0, 15].")

    def process(reader, frames, *_):
        if len(frames) < 16:
            frames = frames + [frames[-1]] * (16 - len(frames))
        windows = sliding_window_for_list(frames, 16, args.overlap_size)
        return batch_features(reader, windows, args.batch_size)

    run_extraction(
        args, "motion",
        lambda: VideoMAEFeatureReader(args.model_name, args.cache_dir, args.device, args.nth_layer),
        process, f"_overlap-{args.overlap_size}",
    )


if __name__ == "__main__":
    main()
