"""Extract per-frame CLIP features, optionally at multiple S2 scales."""

import torch
from transformers import AutoImageProcessor, CLIPVisionModel

from utils.feature_pipeline import batch_features, feature_parser, run_extraction
from utils.s2wrapper import forward as multiscale_forward


class ViTFeatureReader:
    def __init__(self, model_name="openai/clip-vit-large-patch14", cache_dir=None,
                 device="cuda:0", s2_mode="", scales=None, nth_layer=-1):
        self.device = device
        self.s2_mode = s2_mode
        self.scales = scales or [1, 2]
        self.nth_layer = nth_layer
        self.model = CLIPVisionModel.from_pretrained(
            model_name, output_hidden_states=True, cache_dir=cache_dir
        ).to(device).eval()
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )

    @torch.no_grad()
    def forward_features(self, inputs):
        return self.model(inputs).hidden_states[self.nth_layer]

    @torch.no_grad()
    def get_feats(self, frames):
        inputs = self.image_processor(frames, return_tensors="pt").to(self.device).pixel_values
        if self.s2_mode == "s2wrapping":
            outputs = multiscale_forward(
                self.forward_features, inputs, scales=self.scales, num_prefix_token=1
            )
        else:
            outputs = self.forward_features(inputs)
        return outputs[:, 0]


def main():
    parser = feature_parser(__doc__, "openai/clip-vit-large-patch14")
    parser.add_argument("--s2_mode", choices=("", "s2wrapping"), default="")
    parser.add_argument("--scales", nargs="+", type=int, default=[1, 2])
    args = parser.parse_args()
    if any(scale < 1 for scale in args.scales):
        parser.error("scales must be positive integers.")
    postfix = "_s2wrapping" if args.s2_mode else ""
    if args.s2_mode and len(args.scales) == 3:
        postfix += "_large"
    run_extraction(
        args, "spatial",
        lambda: ViTFeatureReader(args.model_name, args.cache_dir, args.device,
                                 args.s2_mode, args.scales, args.nth_layer),
        lambda reader, frames, *_: batch_features(reader, frames, args.batch_size),
        postfix,
    )


if __name__ == "__main__":
    main()
