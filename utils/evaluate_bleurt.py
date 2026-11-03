import json
import numpy as np
from pathlib import Path
import os
import logging
import argparse
from tqdm import tqdm

from utils.text import normalize_text

logger = logging.getLogger(__name__)


class BLEURTEvaluator:
    def __init__(self, model_path: str = "BLEURT-20"):
        self.model_path = Path(model_path).absolute()
        self.scorer = None
        self.load_model()

    def load_model(self):
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"BLEURT checkpoint directory not found: {self.model_path}")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false")
        from bleurt.score import BleurtScorer

        self.scorer = BleurtScorer(str(self.model_path))
        logger.info("BLEURT model loaded successfully")

    def load_data(self, file_path: str):
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        logger.info(f"Loading data from: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "references" not in data or "predictions" not in data:
            raise ValueError("JSON must contain 'references' and 'predictions' arrays")

        refs = data["references"]
        preds = data["predictions"]

        if len(refs) != len(preds):
            raise ValueError(
                f"Arrays have different lengths: references={len(refs)}, predictions={len(preds)}"
            )

        if len(refs) == 0:
            raise ValueError("Arrays are empty")
        if not all(isinstance(text, str) for text in refs + preds):
            raise ValueError("References and predictions must contain strings.")
        language = data.get("language") or (
            "chinese" if data.get("is_chinese") or
            any("\u4e00" <= char <= "\u9fff" for text in refs for char in text) else "german")
        refs = [normalize_text(text, language) for text in refs]
        preds = [normalize_text(text, language) for text in preds]

        logger.info(f"Data loaded successfully: {len(refs)} pairs")
        return refs, preds

    def calculate_scores(self, references, predictions):
        logger.info("Calculating scores...")
        scores = []
        for ref, pred in tqdm(
            zip(references, predictions),
            total=len(references),
            desc="Calculating BLEURT scores",
        ):
            score = self.scorer.score(references=[ref], candidates=[pred])
            scores.extend(float(value) * 100 for value in score)
        logger.info("Scores calculated successfully")

        return scores

    def calculate_statistics(self, scores):
        """
        Calculate basic statistics.
        """
        if not scores:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "count": 0}

        stats = {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "count": len(scores),
        }

        logger.info(f"Statistics: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        return stats

    def save_results(self, results, output_json: str):
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to: {output_json}")

    def process_file(
        self,
        input_file: str,
        output_json: str = "bleurt_results.json",
    ):
        references, predictions = self.load_data(input_file)
        scores = self.calculate_scores(references, predictions)
        stats = self.calculate_statistics(scores)

        individual_scores = []
        for i, (ref, pred, score_val) in enumerate(
            zip(references, predictions, scores)
        ):
            individual_scores.append(
                {
                    "index": i,
                    "reference": ref,
                    "prediction": pred,
                    "score": score_val,
                }
            )

        results = {
            "input_file": input_file,
            "statistics": stats,
            "individual_scores": individual_scores,
            "total_pairs": len(scores),
            "scale": 100,
        }

        self.save_results(results, output_json)
        logger.info("Processing completed successfully")

        return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(
        description="Calculate BLEURT scores for translation results"
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to input JSON file"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to output JSON file (default: input_file_bleurt.json)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="BLEURT-20",
        help="Path to BLEURT model (default: BLEURT-20)",
    )

    args = parser.parse_args()

    input_file = args.input_file
    model_path = args.model_path

    if args.output_json is None:
        input_path = Path(input_file)
        output_json = str(input_path.with_suffix("")) + "_bleurt.json"
    else:
        output_json = args.output_json

    evaluator = BLEURTEvaluator(model_path=model_path)
    results = evaluator.process_file(input_file, output_json)

    print("\n" + "=" * 60)
    print(f"Input file: {results['input_file']}")
    print(f"Total pairs evaluated: {results['total_pairs']}")
    print(f"Mean score: {results['statistics']['mean']:.4f}")
    print(f"Standard deviation: {results['statistics']['std']:.4f}")
    print(f"Results saved to: {output_json}")
    print("=" * 60)


if __name__ == "__main__":
    main()
