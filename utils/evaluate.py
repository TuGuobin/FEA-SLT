from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
import json
import os
from datetime import datetime
from utils.text import normalize_text


def chinese_tokenizer(text):
    return "".join(text.split())


def tokenize_with_punctuation(text):
    return normalize_text(text, "german").split()


class PaperTokenizer:
    """Preserve German letters and Chinese punctuation in ROUGE-L."""

    def __init__(self, language):
        self.language = language

    def tokenize(self, text):
        return list(text) if self.language == "chinese" else text.split()

def rouge_l(prediction, reference, language="auto"):
    """
    Universal ROUGE-L implementation for both Chinese and English text.
    This calculates the longest common subsequence (LCS) based ROUGE score.

    Args:
        prediction (str): Predicted text
        reference (str): Reference text
        language (str): Language type ('chinese', 'english', or 'auto')

    Returns:
        tuple: (precision, recall, f1) scores
    """
    if language == "auto":
        if any("\u4e00" <= char <= "\u9fff" for char in prediction + reference):
            language = "chinese"
        else:
            language = "english"

    if language == "chinese":
        pred_tokens = list(prediction)
        ref_tokens = list(reference)
    else:
        # For English, use word-level tokenization with punctuation as separate tokens
        # Convert to lowercase and tokenize with punctuation handling
        pred_tokens = tokenize_with_punctuation(prediction.lower())
        ref_tokens = tokenize_with_punctuation(reference.lower())

    # Calculate LCS length using dynamic programming
    m, n = len(pred_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0, 0.0, 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]

    # Calculate precision, recall, and F1
    precision = lcs_length / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = lcs_length / len(ref_tokens) if len(ref_tokens) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def evaluate_results(
    predictions,
    references,
    split="train",
    tokenizer=None,
    save_results=False,
    output_dir="./results",
    filename_prefix=None,
    ids=None,
    languages=None,
):
    """
    Evaluate prediction results using BLEU and ROUGE metrics.

    Args:
        predictions (list): List of predicted sequences.
        references (list): List of reference sequences.
        tokenizer (object, optional): Tokenizer if needed for evaluation.
        split (str): The data split being evaluated.
        save_results (bool): Whether to save predictions and references to file.
        output_dir (str): Directory to save the results.
        filename_prefix (str, optional): Prefix for the saved filename. If provided,
            the file will be named ``{prefix}.json`` instead of the default timestamp.

    Returns:
        dict: A dictionary of evaluation scores.
    """
    if len(predictions) != len(references) or not references:
        raise ValueError("Expected equally sized, nonempty predictions and references.")
    if not all(isinstance(text, str) for text in list(predictions) + list(references)):
        raise ValueError("Predictions and references must contain strings.")
    if ids is not None and len(ids) != len(references):
        raise ValueError("IDs must match the number of evaluated samples.")
    if languages is not None and (len(languages) != len(references) or len(set(languages)) != 1):
        raise ValueError("Languages must match the samples and identify one dataset language.")
    is_chinese = (languages[0].lower() == "chinese" if languages else
                  any("\u4e00" <= char <= "\u9fff" for text in references for char in text))
    language = "chinese" if is_chinese else "german"
    normalized_predictions = [normalize_text(text, language) for text in predictions]
    normalized_references = [normalize_text(text, language) for text in references]
    if tokenizer is None:
        tokenizer = "zh" if is_chinese else "13a"
    log_dicts = {}

    for i in range(1, 5):
        score = (
            BLEU(max_ngram_order=i, tokenize=tokenizer)
            .corpus_score(normalized_predictions, [normalized_references])
            .score
        )
        log_dicts[f"{split}/bleu" + str(i)] = score

    scorer = rouge_scorer.RougeScorer(["rougeL"], tokenizer=PaperTokenizer(language))
    rouge_scores = [scorer.score(ref, pred)["rougeL"]
                    for ref, pred in zip(normalized_references, normalized_predictions)]
    for name, attribute in (("precision", "precision"), ("recall", "recall"), ("f1", "fmeasure")):
        log_dicts[f"{split}/rougeL_{name}"] = (
            sum(getattr(score, attribute) for score in rouge_scores) / len(rouge_scores) * 100)

    if save_results:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

        results_data = {
            "predictions": predictions,
            "references": references,
            "metrics": log_dicts,
            "split": split,
            "is_chinese": is_chinese,
            "language": language,
            "normalization": "preserve punctuation; remove whitespace" if is_chinese
                             else "lowercase; remove punctuation; collapse whitespace",
            "bleu_tokenizer": tokenizer,
        }
        if ids is not None:
            results_data["ids"] = ids

        if filename_prefix:
            output_file = os.path.join(output_dir, split, f"{filename_prefix}_bleu4={log_dicts[f'{split}/bleu4']:.2f}.json")
        else:
            output_file = os.path.join(output_dir, split, f"results_{log_dicts[f'{split}/bleu4']:.2f}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

        print(f"Results saved to {output_file}")

    return log_dicts
