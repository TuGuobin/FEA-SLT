# -*- coding: utf-8 -*-
import argparse
import json
import re


def extract_punctuation_sequence(text):
    punctuation_pattern = r'[，。？！；：""''（）【】《》、,.?!;:\'"()\[\]<>]'
    return re.findall(punctuation_pattern, text)


def lcs_length(seq1, seq2):
    m, n = len(seq1), len(seq2)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev

    return prev[n]


def calculate_rouge_l_punctuation(reference, prediction):
    ref_punct = extract_punctuation_sequence(reference)
    pred_punct = extract_punctuation_sequence(prediction)

    if len(ref_punct) == 0 and len(pred_punct) == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0

    if len(ref_punct) == 0:
        return 0.0, 1.0, 0.0, 0, len(pred_punct), 0

    if len(pred_punct) == 0:
        return 1.0, 0.0, 0.0, 0, 0, len(ref_punct)

    lcs_len = lcs_length(ref_punct, pred_punct)

    recall = lcs_len / len(ref_punct)
    precision = lcs_len / len(pred_punct)

    if recall + precision > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0.0

    return precision, recall, f1, lcs_len, len(pred_punct), len(ref_punct)


def evaluate_dataset(references, predictions, dataset_name="dataset"):
    if len(references) != len(predictions) or not references:
        raise ValueError("Expected equally sized, nonempty references and predictions.")

    n = len(references)

    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_lcs = 0
    total_pred = 0
    total_ref = 0

    for ref, pred in zip(references, predictions):
        precision, recall, f1, lcs_len, pred_count, ref_count = calculate_rouge_l_punctuation(ref, pred)

        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_lcs += lcs_len
        total_pred += pred_count
        total_ref += ref_count

    avg_precision = total_precision / n
    avg_recall = total_recall / n
    avg_f1 = total_f1 / n

    return {
        "dataset": dataset_name,
        "n": n,
        "macro_p": avg_precision * 100,
        "macro_r": avg_recall * 100,
        "macro_f1": avg_f1 * 100,
        "scale": 100,
    }


def print_table(results_list):
    header = f"{'Dataset':<20} {'N':>6} {'P':>10} {'R':>10} {'F1':>10}"
    print(header)
    print("-" * len(header))

    for r in results_list:
        print(f"{r['dataset']:<20} {r['n']:>6} {r['macro_p']:>10.4f} {r['macro_r']:>10.4f} {r['macro_f1']:>10.4f}")


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--file", "-f", type=str, required=True, help="JSON file path containing 'references' and 'predictions' arrays")
    args = parser.parse_args()

    with open(args.file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    references = data.get("references", [])
    predictions = data.get("predictions", [])

    if not references or not predictions:
        parser.error("'references' and 'predictions' must be nonempty arrays.")

    if len(references) != len(predictions):
        parser.error(f"references ({len(references)}) and predictions ({len(predictions)}) have different lengths")

    results = []

    results.append(evaluate_dataset(references, predictions, dataset_name="All"))

    question_refs = []
    question_preds = []
    for ref, pred in zip(references, predictions):
        if '？' in ref or '?' in ref:
            question_refs.append(ref)
            question_preds.append(pred)

    if question_refs:
        results.append(evaluate_dataset(question_refs, question_preds, dataset_name="Question"))

    print_table(results)


if __name__ == '__main__':
    main()
