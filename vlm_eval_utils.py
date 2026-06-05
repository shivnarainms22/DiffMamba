"""Small, dependency-light helpers for VQA-style evaluation."""

import re
from collections import Counter, defaultdict


def normalize_answer(value: str) -> str:
    return re.sub(r'[^a-z0-9 ]', '', str(value).lower()).strip()


def score_vqa_answer(generated: str, gold: str) -> tuple[bool, bool]:
    gen = normalize_answer(generated)
    target = normalize_answer(gold)
    exact = bool(target) and (gen == target or gen.startswith(target + ' '))
    recall = bool(target) and target in gen
    return exact, recall


def bucket_vqa_answer(gold: str) -> str:
    value = normalize_answer(gold)
    if value in {'yes', 'no'}:
        return 'yes_no'
    if value.isdigit():
        return 'number'
    return 'other'


def _mean(rows, key):
    if not rows:
        return 0.0
    return sum(1 for r in rows if r[key]) / len(rows)


def summarize_vqa_rows(rows, sampling_steps: int) -> dict:
    buckets = defaultdict(list)
    for row in rows:
        buckets[bucket_vqa_answer(row['gold'])].append(row)

    bucket_summary = {}
    for name, group in buckets.items():
        bucket_summary[name] = {
            'n': len(group),
            'correct_exact_match': _mean(group, 'correct_exact'),
            'correct_gold_recall': _mean(group, 'correct_recall'),
            'shuffled_exact_match': _mean(group, 'shuffled_exact'),
            'shuffled_gold_recall': _mean(group, 'shuffled_recall'),
        }

    answer_hist = Counter(normalize_answer(r['generated_correct'])
                          for r in rows if r.get('generated_correct'))
    return {
        'n': len(rows),
        'sampling_steps': sampling_steps,
        'correct_exact_match': _mean(rows, 'correct_exact'),
        'correct_gold_recall': _mean(rows, 'correct_recall'),
        'shuffled_exact_match': _mean(rows, 'shuffled_exact'),
        'shuffled_gold_recall': _mean(rows, 'shuffled_recall'),
        'image_ablation_exact_delta': (
            _mean(rows, 'correct_exact') - _mean(rows, 'shuffled_exact')),
        'image_ablation_recall_delta': (
            _mean(rows, 'correct_recall') - _mean(rows, 'shuffled_recall')),
        'buckets': bucket_summary,
        'top_generated_answers': answer_hist.most_common(20),
    }
