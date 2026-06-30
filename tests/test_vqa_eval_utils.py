import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm_eval_utils import (
    bucket_vqa_answer,
    normalize_answer,
    score_vqa_answer,
    summarize_vqa_rows,
)


def test_normalize_answer_strips_case_and_punctuation():
    assert normalize_answer('  Blue, sky! ') == 'blue sky'


def test_score_vqa_answer_exact_and_recall():
    exact, recall = score_vqa_answer('blue and white', 'blue')
    assert exact
    assert recall
    exact, recall = score_vqa_answer('white', 'blue')
    assert not exact
    assert not recall


def test_bucket_vqa_answer():
    assert bucket_vqa_answer('yes') == 'yes_no'
    assert bucket_vqa_answer('3') == 'number'
    assert bucket_vqa_answer('tennis racket') == 'other'


def test_summarize_vqa_rows_reports_correct_and_shuffled_breakdowns():
    rows = [
        {'gold': 'yes', 'correct_exact': True, 'correct_recall': True,
         'shuffled_exact': False, 'shuffled_recall': False},
        {'gold': '3', 'correct_exact': False, 'correct_recall': False,
         'shuffled_exact': True, 'shuffled_recall': True},
    ]
    summary = summarize_vqa_rows(rows, sampling_steps=64)
    assert summary['n'] == 2
    assert summary['sampling_steps'] == 64
    assert summary['correct_exact_match'] == 0.5
    assert summary['shuffled_exact_match'] == 0.5
    assert summary['buckets']['yes_no']['n'] == 1
    assert summary['buckets']['number']['n'] == 1
