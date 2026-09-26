import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "score_recall.py"
_spec = importlib.util.spec_from_file_location("score_recall", _SCRIPT_PATH)
score_recall = importlib.util.module_from_spec(_spec)
sys.modules["score_recall"] = score_recall
_spec.loader.exec_module(score_recall)


def test_is_match_exact_after_normalization():
    assert score_recall.is_match("Tim Barker", "  tim   barker  ")


def test_is_match_substring_either_way():
    assert score_recall.is_match("the caller, Tim Barker", "Tim Barker")
    assert score_recall.is_match("Tim Barker", "Tim")


def test_is_match_false_for_unrelated_values():
    assert not score_recall.is_match("Tim Barker", "555-0134")


def test_is_match_false_for_numeric_substring_that_is_not_a_whole_token():
    # "12" is a plain substring of "212", but they're different unit numbers/prices -- a match
    # here would be a false positive recall hit for a wrong extraction.
    assert not score_recall.is_match("212", "12")
    assert not score_recall.is_match("103", "3")


def test_is_match_true_for_whole_numeric_token_within_longer_text():
    assert score_recall.is_match("apartment 204", "204")


def test_is_match_false_for_empty_values():
    assert not score_recall.is_match("", "Tim Barker")
    assert not score_recall.is_match("Tim Barker", "")


def test_score_recall_computes_per_field_stats():
    ground_truths = {
        1: {"caller_name": "Tim Barker", "email": None, "phone_number": "555-0134"},
        2: {"caller_name": "Ana Ruiz", "email": "ana@example.com", "phone_number": None},
    }
    committed = {
        1: {"caller_name": "Tim Barker", "phone_number": "555-9999"},  # phone wrong
        2: {"caller_name": "Ana Ruiz"},  # email missing entirely
    }

    stats = score_recall.score_recall(
        ground_truths, committed, fields=("caller_name", "email", "phone_number")
    )

    assert stats["caller_name"] == {"n_expected": 2, "n_matched": 2, "recall": 1.0}
    assert stats["email"] == {"n_expected": 1, "n_matched": 0, "recall": 0.0}
    assert stats["phone_number"] == {"n_expected": 1, "n_matched": 0, "recall": 0.0}


def test_score_recall_recall_is_none_when_no_call_has_ground_truth():
    ground_truths = {1: {"pet_info": None}}
    committed = {}

    stats = score_recall.score_recall(ground_truths, committed, fields=("pet_info",))

    assert stats["pet_info"] == {"n_expected": 0, "n_matched": 0, "recall": None}


def test_score_recall_missing_call_id_in_committed_counts_as_no_match():
    ground_truths = {1: {"caller_name": "Tim Barker"}}
    committed = {}  # call_id 1 never appears -- no extraction at all

    stats = score_recall.score_recall(ground_truths, committed, fields=("caller_name",))

    assert stats["caller_name"] == {"n_expected": 1, "n_matched": 0, "recall": 0.0}


def test_diff_against_baseline_flags_regression_past_tolerance():
    baseline = {"caller_name": {"recall": 0.9}, "email": {"recall": 0.8}}
    current = {
        "caller_name": {"recall": 0.9, "n_expected": 10, "n_matched": 9},
        "email": {"recall": 0.5, "n_expected": 10, "n_matched": 5},  # big drop
    }

    regressions = score_recall.diff_against_baseline(current, baseline, tolerance=0.05)

    assert len(regressions) == 1
    assert "email" in regressions[0]


def test_diff_against_baseline_tolerates_small_drop():
    baseline = {"caller_name": {"recall": 0.9}}
    current = {"caller_name": {"recall": 0.87, "n_expected": 10, "n_matched": 9}}

    regressions = score_recall.diff_against_baseline(current, baseline, tolerance=0.05)

    assert regressions == []


def test_diff_against_baseline_skips_fields_with_no_recall_in_either_side():
    baseline = {"caller_name": {"recall": None}}
    current = {"caller_name": {"recall": None, "n_expected": 0, "n_matched": 0}}

    regressions = score_recall.diff_against_baseline(current, baseline, tolerance=0.05)

    assert regressions == []
