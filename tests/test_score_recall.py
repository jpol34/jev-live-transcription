import importlib.util
import json
import sys
from pathlib import Path

from jev_live_transcription import db

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "score_recall.py"
_spec = importlib.util.spec_from_file_location("score_recall", _SCRIPT_PATH)
score_recall = importlib.util.module_from_spec(_spec)
sys.modules["score_recall"] = score_recall
_spec.loader.exec_module(score_recall)


def _make_call_fields(call_id: int) -> dict:
    return dict(
        call_id=call_id,
        scenario_json=json.dumps({"id": call_id}),
        category="resident",
        subtype="test_subtype",
        edge_case=0,
        ground_truth_json=json.dumps({"caller_name": "Jamie Test"}),
        target_seconds=180.0,
        full_transcript_word_count=42,
    )


def _make_pipeline_run_fields(**overrides) -> dict:
    fields = dict(
        latency_ms=10.0,
        input_tokens=None,
        cached_input_tokens=None,
        output_tokens=None,
        estimated_cost_usd=None,
        raw_output_json=None,
        error=None,
    )
    fields.update(overrides)
    return fields


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


def test_is_match_true_for_dollar_prefixed_value_preceded_by_whitespace():
    assert score_recall.is_match("around $950", "$950")
    assert score_recall.is_match("$950", "the price quoted was $950")


def test_is_match_false_for_unrelated_dollar_values():
    # Substring matching for a symbol-prefixed value stays bounded -- two genuinely different
    # dollar amounts still don't match.
    assert not score_recall.is_match("$950", "$1,900")


def test_is_match_true_for_dollar_value_directly_abutting_a_word_with_no_separator():
    # A symbol-prefixed value like "$950" is unambiguously delimited by the symbol itself, so it
    # matches whether it's preceded by whitespace or abuts a word with no separator at all.
    assert score_recall.is_match("$950", "the offer was$950 total")


def test_is_match_false_for_empty_values():
    assert not score_recall.is_match("", "Tim Barker")
    assert not score_recall.is_match("Tim Barker", "")


def test_is_match_true_when_extracted_matches_any_item_in_list_truth():
    assert score_recall.is_match("gym", ["pool", "gym"])


def test_is_match_false_when_extracted_matches_no_item_in_list_truth():
    assert not score_recall.is_match("gym", ["pool", "clubhouse"])


def test_is_match_false_for_empty_list_truth():
    assert not score_recall.is_match("gym", [])


def test_is_match_ignores_none_items_in_list_truth():
    assert score_recall.is_match("gym", [None, "gym"])
    assert not score_recall.is_match("pool", [None, "gym"])


def test_is_match_ignores_empty_string_items_in_list_truth():
    assert not score_recall.is_match("gym", [""])


def test_score_recall_matches_against_list_valued_ground_truth():
    ground_truths = {1: {"amenities_requested": ["pool", "gym"]}}
    committed = {1: {"amenities_requested": "gym"}}

    stats = score_recall.score_recall(ground_truths, committed, fields=("amenities_requested",))

    assert stats["amenities_requested"] == {"n_expected": 1, "n_matched": 1, "recall": 1.0}


def test_score_recall_empty_list_ground_truth_does_not_count_as_disclosed():
    ground_truths = {1: {"amenities_requested": []}}
    committed = {}

    stats = score_recall.score_recall(ground_truths, committed, fields=("amenities_requested",))

    assert stats["amenities_requested"] == {"n_expected": 0, "n_matched": 0, "recall": None}


def test_score_recall_list_of_only_empty_strings_does_not_count_as_disclosed():
    ground_truths = {1: {"amenities_requested": [""]}}
    committed = {}

    stats = score_recall.score_recall(ground_truths, committed, fields=("amenities_requested",))

    assert stats["amenities_requested"] == {"n_expected": 0, "n_matched": 0, "recall": None}


def test_score_recall_list_with_none_item_does_not_crash_and_matches_remaining_item():
    ground_truths = {1: {"amenities_requested": [None, "gym"]}}
    committed = {1: {"amenities_requested": "pool"}}  # doesn't match the non-None item

    stats = score_recall.score_recall(ground_truths, committed, fields=("amenities_requested",))

    assert stats["amenities_requested"] == {"n_expected": 1, "n_matched": 0, "recall": 0.0}


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


# -- precision --------------------------------------------------------------


def test_score_precision_true_positive_for_matching_non_null_ground_truth():
    ground_truths = {1: {"caller_name": "Tim Barker"}}
    committed = {1: {"caller_name": "Tim Barker"}}

    stats = score_recall.score_precision(ground_truths, committed, fields=("caller_name",))

    assert stats["caller_name"] == {
        "n_committed": 1,
        "n_correct": 1,
        "n_false_positives": 0,
        "precision": 1.0,
    }


def test_score_precision_false_positive_for_mismatch_against_non_null_ground_truth():
    # Same (call, field) row is simultaneously a recall miss and a precision false positive.
    ground_truths = {1: {"phone_number": "555-0134"}}
    committed = {1: {"phone_number": "555-9999"}}

    recall_stats = score_recall.score_recall(ground_truths, committed, fields=("phone_number",))
    precision_stats = score_recall.score_precision(ground_truths, committed, fields=("phone_number",))

    assert recall_stats["phone_number"]["recall"] == 0.0
    assert precision_stats["phone_number"] == {
        "n_committed": 1,
        "n_correct": 0,
        "n_false_positives": 1,
        "precision": 0.0,
    }


def test_score_precision_false_positive_for_committed_value_against_null_ground_truth():
    ground_truths = {1: {"email": None}}
    committed = {1: {"email": "someone@example.com"}}

    stats = score_recall.score_precision(ground_truths, committed, fields=("email",))

    assert stats["email"] == {
        "n_committed": 1,
        "n_correct": 0,
        "n_false_positives": 1,
        "precision": 0.0,
    }


def test_score_precision_true_positive_for_list_span_matching_any_item():
    ground_truths = {1: {"amenities_requested": ["pool", "gym"]}}
    committed = {1: {"amenities_requested": "gym"}}

    stats = score_recall.score_precision(ground_truths, committed, fields=("amenities_requested",))

    assert stats["amenities_requested"] == {
        "n_committed": 1,
        "n_correct": 1,
        "n_false_positives": 0,
        "precision": 1.0,
    }


def test_score_precision_false_positive_for_list_span_matching_no_item():
    ground_truths = {1: {"amenities_requested": ["pool", "clubhouse"]}}
    committed = {1: {"amenities_requested": "gym"}}

    stats = score_recall.score_precision(ground_truths, committed, fields=("amenities_requested",))

    assert stats["amenities_requested"] == {
        "n_committed": 1,
        "n_correct": 0,
        "n_false_positives": 1,
        "precision": 0.0,
    }


def test_score_precision_false_positive_for_empty_list_ground_truth():
    # An empty list discloses nothing -- same as null ground truth.
    ground_truths = {1: {"amenities_requested": []}}
    committed = {1: {"amenities_requested": "gym"}}

    stats = score_recall.score_precision(ground_truths, committed, fields=("amenities_requested",))

    assert stats["amenities_requested"] == {
        "n_committed": 1,
        "n_correct": 0,
        "n_false_positives": 1,
        "precision": 0.0,
    }


def test_score_precision_is_none_when_field_never_committed():
    ground_truths = {1: {"pet_info": "a cat"}}
    committed = {}

    stats = score_recall.score_precision(ground_truths, committed, fields=("pet_info",))

    assert stats["pet_info"] == {
        "n_committed": 0,
        "n_correct": 0,
        "n_false_positives": 0,
        "precision": None,
    }


def test_score_precision_uncommitted_field_for_one_call_does_not_count_against_precision():
    # Only committed values are scored for precision -- a call with no committed value for this
    # field contributes nothing (it's a recall miss, not a precision false positive).
    ground_truths = {1: {"caller_name": "Tim Barker"}, 2: {"caller_name": "Ana Ruiz"}}
    committed = {2: {"caller_name": "Ana Ruiz"}}  # call 1 has no committed caller_name at all

    stats = score_recall.score_precision(ground_truths, committed, fields=("caller_name",))

    assert stats["caller_name"] == {
        "n_committed": 1,
        "n_correct": 1,
        "n_false_positives": 0,
        "precision": 1.0,
    }


def test_score_precision_computes_per_field_stats_across_multiple_calls():
    ground_truths = {
        1: {"caller_name": "Tim Barker", "email": None},
        2: {"caller_name": "Ana Ruiz", "email": "ana@example.com"},
    }
    committed = {
        1: {"caller_name": "Tim Barker", "email": "spurious@example.com"},  # email FP (null truth)
        2: {"caller_name": "Wrong Name", "email": "ana@example.com"},  # name FP, email TP
    }

    stats = score_recall.score_precision(ground_truths, committed, fields=("caller_name", "email"))

    assert stats["caller_name"] == {
        "n_committed": 2,
        "n_correct": 1,
        "n_false_positives": 1,
        "precision": 0.5,
    }
    assert stats["email"] == {
        "n_committed": 2,
        "n_correct": 1,
        "n_false_positives": 1,
        "precision": 0.5,
    }


# -- jev call volume ----------------------------------------------------------


def test_score_call_volume_counts_choice_and_noul_calls_and_distinct_candidate_sizes():
    jev_calls = {
        "phone_number": [
            {"question_type": "noul", "distinct_candidates": ["555-0134"]},
            {"question_type": "choice", "distinct_candidates": ["555-0134", "555-9999"]},
            {"question_type": "choice", "distinct_candidates": ["555-0134", "555-9999", "555-1111"]},
        ]
    }

    stats = score_recall.score_call_volume(jev_calls, fields=("phone_number",))

    assert stats["phone_number"] == {
        "n_jev_calls": 3,
        "n_choice_calls": 2,
        "avg_distinct_candidates": (1 + 2 + 3) / 3,
        "max_distinct_candidates": 3,
    }


def test_score_call_volume_is_zero_and_none_for_field_with_no_jev_calls():
    stats = score_recall.score_call_volume({}, fields=("caller_name",))

    assert stats["caller_name"] == {
        "n_jev_calls": 0,
        "n_choice_calls": 0,
        "avg_distinct_candidates": None,
        "max_distinct_candidates": None,
    }


def test_score_call_volume_treats_missing_distinct_candidates_as_empty():
    jev_calls = {"unit_number": [{"question_type": "noul"}]}

    stats = score_recall.score_call_volume(jev_calls, fields=("unit_number",))

    assert stats["unit_number"]["n_jev_calls"] == 1
    assert stats["unit_number"]["avg_distinct_candidates"] == 0


# -- merge_stats --------------------------------------------------------------


def test_merge_stats_combines_keys_from_every_dict_per_field():
    recall_stats = {"caller_name": {"n_expected": 2, "n_matched": 2, "recall": 1.0}}
    precision_stats = {"caller_name": {"n_committed": 2, "n_correct": 2, "n_false_positives": 0, "precision": 1.0}}
    call_volume_stats = {
        "caller_name": {
            "n_jev_calls": 2,
            "n_choice_calls": 0,
            "avg_distinct_candidates": 1.0,
            "max_distinct_candidates": 1,
        }
    }

    merged = score_recall.merge_stats(
        recall_stats, precision_stats, call_volume_stats, fields=("caller_name",)
    )

    assert merged["caller_name"] == {
        "n_expected": 2,
        "n_matched": 2,
        "recall": 1.0,
        "n_committed": 2,
        "n_correct": 2,
        "n_false_positives": 0,
        "precision": 1.0,
        "n_jev_calls": 2,
        "n_choice_calls": 0,
        "avg_distinct_candidates": 1.0,
        "max_distinct_candidates": 1,
    }


# -- extended baseline diff (precision, jev call volume) ----------------------


def test_diff_against_baseline_flags_precision_regression_past_tolerance():
    baseline = {"email": {"precision": 0.8}}
    current = {"email": {"precision": 0.5, "n_committed": 10, "n_correct": 5}}

    regressions = score_recall.diff_against_baseline(current, baseline, tolerance=0.05)

    assert len(regressions) == 1
    assert "email" in regressions[0]
    assert "precision" in regressions[0]


def test_diff_against_baseline_tolerates_small_precision_drop():
    baseline = {"caller_name": {"precision": 0.9}}
    current = {"caller_name": {"precision": 0.87}}

    regressions = score_recall.diff_against_baseline(current, baseline, tolerance=0.05)

    assert regressions == []


def test_diff_against_baseline_skips_fields_with_no_precision_in_either_side():
    baseline = {"caller_name": {"precision": None}}
    current = {"caller_name": {"precision": None}}

    regressions = score_recall.diff_against_baseline(current, baseline, tolerance=0.05)

    assert regressions == []


def test_diff_against_baseline_flags_jev_choice_call_volume_increase_past_tolerance():
    baseline = {"phone_number": {"n_choice_calls": 10}}
    current = {"phone_number": {"n_choice_calls": 20}}  # +100%, well past a 5% tolerance

    regressions = score_recall.diff_against_baseline(current, baseline, tolerance=0.05)

    assert len(regressions) == 1
    assert "phone_number" in regressions[0]
    assert "choice-call volume" in regressions[0]


def test_diff_against_baseline_tolerates_small_jev_choice_call_volume_increase():
    baseline = {"phone_number": {"n_choice_calls": 100}}
    current = {"phone_number": {"n_choice_calls": 102}}  # +2%, within a 5% tolerance

    regressions = score_recall.diff_against_baseline(current, baseline, tolerance=0.05)

    assert regressions == []


def test_diff_against_baseline_skips_fields_with_no_call_volume_in_either_side():
    baseline = {"caller_name": {}}
    current = {"caller_name": {}}

    regressions = score_recall.diff_against_baseline(current, baseline, tolerance=0.05)

    assert regressions == []


def test_diff_against_baseline_reports_multiple_metric_regressions_for_same_field():
    baseline = {"phone_number": {"recall": 0.9, "precision": 0.9, "n_choice_calls": 10}}
    current = {"phone_number": {"recall": 0.5, "precision": 0.5, "n_choice_calls": 30}}

    regressions = score_recall.diff_against_baseline(current, baseline, tolerance=0.05)

    assert len(regressions) == 3


# -- load_jev_calls (DB integration) -------------------------------------------


def test_load_jev_calls_groups_by_field_name_and_skips_other_pipelines_and_stages(tmp_path):
    store = db.CaptureStore(tmp_path / "capture.db")
    try:
        store.insert_call(**_make_call_fields(1))
        tick_id = store.enqueue_tick(
            call_id=1,
            tick_number=0,
            wall_clock_ts=1.0,
            transcript_char_offset=10,
            transcript_snapshot="Agent: Hello",
        ).result(timeout=5)

        # A gliner_standard stage row for the same pipeline -- must be excluded (not a jev call).
        gliner_run_id = store.enqueue_pipeline_run(
            **_make_pipeline_run_fields(
                tick_id=tick_id,
                call_id=1,
                pipeline="gliner_jev",
                stage="gliner_standard",
                raw_output_json=json.dumps({}),
            )
        ).result(timeout=5)
        store.enqueue_field_extraction(
            run_id=gliner_run_id,
            call_id=1,
            tick_number=0,
            pipeline="gliner_jev",
            field_name="phone_number",
            candidate_value="555-0134",
            confidence=None,
            is_committed=0,
        ).result(timeout=5)

        # A jev-stage row for a different pipeline -- must be excluded.
        other_pipeline_run_id = store.enqueue_pipeline_run(
            **_make_pipeline_run_fields(
                tick_id=tick_id,
                call_id=1,
                pipeline="some_other_pipeline",
                stage="jev",
                raw_output_json=json.dumps({"question_type": "noul", "distinct_candidates": ["x"]}),
            )
        ).result(timeout=5)
        store.enqueue_field_extraction(
            run_id=other_pipeline_run_id,
            call_id=1,
            tick_number=0,
            pipeline="some_other_pipeline",
            field_name="phone_number",
            candidate_value="x",
            confidence=0.9,
            is_committed=1,
        ).result(timeout=5)

        # Two real jev calls for gliner_jev, on two different fields.
        noul_run_id = store.enqueue_pipeline_run(
            **_make_pipeline_run_fields(
                tick_id=tick_id,
                call_id=1,
                pipeline="gliner_jev",
                stage="jev",
                raw_output_json=json.dumps(
                    {
                        "question_type": "noul",
                        "candidate": "555-0134",
                        "confidence": 0.95,
                        "is_none_of_these": False,
                        "distinct_candidates": ["555-0134"],
                    }
                ),
            )
        ).result(timeout=5)
        store.enqueue_field_extraction(
            run_id=noul_run_id,
            call_id=1,
            tick_number=0,
            pipeline="gliner_jev",
            field_name="phone_number",
            candidate_value="555-0134",
            confidence=0.95,
            is_committed=1,
        ).result(timeout=5)

        choice_run_id = store.enqueue_pipeline_run(
            **_make_pipeline_run_fields(
                tick_id=tick_id,
                call_id=1,
                pipeline="gliner_jev",
                stage="jev",
                raw_output_json=json.dumps(
                    {
                        "question_type": "choice",
                        "candidate": "Tim Barker",
                        "confidence": 0.8,
                        "is_none_of_these": False,
                        "distinct_candidates": ["Tim Barker", "Tim B."],
                    }
                ),
            )
        ).result(timeout=5)
        store.enqueue_field_extraction(
            run_id=choice_run_id,
            call_id=1,
            tick_number=1,
            pipeline="gliner_jev",
            field_name="caller_name",
            candidate_value="Tim Barker",
            confidence=0.8,
            is_committed=1,
        ).result(timeout=5)
    finally:
        store.close()

    conn = db.connect(store.db_path)
    try:
        jev_calls = score_recall.load_jev_calls(conn, "gliner_jev")
    finally:
        conn.close()

    assert set(jev_calls) == {"phone_number", "caller_name"}
    assert len(jev_calls["phone_number"]) == 1
    assert jev_calls["phone_number"][0]["question_type"] == "noul"
    assert len(jev_calls["caller_name"]) == 1
    assert jev_calls["caller_name"][0]["question_type"] == "choice"
    assert jev_calls["caller_name"][0]["distinct_candidates"] == ["Tim Barker", "Tim B."]
