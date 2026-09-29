"""Tests for gliner_only_resolver -- fully local, synchronous, no mocking needed."""

from jev_live_transcription import config
from jev_live_transcription.gliner_only_resolver import GlinerOnlyResolver


def _span(text: str, score: float | None, start: int | None = None, end: int | None = None) -> dict:
    return {"text": text, "score": score, "start": start, "end": end}


# --- config constants ---------------------------------------------------------------------------


def test_gliner_only_constants_match_spec():
    assert config.GLINER_ONLY_COMMIT_THRESHOLD == 0.50
    assert config.GLINER_ONLY_PER_FIELD_COMMIT_THRESHOLDS == {
        "price_quoted": 0.18,
        "budget_amount": 0.15,
        "pet_info": 0.40,
        "work_order_issue": 0.40,
    }
    assert config.GLINER_ONLY_MARGIN_THRESHOLD == 0.15
    assert config.GLINER_ONLY_DETERMINATION_COMMIT_FLOOR == 0.55
    assert config.GLINER_ONLY_SETTLE_TICKS == config.JEV_RECONFIRM_SETTLE_TICKS
    assert config.GLINER_ONLY_SELF_CORRECTION_SCORE_BONUS == 0.20


# --- zero/one distinct candidate ----------------------------------------------------------------


def test_zero_candidates_returns_none():
    resolver = GlinerOnlyResolver(settle_ticks=1)

    result = resolver.resolve_field(
        call_id=1, field_name="caller_name", candidate_spans=[], context_window="ctx"
    )

    assert result is None


def test_single_candidate_above_floor_commits():
    resolver = GlinerOnlyResolver(settle_ticks=1)

    result = resolver.resolve_field(
        call_id=1,
        field_name="caller_name",
        candidate_spans=[_span("Lindsey Perkins", 0.5, 0, 15)],
        context_window="ctx",
    )

    assert result is not None
    assert result.candidate == "Lindsey Perkins"
    assert result.confidence == 0.5
    assert result.is_committed is True
    assert result.is_none_of_these is False
    assert result.distinct_candidates == ("Lindsey Perkins",)
    assert result.decision_reason == "single_above_floor"


def test_single_candidate_below_floor_not_committed_but_returned():
    resolver = GlinerOnlyResolver(settle_ticks=1)

    result = resolver.resolve_field(
        call_id=1,
        field_name="caller_name",
        candidate_spans=[_span("Lindsey Perkins", 0.2)],
        context_window="ctx",
    )

    assert result is not None
    assert result.is_committed is False
    assert result.decision_reason == "single_below_floor"


def test_single_candidate_already_committed_returns_none_next_tick():
    resolver = GlinerOnlyResolver(settle_ticks=1)
    key_kwargs = dict(call_id=1, field_name="caller_name", context_window="ctx")

    first = resolver.resolve_field(candidate_spans=[_span("Someone", 0.9)], **key_kwargs)
    assert first is not None and first.is_committed is True

    second = resolver.resolve_field(candidate_spans=[_span("Someone", 0.9)], **key_kwargs)

    assert second is None


def test_single_candidate_determination_field_uses_its_own_floor():
    resolver = GlinerOnlyResolver(settle_ticks=1)

    below = resolver.resolve_field(
        call_id=1,
        field_name="permission_to_enter",
        candidate_spans=[_span("yes", 0.5)],
        context_window="ctx",
    )
    assert below is not None and below.is_committed is False

    resolver2 = GlinerOnlyResolver(settle_ticks=1)
    above = resolver2.resolve_field(
        call_id=1,
        field_name="permission_to_enter",
        candidate_spans=[_span("yes", 0.6)],
        context_window="ctx",
    )
    assert above is not None and above.is_committed is True


def test_per_field_commit_threshold_override_takes_precedence_over_default():
    # price_quoted's override (0.18) is well below the 0.50 default -- a score that would fail
    # the default floor must still commit under the override.
    resolver = GlinerOnlyResolver(settle_ticks=1)

    result = resolver.resolve_field(
        call_id=1,
        field_name="price_quoted",
        candidate_spans=[_span("$1,200", 0.2)],
        context_window="ctx",
    )

    assert result is not None
    assert result.is_committed is True
    assert result.decision_reason == "single_above_floor"


def test_single_candidate_score_freshness_updates_across_ticks():
    resolver = GlinerOnlyResolver(settle_ticks=1)
    key_kwargs = dict(call_id=1, field_name="caller_name", context_window="ctx")

    first = resolver.resolve_field(candidate_spans=[_span("Someone", 0.2)], **key_kwargs)
    assert first is not None and first.confidence == 0.2 and first.is_committed is False

    second = resolver.resolve_field(candidate_spans=[_span("Someone", 0.9)], **key_kwargs)
    assert second is not None and second.confidence == 0.9 and second.is_committed is True


# --- normalization/dedup reuse (parity with jev_pipeline) --------------------------------------


def test_duplicate_phone_candidate_by_normalized_value_stays_single_candidate():
    resolver = GlinerOnlyResolver(settle_ticks=1)

    result = resolver.resolve_field(
        call_id=1,
        field_name="phone_number",
        candidate_spans=[_span("555-4321", 0.5), _span("5554321", 0.6)],  # same digits
        context_window="ctx",
    )

    assert result is not None
    assert result.distinct_candidates == ("555-4321",)  # first-seen display text kept
    assert result.confidence == 0.6  # freshest score within the same tick


def test_phone_country_code_variant_dedups_to_single_candidate():
    resolver = GlinerOnlyResolver(settle_ticks=1)

    result = resolver.resolve_field(
        call_id=1,
        field_name="phone_number",
        candidate_spans=[_span("+1 555-432-1123", 0.5), _span("555-432-1123", 0.5)],
        context_window="ctx",
    )

    assert result is not None
    assert len(result.distinct_candidates) == 1


def test_single_email_candidate_normalized_from_spoken_form():
    resolver = GlinerOnlyResolver(settle_ticks=1)

    result = resolver.resolve_field(
        call_id=1,
        field_name="email",
        candidate_spans=[_span("e-t-h-a-n dot r-o-b-e-r-t-s at mail dot com", 0.5)],
        context_window="ctx",
    )

    assert result is not None
    assert result.candidate == "ethan.roberts@mail.com"


def test_duplicate_email_candidate_spoken_and_literal_forms_dedup():
    resolver = GlinerOnlyResolver(settle_ticks=1)

    result = resolver.resolve_field(
        call_id=1,
        field_name="email",
        candidate_spans=[
            _span("e-t-h-a-n dot r-o-b-e-r-t-s at mail dot com", 0.4),
            _span("Ethan.Roberts@Mail.com", 0.6),
        ],
        context_window="ctx",
    )

    assert result is not None
    assert len(result.distinct_candidates) == 1
    assert result.confidence == 0.6


# --- 2+ distinct candidates: settle gating --------------------------------------------------


def test_two_distinct_candidates_gated_by_settle_ticks():
    resolver = GlinerOnlyResolver(settle_ticks=2)
    key_kwargs = dict(call_id=1, field_name="phone_number", context_window="ctx")

    first = resolver.resolve_field(
        candidate_spans=[_span("555-3212", 0.5), _span("555-4321", 0.1)], **key_kwargs
    )
    assert first is None

    second = resolver.resolve_field(
        candidate_spans=[_span("555-3212", 0.5), _span("555-4321", 0.1)], **key_kwargs
    )
    assert second is not None


def test_repeated_identical_settled_candidates_do_not_redecide():
    resolver = GlinerOnlyResolver(settle_ticks=1)
    key_kwargs = dict(call_id=1, field_name="phone_number", context_window="ctx")

    first = resolver.resolve_field(
        candidate_spans=[_span("555-3212", 0.5), _span("555-4321", 0.1)], **key_kwargs
    )
    assert first is not None

    for _ in range(5):
        again = resolver.resolve_field(
            candidate_spans=[_span("555-3212", 0.5), _span("555-4321", 0.1)], **key_kwargs
        )
        assert again is None


def test_aged_out_candidate_keeps_last_known_score_for_settled_ranking():
    # Tick 1 reports both candidates; tick 2 only re-reports the top one (the runner-up has aged
    # out of the extraction stage's window this tick) -- but since the distinct-candidate set
    # never shrinks, the runner-up must still be ranked using its last-known score from tick 1,
    # not silently reset to 0.
    resolver = GlinerOnlyResolver(settle_ticks=2)
    key_kwargs = dict(call_id=1, field_name="phone_number", context_window="ctx")

    first = resolver.resolve_field(
        candidate_spans=[_span("555-3212", 0.5), _span("555-4321", 0.3)], **key_kwargs
    )
    assert first is None  # not yet settled

    second = resolver.resolve_field(candidate_spans=[_span("555-3212", 0.6)], **key_kwargs)

    assert second is not None
    assert second.decision_reason == "margin_winner"
    assert second.candidate == "555-3212"
    assert second.confidence == 0.6
    assert set(second.distinct_candidates) == {"555-3212", "555-4321"}


# --- 2+ distinct candidates: margin/floor outcomes ------------------------------------------


def test_settled_top_candidate_wins_when_floor_and_margin_both_clear():
    resolver = GlinerOnlyResolver(settle_ticks=1)

    result = resolver.resolve_field(
        call_id=1,
        field_name="caller_name",
        candidate_spans=[_span("Jordan Lee", 0.6), _span("Jordan Leigh", 0.2)],
        context_window="ctx",
    )

    assert result is not None
    assert result.decision_reason == "margin_winner"
    assert result.candidate == "Jordan Lee"
    assert result.is_committed is True
    assert result.is_none_of_these is False


def test_settled_top_clears_floor_but_margin_too_close_commits_none_of_these():
    resolver = GlinerOnlyResolver(settle_ticks=1)

    result = resolver.resolve_field(
        call_id=1,
        field_name="caller_name",
        candidate_spans=[_span("Jordan Lee", 0.5), _span("Jordan Leigh", 0.45)],
        context_window="ctx",
    )

    assert result is not None
    assert result.decision_reason == "margin_too_close"
    assert result.candidate == "none_of_these"
    assert result.is_committed is True
    assert result.is_none_of_these is True
    assert result.confidence == 0.5


def test_settled_top_below_floor_commits_none_of_these():
    resolver = GlinerOnlyResolver(settle_ticks=1)

    result = resolver.resolve_field(
        call_id=1,
        field_name="caller_name",
        candidate_spans=[_span("Jordan Lee", 0.3), _span("Jordan Leigh", 0.1)],
        context_window="ctx",
    )

    assert result is not None
    assert result.decision_reason == "below_floor"
    assert result.candidate == "none_of_these"
    assert result.is_committed is True
    assert result.is_none_of_these is True
    assert result.confidence == 0.3


# --- self-correction bias ---------------------------------------------------------------------


def test_self_correction_bonus_flips_span_field_margin_decision():
    # Without the bonus, 0.5 vs 0.45 is a "margin_too_close" none-of-these; the self-correction
    # cue ("no wait") points at "555-4321", which appears after it -- the bonus should flip the
    # decision to a confident win for the corrected value.
    context = "it's 555-3212, no wait, it's actually 555-4321."
    start_a = context.find("555-3212")
    start_b = context.find("555-4321")
    resolver = GlinerOnlyResolver(settle_ticks=1)

    result = resolver.resolve_field(
        call_id=1,
        field_name="phone_number",
        candidate_spans=[
            _span("555-3212", 0.5, start_a, start_a + len("555-3212")),
            _span("555-4321", 0.45, start_b, start_b + len("555-4321")),
        ],
        context_window=context,
    )

    assert result is not None
    assert result.decision_reason == "margin_winner"
    assert result.candidate == "555-4321"
    assert result.confidence == 0.45 + config.GLINER_ONLY_SELF_CORRECTION_SCORE_BONUS


def test_no_self_correction_cue_present_does_not_apply_bonus():
    context = "the caller's number is 555-3212, and also mentioned 555-4321 as a backup."
    start_a = context.find("555-3212")
    start_b = context.find("555-4321")
    resolver = GlinerOnlyResolver(settle_ticks=1)

    result = resolver.resolve_field(
        call_id=1,
        field_name="phone_number",
        candidate_spans=[
            _span("555-3212", 0.5, start_a, start_a + len("555-3212")),
            _span("555-4321", 0.45, start_b, start_b + len("555-4321")),
        ],
        context_window=context,
    )

    assert result is not None
    assert result.decision_reason == "margin_too_close"
    assert result.is_none_of_these is True


def test_self_correction_bonus_flips_determination_field_using_freshest_classification():
    # Determination candidates carry no real span offsets, so the bias falls back to this tick's
    # freshest classification (the last entry in this tick's own candidate report) rather than
    # offset comparison.
    resolver = GlinerOnlyResolver(settle_ticks=1)

    result = resolver.resolve_field(
        call_id=1,
        field_name="permission_to_enter",
        candidate_spans=[_span("no", 0.5), _span("yes", 0.45)],
        context_window="caller said no, actually yes go ahead",
    )

    assert result is not None
    assert result.decision_reason == "margin_winner"
    assert result.candidate == "yes"
    assert result.confidence == 0.45 + config.GLINER_ONLY_SELF_CORRECTION_SCORE_BONUS


def test_self_correction_bonus_applies_to_aged_out_candidate_pointed_to_by_cue():
    # The corrected-to value ("555-9999") aged out of this tick's own candidate_spans -- GLiNER
    # didn't re-report it as a scored span -- but it's already a known distinct candidate from an
    # earlier tick, and this tick's own context literally re-speaks it after the cue. The bonus
    # must still find and apply to it via its cached display text, not just live spans.
    resolver = GlinerOnlyResolver(settle_ticks=2)
    key_kwargs = dict(call_id=1, field_name="phone_number")

    first = resolver.resolve_field(
        candidate_spans=[_span("555-3212", 0.5, 0, 8), _span("555-9999", 0.45, 20, 28)],
        context_window="tick one context, 555-3212 and 555-9999",
        **key_kwargs,
    )
    assert first is None  # not yet settled

    context = "555-3212, no wait, it's actually 555-9999"
    result = resolver.resolve_field(
        candidate_spans=[_span("555-3212", 0.5, 0, 8)],  # "555-9999" aged out this tick
        context_window=context,
        **key_kwargs,
    )

    assert result is not None
    assert result.decision_reason == "margin_winner"
    assert result.candidate == "555-9999"
    assert result.confidence == 0.45 + config.GLINER_ONLY_SELF_CORRECTION_SCORE_BONUS


def test_self_correction_cue_pointing_to_unknown_candidate_has_no_effect():
    # The cue is present but nothing in this tick's report starts after it -- the bonus target
    # resolves to nothing usable, so ranking falls back to the raw scores.
    context = "actually, never mind."
    resolver = GlinerOnlyResolver(settle_ticks=1)

    result = resolver.resolve_field(
        call_id=1,
        field_name="caller_name",
        candidate_spans=[_span("Jordan Lee", 0.5, 0, 10), _span("Jordan Leigh", 0.45, 0, 12)],
        context_window=context,
    )

    assert result is not None
    assert result.decision_reason == "margin_too_close"


# --- reopening a locked settled set on a later self-correction cue -----------------------------


def test_locked_none_of_these_is_reopened_by_a_later_self_correction_cue():
    resolver = GlinerOnlyResolver(settle_ticks=1)
    key_kwargs = dict(call_id=1, field_name="phone_number")

    locked = resolver.resolve_field(
        candidate_spans=[_span("555-3212", 0.5), _span("555-4321", 0.45)],
        context_window="the caller gave two numbers, 555-3212 and 555-4321",
        **key_kwargs,
    )
    assert locked is not None
    assert locked.decision_reason == "margin_too_close"
    assert locked.is_none_of_these is True

    # Same candidate set -- the gate alone would stay locked here, since no new distinct candidate
    # has appeared -- but this tick's context carries a self-correction cue pointing at "555-4321".
    context = "555-3212, no wait, it's actually 555-4321"
    start_b = context.find("555-4321")
    reopened = resolver.resolve_field(
        candidate_spans=[
            _span("555-3212", 0.5, 0, len("555-3212")),
            _span("555-4321", 0.45, start_b, start_b + len("555-4321")),
        ],
        context_window=context,
        **key_kwargs,
    )

    assert reopened is not None
    assert reopened.decision_reason == "margin_winner"
    assert reopened.candidate == "555-4321"
    assert reopened.is_committed is True
    assert reopened.is_none_of_these is False


def test_locked_margin_winner_is_not_reopened_by_a_later_self_correction_cue():
    # A confident commit is never re-litigated by this mechanism -- only a locked ambiguous
    # rejection is eligible.
    resolver = GlinerOnlyResolver(settle_ticks=1)
    key_kwargs = dict(call_id=1, field_name="phone_number")

    committed = resolver.resolve_field(
        candidate_spans=[_span("555-3212", 0.6), _span("555-4321", 0.1)],
        context_window="the caller gave 555-3212",
        **key_kwargs,
    )
    assert committed is not None
    assert committed.decision_reason == "margin_winner"
    assert committed.is_committed is True

    context = "no wait, it's actually 555-4321"
    start_b = context.find("555-4321")
    reopened = resolver.resolve_field(
        candidate_spans=[
            _span("555-3212", 0.6, 0, len("555-3212")),
            _span("555-4321", 0.1, start_b, start_b + len("555-4321")),
        ],
        context_window=context,
        **key_kwargs,
    )

    assert reopened is None
