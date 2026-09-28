"""Mocked tests for jev_pipeline -- no real typesafe.ai calls (kept fast/offline)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from typesafe_sdk import TypeSafeAPITimeoutError, TypeSafeAuthenticationError, TypeSafeRateLimitError

from jev_live_transcription import config
from jev_live_transcription.gliner_pipeline import FIELD_TAXONOMY
from jev_live_transcription.jev_pipeline import (
    JevFieldResolver,
    JevResolutionError,
    _noul_instructions,
    field_description,
    normalize_candidate,
    normalize_email_span,
)


def _noul_response(value: float, input_tokens: int = 10, output_tokens: int = 2):
    return SimpleNamespace(
        nouls={"field": SimpleNamespace(noul=value)},
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _choice_response(choice: str, confidence: float, input_tokens: int = 15, output_tokens: int = 3):
    return SimpleNamespace(
        choices={"field": SimpleNamespace(choice=choice, confidence=confidence)},
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _resolver(system_one: AsyncMock, *, settle_ticks: int = 1) -> JevFieldResolver:
    # `settle_ticks=1` (trigger on the very first observation) is the default here so tests that
    # aren't specifically about the settle gate can exercise a Choice call in a single
    # `resolve_field` call, as before; settle-gating behavior itself is covered separately below.
    fake_client = SimpleNamespace(system_one=system_one, aclose=AsyncMock())
    return JevFieldResolver(client=fake_client, settle_ticks=settle_ticks)


# --- normalization -----------------------------------------------------------------------------


def test_normalize_phone_number_digits_only():
    assert normalize_candidate("phone_number", "(555) 432-1123") == "5554321123"
    assert normalize_candidate("phone_number", "555-4321") == normalize_candidate(
        "phone_number", "5554321"
    )


def test_normalize_email_and_caller_name_lowercase_trimmed():
    assert normalize_candidate("email", "  Jordan@Example.com ") == "jordan@example.com"
    assert normalize_candidate("caller_name", "  Lindsey Perkins ") == "lindsey perkins"


def test_normalize_other_field_trimmed_text():
    assert normalize_candidate("unit_number", "  B-207 ") == "B-207"


def test_normalize_email_span_letter_by_letter_hyphenated():
    assert (
        normalize_email_span("e-t-h-a-n dot r-o-b-e-r-t-s at mail dot com")
        == "ethan.roberts@mail.com"
    )


def test_normalize_email_span_letter_by_letter_space_separated():
    assert (
        normalize_email_span("j o h n dot s m i t h at y a h o o dot c o m")
        == "john.smith@yahoo.com"
    )


def test_normalize_email_span_word_at_word_dot_word():
    assert normalize_email_span("jsmith at gmail dot com") == "jsmith@gmail.com"


def test_normalize_email_span_mixed_single_initial_and_words():
    assert normalize_email_span("j dot smith at yahoo dot com") == "j.smith@yahoo.com"


def test_normalize_email_span_already_literal_is_noop_passthrough():
    assert normalize_email_span("ethan.roberts@mail.com") == "ethan.roberts@mail.com"
    # Idempotent: normalizing an already-normalized value must not double-transform it.
    once = normalize_email_span("e-t-h-a-n dot r-o-b-e-r-t-s at mail dot com")
    assert normalize_email_span(once) == once


def test_normalize_email_span_preserves_real_hyphen_in_address():
    # Only a hyphen-joined run of single-character segments ("e-t-h-a-n") is a letter-spelling
    # separator; a real hyphen that's part of the address itself must survive.
    assert normalize_email_span("mary-jane at gmail dot com") == "mary-jane@gmail.com"
    assert normalize_email_span("jordan at big-corp dot com") == "jordan@big-corp.com"


def test_normalize_email_span_literal_case_is_lowercased_for_consistency():
    # The same address must normalize to the same literal string whether it arrives already
    # typed (mixed case) or spoken letter-by-letter (always lowercased by the tokenizing path) --
    # otherwise the two phrasings would commit as different strings.
    assert normalize_email_span("Ethan.Roberts@Mail.com") == "ethan.roberts@mail.com"


def test_normalize_email_span_literal_with_surrounding_whitespace_and_punctuation():
    assert normalize_email_span("  ethan.roberts@mail.com. ") == "ethan.roberts@mail.com"


def test_normalize_email_span_non_email_text_falls_back_unchanged():
    # Not actually an email span (no "at"/"dot" structure) -- returned unchanged rather than
    # mangled into something address-shaped.
    assert normalize_email_span("just some other text") == "just some other text"


def test_field_description_known_and_fallback():
    assert field_description("phone_number") == "phone number"
    assert field_description("weird_new_field") == "weird new field"


# --- taxonomy-gated Noul question template ---------------------------------------------------


def test_noul_instructions_span_field_uses_generic_is_x_correct_phrasing():
    instructions = _noul_instructions("caller_name", "Lindsey Perkins")

    assert "Lindsey Perkins" in instructions
    assert "is 'Lindsey Perkins' the caller's correct name?" in instructions


def test_noul_instructions_determination_field_asks_direct_yes_no_question():
    instructions = _noul_instructions("permission_to_enter", "yes")

    assert "did the caller give permission to enter the unit" in instructions
    assert "'yes'" in instructions
    # Not the generic span phrasing -- a determination isn't "the caller's correct X".
    assert "the caller's correct" not in instructions


def test_noul_instructions_determination_field_falls_back_without_explicit_question():
    # A hypothetical future "determination"-taxonomy field with no entry in
    # `_DETERMINATION_QUESTIONS` yet still gets a taxonomy-gated (not generic span) template,
    # built from its own field description.
    original = dict(FIELD_TAXONOMY)
    FIELD_TAXONOMY["future_determination_field"] = "determination"
    try:
        instructions = _noul_instructions("future_determination_field", "yes")
    finally:
        FIELD_TAXONOMY.clear()
        FIELD_TAXONOMY.update(original)

    assert "future determination field" in instructions
    assert "the caller's correct" not in instructions


# --- resolution decision logic -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_candidates_skips_jev_entirely():
    system_one = AsyncMock()
    resolver = _resolver(system_one)

    result = await resolver.resolve_field(
        call_id=1, field_name="caller_name", candidates=[], context_window="ctx"
    )

    assert result is None
    system_one.assert_not_called()


@pytest.mark.asyncio
async def test_single_candidate_calls_noul():
    system_one = AsyncMock(return_value=_noul_response(0.9))
    resolver = _resolver(system_one)

    result = await resolver.resolve_field(
        call_id=1, field_name="caller_name", candidates=["Lindsey Perkins"], context_window="ctx"
    )

    assert result is not None
    assert result.question_type == "noul"
    assert result.candidate == "Lindsey Perkins"
    assert result.confidence == 0.9
    assert result.is_committed is True
    system_one.assert_awaited_once()
    _, kwargs = system_one.call_args
    assert "Lindsey Perkins" in kwargs["questions"]["field"].instructions


@pytest.mark.asyncio
async def test_single_email_candidate_is_normalized_before_reaching_jev():
    system_one = AsyncMock(return_value=_noul_response(0.9))
    resolver = _resolver(system_one)

    result = await resolver.resolve_field(
        call_id=1,
        field_name="email",
        candidates=["e-t-h-a-n dot r-o-b-e-r-t-s at mail dot com"],
        context_window="ctx",
    )

    assert result is not None
    assert result.candidate == "ethan.roberts@mail.com"
    _, kwargs = system_one.call_args
    assert "ethan.roberts@mail.com" in kwargs["questions"]["field"].instructions


@pytest.mark.asyncio
async def test_single_candidate_determination_field_uses_direct_question():
    system_one = AsyncMock(return_value=_noul_response(0.9))
    resolver = _resolver(system_one)

    result = await resolver.resolve_field(
        call_id=1, field_name="permission_to_enter", candidates=["yes"], context_window="ctx"
    )

    assert result is not None
    assert result.candidate == "yes"
    assert result.is_committed is True
    _, kwargs = system_one.call_args
    instructions = kwargs["questions"]["field"].instructions
    assert "did the caller give permission to enter the unit" in instructions
    assert "the caller's correct" not in instructions


@pytest.mark.asyncio
async def test_single_candidate_below_threshold_not_committed_but_returned():
    system_one = AsyncMock(return_value=_noul_response(0.3))
    resolver = _resolver(system_one)

    result = await resolver.resolve_field(
        call_id=1, field_name="caller_name", candidates=["Lindsey Perkins"], context_window="ctx"
    )

    assert result is not None
    assert result.confidence == 0.3
    assert result.is_committed is False


@pytest.mark.asyncio
async def test_single_candidate_below_threshold_asks_again_next_tick():
    system_one = AsyncMock(return_value=_noul_response(0.3))
    resolver = _resolver(system_one)
    key_kwargs = dict(call_id=1, field_name="caller_name", context_window="ctx")

    await resolver.resolve_field(candidates=["Lindsey Perkins"], **key_kwargs)
    await resolver.resolve_field(candidates=["Lindsey Perkins"], **key_kwargs)

    assert system_one.await_count == 2


@pytest.mark.asyncio
async def test_single_candidate_already_committed_skips_further_calls():
    system_one = AsyncMock(return_value=_noul_response(0.95))
    resolver = _resolver(system_one)
    key_kwargs = dict(call_id=1, field_name="caller_name", context_window="ctx")

    first = await resolver.resolve_field(candidates=["Lindsey Perkins"], **key_kwargs)
    assert first is not None and first.is_committed is True

    second = await resolver.resolve_field(candidates=["Lindsey Perkins"], **key_kwargs)

    assert second is None
    system_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_duplicate_candidate_by_normalized_value_stays_single_candidate():
    system_one = AsyncMock(return_value=_noul_response(0.9))
    resolver = _resolver(system_one)

    result = await resolver.resolve_field(
        call_id=1,
        field_name="phone_number",
        candidates=["555-4321", "5554321"],  # same digits, different formatting
        context_window="ctx",
    )

    assert result is not None
    assert result.question_type == "noul"
    system_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_two_distinct_candidates_calls_choice_with_none_of_these():
    system_one = AsyncMock(return_value=_choice_response("555-4321", 0.85))
    resolver = _resolver(system_one)

    result = await resolver.resolve_field(
        call_id=1,
        field_name="phone_number",
        candidates=["555-3212", "555-4321"],
        context_window="ctx",
    )

    assert result is not None
    assert result.question_type == "choice"
    assert result.candidate == "555-4321"
    assert result.confidence == 0.85
    assert result.is_committed is True
    assert set(result.distinct_candidates) == {"555-3212", "555-4321"}

    _, kwargs = system_one.call_args
    criteria = kwargs["questions"]["field"].criteria
    assert set(criteria.keys()) == {"555-3212", "555-4321", "none_of_these"}


@pytest.mark.asyncio
async def test_self_correction_after_committed_noul_escalates_to_choice():
    system_one = AsyncMock(
        side_effect=[_noul_response(0.9), _choice_response("555-4321", 0.8)]
    )
    resolver = _resolver(system_one)
    key_kwargs = dict(call_id=1, field_name="phone_number", context_window="ctx")

    first = await resolver.resolve_field(candidates=["555-3212"], **key_kwargs)
    assert first is not None and first.question_type == "noul" and first.is_committed is True

    second = await resolver.resolve_field(candidates=["555-3212", "555-4321"], **key_kwargs)

    assert second is not None
    assert second.question_type == "choice"
    assert second.candidate == "555-4321"
    assert system_one.await_count == 2


@pytest.mark.asyncio
async def test_repeated_identical_choice_candidates_trigger_exactly_one_real_call():
    # GLiNER re-detects the same candidates most ticks -- repeated identical observations of an
    # already-resolved distinct-candidate set must not re-call jev.
    system_one = AsyncMock(return_value=_choice_response("555-4321", 0.9))
    resolver = _resolver(system_one)
    key_kwargs = dict(call_id=1, field_name="phone_number", context_window="ctx")

    first = await resolver.resolve_field(candidates=["555-3212", "555-4321"], **key_kwargs)
    assert first is not None

    for _ in range(5):
        again = await resolver.resolve_field(candidates=["555-3212", "555-4321"], **key_kwargs)
        assert again is None

    system_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_choice_settle_gate_requires_persistence_before_first_real_call():
    # With settle_ticks=2 (the config default), a candidate set must be observed unchanged on 2
    # consecutive ticks before it triggers a real jev call -- a single glitchy tick is not enough.
    system_one = AsyncMock(return_value=_choice_response("555-4321", 0.9))
    resolver = _resolver(system_one, settle_ticks=2)
    key_kwargs = dict(call_id=1, field_name="phone_number", context_window="ctx")

    first = await resolver.resolve_field(candidates=["555-3212", "555-4321"], **key_kwargs)
    assert first is None
    system_one.assert_not_awaited()

    second = await resolver.resolve_field(candidates=["555-3212", "555-4321"], **key_kwargs)
    assert second is not None
    system_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_choice_uncommitted_result_is_retried_next_tick_not_stuck():
    # A low-confidence Choice answer is not treated as "resolved" by the gate -- the set must
    # keep being retried every tick it's observed until jev returns a confident answer, matching
    # the Noul path's below-threshold retry behavior rather than getting silently stuck forever.
    system_one = AsyncMock(return_value=_choice_response("555-4321", 0.2))
    resolver = _resolver(system_one, settle_ticks=1)
    key_kwargs = dict(
        call_id=1,
        field_name="phone_number",
        candidates=["555-3212", "555-4321"],
        context_window="ctx",
    )

    first = await resolver.resolve_field(**key_kwargs)
    assert first is not None and first.is_committed is False

    second = await resolver.resolve_field(**key_kwargs)
    assert second is not None and second.is_committed is False
    assert system_one.await_count == 2


@pytest.mark.asyncio
async def test_self_correction_resolves_once_new_candidate_settles_bounded_latency():
    # The real self-correction scenario: a phone number is committed via Noul, then the caller
    # restates a different one. The Choice re-resolution should fire promptly once the new
    # candidate set settles -- at most `settle_ticks` ticks after it first appears, not never.
    system_one = AsyncMock(
        side_effect=[_noul_response(0.9), _choice_response("555-4321", 0.8)]
    )
    resolver = _resolver(system_one, settle_ticks=config.JEV_RECONFIRM_SETTLE_TICKS)
    key_kwargs = dict(call_id=1, field_name="phone_number", context_window="ctx")

    committed = await resolver.resolve_field(candidates=["555-3212"], **key_kwargs)
    assert committed is not None and committed.is_committed is True

    # The corrected value first appears alongside the original -- not settled yet, so no Choice
    # call fires on any of the first `settle_ticks - 1` ticks it's observed.
    for _ in range(config.JEV_RECONFIRM_SETTLE_TICKS - 1):
        pending = await resolver.resolve_field(
            candidates=["555-3212", "555-4321"], **key_kwargs
        )
        assert pending is None
    system_one.assert_awaited_once()

    settled = await resolver.resolve_field(candidates=["555-3212", "555-4321"], **key_kwargs)

    assert settled is not None
    assert settled.question_type == "choice"
    assert settled.candidate == "555-4321"
    assert system_one.await_count == 2


def test_commit_threshold_matches_config():
    assert config.JEV_COMMIT_THRESHOLD == 0.6


# --- per-candidate context caching ------------------------------------------------------------


@pytest.mark.asyncio
async def test_noul_uses_candidates_own_context_window_not_field_fallback():
    system_one = AsyncMock(return_value=_noul_response(0.9))
    resolver = _resolver(system_one)

    await resolver.resolve_field(
        call_id=1,
        field_name="caller_name",
        candidates=["Lindsey Perkins"],
        context_window="unrelated fallback text",
        candidate_context_windows={"Lindsey Perkins": "My name is Lindsey Perkins."},
    )

    _, kwargs = system_one.call_args
    assert kwargs["state"]["context_window"] == "My name is Lindsey Perkins."


@pytest.mark.asyncio
async def test_choice_context_includes_each_candidates_own_cached_snippet_after_aging_out():
    # Simulates ticket #19's GLiNER windowing: the only tick that ever reports "555-3212" is the
    # first one -- by the time "555-4321" is detected many ticks later, the candidate-extraction
    # stage's bounded window no longer covers "555-3212"'s original mention, so that later tick's
    # own candidate list and per-candidate context map only ever include "555-4321". The Choice
    # call must still see "555-3212"'s original justifying text, cached from the first tick.
    system_one = AsyncMock(
        side_effect=[
            _noul_response(0.9),
            _choice_response("555-4321", 0.85),
        ]
    )
    resolver = _resolver(system_one)

    first = await resolver.resolve_field(
        call_id=1,
        field_name="phone_number",
        candidates=["555-3212"],
        context_window="fallback tick 1",
        candidate_context_windows={"555-3212": "Caller: my number is 555-3212."},
    )
    assert first is not None and first.question_type == "noul"

    # Many ticks later: "555-3212" has aged out of GLiNER's window and is no longer reported at
    # all this tick -- only the genuinely new candidate is.
    result = await resolver.resolve_field(
        call_id=1,
        field_name="phone_number",
        candidates=["555-4321"],
        context_window="fallback tick N",
        candidate_context_windows={"555-4321": "Caller: actually it's 555-4321."},
    )

    assert result is not None
    assert result.question_type == "choice"
    assert set(result.distinct_candidates) == {"555-3212", "555-4321"}

    _, kwargs = system_one.call_args
    combined_context = kwargs["state"]["context_window"]
    assert "Caller: my number is 555-3212." in combined_context
    assert "Caller: actually it's 555-4321." in combined_context


@pytest.mark.asyncio
async def test_candidate_context_cached_on_first_sighting_is_not_overwritten_later():
    system_one = AsyncMock(
        side_effect=[_noul_response(0.3), _noul_response(0.9)]
    )
    resolver = _resolver(system_one)
    key_kwargs = dict(call_id=1, field_name="caller_name")

    await resolver.resolve_field(
        candidates=["Someone"],
        context_window="first tick fallback",
        candidate_context_windows={"Someone": "original justifying sentence"},
        **key_kwargs,
    )
    await resolver.resolve_field(
        candidates=["Someone"],
        context_window="second tick fallback",
        candidate_context_windows={"Someone": "a different sentence entirely"},
        **key_kwargs,
    )

    _, kwargs = system_one.call_args
    assert kwargs["state"]["context_window"] == "original justifying sentence"


@pytest.mark.asyncio
async def test_fallback_context_is_never_permanently_cached_over_a_later_precise_one():
    # A candidate's first sighting has no per-candidate snippet (e.g. its span was missing
    # start/end that tick), so it must fall back to the field-level context_window for that call
    # only -- not lock that fallback in forever. Once a later tick supplies the candidate's own
    # precise snippet, that should get cached and used instead.
    system_one = AsyncMock(side_effect=[_noul_response(0.3), _noul_response(0.9)])
    resolver = _resolver(system_one)
    key_kwargs = dict(call_id=1, field_name="caller_name", candidates=["Someone"])

    await resolver.resolve_field(context_window="whole transcript fallback", **key_kwargs)
    await resolver.resolve_field(
        context_window="whole transcript fallback",
        candidate_context_windows={"Someone": "Caller: my name is Someone."},
        **key_kwargs,
    )

    _, kwargs = system_one.call_args
    assert kwargs["state"]["context_window"] == "Caller: my name is Someone."


@pytest.mark.asyncio
async def test_missing_candidate_context_map_falls_back_to_field_context_window():
    system_one = AsyncMock(return_value=_noul_response(0.9))
    resolver = _resolver(system_one)

    await resolver.resolve_field(
        call_id=1, field_name="caller_name", candidates=["Someone"], context_window="ctx"
    )

    _, kwargs = system_one.call_args
    assert kwargs["state"]["context_window"] == "ctx"


# --- retries and failures -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transient_error_then_success_retries_and_succeeds(monkeypatch):
    monkeypatch.setattr("jev_live_transcription.jev_pipeline.asyncio.sleep", AsyncMock())
    system_one = AsyncMock(
        side_effect=[TypeSafeAPITimeoutError(timeout=6.0), _noul_response(0.9)]
    )
    resolver = _resolver(system_one)

    result = await resolver.resolve_field(
        call_id=1, field_name="caller_name", candidates=["Someone"], context_window="ctx"
    )

    assert result is not None
    assert result.confidence == 0.9
    assert system_one.await_count == 2


@pytest.mark.asyncio
async def test_permanent_error_raises_immediately_without_retry(monkeypatch):
    sleep_mock = AsyncMock()
    monkeypatch.setattr("jev_live_transcription.jev_pipeline.asyncio.sleep", sleep_mock)
    auth_error = TypeSafeAuthenticationError(status=401, body=None, headers={})
    system_one = AsyncMock(side_effect=auth_error)
    resolver = _resolver(system_one)

    with pytest.raises(JevResolutionError) as exc_info:
        await resolver.resolve_field(
            call_id=1, field_name="caller_name", candidates=["Someone"], context_window="ctx"
        )

    assert exc_info.value.cause is auth_error
    system_one.assert_awaited_once()
    sleep_mock.assert_not_called()


@pytest.mark.asyncio
async def test_transient_error_exhausts_retries_and_raises(monkeypatch):
    monkeypatch.setattr("jev_live_transcription.jev_pipeline.asyncio.sleep", AsyncMock())
    rate_limit_error = TypeSafeRateLimitError(status=429, body=None, headers={})
    system_one = AsyncMock(side_effect=rate_limit_error)
    resolver = _resolver(system_one)

    with pytest.raises(JevResolutionError) as exc_info:
        await resolver.resolve_field(
            call_id=1, field_name="caller_name", candidates=["Someone"], context_window="ctx"
        )

    assert exc_info.value.cause is rate_limit_error
    # Initial attempt + 2 retries = 3 total calls.
    assert system_one.await_count == 3


@pytest.mark.asyncio
async def test_timeout_is_treated_as_transient_and_retried(monkeypatch):
    monkeypatch.setattr("jev_live_transcription.jev_pipeline.asyncio.sleep", AsyncMock())
    system_one = AsyncMock(
        side_effect=[TypeSafeAPITimeoutError(timeout=6.0), _noul_response(0.9)]
    )
    resolver = _resolver(system_one)

    result = await resolver.resolve_field(
        call_id=1, field_name="caller_name", candidates=["Someone"], context_window="ctx"
    )

    assert result is not None
    assert system_one.await_count == 2
    # The timeout is passed to the SDK's own per-call `timeout=` rather than wrapped in
    # asyncio.wait_for, so the shared client's pooled connection is never cancelled externally.
    _, kwargs = system_one.call_args
    assert kwargs["timeout"] == 6.0


# --- none_of_these and phone-number country-code normalization ------------------------------


@pytest.mark.asyncio
async def test_choice_none_of_these_is_flagged_not_treated_as_a_real_candidate():
    system_one = AsyncMock(return_value=_choice_response("none_of_these", 0.9))
    resolver = _resolver(system_one)

    result = await resolver.resolve_field(
        call_id=1,
        field_name="phone_number",
        candidates=["555-3212", "555-4321"],
        context_window="ctx",
    )

    assert result is not None
    assert result.candidate == "none_of_these"
    assert result.is_none_of_these is True


@pytest.mark.asyncio
async def test_choice_real_candidate_is_not_flagged_as_none_of_these():
    system_one = AsyncMock(return_value=_choice_response("555-4321", 0.9))
    resolver = _resolver(system_one)

    result = await resolver.resolve_field(
        call_id=1,
        field_name="phone_number",
        candidates=["555-3212", "555-4321"],
        context_window="ctx",
    )

    assert result is not None
    assert result.is_none_of_these is False


def test_normalize_phone_number_strips_leading_country_code():
    assert normalize_candidate("phone_number", "+1 555-432-1123") == normalize_candidate(
        "phone_number", "555-432-1123"
    )


@pytest.mark.asyncio
async def test_country_code_variant_stays_single_candidate_not_choice():
    system_one = AsyncMock(return_value=_noul_response(0.9))
    resolver = _resolver(system_one)

    result = await resolver.resolve_field(
        call_id=1,
        field_name="phone_number",
        candidates=["+1 555-432-1123", "555-432-1123"],
        context_window="ctx",
    )

    assert result is not None
    assert result.question_type == "noul"
    system_one.assert_awaited_once()


# --- concurrency safety -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_calls_for_same_key_are_serialized():
    async def system_one(*args, **kwargs):
        await asyncio.sleep(0.01)
        return _noul_response(0.9)

    resolver = _resolver(AsyncMock(side_effect=system_one))
    key_kwargs = dict(
        call_id=1, field_name="caller_name", candidates=["Someone"], context_window="ctx"
    )

    results = await asyncio.gather(
        resolver.resolve_field(**key_kwargs), resolver.resolve_field(**key_kwargs)
    )

    # The lock serializes the two calls for the same key: whichever runs first commits the
    # field, so the second sees an already-committed single candidate and skips its own jev
    # call entirely rather than racing the first for the write to `self._committed`.
    assert sum(r is not None for r in results) == 1
    resolver._client.system_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_calls_for_different_keys_run_concurrently():
    started = asyncio.Event()

    async def system_one(*args, **kwargs):
        started.set()
        await asyncio.sleep(0.01)
        return _noul_response(0.9)

    resolver = _resolver(AsyncMock(side_effect=system_one))

    results = await asyncio.gather(
        resolver.resolve_field(
            call_id=1, field_name="caller_name", candidates=["A"], context_window="ctx"
        ),
        resolver.resolve_field(
            call_id=2, field_name="caller_name", candidates=["B"], context_window="ctx"
        ),
    )

    assert all(r is not None for r in results)
    assert resolver._client.system_one.await_count == 2
