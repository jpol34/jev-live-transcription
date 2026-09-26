"""Mocked tests for jev_pipeline -- no real typesafe.ai calls (kept fast/offline)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from typesafe_sdk import TypeSafeAPITimeoutError, TypeSafeAuthenticationError, TypeSafeRateLimitError

from jev_live_transcription import config
from jev_live_transcription.jev_pipeline import (
    JevFieldResolver,
    JevResolutionError,
    field_description,
    normalize_candidate,
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


def test_field_description_known_and_fallback():
    assert field_description("phone_number") == "phone number"
    assert field_description("weird_new_field") == "weird new field"


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
