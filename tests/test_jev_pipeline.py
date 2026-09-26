"""Mocked tests for jev_pipeline -- no real typesafe.ai calls (kept fast/offline)."""

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


def _resolver(system_one: AsyncMock) -> JevFieldResolver:
    fake_client = SimpleNamespace(system_one=system_one, aclose=AsyncMock())
    return JevFieldResolver(client=fake_client)


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
async def test_more_than_one_distinct_always_calls_choice_even_when_committed():
    system_one = AsyncMock(return_value=_choice_response("555-4321", 0.9))
    resolver = _resolver(system_one)
    key_kwargs = dict(call_id=1, field_name="phone_number", context_window="ctx")

    await resolver.resolve_field(candidates=["555-3212", "555-4321"], **key_kwargs)
    await resolver.resolve_field(candidates=["555-3212", "555-4321"], **key_kwargs)

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

    async def slow_then_fast(*args, **kwargs):
        if slow_then_fast.calls == 0:
            slow_then_fast.calls += 1
            raise TimeoutError("simulated wait_for timeout")
        return _noul_response(0.9)

    slow_then_fast.calls = 0
    system_one = AsyncMock(side_effect=slow_then_fast)
    resolver = _resolver(system_one)

    result = await resolver.resolve_field(
        call_id=1, field_name="caller_name", candidates=["Someone"], context_window="ctx"
    )

    assert result is not None
    assert system_one.await_count == 2
