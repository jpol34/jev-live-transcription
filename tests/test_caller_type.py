"""Mocked tests for webapp.caller_type -- no real typesafe.ai calls (kept fast/offline)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from typesafe_sdk import TypeSafeAPITimeoutError

from jev_live_transcription import config
from jev_live_transcription.jev_pipeline import JevFieldResolver
from jev_live_transcription.webapp.caller_type import (
    MIN_TICKS,
    MIN_TRANSCRIPT_CHARS,
    CallerTypeClassifier,
)


def _choice_response(choice: str, confidence: float):
    return SimpleNamespace(
        choices={"field": SimpleNamespace(choice=choice, confidence=confidence)},
        usage=SimpleNamespace(input_tokens=15, output_tokens=3),
    )


def _resolver(system_one: AsyncMock) -> JevFieldResolver:
    fake_client = SimpleNamespace(system_one=system_one, aclose=AsyncMock())
    return JevFieldResolver(client=fake_client)


_LONG_SNAPSHOT = "Agent: hi. Caller: I'm looking for a two bedroom apartment to rent. " * 2


def test_classify_noops_before_min_ticks():
    resolver = _resolver(AsyncMock())
    classifier = CallerTypeClassifier(resolver, call_id=1)

    _run(classifier.classify(MIN_TICKS - 1, _LONG_SNAPSHOT))

    resolver._client.system_one.assert_not_called()
    assert classifier.status == "pending"


def test_classify_noops_before_min_transcript_length():
    resolver = _resolver(AsyncMock())
    classifier = CallerTypeClassifier(resolver, call_id=1)

    short_snapshot = "a" * (MIN_TRANSCRIPT_CHARS - 1)
    _run(classifier.classify(MIN_TICKS, short_snapshot))

    resolver._client.system_one.assert_not_called()
    assert classifier.status == "pending"


def test_classify_commits_when_confidence_meets_threshold():
    system_one = AsyncMock(return_value=_choice_response("prospect", config.JEV_COMMIT_THRESHOLD))
    resolver = _resolver(system_one)
    classifier = CallerTypeClassifier(resolver, call_id=1)

    _run(classifier.classify(MIN_TICKS, _LONG_SNAPSHOT))

    assert classifier.committed is True
    assert classifier.value == "prospect"
    assert classifier.status == "prospect"


def test_classify_holds_pending_when_below_threshold():
    system_one = AsyncMock(
        return_value=_choice_response("resident", config.JEV_COMMIT_THRESHOLD - 0.1)
    )
    resolver = _resolver(system_one)
    classifier = CallerTypeClassifier(resolver, call_id=1)

    _run(classifier.classify(MIN_TICKS, _LONG_SNAPSHOT))

    assert classifier.committed is False
    assert classifier.status == "pending"


def test_classify_keeps_calling_until_committed():
    system_one = AsyncMock(
        side_effect=[
            _choice_response("other", config.JEV_COMMIT_THRESHOLD - 0.2),
            _choice_response("other", config.JEV_COMMIT_THRESHOLD),
        ]
    )
    resolver = _resolver(system_one)
    classifier = CallerTypeClassifier(resolver, call_id=1)

    _run(classifier.classify(MIN_TICKS, _LONG_SNAPSHOT))
    assert classifier.committed is False
    _run(classifier.classify(MIN_TICKS + 1, _LONG_SNAPSHOT))
    assert classifier.committed is True

    assert system_one.call_count == 2


def test_classify_stops_calling_once_committed():
    system_one = AsyncMock(return_value=_choice_response("resident", config.JEV_COMMIT_THRESHOLD))
    resolver = _resolver(system_one)
    classifier = CallerTypeClassifier(resolver, call_id=1)

    _run(classifier.classify(MIN_TICKS, _LONG_SNAPSHOT))
    _run(classifier.classify(MIN_TICKS + 1, _LONG_SNAPSHOT))

    assert system_one.call_count == 1  # second call short-circuited: already committed


def test_classify_degrades_gracefully_on_jev_failure(monkeypatch):
    monkeypatch.setattr("jev_live_transcription.jev_pipeline.asyncio.sleep", AsyncMock())
    system_one = AsyncMock(side_effect=TypeSafeAPITimeoutError("timed out"))
    resolver = _resolver(system_one)
    classifier = CallerTypeClassifier(resolver, call_id=1)

    _run(classifier.classify(MIN_TICKS, _LONG_SNAPSHOT))  # must not raise

    assert classifier.committed is False
    assert classifier.status == "pending"


def _run(coro):
    import asyncio

    return asyncio.run(coro)
