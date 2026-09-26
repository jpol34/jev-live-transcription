"""Jev resolution stage: disambiguates/confirms GLiNER candidate spans for one field.

Given the candidate spans a (separate, upstream) GLiNER extraction stage found for a field at a
tick, this module decides whether and how to ask the hosted "jev" resolver (typesafe.ai) to
confirm a value, using only a local context window around the candidate(s) -- never the full
transcript. It is deliberately decoupled from the extraction stage: callers pass a plain list of
candidate strings per field, so this composes with whatever shape `extract_candidates` ends up
returning.

Per (call_id, field_name), distinct candidate values (deduped by a per-field-type normalized
form) accumulate across ticks and never shrink:

- 0 distinct candidates: no jev call is made.
- Exactly 1 distinct candidate, not yet committed: ask a yes/no ("Noul") question.
- More than 1 distinct candidate ever seen: ask a multiple-choice ("Choice") question over all
  of them plus "none_of_these" -- this is what lets a previously committed value change when the
  caller self-corrects (a phone number restated, for example).
- Exactly 1 distinct candidate, already committed: nothing new to resolve, so no call is made.

Failures (after retries are exhausted) are raised as `JevResolutionError` rather than degraded to
`None` -- this benchmark needs jev failures visible as data, unlike a best-effort production
caller that would rather fall back silently.
"""

import asyncio
import re
from dataclasses import dataclass
from typing import Literal

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    Noul,
    RetryPolicy,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
)

from jev_live_transcription import config

CallId = int | str

_TIMEOUT_SECONDS = 6.0
_BACKOFF_SECONDS = (0.5, 1.5)
_MAX_RETRIES = len(_BACKOFF_SECONDS)

_FIELD_DESCRIPTIONS = {
    "caller_name": "name",
    "phone_number": "phone number",
    "email": "email address",
    "unit_number": "unit number",
}

_DIGITS_RE = re.compile(r"\D+")

_NONE_OF_THESE = "none_of_these"


def field_description(field_name: str) -> str:
    """Human-readable description of a field for use in jev question text."""
    return _FIELD_DESCRIPTIONS.get(field_name, field_name.replace("_", " "))


def normalize_candidate(field_name: str, value: str) -> str:
    """Normalize a raw candidate value for dedup, per the field's type.

    `phone_number` compares digits only; `email` and `caller_name` compare case- and
    whitespace-insensitively; every other field compares trimmed text.
    """
    if field_name == "phone_number":
        return _DIGITS_RE.sub("", value)
    if field_name in ("email", "caller_name"):
        return value.strip().lower()
    return value.strip()


def _is_transient(exc: BaseException) -> bool:
    """Whether `exc` looks like a timeout, network error, or 429/5xx -- worth retrying.

    Auth/config/malformed-request failures (400/401/403/404/422, or a validation error on an
    otherwise-successful response) are not retried.
    """
    if isinstance(exc, TimeoutError):  # covers asyncio.wait_for and TypeSafeAPITimeoutError
        return True
    if isinstance(exc, TypeSafeAPIConnectionError):
        return True
    status = getattr(exc, "status", None)
    return isinstance(status, int) and (status == 429 or status >= 500)


class JevResolutionError(RuntimeError):
    """Raised when a jev call fails and retries are exhausted.

    Carries the field/call identity and the underlying cause so failures are visible as data
    rather than silently degrading to a missing result.
    """

    def __init__(self, call_id: CallId, field_name: str, cause: BaseException) -> None:
        super().__init__(
            f"jev resolution failed for call_id={call_id!r} field={field_name!r}: {cause}"
        )
        self.call_id = call_id
        self.field_name = field_name
        self.cause = cause


@dataclass(frozen=True)
class JevResolution:
    """The outcome of one jev call for one field at one tick.

    `is_committed` only reflects this call's confidence against `config.JEV_COMMIT_THRESHOLD`;
    tracking which value is "currently committed" across ticks (and holding it steady when a
    later call comes back under threshold) is the calling orchestrator's job.
    """

    call_id: CallId
    field_name: str
    question_type: Literal["noul", "choice"]
    candidate: str
    confidence: float
    is_committed: bool
    distinct_candidates: tuple[str, ...]
    input_tokens: int | None
    output_tokens: int | None


class JevFieldResolver:
    """Resolves candidate spans for fields across the life of one or more calls.

    Wraps one shared `AsyncTypeSafeClient`, safe to reuse concurrently across many async tasks:
    it's backed by an `httpx2.AsyncClient`, which manages its own pooled connections and is
    documented safe for concurrent requests from multiple coroutines.
    """

    def __init__(self, client: AsyncTypeSafeClient | None = None) -> None:
        # Retries are driven explicitly by `_call_with_retry` below (fixed backoff, transient
        # errors only) rather than the SDK's own default policy, so the benchmark has a single,
        # visible retry/timeout budget per call instead of two overlapping ones.
        self._client = client or AsyncTypeSafeClient(retry=RetryPolicy(max_retries=0))
        self._candidates: dict[tuple[CallId, str], dict[str, str]] = {}
        self._committed: dict[tuple[CallId, str], bool] = {}

    async def resolve_field(
        self,
        call_id: CallId,
        field_name: str,
        candidates: list[str],
        context_window: str,
    ) -> JevResolution | None:
        """Fold this tick's `candidates` into the field's dedup set and resolve if warranted.

        Returns `None` when no jev call is warranted this tick (no candidates seen yet, or a
        single already-committed candidate with nothing new to resolve).
        """
        key = (call_id, field_name)
        seen = self._candidates.setdefault(key, {})
        for raw in candidates:
            value = raw.strip()
            if not value:
                continue
            normalized = normalize_candidate(field_name, value)
            if normalized and normalized not in seen:
                seen[normalized] = value

        distinct = list(seen.values())
        if not distinct:
            return None

        if len(distinct) == 1:
            if self._committed.get(key, False):
                return None
            result = await self._resolve_noul(call_id, field_name, distinct[0], context_window)
        else:
            result = await self._resolve_choice(call_id, field_name, distinct, context_window)

        self._committed[key] = result.is_committed
        return result

    async def _resolve_noul(
        self, call_id: CallId, field_name: str, candidate: str, context_window: str
    ) -> JevResolution:
        description = field_description(field_name)
        response = await self._call_with_retry(
            call_id=call_id,
            field_name=field_name,
            state={"context_window": context_window},
            questions={
                "field": Noul(
                    instructions=(
                        f"Based only on the given context, is {candidate!r} the caller's "
                        f"correct {description}?"
                    ),
                ),
            },
        )
        confidence = response.nouls["field"].noul
        return JevResolution(
            call_id=call_id,
            field_name=field_name,
            question_type="noul",
            candidate=candidate,
            confidence=confidence,
            is_committed=confidence >= config.JEV_COMMIT_THRESHOLD,
            distinct_candidates=(candidate,),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def _resolve_choice(
        self, call_id: CallId, field_name: str, distinct: list[str], context_window: str
    ) -> JevResolution:
        description = field_description(field_name)
        criteria: dict[str, None] = {value: None for value in distinct}
        criteria[_NONE_OF_THESE] = None
        response = await self._call_with_retry(
            call_id=call_id,
            field_name=field_name,
            state={"context_window": context_window},
            questions={
                "field": Choice(
                    instructions=(
                        f"Based only on the given context, which of these is the caller's "
                        f"correct {description}? If the caller corrected themselves, favor the "
                        f"value they most recently confirmed."
                    ),
                    criteria=criteria,
                ),
            },
        )
        answer = response.choices["field"]
        return JevResolution(
            call_id=call_id,
            field_name=field_name,
            question_type="choice",
            candidate=answer.choice,
            confidence=answer.confidence,
            is_committed=answer.confidence >= config.JEV_COMMIT_THRESHOLD,
            distinct_candidates=tuple(distinct),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def _call_with_retry(self, *, call_id: CallId, field_name: str, state, questions):
        last_exc: BaseException | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return await asyncio.wait_for(
                    self._client.system_one(state=state, questions=questions),
                    timeout=_TIMEOUT_SECONDS,
                )
            except (TimeoutError, TypeSafeAPIError, TypeSafeAPIConnectionError) as exc:
                last_exc = exc
                if attempt == _MAX_RETRIES or not _is_transient(exc):
                    raise JevResolutionError(call_id, field_name, exc) from exc
                await asyncio.sleep(_BACKOFF_SECONDS[attempt])
        # Unreachable: the loop above always returns or raises.
        raise JevResolutionError(call_id, field_name, last_exc or RuntimeError("unknown failure"))

    async def aclose(self) -> None:
        await self._client.aclose()
