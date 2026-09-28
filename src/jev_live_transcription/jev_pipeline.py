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
  caller self-corrects (a phone number restated, for example). Gated by a `SettleGate` (see
  `settle_gate.py`) so a Choice call only fires once the distinct-candidate set itself has changed
  and persisted for `config.JEV_RECONFIRM_SETTLE_TICKS` consecutive observations -- GLiNER
  re-detects the same entities most ticks, so without this a field with 2+ distinct candidates
  would re-call jev on essentially every tick even when nothing has changed.
- Exactly 1 distinct candidate, already committed: nothing new to resolve, so no call is made.

Each distinct candidate's own context snippet is cached the first time it is detected, keyed by
its normalized value, and reused for every later call involving that candidate -- including once
the candidate-extraction stage stops re-detecting it on later ticks. A Noul question uses its one
candidate's cached snippet; a Choice question's context is the concatenation of every distinct
candidate's own cached snippet, not just whatever text the triggering tick's own candidates cover.

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
    JSONContent,
    Noul,
    Questions,
    RetryPolicy,
    SystemOneResponse,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
)

from jev_live_transcription import config
from jev_live_transcription.settle_gate import SettleGate

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

# A literal `user@domain.tld` address -- no whitespace, exactly one "@", at least one "." in the
# domain part. Used both to short-circuit already-canonical spans (no-op passthrough) and to
# validate a spoken-form span was normalized into something address-shaped before trusting it.
_LITERAL_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

_EMAIL_WORD_RE = re.compile(r"[a-z0-9]+")

_EMAIL_SYMBOL_WORDS = {"at": "@", "dot": "."}


def field_description(field_name: str) -> str:
    """Human-readable description of a field for use in jev question text."""
    return _FIELD_DESCRIPTIONS.get(field_name, field_name.replace("_", " "))


def normalize_email_span(value: str) -> str:
    """Collapse a spoken-form `email` span into canonical `user@domain.com` form.

    Callers speak an email address as words rather than reading a literal string, e.g. "It's
    ethan dot roberts, let me spell that, e-t-h-a-n dot r-o-b-e-r-t-s at mail dot com" -- GLiNER
    can detect the full span with good confidence, but the raw text can never string-match a
    ground-truth `user@domain.com` value without this normalization.

    Handles both hyphen-joined ("e-t-h-a-n") and space-separated ("e t h a n") letter-by-letter
    spelling by collapsing any run of single-character tokens into one word, then maps the "at"/
    "dot" words to `@`/`.`. Already-canonical text (a literal address with no whitespace) is
    returned unchanged -- this must be idempotent, since the same candidate value can be folded
    in again on a later tick after already being normalized once.

    Falls back to returning `value` unchanged if the result still doesn't look like an email
    address, rather than emitting a mangled string for a span that wasn't actually spoken email.
    """
    stripped = value.strip().strip(".,;:!?\"'")
    if not stripped:
        return value
    if _LITERAL_EMAIL_RE.match(stripped):
        return stripped

    tokens = _EMAIL_WORD_RE.findall(stripped.lower().replace("-", " "))
    if not tokens:
        return value

    collapsed: list[str] = []
    letter_run: list[str] = []
    for token in tokens:
        if len(token) == 1:
            letter_run.append(token)
            continue
        if letter_run:
            collapsed.append("".join(letter_run))
            letter_run = []
        collapsed.append(token)
    if letter_run:
        collapsed.append("".join(letter_run))

    candidate = "".join(_EMAIL_SYMBOL_WORDS.get(token, token) for token in collapsed)
    return candidate if _LITERAL_EMAIL_RE.match(candidate) else value


def normalize_candidate(field_name: str, value: str) -> str:
    """Normalize a raw candidate value for dedup, per the field's type.

    `phone_number` compares digits only, with a leading US country code ("1" prefix on an
    11-digit number) dropped so "+1 555-432-1123" and "555-432-1123" dedup to the same value;
    `email` and `caller_name` compare case- and whitespace-insensitively; every other field
    compares trimmed text.
    """
    if field_name == "phone_number":
        digits = _DIGITS_RE.sub("", value)
        if len(digits) == 11 and digits[0] == "1":
            digits = digits[1:]
        return digits
    if field_name in ("email", "caller_name"):
        return value.strip().lower()
    return value.strip()


def _is_transient(exc: BaseException) -> bool:
    """Whether `exc` looks like a timeout, network error, or 429/5xx -- worth retrying.

    Auth/config/malformed-request failures (400/401/403/404/422, or a validation error on an
    otherwise-successful response) are not retried.
    """
    if isinstance(exc, TimeoutError):  # TypeSafeAPITimeoutError subclasses TimeoutError
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

    `candidate` holds jev's raw selection, which for a `choice` question can be the synthetic
    `"none_of_these"` label rather than an actual candidate value -- callers must check
    `is_none_of_these` before treating `candidate` as a real field value.
    """

    call_id: CallId
    field_name: str
    question_type: Literal["noul", "choice"]
    candidate: str
    confidence: float
    is_committed: bool
    is_none_of_these: bool
    distinct_candidates: tuple[str, ...]
    input_tokens: int | None
    output_tokens: int | None


class JevFieldResolver:
    """Resolves candidate spans for fields across the life of one or more calls.

    Wraps one shared `AsyncTypeSafeClient`, safe to reuse concurrently across many async tasks:
    it's backed by an `httpx2.AsyncClient`, which manages its own pooled connections and is
    documented safe for concurrent requests from multiple coroutines.
    """

    def __init__(
        self,
        client: AsyncTypeSafeClient | None = None,
        *,
        settle_ticks: int = config.JEV_RECONFIRM_SETTLE_TICKS,
    ) -> None:
        # Retries are driven explicitly by `_call_with_retry` below (fixed backoff, transient
        # errors only) rather than the SDK's own default policy, so the benchmark has a single,
        # visible retry/timeout budget per call instead of two overlapping ones.
        self._client = client or AsyncTypeSafeClient(retry=RetryPolicy(max_retries=0))
        self._candidates: dict[tuple[CallId, str], dict[str, str]] = {}
        # Each distinct candidate's own context snippet, cached the first time it is detected and
        # never overwritten afterward -- keyed the same way as `_candidates` (per (call_id,
        # field_name), then by normalized candidate value) so it survives a candidate aging out of
        # whatever the extraction stage reports on later ticks.
        self._contexts: dict[tuple[CallId, str], dict[str, str]] = {}
        self._committed: dict[tuple[CallId, str], bool] = {}
        # Gates the Choice (multi-candidate) path only -- see `resolve_field` -- against
        # re-resolving on every tick a field's distinct-candidate set happens to be re-reported
        # unchanged.
        self._settle_gate: SettleGate[tuple[CallId, str], frozenset[str]] = SettleGate(settle_ticks)
        # Serializes resolve_field calls per (call_id, field_name): each call reads then awaits a
        # jev round-trip before writing self._committed, so concurrent calls for the same key
        # would otherwise race on which result's commit outcome wins. Different keys stay fully
        # concurrent.
        self._locks: dict[tuple[CallId, str], asyncio.Lock] = {}

    async def resolve_field(
        self,
        call_id: CallId,
        field_name: str,
        candidates: list[str],
        context_window: str,
        *,
        candidate_context_windows: dict[str, str] | None = None,
    ) -> JevResolution | None:
        """Fold this tick's `candidates` into the field's dedup set and resolve if warranted.

        `candidate_context_windows`, when given, maps each raw string in `candidates` to its own
        individually-sliced context snippet (the text surrounding just that candidate's span,
        distinct from any other candidate's). The first time a candidate's own snippet is
        available this way, it is cached under the candidate's normalized value and reused for
        every later call involving that candidate, regardless of what either argument holds on a
        later tick. A candidate missing from `candidate_context_windows` (or seen before this
        argument existed) falls back to `context_window` for that call only -- this fallback is
        never itself cached, so a later tick supplying the candidate's own snippet still gets
        cached once it's available, instead of being permanently shadowed by an earlier, less
        precise substitute.

        Returns `None` when no jev call is warranted this tick: no candidates seen yet, a single
        already-committed candidate with nothing new to resolve, or -- for 2+ distinct candidates
        -- the set hasn't yet settled per `SettleGate` (it will resolve within
        `config.JEV_RECONFIRM_SETTLE_TICKS` further ticks of the set persisting unchanged).
        """
        key = (call_id, field_name)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            seen = self._candidates.setdefault(key, {})
            contexts = self._contexts.setdefault(key, {})
            per_candidate = candidate_context_windows or {}
            for raw in candidates:
                value = raw.strip()
                if not value:
                    continue
                if field_name == "email":
                    value = normalize_email_span(value)
                normalized = normalize_candidate(field_name, value)
                if not normalized:
                    continue
                if normalized not in seen:
                    seen[normalized] = value
                if normalized not in contexts and raw in per_candidate:
                    contexts[normalized] = per_candidate[raw]

            distinct = list(seen.values())
            if not distinct:
                return None

            if len(distinct) == 1:
                if self._committed.get(key, False):
                    return None
                normalized = normalize_candidate(field_name, distinct[0])
                candidate_context = contexts.get(normalized) or per_candidate.get(
                    distinct[0], context_window
                )
                result = await self._resolve_noul(
                    call_id, field_name, distinct[0], candidate_context
                )
            else:
                candidate_set = frozenset(seen.keys())
                if not self._settle_gate.observe(key, candidate_set):
                    return None
                combined_context = self._combined_context(
                    field_name, distinct, contexts, per_candidate, context_window
                )
                result = await self._resolve_choice(call_id, field_name, distinct, combined_context)
                if result.is_committed:
                    # Only a confident answer counts as "resolved" for gating purposes -- an
                    # uncommitted (low-confidence) Choice call must keep being retried every tick
                    # this same set is observed, matching the Noul path's below-threshold retry
                    # behavior, rather than getting silently stuck unresolved until a genuinely
                    # new candidate appears.
                    self._settle_gate.record_resolved(key, candidate_set)

            self._committed[key] = result.is_committed
            return result

    @staticmethod
    def _combined_context(
        field_name: str,
        distinct: list[str],
        contexts: dict[str, str],
        per_candidate: dict[str, str],
        fallback: str,
    ) -> str:
        """Join every distinct candidate's own context snippet into one Choice-call context.

        Prefers each candidate's cached snippet; falls back to its snippet from this tick's own
        `per_candidate` map (uncached, e.g. offsets were missing on every sighting so far) and
        finally to the field-level `fallback` when neither is available (e.g. the candidate isn't
        present at all this tick). Snippets are kept in first-seen order and deduplicated by exact
        text, so two candidates whose spans landed in the same sentence don't repeat it, while a
        genuinely older candidate's snippet still appears alongside a newer one's even when they
        were detected many ticks apart.
        """
        ordered: list[str] = []
        for value in distinct:
            normalized = normalize_candidate(field_name, value)
            snippet = contexts.get(normalized) or per_candidate.get(value, fallback)
            if snippet not in ordered:
                ordered.append(snippet)
        return "\n\n".join(ordered)

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
            is_none_of_these=False,
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
            is_none_of_these=answer.choice == _NONE_OF_THESE,
            distinct_candidates=tuple(distinct),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def _call_with_retry(
        self,
        *,
        call_id: CallId,
        field_name: str,
        state: JSONContent,
        questions: Questions,
    ) -> SystemOneResponse:
        # The per-call timeout is passed to the SDK itself rather than wrapped in
        # `asyncio.wait_for`: cancelling a coroutine from outside mid-request can leave the
        # shared client's pooled connection in a stale state for the next caller to reuse,
        # whereas the SDK's own timeout raises `TypeSafeAPITimeoutError` through its own
        # cancellation path.
        last_exc: BaseException | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return await self._client.system_one(
                    state=state, questions=questions, timeout=_TIMEOUT_SECONDS
                )
            except (TypeSafeAPIError, TypeSafeAPIConnectionError) as exc:
                last_exc = exc
                if attempt == _MAX_RETRIES or not _is_transient(exc):
                    raise JevResolutionError(call_id, field_name, exc) from exc
                await asyncio.sleep(_BACKOFF_SECONDS[attempt])
        # Unreachable: the loop above always returns or raises.
        raise JevResolutionError(call_id, field_name, last_exc or RuntimeError("unknown failure"))

    async def aclose(self) -> None:
        await self._client.aclose()
