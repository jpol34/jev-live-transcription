"""GPT-5.1 comparison arm for the live-transcription benchmark.

Completely independent of the GLiNER+jev pipeline: on every LLM-cadence tick,
the full 11-field extraction is done in a single Responses API call, with no
local model or jev resolver involved. Calls for a given `call_id` are chained
via `previous_response_id` so the server retains full prior context; only the
transcript delta since the last *successful* call is sent each time.

`usage.input_tokens` reports the full reconstructed context size on every
call, cached or not — it does not shrink as the chain grows. The evidence
that caching is active is `usage.input_tokens_details.cached_tokens`, a
separate field for the subset of those input tokens billed at a discount.
`extract` captures both and prices the call accordingly.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field

import openai
from openai import AsyncOpenAI

from . import config

MODEL_NAME = "gpt-5.1"
MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 15.0


def _is_retryable_status(status_code: int) -> bool:
    """Mirrors openai._base_client.BaseClient._should_retry's status-code rules.

    A status code the SDK itself won't retry has already reached us as a
    final, non-retried failure, so it's treated as permanent rather than
    transient.
    """
    return status_code in (408, 409, 429) or status_code >= 500


# The 11 target fields, shared with the GLiNER+jev pipeline for a fair,
# apples-to-apples comparison (see gliner_pipeline.PII_FIELD_LABELS /
# ZERO_SHOT_FIELD_LABELS for that side's equivalent field list).
FIELDS: tuple[str, ...] = (
    "caller_name",
    "email",
    "phone_number",
    "unit_number",
    "amenities_requested",
    "pet_info",
    "permission_to_enter",
    "work_order_issue",
    "move_in_date",
    "price_quoted",
    "budget_amount",
)

SYSTEM_INSTRUCTIONS = """You are extracting structured fields from a live phone call between a \
leasing/property-management agent and a caller. You receive the transcript in \
incremental chunks as the call progresses; each message is the new transcript \
text since your last reply, and you retain everything said earlier in the call.

For each of the following 11 fields, report the best value you can infer from \
the call so far (or null if nothing relevant has been said yet) and a \
confidence between 0 and 1 reflecting how sure you are that the value is \
correct and complete:

- caller_name: the caller's full name
- email: the caller's email address
- phone_number: the caller's phone number
- unit_number: the apartment/unit number discussed
- amenities_requested: amenities the caller asked about or requested
- pet_info: any pet type, breed, or description the caller mentioned
- permission_to_enter: the caller's stated permission (or refusal) to enter their unit
- work_order_issue: the maintenance or work order issue described
- move_in_date: a move-in or lease date mentioned
- price_quoted: a rent price or dollar amount quoted to the caller
- budget_amount: a budget or price range the caller says they can afford

Always answer for every field, using your best current understanding of the \
whole call so far, not just the latest chunk. It's fine for a confidence to \
move up or down as later chunks arrive."""


def _field_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "value": {"type": ["string", "null"]},
            "confidence": {"type": "number"},
        },
        "required": ["value", "confidence"],
        "additionalProperties": False,
    }


_RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {field_name: _field_schema() for field_name in FIELDS},
    "required": list(FIELDS),
    "additionalProperties": False,
}

_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "call_field_extraction",
    "schema": _RESPONSE_JSON_SCHEMA,
    "strict": True,
}


class PermanentLLMError(RuntimeError):
    """Raised when the API rejects a call in a way retrying won't fix.

    Covers auth/permission/bad-request/not-found style failures (e.g. the
    account not having access to `MODEL_NAME`) — the SDK's own retry loop
    already leaves these alone (see `_is_retryable_status`), so this just
    makes the failure loud instead of letting a generic exception through.
    """


@dataclass
class _ChainState:
    """Per-`call_id` Responses API chaining state.

    `last_good_offset` is the length of `transcript_snapshot` already
    delivered as of `last_good_response_id` — tracked separately from "an
    attempt was made" so a transient failure never causes the transcript
    content sent since the last success to be silently dropped: the next
    successful call's delta is computed from here, not from the failed
    attempt.
    """

    last_good_response_id: str | None = None
    last_good_offset: int = 0
    # The result `extract` returned for `last_good_response_id`, re-served
    # as-is (at zero additional cost) when a later tick has no new transcript
    # text to send — see the empty-delta branch in `extract`.
    last_result: dict | None = None
    # asyncio.Lock, not threading.Lock: this is held across an `await`
    # (the API call itself), and blocking the whole event-loop thread on a
    # contended threading.Lock there would stall every other call's tick,
    # not just this call_id's.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_chain_states: dict[str, _ChainState] = {}
_chain_states_guard = threading.Lock()

_client: AsyncOpenAI | None = None
_client_guard = threading.Lock()


def _get_state(call_id: str) -> _ChainState:
    with _chain_states_guard:
        state = _chain_states.get(call_id)
        if state is None:
            state = _ChainState()
            _chain_states[call_id] = state
        return state


def reset_call(call_id: str) -> None:
    """Drop chaining state for `call_id`, e.g. once a call finishes."""
    with _chain_states_guard:
        _chain_states.pop(call_id, None)


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        with _client_guard:
            if _client is None:
                _client = AsyncOpenAI(
                    max_retries=MAX_RETRIES, timeout=REQUEST_TIMEOUT_SECONDS
                )
    return _client


def _estimate_cost_usd(input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    pricing = config.PRICING_PER_MILLION_TOKENS[MODEL_NAME]
    price_in = pricing["input"] / 1_000_000
    price_out = pricing["output"] / 1_000_000
    uncached_tokens = input_tokens - cached_tokens
    return (
        uncached_tokens * price_in
        + cached_tokens * price_in * config.PROMPT_CACHE_DISCOUNT
        + output_tokens * price_out
    )


async def extract(call_id: str, transcript_snapshot: str) -> dict:
    """Run one chained GPT-5.1 extraction call for `call_id`.

    Sends only the transcript text new since the last call that got a
    completed response for this `call_id` (chained via
    `previous_response_id`), and returns the 11 fields, their self-reported
    confidences, `is_committed` per field, and usage/cost accounting. Raises
    `PermanentLLMError` on an unretryable API failure (bad request, auth,
    missing model, ...); any other exception raised before a response comes
    back means the SDK's own retries were exhausted, and this call's
    transcript delta will be re-sent (on top of anything new) the next time
    `extract` succeeds for this `call_id`. A response that completes but
    can't be parsed into the 11 fields still advances the chain — the delta
    was billed and is now part of the server-side context either way — and
    the parse error propagates instead.

    When `transcript_snapshot` has grown by nothing since the last successful
    call, no API call is made — the previous result is re-served at zero
    additional cost/latency, mirroring `gliner_pipeline`'s equivalent
    no-new-text skip.
    """
    state = _get_state(call_id)
    async with state.lock:
        if state.last_good_offset > len(transcript_snapshot):
            raise ValueError(
                f"transcript_snapshot for call_id={call_id!r} is shorter than "
                f"what was already sent for it ({len(transcript_snapshot)} "
                f"chars vs. {state.last_good_offset} already delivered) — call "
                f"reset_call({call_id!r}) before reusing this call_id for a "
                f"different transcript."
            )

        delta = transcript_snapshot[state.last_good_offset :]
        if not delta and state.last_result is not None:
            return state.last_result

        # Only reached with an empty `delta` on the very first call for a
        # `call_id` whose opening transcript snapshot is itself empty.
        input_text = delta if delta else "(no transcript content yet)"

        kwargs = {
            "model": MODEL_NAME,
            "instructions": SYSTEM_INSTRUCTIONS,
            "input": input_text,
            "text": {"format": _RESPONSE_FORMAT},
        }
        if state.last_good_response_id is not None:
            kwargs["previous_response_id"] = state.last_good_response_id

        client = _get_client()
        start = time.monotonic()
        try:
            response = await client.responses.create(**kwargs)
        except openai.APIStatusError as exc:
            if not _is_retryable_status(exc.status_code):
                raise PermanentLLMError(
                    f"gpt-5.1 call failed permanently ({type(exc).__name__}, "
                    f"status {exc.status_code}): {exc}"
                ) from exc
            raise
        latency_ms = (time.monotonic() - start) * 1000

        if response.status != "completed" or response.error is not None:
            raise RuntimeError(
                f"gpt-5.1 response did not complete cleanly: "
                f"status={response.status} error={response.error}"
            )

        # The call is billed and chained server-side as of here, regardless of
        # whether we can parse its body below — advance now so a parse
        # failure doesn't cause this same delta to be re-sent (and re-billed)
        # on the next call. `last_result` stays behind until parsing succeeds.
        state.last_good_response_id = response.id
        state.last_good_offset = len(transcript_snapshot)

        parsed = json.loads(response.output_text)
        fields = {name: parsed[name]["value"] for name in FIELDS}
        confidences = {name: float(parsed[name]["confidence"]) for name in FIELDS}
        is_committed = {
            name: confidences[name] >= config.JEV_COMMIT_THRESHOLD for name in FIELDS
        }

        usage = response.usage
        input_tokens = usage.input_tokens
        cached_tokens = usage.input_tokens_details.cached_tokens
        output_tokens = usage.output_tokens
        estimated_cost_usd = _estimate_cost_usd(input_tokens, cached_tokens, output_tokens)

        result = {
            "fields": fields,
            "confidences": confidences,
            "is_committed": is_committed,
            "response_id": response.id,
            "input_tokens": input_tokens,
            "cached_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": estimated_cost_usd,
            "latency_ms": latency_ms,
        }
        state.last_result = result

        return result
