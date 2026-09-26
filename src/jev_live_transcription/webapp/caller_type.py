"""Caller-type classification for the live-demo web page: infers whether the caller is a
Prospect, Resident, or Other from the transcript-so-far via a dedicated jev Choice call.

Demo-specific -- deliberately not part of `jev_pipeline.py`, which stays focused on the 11
benchmark fields. Reuses `JevFieldResolver`'s retry/timeout resilience (`_call_with_retry`,
`JevResolutionError`) rather than a bare `client.system_one(...)` call: a benchmark treats an
unhandled jev failure as data to record, but a live demo should degrade gracefully (hold the last
known classification, or stay "pending" if none yet) rather than crash or hang the WebSocket on a
jev hiccup.

Mirrors `jev_pipeline`'s Noul commit-and-hold pattern, not its Choice/SettleGate machinery -- once
confidently classified, the result is final for the rest of the call (this is a single
classification, not a correctable extracted value that might need to change), so there's nothing
to re-settle. Doesn't fire from tick 1 either: `MIN_TICKS`/`MIN_TRANSCRIPT_CHARS` gate the first
attempt so a jev call isn't burned on transcript too thin to classify from.
"""

from __future__ import annotations

from typing import Literal

from typesafe_sdk import Choice

from .. import config
from ..jev_pipeline import JevFieldResolver, JevResolutionError

CallerType = Literal["prospect", "resident", "other"]

OPTIONS: tuple[CallerType, ...] = ("prospect", "resident", "other")

# Minimum simulated ticks and transcript length before the first classification attempt --
# burning a jev call against a call's first couple of words wastes it on data too thin to
# classify from.
MIN_TICKS = 5
MIN_TRANSCRIPT_CHARS = 40

_INSTRUCTIONS = (
    "Based only on the call transcript so far, is the caller a Prospect (interested in renting "
    "an apartment they don't currently live in), a Resident (an existing tenant), or something "
    "else (Other)?"
)


class CallerTypeClassifier:
    """Tracks one call's caller-type classification across ticks.

    Not safe to share across concurrent calls -- construct a fresh instance per replay session,
    matching how one WebSocket session replays exactly one call.
    """

    def __init__(self, resolver: JevFieldResolver, call_id) -> None:
        self._resolver = resolver
        self._call_id = call_id
        self.value: CallerType | None = None
        self.confidence: float = 0.0
        self.committed: bool = False

    @property
    def status(self) -> CallerType | Literal["pending"]:
        return self.value if self.value is not None else "pending"

    async def classify(self, tick_number: int, transcript_snapshot: str) -> None:
        """Attempt a classification this tick, if warranted. No-ops once already committed, or
        before the minimum-signal threshold has accumulated."""
        if self.committed:
            return
        if tick_number < MIN_TICKS or len(transcript_snapshot) < MIN_TRANSCRIPT_CHARS:
            return

        try:
            response = await self._resolver._call_with_retry(
                call_id=self._call_id,
                field_name="caller_type",
                state={"context_window": transcript_snapshot},
                questions={
                    "field": Choice(
                        instructions=_INSTRUCTIONS,
                        criteria={option: None for option in OPTIONS},
                    ),
                },
            )
        except JevResolutionError:
            # Degrade gracefully: hold whatever was last known (possibly still "pending") rather
            # than crash or hang the WebSocket on a jev hiccup.
            return

        answer = response.choices["field"]
        if answer.confidence >= config.JEV_COMMIT_THRESHOLD:
            # Only a confident answer counts as a real classification -- a below-threshold guess
            # is discarded rather than shown, so the displayed status stays "pending" (not a
            # possibly-wrong flicker) until a genuinely confident call commits it.
            self.value = answer.choice
            self.confidence = answer.confidence
            self.committed = True
