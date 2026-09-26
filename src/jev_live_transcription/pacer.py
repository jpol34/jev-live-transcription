"""Replays a completed call transcript as if it were arriving live.

Each word in the transcript is assigned a synthetic release timestamp so that,
at any simulated tick, `CallPacer.snapshot_at` can return the exact
"transcript-so-far" text a live captioning system would have produced by that
point. Timing is deterministic per `call_id` so a benchmark run is
reproducible.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass, field

from . import config

# Each word's gap is the target mean gap times a uniform jitter factor in this
# range, so per-word timing looks natural while the long-run average across a
# full transcript still converges to the sampled target words-per-minute.
_JITTER_RANGE = (0.6, 1.4)


@dataclass(frozen=True)
class WordEvent:
    """One word's position in the transcript and when it is released."""

    turn_index: int
    word: str
    release_time: float  # simulated seconds since call start


@dataclass
class CallPacer:
    """Builds and replays a word-release timeline for one call's transcript."""

    call_id: int
    transcript_turns: list[dict]
    wpm_range: tuple[int, int] = config.WPM_RANGE

    speakers: list[str] = field(init=False)
    events: list[WordEvent] = field(init=False)
    total_seconds: float = field(init=False)
    total_ticks: int = field(init=False)

    def __post_init__(self) -> None:
        turn_words = [turn["text"].split() for turn in self.transcript_turns]
        self.speakers = [turn["speaker"] for turn in self.transcript_turns]
        self.events = self._build_timeline(turn_words)
        self.total_seconds = self.events[-1].release_time if self.events else 0.0
        self.total_ticks = (
            math.ceil(self.total_seconds / config.TICK_SECONDS) if self.events else 0
        )

    def _build_timeline(self, turn_words: list[list[str]]) -> list[WordEvent]:
        rng = random.Random(self.call_id)
        target_wpm = rng.uniform(*self.wpm_range)
        mean_gap_seconds = 60.0 / target_wpm

        events: list[WordEvent] = []
        elapsed = 0.0
        for turn_index, words in enumerate(turn_words):
            for word in words:
                elapsed += mean_gap_seconds * rng.uniform(*_JITTER_RANGE)
                events.append(WordEvent(turn_index, word, elapsed))
        return events

    @property
    def average_wpm(self) -> float:
        """Combined words-per-minute across the whole transcript."""
        if not self.events or self.total_seconds <= 0:
            return 0.0
        return len(self.events) / (self.total_seconds / 60.0)

    def full_text(self) -> str:
        """The complete transcript text, once every word has been released."""
        return self._render(self.events)

    def snapshot_at(self, tick_number: int) -> tuple[str, int]:
        """Return `(transcript_text_so_far, character_offset)` for a simulated tick."""
        current_time = tick_number * config.TICK_SECONDS
        released = [event for event in self.events if event.release_time <= current_time]
        text = self._render(released)
        return text, len(text)

    def _render(self, events: list[WordEvent]) -> str:
        if not events:
            return ""
        lines: list[str] = []
        current_turn = events[0].turn_index
        current_words: list[str] = []
        for event in events:
            if event.turn_index != current_turn:
                lines.append(f"{self.speakers[current_turn]}: {' '.join(current_words)}")
                current_turn = event.turn_index
                current_words = []
            current_words.append(event.word)
        lines.append(f"{self.speakers[current_turn]}: {' '.join(current_words)}")
        return "\n".join(lines)


def iter_batch_ticks(pacer: CallPacer):
    """Yield `(tick_number, text, offset)` for every tick, back-to-back with no sleeping."""
    for tick in range(pacer.total_ticks + 1):
        text, offset = pacer.snapshot_at(tick)
        yield tick, text, offset


async def iter_realtime_ticks(pacer: CallPacer):
    """Yield `(tick_number, text, offset)` for every tick, sleeping to match wall-clock pace."""
    start = time.monotonic()
    for tick in range(pacer.total_ticks + 1):
        target = start + tick * config.TICK_SECONDS
        remaining = target - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(remaining)
        text, offset = pacer.snapshot_at(tick)
        yield tick, text, offset
