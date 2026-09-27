"""One-off local script that regenerates `fixtures/sample_windows.json`.

Slices realistic ~200-char trailing windows out of this repo's own generated call corpus, the same
way `jev_live_transcription.gliner_pipeline._zero_shot_window` slices the transcript-so-far on
every tick, and pairs them with the project's 11-entry zero-shot label dict. The resulting fixture
is the only thing the `gliner_serve` benchmark's pod-side code reads -- this script only reads from
`jev_live_transcription.corpus`, `jev_live_transcription.gliner_pipeline`, and
`jev_live_transcription.config`; it never writes to anything under `src/jev_live_transcription/`.
"""

from __future__ import annotations

import json
from pathlib import Path

from jev_live_transcription import corpus
from jev_live_transcription.gliner_pipeline import ZERO_SHOT_FIELD_LABELS, _zero_shot_window

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_windows.json"

# Number of calls to sample from the corpus, spaced evenly across sorted call ids for scenario
# variety rather than clustering on whichever calls happen to sort first.
_NUM_CALLS = 10

# Fractions of each sampled call's rendered transcript length at which to take a trailing-window
# snapshot, approximating several different points in a call's progress -- the same way
# `_zero_shot_window` sees a growing transcript-so-far on each tick -- rather than only the single
# longest snapshot.
_TRUNCATION_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


def _sample_call_ids(call_ids: list[int], count: int) -> list[int]:
    if len(call_ids) <= count:
        return call_ids
    step = len(call_ids) / count
    return [call_ids[int(i * step)] for i in range(count)]


def generate_windows() -> list[str]:
    """Return de-duplicated trailing-window slices sampled across the corpus."""
    calls = corpus.load_all()
    call_ids = _sample_call_ids(sorted(calls), _NUM_CALLS)
    windows: list[str] = []
    seen: set[str] = set()
    for call_id in call_ids:
        full_text = corpus.render_turns(calls[call_id]["transcript_turns"])
        for fraction in _TRUNCATION_FRACTIONS:
            cutoff = max(1, int(len(full_text) * fraction))
            snapshot = full_text[:cutoff]
            window_text, _ = _zero_shot_window(snapshot)
            window_text = window_text.strip()
            if window_text and window_text not in seen:
                seen.add(window_text)
                windows.append(window_text)
    return windows


def main() -> None:
    windows = generate_windows()
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(
        json.dumps({"labels": ZERO_SHOT_FIELD_LABELS, "windows": windows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(windows)} windows to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
