"""Loads the generated call corpus from `output/transcripts` and `output/metadata`.

Each call is a matched pair of files sharing a `{id:03d}_{subtype}` stem: a
`.txt` transcript (a couple of `#` comment lines followed by `Speaker: text`
turns) in `output/transcripts/`, and a `.json` record in `output/metadata/`
shaped `{"scenario": {...}, "metadata": {...}}`, written by `scripts/generate.py`.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TRANSCRIPTS_DIR = ROOT / "output" / "transcripts"
METADATA_DIR = ROOT / "output" / "metadata"


_SPEAKERS = ("Agent", "Caller")


def _parse_transcript(text: str) -> list[dict]:
    turns: list[dict] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        speaker, sep, content = stripped.partition(": ")
        if sep and speaker in _SPEAKERS:
            turns.append({"speaker": speaker, "text": content})
        elif turns:
            # A line break inside a turn's text (no "Speaker: " prefix) —
            # append it to the turn in progress instead of dropping it.
            turns[-1]["text"] += "\n" + stripped
    return turns


def load_all(
    transcripts_dir: Path = TRANSCRIPTS_DIR, metadata_dir: Path = METADATA_DIR
) -> dict[int, dict]:
    """Return `{call_id: {"scenario", "ground_truth", "transcript_turns"}}` for every call."""
    calls: dict[int, dict] = {}
    for meta_path in sorted(metadata_dir.glob("*.json")):
        record = json.loads(meta_path.read_text(encoding="utf-8"))
        scenario = record["scenario"]
        transcript_path = transcripts_dir / f"{meta_path.stem}.txt"
        transcript_turns = _parse_transcript(transcript_path.read_text(encoding="utf-8"))
        calls[scenario["id"]] = {
            "scenario": scenario,
            "ground_truth": record["metadata"],
            "transcript_turns": transcript_turns,
        }
    return calls
