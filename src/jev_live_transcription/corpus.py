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


def _parse_transcript(text: str) -> list[dict]:
    turns = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        speaker, sep, content = line.partition(": ")
        if not sep:
            continue
        turns.append({"speaker": speaker, "text": content})
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
