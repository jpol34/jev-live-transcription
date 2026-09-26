# jev-live-transcription

A benchmark testing whether a local GLiNER NER model plus a hosted "jev" resolver
model (typesafe.ai) can extract structured fields from a live phone call
transcript faster than a general LLM (GPT-5.1), while tracking accuracy,
latency, tokens, and cost for both approaches.

The corpus is 100 synthetic property-management call transcripts (leasing
prospect calls and resident calls: work orders, billing, complaints), each
paired with ground-truth structured fields (caller name, unit number, pet
info, work order issue, etc.) and a scenario descriptor (category, subtype,
edge case flag, target length, caller disclosure style).

## Project layout

- `src/jev_live_transcription/` — the installable package.
  - `secrets.py` — loads `OPENAI_API_KEY` and `TYPESAFE_API_KEY` from
    Strongbox into the process environment.
  - `config.py` — shared constants (call clock, concurrency, pricing
    estimates).
  - `corpus.py` — loads the 100 transcript/ground-truth pairs from `output/`.
  - `pacer.py`, `db.py`, `pipeline_core.py` — replays a call's transcript
    paced to wall-clock time (or as fast as possible for batch runs), and
    captures both pipelines' per-tick activity to a SQLite database.
  - `cli.py`, `tui.py` — the `jlt` command-line entry point; `jlt tui
    <call_id>` replays one call live in a terminal UI showing both
    pipelines' current committed field values side by side.
- `scripts/scenarios.py` — deterministic scenario builder for the 100 calls.
- `scripts/generate.py` — async OpenAI-based transcript + ground-truth
  generator that writes into `output/transcripts/` and `output/metadata/`.
- `output/transcripts/`, `output/metadata/` — the generated corpus (100 calls
  each, `{id:03d}_{subtype}` filename stems).
- `output/transcript_short/` — an earlier, shorter-format batch kept for
  reference only.

## Setup

```
uv venv
uv pip install -e .
```

Requires `pwsh` with the `Strongbox` module available, and Strongbox entries
for `OPENAI_API_KEY` (and, once the jev resolver is wired up, `TYPESAFE_API_KEY`).

## Regenerating the corpus

```
uv run python scripts/generate.py
```

Pulls `OPENAI_API_KEY` from Strongbox, generates all 100 calls concurrently,
and writes each transcript/metadata pair to `output/transcripts/` and
`output/metadata/` as it completes. A manifest and generation log land in
`output/`.

## Verifying the corpus loads

```
uv run python -c "from jev_live_transcription import corpus; print(len(corpus.load_all()))"
```

Should print `100`.

## Watching a call live

```
uv run jlt tui 036
```

Replays call `036`'s transcript paced to wall-clock time and shows both pipelines' current
committed value and confidence for all 11 fields, updating roughly once per simulated second.
Requires `OPENAI_API_KEY` and `TYPESAFE_API_KEY` (loaded from Strongbox, same as above). Writes
its capture DB to `output/tui_captures/call_<id>.db` by default (`--db-path` overrides this).

## Tests

```
uv run pytest
```
