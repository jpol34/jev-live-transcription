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
  - `webapp/` — a FastAPI stakeholder-demo web page: replays a corpus call
    live over a WebSocket, rendered as a leasing-office agent's screen pop
    (`app.py`, `caller_type.py`, `static/`). See "Stakeholder demo web page"
    below.
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

## Running the full corpus benchmark

```
uv run jlt gpu-run --ssh-key ~/.ssh/<key>
```

Creates a RunPod GPU pod, runs `jlt batch` against the full 100-call corpus on it (GLiNER on
CUDA, sequential by default), copies the resulting capture DB back to `data/benchmark.sqlite3`,
and tears the pod down. `--ssh-key` must point at the private half of a key pair registered on
the RunPod account; `--subset N` limits the run to the first N calls, and `--keep-pod` skips
teardown for debugging. GPU is the canonical device for this benchmark's numbers -- a local,
CPU-only run remains available via `jlt batch` directly, but its latency is not representative.

On an NVIDIA A100-SXM4-80GB, a single call's GLiNER forward pass averages ~19ms/tick at the current
`GLINER_ZERO_SHOT_WINDOW_CHARS=200` (mean latency rises to ~21ms/~25ms at the wider 400/800-char
windows measured and rejected -- see that constant's comment; a separate, higher figure appears in
`config.GLINER_CONCURRENCY`'s comment because that measurement runs under real multi-call
contention, not in isolation), against a ~226ms/tick CPU baseline
for the same model.

Score per-field recall against the corpus's ground truth from any capture DB:

```
uv run python scripts/score_recall.py data/benchmark.sqlite3
```

`--save <path>` writes a JSON snapshot for a later `--baseline <path>` comparison, which exits
non-zero if any field's recall drops by more than `--tolerance` (default 5%).

## Stakeholder demo web page

A separate, hosted web page that replays a corpus call tick-by-tick over a WebSocket, rendered as
a leasing-office agent's live screen pop -- a stakeholder demo, not a production tool. Fully
public, no auth gate; a `MAX_CONCURRENT_SESSIONS` cap and an `ActiveCallGuard` (one live session
per call_id at a time) bound concurrent load/cost instead.

Run it locally:

```
uv run uvicorn jev_live_transcription.webapp.app:app --reload
```

Requires `TYPESAFE_API_KEY` (loaded from Strongbox, same as above) -- no `OPENAI_API_KEY`, since
the GPT-5.1 baseline is never shown here. Open `http://127.0.0.1:8000/` to pick a call.

### Deployment

Deployed as its own Railway service (`railway.json` pins the Nixpacks builder explicitly, since
the repo's `Dockerfile` -- used for Part 1's GPU pod image, unrelated to this service -- would
otherwise be Railway's default auto-detected build path). After linking the service
(`railway link`), set its `TYPESAFE_API_KEY` by running `scripts/set-railway-webapp-secrets.ps1`
yourself -- it sources the value from Strongbox and never routes it through anything else.

## Tests

```
uv run pytest
```
