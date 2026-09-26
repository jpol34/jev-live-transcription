# GLiNER checkpoint research: third-party usage

Findings from researching how third-party projects (not the official GLiNER repo, not the
checkpoint authors, not this project) actually use the two GLiNER checkpoints this project depends
on (`src/jev_live_transcription/gliner_pipeline.py`). Gathered to sanity-check this project's own
usage against real-world consumption, since the official docs for one of these checkpoints are
thin.

## `knowledgator/gliner-stream-pii-v1.0` (streaming PII model)

Third-party adoption is thin but real: three genuine third-party consumers found. No GitHub issues
or discussions anywhere mention this checkpoint or its streaming API specifically.

**Real third-party repos:**
- **`gravitee-io/GLiNER4j`** (Apache-2.0) — a Java/JVM reimplementation via ONNX Runtime and
  llama.cpp, with a dedicated `StreamingSpanNer` façade built specifically for this checkpoint
  (`gliner4j-llamacpp/src/main/java/.../StreamingSpanNer.java`).
- **`slavadubrov/ner-field-guide`** — a `BufferedPII` class (`ner_demo/streaming.py`) wrapping the
  session API for PII-masking. Pins exactly `gliner==0.2.28` for the streaming API to work at all.
- **`CarlosZiegler/better-privacy`** — an internal licensing due-diligence doc (not code).

**Validated against this project's usage:**
- **Delta-only feeding is correct.** Every real consumer feeds only new text per call and treats
  the response as the full current entity snapshot, not a delta to merge — matches
  `_run_pii_tick`.
- **Explicit `clear_session` on teardown is correct.** Both real consumers close sessions
  explicitly (context managers / try-with-resources); nobody relies on TTL eviction except the
  *official* GLiNER repo's own example (out of scope here, but the only TTL-based pattern found at
  all).
- **The global inference lock is independently validated, not overcaution.** GLiNER4j hit the
  identical constraint from a different angle (llama.cpp instead of `transformers`): concurrent
  calls across *different* sessions still share one forward pass through the same decoder, so they
  serialize through a dispatcher. Two independent implementations landing on "serialize
  cross-session calls" is strong confirmation this project's lock is required.
- **`threshold=0.5` matches every third-party example found.**

**Action items:**
1. **No declared license on the model card** (flagged by an unrelated third party's own licensing
   due-diligence doc, as of 2026-07-30) — worth checking directly on the HF model card before this
   ships anywhere beyond a benchmark.
2. **`pyproject.toml` pins `gliner>=0.2.28` (open floor, no ceiling).** The one confirmed
   real-world user of the streaming API pins exactly `0.2.28`. Given near-zero other adoption to
   have shaken out compatibility on later releases, an open-ended pin is riskier here than for a
   mainstream checkpoint — worth capping the upper bound.

## `urchade/gliner_medium-v2.1` (zero-shot domain model)

Widely adopted — 10+ real third-party production/benchmark consumers found, including Presidio's
own official PII-recognizer docs.

**Real third-party repos (non-exhaustive):** `FalkorDB/GraphRAG-SDK`, `DataFog/datafog-python`,
`C0oki3s/ScribdT`, `ksenia007/alvessa_agent`, `redhat-ai-americas/memory-hub`,
`pbernet/akka_streams_tutorial`, `hongbo-miao/hongbomiao.com`, `Kwaai-AI-Lab/KwaaiNet`,
`hopit-ai/Moda`, `Thibault-GAREL/LyRIDS_OPENER`, `zeroc00I/DontFeedTheAI`.

**Validated / refined against this project's usage:**
- **Windowing is a correctness requirement, not just a latency optimization.** Confirmed directly
  from `gliner/model.py` and the checkpoint's own config: `max_len: 384` tokens, and GLiNER does
  **not** auto-window long input — text past `max_len` is silently truncated as a **prefix cut**
  (keeps the beginning, drops the end), confirmed via a still-open GitHub issue
  (`urchade/GLiNER#378`, plus #82/#95/#185/#231 going back to 2024). For a growing live transcript,
  feeding the full text instead of a trailing window would silently drop the most recent, most
  relevant content — exactly backwards. This project's `GLINER_ZERO_SHOT_WINDOW_CHARS = 200`
  (~30-40 words) is well under the 384-token ceiling, so no truncation risk at the current size.
- **`threshold` is unset (library default 0.5) — worth reconsidering.** Real usage clusters
  *below* 0.5 (0.3-0.45) when recall matters and there's no downstream verification step. The one
  project using a *higher* threshold (0.75, FalkorDB) does so specifically because it hands the
  mid-confidence band to an LLM verifier, which this project doesn't have. Given false negatives
  (missing a unit number, price, pet info) plausibly cost more than false positives here, an
  explicit lower threshold may be worth testing.
- **Label phrasing diverges from real-world norms, with no precedent either way.** Every
  third-party label list found stays at short noun-phrase length (e.g. "phone number", "api key").
  This project's longer, sentence-style labels (e.g. "statement about permission to enter the
  unit") aren't necessarily wrong — GLiNER v2.1 has no separate description field, so packing more
  signal into the label string is legitimate — but nothing in real-world usage validates it
  specifically. Worth an empirical eval rather than assuming it behaves like short-label norms.
- **Minor, not actionable:** lowercase labels (what this project uses) match the safer real-world
  default; label order and label-set size can each perturb other labels' confidence scores slightly
  (a model quirk, not a bug, if score drift is ever observed after changing
  `ZERO_SHOT_FIELD_LABELS`).

## Research method

Both checkpoints were researched via GitHub code/repo/issue search for real third-party usage,
cloning promising repos to inspect actual source rather than trusting model-card blurbs. Full
agent transcripts are not preserved; this file is the synthesized findings.
