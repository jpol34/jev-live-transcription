# GLiNER checkpoint research: third-party usage and official documentation

Findings from researching how third-party projects and the official GLiNER project itself document
and use the two GLiNER checkpoints this project depends on
(`src/jev_live_transcription/gliner_pipeline.py`). Gathered to sanity-check this project's own
usage against both real-world consumption and upstream documentation.

## `knowledgator/gliner-stream-pii-v1.0` (streaming PII model)

Third-party adoption is thin but real: three genuine third-party consumers found. No GitHub issues
or discussions anywhere mention this checkpoint or its streaming API specifically.

**Real third-party repos:**
- **`gravitee-io/GLiNER4j`** (Apache-2.0) — a Java/JVM reimplementation via ONNX Runtime and
  llama.cpp, with a dedicated `StreamingSpanNer` façade built specifically for this checkpoint
  (`gliner4j-llamacpp/src/main/java/.../StreamingSpanNer.java`).
- **`slavadubrov/ner-field-guide`** — a `BufferedPII` class (`ner_demo/streaming.py`) wrapping the
  session API for PII-masking. Pins exactly `gliner==0.2.28`, added in a blanket reproducibility
  pin (alongside exact `transformers`/`torch` pins) for an article/demo, not in reaction to any
  known incompatibility — `0.2.29` didn't exist yet when that pin was made.
- **`CarlosZiegler/better-privacy`** — an internal licensing due-diligence doc (not code).

**Official documentation** (huggingface.co/knowledgator/gliner-stream-pii-v1.0): confirms this
project's usage pattern directly — `session_id=[id]` for cached incremental inference that reuses
decoder KV state/labels/words/span history, each call returning the complete current session
snapshot rather than only new entities (matches `_run_pii_tick`'s delta-only feeding), explicit
guidance to always clear finished sessions (matches `reset_call`'s `clear_session`), and a worked
code example using `threshold=0.5` with the same `model.inference([chunk], labels,
session_id=[id], threshold=0.5)` shape this project uses. The model card documents no
thread-safety/concurrency guarantees of its own.

**Thread-safety**, confirmed by reading the installed `gliner` package source directly: the
streaming `inference()` path's batch-execution method holds an internal `RLock` around its entire
body — cache read, forward pass, and cache write-back — which fully serializes calls across every
`session_id` at the library level. This is why `gliner_pipeline.py` relies on that guarantee rather
than adding its own redundant lock around the `inference()` call. Separately, `clear_session()`
goes through a different internal lock than the one guarding `inference()` batches; this project's
own call lifecycle never runs `reset_call()` concurrently with an in-flight tick for the same
call_id (each call's ticks run strictly sequentially, and teardown only happens after the tick loop
fully exits), so that library-level gap has no exploitable window here.

Knowledgator's own docs cookbook (docs.knowledgator.com) gives general first-party threshold
guidance for their PII checkpoints: start at 0.5, lower to 0.3-0.4 for redaction-sensitive use
cases, raise to 0.6-0.7 for precision-critical ones. This project keeps `threshold=0.5`, matching
both the model card's own worked example and every real third-party consumer found — the strongest
combination of official-example and real-consumer confirmation among any value tried here.

**Open item:** the model card declares no license, flagged independently by a third party's own
licensing due-diligence doc (as of 2026-07-30) — worth checking directly on the HF model card
before this ships anywhere beyond a benchmark.

## `urchade/gliner_medium-v2.1` (zero-shot domain model)

Widely adopted — 10+ real third-party production/benchmark consumers found, including Presidio's
own official PII-recognizer docs.

**Real third-party repos (non-exhaustive):** `FalkorDB/GraphRAG-SDK`, `DataFog/datafog-python`,
`C0oki3s/ScribdT`, `ksenia007/alvessa_agent`, `redhat-ai-americas/memory-hub`,
`pbernet/akka_streams_tutorial`, `hongbo-miao/hongbomiao.com`, `Kwaai-AI-Lab/KwaaiNet`,
`hopit-ai/Moda`, `Thibault-GAREL/LyRIDS_OPENER`, `zeroc00I/DontFeedTheAI`.

**Windowing is a correctness requirement, not just a latency optimization.** Confirmed directly
from `gliner`'s tokenization/data-processing source and the checkpoint's own config: `max_len: 384`
tokens, and GLiNER does **not** auto-window long input — text past `max_len` is silently truncated
as a **prefix cut** (keeps the beginning, drops the end), confirmed via still-open GitHub issues
(`urchade/GLiNER#378`, plus #82/#95/#185/#231) with no auto-windowing/sliding-window feature added
in any later release; the community's own recommended workaround is exactly the manual chunking
this project already does. For a growing live transcript, feeding the full text instead of a
trailing window would silently drop the most recent, most relevant content — exactly backwards.
This project's `GLINER_ZERO_SHOT_WINDOW_CHARS = 200` (~30-40 words) is well under the 384-token
ceiling, so no truncation risk at the current size.

**Threshold is an explicit, documented precision/recall dial.** GLiNER's official docs
(urchade.github.io/GLiNER/usage.html) document 0.7 as "high precision, lower recall," 0.3 as "lower
precision, higher recall," and 0.5 as the "balanced" default — with no particular justification for
0.5 beyond "balanced" (per the maintainer directly, in `urchade/GLiNER#100`: "I have used the
default 0.5 threshold value" for the paper's own benchmarks, no tuning rationale given). Microsoft
Presidio's own official GLiNER-based PII recognizer — a closely related checkpoint used for the
same shape of task this project has (extraction with no downstream confidence-verification step)
— defaults to **0.30**, with real users configuring 0.25-0.4 in practice. This project uses
`config.GLINER_ZERO_SHOT_THRESHOLD = 0.30` to match: a missed field is unrecoverable, while a
low-confidence false positive is just one more candidate for jev's resolution stage to weigh and
reject, so the asymmetry favors recall.

**Label descriptions.** GLiNER's `predict_entities`/`inference` accept labels either as a plain
list of strings or as a `{name: description}` mapping (added in GLiNER v0.2.29): the mapping key
comes back as `entity["label"]` in results, while the value is the actual prompt text fed to the
model for zero-shot matching. This project passes `ZERO_SHOT_FIELD_LABELS` directly as such a
mapping — its field names are the returned labels, and its existing longer, sentence-style values
(e.g. "statement about permission to enter the unit") are the model-facing prompts, which official
docs confirm benefit from being specific ("specific labels work better than generic ones") even
though most third-party examples found use short 1-3 word noun phrases for the returned label
itself. This mechanism is three weeks old as of this research and has no third-party adoption yet
to validate it further, but it's the current, official way to keep a clean returned label while
still feeding the model a detailed zero-shot prompt.

**`multi_label=True` is correct** for this project's usage: official docs confirm it allows one
span to receive multiple labels simultaneously, appropriate since this project's fields aren't
mutually exclusive categories.

## GLiNER library version compatibility

`v0.2.28` (2026-07-24) is the release that added streaming/session support at all — confirmed
directly in that release's changelog — so `pyproject.toml`'s `gliner>=0.2.29` floor (bumped from
`>=0.2.28` to pick up the label-description mapping feature above) requires nothing older than
streaming support ever needed anyway. `v0.2.29` (2026-09-08, the latest release and this project's
installed version) is additive only — OpenVINO export, label descriptions, contextual embeddings,
batching/training/offline-loading fixes — with no breaking changes to `from_pretrained`,
`map_location`, `predict_entities`, or the streaming `inference()`/`session_id`/`clear_session`
API. No upper-bound version ceiling is needed based on actual release notes.

`from_pretrained(..., map_location=<device>)` remains the current, live API for device placement —
no newer replacement exists. Both inference paths this project uses already wrap themselves in
`torch.no_grad()`/`torch.inference_mode()` internally, so `gliner_pipeline.py` doesn't need to add
its own.

Installing `gliner[gpu]` (the ONNX Runtime GPU path) is avoided entirely: a real filed issue
(`urchade/GLiNER#267`) reproduces `onnxruntime`'s CPU package silently winning over
`onnxruntime-gpu` even when both are installed, with no error raised — confirmed still unresolved
in the library, not just a rumor.

## Research method

Both checkpoints were researched via GitHub code/repo/issue search for real third-party usage,
cloning promising repos to inspect actual source rather than trusting model-card blurbs, reading
official GLiNER/Presidio/Knowledgator documentation directly, and reading the installed `gliner`
package's own source for behavior no documentation covers (thread-safety, lock scope). Full agent
transcripts are not preserved; this file is the synthesized findings.
