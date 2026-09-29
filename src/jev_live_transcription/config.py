"""Shared constants for the benchmark harness."""

from typing import Literal

# Device the GLiNER checkpoint loads onto: "auto" (resolved to "cuda" if available, else "cpu" --
# see gliner_pipeline._resolve_device), or an explicit "cpu"/"cuda" override.
GLINER_DEVICE = "auto"

# Simulated call clock.
TICK_SECONDS = 1
WPM_RANGE = (130, 160)

# How often (in ticks) the GPT-5.1 baseline is invoked against the
# accumulating transcript, vs. the local GLiNER model running every tick.
LLM_CADENCE_TICKS = 3

# Minimum jev resolver confidence to accept a field extraction as committed
# rather than held for a later tick.
JEV_COMMIT_THRESHOLD = 0.6

# Consecutive ticks a field's distinct-candidate set must persist unchanged before jev's
# multi-candidate ("Choice") path re-resolves it -- GLiNER re-detects the same entities most
# ticks, so gating on a settled set avoids re-confirming jev on every one of them.
JEV_RECONFIRM_SETTLE_TICKS = 2

# Maximum trailing character count of the transcript-so-far fed to the GLiNER model per tick.
# Bounding it to a recent sliding window, rather than the full growing transcript, keeps its
# per-tick latency flat regardless of call length instead of growing unboundedly -- the property
# this constant guarantees is flatness with call length, not a hard per-tick ceiling (occasional
# spikes above the ~400ms/tick target, relative to TICK_SECONDS=1, are still possible under
# contention). 200 is also the best of the candidates measured on real GPU data across all 11
# fields (scripts/measure_window_size.py, full corpus): wider windows (400/800 chars) cost
# negligible extra latency but net-hurt recall -- amenities_requested and move_in_date both
# decline sharply with more context, outweighing smaller gains elsewhere.
GLINER_ZERO_SHOT_WINDOW_CHARS = 200

# Floor confidence passed to the GLiNER model call itself, on GLiNER's 0-1 scale (library default
# is 0.5). Deliberately permissive: the real per-field cutoff is applied post-hoc in
# `gliner_pipeline._postprocess_entities` via `PER_FIELD_THRESHOLDS`/`GLINER_DEFAULT_FIELD_THRESHOLD`
# below, since fields differ in how confidently GLiNER scores their correct span (e.g.
# `price_quoted`'s correct candidate typically scores 0.05-0.30). This floor only needs to sit
# below the lowest per-field threshold in use so no field's candidates get cut before that
# per-field filter ever sees them.
GLINER_ZERO_SHOT_THRESHOLD = 0.05

# Default post-hoc confidence cutoff (`gliner_pipeline._postprocess_entities`) for any field with
# no entry in `gliner_pipeline.PER_FIELD_THRESHOLDS`. 0.30 is the same value Microsoft's own
# Presidio project uses in its official GLiNER-based PII recognizer, which has the same
# no-verification-step shape as this pipeline -- kept as the default since nothing downstream
# double-checks a candidate's confidence before jev resolution sees it: a field GLiNER never
# surfaces at all is unrecoverable, while a low-confidence false positive is just one more
# candidate for jev to weigh and reject.
GLINER_DEFAULT_FIELD_THRESHOLD = 0.30

# gliner_only pipeline arm: local commit-policy constants for `gliner_only_resolver`. Unlike
# jev's own thresholds above, these gate a purely local decision made from GLiNER's own candidate
# scores -- no jev/typesafe.ai call. Empirically tuned against the real 100-call corpus via
# `scripts/score_recall.py --pipeline gliner_only` (same discipline as
# `gliner_pipeline.PER_FIELD_THRESHOLDS`) -- see each constant's comment below for what the data
# showed. The dominant pattern across nearly every field: a settled 2+-candidate commit
# (`margin_winner`) is far more reliable than a single-candidate commit (`single_above_floor`,
# which has no settle-gate or margin check at all -- it fires the instant one candidate clears its
# floor) -- e.g. `caller_name`'s single-candidate commits were 0% correct (0/15) against
# `margin_winner`'s 65% (55/85) in the tuning run; `email` and `unit_number` showed the same shape.
# `GLINER_ONLY_COMMIT_THRESHOLD` gates both paths identically (no separate single-candidate floor
# exists among these four constants), so raising it trades away the worst single-candidate noise at
# the cost of also raising the bar for `margin_winner`.

# Default minimum GLiNER confidence a field's single top candidate must clear to commit. Distinct
# from `gliner_pipeline.PER_FIELD_THRESHOLDS`/`GLINER_DEFAULT_FIELD_THRESHOLD` (which gate whether a
# candidate ever reaches the resolver at all) and from `JEV_COMMIT_THRESHOLD` above (which gates a
# hosted jev call's own confidence, a different scale entirely). 0.50 costs
# `caller_name`/`email`/`unit_number`/`amenities_requested` no recall -- every correct
# `margin_winner` commit for these fields scores at or above 0.50 in the real corpus -- while
# filtering out a large share of low-scoring, always-wrong single-candidate commits (e.g.
# `caller_name`'s wrong single-candidate commits cluster at 0.46-0.54). `pet_info` and
# `work_order_issue` are the exception -- their correct single-candidate commits score as low as
# 0.41 -- and are carved out below via `GLINER_ONLY_PER_FIELD_COMMIT_THRESHOLDS` so this default
# doesn't regress them.
GLINER_ONLY_COMMIT_THRESHOLD = 0.50

# Per-field overrides of `GLINER_ONLY_COMMIT_THRESHOLD`.
#   - `price_quoted`: correct and wrong single-candidate commits overlap heavily in the low-score
#     band (both cluster around 0.12-0.35 -- the same label-collision behavior
#     `gliner_pipeline.PER_FIELD_THRESHOLDS` documents), so no floor cleanly separates them, but
#     0.18 clears the densest cluster of wrong commits for a modest recall cost.
#   - `budget_amount`: this field's correct commits (score 0.16-0.20) sit inside the same dense
#     wrong-commit cluster (0.15-0.45) as `price_quoted`, but with far fewer correct examples to
#     preserve -- a floor high enough to meaningfully cut wrong commits would cost roughly half of
#     this field's already-small recall for a negligible precision gain, so 0.15 stays: this
#     field's imprecision looks like candidate-level ambiguity, not a threshold miscalibration a
#     floor change can fix.
#   - `pet_info`, `work_order_issue`: both fields' correct single-candidate commits score as low as
#     0.41/0.46 -- well under `GLINER_ONLY_COMMIT_THRESHOLD` -- so each needs its own floor to avoid
#     near-zero recall.
GLINER_ONLY_PER_FIELD_COMMIT_THRESHOLDS = {
    "price_quoted": 0.18,
    "budget_amount": 0.15,
    "pet_info": 0.40,
    "work_order_issue": 0.40,
}

# Minimum score gap between the top-ranked and runner-up candidate, once a 2+-candidate set has
# settled, to treat the top one as a confident winner rather than an ambiguous ("none of these")
# rejection. 0.15 has no strong signal against it: the capture DB doesn't record a settled
# decision's runner-up score (only the winner's), so the margin actually applied per decision isn't
# directly recoverable from real data for a targeted comparison; what is directly visible --
# per-field final-outcome misses attributable to a locked `margin_too_close` rejection -- is
# consistently small (1-3 calls per field) next to the bigger, clearly-attributable problems the
# other three constants address (single-candidate noise, and price_quoted/budget_amount's
# candidate-level ambiguity).
GLINER_ONLY_MARGIN_THRESHOLD = 0.15

# Separate commit floor for determination-taxonomy fields (`gliner_pipeline.FIELD_TAXONOMY`),
# used in place of `GLINER_ONLY_COMMIT_THRESHOLD`/`GLINER_ONLY_PER_FIELD_COMMIT_THRESHOLDS` --
# a determination field's candidate score comes from `determination_classifier`'s hand-assigned
# confidence tiers, not GLiNER's span-score distribution, so it isn't comparable to either. 0.55
# has no strong signal against it: `permission_to_enter` (the only determination field) has
# correct and wrong single-candidate commits clustering at the same score (0.9) in the real corpus,
# so no floor value separates them -- its dominant miss mode is GLiNER/`determination_classifier`
# never producing a candidate at all (16 of 38 expected calls), a recall gap this floor can't
# address either way.
GLINER_ONLY_DETERMINATION_COMMIT_FLOOR = 0.55

# Consecutive observations a field's distinct-candidate set must persist unchanged before
# `GlinerOnlyResolver` re-decides it, same purpose as `JEV_RECONFIRM_SETTLE_TICKS` for jev's Choice
# path but independently tunable -- defaults to that constant's value.
GLINER_ONLY_SETTLE_TICKS = JEV_RECONFIRM_SETTLE_TICKS

# One-tick-only additive score bonus for whichever candidate a lexical self-correction cue (e.g.
# "actually", "scratch that") points to this tick, applied before ranking a settled candidate set.
# Set above `GLINER_ONLY_MARGIN_THRESHOLD` so it can flip an otherwise-too-close decision.
GLINER_ONLY_SELF_CORRECTION_SCORE_BONUS = 0.20

# This machine's installed RAM, used to size call-level concurrency: reserve
# 4GB headroom for the OS/other sessions and budget ~4GB per concurrent call.
RAM_GB = 16
CALL_CONCURRENCY = (RAM_GB - 4) // 4  # 3

# Serving mode for gliner_pipeline's extraction path: "inline" wraps the local model singleton
# directly in this process (no network hop, but still gets real batching for ticks that land
# concurrently within one process); "http" posts to GLINER_SERVICE_URL instead, for when a
# separate `jlt serve` process does the actual inference. Defaults to "inline" to preserve today's
# single-process behavior.
GLINER_SERVING_MODE: Literal["inline", "http"] = "inline"
GLINER_SERVICE_URL = "http://localhost:8000"

# GlinerBatchEngine tuning, decided from a real A100 batch-tuning sweep at concurrency=200 (the low
# end of this project's real 200-500 concurrent-call target), matching this project's existing
# discipline of deciding window size/threshold from measurement rather than guessing
# (benchmarks/gliner_batch_service/RESULTS.md). Raising max_batch_size dominated the sweep --
# larger batches monotonically cut both queueing rounds and tail latency up to the point where a
# batch covers the full concurrent load, with no benefit from going higher (max_batch_size caps
# naturally at whatever's actually queued). At that point, batch_wait_timeout_ms=0 (grab whatever's
# immediately queued, don't wait for more) beat any positive wait, since there's nothing left to
# usefully wait for once one batch already covers the load.
GLINER_BATCH_MAX_SIZE = 200
GLINER_BATCH_WAIT_TIMEOUT_MS = 0.0

# Bounds how many ticks may be concurrently in flight toward GlinerBatchEngine at once. Set equal
# to GLINER_BATCH_MAX_SIZE rather than independently: below that, this semaphore -- not the
# engine's own batching capacity -- would be the thing capping how large a real batch can ever get,
# which would silently waste headroom the engine is otherwise willing to use.
GLINER_CONCURRENCY = GLINER_BATCH_MAX_SIZE

# Rough cost-per-million-token estimates, USD. These are NOT authoritative —
# GPT-5.1 pricing is from public list pricing and may drift, and the jev
# figure is typesafe.ai's publicly claimed homepage rate, unconfirmed against
# an actual billing dashboard. Replace once real invoices/dashboards exist.
PRICING_PER_MILLION_TOKENS = {
    "gpt-5.1": {
        "input": 1.25,
        "output": 10.00,
    },
    "jev": {
        "input": 0.042,
        "output": 0.0,  # typesafe.ai advertises output tokens as free
    },
}

# Discount applied to prompt-cache-hit input tokens relative to the standard
# input rate above. OpenAI's `usage.input_tokens` always reports the full
# reconstructed context size, cached or not — `input_tokens_details.cached_tokens`
# is the subset of those tokens actually billed at this discount.
PROMPT_CACHE_DISCOUNT = 0.5

# `jlt gpu-run` pod GPU type. RunPod stock fluctuates within minutes (confirmed live during
# development), so this is a single best-effort choice, not pinned to a data center -- letting
# RunPod's scheduler place the pod keeps it working as stock shifts between regions. If this type
# has no stock when you run it, check current availability (hangar.runpod_client / the runpod
# MCP's get-capacity) and update this constant. RTX 6000 Ada Generation (the original pick) had no
# stock across two separate real attempts; A100 SXM was reliably available and its 80GB VRAM is
# far more than this ~184M-param model needs, but that headroom is the tradeoff for actually
# getting a pod at all right now.
RUNPOD_GPU_TYPE_ID = "NVIDIA A100-SXM4-80GB"
