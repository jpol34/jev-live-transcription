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

# Minimum confidence the GLiNER model requires to report a candidate span, on
# GLiNER's 0-1 scale (library default is 0.5). Set below that default because nothing downstream
# double-checks a candidate's confidence before jev resolution sees it: a field GLiNER never
# surfaces at all is unrecoverable, while a low-confidence false positive is just one more
# candidate for jev to weigh and reject. 0.30 is the same value Microsoft's own Presidio project
# uses in its official GLiNER-based PII recognizer, which has the same no-verification-step shape
# as this pipeline.
GLINER_ZERO_SHOT_THRESHOLD = 0.30

# This machine's installed RAM, used to size call-level concurrency: reserve
# 4GB headroom for the OS/other sessions and budget ~4GB per concurrent call.
RAM_GB = 16
CALL_CONCURRENCY = (RAM_GB - 4) // 4  # 3

# GLiNER concurrency is tuned independently of call concurrency since the model runs locally
# rather than per-call against a remote API. GlinerBatchEngine batches concurrent ticks into one
# shared model.inference() call instead of serializing them, so raising this value can yield a real
# throughput benefit rather than pure queueing overhead. Placeholder until re-tuned from real
# batched measurement.
GLINER_CONCURRENCY = 1

# Serving mode for gliner_pipeline's extraction path: "inline" wraps the local model singleton
# directly in this process (no network hop, but still gets real batching for ticks that land
# concurrently within one process); "http" posts to GLINER_SERVICE_URL instead, for when a
# separate `jlt serve` process does the actual inference. Defaults to "inline" to preserve today's
# single-process behavior.
GLINER_SERVING_MODE: Literal["inline", "http"] = "inline"
GLINER_SERVICE_URL = "http://localhost:8000"

# GlinerBatchEngine tuning. Placeholder defaults -- tune from real A100 pod measurement, matching
# this project's existing discipline of deciding window size/threshold from measurement rather
# than guessing.
GLINER_BATCH_MAX_SIZE = 16
GLINER_BATCH_WAIT_TIMEOUT_MS = 20.0

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
