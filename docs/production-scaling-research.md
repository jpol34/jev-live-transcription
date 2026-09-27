# Production-scale GLiNER serving: research and design decisions

This is a research and decision record, not a description of this repo's current behavior --
`jev-live-transcription` is a benchmark, not a production call-handling service, and nothing in
this document is implemented here. It exists to capture the findings and decisions from a design
pass on how a *separate, future* production service would need to serve GLiNER-based extraction
at real concurrency (dozens to hundreds of live calls), building on facts this benchmark project
already established (recall thresholds, window size, label set).

## The problem this started from

Ticket 8's real GPU measurement (`scripts/measure_gliner_concurrency.py`, see
`config.GLINER_CONCURRENCY`'s comment) found that raising `gliner_concurrency` above 1 makes
latency *worse*, not better: mean latency rose from 23.2ms (concurrency=1) to 50.5ms
(concurrency=4), with p95 blowing out to 86ms at 4. The root cause, confirmed by direct code
reading: `gliner_pipeline._zero_shot_inference_lock` fully serializes every model call regardless
of concurrency setting -- raising it only adds more concurrent waiters for that one lock, not real
parallelism. This is not a GPU-device-contention effect and is not GPU-specific.

The practical ceiling of the current architecture (one process, one model, fully serialized by a
Python lock) is roughly 10-20 truly concurrent live calls before queueing delay becomes visible on
a live call -- nowhere near the "dozens to hundreds" a production system would need.

## Research findings

### GLiNER ships an official fix for exactly this (not something to hand-roll)

The `gliner` library itself provides:

- **`model.batch_predict_entities()`** -- a real batched-inference API: multiple separate texts in
  one forward pass instead of N serialized single-text calls. Confirmed via
  [urchade/GLiNER#88](https://github.com/urchade/GLiNER/issues/88), where a maintainer/community
  response specifically recommends switching to it to fix low GPU utilization under concurrent
  single-text calls -- the exact symptom this project measured.
- **`gliner[serve]`** (`pip install gliner[serve]`, `python -m gliner.serve`) -- a first-party
  production serving layer built on Ray Serve
  ([docs](https://urchade.github.io/GLiNER/serving.html)). It automatically coalesces concurrent
  incoming requests into single batched forward passes via Ray Serve's `@serve.batch`, with:
  - `--max-batch-size` (default 32), `--batch-wait-timeout-ms` (default 10ms -- the
    batch-collection window), `--precompiled-batch-sizes` (default `1,2,4,8,16,32`).
  - Multi-replica/multi-GPU horizontal scaling: `--num-replicas`, `--num-gpus-per-replica`.
  - Memory-aware batch sizing (`--target-memory-fraction`, `--memory-overhead-factor`) and
    `--max-ongoing-requests` (default 256 concurrent in-flight per replica).
  - No official throughput/latency benchmark numbers are published for this serving layer itself --
    a real gap; see "run our own benchmark" below.
- Multi-GPU support via `nn.DataParallel` is a known *unresolved* gap for `batch_predict()` used
  directly (same GitHub issue thread) -- `gliner[serve]`'s replica-based scaling sidesteps this by
  running one model per process/GPU rather than splitting one batch across GPUs.

### Real measured throughput numbers (a similarly-sized model, not this exact one)

**GLiNER Guard** (arXiv:2605.05277, 147M params -- close to this project's 184M
`gliner_medium-v2.1`), benchmarked on one NVIDIA A100 80GB:

| config | throughput | p50 | p95 | p99 |
|---|---|---|---|---|
| batch=1 (unbatched) | ~54 req/s | -- | -- | -- |
| PyTorch FP16 + dynamic batching (LitServe, batch 64, 50ms window) | 148.2 req/s | 570ms | 1500ms | 1700ms |
| ONNX CUDA FP16 + same batching | 170.6 req/s | 540ms | 870ms | 1000ms |
| ONNX TensorRT FP16 + same batching | 193.6 req/s | 480ms | 750ms | 900ms |

**This is the central open tension for this project's own latency target (see Decisions below):**
every one of these batched configurations' own p50 (480-570ms) already exceeds a 400ms whole-pipeline
ceiling, before this project's own downstream jev-resolution step (~150ms observed) or any network
hop to a separate serving process are even added. These numbers come from a *different* model
config, batch-window tuning, and input shape than this project's own workload (~200-char inputs,
11 labels) -- they establish that real throughput gains are achievable, not that this exact
latency/throughput tradeoff point is the right one for this project. A fresh benchmark tuned for a
*low-latency* profile (small batch, short window) rather than the throughput-maximizing defaults
above is a real open question, not yet answered by any source found.

### Production analogs

Real-time call-center PII redaction in practice favors a **hybrid** pipeline, not NER-only:
deterministic regex for structured PII (phone/email/card numbers) on every request, transformer
NER reserved for harder free-text fields (names, addresses), with heavier LLM-based redaction
gated to only flagged/high-risk samples (moderate-confidence sources: practitioner blogs, not peer
reviewed). One named real product doing live in-call PII redaction is **Trustera**, but its
internal serving architecture is not publicly documented (existence proof only).

CPU-fleet vs. GPU-fleet cost-per-request-second for this specific workload shape (many small,
short-latency-budget requests) has no direct authoritative source -- general small-transformer
literature suggests CPU is competitive at low-to-moderate throughput and GPUs pull ahead once
batched throughput is high, consistent with the GLiNER Guard numbers above, but this is a reasoned
synthesis, not a single controlled study.

## Design decisions (settled)

Reached via `/grill` on 2026-09-27. All four recommendations below were accepted as stated.

1. **Scope**: design-only for now. Not an active build in this repo or elsewhere yet.
2. **Serving approach**: `gliner[serve]` (Ray Serve) over hand-rolled in-process batching --
   purpose-built, maintained by GLiNER's own team, already solves dynamic batching, memory-aware
   batch sizing, and multi-GPU replica scaling.
3. **Target scale**: ~200-500 concurrent live calls.
4. **Latency ceiling**: ~400ms applies to the *whole* per-tick pipeline (GLiNER extraction + jev
   resolution + everything else), not just the GLiNER-call portion -- that's what a live user
   experiences, and jev's own ~150ms already consumes a large share of that budget before
   extraction latency is even added.
5. **Deployment topology**: a separate, persistent service reached over the network -- not
   co-located with a single caller. A shared batching server only pays off with many independent
   client processes routing through one pool of GPU capacity.
6. **Benchmark before committing to a replica count**: run a real `gliner[serve]` deployment
   against this project's own workload shape (window size, label set, input length) and tune
   batch-window/replica-count from that data, rather than sizing capacity off the GLiNER Guard
   numbers above (different model/config/input shape). Matches this project's own established
   practice (window size, concurrency, and threshold were all decided from real measurement, not
   borrowed numbers).
7. **Repo boundary**: this is design for a *separate future production service*, not an evolution
   of `jev-live-transcription`'s own architecture. This repo's job stays "benchmark GLiNER+jev vs.
   an LLM"; a future service would *reuse* this project's findings (recall threshold, window size)
   but live in its own codebase with its own deployment lifecycle.

## Open items (not yet researched / not yet decided)

- **Model-level throughput options** (in progress as of this writing): real low-latency-profile
  ONNX/TensorRT quantization numbers (the table above is throughput-optimized, not
  latency-optimized), smaller/distilled GLiNER checkpoints and their recall/speed tradeoff, and
  whether `gliner[serve]` has been tuned anywhere for a tight-latency-SLA profile (short
  batch-collection window) rather than the throughput-maximizing defaults.
- **Backpressure and cost modeling**: not yet researched. Open questions: what happens when
  instantaneous demand exceeds capacity (queue with a drop policy, autoscaling, graceful
  degradation), and the realistic infrastructure cost shape (RunPod A100 SXM is $1.59/hr secure
  cloud as of this research) for each scaling approach at the 200-500-concurrent-call target.
- **The core latency/throughput tension above is unresolved**: whether a `gliner[serve]`
  configuration exists that hits both the ~400ms whole-pipeline ceiling and meaningful throughput
  gain over the ~10-20-concurrent-call ceiling of the current unbatched architecture is an open
  empirical question, not yet answered by any source found. This is exactly what decision 6's
  own-workload benchmark would need to resolve.
