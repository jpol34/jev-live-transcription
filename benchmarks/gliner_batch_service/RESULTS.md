# gliner_batch_service benchmark results

Measured on a single RunPod A100-SXM4-80GB, `urchade/gliner_medium-v2.1`, against this project's
own workload shape: the 11-field zero-shot label set (`gliner_pipeline.ZERO_SHOT_FIELD_LABELS`)
over ~200-char trailing-window transcript text (the same `fixtures/sample_windows.json` corpus
`gliner_serve`'s own benchmark uses), at two closed-loop concurrency points (`pod_bench.py bench
--concurrencies 50,200 --duration-s 25`), against `jlt serve` at its placeholder tuning
(`GLINER_BATCH_MAX_SIZE=16`, `GLINER_BATCH_WAIT_TIMEOUT_MS=20ms` -- unmodified defaults, not
swept). Raw data: `results.json`.

## Results

| concurrency | n | n_failed | p50 | p95 | p99 | mean | throughput |
|---|---|---|---|---|---|---|---|
| 50 | 6450 | 0 | 176.2ms | 284.4ms | 411.7ms | 194.5ms | 258.0 req/s |
| 200 | 6616 | 0 | 742.8ms | 918.8ms | 1042.6ms | 765.6ms | 264.6 req/s |

Zero failures at both points -- the service stayed correct under load, this is purely a latency/
throughput measurement, not a stability one.

### Methodology note: a load-tester bug, not a server pathology, in the first run

The first attempt at the concurrency=200 point produced a p99 of ~25 **seconds** against a clean
p95 of ~526ms -- an order-of-magnitude cliff inconsistent with the rest of the distribution.
Traced to `load_test.py`'s HTTP client: `aiohttp.ClientSession` defaults to a 100-connection pool
cap, so at `--concurrency 200` roughly half the workers were queueing for a client-side connection
slot before ever reaching the server, not measuring the server at all. Fixed by sizing the
connector's `limit` to match `--concurrency` (each closed-loop worker holds at most one connection
at a time, so `limit=concurrency` is exactly sufficient) and re-run. The numbers above are from the
corrected run; the table has no artifact of the original bug in it.

## Comparison against the diagnosed pathology and the real target

Three-way comparison, at the directly-comparable concurrency=50 point:

- **Raw single-call baseline** (`scripts/measure_gliner_concurrency.py`, concurrency=1, no serving
  layer): 23.2ms mean.
- **`gliner[serve]`'s best measured config** (20ms batch-wait, bfloat16, same concurrency=50 --
  `benchmarks/gliner_serve/RESULTS.md`): p50=3775ms, throughput=4.32 req/s.
- **`GlinerBatchEngine` (this suite), concurrency=50**: p50=176.2ms, throughput=258.0 req/s.

That's a **~21x** improvement in p50 and a **~60x** improvement in throughput over `gliner[serve]`'s
best-performing config (and ~286x/~129x over its worst, the two 10ms-batch-wait configs). This
confirms the diagnostic pass's conclusion (`benchmarks/gliner_serve/RESULTS.md`'s Decision 6): the
multi-second latencies measured there were a serving-layer pathology (a blocking `async def`
handler plus batch-size churn defeating `torch.compile`), not an inherent ceiling on GLiNER-on-GPU
serving. Fixing those two root causes directly, as this plan set out to do, recovers serving
latency to within two orders of magnitude of the raw single-call baseline instead of three.

Verdict against the real target (`docs/production-scaling-research.md` decisions 3-5: ~250ms
GLiNER sub-budget within a ~400ms whole-pipeline ceiling, at 200-500 concurrent live calls) is more
mixed:

- **At concurrency=50**: p50 (176ms) clears the 250ms sub-budget; p95 (284ms) exceeds the
  sub-budget but clears the 400ms whole-pipeline ceiling; p99 (412ms) is essentially at that
  ceiling and over the GLiNER-specific sub-budget.
- **At concurrency=200** (the *low end* of the real target range): p50 (743ms) is already ~3x over
  the sub-budget and ~1.9x over the whole-pipeline ceiling; p95/p99 further over both.

**This service does not yet meet the real per-tick latency budget at real target concurrency, at
its current placeholder tuning.** It is not a dead end, though: throughput (258-265 req/s) already
sits within the 200-500 req/s implied by the target call volume even at these untuned settings, and
the bottleneck looks like queueing, not raw compute -- `GLINER_BATCH_MAX_SIZE=16` means 200
concurrent requests need ~13 sequential batches to all clear once, and latency scales with that
round count, not with the model's actual forward-pass cost (which the raw 23.2ms baseline shows is
still tiny). A larger `max_batch_size` (trading batch-assembly latency for fewer sequential rounds)
and/or horizontal replica scaling (dividing concurrent load across more than one `GlinerBatchEngine`
instance) are both plausible ways to close this gap -- neither was tested in this pass, since it
only swept concurrency, not the batch-tuning knobs. `GLINER_BATCH_MAX_SIZE`/
`GLINER_BATCH_WAIT_TIMEOUT_MS` are left at their placeholder values pending that sweep; this run's
job was to validate the mechanism works correctly and measure where it currently stands, not to
retune it blind.

## Open follow-up

- **Batch-tuning sweep**: `pod_bench.py bench` already supports `--max-batch-size`/
  `--batch-wait-timeout-ms` overrides (unused in this pass) -- a follow-up run sweeping those
  against concurrency=200-500 is the natural next step before treating this service as
  production-tuned.
- **Horizontal scaling**: whether N replicas behind a load balancer closes the gap faster/cheaper
  than a larger batch size is an open question this pass didn't test.
- **Phase B gate** (per the original plan): whether a streaming-checkpoint architecture swap is
  worth pursuing depends partly on whether the gap above gets closed by tuning alone -- tracked
  separately.

Every pod created for this validation was confirmed terminated (`mcp__runpod__list-pods`) after
each run.
