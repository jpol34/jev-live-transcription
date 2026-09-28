# gliner_batch_service benchmark results

Measured on a single RunPod A100-SXM4-80GB, `urchade/gliner_medium-v2.1`, against this project's
own workload shape: the 11-field zero-shot label set (`gliner_pipeline.ZERO_SHOT_FIELD_LABELS`)
over ~200-char trailing-window transcript text (the same `fixtures/sample_windows.json` corpus
`gliner_serve`'s own benchmark uses). Two passes: a baseline concurrency sweep at the placeholder
tuning (`results.json`), then a batch-tuning sweep at concurrency=200 (`tuning_sweep.json`) and a
final verification of the winning config at both concurrency points (`tuned_results.json`). The
config values this repo actually ships (`config.GLINER_BATCH_MAX_SIZE`/`GLINER_BATCH_WAIT_TIMEOUT_MS`)
are the tuning sweep's winner, not the placeholder.

## Baseline: placeholder tuning (`GLINER_BATCH_MAX_SIZE=16`, `GLINER_BATCH_WAIT_TIMEOUT_MS=20ms`)

`pod_bench.py bench --concurrencies 50,200 --duration-s 25`, unmodified defaults, not yet swept.
Raw data: `results.json`.

| concurrency | n | n_failed | p50 | p95 | p99 | mean | throughput |
|---|---|---|---|---|---|---|---|
| 50 | 6450 | 0 | 176.2ms | 284.4ms | 411.7ms | 194.5ms | 258.0 req/s |
| 200 | 6616 | 0 | 742.8ms | 918.8ms | 1042.6ms | 765.6ms | 264.6 req/s |

Zero failures at either point -- the service stayed correct under load throughout this whole
benchmark pass, this is purely a latency/throughput measurement, not a stability one.

### Methodology note: a load-tester bug, not a server pathology, in the first run

The first attempt at the concurrency=200 point produced a p99 of ~25 **seconds** against a clean
p95 of ~526ms -- an order-of-magnitude cliff inconsistent with the rest of the distribution.
Traced to `load_test.py`'s HTTP client: `aiohttp.ClientSession` defaults to a 100-connection pool
cap, so at `--concurrency 200` roughly half the workers were queueing for a client-side connection
slot before ever reaching the server, not measuring the server at all. Fixed by sizing the
connector's `limit` to match `--concurrency` (each closed-loop worker holds at most one connection
at a time, so `limit=concurrency` is exactly sufficient) and re-run. The baseline table above is
from the corrected run; it has no artifact of the original bug in it.

## Comparison against the diagnosed pathology

Three-way comparison, at the directly-comparable concurrency=50 point (baseline tuning):

- **Raw single-call baseline** (`scripts/measure_gliner_concurrency.py`, concurrency=1, no serving
  layer): 23.2ms mean.
- **`gliner[serve]`'s best measured config** (20ms batch-wait, bfloat16, same concurrency=50 --
  `benchmarks/gliner_serve/RESULTS.md`): p50=3775ms, throughput=4.32 req/s.
- **`GlinerBatchEngine` (this suite), concurrency=50, baseline tuning**: p50=176.2ms,
  throughput=258.0 req/s.

That's a **~21x** improvement in p50 and a **~60x** improvement in throughput over `gliner[serve]`'s
best-performing config (and ~286x/~129x over its worst, the two 10ms-batch-wait configs), *before*
any tuning of this service's own knobs. This confirms the diagnostic pass's conclusion
(`benchmarks/gliner_serve/RESULTS.md`'s Decision 6): the multi-second latencies measured there were
a serving-layer pathology (a blocking `async def` handler plus batch-size churn defeating
`torch.compile`), not an inherent ceiling on GLiNER-on-GPU serving.

## Batch-tuning sweep (concurrency=200, the low end of the real target range)

Baseline tuning missed the real target at concurrency=200 (see below), with queueing behind
`GLINER_BATCH_MAX_SIZE`, not raw compute, the suspected cause (200 concurrent requests need ~13
sequential batches of 16 to all clear once). `pod_bench.py tune --concurrency 200 --duration-s 25
--configs <max_batch_size:batch_wait_timeout_ms,...>` tests that directly. Raw data:
`tuning_sweep.json`.

| max_batch_size | batch_wait_timeout_ms | p50 | p95 | p99 | throughput |
|---|---|---|---|---|---|
| 16 | 20 | 784.4ms | 1007.4ms | 1496.2ms | 246.7 req/s |
| 32 | 20 | 713.7ms | 1021.8ms | 1331.6ms | 266.6 req/s |
| 64 | 20 | 643.5ms | 923.6ms | 1138.2ms | 292.2 req/s |
| 16 | 10 | 793.7ms | 1030.4ms | 1349.3ms | 246.1 req/s |
| 32 | 10 | 698.5ms | 967.1ms | 1244.0ms | 273.8 req/s |
| 64 | 10 | 640.0ms | 924.4ms | 1255.5ms | 295.8 req/s |
| 128 | 10 | 570.0ms | 859.9ms | 1048.8ms | 320.0 req/s |
| 200 | 10 | 676.5ms | 958.9ms | 1350.8ms | 292.8 req/s |
| 128 | 0 | 569.4ms | 789.7ms | 1193.6ms | 324.0 req/s |
| **200** | **0** | **562.1ms** | **770.1ms** | 1355.3ms | **328.0 req/s** |

`max_batch_size` dominates the sweep: raising it from 16 to 64 monotonically cuts p50 by ~18% and
lifts throughput by ~20%, and continuing to 128/200 keeps improving p50/p95 further, with returns
visibly flattening past 128 (128->200 only gains ~7ms at `batch_wait_timeout_ms=0`) -- consistent
with having shifted the bottleneck from queueing rounds to the batch's own GPU forward-pass time.
`batch_wait_timeout_ms` matters far less and its effect flips sign once `max_batch_size` covers the
full concurrent load: at small batch sizes a longer wait helps marginally (more chances to fill the
batch); at `max_batch_size=128` or `200`, `batch_wait_timeout_ms=0` (grab whatever's already queued,
don't wait for more) wins outright, since there's nothing left to usefully wait for once one batch
already covers the load.

**Winner: `max_batch_size=200, batch_wait_timeout_ms=0`** -- best p50 and p95 in the sweep, and
`config.py` now ships these as the real defaults (not the placeholder).

## Tuned config verification (both concurrency points)

`pod_bench.py bench --concurrencies 50,200 --max-batch-size 200 --batch-wait-timeout-ms 0
--duration-s 25`, confirming the winning config doesn't regress the already-good concurrency=50
case. Raw data: `tuned_results.json`.

| concurrency | n | n_failed | p50 | p95 | p99 | throughput |
|---|---|---|---|---|---|---|
| 50 | 7101 | 0 | 164.6ms | 201.1ms | 406.2ms | 284.0 req/s |
| 200 | 8400 | 0 | 560.2ms | 724.7ms | 830.7ms | 336.0 req/s |

No regression -- concurrency=50 *improved* too (p95 284.4ms -> 201.1ms), now clearing both the
250ms sub-budget and the 400ms whole-pipeline ceiling on p50 and p95. concurrency=200's p99 also
tightened substantially (1042.6ms -> 830.7ms) versus the baseline-tuning run.

## Verdict against the real target

Real target: `docs/production-scaling-research.md` decisions 3-5, ~250ms GLiNER sub-budget within a
~400ms whole-pipeline ceiling, at 200-500 concurrent live calls.

- **At concurrency=50, tuned**: p50 (165ms) and p95 (201ms) both clear the 250ms sub-budget --
  fully within target.
- **At concurrency=200 (the *low end* of the real target range), tuned**: p50 (560ms) is still
  ~2.2x over the sub-budget and ~1.4x over the whole-pipeline ceiling -- improved substantially from
  baseline tuning (743ms, ~3x/~1.9x over) but not yet within budget.

**This service does not yet meet the real per-tick latency budget at the low end of real target
concurrency, even after batch-size tuning -- but tuning alone closed roughly a third of the gap**
(p50 784ms -> 562ms at the tuning sweep's own concurrency=200 point, a ~28% cut), and returns were
still visibly flattening rather than hitting a hard wall, suggesting the remaining gap is now
closer to the batch's real GPU forward-pass compute time than to queueing overhead. Whether that
compute-bound floor can be closed further on a single instance (e.g. quantization, compile tuning)
or needs horizontal replica scaling to reach 200-500 concurrent calls within budget is not decided
by this pass -- per direction, horizontal scaling is being tracked as a future goal, not pursued
now.

## Open follow-up

- **Horizontal scaling** (deferred, future goal, not pursued in this pass): whether N replicas
  behind a load balancer closes the remaining gap, and at what cost, is untested here.
- **Single-instance compute-bound tuning**: quantization or `torch.compile` tuning might shrink the
  per-batch forward-pass time itself, the now-likely bottleneck at `max_batch_size=200` -- untested
  in this pass, which only swept batch collection parameters, not model execution.
- **Phase B gate** (per the original plan): whether a streaming-checkpoint architecture swap is
  worth pursuing depends partly on whether the remaining gap gets closed by the above -- tracked
  separately.

Every pod created for this validation (baseline sweep, two tuning-sweep rounds, and the tuned
verification run) was confirmed terminated (`mcp__runpod__list-pods`) after each run.
