# gliner[serve] benchmark results

Measured on a single RunPod A100-SXM4-80GB, `urchade/gliner_medium-v2.1`, against this project's
own workload shape: the 11-field zero-shot label set
(`gliner_pipeline.ZERO_SHOT_FIELD_LABELS`) over ~200-char trailing-window transcript text
(`fixtures/sample_windows.json`), at 50 concurrent closed-loop callers (`load_test.py
--concurrency 50 --duration-s 25`) per config. Raw data: `results.json`.

## Results

| config | dtype/quant | batch wait | n | n_failed | p50 | p95 | p99 | mean | throughput |
|---|---|---|---|---|---|---|---|---|---|
| bfloat16_10ms | bfloat16 (default) | 10ms | 50 | 0 | 50490ms | 50547ms | 50553ms | 49949ms | 2.0 req/s |
| float16_10ms | float16 | 10ms | 52 | 0 | 48122ms | 51837ms | 51838ms | 47327ms | 2.08 req/s |
| int8_10ms | int8 | 10ms | -- | -- | -- | -- | -- | -- | server did not become ready within 600s |
| bfloat16_10ms_5ms | bfloat16 | 5ms | 101 | 0 | 3814ms | 28215ms | 28217ms | 14103ms | 4.04 req/s |
| bfloat16_10ms_20ms | bfloat16 | 20ms | 108 | 0 | 3775ms | 27233ms | 27242ms | 12750ms | 4.32 req/s |
| bfloat16_10ms_30ms | bfloat16 | 30ms | 55 | 0 | 26989ms | 27027ms | 27029ms | 24540ms | 2.20 req/s |

`throughput_req_s` is `n` divided by the requested 25s test window, not actual measured wall-clock
duration; for the two 10ms configs, per-request latency (~50s) exceeds the test window, so the
true sustained rate is lower than the reported figure, not higher. The `int8` quantization path
never answered a single request within the 600s startup timeout applied to every config (the same
timeout that covers this model's own bfloat16 torch.compile warmup, which took up to ~7 minutes) --
it is excluded from the batch-window sweep rather than compared unfavorably.

The default dtype (bfloat16) beat float16 on p95 and was carried into the batch-window sweep;
20ms batch-wait was the best-performing point in that sweep (lowest p50/mean, highest throughput,
p95/p99 within noise of 30ms's).

## Decision 6: can a sensibly-tuned deployment meet the latency/throughput bar?

No. Every config's p50 is at minimum 15x over the ~250ms GLiNER sub-budget of the ~400ms
whole-pipeline latency ceiling (best case: 3775ms at 20ms batch-wait), and p95/p99 land in the
tens of seconds. Batch-window tuning across 5/10/20/30ms changes the picture by at most ~2x and
does not close a three-orders-of-magnitude gap.

Per-replica throughput in this sweep tops out at 4.32 req/s (20ms batch-wait; the true sustained
rate is likely lower per the caveat above). Reaching the 200-500 req/s target from
`docs/production-scaling-research.md` (200-500 concurrent live calls) by adding replicas alone
would need roughly 46-116 A100 replicas at that per-replica rate -- not a "meaningful throughput
gain" over the current architecture's ~10-20-concurrent-call ceiling, but a GPU fleet far larger
and more expensive than anything else considered in that research pass.

This rules out shipping `gliner[serve]`'s dtype/quantization/batch-window knobs as-is for a
low-latency deployment at this label count and input shape. It does not rule out GLiNER-based
serving in general -- it rules out this specific serving layer's default request-handling path
(one label set per request, no compilation-friendly batch size stability) at this concurrency for
this model. Whoever builds the real service needs a different approach to get near target: fewer
labels per request, a smaller model, request coalescing tuned around `--precompiled-batch-sizes`,
or a different serving layer entirely -- each a real design change, not a config flag, and out of
scope for this benchmark pass.

## Schema note

The real installed `gliner[serve]` (`gliner==0.2.29`) request schema accepts
`{"text", "labels", "threshold", "relation_threshold", "adapter_id"}` with no `multi_label` field.
This project's own pipeline calls `predict_entities(..., multi_label=True)`
(`gliner_pipeline.py`). A future service reusing this project's label set would get `multi_label`
at gliner[serve]'s own default (`False`) unless a version of gliner[serve] adds that parameter --
worth checking at build time rather than assumed.
