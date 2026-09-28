# gliner_batch_service benchmark

Benchmark suite for `src/jev_live_transcription/serving/app.py` (`jlt serve`) -- the standalone
HTTP service wrapping `GlinerBatchEngine`, this project's own hand-rolled async dynamic-batching
fix for the pathology `benchmarks/gliner_serve/RESULTS.md` diagnosed in GLiNER's official
`gliner[serve]` serving layer. Unlike that suite, this one benchmarks this repo's own code
(`src/jev_live_transcription/`), not a third-party stack.

Sibling to `benchmarks/gliner_serve/`, following the same shape (`load_test.py`'s closed-loop
worker-pool/percentile/summarize methodology, `pod_bench.py`'s hangar-based pod lifecycle) so both
suites' numbers are directly comparable.

## Fixture

Reuses `benchmarks/gliner_serve/fixtures/sample_windows.json` rather than duplicating it, so both
suites measure the identical input corpus -- only the `"windows"` list is used here (this service's
label set is fixed server-side, unlike `gliner_serve`'s request shape).

## Usage

Local, against a `jlt serve` instance already running on this machine:

```
uv run python -m jev_live_transcription.cli serve &
uv run python benchmarks/gliner_batch_service/load_test.py --concurrency 5 --duration-s 10
```

Real A100 pod (creates, benchmarks, always terminates the pod on exit):

```
uv run python benchmarks/gliner_batch_service/pod_bench.py smoke --ssh-key ~/.ssh/id_ed25519
uv run python benchmarks/gliner_batch_service/pod_bench.py bench --ssh-key ~/.ssh/id_ed25519 \
    --concurrencies 50,200 --duration-s 25
```

`bench` mode writes `results.json`, rewritten after each concurrency point completes so a crash
partway through the sweep doesn't lose already-collected data.
