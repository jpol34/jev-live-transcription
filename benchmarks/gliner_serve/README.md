# gliner[serve] benchmark

Benchmark and prototyping code for a hypothetical future production GLiNER-serving service, kept
in this repo for convenience rather than as a separate project. It measures GLiNER's official
`gliner[serve]` serving layer against this project's own workload shape (11 zero-shot field labels,
~200-char trailing-window inputs). It is not part of, and never modifies, this repo's own
extraction pipeline (`src/jev_live_transcription/`), which remains a benchmark of GLiNER+jev
candidate extraction against an LLM baseline.

## Fixture

`fixtures/sample_windows.json` holds realistic sample inputs for the benchmark: the project's
11-entry zero-shot label dict (`jev_live_transcription.gliner_pipeline.ZERO_SHOT_FIELD_LABELS`)
alongside ~30-50 trailing transcript windows sampled from the real generated call corpus, sliced
the same way `gliner_pipeline._zero_shot_window` slices the transcript-so-far on every tick. This
fixture is the only thing the benchmark's pod-side code reads -- it never imports
`jev_live_transcription` or needs any secrets.

Regenerate it with:

```
uv run python benchmarks/gliner_serve/gen_fixtures.py
```
