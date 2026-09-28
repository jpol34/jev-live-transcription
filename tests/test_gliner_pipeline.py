import asyncio
import threading
import time

from jev_live_transcription import config, gliner_pipeline


class FakeZeroShotModel:
    """Stand-in for the GLiNER checkpoint's `.inference()` surface -- the batched, order-preserving
    method `GlinerBatchEngine` calls (the same method `predict_entities` itself delegates to for a
    single text). `responses` is a flat queue of per-text entity lists, popped one per text in
    `texts`, in order -- a single-text call pops exactly one, a batched call of N texts pops N.
    """

    def __init__(self, responses):
        self.calls = []
        self._responses = list(responses)

    def inference(self, texts, labels, batch_size=None, multi_label=False, threshold=None):
        self.calls.append(
            {"texts": list(texts), "labels": labels, "multi_label": multi_label, "threshold": threshold}
        )
        return [self._responses.pop(0) for _ in texts]


def _install_fake(monkeypatch, responses=()):
    model = FakeZeroShotModel(responses)
    monkeypatch.setattr(gliner_pipeline, "_zero_shot_model", model)
    return model


def test_extract_candidates_covers_all_eleven_fields(monkeypatch):
    _install_fake(monkeypatch, responses=[[]])

    result = asyncio.run(gliner_pipeline.extract_candidates("some transcript text", "call-1"))

    expected_fields = set(gliner_pipeline.ZERO_SHOT_FIELD_LABELS)
    assert set(result.keys()) == expected_fields
    assert len(expected_fields) == 11
    assert all(spans == [] for spans in result.values())


def test_extract_candidates_returns_candidates_for_multiple_fields(monkeypatch):
    snapshot = "Jane called about unit 204"
    model = _install_fake(
        monkeypatch,
        responses=[
            [
                {"start": 0, "end": 4, "text": "Jane", "label": "caller_name", "score": 0.9},
                {"start": 18, "end": 26, "text": "unit 204", "label": "unit_number", "score": 0.8},
            ]
        ],
    )

    result = asyncio.run(gliner_pipeline.extract_candidates(snapshot, "call-1"))

    assert result["caller_name"] == [{"text": "Jane", "score": 0.9, "start": 0, "end": 4}]
    assert result["unit_number"] == [{"text": "unit 204", "score": 0.8, "start": 18, "end": 26}]
    assert result["email"] == []
    assert result["phone_number"] == []
    assert model.calls[0]["multi_label"] is True


def test_extract_candidates_timed_returns_latency(monkeypatch):
    class SlowModel(FakeZeroShotModel):
        def inference(self, texts, labels, batch_size=None, multi_label=False, threshold=None):
            time.sleep(0.05)
            return super().inference(
                texts, labels, batch_size=batch_size, multi_label=multi_label, threshold=threshold
            )

    monkeypatch.setattr(gliner_pipeline, "_zero_shot_model", SlowModel([[]]))

    candidates, latency_ms = asyncio.run(
        gliner_pipeline.extract_candidates_timed("some text", "call-timed")
    )

    assert set(candidates.keys()) == set(gliner_pipeline.ZERO_SHOT_FIELD_LABELS)
    assert 40 <= latency_ms < 500


def test_entities_with_unrecognized_labels_are_dropped(monkeypatch):
    _install_fake(
        monkeypatch,
        responses=[[{"start": 0, "end": 3, "text": "xyz", "label": "not-a-real-label", "score": 0.5}]],
    )

    result = asyncio.run(gliner_pipeline.extract_candidates("xyz text", "call-4"))

    assert all(spans == [] for spans in result.values())


def test_zero_shot_model_receives_only_the_trailing_window(monkeypatch):
    # A snapshot much longer than the configured window -- if the model still received the whole
    # thing, its input size (and therefore latency) would keep growing with call length, which is
    # exactly the unbounded-latency behavior this window bounds.
    long_snapshot = "x" * (config.GLINER_ZERO_SHOT_WINDOW_CHARS * 5)
    model = _install_fake(monkeypatch, responses=[[]])

    asyncio.run(gliner_pipeline.extract_candidates(long_snapshot, "call-window"))

    sent_text = model.calls[0]["texts"][0]
    assert len(sent_text) == config.GLINER_ZERO_SHOT_WINDOW_CHARS
    assert sent_text == long_snapshot[-config.GLINER_ZERO_SHOT_WINDOW_CHARS :]


def test_zero_shot_input_length_stays_capped_as_transcript_grows(monkeypatch):
    # What actually keeps latency flat regardless of call length is that the model's input size
    # stays capped no matter how long the transcript gets -- a wall-clock timing assertion here
    # would be too noisy (and too close to unrelated scheduling overhead) to reliably catch a
    # regression, so this checks the input size directly across a sequence of growing snapshots.
    call_id = "call-growing-window"
    model = _install_fake(monkeypatch, responses=[[]] * 5)
    turn = "The caller mentioned unit 204 needs repairs soon. "
    last_snapshot = ""
    for n in (1, 2, 5, 20, 100):
        last_snapshot = turn * n
        asyncio.run(gliner_pipeline.extract_candidates(last_snapshot, call_id))
    gliner_pipeline.reset_call(call_id)

    sent_lengths = [len(call["texts"][0]) for call in model.calls]
    assert all(length <= config.GLINER_ZERO_SHOT_WINDOW_CHARS for length in sent_lengths)
    # The final, largest snapshot's window is close to the configured cap (word-boundary snapping
    # can shrink it slightly, never grow it) and far smaller than the full snapshot it was cut
    # from -- proof the window actually bit, not just an incidentally-small input.
    assert sent_lengths[-1] > config.GLINER_ZERO_SHOT_WINDOW_CHARS - len(turn)
    assert sent_lengths[-1] < len(last_snapshot)


def test_candidate_offsets_are_translated_to_full_snapshot_coordinates(monkeypatch):
    prefix = "word " * 100  # plenty of word-boundary-aligned filler, well past the window
    candidate_text = "unit 204"
    snapshot = prefix + candidate_text
    # Ground truth for where the model's window actually starts (word-boundary snapping means this
    # isn't simply `len(snapshot) - GLINER_ZERO_SHOT_WINDOW_CHARS`).
    window_text, window_start = gliner_pipeline._zero_shot_window(snapshot)
    assert snapshot[window_start:] == window_text
    # The offsets GLiNER would report are relative to the windowed slice actually fed to the
    # model, not the full snapshot.
    start_in_window = window_text.index(candidate_text)
    end_in_window = start_in_window + len(candidate_text)
    _install_fake(
        monkeypatch,
        responses=[
            [
                {
                    "start": start_in_window,
                    "end": end_in_window,
                    "text": candidate_text,
                    "label": "unit_number",
                    "score": 0.8,
                }
            ]
        ],
    )

    result = asyncio.run(gliner_pipeline.extract_candidates(snapshot, "call-offset"))

    span = result["unit_number"][0]
    assert span["start"] == window_start + start_in_window
    assert span["end"] == window_start + end_in_window
    assert snapshot[span["start"] : span["end"]] == candidate_text


def test_offsets_unchanged_when_snapshot_shorter_than_window(monkeypatch):
    snapshot = "Jane called about unit 204"
    _install_fake(
        monkeypatch,
        responses=[[{"start": 18, "end": 26, "text": "unit 204", "label": "unit_number", "score": 0.8}]],
    )

    result = asyncio.run(gliner_pipeline.extract_candidates(snapshot, "call-short"))

    assert result["unit_number"] == [{"text": "unit 204", "score": 0.8, "start": 18, "end": 26}]


def test_model_receives_configured_threshold_and_label_descriptions(monkeypatch):
    model = _install_fake(monkeypatch, responses=[[]])

    asyncio.run(gliner_pipeline.extract_candidates("some transcript text", "call-threshold"))

    call = model.calls[0]
    assert call["threshold"] == config.GLINER_ZERO_SHOT_THRESHOLD
    assert call["labels"] == gliner_pipeline.ZERO_SHOT_FIELD_LABELS


def test_concurrent_ticks_share_one_batched_inference_call(monkeypatch):
    # The whole point of GlinerBatchEngine: concurrent ticks for different calls should land in one
    # real batched model.inference() call, not be serialized into separate single-text calls.
    model = _install_fake(
        monkeypatch,
        responses=[
            [{"start": 0, "end": 4, "text": "Jane", "label": "caller_name", "score": 0.9}],
            [{"start": 0, "end": 8, "text": "unit 204", "label": "unit_number", "score": 0.8}],
        ],
    )

    async def run_two_concurrently():
        return await asyncio.gather(
            gliner_pipeline.extract_candidates("Jane called", "call-a"),
            gliner_pipeline.extract_candidates("unit 204 leaking", "call-b"),
        )

    result_a, result_b = asyncio.run(run_two_concurrently())

    assert len(model.calls) == 1
    assert len(model.calls[0]["texts"]) == 2
    assert result_a["caller_name"][0]["text"] == "Jane"
    assert result_b["unit_number"][0]["text"] == "unit 204"


def test_worker_never_runs_two_inference_calls_concurrently(monkeypatch):
    # GLiNER's stateless inference has no internal lock of its own, so the property that must hold
    # is that GlinerBatchEngine's single-worker executor never lets two separate batch dispatches
    # run at the same time, even when they land in separate batches rather than being merged into
    # one.
    active_lock = threading.Lock()
    active = {"count": 0, "max_seen": 0}

    class ConcurrencyCheckingModel(FakeZeroShotModel):
        def inference(self, texts, labels, batch_size=None, multi_label=False, threshold=None):
            with active_lock:
                active["count"] += 1
                active["max_seen"] = max(active["max_seen"], active["count"])
            time.sleep(0.02)
            result = super().inference(
                texts, labels, batch_size=batch_size, multi_label=multi_label, threshold=threshold
            )
            with active_lock:
                active["count"] -= 1
            return result

    monkeypatch.setattr(config, "GLINER_BATCH_WAIT_TIMEOUT_MS", 0)
    monkeypatch.setattr(gliner_pipeline, "_zero_shot_model", ConcurrencyCheckingModel([[], []]))

    async def run_sequence():
        first = asyncio.create_task(gliner_pipeline.extract_candidates("first call text", "call-1"))
        # Give the worker time to pick up and dispatch the first request to its executor thread
        # before the second one is even submitted, forcing two separate batches.
        await asyncio.sleep(0.005)
        second = asyncio.create_task(gliner_pipeline.extract_candidates("second call text", "call-2"))
        await asyncio.gather(first, second)

    asyncio.run(run_sequence())

    assert active["max_seen"] == 1


def test_batch_engine_gets_a_fresh_instance_per_event_loop(monkeypatch):
    # GlinerBatchEngine (like the vendor AsyncStreamingEngine it's modeled on) raises if reused
    # across event loops. A bare module-level singleton would break the moment a second loop
    # touched it -- exactly what happens here, since each asyncio.run() call is its own fresh loop.
    _install_fake(monkeypatch, responses=[[], []])
    engine_ids = []

    async def grab_engine_id():
        engine = await gliner_pipeline._get_batch_engine()
        engine_ids.append(id(engine))

    asyncio.run(grab_engine_id())
    asyncio.run(grab_engine_id())

    assert len(engine_ids) == 2
    assert engine_ids[0] != engine_ids[1]


def test_batch_dispatch_failure_rejects_every_request_in_the_batch(monkeypatch):
    # model.inference() has no per-item exception isolation, so a batch-wide failure must fail
    # every request sharing that batch with the same exception, not hang or silently drop some.
    class FailingModel(FakeZeroShotModel):
        def inference(self, texts, labels, batch_size=None, multi_label=False, threshold=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(gliner_pipeline, "_zero_shot_model", FailingModel([]))

    async def run_and_collect_errors():
        results = await asyncio.gather(
            gliner_pipeline.extract_candidates("first", "call-1"),
            gliner_pipeline.extract_candidates("second", "call-2"),
            return_exceptions=True,
        )
        return results

    results = asyncio.run(run_and_collect_errors())

    assert len(results) == 2
    assert all(isinstance(result, RuntimeError) for result in results)


def test_reset_call_is_a_safe_no_op():
    gliner_pipeline.reset_call("call-5")


def test_resolve_device_auto_resolves_to_cuda_when_available(monkeypatch):
    monkeypatch.setattr(config, "GLINER_DEVICE", "auto")
    monkeypatch.setattr(gliner_pipeline.torch.cuda, "is_available", lambda: True)

    assert gliner_pipeline._resolve_device() == "cuda"


def test_resolve_device_auto_resolves_to_cpu_when_unavailable(monkeypatch):
    monkeypatch.setattr(config, "GLINER_DEVICE", "auto")
    monkeypatch.setattr(gliner_pipeline.torch.cuda, "is_available", lambda: False)

    assert gliner_pipeline._resolve_device() == "cpu"


def test_resolve_device_explicit_override_passes_through_unchanged(monkeypatch):
    monkeypatch.setattr(config, "GLINER_DEVICE", "cpu")
    monkeypatch.setattr(gliner_pipeline.torch.cuda, "is_available", lambda: True)

    assert gliner_pipeline._resolve_device() == "cpu"


def test_get_zero_shot_model_loads_with_resolved_device(monkeypatch):
    monkeypatch.setattr(gliner_pipeline, "_zero_shot_model", None)
    monkeypatch.setattr(gliner_pipeline, "_resolve_device", lambda: "cuda")
    captured = {}

    def fake_from_pretrained(name, map_location=None):
        captured["name"] = name
        captured["map_location"] = map_location
        return "fake-zero-shot-model"

    monkeypatch.setattr(gliner_pipeline.GLiNER, "from_pretrained", fake_from_pretrained)

    model = gliner_pipeline._get_zero_shot_model()

    assert model == "fake-zero-shot-model"
    assert captured == {"name": gliner_pipeline.ZERO_SHOT_MODEL_NAME, "map_location": "cuda"}
