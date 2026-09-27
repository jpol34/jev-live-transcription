import asyncio
import time

from jev_live_transcription import config, gliner_pipeline


class FakeZeroShotModel:
    """Stand-in for the GLiNER checkpoint's `.predict_entities()` surface."""

    def __init__(self, responses):
        self.calls = []
        self._responses = list(responses)

    def predict_entities(self, text, labels, multi_label=False, threshold=None):
        self.calls.append({"text": text, "labels": labels, "multi_label": multi_label, "threshold": threshold})
        return self._responses.pop(0)


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
    def slow_tick(transcript_snapshot):
        time.sleep(0.05)
        return {field: [] for field in gliner_pipeline.ZERO_SHOT_FIELD_LABELS}

    monkeypatch.setattr(gliner_pipeline, "_run_zero_shot_tick", slow_tick)

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

    sent_text = model.calls[0]["text"]
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

    sent_lengths = [len(call["text"]) for call in model.calls]
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


def test_inference_lock_is_held_during_the_model_call(monkeypatch):
    lock_states = []

    class LockCheckingModel(FakeZeroShotModel):
        def predict_entities(self, text, labels, multi_label=False, threshold=None):
            lock_states.append(gliner_pipeline._zero_shot_inference_lock.locked())
            return super().predict_entities(text, labels, multi_label=multi_label, threshold=threshold)

    model = LockCheckingModel([[]])
    monkeypatch.setattr(gliner_pipeline, "_zero_shot_model", model)

    asyncio.run(gliner_pipeline.extract_candidates("some text", "call-lock"))

    assert lock_states == [True]


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
