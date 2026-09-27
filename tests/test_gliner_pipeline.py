import asyncio
import time

from jev_live_transcription import config, gliner_pipeline


class FakePIIModel:
    """Stand-in for the PII checkpoint's `.predict_entities()` surface."""

    def __init__(self, responses):
        self.calls = []
        self._responses = list(responses)

    def predict_entities(self, text, labels, threshold=0.5):
        self.calls.append({"text": text, "labels": labels, "threshold": threshold})
        return self._responses.pop(0)


class FakeZeroShotModel:
    """Stand-in for the standard zero-shot checkpoint's `.predict_entities()` surface."""

    def __init__(self, responses):
        self.calls = []
        self._responses = list(responses)

    def predict_entities(self, text, labels, multi_label=False, threshold=None):
        self.calls.append({"text": text, "labels": labels, "multi_label": multi_label, "threshold": threshold})
        return self._responses.pop(0)


def _install_fakes(monkeypatch, pii_responses=(), zero_shot_responses=()):
    pii_model = FakePIIModel(pii_responses)
    zero_shot_model = FakeZeroShotModel(zero_shot_responses)
    monkeypatch.setattr(gliner_pipeline, "_pii_model", pii_model)
    monkeypatch.setattr(gliner_pipeline, "_zero_shot_model", zero_shot_model)
    return pii_model, zero_shot_model


def test_extract_candidates_covers_all_eleven_fields(monkeypatch):
    # No caller turn in this snapshot, so the PII model is never called -- it doesn't need a
    # response queued.
    _install_fakes(monkeypatch, zero_shot_responses=[[]])

    result = asyncio.run(gliner_pipeline.extract_candidates("some transcript text", "call-1"))

    expected_fields = set(gliner_pipeline.PII_FIELD_LABELS) | set(gliner_pipeline.ZERO_SHOT_FIELD_LABELS)
    assert set(result.keys()) == expected_fields
    assert len(expected_fields) == 11
    assert all(spans == [] for spans in result.values())


def test_extract_candidates_merges_pii_and_zero_shot_candidates(monkeypatch):
    snapshot = "Agent: hi\nCaller: Jane called about unit 204"
    # Snapshot is far shorter than GLINER_ZERO_SHOT_WINDOW_CHARS, so the zero-shot model's window
    # is the whole snapshot and its reported offsets equal full-snapshot offsets directly.
    unit_start = snapshot.index("unit 204")
    unit_end = unit_start + len("unit 204")
    pii_model, zero_shot_model = _install_fakes(
        monkeypatch,
        pii_responses=[[{"start": 0, "end": 4, "text": "Jane", "label": "person", "score": 0.9}]],
        zero_shot_responses=[
            [{"start": unit_start, "end": unit_end, "text": "unit 204", "label": "unit_number", "score": 0.8}]
        ],
    )

    result = asyncio.run(gliner_pipeline.extract_candidates(snapshot, "call-1"))

    caller_turn_start = snapshot.index("Jane")
    assert result["caller_name"] == [
        {"text": "Jane", "score": 0.9, "start": caller_turn_start, "end": caller_turn_start + 4}
    ]
    assert result["unit_number"] == [{"text": "unit 204", "score": 0.8, "start": unit_start, "end": unit_end}]
    assert result["email"] == []
    assert result["phone_number"] == []
    assert pii_model.calls[0]["text"] == "Jane called about unit 204"
    assert zero_shot_model.calls[0]["multi_label"] is True


def test_extract_candidates_timed_times_each_model_independently(monkeypatch):
    # _run_pii_tick/_run_zero_shot_tick are patched directly (below extract_candidates_timed's
    # own asyncio.to_thread calls) rather than the model objects, so each fake sleeps for a known,
    # different duration -- proving the two returned latencies reflect each call's own wall time
    # instead of one shared measurement of the outer gather.
    def slow_pii(transcript_snapshot):
        time.sleep(0.05)
        return {field: [] for field in gliner_pipeline.PII_FIELD_LABELS}

    def slow_zero_shot(transcript_snapshot):
        time.sleep(0.2)
        return {field: [] for field in gliner_pipeline.ZERO_SHOT_FIELD_LABELS}

    monkeypatch.setattr(gliner_pipeline, "_run_pii_tick", slow_pii)
    monkeypatch.setattr(gliner_pipeline, "_run_zero_shot_tick", slow_zero_shot)

    candidates, pii_latency_ms, zero_shot_latency_ms = asyncio.run(
        gliner_pipeline.extract_candidates_timed("some text", "call-timed")
    )

    expected_fields = set(gliner_pipeline.PII_FIELD_LABELS) | set(gliner_pipeline.ZERO_SHOT_FIELD_LABELS)
    assert set(candidates.keys()) == expected_fields
    assert 40 <= pii_latency_ms < 150
    assert 150 <= zero_shot_latency_ms < 400
    assert pii_latency_ms != zero_shot_latency_ms


def test_current_caller_turn_returns_text_and_offset_mid_turn():
    snapshot = "Agent: Thank you for calling.\nCaller: Hi, it's Tom"

    text, start = gliner_pipeline._current_caller_turn(snapshot)

    assert text == "Hi, it's Tom"
    assert start == snapshot.index("Hi, it's Tom")
    assert snapshot[start:] == text


def test_current_caller_turn_empty_when_agent_is_currently_speaking():
    snapshot = "Caller: Hi, it's Tom.\nAgent: Nice to meet you, Tom"

    text, start = gliner_pipeline._current_caller_turn(snapshot)

    assert text == ""
    assert start == 0


def test_current_caller_turn_empty_when_snapshot_is_empty():
    text, start = gliner_pipeline._current_caller_turn("")

    assert text == ""
    assert start == 0


def test_current_caller_turn_empty_just_as_the_turn_starts():
    # The turn line exists but no words have been released into it yet this tick.
    snapshot = "Agent: hello\nCaller: "

    text, start = gliner_pipeline._current_caller_turn(snapshot)

    assert text == ""


def test_current_caller_turn_ignores_earlier_turns_in_a_multi_turn_snapshot():
    snapshot = "Agent: turn one\nCaller: turn two\nAgent: turn three\nCaller: turn four"

    text, _start = gliner_pipeline._current_caller_turn(snapshot)

    assert text == "turn four"


def test_pii_model_not_called_when_agent_is_currently_speaking(monkeypatch):
    pii_model, zero_shot_model = _install_fakes(monkeypatch, zero_shot_responses=[[]])

    result = asyncio.run(
        gliner_pipeline.extract_candidates("Caller: Hi it's Tom.\nAgent: hello", "call-agent-turn")
    )

    assert pii_model.calls == []
    assert all(result[field] == [] for field in gliner_pipeline.PII_FIELD_LABELS)


def test_pii_model_receives_only_the_current_caller_turn_not_prior_turns(monkeypatch):
    # Regression guard for the bug this scoping fixes: an earlier turn (agent or caller) must
    # never reach the PII model call, since this checkpoint's "person" label loses essentially
    # all confidence once any prior turn is present in its input.
    snapshot = "Agent: Thank you for calling Willow Creek.\nCaller: Hi, it's Tom Barker"
    pii_model, _zero_shot_model = _install_fakes(monkeypatch, pii_responses=[[]], zero_shot_responses=[[]])

    asyncio.run(gliner_pipeline.extract_candidates(snapshot, "call-scoped"))

    assert pii_model.calls[0]["text"] == "Hi, it's Tom Barker"
    assert "Willow Creek" not in pii_model.calls[0]["text"]


def test_pii_model_recomputes_every_tick_as_the_turn_grows(monkeypatch):
    pii_model, _zero_shot_model = _install_fakes(
        monkeypatch, pii_responses=[[], []], zero_shot_responses=[[], []]
    )

    asyncio.run(gliner_pipeline.extract_candidates("Caller: Hi", "call-growing"))
    asyncio.run(gliner_pipeline.extract_candidates("Caller: Hi, it's Tom", "call-growing"))

    assert pii_model.calls[0]["text"] == "Hi"
    assert pii_model.calls[1]["text"] == "Hi, it's Tom"


def test_pii_candidate_offsets_are_translated_to_full_snapshot_coordinates(monkeypatch):
    snapshot = "Agent: hello there\nCaller: it's Tom Barker calling"
    turn_text, turn_start = gliner_pipeline._current_caller_turn(snapshot)
    name_start_in_turn = turn_text.index("Tom Barker")
    name_end_in_turn = name_start_in_turn + len("Tom Barker")
    _pii_model, _zero_shot_model = _install_fakes(
        monkeypatch,
        pii_responses=[
            [
                {
                    "start": name_start_in_turn,
                    "end": name_end_in_turn,
                    "text": "Tom Barker",
                    "label": "person",
                    "score": 0.9,
                }
            ]
        ],
        zero_shot_responses=[[]],
    )

    result = asyncio.run(gliner_pipeline.extract_candidates(snapshot, "call-pii-offset"))

    span = result["caller_name"][0]
    assert span["start"] == turn_start + name_start_in_turn
    assert span["end"] == turn_start + name_end_in_turn
    assert snapshot[span["start"] : span["end"]] == "Tom Barker"


def test_entities_with_unrecognized_labels_are_dropped(monkeypatch):
    _install_fakes(
        monkeypatch,
        pii_responses=[[{"start": 0, "end": 3, "text": "xyz", "label": "not-a-real-label", "score": 0.5}]],
        zero_shot_responses=[[]],
    )

    result = asyncio.run(gliner_pipeline.extract_candidates("Caller: xyz text", "call-4"))

    assert all(spans == [] for spans in result.values())


def test_zero_shot_model_receives_only_the_trailing_window(monkeypatch):
    # A snapshot much longer than the configured window -- if the model still received the whole
    # thing, its input size (and therefore latency) would keep growing with call length, which is
    # exactly the unbounded-latency behavior this window bounds.
    long_snapshot = "x" * (config.GLINER_ZERO_SHOT_WINDOW_CHARS * 5)
    _pii_model, zero_shot_model = _install_fakes(monkeypatch, zero_shot_responses=[[]])

    asyncio.run(gliner_pipeline.extract_candidates(long_snapshot, "call-window"))

    sent_text = zero_shot_model.calls[0]["text"]
    assert len(sent_text) == config.GLINER_ZERO_SHOT_WINDOW_CHARS
    assert sent_text == long_snapshot[-config.GLINER_ZERO_SHOT_WINDOW_CHARS :]


def test_zero_shot_input_length_stays_capped_as_transcript_grows(monkeypatch):
    # What actually keeps latency flat regardless of call length is that the model's input size
    # stays capped no matter how long the transcript gets -- a wall-clock timing assertion here
    # would be too noisy (and too close to unrelated scheduling overhead) to reliably catch a
    # regression, so this checks the input size directly across a sequence of growing snapshots.
    call_id = "call-growing-window"
    _pii_model, zero_shot_model = _install_fakes(monkeypatch, zero_shot_responses=[[]] * 5)
    turn = "The caller mentioned unit 204 needs repairs soon. "
    last_snapshot = ""
    for n in (1, 2, 5, 20, 100):
        last_snapshot = turn * n
        asyncio.run(gliner_pipeline.extract_candidates(last_snapshot, call_id))
    gliner_pipeline.reset_call(call_id)

    sent_lengths = [len(call["text"]) for call in zero_shot_model.calls]
    assert all(length <= config.GLINER_ZERO_SHOT_WINDOW_CHARS for length in sent_lengths)
    # The final, largest snapshot's window is close to the configured cap (word-boundary snapping
    # can shrink it slightly, never grow it) and far smaller than the full snapshot it was cut
    # from -- proof the window actually bit, not just an incidentally-small input.
    assert sent_lengths[-1] > config.GLINER_ZERO_SHOT_WINDOW_CHARS - len(turn)
    assert sent_lengths[-1] < len(last_snapshot)


def test_zero_shot_candidate_offsets_are_translated_to_full_snapshot_coordinates(monkeypatch):
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
    _pii_model, _zero_shot_model = _install_fakes(
        monkeypatch,
        zero_shot_responses=[
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


def test_zero_shot_offsets_unchanged_when_snapshot_shorter_than_window(monkeypatch):
    snapshot = "Jane called about unit 204"
    _pii_model, _zero_shot_model = _install_fakes(
        monkeypatch,
        zero_shot_responses=[
            [{"start": 18, "end": 26, "text": "unit 204", "label": "unit_number", "score": 0.8}]
        ],
    )

    result = asyncio.run(gliner_pipeline.extract_candidates(snapshot, "call-short"))

    assert result["unit_number"] == [{"text": "unit 204", "score": 0.8, "start": 18, "end": 26}]


def test_zero_shot_model_receives_configured_threshold_and_label_descriptions(monkeypatch):
    _pii_model, zero_shot_model = _install_fakes(monkeypatch, zero_shot_responses=[[]])

    asyncio.run(gliner_pipeline.extract_candidates("some transcript text", "call-threshold"))

    call = zero_shot_model.calls[0]
    assert call["threshold"] == config.GLINER_ZERO_SHOT_THRESHOLD
    assert call["labels"] == gliner_pipeline.ZERO_SHOT_FIELD_LABELS


def test_zero_shot_inference_lock_is_held_during_the_model_call(monkeypatch):
    lock_states = []

    class LockCheckingZeroShotModel(FakeZeroShotModel):
        def predict_entities(self, text, labels, multi_label=False, threshold=None):
            lock_states.append(gliner_pipeline._zero_shot_inference_lock.locked())
            return super().predict_entities(text, labels, multi_label=multi_label, threshold=threshold)

    pii_model = FakePIIModel([])
    zero_shot_model = LockCheckingZeroShotModel([[]])
    monkeypatch.setattr(gliner_pipeline, "_pii_model", pii_model)
    monkeypatch.setattr(gliner_pipeline, "_zero_shot_model", zero_shot_model)

    asyncio.run(gliner_pipeline.extract_candidates("some text", "call-lock"))

    assert lock_states == [True]


def test_pii_inference_lock_is_held_during_the_model_call(monkeypatch):
    lock_states = []

    class LockCheckingPIIModel(FakePIIModel):
        def predict_entities(self, text, labels, threshold=0.5):
            lock_states.append(gliner_pipeline._pii_inference_lock.locked())
            return super().predict_entities(text, labels, threshold=threshold)

    pii_model = LockCheckingPIIModel([[]])
    zero_shot_model = FakeZeroShotModel([[]])
    monkeypatch.setattr(gliner_pipeline, "_pii_model", pii_model)
    monkeypatch.setattr(gliner_pipeline, "_zero_shot_model", zero_shot_model)

    asyncio.run(gliner_pipeline.extract_candidates("Caller: some text", "call-pii-lock"))

    assert lock_states == [True]


def test_reset_call_is_a_safe_no_op():
    # Both models are stateless now -- reset_call has nothing to discard, but stays callable so
    # its existing callers (call teardown, warm-up, smoke_gliner.py) don't need special-casing.
    gliner_pipeline.reset_call("call-5")


def test_reset_call_is_safe_when_pii_model_never_loaded(monkeypatch):
    monkeypatch.setattr(gliner_pipeline, "_pii_model", None)

    gliner_pipeline.reset_call("call-6")


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


def test_get_pii_model_loads_with_resolved_device(monkeypatch):
    monkeypatch.setattr(gliner_pipeline, "_pii_model", None)
    monkeypatch.setattr(gliner_pipeline, "_resolve_device", lambda: "cuda")
    captured = {}

    def fake_from_pretrained(name, map_location=None):
        captured["name"] = name
        captured["map_location"] = map_location
        return "fake-pii-model"

    monkeypatch.setattr(gliner_pipeline.GLiNER, "from_pretrained", fake_from_pretrained)

    model = gliner_pipeline._get_pii_model()

    assert model == "fake-pii-model"
    assert captured == {"name": gliner_pipeline.PII_MODEL_NAME, "map_location": "cuda"}


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
