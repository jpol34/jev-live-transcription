import asyncio
import time

from jev_live_transcription import gliner_pipeline


class FakePIIModel:
    """Stand-in for the streaming PII checkpoint's `.inference()` surface."""

    def __init__(self, responses):
        self.calls = []
        self.cleared = []
        self._responses = list(responses)

    def inference(self, texts, labels, *, session_id, threshold=0.5):
        self.calls.append(
            {"texts": list(texts), "labels": list(labels), "session_id": list(session_id), "threshold": threshold}
        )
        return [self._responses.pop(0)]

    def clear_session(self, session_id):
        self.cleared.append(session_id)


class FakeZeroShotModel:
    """Stand-in for the standard zero-shot checkpoint's `.predict_entities()` surface."""

    def __init__(self, responses):
        self.calls = []
        self._responses = list(responses)

    def predict_entities(self, text, labels, multi_label=False):
        self.calls.append({"text": text, "labels": list(labels), "multi_label": multi_label})
        return self._responses.pop(0)


def _install_fakes(monkeypatch, pii_responses=(), zero_shot_responses=()):
    monkeypatch.setattr(gliner_pipeline, "_pii_sent_length", {})
    pii_model = FakePIIModel(pii_responses)
    zero_shot_model = FakeZeroShotModel(zero_shot_responses)
    monkeypatch.setattr(gliner_pipeline, "_pii_model", pii_model)
    monkeypatch.setattr(gliner_pipeline, "_zero_shot_model", zero_shot_model)
    return pii_model, zero_shot_model


def test_extract_candidates_covers_all_eleven_fields(monkeypatch):
    _install_fakes(monkeypatch, pii_responses=[[]], zero_shot_responses=[[]])

    result = asyncio.run(gliner_pipeline.extract_candidates("some transcript text", "call-1"))

    expected_fields = set(gliner_pipeline.PII_FIELD_LABELS) | set(gliner_pipeline.ZERO_SHOT_FIELD_LABELS)
    assert set(result.keys()) == expected_fields
    assert len(expected_fields) == 11
    assert all(spans == [] for spans in result.values())


def test_extract_candidates_merges_pii_and_zero_shot_candidates(monkeypatch):
    pii_model, zero_shot_model = _install_fakes(
        monkeypatch,
        pii_responses=[[{"start": 0, "end": 4, "text": "Jane", "label": "person", "score": 0.9}]],
        zero_shot_responses=[
            [{"start": 18, "end": 26, "text": "unit 204", "label": "apartment unit number", "score": 0.8}]
        ],
    )

    result = asyncio.run(gliner_pipeline.extract_candidates("Jane called about unit 204", "call-1"))

    assert result["caller_name"] == [{"text": "Jane", "score": 0.9, "start": 0, "end": 4}]
    assert result["unit_number"] == [{"text": "unit 204", "score": 0.8, "start": 18, "end": 26}]
    assert result["email"] == []
    assert result["phone_number"] == []
    assert pii_model.calls[0]["session_id"] == ["call-1"]
    assert zero_shot_model.calls[0]["multi_label"] is True


def test_extract_candidates_timed_times_each_model_independently(monkeypatch):
    # _run_pii_tick/_run_zero_shot_tick are patched directly (below extract_candidates_timed's
    # own asyncio.to_thread calls) rather than the model objects, so each fake sleeps for a known,
    # different duration -- proving the two returned latencies reflect each call's own wall time
    # instead of one shared measurement of the outer gather.
    def slow_pii(transcript_snapshot, call_id):
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


def test_pii_model_receives_only_the_new_transcript_suffix(monkeypatch):
    pii_model, zero_shot_model = _install_fakes(
        monkeypatch, pii_responses=[[], []], zero_shot_responses=[[], []]
    )

    asyncio.run(gliner_pipeline.extract_candidates("Hello there.", "call-2"))
    asyncio.run(gliner_pipeline.extract_candidates("Hello there. More text.", "call-2"))

    assert pii_model.calls[0]["texts"] == ["Hello there."]
    assert pii_model.calls[1]["texts"] == [" More text."]
    # The zero-shot model always re-encodes the full snapshot, unlike the
    # streaming PII model's delta-only input.
    assert zero_shot_model.calls[0]["text"] == "Hello there."
    assert zero_shot_model.calls[1]["text"] == "Hello there. More text."


def test_pii_model_is_not_called_when_transcript_has_not_grown(monkeypatch):
    pii_model, _zero_shot_model = _install_fakes(
        monkeypatch, pii_responses=[[]], zero_shot_responses=[[], []]
    )

    asyncio.run(gliner_pipeline.extract_candidates("Same text.", "call-3"))
    result = asyncio.run(gliner_pipeline.extract_candidates("Same text.", "call-3"))

    assert len(pii_model.calls) == 1
    assert all(result[field] == [] for field in gliner_pipeline.PII_FIELD_LABELS)


def test_different_call_ids_track_pii_state_independently(monkeypatch):
    pii_model, _zero_shot_model = _install_fakes(
        monkeypatch, pii_responses=[[], []], zero_shot_responses=[[], []]
    )

    asyncio.run(gliner_pipeline.extract_candidates("Call A transcript.", "call-a"))
    asyncio.run(gliner_pipeline.extract_candidates("Call B transcript.", "call-b"))

    assert pii_model.calls[0]["texts"] == ["Call A transcript."]
    assert pii_model.calls[0]["session_id"] == ["call-a"]
    assert pii_model.calls[1]["texts"] == ["Call B transcript."]
    assert pii_model.calls[1]["session_id"] == ["call-b"]


def test_entities_with_unrecognized_labels_are_dropped(monkeypatch):
    _install_fakes(
        monkeypatch,
        pii_responses=[[{"start": 0, "end": 3, "text": "xyz", "label": "not-a-real-label", "score": 0.5}]],
        zero_shot_responses=[[]],
    )

    result = asyncio.run(gliner_pipeline.extract_candidates("xyz text", "call-4"))

    assert all(spans == [] for spans in result.values())


def test_reset_call_clears_tracked_length_and_model_session(monkeypatch):
    monkeypatch.setattr(gliner_pipeline, "_pii_sent_length", {"call-5": 42})
    pii_model = FakePIIModel([])
    monkeypatch.setattr(gliner_pipeline, "_pii_model", pii_model)

    gliner_pipeline.reset_call("call-5")

    assert "call-5" not in gliner_pipeline._pii_sent_length
    assert pii_model.cleared == ["call-5"]


def test_reset_call_is_safe_when_pii_model_never_loaded(monkeypatch):
    monkeypatch.setattr(gliner_pipeline, "_pii_sent_length", {"call-6": 10})
    monkeypatch.setattr(gliner_pipeline, "_pii_model", None)

    gliner_pipeline.reset_call("call-6")

    assert "call-6" not in gliner_pipeline._pii_sent_length
