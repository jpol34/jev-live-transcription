"""Mocked tests for the standalone GLiNER serving app -- no real GLiNER model load."""

import pytest
from fastapi.testclient import TestClient

from jev_live_transcription.serving import app as serving_app_module


class FakeModel:
    """Stand-in for the GLiNER checkpoint's `.inference()` surface (same shape
    `GlinerBatchEngine` calls in production -- see tests/test_gliner_pipeline.py)."""

    def inference(self, texts, labels, batch_size=None, multi_label=False, threshold=None):
        return [
            [{"start": 0, "end": 4, "text": text[:4], "label": "caller_name", "score": 0.9}]
            for text in texts
        ]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(serving_app_module.gliner_pipeline, "_get_zero_shot_model", lambda: FakeModel())
    with TestClient(serving_app_module.app) as test_client:
        yield test_client


def test_extract_returns_the_same_entity_shape_predict_entities_uses(client):
    response = client.post("/extract", json={"text": "Jane called"})

    assert response.status_code == 200
    assert response.json() == {
        "entities": [{"start": 0, "end": 4, "text": "Jane", "label": "caller_name", "score": 0.9}]
    }


def test_healthz_reports_ok_once_started(client):
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_extract_rejects_missing_text_field(client):
    response = client.post("/extract", json={})

    assert response.status_code == 422


def test_extract_returns_500_with_detail_on_inference_failure(monkeypatch):
    class FailingModel:
        def inference(self, texts, labels, batch_size=None, multi_label=False, threshold=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        serving_app_module.gliner_pipeline, "_get_zero_shot_model", lambda: FailingModel()
    )
    app = serving_app_module.create_app()

    with TestClient(app) as test_client:
        response = test_client.post("/extract", json={"text": "hello"})

    assert response.status_code == 500
    assert "boom" in response.json()["detail"]


def test_healthz_reports_503_when_worker_is_not_running(client, monkeypatch):
    monkeypatch.setattr(
        type(serving_app_module.app.state.batch_engine),
        "is_running",
        property(lambda self: False),
    )

    response = client.get("/healthz")

    assert response.status_code == 503


def test_create_app_forwards_batch_tuning_overrides(monkeypatch):
    captured = {}

    def fake_make_batch_engine(*, max_batch_size=None, batch_wait_timeout_ms=None):
        captured["max_batch_size"] = max_batch_size
        captured["batch_wait_timeout_ms"] = batch_wait_timeout_ms
        return serving_app_module.gliner_pipeline.GlinerBatchEngine(
            FakeModel(),
            {"a": "a"},
            max_batch_size=max_batch_size or 8,
            batch_wait_timeout_ms=batch_wait_timeout_ms or 0,
            threshold=0.3,
        )

    monkeypatch.setattr(
        serving_app_module.gliner_pipeline, "_make_batch_engine", fake_make_batch_engine
    )
    app = serving_app_module.create_app(max_batch_size=32, batch_wait_timeout_ms=5.0)

    with TestClient(app):
        pass

    assert captured == {"max_batch_size": 32, "batch_wait_timeout_ms": 5.0}
