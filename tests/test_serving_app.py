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
