"""Mocked tests for the webapp -- no real GLiNER/jev/Strongbox calls (kept fast/offline)."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from jev_live_transcription.webapp import app as app_module


def _calls_fixture() -> dict[int, dict]:
    return {
        1: {"scenario": {"id": 1, "category": "prospect", "subtype": "availability_inquiry", "edge_case": False, "target_minutes": 3.0}},
        2: {"scenario": {"id": 2, "category": "prospect", "subtype": "availability_inquiry", "edge_case": True, "target_minutes": 2.0}},
        3: {"scenario": {"id": 3, "category": "resident", "subtype": "work_order", "edge_case": False, "target_minutes": 4.0}},
    }


def test_group_calls_groups_by_category_then_subtype():
    grouped = app_module._group_calls(_calls_fixture())

    assert set(grouped) == {"prospect", "resident"}
    assert [c["call_id"] for c in grouped["prospect"]["availability_inquiry"]] == [1, 2]
    assert grouped["prospect"]["availability_inquiry"][1]["edge_case"] is True
    assert [c["call_id"] for c in grouped["resident"]["work_order"]] == [3]


def test_serialize_committed_reshapes_tuple_keys():
    committed = {
        ("gliner_jev", "caller_name"): ("Tim Barker", 0.9),
        ("llm", "caller_name"): ("Tim", 0.7),
    }

    serialized = app_module._serialize_committed(committed)

    assert serialized == {
        "gliner_jev": {"caller_name": {"value": "Tim Barker", "confidence": 0.9}},
        "llm": {"caller_name": {"value": "Tim", "confidence": 0.7}},
    }


def test_serialize_committed_empty_is_empty():
    assert app_module._serialize_committed({}) == {}


async def test_session_limiter_blocks_past_max_and_releases():
    limiter = app_module.SessionLimiter(max_sessions=1)

    assert await limiter.try_acquire() is True
    assert await limiter.try_acquire() is False  # over cap

    await limiter.release()
    assert await limiter.try_acquire() is True  # slot freed


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module.secrets, "load_typesafe_key", lambda: None)
    monkeypatch.setattr(app_module.batch_runner, "_warm_up_gliner", AsyncMock())
    with TestClient(app_module.app) as test_client:
        app_module.app.state.calls = _calls_fixture()
        yield test_client


def test_list_calls_endpoint(client):
    response = client.get("/calls")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prospect", "resident"}


def test_websocket_replay_streams_ticks_and_done(client, monkeypatch):
    async def fake_run_call(call_id, db_path, *, pacer_mode, calls, on_tick):
        on_tick(1, 3, {("gliner_jev", "caller_name"): ("Tim Barker", 0.9)})
        on_tick(2, 3, {("gliner_jev", "caller_name"): ("Tim Barker", 0.9)})

    monkeypatch.setattr(app_module.pipeline_core, "run_call", fake_run_call)

    with client.websocket_connect("/ws/1") as websocket:
        first = websocket.receive_json()
        second = websocket.receive_json()
        done = websocket.receive_json()

    assert first["tick_number"] == 1
    assert first["committed"]["gliner_jev"]["caller_name"]["value"] == "Tim Barker"
    assert second["tick_number"] == 2
    assert done == {"type": "done"}


def test_websocket_replay_rejects_unknown_call_id(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/999") as websocket:
            websocket.receive_json()


def test_websocket_replay_sends_busy_when_over_session_cap(client, monkeypatch):
    async def fake_try_acquire():
        return False

    monkeypatch.setattr(app_module.app.state.session_limiter, "try_acquire", fake_try_acquire)

    with client.websocket_connect("/ws/1") as websocket:
        message = websocket.receive_json()

    assert message["type"] == "busy"
