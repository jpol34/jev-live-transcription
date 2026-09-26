"""Mocked tests for the webapp -- no real GLiNER/jev/Strongbox calls (kept fast/offline)."""

import asyncio
import time
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


async def test_active_call_guard_blocks_same_call_id_until_released():
    guard = app_module.ActiveCallGuard()

    assert await guard.try_acquire(1) is True
    assert await guard.try_acquire(1) is False  # same call_id, still active
    assert await guard.try_acquire(2) is True  # different call_id unaffected

    await guard.release(1)
    assert await guard.try_acquire(1) is True  # freed


class FakeResolver:
    """Stands in for JevFieldResolver -- constructing a real one requires a real
    TYPESAFE_API_KEY, which these mocked tests never set."""

    aclose = AsyncMock()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module.secrets, "load_typesafe_key", lambda: None)
    monkeypatch.setattr(app_module.batch_runner, "warm_up_gliner", AsyncMock())
    monkeypatch.setattr(app_module, "JevFieldResolver", FakeResolver)
    with TestClient(app_module.app) as test_client:
        app_module.app.state.calls = _calls_fixture()
        yield test_client


def test_list_calls_endpoint(client):
    response = client.get("/calls")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prospect", "resident"}


def test_index_serves_selector_page(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "selector.js" in response.text


def test_call_page_serves_live_view(client):
    response = client.get("/call/1")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "live.js" in response.text


def test_static_assets_are_served(client):
    for path in ("/static/style.css", "/static/selector.js", "/static/live.js"):
        response = client.get(path)
        assert response.status_code == 200, path


def test_websocket_replay_streams_ticks_and_done(client, monkeypatch):
    async def fake_run_call(call_id, db_path, *, pacer_mode, calls, on_tick):
        on_tick(1, 3, "Agent: hi", {("gliner_jev", "caller_name"): ("Tim Barker", 0.9)})
        on_tick(2, 3, "Agent: hi there", {("gliner_jev", "caller_name"): ("Tim Barker", 0.9)})

    monkeypatch.setattr(app_module.pipeline_core, "run_call", fake_run_call)

    with client.websocket_connect("/ws/1") as websocket:
        first = websocket.receive_json()
        second = websocket.receive_json()
        done = websocket.receive_json()

    assert first["tick_number"] == 1
    assert first["transcript"] == "Agent: hi"
    assert first["caller_type"] == {"status": "pending", "confidence": 0.0}
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


def test_websocket_replay_sends_busy_when_call_id_already_active(client, monkeypatch):
    async def slow_run_call(call_id, db_path, *, pacer_mode, calls, on_tick):
        on_tick(1, 5, "Agent: hi", {})
        await asyncio.sleep(0.3)

    monkeypatch.setattr(app_module.pipeline_core, "run_call", slow_run_call)

    with client.websocket_connect("/ws/1") as first_ws:
        first_ws.receive_json()  # confirms the first session is actually running
        with client.websocket_connect("/ws/1") as second_ws:
            message = second_ws.receive_json()
        assert message["type"] == "busy"


def test_websocket_disconnect_cancels_the_underlying_replay(client, monkeypatch):
    ticks_emitted = []
    finished_normally = False

    async def fake_run_call(call_id, db_path, *, pacer_mode, calls, on_tick):
        nonlocal finished_normally
        for tick in range(1, 20):
            on_tick(tick, 20, f"Agent: tick {tick}", {})
            ticks_emitted.append(tick)
            await asyncio.sleep(0.05)
        finished_normally = True

    monkeypatch.setattr(app_module.pipeline_core, "run_call", fake_run_call)

    with client.websocket_connect("/ws/1") as websocket:
        websocket.receive_json()  # first tick only, then the client goes away

    time.sleep(0.5)  # give the server-side task a moment to notice the cancellation

    assert finished_normally is False
    assert len(ticks_emitted) < 19
