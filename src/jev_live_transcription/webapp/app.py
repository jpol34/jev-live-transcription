"""FastAPI stakeholder-demo app: replays a corpus call tick-by-tick over a WebSocket, rendered by
the frontend as a leasing-office agent's live screen pop.

Purely a visualization layer over the existing GLiNER+jev pipeline -- `pipeline_core.run_call` is
never modified or reimplemented here. `run_call` requires *a* capture DB to write ticks/pipeline
runs to; since this app has no use for that data once a session ends, each replay session gets its
own throwaway SQLite file in a temp directory, deleted as soon as the session finishes -- a bare
`":memory:"` path doesn't work here, since `CaptureStore` opens a second connection from its
writer thread and each `sqlite3.connect(":memory:")` call gets its own unshared, empty database.

Fully public and unauthenticated by design (viewer-triggered jev calls cost trivial, already-
established amounts) -- `MAX_CONCURRENT_SESSIONS` bounds concurrent replay load/cost instead, so
one visitor can't degrade the demo for everyone else.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .. import batch_runner, corpus, pipeline_core, secrets

_LOGGER = logging.getLogger(__name__)

# Caps concurrent live replay sessions so a fully public, unauthenticated page can't be degraded
# for every viewer by an unbounded number of concurrent jev-calling sessions.
MAX_CONCURRENT_SESSIONS = 10

_STATIC_DIR = Path(__file__).resolve().parent / "static"

# Sentinel telling a session's queue-consumer task to stop draining -- distinct from any real tick
# payload (which is always a dict), so it can never collide with legitimate data.
_DONE = object()


class SessionLimiter:
    """Bounds how many replay sessions may be in flight at once, with a non-blocking
    `try_acquire` -- unlike `asyncio.Semaphore.acquire()`, a caller over the cap must be told
    "busy" immediately rather than queued to wait for a slot."""

    def __init__(self, max_sessions: int) -> None:
        self._max_sessions = max_sessions
        self._count = 0
        self._lock = asyncio.Lock()

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._count >= self._max_sessions:
                return False
            self._count += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            self._count -= 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    secrets.load_typesafe_key()
    app.state.calls = corpus.load_all()
    app.state.session_limiter = SessionLimiter(MAX_CONCURRENT_SESSIONS)
    # Warms up both GLiNER checkpoints now, same as batch_runner does for the benchmark harness,
    # so the first real viewer's replay doesn't silently stall through the one-time model-load
    # cost before any tick is sent.
    await batch_runner._warm_up_gliner()
    yield


app = FastAPI(lifespan=lifespan)
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


def _group_calls(calls: dict[int, dict]) -> dict[str, dict[str, list[dict]]]:
    """Group every call's scenario summary by category, then subtype, for the selector view."""
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for call_id, call_data in sorted(calls.items()):
        scenario = call_data["scenario"]
        grouped[scenario["category"]][scenario["subtype"]].append(
            {
                "call_id": call_id,
                "edge_case": bool(scenario["edge_case"]),
                "target_minutes": scenario["target_minutes"],
            }
        )
    return {category: dict(subtypes) for category, subtypes in grouped.items()}


@app.get("/calls")
def list_calls() -> dict:
    return _group_calls(app.state.calls)


def _serialize_committed(committed: dict[tuple[str, str], tuple[str, float]]) -> dict:
    """Reshape `run_call`'s `(pipeline, field_name) -> (value, confidence)` snapshot into
    JSON-safe nesting -- a tuple can't be a JSON object key."""
    by_pipeline: dict[str, dict[str, dict]] = defaultdict(dict)
    for (pipeline, field_name), (value, confidence) in committed.items():
        by_pipeline[pipeline][field_name] = {"value": value, "confidence": confidence}
    return dict(by_pipeline)


async def _run_replay_session(websocket: WebSocket, call_id: int) -> None:
    calls: dict[int, dict] = websocket.app.state.calls
    queue: asyncio.Queue = asyncio.Queue()

    def on_tick(tick_number: int, total_ticks: int, committed: dict) -> None:
        # Called synchronously from inside run_call's async tick loop -- queue.put_nowait is the
        # sync-safe way to hand a tick off to the consumer task below, which does the actual
        # (necessarily async) websocket.send_json.
        queue.put_nowait(
            {
                "tick_number": tick_number,
                "total_ticks": total_ticks,
                "committed": _serialize_committed(committed),
            }
        )

    async def consume() -> None:
        while True:
            item = await queue.get()
            if item is _DONE:
                return
            await websocket.send_json(item)

    consumer_task = asyncio.create_task(consume())
    with tempfile.TemporaryDirectory(prefix="jlt-webapp-") as tmp_dir:
        tmp_db_path = Path(tmp_dir) / f"call_{call_id}.sqlite3"
        try:
            await pipeline_core.run_call(
                call_id, tmp_db_path, pacer_mode="realtime", calls=calls, on_tick=on_tick
            )
        finally:
            queue.put_nowait(_DONE)
            await consumer_task
    await websocket.send_json({"type": "done"})


@app.websocket("/ws/{call_id}")
async def replay(websocket: WebSocket, call_id: int) -> None:
    calls: dict[int, dict] = websocket.app.state.calls
    if call_id not in calls:
        await websocket.close(code=4404, reason="call_id not found")
        return

    limiter: SessionLimiter = websocket.app.state.session_limiter
    await websocket.accept()

    if not await limiter.try_acquire():
        await websocket.send_json({"type": "busy", "message": "Busy -- try again shortly."})
        await websocket.close(code=1013, reason="busy")
        return

    try:
        await _run_replay_session(websocket, call_id)
    except WebSocketDisconnect:
        _LOGGER.info("viewer disconnected mid-replay for call_id=%r", call_id)
    except Exception:
        _LOGGER.exception("replay session failed for call_id=%r", call_id)
        try:
            await websocket.close(code=1011, reason="internal error")
        except Exception:
            pass
    finally:
        await limiter.release()
