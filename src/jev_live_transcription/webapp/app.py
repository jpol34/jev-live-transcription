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
one visitor can't degrade the demo for everyone else. `ActiveCallGuard` separately prevents two
viewers from replaying the *same* call_id at once: `llm_baseline`'s GPT-5.1 chain state is keyed
only by call_id at module scope, so two concurrent sessions for the same call_id would race on
(and, on cleanup, destroy) each other's chain state.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import batch_runner, corpus, pipeline_core, secrets
from ..jev_pipeline import JevFieldResolver
from .caller_type import CallerTypeClassifier

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


class ActiveCallGuard:
    """Tracks which call_ids currently have a live replay session, so a second viewer can't start
    a concurrent replay of the same call_id -- see module docstring for why that would corrupt
    shared `gliner_pipeline` state."""

    def __init__(self) -> None:
        self._active: set[int] = set()
        self._lock = asyncio.Lock()

    async def try_acquire(self, call_id: int) -> bool:
        async with self._lock:
            if call_id in self._active:
                return False
            self._active.add(call_id)
            return True

    async def release(self, call_id: int) -> None:
        async with self._lock:
            self._active.discard(call_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    secrets.load_typesafe_key()
    app.state.calls = corpus.load_all()
    app.state.session_limiter = SessionLimiter(MAX_CONCURRENT_SESSIONS)
    app.state.active_call_guard = ActiveCallGuard()
    # Warms up both GLiNER checkpoints now, same as batch_runner does for the benchmark harness,
    # so the first real viewer's replay doesn't silently stall through the one-time model-load
    # cost before any tick is sent.
    await batch_runner.warm_up_gliner()
    yield


app = FastAPI(lifespan=lifespan)
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/call/{call_id}")
def call_page(call_id: int) -> FileResponse:
    if call_id not in app.state.calls:
        raise HTTPException(status_code=404, detail="call_id not found")
    return FileResponse(_STATIC_DIR / "call.html")


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


async def _run_replay(websocket: WebSocket, call_id: int, calls: dict, queue: asyncio.Queue) -> None:
    """Runs `run_call` to completion (or until cancelled by `_run_replay_session` on a send
    failure), always leaving a `_DONE` sentinel on `queue` so `consume` in `_run_replay_session`
    can't block forever waiting for one more item that will never arrive."""

    # A resolver of its own, rather than sharing run_call's internal one (which would require
    # threading it through run_call's signature) -- what actually matters here is reusing
    # jev_pipeline's retry/timeout machinery, not sharing the connection pool.
    resolver = JevFieldResolver()
    classifier = CallerTypeClassifier(resolver, call_id)
    classify_task: asyncio.Task | None = None

    def on_tick(tick_number: int, total_ticks: int, snapshot: str, committed: dict) -> None:
        # Called synchronously from inside run_call's async tick loop -- queue.put_nowait is the
        # sync-safe way to hand a tick off to the consumer task, which does the actual
        # (necessarily async) websocket.send_json. Classification is likewise fired off as a
        # background task rather than awaited here (on_tick can't await); `classify_task` guards
        # against overlapping attempts if a prior classification call is still in flight when a
        # later tick fires.
        nonlocal classify_task
        if classify_task is None or classify_task.done():
            classify_task = asyncio.create_task(classifier.classify(tick_number, snapshot))
        queue.put_nowait(
            {
                "tick_number": tick_number,
                "total_ticks": total_ticks,
                "transcript": snapshot,
                "caller_type": {"status": classifier.status, "confidence": classifier.confidence},
                "committed": _serialize_committed(committed),
            }
        )

    # tempfile.TemporaryDirectory's own __exit__ can raise on Windows if a WAL/SHM sidecar file's
    # handle briefly outlives store.close() -- using mkdtemp/rmtree directly instead means that
    # cleanup failure can never be mistaken for run_call itself having failed (rmtree's errors are
    # swallowed below, after the replay's own success/failure is already determined).
    tmp_dir = tempfile.mkdtemp(prefix="jlt-webapp-")
    try:
        tmp_db_path = Path(tmp_dir) / f"call_{call_id}.sqlite3"
        try:
            await pipeline_core.run_call(
                call_id, tmp_db_path, pacer_mode="realtime", calls=calls, on_tick=on_tick
            )
        finally:
            # Cancelled rather than awaited: by the time we reach here (whether run_call finished
            # normally or this whole task was itself cancelled on viewer disconnect), no more
            # payloads will be sent, so there's nothing left for a still-in-flight classification
            # call to usefully report -- awaiting it would just delay teardown for no benefit.
            if classify_task is not None and not classify_task.done():
                classify_task.cancel()
            queue.put_nowait(_DONE)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        await resolver.aclose()


async def _run_replay_session(websocket: WebSocket, call_id: int) -> None:
    calls: dict[int, dict] = websocket.app.state.calls
    queue: asyncio.Queue = asyncio.Queue()

    async def consume() -> None:
        while True:
            item = await queue.get()
            if item is _DONE:
                return
            await websocket.send_json(item)

    run_task = asyncio.create_task(_run_replay(websocket, call_id, calls, queue))
    consumer_task = asyncio.create_task(consume())
    try:
        done, _pending = await asyncio.wait(
            {run_task, consumer_task}, return_when=asyncio.FIRST_EXCEPTION
        )
        for task in done:
            exc = task.exception()
            if exc is not None:
                raise exc
        if not consumer_task.done():
            # run_task finished first (the ordinary case): let the consumer drain whatever's left
            # in the queue, including the _DONE sentinel run_task's own finally always enqueues.
            await consumer_task
    finally:
        # A viewer disconnecting mid-replay fails consumer_task's websocket.send_json above, which
        # is re-raised out of the `try` -- run_task must be cancelled here rather than left running
        # to completion, or a closed tab keeps consuming one of MAX_CONCURRENT_SESSIONS slots and
        # triggering paid jev calls for the rest of the call's (multi-minute) simulated duration.
        for task in (run_task, consumer_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(run_task, consumer_task, return_exceptions=True)
    await websocket.send_json({"type": "done"})


@app.websocket("/ws/{call_id}")
async def replay(websocket: WebSocket, call_id: int) -> None:
    calls: dict[int, dict] = websocket.app.state.calls
    if call_id not in calls:
        await websocket.close(code=4404, reason="call_id not found")
        return

    limiter: SessionLimiter = websocket.app.state.session_limiter
    active_calls: ActiveCallGuard = websocket.app.state.active_call_guard
    await websocket.accept()

    if not await limiter.try_acquire():
        await websocket.send_json({"type": "busy", "message": "Busy -- try again shortly."})
        await websocket.close(code=1013, reason="busy")
        return

    try:
        if not await active_calls.try_acquire(call_id):
            await websocket.send_json(
                {"type": "busy", "message": "This call is already being viewed -- try again shortly."}
            )
            await websocket.close(code=1013, reason="call already active")
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
            await active_calls.release(call_id)
    finally:
        await limiter.release()
