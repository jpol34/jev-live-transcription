"""FastAPI service exposing `GlinerBatchEngine` over HTTP -- a standalone process doing the actual
GLiNER inference for `config.GLINER_SERVING_MODE="http"` callers, instead of each pipeline process
running its own model. Started via `jlt serve` (see `cli.py`).

The label set is fixed server-side (`gliner_pipeline.ZERO_SHOT_FIELD_LABELS`); callers only send the
raw text to extract from. Offsets in the returned entities are relative to that text, same as
`predict_entities` already returns today -- offset-translation back to a caller's own full-snapshot
coordinates stays a client-side concern (see `gliner_pipeline._postprocess_entities`).
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .. import gliner_pipeline
from ..gliner_batch_engine import GlinerBatchEngine

_LOGGER = logging.getLogger(__name__)


class ExtractRequest(BaseModel):
    text: str


def create_app(
    *, max_batch_size: int | None = None, batch_wait_timeout_ms: float | None = None
) -> FastAPI:
    """Build a serving app, optionally overriding the batch engine's tuning knobs (used by `jlt
    serve`'s CLI flags) instead of only ever reading `config`'s defaults -- keeps CLI overrides
    scoped to the app instance they were requested for, rather than mutating global state that
    other code in the same process (e.g. `gliner_pipeline`'s own inline engine) also reads.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # `_make_batch_engine` loads the GLiNER checkpoint on first call, which can take tens of
        # seconds to minutes on a cold cache -- run it off the event loop so ASGI startup doesn't
        # block (and risk tripping a container platform's startup/health-check timeout).
        engine = await asyncio.to_thread(
            gliner_pipeline._make_batch_engine,
            max_batch_size=max_batch_size,
            batch_wait_timeout_ms=batch_wait_timeout_ms,
        )
        await engine.start()
        app.state.batch_engine = engine
        try:
            yield
        finally:
            await engine.close()

    app = FastAPI(lifespan=lifespan)

    @app.post("/extract")
    async def extract(request: ExtractRequest) -> dict:
        engine: GlinerBatchEngine = app.state.batch_engine
        try:
            entities = await engine.submit(request.text)
        except Exception as exc:
            # A batch-dispatch failure has no per-item isolation (see GlinerBatchEngine._worker),
            # so surface it as a clear 500 with detail rather than letting it fall through to
            # FastAPI's default opaque, bodyless error response.
            _LOGGER.warning("extraction failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"entities": entities}

    @app.get("/healthz")
    async def healthz() -> dict:
        engine: GlinerBatchEngine = app.state.batch_engine
        if not engine.is_running:
            raise HTTPException(status_code=503, detail="batch engine worker is not running")
        return {"status": "ok"}

    return app


app = create_app()
