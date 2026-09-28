"""FastAPI service exposing `GlinerBatchEngine` over HTTP -- a standalone process doing the actual
GLiNER inference for `config.GLINER_SERVING_MODE="http"` callers, instead of each pipeline process
running its own model. Started via `jlt serve` (see `cli.py`).

The label set is fixed server-side (`gliner_pipeline.ZERO_SHOT_FIELD_LABELS`); callers only send the
raw text to extract from. Offsets in the returned entities are relative to that text, same as
`predict_entities` already returns today -- offset-translation back to a caller's own full-snapshot
coordinates stays a client-side concern (see `gliner_pipeline._postprocess_entities`).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from .. import config, gliner_pipeline
from ..gliner_batch_engine import GlinerBatchEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = gliner_pipeline._get_zero_shot_model()
    engine = GlinerBatchEngine(
        model,
        gliner_pipeline.ZERO_SHOT_FIELD_LABELS,
        max_batch_size=config.GLINER_BATCH_MAX_SIZE,
        batch_wait_timeout_ms=config.GLINER_BATCH_WAIT_TIMEOUT_MS,
        threshold=config.GLINER_ZERO_SHOT_THRESHOLD,
        multi_label=True,
    )
    await engine.start()
    app.state.batch_engine = engine
    try:
        yield
    finally:
        await engine.close()


app = FastAPI(lifespan=lifespan)


class ExtractRequest(BaseModel):
    text: str


@app.post("/extract")
async def extract(request: ExtractRequest) -> dict:
    engine: GlinerBatchEngine = app.state.batch_engine
    entities = await engine.submit(request.text)
    return {"entities": entities}


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}
