"""Tests for `benchmarks/gliner_serve/load_test.py`.

The script is standalone (no `jev_live_transcription` import) so it's loaded here the same way
`tests/test_measure_gliner_concurrency.py` loads its script under test, via
`importlib.util.spec_from_file_location`.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestServer

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "gliner_serve" / "load_test.py"
_spec = importlib.util.spec_from_file_location("load_test", _SCRIPT_PATH)
load_test = importlib.util.module_from_spec(_spec)
sys.modules["load_test"] = load_test
_spec.loader.exec_module(load_test)


def test_percentile_empty_list_is_none():
    assert load_test.percentile([], 50) is None


def test_percentile_p50_of_odd_length():
    assert load_test.percentile([10, 30, 20], 50) == 20


def test_percentile_p95_biases_toward_high_end():
    values = list(range(1, 101))  # 1..100
    assert load_test.percentile(values, 95) >= 95


def test_build_request_matches_confirmed_gliner_serve_schema():
    # Confirmed against gliner==0.2.29's gliner/serve/server.py GLiNERDeployment.__call__: the
    # request body only needs "text" and "labels" (every other field is optional server-side).
    payload = load_test.build_request("hello world", {"caller_name": "caller's full name"})
    assert payload == {"text": "hello world", "labels": {"caller_name": "caller's full name"}}


def test_summarize_empty_latencies():
    stats = load_test.summarize([], n_failed=3, duration_s=1.0)
    assert stats == {
        "n": 0,
        "n_failed": 3,
        "p50_ms": None,
        "p95_ms": None,
        "p99_ms": None,
        "mean_ms": None,
        "throughput_req_s": 0.0,
    }


def test_summarize_known_synthetic_distribution():
    # A hand-picked, known latency distribution -- 10, 20, ..., 100ms -- to check the percentile
    # and throughput math against exact expected values, independent of any real HTTP timing.
    latencies = [float(x) for x in range(10, 101, 10)]

    stats = load_test.summarize(latencies, n_failed=0, duration_s=2.0)

    assert stats["n"] == 10
    assert stats["n_failed"] == 0
    assert stats["p50_ms"] == 50.0
    assert stats["p95_ms"] == 100.0
    assert stats["p99_ms"] == 100.0
    assert stats["mean_ms"] == 55.0
    assert stats["throughput_req_s"] == 5.0


def test_load_fixture_reads_real_fixture_file():
    labels, windows = load_test.load_fixture()
    assert len(labels) == 11
    assert len(windows) == 50
    assert all(isinstance(w, str) and w for w in windows)


async def test_run_load_test_against_stub_server_reports_well_formed_stats():
    """Runs the real closed-loop worker/HTTP path against a local aiohttp stub server with a
    known artificial per-request delay, and checks the reported stats are consistent with it."""
    delay_s = 0.02

    async def handler(request: web.Request) -> web.Response:
        await request.json()
        await asyncio.sleep(delay_s)
        return web.json_response({"entities": []})

    app = web.Application()
    app.router.add_post(load_test.ROUTE_PREFIX, handler)
    server = TestServer(app)
    await server.start_server()
    try:
        url = f"http://{server.host}:{server.port}"
        result = await load_test.run_load_test(
            url, concurrency=4, duration_s=0.6, windows=["hello there, this is a test window"], labels={"a": "A"}
        )
    finally:
        await server.close()

    assert result["n"] > 0
    assert result["n_failed"] == 0
    assert result["p50_ms"] <= result["p95_ms"] <= result["p99_ms"]
    # Every request sleeps delay_s server-side, so measured latency must be at least that much.
    assert result["mean_ms"] >= delay_s * 1000
    assert result["throughput_req_s"] > 0


async def test_run_load_test_records_failures_from_error_responses():
    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"error": "boom"}, status=500)

    app = web.Application()
    app.router.add_post(load_test.ROUTE_PREFIX, handler)
    server = TestServer(app)
    await server.start_server()
    try:
        url = f"http://{server.host}:{server.port}"
        result = await load_test.run_load_test(
            url, concurrency=2, duration_s=0.3, windows=["hello"], labels={"a": "A"}
        )
    finally:
        await server.close()

    assert result["n"] > 0
    assert result["n_failed"] == result["n"]
