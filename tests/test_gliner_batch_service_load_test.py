"""Tests for `benchmarks/gliner_batch_service/load_test.py`.

The script is standalone (no `jev_live_transcription` import) so it's loaded here the same way
`tests/test_load_test.py` loads the sibling `gliner_serve` suite's script under test, via
`importlib.util.spec_from_file_location`.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestServer

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "gliner_batch_service"
    / "load_test.py"
)
_spec = importlib.util.spec_from_file_location("gliner_batch_service_load_test", _SCRIPT_PATH)
load_test = importlib.util.module_from_spec(_spec)
sys.modules["gliner_batch_service_load_test"] = load_test
_spec.loader.exec_module(load_test)


def test_percentile_empty_list_is_none():
    assert load_test.percentile([], 50) is None


def test_percentile_p50_of_odd_length():
    assert load_test.percentile([10, 30, 20], 50) == 20


def test_percentile_p95_biases_toward_high_end():
    values = list(range(1, 101))  # 1..100
    assert load_test.percentile(values, 95) >= 95


def test_build_request_has_no_labels_field():
    # Unlike gliner_serve's request shape, this service's label set is fixed server-side -- the
    # request body carries only the text.
    payload = load_test.build_request("hello world")
    assert payload == {"text": "hello world"}


def test_summarize_empty_latencies():
    # n counts every attempt (successes + failures), not just latencies recorded -- an empty
    # `latencies` list with 3 failures still means 3 real attempts were made.
    stats = load_test.summarize([], n_failed=3, duration_s=1.0)
    assert stats == {
        "n": 3,
        "n_failed": 3,
        "p50_ms": None,
        "p95_ms": None,
        "p99_ms": None,
        "mean_ms": None,
        "throughput_req_s": 3.0,
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


def test_load_fixture_reads_the_shared_gliner_serve_fixture_file():
    # No FIXTURE_PATH duplication -- this suite reads gliner_serve's own fixture file directly, so
    # both suites measure the identical corpus.
    windows = load_test.load_fixture()
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
            url, concurrency=4, duration_s=0.6, windows=["hello there, this is a test window"]
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
        return web.json_response({"detail": "boom"}, status=500)

    app = web.Application()
    app.router.add_post(load_test.ROUTE_PREFIX, handler)
    server = TestServer(app)
    await server.start_server()
    try:
        url = f"http://{server.host}:{server.port}"
        result = await load_test.run_load_test(url, concurrency=2, duration_s=0.3, windows=["hello"])
    finally:
        await server.close()

    assert result["n"] > 0
    assert result["n_failed"] == result["n"]


async def test_run_load_test_excludes_failed_request_latency_from_percentiles():
    """A failed request's latency (which can be inflated, e.g. by a slow response before an error
    status) must not pollute the reported percentiles/mean -- only n_failed should reflect it."""
    call_count = {"n": 0}

    async def handler(request: web.Request) -> web.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            await asyncio.sleep(0.2)  # one slow failure
            return web.json_response({"detail": "boom"}, status=500)
        return web.json_response({"entities": []})  # fast successes after it

    app = web.Application()
    app.router.add_post(load_test.ROUTE_PREFIX, handler)
    server = TestServer(app)
    await server.start_server()
    try:
        url = f"http://{server.host}:{server.port}"
        result = await load_test.run_load_test(url, concurrency=1, duration_s=0.5, windows=["hello"])
    finally:
        await server.close()

    assert result["n_failed"] == 1
    assert result["n"] > result["n_failed"]
    # The one slow (200ms) failure must not show up in p99/mean -- every recorded latency should
    # be a fast local success, far below it.
    assert result["p99_ms"] < 100
    assert result["mean_ms"] < 100


async def test_run_load_test_configures_connector_limit_to_match_concurrency(monkeypatch):
    # aiohttp.TCPConnector's default limit (100) would silently bottleneck any --concurrency above
    # that with client-side connection-pool contention instead of measuring the server -- each
    # closed-loop worker holds at most one connection at a time, so the limit must scale with
    # --concurrency rather than stay at the library default.
    import aiohttp

    captured = {}
    real_connector = aiohttp.TCPConnector

    def tracking_connector(*args, **kwargs):
        captured["limit"] = kwargs.get("limit")
        return real_connector(*args, **kwargs)

    monkeypatch.setattr(load_test.aiohttp, "TCPConnector", tracking_connector)

    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"entities": []})

    app = web.Application()
    app.router.add_post(load_test.ROUTE_PREFIX, handler)
    server = TestServer(app)
    await server.start_server()
    try:
        url = f"http://{server.host}:{server.port}"
        await load_test.run_load_test(url, concurrency=150, duration_s=0.2, windows=["hello"])
    finally:
        await server.close()

    assert captured["limit"] == 150
