"""Async closed-loop HTTP load generator for a running `jlt serve` deployment
(`src/jev_live_transcription/serving/app.py`, GlinerBatchEngine over HTTP).

Meant to run *on the pod itself* against `http://localhost:8000` (matches `jlt serve`'s own
default), so measured latency is real serving latency, not internet RTT. Same closed-loop
worker-pool/`percentile`/`summarize` shape as `benchmarks/gliner_serve/load_test.py`, adapted for
this service's request shape (no `labels` field -- fixed server-side) so both suites are directly
comparable. Standalone script with no dependency on this repo's `jev_live_transcription` package --
only `aiohttp` and the stdlib -- consistent with the `gliner_serve` suite, even though `aiohttp` is
already a project dependency (this keeps both scripts usable if ever copied somewhere without the
full package installed).

Spawns `--concurrency` worker tasks that each loop "send one request, record its latency, send
the next" for `--duration-s` wall-clock seconds -- closed-loop saturation load, the right model for
finding a config's real capacity, as opposed to an open-loop fixed-rate model.

Usage:
    python load_test.py --concurrency 50 --duration-s 25 --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

import aiohttp

# Reuses the existing gliner_serve suite's fixture corpus rather than duplicating it, so both
# suites measure the identical input windows -- an apples-to-apples comparison.
FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "gliner_serve" / "fixtures" / "sample_windows.json"
)
DEFAULT_URL = "http://localhost:8000"
# Bounds each individual request -- see run_load_test's docstring for why this must be well below
# aiohttp's own 300s default.
DEFAULT_REQUEST_TIMEOUT_S = 30.0

# `src/jev_live_transcription/serving/app.py`'s one real route -- label set is fixed server-side,
# unlike gliner[serve]'s `/gliner`, so requests here carry no `labels` field at all.
ROUTE_PREFIX = "/extract"


def percentile(values: list[float], p: float) -> float | None:
    """Return the `p`th percentile (0-100) of `values` via nearest-rank, or `None` if empty.

    Same nearest-rank approach as `benchmarks/gliner_serve/load_test.py`'s `percentile()`.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1))))
    return ordered[rank]


def summarize(latencies: list[float], *, n_failed: int, duration_s: float) -> dict[str, Any]:
    """Build the final result dict from successful-request latencies (ms), a separate failed-request
    count, and the run's wall time.

    `latencies` holds only *successful* requests' timings -- a failed request's latency (which can
    be inflated by the client's own timeout, see `_send_one`) says nothing about real serving
    speed, so mixing it into `p50_ms`/`p95_ms`/`mean_ms` would silently skew them. `n` (used for
    `throughput_req_s`) still counts every attempt, successful or not, since a real caller's
    request rate includes the ones that failed.
    """
    n = len(latencies) + n_failed
    return {
        "n": n,
        "n_failed": n_failed,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "mean_ms": statistics.fmean(latencies) if latencies else None,
        "throughput_req_s": n / duration_s if duration_s > 0 else None,
    }


def load_fixture(path: Path = FIXTURE_PATH) -> list[str]:
    """Load the sample transcript windows from `fixtures/sample_windows.json`.

    Only the `"windows"` list is used -- this service's label set is fixed server-side, unlike
    `gliner_serve`'s, so the fixture's `"labels"` dict (kept for `gliner_serve`'s own use) is
    ignored here.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["windows"]


def build_request(text: str) -> dict[str, Any]:
    """Build a `jlt serve` `/extract` request body for `text`."""
    return {"text": text}


async def _send_one(session: aiohttp.ClientSession, url: str, text: str) -> tuple[float, bool]:
    """Send one request and return `(latency_ms, success)`."""
    payload = build_request(text)
    start = time.monotonic()
    try:
        async with session.post(url, json=payload) as resp:
            await resp.read()
            success = resp.status == 200
    except (aiohttp.ClientError, asyncio.TimeoutError):
        success = False
    latency_ms = (time.monotonic() - start) * 1000
    return latency_ms, success


async def _worker(
    session: aiohttp.ClientSession,
    url: str,
    windows: list[str],
    end_time: float,
    latencies: list[float],
    failures: list[bool],
) -> None:
    """Loop "send one request, record latency, send the next" until `end_time` (monotonic)."""
    i = 0
    n_windows = len(windows)
    while time.monotonic() < end_time:
        text = windows[i % n_windows]
        i += 1
        latency_ms, success = await _send_one(session, url, text)
        if success:
            latencies.append(latency_ms)
        else:
            failures.append(True)


async def run_load_test(
    url: str,
    concurrency: int,
    duration_s: float,
    windows: list[str],
    *,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
) -> dict[str, Any]:
    """Run `concurrency` closed-loop workers against `url` for `duration_s` seconds.

    `request_timeout_s` bounds each individual request -- aiohttp's own default (300s) would let
    one stuck request under overload make the whole run (and, for `pod_bench.py`'s SSH-wrapped
    invocation, the wrapping SSH call) run far longer than `duration_s`.
    """
    endpoint = url.rstrip("/") + ROUTE_PREFIX
    latencies: list[float] = []
    failures: list[bool] = []
    timeout = aiohttp.ClientTimeout(total=request_timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        end_time = time.monotonic() + duration_s
        await asyncio.gather(
            *(
                _worker(session, endpoint, windows, end_time, latencies, failures)
                for _ in range(concurrency)
            )
        )
    return summarize(latencies, n_failed=len(failures), duration_s=duration_s)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, required=True, help="Number of worker tasks.")
    parser.add_argument("--duration-s", type=float, required=True, help="Wall-clock run duration.")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Base URL of the jlt serve server.")
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_S,
        help=f"Per-request timeout, seconds (default: {DEFAULT_REQUEST_TIMEOUT_S}).",
    )
    parser.add_argument(
        "--fixture-path",
        type=Path,
        default=FIXTURE_PATH,
        help=(
            "Path to sample_windows.json. Defaults to the gliner_serve suite's fixture (sibling "
            "directory) for local runs -- override when this script runs standalone on a pod "
            "without that sibling directory present (see pod_bench.py)."
        ),
    )
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error(f"--concurrency must be positive, got {args.concurrency}")
    if args.duration_s <= 0:
        parser.error(f"--duration-s must be positive, got {args.duration_s}")

    windows = load_fixture(args.fixture_path)
    result = asyncio.run(
        run_load_test(
            args.url,
            args.concurrency,
            args.duration_s,
            windows,
            request_timeout_s=args.request_timeout_s,
        )
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
