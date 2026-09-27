"""Async closed-loop HTTP load generator for a running gliner[serve] deployment.

Meant to run *on the pod itself* against `http://localhost:8000` (gliner[serve]'s default),
so measured latency is real serving latency, not internet RTT. It is a standalone script with
no dependency on this repo's `jev_live_transcription` package -- only `aiohttp` and the stdlib --
since it runs on a pod that never installs that package.

Spawns `--concurrency` worker tasks that each loop "send one request, record its latency, send
the next" for `--duration-s` wall-clock seconds. This is closed-loop saturation load (each worker
only ever has one request in flight), the right model for finding a config's real capacity, as
opposed to an open-loop fixed-rate model.

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

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_windows.json"
DEFAULT_URL = "http://localhost:8000"

# Confirmed against the actually-installed gliner==0.2.29 package (gliner/serve/server.py's
# GLiNERDeployment.__call__ and gliner/serve/client.py's GLiNERClient), not assumed from docs:
# the Ray Serve deployment is mounted under this route prefix on top of the server's HTTP port,
# and reads its request body as JSON with a "text" and "labels" field.
ROUTE_PREFIX = "/gliner"


def percentile(values: list[float], p: float) -> float | None:
    """Return the `p`th percentile (0-100) of `values` via nearest-rank, or `None` if empty.

    Same nearest-rank approach as `scripts/measure_gliner_concurrency.py`'s `percentile()`.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1))))
    return ordered[rank]


def summarize(latencies: list[float], *, n_failed: int, duration_s: float) -> dict[str, Any]:
    """Build the final result dict from raw per-request latencies (ms) and the run's wall time."""
    n = len(latencies)
    return {
        "n": n,
        "n_failed": n_failed,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "mean_ms": statistics.fmean(latencies) if latencies else None,
        "throughput_req_s": n / duration_s if duration_s > 0 else None,
    }


def load_fixture(path: Path = FIXTURE_PATH) -> tuple[dict[str, str], list[str]]:
    """Load the label dict and sample transcript windows from `fixtures/sample_windows.json`."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["labels"], data["windows"]


def build_request(text: str, labels: dict[str, str]) -> dict[str, Any]:
    """Build a gliner[serve] request body for `text` against `labels`.

    This project's own pipeline calls `predict_entities(..., multi_label=True)`, but the real
    installed gliner[serve] request schema's `multi_label` defaults to `False` and there is no
    field for anything else this benchmark needs -- so `multi_label` is left at the server's
    default here rather than invented to match the project's own pipeline.
    """
    return {"text": text, "labels": labels}


async def _send_one(
    session: aiohttp.ClientSession, url: str, text: str, labels: dict[str, str]
) -> tuple[float, bool]:
    """Send one request and return `(latency_ms, success)`."""
    payload = build_request(text, labels)
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
    labels: dict[str, str],
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
        latency_ms, success = await _send_one(session, url, text, labels)
        latencies.append(latency_ms)
        if not success:
            failures.append(True)


async def run_load_test(
    url: str, concurrency: int, duration_s: float, windows: list[str], labels: dict[str, str]
) -> dict[str, Any]:
    """Run `concurrency` closed-loop workers against `url` for `duration_s` seconds."""
    endpoint = url.rstrip("/") + ROUTE_PREFIX
    latencies: list[float] = []
    failures: list[bool] = []
    async with aiohttp.ClientSession() as session:
        end_time = time.monotonic() + duration_s
        await asyncio.gather(
            *(
                _worker(session, endpoint, windows, labels, end_time, latencies, failures)
                for _ in range(concurrency)
            )
        )
    return summarize(latencies, n_failed=len(failures), duration_s=duration_s)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, required=True, help="Number of worker tasks.")
    parser.add_argument("--duration-s", type=float, required=True, help="Wall-clock run duration.")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Base URL of the gliner[serve] server.")
    args = parser.parse_args()

    labels, windows = load_fixture()
    result = asyncio.run(run_load_test(args.url, args.concurrency, args.duration_s, windows, labels))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
