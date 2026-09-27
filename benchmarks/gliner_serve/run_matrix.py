"""Sweeps gliner[serve] server configs and load-tests each one, run entirely on the pod.

For each config, launches `python -m gliner.serve` as a local background subprocess (a plain
subprocess -- no SSH involved, since this script already runs on the pod itself), waits for it to
answer on its HTTP port, drives `load_test.py`'s closed-loop load test against it, kills the
server, records the result, and moves to the next config.

Sweep: (1) default dtype (bfloat16) at 10ms batch-wait, (2) float16 at 10ms, (3) int8 quantization
at 10ms, then the best-looking (lowest p95_ms among configs with no failures) of those three at
(4) 5ms, (5) 20ms, (6) 30ms. Falls back to the plain default (no dtype/quantization flag) as
"best" if all of (1)-(3) failed, rather than crashing.

Usage:
    python run_matrix.py --results-path results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

# Both benchmark scripts live side by side in this directory and run standalone on the pod (no
# jev_live_transcription package installed there) -- make sure `import load_test` resolves
# whether this file is run directly or exec'd from elsewhere (e.g. loaded by a test via
# importlib), since a script's own directory isn't always on sys.path in the latter case.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import load_test  # noqa: E402

MODEL = "urchade/gliner_medium-v2.1"
DEFAULT_URL = "http://localhost:8000"

# gliner.serve enables torch.compile by default, and its first real request pays a one-time
# warmup cost measured directly (via pod_bench.py's own smoke test) to take up to ~7 minutes on
# an A100 with this model -- dynamo hits its recompile limit partway through. 120s was nowhere
# near enough; this is a real property of the default config, not a bug to shrink away.
READY_TIMEOUT_S = 600
READY_POLL_INTERVAL_S = 2
SERVER_STOP_TIMEOUT_S = 30

LOAD_TEST_CONCURRENCY = 50
LOAD_TEST_DURATION_S = 25

STAGE_1_CONFIGS: list[dict[str, Any]] = [
    {"name": "bfloat16_10ms", "batch_wait_timeout_ms": 10, "dtype": None, "quantization": None},
    {"name": "float16_10ms", "batch_wait_timeout_ms": 10, "dtype": "float16", "quantization": None},
    {"name": "int8_10ms", "batch_wait_timeout_ms": 10, "dtype": None, "quantization": "int8"},
]

# Used as "best" when every stage-1 config fails, per the ticket's explicit fallback rule.
FALLBACK_BEST_CONFIG: dict[str, Any] = {"name": "default", "dtype": None, "quantization": None}


def build_server_cmd(config: dict[str, Any]) -> list[str]:
    """Build the `python -m gliner.serve ...` argv for a sweep config."""
    cmd = [
        sys.executable,
        "-m",
        "gliner.serve",
        "--model",
        MODEL,
        "--batch-wait-timeout-ms",
        str(config["batch_wait_timeout_ms"]),
    ]
    if config.get("dtype"):
        cmd += ["--dtype", config["dtype"]]
    if config.get("quantization"):
        cmd += ["--quantization", config["quantization"]]
    return cmd


def launch_server(cmd: list[str], log_path: Path):
    """Start `cmd` in the background, redirecting its output to `log_path`.

    Returns a real `Popen` handle (not a shell-backgrounded process) so the caller can
    `.terminate()`/`.wait()` on it directly -- simpler than the SSH-over-network detachment
    `pod_bench.py` needs, since this script runs locally on the pod.
    """
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file


def stop_server(proc: subprocess.Popen, log_file) -> None:
    """Terminate `proc`, escalating to kill if it doesn't exit promptly, and close its log."""
    proc.terminate()
    try:
        proc.wait(timeout=SERVER_STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log_file.close()


def wait_for_ready(
    endpoint: str,
    labels: dict[str, str],
    probe_text: str,
    *,
    timeout_s: float = READY_TIMEOUT_S,
    poll_interval_s: float = READY_POLL_INTERVAL_S,
) -> bool:
    """Poll `endpoint` with a real request until it answers 200, or `timeout_s` elapses."""
    payload = json.dumps(load_test.build_request(probe_text, labels)).encode("utf-8")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(
                endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(poll_interval_s)
    return False


def run_config(config: dict[str, Any], *, url: str, log_dir: Path) -> dict[str, Any]:
    """Launch `config`'s server, wait for it, load-test it, and kill it. Returns one result dict."""
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_server_cmd(config)
    log_path = log_dir / f"{config['name']}.log"
    proc, log_file = launch_server(cmd, log_path)
    try:
        labels, windows = load_test.load_fixture()
        endpoint = url.rstrip("/") + load_test.ROUTE_PREFIX
        ready = wait_for_ready(endpoint, labels, windows[0])
        if not ready:
            result = {
                "n": 0,
                "n_failed": 0,
                "p50_ms": None,
                "p95_ms": None,
                "p99_ms": None,
                "mean_ms": None,
                "throughput_req_s": None,
                "error": f"server did not become ready within {READY_TIMEOUT_S}s",
            }
        else:
            result = asyncio.run(
                load_test.run_load_test(url, LOAD_TEST_CONCURRENCY, LOAD_TEST_DURATION_S, windows, labels)
            )
    finally:
        stop_server(proc, log_file)
    return {"config": config, "result": result}


def pick_best(stage_1_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the stage-1 config with the lowest p95_ms among configs that had zero failures.

    Falls back to `FALLBACK_BEST_CONFIG` if every stage-1 config had failures -- this runs
    unattended, so a judgment call normally made by eyeballing results must not crash the sweep.
    """
    successful = [
        r for r in stage_1_results if r["result"].get("n_failed") == 0 and r["result"].get("p95_ms") is not None
    ]
    if not successful:
        return FALLBACK_BEST_CONFIG
    return min(successful, key=lambda r: r["result"]["p95_ms"])["config"]


def build_stage_2_configs(best: dict[str, Any]) -> list[dict[str, Any]]:
    """Re-run `best`'s dtype/quantization at the three remaining batch-wait window sizes."""
    return [
        {
            "name": f"{best['name']}_{window_ms}ms",
            "batch_wait_timeout_ms": window_ms,
            "dtype": best["dtype"],
            "quantization": best["quantization"],
        }
        for window_ms in (5, 20, 30)
    ]


def write_results(results_path: Path, results: list[dict[str, Any]]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))


def run_sweep(
    configs: list[dict[str, Any]], *, url: str, log_dir: Path, results: list[dict[str, Any]], results_path: Path
) -> None:
    """Run each config in `configs`, appending to `results` and rewriting `results_path` after each
    one completes so a mid-run failure doesn't lose everything already measured."""
    for config in configs:
        print(f"[{datetime.now()}] starting config {config['name']}")
        result = run_config(config, url=url, log_dir=log_dir)
        results.append(result)
        write_results(results_path, results)
        print(f"[{datetime.now()}] finished config {config['name']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", type=Path, required=True, help="Where to write the results JSON.")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Base URL of the gliner[serve] server.")
    args = parser.parse_args()

    log_dir = args.results_path.resolve().parent / "gliner_serve_logs"
    results: list[dict[str, Any]] = []

    run_sweep(STAGE_1_CONFIGS, url=args.url, log_dir=log_dir, results=results, results_path=args.results_path)

    best = pick_best(results)
    stage_2_configs = build_stage_2_configs(best)
    run_sweep(stage_2_configs, url=args.url, log_dir=log_dir, results=results, results_path=args.results_path)

    print(f"[{datetime.now()}] sweep complete -- wrote {len(results)} results to {args.results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
