"""Diagnoses gliner[serve]'s latency pathology (p50 3.8s-50s in the historical sweep in
results.json on the ticket-52-gliner-serve-results branch) against two mechanisms confirmed by
reading the installed gliner==0.2.29 package directly (gliner/serve/server.py):

1. `GLiNERDeployment._infer_batch` is declared `async def` but calls the synchronous,
   `@torch.inference_mode()`-wrapped `_run_batch_internal` with no `await`/executor -- a classic
   blocking-handler-stalls-the-event-loop pattern.
2. `_infer_batch.set_max_batch_size(...)` is re-called on every dispatch based on observed
   sequence length, which could churn the batch size torch.compile precompiled against.

Runs in two phases so the decision gate below always has a matched latency/utilization pair for
whichever config actually wins, rather than a fixed step list that could leave the winner (e.g.
the historical value) without any utilization number:

- Phase A (p50 only): default (bfloat16, compile-on) at concurrency=1; `--no-compile` at
  concurrency=1 and concurrency=50; and the historical `bfloat16_10ms` concurrency=50 p50 already
  recorded in results.json (compile-on, never GPU-sampled).
- Phase B (GPU-utilization pairing): whichever Phase-A cell has the lowest p50 gets one
  concurrency=50, `nvidia-smi`-sampled run. The `--no-compile` concurrency=50 run from Phase A is
  reused directly if it's the winner; otherwise one additional concurrency=50 pass runs under the
  winning config (freshly, even for the historical config, since that run was never sampled).

`int8` quantization is retried once with an extended warmup timeout, since the historical sweep's
attempt never became ready within its 600s timeout.

Applies the ticket's decision gate to the real numbers: proceeding without vLLM validation
requires the Phase-A winner to reach BOTH p50 <= 500ms AND mean GPU utilization >= 50% during its
Phase-B paired run.

Meant to run entirely on the pod (invoked via `pod_bench.py`'s `diagnose` mode), mirroring
`run_matrix.py`'s shape: a local `python -m gliner.serve` subprocess per config, the same
`load_test` import trick, and the same crash-safe rewrite-after-each-step discipline.

Usage:
    python diagnose.py --results-path diagnosis.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
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
# whether this file is run directly or exec'd from elsewhere, since a script's own directory
# isn't always on sys.path in that case.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import load_test  # noqa: E402

MODEL = "urchade/gliner_medium-v2.1"
DEFAULT_URL = "http://localhost:8000"

# Same rationale as run_matrix.py: gliner.serve's default torch.compile warmup has been observed
# directly to take up to ~7 minutes on an A100 with this model.
READY_TIMEOUT_S = 600
# int8 quantization's warmup never completed within READY_TIMEOUT_S in the historical sweep
# (results.json on ticket-52-gliner-serve-results records "did not become ready within 600s") --
# retried here with a longer budget to get a real number instead of another timeout.
INT8_READY_TIMEOUT_S = 1200
READY_POLL_INTERVAL_S = 2
SERVER_STOP_TIMEOUT_S = 30

# concurrency=1 Phase-A cells run this many seconds of sequential closed-loop requests (~20
# requests at the sub-second-to-low-second latencies plausible for concurrency=1).
PHASE_A_C1_DURATION_S = 20.0
# concurrency=50 passes (Phase A's no-compile cell, Phase B, the int8 retry) match the historical
# matrix's own concurrency=50 duration so results are comparable to results.json.
LOAD_TEST_CONCURRENCY = 50
LOAD_TEST_DURATION_S = 25.0

# Pulled directly from `git show ticket-52-gliner-serve-results:benchmarks/gliner_serve/results.json`
# (the bfloat16_10ms config: batch_wait_timeout_ms=10, no dtype/quantization override, i.e.
# default bfloat16 with compile on, concurrency=50) -- not re-run, since it's already recorded and
# reproducing it exactly would just burn pod time on a number this diagnostic already has.
HISTORICAL_BFLOAT16_10MS_P50_MS = 50490.31532404479

DEFAULT_CONFIG: dict[str, Any] = {
    "name": "default_compile_on",
    "batch_wait_timeout_ms": 10,
    "dtype": None,
    "quantization": None,
    "no_compile": False,
}
NO_COMPILE_CONFIG: dict[str, Any] = {
    "name": "no_compile",
    "batch_wait_timeout_ms": 10,
    "dtype": None,
    "quantization": None,
    "no_compile": True,
}
# Matches the historical sweep's int8_10ms config exactly (same batch-wait window, concurrency,
# duration) so this retry is a like-for-like rerun with only the ready-timeout extended.
INT8_CONFIG: dict[str, Any] = {
    "name": "int8_10ms",
    "batch_wait_timeout_ms": 10,
    "dtype": None,
    "quantization": "int8",
    "no_compile": False,
}

# Maps a Phase-A cell name to the server config Phase B re-runs at concurrency=50 if that cell
# wins but wasn't already GPU-sampled at concurrency=50 (no_compile_c50 is the one exception,
# handled separately since it's reused rather than re-run).
WINNER_CONFIG_BY_CELL: dict[str, dict[str, Any]] = {
    "default_compile_on_c1": DEFAULT_CONFIG,
    "no_compile_c1": NO_COMPILE_CONFIG,
    "historical_bfloat16_10ms_c50": DEFAULT_CONFIG,
}

GPU_SAMPLE_INTERVAL_S = 1
GPU_STOP_TIMEOUT_S = 10

# Decision gate thresholds from the ticket: 500ms is 2x the ~250ms sub-budget this benchmark's
# real workload needs; 50% GPU utilization is the bar for "the GPU itself is the bottleneck, not
# the serving layer around it."
GATE_P50_THRESHOLD_MS = 500.0
GATE_GPU_UTIL_THRESHOLD_PCT = 50.0


def build_server_cmd(config: dict[str, Any]) -> list[str]:
    """Build the `python -m gliner.serve ...` argv for a diagnostic config."""
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
    if config.get("no_compile"):
        cmd += ["--no-compile"]
    return cmd


def launch_server(cmd: list[str], log_path: Path):
    """Start `cmd` in the background, redirecting its output to `log_path`.

    Returns a real `Popen` handle (not a shell-backgrounded process) so the caller can
    `.terminate()`/`.wait()` on it directly -- this script runs locally on the pod, so no
    SSH-over-network detachment tricks are needed.
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


def start_gpu_sampling(log_path: Path):
    """Spawn `nvidia-smi -l 1` sampling GPU utilization/memory to `log_path`, from this process
    directly -- it stays alive for the whole run, so a plain `.terminate()` when the timed load
    test ends is all that's needed (no SSH-detachment tricks, unlike pod_bench.py's server
    launches, which solve a different problem: surviving past the SSH command that started them)."""
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv", "-l", str(GPU_SAMPLE_INTERVAL_S)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return proc, log_file


def stop_gpu_sampling(proc: subprocess.Popen, log_file) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=GPU_STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log_file.close()


def parse_gpu_log(log_path: Path) -> dict[str, Any] | None:
    """Parse `nvidia-smi --format=csv` output (a header row, then one "NN %, NN MiB" row per
    sample) into mean/max GPU utilization and memory. Returns None if no samples were captured."""
    if not log_path.exists():
        return None
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    utils: list[float] = []
    mems: list[float] = []
    for line in lines[1:]:  # skip the header row
        parts = line.split(",")
        if len(parts) != 2:
            continue
        # Parse both fields into locals before appending either -- an unparsable field (e.g. an
        # "N/A" reading) must drop the whole row, not leave `utils`/`mems` desynchronized.
        try:
            util = float(parts[0].strip().rstrip("%").strip())
            mem = float(parts[1].strip().rstrip("MiB").strip())
        except ValueError:
            continue
        utils.append(util)
        mems.append(mem)
    if not utils:
        return None
    return {
        "n_samples": len(utils),
        "mean_gpu_util_pct": statistics.fmean(utils),
        "max_gpu_util_pct": max(utils),
        "mean_mem_used_mib": statistics.fmean(mems) if mems else None,
        "max_mem_used_mib": max(mems) if mems else None,
    }


def run_pass(
    url: str,
    *,
    concurrency: int,
    duration_s: float,
    windows: list[str],
    labels: dict[str, str],
    sample_gpu: bool,
    gpu_log_path: Path | None,
) -> dict[str, Any]:
    """Run one `load_test.run_load_test` pass against an already-ready server, optionally
    GPU-sampling for its duration."""
    gpu_proc = gpu_file = None
    if sample_gpu:
        assert gpu_log_path is not None
        gpu_proc, gpu_file = start_gpu_sampling(gpu_log_path)
    try:
        result = asyncio.run(load_test.run_load_test(url, concurrency, duration_s, windows, labels))
    finally:
        if gpu_proc is not None:
            stop_gpu_sampling(gpu_proc, gpu_file)
    gpu_stats = parse_gpu_log(gpu_log_path) if sample_gpu and gpu_log_path is not None else None
    return {"concurrency": concurrency, "duration_s": duration_s, "result": result, "gpu_stats": gpu_stats}


def run_config_passes(
    config: dict[str, Any],
    passes: list[dict[str, Any]],
    *,
    url: str,
    log_dir: Path,
    ready_timeout_s: float = READY_TIMEOUT_S,
) -> dict[str, Any]:
    """Start `config`'s server once, wait for it, run each of `passes` sequentially against it
    (each pass: `{"concurrency", "duration_s", "sample_gpu"}`), then tear it down once -- avoids
    paying torch.compile warmup more than once per config when it needs multiple passes."""
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_server_cmd(config)
    log_path = log_dir / f"{config['name']}.log"
    proc, log_file = launch_server(cmd, log_path)
    try:
        labels, windows = load_test.load_fixture()
        endpoint = url.rstrip("/") + load_test.ROUTE_PREFIX
        ready = wait_for_ready(endpoint, labels, windows[0], timeout_s=ready_timeout_s)
        if not ready:
            error = f"server did not become ready within {ready_timeout_s}s"
            pass_results = [
                {
                    "concurrency": p["concurrency"],
                    "duration_s": p["duration_s"],
                    "result": {
                        "n": 0,
                        "n_failed": 0,
                        "p50_ms": None,
                        "p95_ms": None,
                        "p99_ms": None,
                        "mean_ms": None,
                        "throughput_req_s": None,
                        "error": error,
                    },
                    "gpu_stats": None,
                }
                for p in passes
            ]
        else:
            pass_results = []
            for p in passes:
                sample_gpu = bool(p.get("sample_gpu"))
                gpu_log_path = (
                    log_dir / f"{config['name']}_c{p['concurrency']}_gpu.csv" if sample_gpu else None
                )
                pass_results.append(
                    run_pass(
                        url,
                        concurrency=p["concurrency"],
                        duration_s=p["duration_s"],
                        windows=windows,
                        labels=labels,
                        sample_gpu=sample_gpu,
                        gpu_log_path=gpu_log_path,
                    )
                )
    finally:
        stop_server(proc, log_file)
    return {"config": config, "passes": pass_results}


def pick_phase_a_winner(phase_a: dict[str, Any]) -> str:
    """Return the Phase-A cell name whose result has the lowest p50_ms (cells with no p50 -- a
    server that never became ready -- are excluded)."""
    candidates = {
        name: cell["result"].get("p50_ms")
        for name, cell in phase_a.items()
        if cell["result"].get("p50_ms") is not None
    }
    if not candidates:
        raise RuntimeError("no Phase-A cell produced a p50 -- can't pick a winner")
    return min(candidates, key=candidates.get)


def run_phase_b(winner: str, phase_a: dict[str, Any], *, url: str, log_dir: Path) -> dict[str, Any]:
    """Get one concurrency=50, GPU-sampled pass for the Phase-A winner: reuse the no-compile
    concurrency=50 run directly if it won (it was already GPU-sampled in Phase A), otherwise run
    one fresh concurrency=50 pass under the winning config."""
    if winner == "no_compile_c50":
        return {"source": "reused_phase_a_no_compile_c50", "pass": phase_a["no_compile_c50"]}
    config = WINNER_CONFIG_BY_CELL[winner]
    run = run_config_passes(
        config,
        [{"concurrency": LOAD_TEST_CONCURRENCY, "duration_s": LOAD_TEST_DURATION_S, "sample_gpu": True}],
        url=url,
        log_dir=log_dir,
    )
    return {"source": f"fresh_c50_pass_under_{config['name']}", "pass": run["passes"][0]}


def apply_gate(phase_b: dict[str, Any]) -> dict[str, Any]:
    """Apply the ticket's decision gate: proceeding without vLLM validation requires the winning
    config's Phase-B pass to reach BOTH p50 <= 500ms AND mean GPU utilization >= 50%."""
    winner_pass = phase_b["pass"]
    p50_ms = winner_pass["result"].get("p50_ms")
    gpu_stats = winner_pass.get("gpu_stats")
    mean_util = gpu_stats.get("mean_gpu_util_pct") if gpu_stats else None

    passes_latency = p50_ms is not None and p50_ms <= GATE_P50_THRESHOLD_MS
    passes_util = mean_util is not None and mean_util >= GATE_GPU_UTIL_THRESHOLD_PCT
    proceed_to_vllm = not (passes_latency and passes_util)

    return {
        "winner_p50_ms": p50_ms,
        "winner_mean_gpu_util_pct": mean_util,
        "passes_latency_threshold": passes_latency,
        "passes_gpu_util_threshold": passes_util,
        "proceed_to_vllm_validation": proceed_to_vllm,
        "recommendation": (
            "yes -- vLLM validation is warranted (the winning config fails p50<=500ms and/or mean GPU util>=50%)"
            if proceed_to_vllm
            else "no -- gliner[serve] is fixable without vLLM (winning config reaches p50<=500ms and mean GPU util>=50%)"
        ),
    }


def write_results(results_path: Path, diagnosis: dict[str, Any]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(diagnosis, indent=2))


def run_diagnosis(*, url: str, results_path: Path, log_dir: Path) -> dict[str, Any]:
    """Runs the full two-phase diagnosis plus the int8 retry, rewriting `results_path` after each
    step completes -- same crash-safe discipline as run_matrix.py's `run_sweep`."""
    diagnosis: dict[str, Any] = {"phase_a": {}}

    print(f"[{datetime.now()}] Phase A: default (compile-on) at concurrency=1")
    default_run = run_config_passes(
        DEFAULT_CONFIG, [{"concurrency": 1, "duration_s": PHASE_A_C1_DURATION_S}], url=url, log_dir=log_dir
    )
    diagnosis["phase_a"]["default_compile_on_c1"] = default_run["passes"][0]
    write_results(results_path, diagnosis)

    print(f"[{datetime.now()}] Phase A: --no-compile at concurrency=1 and concurrency=50")
    no_compile_run = run_config_passes(
        NO_COMPILE_CONFIG,
        [
            {"concurrency": 1, "duration_s": PHASE_A_C1_DURATION_S},
            {"concurrency": LOAD_TEST_CONCURRENCY, "duration_s": LOAD_TEST_DURATION_S, "sample_gpu": True},
        ],
        url=url,
        log_dir=log_dir,
    )
    diagnosis["phase_a"]["no_compile_c1"] = no_compile_run["passes"][0]
    diagnosis["phase_a"]["no_compile_c50"] = no_compile_run["passes"][1]
    write_results(results_path, diagnosis)

    diagnosis["phase_a"]["historical_bfloat16_10ms_c50"] = {
        "concurrency": LOAD_TEST_CONCURRENCY,
        "duration_s": LOAD_TEST_DURATION_S,
        "result": {"p50_ms": HISTORICAL_BFLOAT16_10MS_P50_MS},
        "gpu_stats": None,
        "source": "results.json on ticket-52-gliner-serve-results, bfloat16_10ms config",
    }
    write_results(results_path, diagnosis)

    winner = pick_phase_a_winner(diagnosis["phase_a"])
    diagnosis["phase_a_winner"] = winner
    print(f"[{datetime.now()}] Phase A winner: {winner}")
    write_results(results_path, diagnosis)

    print(f"[{datetime.now()}] Phase B: GPU-sampled concurrency=50 pass for winner {winner}")
    diagnosis["phase_b"] = run_phase_b(winner, diagnosis["phase_a"], url=url, log_dir=log_dir)
    write_results(results_path, diagnosis)

    print(f"[{datetime.now()}] int8 retry with extended ({INT8_READY_TIMEOUT_S}s) ready timeout")
    int8_run = run_config_passes(
        INT8_CONFIG,
        [{"concurrency": LOAD_TEST_CONCURRENCY, "duration_s": LOAD_TEST_DURATION_S}],
        url=url,
        log_dir=log_dir,
        ready_timeout_s=INT8_READY_TIMEOUT_S,
    )
    diagnosis["int8_retry"] = int8_run["passes"][0]
    write_results(results_path, diagnosis)

    diagnosis["gate_outcome"] = apply_gate(diagnosis["phase_b"])
    write_results(results_path, diagnosis)
    print(f"[{datetime.now()}] gate outcome: {diagnosis['gate_outcome']['recommendation']}")

    return diagnosis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", type=Path, required=True, help="Where to write diagnosis.json.")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Base URL of the gliner[serve] server.")
    args = parser.parse_args()

    log_dir = args.results_path.resolve().parent / "gliner_diagnose_logs"
    run_diagnosis(url=args.url, results_path=args.results_path, log_dir=log_dir)

    print(f"[{datetime.now()}] diagnosis complete -- wrote results to {args.results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
