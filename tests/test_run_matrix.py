"""Tests for `benchmarks/gliner_serve/run_matrix.py`'s config-switching logic.

Loaded via `importlib.util.spec_from_file_location`, same as `test_measure_gliner_concurrency.py`.
`run_matrix.py` itself adds its own directory to `sys.path` so its internal `import load_test`
resolves to the real, standalone `load_test.py` next to it.

The real `python -m gliner.serve` launch is swapped out for a tiny stdlib-only stub HTTP server
(no GPU/gliner install needed) by monkeypatching `build_server_cmd` -- the documented seam for
this -- so the full start/wait-for-ready/load-test/kill/next-config orchestration is exercised
against a real subprocess and real HTTP calls, just not real GLiNER inference.
"""

import importlib.util
import json
import socket
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "gliner_serve" / "run_matrix.py"
_spec = importlib.util.spec_from_file_location("run_matrix", _SCRIPT_PATH)
run_matrix = importlib.util.module_from_spec(_spec)
sys.modules["run_matrix"] = run_matrix
_spec.loader.exec_module(run_matrix)


_STUB_SERVER_SRC = '''
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps({"entities": []}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args, _unknown = parser.parse_known_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
'''


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _patch_for_stub(monkeypatch, tmp_path, port):
    stub_script = tmp_path / "stub_server.py"
    stub_script.write_text(_STUB_SERVER_SRC, encoding="utf-8")

    def fake_build_server_cmd(config):
        return [sys.executable, str(stub_script), "--port", str(port)]

    monkeypatch.setattr(run_matrix, "build_server_cmd", fake_build_server_cmd)
    monkeypatch.setattr(run_matrix, "READY_TIMEOUT_S", 10)
    monkeypatch.setattr(run_matrix, "READY_POLL_INTERVAL_S", 0.1)
    monkeypatch.setattr(run_matrix, "LOAD_TEST_CONCURRENCY", 2)
    monkeypatch.setattr(run_matrix, "LOAD_TEST_DURATION_S", 0.3)


def test_run_config_starts_waits_load_tests_and_kills_the_stub_server(monkeypatch, tmp_path):
    port = _free_port()
    _patch_for_stub(monkeypatch, tmp_path, port)
    config = {"name": "bfloat16_10ms", "batch_wait_timeout_ms": 10, "dtype": None, "quantization": None}

    entry = run_matrix.run_config(config, url=f"http://127.0.0.1:{port}", log_dir=tmp_path / "logs")

    assert entry["config"] == config
    assert entry["result"]["n"] > 0
    assert entry["result"]["n_failed"] == 0
    # A second config re-launched on the very same port only succeeds if run_config's kill of the
    # first stub server actually freed it -- a stronger, less timing-sensitive check than probing
    # the socket directly (whose TIME_WAIT-ish teardown timing isn't guaranteed the instant the
    # subprocess handle reports exited).
    second_entry = run_matrix.run_config(config, url=f"http://127.0.0.1:{port}", log_dir=tmp_path / "logs")
    assert second_entry["result"]["n"] > 0
    assert second_entry["result"]["n_failed"] == 0


def test_stop_server_actually_terminates_the_process(tmp_path):
    proc, log_file = run_matrix.launch_server(
        [sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "stub.log"
    )

    run_matrix.stop_server(proc, log_file)

    assert proc.poll() is not None  # process has exited, not left running in the background


def test_pick_best_selects_lowest_p95_among_successful_configs():
    results = [
        {"config": {"name": "a"}, "result": {"n_failed": 0, "p95_ms": 200.0}},
        {"config": {"name": "b"}, "result": {"n_failed": 0, "p95_ms": 100.0}},
        {"config": {"name": "c"}, "result": {"n_failed": 1, "p95_ms": 10.0}},  # failed -- excluded
    ]
    assert run_matrix.pick_best(results)["name"] == "b"


def test_pick_best_falls_back_to_default_when_all_configs_failed():
    results = [
        {"config": {"name": "a"}, "result": {"n_failed": 5, "p95_ms": 200.0}},
        {"config": {"name": "b"}, "result": {"n_failed": 2, "p95_ms": None}},
    ]
    assert run_matrix.pick_best(results) == run_matrix.FALLBACK_BEST_CONFIG


def test_build_stage_2_configs_reuses_best_dtype_and_quantization_at_three_windows():
    best = {"name": "int8_10ms", "dtype": None, "quantization": "int8"}

    configs = run_matrix.build_stage_2_configs(best)

    assert [c["batch_wait_timeout_ms"] for c in configs] == [5, 20, 30]
    assert all(c["dtype"] is None and c["quantization"] == "int8" for c in configs)
    assert [c["name"] for c in configs] == ["int8_10ms_5ms", "int8_10ms_20ms", "int8_10ms_30ms"]


def test_full_sweep_produces_one_result_per_config_against_the_stub_server(monkeypatch, tmp_path):
    port = _free_port()
    _patch_for_stub(monkeypatch, tmp_path, port)
    results_path = tmp_path / "results.json"
    monkeypatch.setattr(
        sys, "argv", ["run_matrix.py", "--results-path", str(results_path), "--url", f"http://127.0.0.1:{port}"]
    )

    run_matrix.main()

    data = json.loads(results_path.read_text())
    assert len(data) == 6  # 3 stage-1 configs + 3 stage-2 configs at the best's dtype/quantization
    stage_1_entries, stage_2_entries = data[:3], data[3:]
    assert [e["config"]["name"] for e in stage_1_entries] == [c["name"] for c in run_matrix.STAGE_1_CONFIGS]
    # The stub server responds near-instantly for every stage-1 config, so which one comes out
    # "best" (lowest p95_ms) is real measured timing noise, not a fixed outcome -- assert stage 2
    # follows whichever stage-1 config actually won, rather than hardcoding one.
    expected_best = run_matrix.pick_best(stage_1_entries)
    assert [e["config"]["name"] for e in stage_2_entries] == [f"{expected_best['name']}_{w}ms" for w in (5, 20, 30)]
    for entry in data:
        assert entry["result"]["n"] > 0
        assert entry["result"]["n_failed"] == 0
