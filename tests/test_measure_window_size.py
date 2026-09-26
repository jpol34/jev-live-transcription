import importlib.util
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

_spec = importlib.util.spec_from_file_location(
    "measure_window_size", _SCRIPTS_DIR / "measure_window_size.py"
)
measure_window_size = importlib.util.module_from_spec(_spec)
sys.modules["measure_window_size"] = measure_window_size
_spec.loader.exec_module(measure_window_size)


def test_format_recall_none_is_na():
    assert measure_window_size._format_recall(None) == "n/a"


def test_format_recall_formats_as_percent():
    assert measure_window_size._format_recall(0.5) == "50%"
    assert measure_window_size._format_recall(1.0) == "100%"


def test_print_recall_table_runs_without_error(capsys):
    recall_by_window = {
        200: {field: {"recall": 0.5, "n_expected": 2, "n_matched": 1} for field in measure_window_size.score_recall.FIELDS},
        400: {field: {"recall": None, "n_expected": 0, "n_matched": 0} for field in measure_window_size.score_recall.FIELDS},
    }

    measure_window_size.print_recall_table(recall_by_window)

    captured = capsys.readouterr()
    assert "window_chars" in captured.out
    assert "50%" in captured.out
    assert "n/a" in captured.out
