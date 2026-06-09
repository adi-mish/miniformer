import importlib
import json
import subprocess
import sys
from pathlib import Path


def test_make_tiny_jsonl_script_writes_all_tasks(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "miniformer.scripts.make_tiny_jsonl",
            "--task",
            "all",
            "--output-dir",
            str(tmp_path),
            "--train-rows",
            "2",
            "--val-rows",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    for task in ["classification", "regression", "language_modeling"]:
        train_path = tmp_path / task / "train.jsonl"
        val_path = tmp_path / task / "val.jsonl"
        assert train_path.exists()
        assert val_path.exists()
        assert len(train_path.read_text().splitlines()) == 2
        assert len(val_path.read_text().splitlines()) == 1


def test_trace_report_script_writes_static_artifacts(tmp_path):
    html_path = tmp_path / "trace.html"
    json_path = tmp_path / "trace.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "miniformer.scripts.write_trace_report",
            "--output-html",
            str(html_path),
            "--output-json",
            str(json_path),
            "--seq-len",
            "4",
            "--d-model",
            "8",
            "--n-heads",
            "2",
            "--n-layers",
            "1",
            "--d-ff",
            "16",
            "--top-k",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Miniformer Trace" in html_path.read_text()
    payload = json.loads(json_path.read_text())
    assert payload["output_shape"] == [1, 4, 64]
    assert payload["cache"]["attempted"] is True


def test_check_script_lists_available_checks():
    result = subprocess.run(
        [sys.executable, "-m", "miniformer.scripts.run_checks", "--list"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "tests: uv run pytest -q" in result.stdout
    assert "build: uv build" in result.stdout


def test_validate_jsonl_script_reports_schema_errors(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"input": "", "label": 0}\n')

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "miniformer.scripts.validate_jsonl",
            str(path),
            "--task",
            "classification",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "errors=1" in result.stdout
    assert "input string" in result.stdout


def test_script_modules_import_without_side_effects():
    scripts_dir = Path(__file__).resolve().parents[2] / "src" / "miniformer" / "scripts"
    script_names = {
        script.stem for script in scripts_dir.glob("*.py") if script.name not in {"__init__.py"}
    }

    assert script_names == {
        "make_tiny_jsonl",
        "run_checks",
        "validate_jsonl",
        "write_trace_report",
    }
    for script_name in script_names:
        module = importlib.import_module(f"miniformer.scripts.{script_name}")
        assert callable(module.main)
