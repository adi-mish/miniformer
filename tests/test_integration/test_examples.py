import importlib.util
import subprocess
import sys
from pathlib import Path


def import_module_from_path(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_examples_import_without_side_effects():
    examples_dir = Path(__file__).resolve().parents[2] / "examples"

    for script in examples_dir.glob("*.py"):
        import_module_from_path(script)


def test_train_model_example_smoke(tmp_path):
    script = Path(__file__).resolve().parents[2] / "examples" / "train_model.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--epochs",
            "0",
            "--batch-size",
            "4",
            "--output-dir",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "final_model" / "model.pt").exists()
    assert (tmp_path / "final_model" / "config.json").exists()
