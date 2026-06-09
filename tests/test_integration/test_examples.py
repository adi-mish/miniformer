import subprocess
import sys
from pathlib import Path


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
