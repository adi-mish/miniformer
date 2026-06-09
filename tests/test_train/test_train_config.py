import json
import sys
import pytest
from miniformer.train.train_config import TrainConfig


def test_defaults(tmp_path):
    cfg = TrainConfig()
    assert cfg.lr == 5e-4
    assert cfg.batch_size == 32
    assert cfg.logger == "csv"

    # test save functionality
    cfg.experiment_name = "testexp"
    file = tmp_path / "cfg.json"
    cfg.save(str(file))
    data = json.loads(file.read_text())
    assert data["experiment_name"] == "testexp"


def test_from_cli(monkeypatch):
    args = ["prog", "--lr", "0.1", "--batch_size", "64", "--logger", "none"]
    monkeypatch.setattr(sys, "argv", args)
    cfg = TrainConfig.from_cli()
    assert isinstance(cfg, TrainConfig)
    assert cfg.lr == pytest.approx(0.1)
    assert cfg.batch_size == 64
    assert cfg.logger == "none"


def test_from_cli_model_config_json(monkeypatch):
    args = [
        "prog",
        "--model_config",
        '{"vocab_size": 128, "d_model": 32, "n_heads": 4}',
        "--precision",
        "32",
        "--deterministic",
    ]
    monkeypatch.setattr(sys, "argv", args)

    cfg = TrainConfig.from_cli()

    assert cfg.model_config == {"vocab_size": 128, "d_model": 32, "n_heads": 4}
    assert cfg.precision == 32
    assert cfg.deterministic is True


def test_from_cli_config_json_overrides_args(monkeypatch, tmp_path):
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps({
        "batch_size": 16,
        "logger": "csv",
        "model_config": {"vocab_size": 256},
    }))
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--batch_size", "64", "--config_json", str(config_path)],
    )

    cfg = TrainConfig.from_cli()

    assert cfg.batch_size == 16
    assert cfg.logger == "csv"
    assert cfg.model_config == {"vocab_size": 256}
