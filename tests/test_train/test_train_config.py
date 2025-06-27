import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / 'src'))

import json
import pytest
from miniformer.train.train_config import TrainConfig


def test_defaults(tmp_path):
    cfg = TrainConfig()
    assert cfg.lr == 5e-4
    assert cfg.batch_size == 32
    assert cfg.logger == "tensorboard"

    # test save functionality
    cfg.experiment_name = "testexp"
    file = tmp_path / "cfg.json"
    cfg.save(str(file))
    data = json.loads(file.read_text())
    assert data["experiment_name"] == "testexp"


def test_from_cli(monkeypatch):
    args = ["prog", "--lr", "0.1", "--batch_size", "64", "--logger", "wandb"]
    monkeypatch.setattr(sys, "argv", args)
    cfg = TrainConfig.from_cli()
    assert isinstance(cfg, TrainConfig)
    assert cfg.lr == pytest.approx(0.1)
    assert cfg.batch_size == 64
    assert cfg.logger == "wandb"
    