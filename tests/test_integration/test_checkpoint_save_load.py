import json
from types import SimpleNamespace

import torch

from miniformer.train.module import MiniFormerModule
from miniformer.train.train_config import TrainConfig
from miniformer.train.trainer import train_model


def make_data(tmp_path):
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    rows = [{"input": f"sample {i}", "label": i % 3} for i in range(12)]
    train_file.write_text("\n".join(json.dumps(row) for row in rows))
    val_file.write_text("\n".join(json.dumps(row) for row in rows[:6]))
    return SimpleNamespace(train_file=str(train_file), val_file=str(val_file))


def make_cfg(tmp_path, data, max_epochs=1):
    return TrainConfig(
        task="classification",
        train_path=data.train_file,
        val_path=data.val_file,
        batch_size=3,
        num_workers=0,
        shuffle_train=False,
        max_epochs=max_epochs,
        lr=0.01,
        scheduler="none",
        logger="none",
        gpus=0,
        early_stopping_patience=0,
        work_dir=str(tmp_path),
        experiment_name="checkpoint_test",
        model_config={
            "vocab_size": 100,
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 1,
            "d_ff": 32,
            "dropout": 0.0,
            "output_dim": 3,
        },
    )


def test_save_and_load_checkpoint(tmp_path):
    data = make_data(tmp_path)
    cfg = make_cfg(tmp_path, data)
    metrics = train_model(cfg)

    checkpoint = tmp_path / "checkpoint_test" / "checkpoints" / "best.pt"
    assert checkpoint.exists()
    assert "val_loss" in metrics

    loaded = MiniFormerModule.load_checkpoint(checkpoint, cfg=cfg)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)

    for name, param in loaded.state_dict().items():
        assert torch.allclose(param, saved["state_dict"][name])


def test_resume_training_from_checkpoint(tmp_path):
    data = make_data(tmp_path)
    cfg = make_cfg(tmp_path, data, max_epochs=1)
    train_model(cfg)

    checkpoint = tmp_path / "checkpoint_test" / "checkpoints" / "last.pt"
    before = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]

    cfg.max_epochs = 1
    resumed = MiniFormerModule(cfg)
    train_model(cfg, module=resumed, ckpt_path=checkpoint)

    after = resumed.state_dict()
    assert any(not torch.allclose(before[name], after[name]) for name in before)
