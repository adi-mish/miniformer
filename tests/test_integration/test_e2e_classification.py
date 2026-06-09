import json

import torch

from miniformer.train.datamodule import MiniFormerDataModule
from miniformer.train.module import MiniFormerModule
from miniformer.train.train_config import TrainConfig
from miniformer.train.trainer import train_model


def test_classification_training(tmp_path):
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    train_rows = [{"input": f"sample {i}", "label": i % 3} for i in range(12)]
    val_rows = [{"input": f"val {i}", "label": i % 3} for i in range(6)]
    train_file.write_text("\n".join(json.dumps(row) for row in train_rows))
    val_file.write_text("\n".join(json.dumps(row) for row in val_rows))

    cfg = TrainConfig(
        task="classification",
        model="encoder",
        train_path=str(train_file),
        val_path=str(val_file),
        batch_size=3,
        num_workers=0,
        shuffle_train=False,
        max_epochs=1,
        lr=0.01,
        scheduler="none",
        logger="none",
        gpus=0,
        early_stopping_patience=0,
        work_dir=str(tmp_path),
        experiment_name="classification",
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
    module = MiniFormerModule(cfg)
    initial = {name: param.detach().clone() for name, param in module.named_parameters()}

    metrics = train_model(cfg, module=module)

    assert "val_loss" in metrics
    assert any(not torch.allclose(initial[name], param) for name, param in module.named_parameters())

    datamodule = MiniFormerDataModule(cfg)
    datamodule.setup()
    batch = next(iter(datamodule.val_dataloader()))
    with torch.no_grad():
        outputs, _ = module.forward_batch(batch)
    assert outputs.shape[-1] == 3
