import json

import torch

from miniformer.train.datamodule import MiniFormerDataModule
from miniformer.train.module import MiniFormerModule
from miniformer.train.train_config import TrainConfig
from miniformer.train.trainer import train_model


class DummyTokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size

    def encode(self, text, add_special_tokens=True):
        return [ord(char) % self.vocab_size for char in text]


def test_language_modeling_training(tmp_path):
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    train_rows = [{"text": f"sample text {i}"} for i in range(12)]
    val_rows = [{"text": f"validation text {i}"} for i in range(6)]
    train_file.write_text("\n".join(json.dumps(row) for row in train_rows))
    val_file.write_text("\n".join(json.dumps(row) for row in val_rows))

    cfg = TrainConfig(
        task="language_modeling",
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
        experiment_name="lm",
        model_config={
            "vocab_size": 100,
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 1,
            "d_ff": 32,
            "dropout": 0.0,
        },
    )
    tokenizer = DummyTokenizer(vocab_size=100)
    module = MiniFormerModule(cfg)
    initial = {name: param.detach().clone() for name, param in module.named_parameters()}

    metrics = train_model(cfg, tokenizer=tokenizer, module=module)

    assert "val_ppl" in metrics
    assert any(
        not torch.allclose(initial[name], param) for name, param in module.named_parameters()
    )

    datamodule = MiniFormerDataModule(cfg, tokenizer=tokenizer)
    datamodule.setup()
    batch = next(iter(datamodule.val_dataloader()))
    with torch.no_grad():
        outputs, _ = module.forward_batch(batch)
    assert outputs.shape == (*batch["input_ids"].shape, cfg.model_config["vocab_size"])
