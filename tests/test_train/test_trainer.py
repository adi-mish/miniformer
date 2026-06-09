import torch

from miniformer.train.train_config import TrainConfig
from miniformer.train.trainer import (
    _mean_metrics,
    evaluate,
    seed_everything,
    train_model,
    train_one_epoch,
)


class TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.cfg = None

    def training_step(self, batch, batch_idx=0):
        x, y = batch
        return ((x * self.weight - y) ** 2).mean()

    def validation_step(self, batch, batch_idx=0):
        return {"val_loss": float(self.training_step(batch, batch_idx).detach())}

    def configure_optimizers(self, steps_per_epoch=None):
        return torch.optim.SGD(self.parameters(), lr=0.1), None

    def save_checkpoint(self, path, *, optimizer=None, epoch=0, metrics=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "metrics": metrics or {}}, path)


def test_seed_everything_reproducible():
    seed_everything(123)
    first = torch.rand(2)
    seed_everything(123)
    second = torch.rand(2)
    assert torch.equal(first, second)


def test_train_one_epoch_updates_parameters():
    module = TinyModule()
    loader = [(torch.tensor([1.0]), torch.tensor([0.0]))]
    optimizer, scheduler = module.configure_optimizers()

    before = module.weight.detach().clone()
    loss = train_one_epoch(module, loader, optimizer, scheduler)

    assert loss > 0
    assert not torch.equal(before, module.weight.detach())


def test_evaluate_averages_metrics():
    module = TinyModule()
    loader = [
        (torch.tensor([1.0]), torch.tensor([0.0])),
        (torch.tensor([2.0]), torch.tensor([0.0])),
    ]
    metrics = evaluate(module, loader)
    assert metrics["val_loss"] == 2.5


def test_mean_metrics_averages_only_present_keys():
    metrics = _mean_metrics(
        [
            {"val_loss": 2.0, "val_accuracy": 1.0},
            {"val_loss": 4.0},
        ]
    )

    assert metrics == {"val_accuracy": 1.0, "val_loss": 3.0}


def test_train_model_smoke(tmp_path):
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    train_path.write_text('{"input": "aa", "label": 0}\n{"input": "bb", "label": 1}\n')
    val_path.write_text('{"input": "aa", "label": 0}\n{"input": "bb", "label": 1}\n')

    cfg = TrainConfig(
        task="classification",
        model="encoder",
        train_path=str(train_path),
        val_path=str(val_path),
        batch_size=2,
        num_workers=0,
        shuffle_train=False,
        max_epochs=1,
        gpus=0,
        logger="none",
        early_stopping_patience=0,
        model_config={
            "vocab_size": 20,
            "d_model": 8,
            "n_heads": 2,
            "n_layers": 1,
            "d_ff": 16,
            "dropout": 0.0,
            "output_dim": 2,
        },
        work_dir=str(tmp_path),
        experiment_name="smoke",
    )

    metrics = train_model(cfg)

    assert "val_loss" in metrics
    assert (tmp_path / "smoke" / "checkpoints" / "best.pt").exists()
    assert (tmp_path / "smoke" / "checkpoints" / "last.pt").exists()
