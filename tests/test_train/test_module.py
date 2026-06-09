from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from miniformer.train.module import MiniFormerModule


def make_cfg(task: str, scheduler: str = "none", output_dim: Optional[int] = None) -> SimpleNamespace:
    model_config = {
        "vocab_size": 20,
        "d_model": 8,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 16,
        "dropout": 0.0,
    }
    if output_dim is not None:
        model_config["output_dim"] = output_dim
    elif task == "classification":
        model_config["output_dim"] = 4
    elif task == "regression":
        model_config["output_dim"] = 1

    return SimpleNamespace(
        model="seq2seq",
        task=task,
        model_config=model_config,
        lr=0.01,
        weight_decay=0.0,
        scheduler=scheduler,
        warmup_steps=2,
        max_epochs=2,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
    )


def test_configure_optimizers_none():
    module = MiniFormerModule(make_cfg("language_modeling", "none"))
    optimizer, scheduler = module.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert scheduler is None


@pytest.mark.parametrize(
    "scheduler_name,scheduler_type",
    [
        ("linear", torch.optim.lr_scheduler.LinearLR),
        ("onecycle", torch.optim.lr_scheduler.OneCycleLR),
        ("cosine", torch.optim.lr_scheduler.CosineAnnealingWarmRestarts),
    ],
)
def test_configure_optimizers_schedulers(scheduler_name, scheduler_type):
    module = MiniFormerModule(make_cfg("language_modeling", scheduler_name))
    optimizer, scheduler = module.configure_optimizers(steps_per_epoch=3)
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert isinstance(scheduler, scheduler_type)


def test_compute_loss_lm():
    module = MiniFormerModule(make_cfg("language_modeling", "none"))
    logits = torch.randn(2, 3, 20)
    labels = torch.tensor([[0, 1, 2], [2, 3, 4]])
    loss, out = module._compute_loss({"labels": labels}, logits)
    assert loss.item() > 0
    assert torch.equal(out, logits)


def test_compute_loss_classification():
    module = MiniFormerModule(make_cfg("classification", "none"))
    logits = torch.randn(2, 4)
    loss, out = module._compute_loss([{"labels": 0}, {"labels": 1}], logits)
    assert loss.item() >= 0
    assert out.shape == (2, 4)


def test_compute_loss_regression():
    module = MiniFormerModule(make_cfg("regression", "none"))
    preds = torch.tensor([[2.0], [3.0]])
    loss, out = module._compute_loss([{"labels": 1.0}, {"labels": 3.0}], preds)
    assert out.squeeze().tolist() == pytest.approx([2.0, 3.0])
    assert loss.item() == pytest.approx(0.5)


def test_training_step_classification_uses_real_model():
    module = MiniFormerModule(make_cfg("classification", "none"))
    batch = [{"input": "aa", "labels": 0}, {"input": "bb", "labels": 1}]
    loss = module.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad


def test_validation_step_returns_metrics():
    module = MiniFormerModule(make_cfg("classification", "none"))
    batch = [{"input": "aa", "labels": 0}, {"input": "bb", "labels": 1}]
    metrics = module.validation_step(batch, 0)
    assert "val_loss" in metrics
    assert "val_accuracy" in metrics
