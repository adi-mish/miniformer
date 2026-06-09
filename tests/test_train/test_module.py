from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from miniformer.data.preprocessing import collate_records
from miniformer.train.module import MiniFormerModule


def make_cfg(
    task: str, scheduler: str = "none", output_dim: Optional[int] = None
) -> SimpleNamespace:
    model_config = {
        "vocab_size": 20,
        "d_model": 8,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 16,
        "dropout": 0.0,
        "output_mode": "vocab" if task == "language_modeling" else "projection",
    }
    if output_dim is not None:
        model_config["output_dim"] = output_dim
    elif task == "classification":
        model_config["output_dim"] = 4
    elif task == "regression":
        model_config["output_dim"] = 1

    return SimpleNamespace(
        model="seq2seq" if task == "language_modeling" else "encoder",
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
    labels = torch.tensor([0, 1])
    loss, out = module._compute_loss((labels,), logits)
    assert loss.item() >= 0
    assert out.shape == (2, 4)


def test_compute_loss_regression():
    module = MiniFormerModule(make_cfg("regression", "none"))
    preds = torch.tensor([[2.0], [3.0]])
    labels = torch.tensor([1.0, 3.0])
    loss, out = module._compute_loss((labels,), preds)
    assert out.squeeze().tolist() == pytest.approx([2.0, 3.0])
    assert loss.item() == pytest.approx(0.5)


def test_training_step_classification_uses_real_model():
    module = MiniFormerModule(make_cfg("classification", "none"))
    batch = collate_records(
        [{"input": "aa", "labels": 0}, {"input": "bb", "labels": 1}],
        task="classification",
        vocab_size=20,
    )
    loss = module.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad


def test_raw_records_are_rejected_by_training_module():
    module = MiniFormerModule(make_cfg("classification", "none"))

    with pytest.raises(TypeError, match="tensor batch dictionary"):
        module.training_step([{"input": "aa", "labels": 0}], 0)


def test_empty_string_batch_raises_clear_preprocessing_error():
    with pytest.raises(ValueError, match="must not be empty"):
        collate_records(
            [{"input": "", "labels": 0}],
            task="classification",
            vocab_size=20,
        )


def test_training_module_rejects_unprocessed_text_batch():
    module = MiniFormerModule(make_cfg("classification", "none"))

    with pytest.raises(TypeError, match="Raw text"):
        module.training_step({"input": "", "labels": torch.tensor([0])}, 0)


def test_non_lm_training_defaults_to_bidirectional_attention():
    module = MiniFormerModule(make_cfg("classification", "none"))

    assert module.cfg.model_config["causal"] is False
    assert module.model.config.causal is False


def test_language_modeling_keeps_causal_attention_default():
    module = MiniFormerModule(make_cfg("language_modeling", "none"))

    assert module.model.config.causal is True


def test_invalid_task_model_combination_raises():
    cfg = make_cfg("classification", "none")
    cfg.model = "seq2seq"

    with pytest.raises(ValueError, match="requires model='encoder'"):
        MiniFormerModule(cfg)


def test_non_lm_task_requires_output_dim():
    cfg = make_cfg("classification", "none")
    del cfg.model_config["output_dim"]

    with pytest.raises(ValueError, match="requires model_config\\['output_dim'\\]"):
        MiniFormerModule(cfg)


def test_validation_step_returns_metrics():
    module = MiniFormerModule(make_cfg("classification", "none"))
    batch = collate_records(
        [{"input": "aa", "labels": 0}, {"input": "bb", "labels": 1}],
        task="classification",
        vocab_size=20,
    )
    metrics = module.validation_step(batch, 0)
    assert "val_loss" in metrics
    assert "val_accuracy" in metrics
