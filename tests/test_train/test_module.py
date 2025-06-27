import sys
import pathlib
from types import SimpleNamespace
from typing import cast, Tuple, List, Any

import torch
import pytest

# adjust this path to point at your src directory
sys.path.insert(
    0,
    str(
        pathlib.Path(__file__)
        .resolve()
        .parent.parent.parent
        / "src"
    ),
)

import miniformer.train.module as module
from miniformer.train.train_config import TrainConfig  # for typing only

# --- dummies and fixtures --------------------------------------------------

class DummySeq2Seq:
    def __init__(self, config: Any):
        pass

class DummyTransformer:
    def __init__(self, config: Any):
        # needed for classification branch
        self.config = SimpleNamespace(output_dim=4)

@pytest.fixture(autouse=True)
def patch_models(monkeypatch):
    monkeypatch.setattr(module, "Seq2SeqTransformer", DummySeq2Seq)
    monkeypatch.setattr(module, "Transformer", DummyTransformer)
    monkeypatch.setattr(module, "TransformerConfig", lambda **kwargs: None)

# --- helper to fake a TrainConfig ------------------------------------------

def make_cfg(task: str, scheduler: str) -> SimpleNamespace:
    return SimpleNamespace(
        model="seq2seq",
        task=task,
        model_config={},
        lr=0.01,
        weight_decay=0.0,
        scheduler=scheduler,
        warmup_steps=10,
        max_epochs=2,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
    )

# --- tests ---------------------------------------------------------------

def test_configure_optimizers_none():
    # cast to TrainConfig so Pylance sees the right type
    cfg = cast(TrainConfig, make_cfg("language_modeling", "none"))
    lit = module.MiniFormerLitModule(cfg)
    optim = lit.configure_optimizers()
    assert isinstance(optim, torch.optim.Optimizer)


def test_configure_optimizers_linear():
    cfg = cast(TrainConfig, make_cfg("language_modeling", "linear"))
    lit = module.MiniFormerLitModule(cfg)
    # bypass LightningModule.trainer type check by using object.__setattr__
    object.__setattr__(
        lit,
        "trainer",
        SimpleNamespace(estimated_stepping_batches=100),
    )
    # tell the type checker this *must* be the two-list return branch
    opts_scheds = cast(
        Tuple[List[torch.optim.Optimizer], List[Any]],
        lit.configure_optimizers(),
    )
    opts, scheds = opts_scheds
    assert isinstance(opts[0], torch.optim.Optimizer)
    from torch.optim.lr_scheduler import LinearLR

    assert isinstance(scheds[0], LinearLR)


def test_configure_optimizers_onecycle():
    cfg = cast(TrainConfig, make_cfg("language_modeling", "onecycle"))
    lit = module.MiniFormerLitModule(cfg)
    object.__setattr__(
        lit,
        "trainer",
        SimpleNamespace(estimated_stepping_batches=20),
    )
    opts_scheds = cast(
        Tuple[List[torch.optim.Optimizer], List[Any]],
        lit.configure_optimizers(),
    )
    _, scheds = opts_scheds
    from torch.optim.lr_scheduler import OneCycleLR

    assert isinstance(scheds[0], OneCycleLR)


def test_configure_optimizers_cosine():
    cfg = cast(TrainConfig, make_cfg("language_modeling", "cosine"))
    lit = module.MiniFormerLitModule(cfg)
    object.__setattr__(lit, "trainer", SimpleNamespace(estimated_stepping_batches=0))
    opts_scheds = cast(
        Tuple[List[torch.optim.Optimizer], List[Any]],
        lit.configure_optimizers(),
    )
    _, scheds = opts_scheds
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    assert isinstance(scheds[0], CosineAnnealingWarmRestarts)


def test_compute_loss_lm():
    cfg = cast(TrainConfig, make_cfg("language_modeling", "none"))
    lit = module.MiniFormerLitModule(cfg)
    logits = torch.randn(2, 3, 5)
    labels = torch.tensor([[0, 1, 2], [2, 3, 4]])
    batch = {"labels": labels}
    loss, out = lit._compute_loss(batch, logits)
    assert loss.item() > 0
    assert torch.equal(out, logits)


def test_compute_loss_classification():
    cfg = cast(TrainConfig, make_cfg("classification", "none"))
    lit = module.MiniFormerLitModule(cfg)
    lit.model.config.output_dim = 4
    logits = torch.randn(2, 4)
    batch = [{"labels": 0}, {"labels": 1}]
    loss, logit = lit._compute_loss(batch, logits)
    assert loss.item() >= 0


def test_compute_loss_regression():
    cfg = cast(TrainConfig, make_cfg("regression", "none"))
    lit = module.MiniFormerLitModule(cfg)
    preds = torch.tensor([[2.0], [3.0]])
    batch = [{"labels": 1.0}, {"labels": 3.0}]
    loss, out = lit._compute_loss(batch, preds)
    assert out.squeeze().tolist() == pytest.approx([2.0, 3.0])
    expected = ((2.0 - 1.0) ** 2 + (3.0 - 3.0) ** 2) / 2
    assert loss.item() == pytest.approx(expected)
