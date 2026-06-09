import torch

from miniformer.train.module import MiniFormerModule
from miniformer.train.train_config import TrainConfig


def make_cfg():
    cfg = TrainConfig()
    cfg.task = "classification"
    cfg.model = "encoder"
    cfg.model_config = {
        "vocab_size": 100,
        "d_model": 16,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 32,
        "dropout": 0.0,
        "output_mode": "projection",
        "output_dim": 3,
    }
    cfg.lr = 0.01
    cfg.max_epochs = 2
    return cfg


def test_lr_scheduler_compatibility():
    expected = {
        "linear": torch.optim.lr_scheduler.LinearLR,
        "onecycle": torch.optim.lr_scheduler.OneCycleLR,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    }
    for scheduler_name in ["none", *expected.keys()]:
        cfg = make_cfg()
        cfg.scheduler = scheduler_name  # type: ignore[assignment]
        model = MiniFormerModule(cfg)
        optimizer, scheduler = model.configure_optimizers(steps_per_epoch=3)
        assert isinstance(optimizer, torch.optim.Optimizer)
        if scheduler_name == "none":
            assert scheduler is None
        else:
            assert isinstance(scheduler, expected[scheduler_name])


def test_gradient_clipping_compatibility():
    for clip_val in [0.1, 0.5, 1.0]:
        cfg = make_cfg()
        model = MiniFormerModule(cfg)
        loss = model.training_step(
            [{"input": "test", "labels": 0}, {"input": "more", "labels": 1}], 0
        )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        total_norm_after = torch.linalg.vector_norm(
            torch.stack(
                [
                    param.grad.detach().norm()
                    for param in model.parameters()
                    if param.grad is not None
                ]
            )
        ).item()

        assert total_norm_after <= clip_val + 1e-5


def test_precision_config_values_do_not_prevent_model_initialization():
    for precision in [16, 32, 64, "bf16"]:
        cfg = make_cfg()
        cfg.precision = precision  # type: ignore[assignment]
        model = MiniFormerModule(cfg)
        assert model.training_step([{"input": "test", "labels": 0}], 0).isfinite()
