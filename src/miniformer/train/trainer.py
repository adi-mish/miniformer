from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch

from .datamodule import MiniFormerDataModule
from .module import MiniFormerModule
from .train_config import TrainConfig


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _mean_metrics(metrics: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(metrics)
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {
        key: sum(row[key] for row in rows if key in row) / sum(1 for row in rows if key in row)
        for key in keys
    }


def _metric_is_better(value: float, best: Optional[float], metric_name: str) -> bool:
    if best is None:
        return True
    if "loss" in metric_name or "mae" in metric_name or "ppl" in metric_name:
        return value < best
    return value > best


def _write_csv_log(path: Path, row: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_one_epoch(
    module: MiniFormerModule,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    *,
    gradient_clip_val: float = 0.0,
    accumulate_grad_batches: int = 1,
) -> float:
    module.train()
    optimizer.zero_grad()
    total_loss = 0.0
    steps = 0

    for batch_idx, batch in enumerate(dataloader):
        loss = module.training_step(batch, batch_idx)
        (loss / max(accumulate_grad_batches, 1)).backward()

        should_step = (batch_idx + 1) % max(accumulate_grad_batches, 1) == 0
        if should_step:
            if gradient_clip_val and gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(module.parameters(), gradient_clip_val)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += float(loss.detach().cpu())
        steps += 1

    if steps and steps % max(accumulate_grad_batches, 1) != 0:
        if gradient_clip_val and gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(module.parameters(), gradient_clip_val)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate(module: MiniFormerModule, dataloader) -> Dict[str, float]:
    module.eval()
    return _mean_metrics(module.validation_step(batch, i) for i, batch in enumerate(dataloader))


def train_model(
    cfg: TrainConfig,
    *,
    tokenizer=None,
    module: Optional[MiniFormerModule] = None,
    datamodule: Optional[MiniFormerDataModule] = None,
    ckpt_path: Optional[str | Path] = None,
) -> Dict[str, float]:
    seed_everything(cfg.seed, cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.gpus > 0 else "cpu")

    datamodule = datamodule or MiniFormerDataModule(cfg, tokenizer)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader() if cfg.val_path else None

    module = module or MiniFormerModule(cfg)
    if ckpt_path is not None:
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        module.load_state_dict(loaded["state_dict"])
    module.to(device)

    optimizer, scheduler = module.configure_optimizers(steps_per_epoch=len(train_loader))

    run_dir = Path(cfg.work_dir) / cfg.experiment_name
    ckpt_dir = run_dir / "checkpoints"
    log_path = run_dir / "metrics.csv"

    best_value: Optional[float] = None
    best_metrics: Dict[str, float] = {}
    epochs_without_improvement = 0

    for epoch in range(cfg.max_epochs):
        train_loss = train_one_epoch(
            module,
            train_loader,
            optimizer,
            scheduler,
            gradient_clip_val=cfg.gradient_clip_val,
            accumulate_grad_batches=cfg.accumulate_grad_batches,
        )

        metrics: Dict[str, float] = {"epoch": float(epoch), "train_loss": train_loss}
        if val_loader is not None:
            metrics.update(evaluate(module, val_loader))

        if cfg.logger == "csv":
            _write_csv_log(log_path, metrics)

        monitor_value = metrics.get(cfg.checkpoint_metric)
        if monitor_value is not None and _metric_is_better(
            monitor_value,
            best_value,
            cfg.checkpoint_metric,
        ):
            best_value = monitor_value
            best_metrics = metrics
            epochs_without_improvement = 0
            module.save_checkpoint(
                ckpt_dir / "best.pt", optimizer=optimizer, epoch=epoch, metrics=metrics
            )
        else:
            epochs_without_improvement += 1

        module.save_checkpoint(
            ckpt_dir / "last.pt", optimizer=optimizer, epoch=epoch, metrics=metrics
        )

        if (
            cfg.early_stopping_patience > 0
            and val_loader is not None
            and epochs_without_improvement >= cfg.early_stopping_patience
        ):
            break

    if cfg.test_path:
        test_metrics = evaluate(module, datamodule.test_dataloader())
        best_metrics.update({f"test_{key}": value for key, value in test_metrics.items()})

    return best_metrics or metrics


def main():
    cfg = TrainConfig.from_cli()

    tokenizer = None
    if cfg.task == "language_modeling":
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except ImportError as exc:
            raise ImportError("Install transformers or provide your own tokenizer") from exc

    metrics = train_model(cfg, tokenizer=tokenizer)
    if metrics:
        print(" ".join(f"{key}={value:.4f}" for key, value in metrics.items()))


if __name__ == "__main__":
    main()
