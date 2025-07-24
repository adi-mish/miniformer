import os
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from .train_config import TrainConfig
from .datamodule import MiniFormerDataModule
from .module import MiniFormerLitModule

def main():
    cfg = TrainConfig.from_cli()

    # Reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # Data & Model
    tokenizer = None
    if cfg.task == "language_modeling":
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except ImportError:
            raise ImportError("Install transformers or provide your own tokenizer")

    datamodule = MiniFormerDataModule(cfg, tokenizer)
    model = MiniFormerLitModule(cfg)

    # Logger
    if cfg.logger == "tensorboard":
        logger = TensorBoardLogger(save_dir=cfg.work_dir, name=cfg.experiment_name)
    elif cfg.logger == "wandb":
        logger = WandbLogger(project=cfg.experiment_name, save_dir=cfg.work_dir)
    elif cfg.logger == "csv":
        logger = CSVLogger(cfg.work_dir, name=cfg.experiment_name)
    else:
        logger = False  # disable logging

    # ------------------ Checkpoint filename that only uses logged metrics ------------------
    def _build_ckpt_filename(cfg):
        # Always include epoch
        pieces = ["{epoch}"]
        # Always include the metric we monitor
        pieces.append(f"{{{cfg.checkpoint_metric}:.3f}}")
        # Add val_mae only if task=="regression"
        if cfg.task == "regression":
            pieces.append("{val_mae:.3f}")
        return "-".join(pieces)

    callbacks = []
    # Use the logger's log_dir (includes version_XX) when available; else fall back
    if logger and hasattr(logger, 'log_dir') and logger.log_dir:
        ckpt_root = logger.log_dir
    else:
        ckpt_root = os.path.join(cfg.work_dir, cfg.experiment_name,
                                 "version_manual")
    ckpt_dir = os.path.join(ckpt_root, "checkpoints")

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=_build_ckpt_filename(cfg),
        monitor=cfg.checkpoint_metric,
        mode="min" if "loss" in cfg.checkpoint_metric else "max",
        save_top_k=3,            # keep a few best checkpoints
        save_last=True,          # also save the very last one
        auto_insert_metric_name=False
    )
    callbacks.append(ckpt_cb)

    if cfg.early_stopping_patience > 0:
        es_cb = EarlyStopping(
            monitor=cfg.checkpoint_metric,
            patience=cfg.early_stopping_patience,
            mode="min" if "loss" in cfg.checkpoint_metric else "max",
            strict=True,
        )
        callbacks.append(es_cb)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=("gpu" if torch.cuda.is_available() and cfg.gpus > 0
                     else "cpu"),
        devices=cfg.gpus if cfg.gpus > 0 else 1,
        precision=cfg.precision,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        gradient_clip_val=cfg.gradient_clip_val,
        deterministic=getattr(cfg, "deterministic", False),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=50,
    )

    # Fit (and optionally test)
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule, verbose=False)
    if cfg.test_path:
        trainer.test(datamodule=datamodule)


if __name__ == "__main__":
    main()
