from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from miniformer.config.model_config import TransformerConfig
from miniformer.data.batch import Batch, BatchKind
from miniformer.model.seq2seq_transformer import Seq2SeqTransformer
from miniformer.model.transformer import Transformer
from miniformer.train.checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    checkpoint_metadata,
    checkpoint_train_config,
    validate_checkpoint_compatibility,
)
from miniformer.train.pooling import PoolingMode, pool_sequence_outputs
from miniformer.train.train_config import TrainConfig


class MiniFormerModule(nn.Module):
    """Small plain-PyTorch wrapper around MiniFormer models."""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.model: nn.Module

        if cfg.task == "language_modeling" and cfg.model != "seq2seq":
            raise ValueError("language_modeling currently requires model='seq2seq'")
        if cfg.task != "language_modeling" and cfg.model != "encoder":
            raise ValueError(f"{cfg.task} currently requires model='encoder'")

        model_config = dict(cfg.model_config)
        if cfg.task == "language_modeling" and model_config.get("output_mode") != "vocab":
            raise ValueError("language_modeling requires model_config['output_mode']='vocab'")
        if cfg.task != "language_modeling" and model_config.get("output_dim") is None:
            raise ValueError(f"{cfg.task} requires model_config['output_dim']")
        if cfg.task != "language_modeling" and model_config.get("output_mode") != "projection":
            raise ValueError(f"{cfg.task} requires model_config['output_mode']='projection'")
        if cfg.task != "language_modeling" and "causal" not in model_config:
            model_config["causal"] = False
        self.cfg.model_config = model_config

        if cfg.task == "language_modeling":
            self.model = Seq2SeqTransformer(TransformerConfig(**model_config))
        else:
            self.model = Transformer(TransformerConfig(**model_config))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _preprocess_batch(self, batch: Any) -> Batch:
        """Normalize already-collated batches into a typed Batch on this module's device."""
        if isinstance(batch, Batch):
            typed = batch
        elif isinstance(batch, Mapping):
            kind: BatchKind = (
                "language_modeling" if self.cfg.task == "language_modeling" else "tokens"
            )
            if "input" in batch:
                kind = "features"
            typed = Batch.from_mapping(batch, kind=kind)
        else:
            raise TypeError(
                "MiniFormerModule expects a Batch or tensor batch dictionary. "
                "Use MiniFormerDataModule or miniformer.data.preprocessing to collate data."
            )

        typed = typed.to(self.device)
        if typed.attention_mask is None and self.cfg.task != "language_modeling":
            attention_mask = torch.ones(
                typed.inputs.size(0),
                typed.inputs.size(1),
                dtype=torch.bool,
                device=typed.inputs.device,
            )
            typed = typed.with_attention_mask(attention_mask)
        return typed

    @staticmethod
    def _to_attention_mask(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        return attention_mask.unsqueeze(1).unsqueeze(2)

    def forward_batch(self, batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        typed = self._preprocess_batch(batch)
        if self.cfg.task == "language_modeling":
            outputs = self.model(typed.inputs, typed.inputs, use_causal_mask=True).output
        else:
            sequence_outputs = self.model(
                typed.inputs,
                mask=self._to_attention_mask(typed.attention_mask),
            ).output
            outputs = pool_sequence_outputs(
                sequence_outputs,
                typed.attention_mask,
                mode=self.pooling,
            )
        return outputs, typed.labels

    @property
    def pooling(self) -> PoolingMode:
        pooling = getattr(self.cfg, "pooling", "masked_mean")
        if pooling not in {"first", "mean", "masked_mean"}:
            raise ValueError(f"Unknown pooling mode: {pooling}")
        return cast(PoolingMode, pooling)

    def configure_optimizers(self, steps_per_epoch: Optional[int] = None):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        if self.cfg.scheduler == "none":
            return optimizer, None
        if self.cfg.scheduler == "linear":
            scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=max(self.cfg.warmup_steps, 1),
            )
        elif self.cfg.scheduler == "onecycle":
            total_steps = max(1, self.cfg.max_epochs * max(steps_per_epoch or 100, 1))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.cfg.lr,
                total_steps=total_steps,
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=max(self.cfg.warmup_steps, 1),
            )
        return optimizer, scheduler

    def _compute_loss(self, batch_or_labels, outputs: torch.Tensor):
        if self.cfg.task == "language_modeling":
            labels = (
                batch_or_labels[-1]
                if isinstance(batch_or_labels, tuple)
                else batch_or_labels.get("labels")
            )
            if labels is None:
                loss = torch.tensor(0.0, device=outputs.device)
            else:
                labels = labels.to(outputs.device)
                loss = F.cross_entropy(
                    outputs.reshape(-1, outputs.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
            return loss, outputs

        if self.cfg.task == "classification":
            logits = outputs[:, 0, :] if outputs.dim() == 3 else outputs
            labels = self._extract_supervised_labels(batch_or_labels, logits.device)
            loss = F.cross_entropy(logits, labels)
            return loss, logits

        preds = outputs.squeeze(-1)
        if preds.dim() == 2:
            preds = preds[:, 0]
        labels = self._extract_supervised_labels(batch_or_labels, preds.device)
        loss = F.mse_loss(preds, labels)
        return loss, preds

    @staticmethod
    def _extract_supervised_labels(batch_or_labels: Any, device: torch.device) -> torch.Tensor:
        if isinstance(batch_or_labels, tuple) and len(batch_or_labels) == 1:
            labels = batch_or_labels[0]
        elif isinstance(batch_or_labels, dict):
            labels = batch_or_labels.get("labels")
        else:
            labels = None

        if not isinstance(labels, torch.Tensor):
            raise TypeError("Supervised losses require tensor labels")
        return labels.to(device)

    def training_step(self, batch, batch_idx: int = 0) -> torch.Tensor:
        outputs, labels = self.forward_batch(batch)
        loss, _ = self._compute_loss((labels,), outputs)
        return loss

    def validation_step(self, batch, batch_idx: int = 0) -> Dict[str, float]:
        outputs, labels = self.forward_batch(batch)
        if labels is None:
            raise ValueError("Validation batches must include labels")
        loss, predictions = self._compute_loss((labels,), outputs)
        metrics = {"val_loss": float(loss.detach().cpu())}

        if self.cfg.task == "language_modeling":
            metrics["val_ppl"] = (
                float(torch.exp(loss.detach()).cpu()) if loss.isfinite() else math.inf
            )
        elif self.cfg.task == "classification":
            preds = torch.argmax(predictions, dim=-1)
            metrics["val_accuracy"] = float((preds == labels).float().mean().detach().cpu())
        elif self.cfg.task == "regression":
            metrics["val_mae"] = float(torch.mean(torch.abs(predictions - labels)).detach().cpu())

        return metrics

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        checkpoint_metrics = metrics or {}
        model_config = getattr(self.model, "config").to_dict()
        checkpoint = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "cfg": asdict(self.cfg),
            "train_config": checkpoint_train_config(self.cfg),
            "model_config": model_config,
            "metadata": checkpoint_metadata(
                self.cfg,
                epoch=epoch,
                metrics=checkpoint_metrics,
            ),
            "state_dict": self.state_dict(),
            "epoch": epoch,
            "metrics": checkpoint_metrics,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(
        cls, path: str | Path, cfg: Optional[TrainConfig] = None
    ) -> "MiniFormerModule":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if cfg is None:
            cfg_payload = checkpoint.get("train_config") or checkpoint["cfg"]
            cfg = TrainConfig(**cfg_payload)
        else:
            validate_checkpoint_compatibility(checkpoint, cfg)
        module = cls(cfg)
        module.load_state_dict(checkpoint["state_dict"])
        return module
