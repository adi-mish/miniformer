from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from miniformer.config.model_config import TransformerConfig
from miniformer.model.seq2seq_transformer import Seq2SeqTransformer
from miniformer.model.transformer import Transformer
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

    def _preprocess_batch(
        self, batch: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Move already-collated tensor batches to this module's device."""
        if not isinstance(batch, dict):
            raise TypeError(
                "MiniFormerModule expects a tensor batch dictionary. "
                "Use MiniFormerDataModule or miniformer.data.preprocessing to collate data."
            )

        if isinstance(batch.get("input_ids"), torch.Tensor):
            inputs = batch["input_ids"].to(self.device)
        elif isinstance(batch.get("input"), torch.Tensor):
            inputs = batch["input"].to(self.device)
        else:
            raise TypeError(
                "MiniFormerModule expects batch['input_ids'] or batch['input'] to be a tensor. "
                "Raw text and records belong in miniformer.data.preprocessing."
            )

        labels = batch.get("labels")
        if labels is not None and not isinstance(labels, torch.Tensor):
            raise TypeError("MiniFormerModule expects batch['labels'] to be a tensor when present")

        attention_mask = batch.get("attention_mask")
        if attention_mask is None and self.cfg.task != "language_modeling":
            attention_mask = torch.ones(
                inputs.size(0),
                inputs.size(1),
                dtype=torch.bool,
                device=inputs.device,
            )
        elif attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                raise TypeError("MiniFormerModule expects batch['attention_mask'] to be a tensor")
            if attention_mask.shape != inputs.shape[:2]:
                raise ValueError("attention_mask must match input batch and sequence dimensions")
            attention_mask = attention_mask.to(self.device, dtype=torch.bool)

        return inputs, labels.to(self.device) if labels is not None else None, attention_mask

    @staticmethod
    def _to_attention_mask(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        return attention_mask.unsqueeze(1).unsqueeze(2)

    def forward_batch(self, batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x, y, attention_mask = self._preprocess_batch(batch)
        if self.cfg.task == "language_modeling":
            outputs = self.model(x, x, use_causal_mask=True).output
        else:
            sequence_outputs = self.model(x, mask=self._to_attention_mask(attention_mask)).output
            outputs = pool_sequence_outputs(
                sequence_outputs,
                attention_mask,
                mode=self.pooling,
            )
        return outputs, y

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
        checkpoint = {
            "cfg": asdict(self.cfg),
            "state_dict": self.state_dict(),
            "epoch": epoch,
            "metrics": metrics or {},
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
            cfg = TrainConfig(**checkpoint["cfg"])
        module = cls(cfg)
        module.load_state_dict(checkpoint["state_dict"])
        return module
