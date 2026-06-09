from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from miniformer.config.model_config import TransformerConfig
from miniformer.model.seq2seq_transformer import Seq2SeqTransformer
from miniformer.model.transformer import Transformer
from miniformer.train.train_config import TrainConfig
from miniformer.utils.tokenization import stable_token_id


class MiniFormerModule(nn.Module):
    """Small plain-PyTorch wrapper around MiniFormer models."""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = None
        self.pad_id = 0

        if cfg.task == "language_modeling" and cfg.model != "seq2seq":
            raise ValueError("language_modeling currently requires model='seq2seq'")
        if cfg.task != "language_modeling" and cfg.model != "encoder":
            raise ValueError(f"{cfg.task} currently requires model='encoder'")

        model_config = dict(cfg.model_config)
        if cfg.task == "language_modeling" and model_config.get("output_dim") is None:
            model_config["output_dim"] = model_config.get("vocab_size")
        if cfg.task != "language_modeling" and model_config.get("output_dim") is None:
            raise ValueError(f"{cfg.task} requires model_config['output_dim']")
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

    def _preprocess_batch(self, batch):
        """Convert supported dataset batches into model inputs and labels."""
        if (
            isinstance(batch, list)
            and batch
            and isinstance(batch[0], dict)
            and "input" in batch[0]
            and isinstance(batch[0]["input"], str)
        ):
            vocab_size = self.cfg.model_config["vocab_size"]
            tokenized = [
                torch.tensor([ord(char) % vocab_size for char in sample["input"]], dtype=torch.long)
                for sample in batch
            ]
            if any(ids.numel() == 0 for ids in tokenized):
                raise ValueError("String inputs must not be empty")
            max_len = max(ids.size(0) for ids in tokenized)
            input_ids = torch.zeros(len(batch), max_len, dtype=torch.long, device=self.device)
            for i, ids in enumerate(tokenized):
                input_ids[i, : ids.size(0)] = ids.to(self.device)

            label_dtype = torch.long if self.cfg.task == "classification" else torch.float
            labels = torch.tensor(
                [sample["labels"] for sample in batch], dtype=label_dtype, device=self.device
            )
            return input_ids, labels

        if (
            isinstance(batch, dict)
            and "input" in batch
            and isinstance(batch["input"], torch.Tensor)
        ):
            return batch["input"].to(self.device), batch["labels"].to(self.device)

        if isinstance(batch, dict) and "input_ids" in batch:
            return batch["input_ids"].to(self.device), batch["labels"].to(self.device)

        if (
            isinstance(batch, list)
            and batch
            and isinstance(batch[0], dict)
            and "input" not in batch[0]
        ):
            labels = None
            if "labels" in batch[0]:
                dtype = torch.long if self.cfg.task == "classification" else torch.float
                labels = torch.tensor(
                    [item["labels"] for item in batch], dtype=dtype, device=self.device
                )
            return batch, labels

        if isinstance(batch, list) and batch and isinstance(batch[0], dict) and "input" in batch[0]:
            texts = [item["input"] for item in batch]
            labels = (
                torch.stack([item["labels"] for item in batch]).to(self.device)
                if "labels" in batch[0]
                else None
            )
        else:
            return batch, None

        if self.tokenizer is not None:
            enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            input_ids = enc["input_ids"].to(self.device)
        else:
            vocab_size = self.cfg.model_config.get("vocab_size", 30522)
            max_len = max(len(str(text).split()) for text in texts)
            input_ids = torch.zeros(len(texts), max_len, dtype=torch.long, device=self.device)
            for i, text in enumerate(texts):
                ids = torch.tensor(
                    [stable_token_id(word, vocab_size) for word in str(text).split()],
                    dtype=torch.long,
                    device=self.device,
                )
                input_ids[i, : ids.size(0)] = ids

        return input_ids, labels

    def forward_batch(self, batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x, y = self._preprocess_batch(batch)
        if self.cfg.task == "language_modeling":
            outputs = tuple(self.model(x, x, use_causal_mask=True))[0]
        else:
            outputs = self.model(x)
        return outputs, y

    def configure_optimizers(self, steps_per_epoch: Optional[int] = None):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        if self.cfg.scheduler == "none":
            return optimizer, None
        if self.cfg.scheduler == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
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
            if isinstance(batch_or_labels, tuple) and len(batch_or_labels) == 1:
                labels = batch_or_labels[0].to(logits.device)
            else:
                labels = torch.tensor([b["labels"] for b in batch_or_labels], device=logits.device)
            loss = F.cross_entropy(logits, labels)
            return loss, logits

        preds = outputs.squeeze(-1)
        if preds.dim() == 2:
            preds = preds[:, 0]
        if isinstance(batch_or_labels, tuple) and len(batch_or_labels) == 1:
            labels = batch_or_labels[0].to(preds.device)
        else:
            labels = torch.tensor([b["labels"] for b in batch_or_labels], device=preds.device)
        loss = F.mse_loss(preds, labels)
        return loss, preds

    def training_step(self, batch, batch_idx: int = 0) -> torch.Tensor:
        outputs, labels = self.forward_batch(batch)
        loss, _ = self._compute_loss((labels,), outputs)
        return loss

    def validation_step(self, batch, batch_idx: int = 0) -> Dict[str, float]:
        outputs, labels = self.forward_batch(batch)
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
