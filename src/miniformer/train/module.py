import math
from dataclasses import asdict
import torch
import torch.nn.functional as F
import lightning as L
import torchmetrics as tm

from miniformer.model.seq2seq_transformer import Seq2SeqTransformer
from miniformer.model.transformer import Transformer
from miniformer.config.model_config import TransformerConfig
from .train_config import TrainConfig

class MiniFormerLitModule(L.LightningModule):
    """Wraps MiniFormer models to provide Lightning hooks."""
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        # let Lightning capture whatever was passed in here,
        # whether it's a dataclass or a SimpleNamespace
        self.save_hyperparameters()
        self.cfg = cfg

        if cfg.model == "seq2seq":
            self.model = Seq2SeqTransformer(TransformerConfig(**cfg.model_config))
        else:
            self.model = Transformer(TransformerConfig(**cfg.model_config))

        if cfg.task == "language_modeling":
            self.val_ppl = tm.Perplexity(ignore_index=-100)
        elif cfg.task == "classification":
            self.train_acc = tm.Accuracy(task="multiclass", num_classes=self.model.config.output_dim)
            self.val_acc = tm.Accuracy(task="multiclass", num_classes=self.model.config.output_dim)
        elif cfg.task == "regression":
            self.val_mae = tm.MeanAbsoluteError()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        if self.cfg.scheduler == "none":
            return optimizer
        elif self.cfg.scheduler == "linear":
            sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=self.cfg.warmup_steps)
        elif self.cfg.scheduler == "onecycle":
            steps = self.cfg.max_epochs * math.ceil(self.trainer.estimated_stepping_batches / self.cfg.max_epochs)
            sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.lr, total_steps=steps)
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.cfg.warmup_steps)
        return [optimizer], [sched]

    def _compute_loss(self, batch, outputs):
        if self.cfg.task == "language_modeling":
            logits = outputs
            labels = batch["labels"].to(logits.device)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            return loss, logits
        elif self.cfg.task == "classification":
            logits = outputs
            if logits.dim() == 3:
                logits = logits[:, 0, :]
            labels = torch.tensor([b["labels"] for b in batch], device=logits.device)
            loss = F.cross_entropy(logits, labels)
            return loss, logits
        else:  # regression
            preds = outputs.squeeze(-1)
            labels = torch.tensor([b["labels"] for b in batch], device=preds.device)
            loss = F.mse_loss(preds, labels)
            return loss, preds

    def training_step(self, batch, batch_idx):
        if self.cfg.task == "language_modeling":
            outputs = self.model(batch["input_ids"], batch["input_ids"], use_causal_mask=True)[0]
        else:
            outputs = self.model(batch)
        loss, _ = self._compute_loss(batch, outputs)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        if self.cfg.task == "classification":
            self.train_acc(torch.argmax(outputs, dim=-1), torch.tensor([b["labels"] for b in batch], device=self.device))
            self.log("train_acc", self.train_acc, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.cfg.task == "language_modeling":
            outputs = self.model(batch["input_ids"], batch["input_ids"], use_causal_mask=True)[0]
        else:
            outputs = self.model(batch)
        loss, logits_or_preds = self._compute_loss(batch, outputs)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        if self.cfg.task == "language_modeling":
            self.val_ppl(logits_or_preds, batch["labels"].to(self.device))
            self.log("val_ppl", self.val_ppl, prog_bar=True, sync_dist=True)
        elif self.cfg.task == "classification":
            self.val_acc(torch.argmax(logits_or_preds, dim=-1), torch.tensor([b["labels"] for b in batch], device=self.device))
            self.log("val_acc", self.val_acc, prog_bar=True, sync_dist=True)
        elif self.cfg.task == "regression":
            self.val_mae(logits_or_preds, torch.tensor([b["labels"] for b in batch], device=self.device))
            self.log("val_mae", self.val_mae, prog_bar=True, sync_dist=True)
