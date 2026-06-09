from __future__ import annotations

import ast
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from miniformer.data.preprocessing import collate_records, encode_text
from miniformer.data.tokenizers import TokenizerProtocol, ensure_tokenizer


class JSONLinesDataset(Dataset):
    """A minimal Dataset reading line-separated JSON records."""

    def __init__(
        self,
        path: str,
        tokenizer: TokenizerProtocol | None = None,
        task: str = "language_modeling",
    ):
        super().__init__()
        if task not in {"language_modeling", "classification", "regression"}:
            raise ValueError(f"Unsupported dataset task: {task}")
        self.data = []
        for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                try:
                    record = ast.literal_eval(line)
                except (SyntaxError, ValueError) as literal_error:
                    raise ValueError(
                        f"Invalid JSONL record at line {line_number}"
                    ) from literal_error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record at line {line_number} must be an object")
            self.data.append(record)
        self.tokenizer = tokenizer
        self.task = task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.task == "language_modeling":
            txt = item["text"]
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for LM task")
            ids = encode_text(txt, tokenizer=self.tokenizer)
            if ids.numel() < 2:
                raise ValueError("Language modeling records must produce at least two tokens")
            return {"input_ids": ids[:-1], "labels": ids[1:]}

        elif self.task == "classification":
            return {"input": item["input"], "labels": torch.tensor(item["label"], dtype=torch.long)}

        else:  # regression – accept either `value` or `labels`
            val = item.get("value", item.get("labels"))
            return {"input": item["input"], "labels": torch.tensor(val, dtype=torch.float)}


class MiniFormerDataModule:
    """Small data loader factory for JSONL datasets."""

    def __init__(self, cfg, tokenizer: TokenizerProtocol | None = None):
        self.cfg = cfg
        self.tokenizer = ensure_tokenizer(tokenizer, vocab_size=self._vocab_size())

    def _vocab_size(self) -> int:
        return int(getattr(self.cfg, "model_config", {}).get("vocab_size", 30522))

    def setup(self, stage: str | None = None):
        if self.cfg.train_path:
            self.train_data = JSONLinesDataset(self.cfg.train_path, self.tokenizer, self.cfg.task)
        if self.cfg.val_path:
            self.val_data = JSONLinesDataset(self.cfg.val_path, self.tokenizer, self.cfg.task)
        if self.cfg.test_path:
            self.test_data = JSONLinesDataset(self.cfg.test_path, self.tokenizer, self.cfg.task)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            shuffle=getattr(self.cfg, "shuffle_train", True),
            num_workers=self.cfg.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    def _collate_fn(self, batch):
        return collate_records(
            batch,
            task=self.cfg.task,
            vocab_size=self._vocab_size(),
            tokenizer=self.tokenizer,
        )
