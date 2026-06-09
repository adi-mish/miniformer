from __future__ import annotations

import ast
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class JSONLinesDataset(Dataset):
    """A minimal Dataset reading line-separated JSON records."""

    def __init__(self, path: str, tokenizer=None, task: str = "language_modeling"):
        super().__init__()
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
            ids = torch.tensor(
                self.tokenizer.encode(txt, add_special_tokens=True), dtype=torch.long
            )
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

    def __init__(self, cfg, tokenizer=None):
        self.cfg = cfg
        self.tokenizer = tokenizer

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
        """
        Language-modeling batches are padded tensors. String classification and
        regression batches stay as raw records so tokenization can happen in the
        training module. Numeric feature sequences are padded into tensors.
        """
        task = self.cfg.task

        # ------------------------------------------------------------------ 1. LM
        if task == "language_modeling":
            lengths = [b["input_ids"].size(0) for b in batch]
            max_len = max(lengths)
            input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
            labels = torch.full_like(input_ids, -100)
            for i, b in enumerate(batch):
                seq_len = lengths[i]
                input_ids[i, :seq_len] = b["input_ids"]
                labels[i, :seq_len] = b["labels"]
            return {"input_ids": input_ids, "labels": labels}

        # ---------------------------------------------------- 2. string inputs → return list
        if task in {"classification", "regression"} and isinstance(batch[0]["input"], str):
            return batch

        # ------------------------------------------------ 3. numeric sequence features
        if task in {"classification", "regression"}:
            seq_lens = [len(s["input"]) for s in batch]
            if min(seq_lens) == 0:
                raise ValueError("Numeric feature sequences must not be empty")
            max_len = max(seq_lens)
            feat_keys = list(batch[0]["input"][0].keys())
            feat_dim = len(feat_keys)

            x = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
            label_dtype = torch.long if task == "classification" else torch.float32
            y = torch.as_tensor([b["labels"] for b in batch], dtype=label_dtype)
            for i, sample in enumerate(batch):
                seq = torch.tensor(
                    [[step[k] for k in feat_keys] for step in sample["input"]],
                    dtype=torch.float32,
                )
                x[i, : seq.size(0)] = seq
            return {"input": x, "labels": y}

        raise ValueError(f"Unsupported task for collation: {task}")
