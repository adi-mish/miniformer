import os
import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L  # PyTorch Lightning ≥ 2.x

class JSONLinesDataset(Dataset):
    """A minimal Dataset reading line-separated JSON records."""
    def __init__(self, path: str, tokenizer=None, task: str = "language_modeling"):
        super().__init__()
        import json, pathlib, ast
        raw = pathlib.Path(path).read_text().splitlines()
        self.data = []
        for idx, line in enumerate(raw):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # fall back to Python literal eval for single-quoted dicts
                record = ast.literal_eval(line)
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
            ids = torch.tensor(self.tokenizer.encode(txt, add_special_tokens=True), dtype=torch.long)
            return {"input_ids": ids[:-1], "labels": ids[1:]}

        elif self.task == "classification":
            return {"input": item["input"], "labels": torch.tensor(item["label"], dtype=torch.long)}

        else:  # regression – accept either `value` or `labels`
            val = item.get("value", item.get("labels"))
            return {"input": item["input"], "labels": torch.tensor(val, dtype=torch.float)}

class MiniFormerDataModule(L.LightningDataModule):
    """Lightweight DataModule placeholder - replace with task-specific logic."""
    def __init__(self, cfg, tokenizer=None):
        super().__init__()
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
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def _collate_fn(self, batch):
        """
        Unit tests expect three different behaviours:

        • language_modeling  – return a dict with padded Long tensors
        • classification/regression when input is a *string*  – **return the raw list** unchanged
        • classification/regression when input is a list[dict] of numeric features –
        return padded float tensors (keep the code you already wrote).

        Anything else falls back to the raw list.
        """
        task = self.cfg.task

        # ------------------------------------------------------------------ 1. LM
        if task == "language_modeling":
            lengths  = [b["input_ids"].size(0) for b in batch]
            max_len  = max(lengths)
            input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
            labels    = torch.full_like(input_ids, -100)
            for i, b in enumerate(batch):
                l = lengths[i]
                input_ids[i, :l] = b["input_ids"]
                labels[i,   :l]  = b["labels"]
            return {"input_ids": input_ids, "labels": labels}

        # ---------------------------------------------------- 2. string inputs → return list
        if task in {"classification", "regression"} and isinstance(batch[0]["input"], str):
            return batch                                           # ← unit-test expects list

        # ------------------------------------------------ 3. numeric sequence features (your code)
        if task in {"classification", "regression"}:
            seq_lens  = [len(s["input"]) for s in batch]
            max_len   = max(seq_lens)
            feat_keys = list(batch[0]["input"][0].keys())
            feat_dim  = len(feat_keys)

            x = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
            y = torch.tensor([b["labels"] for b in batch])
            for i, sample in enumerate(batch):
                seq = torch.tensor([[step[k] for k in feat_keys] for step in sample["input"]])
                x[i, : seq.size(0)] = seq
            return {"input": x, "labels": y}

        # ---------------------------------------------------------- fallback (safety-net)
        return batch
