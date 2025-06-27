import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / 'src'))

import json
import torch
import pytest
from types import SimpleNamespace
from miniformer.train.datamodule import JSONLinesDataset, MiniFormerDataModule


class DummyTokenizer:
    def encode(self, text, add_special_tokens=True):
        # simple char-to-int mapping
        return [ord(c) for c in text]


def create_jsonlines_file(tmp_path, records):
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return str(path)


def test_jsonlines_dataset_lm(tmp_path):
    records = [{"text": "ab"}, {"text": "c"}]
    path = create_jsonlines_file(tmp_path, records)
    ds = JSONLinesDataset(path, tokenizer=DummyTokenizer(), task="language_modeling")
    assert len(ds) == 2
    item0 = ds[0]
    # "ab" -> [97,98] => input_ids=[97], labels=[98]
    assert torch.equal(item0["input_ids"], torch.tensor([97], dtype=torch.long))
    assert torch.equal(item0["labels"], torch.tensor([98], dtype=torch.long))

    # missing tokenizer should error
    with pytest.raises(ValueError):
        JSONLinesDataset(path, tokenizer=None, task="language_modeling")[0]


def test_jsonlines_dataset_classification_and_regression(tmp_path):
    records = [
        {"input": "foo", "label": 1},
        {"input": "bar", "value": 2.5}
    ]
    path = create_jsonlines_file(tmp_path, records)
    ds_clf = JSONLinesDataset(path, tokenizer=None, task="classification")
    item = ds_clf[0]
    assert item["input"] == "foo"
    assert item["labels"].item() == 1

    ds_reg = JSONLinesDataset(path, tokenizer=None, task="regression")
    item = ds_reg[1]
    assert item["input"] == "bar"
    assert item["labels"].item() == 2.5


def test_datamodule_lm(tmp_path):
    records = [{"text": "aaa"}, {"text": "b"}]
    path = create_jsonlines_file(tmp_path, records)
    cfg = SimpleNamespace(
        train_path=path, val_path="", test_path="",
        batch_size=2, num_workers=0, task="language_modeling"
    )
    dm = MiniFormerDataModule(cfg, tokenizer=DummyTokenizer())
    dm.setup()
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    # batch should be a dict with padded tensors
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape == batch["input_ids"].shape
    # second sample is length 1, so padded positions should be -100
    assert (batch["labels"][1,1:] == -100).all()


def test_datamodule_classification(tmp_path):
    records = [{"input": "x", "label": 0}, {"input": "y", "label": 1}]
    path = create_jsonlines_file(tmp_path, records)
    cfg = SimpleNamespace(
        train_path=path, val_path="", test_path="",
        batch_size=2, num_workers=0, task="classification"
    )
    dm = MiniFormerDataModule(cfg, tokenizer=None)
    dm.setup()
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    # non-LM batches returned as list of dicts
    assert isinstance(batch, list)
    assert batch[0]["input"] == "x"
