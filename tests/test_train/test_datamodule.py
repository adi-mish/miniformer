import json
from types import SimpleNamespace

import pytest
import torch

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
    records = [{"text": "ab"}, {"text": "cd"}]
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


def test_jsonlines_dataset_lm_rejects_too_short_text(tmp_path):
    path = create_jsonlines_file(tmp_path, [{"text": "a"}])
    ds = JSONLinesDataset(path, tokenizer=DummyTokenizer(), task="language_modeling")

    with pytest.raises(ValueError, match="at least two tokens"):
        ds[0]


def test_jsonlines_dataset_rejects_invalid_records(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("not json\n")

    with pytest.raises(ValueError, match="Invalid JSONL record"):
        JSONLinesDataset(str(path), tokenizer=None, task="classification")


def test_jsonlines_dataset_rejects_non_object_records(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("[1, 2, 3]\n")

    with pytest.raises(ValueError, match="must be an object"):
        JSONLinesDataset(str(path), tokenizer=None, task="classification")


def test_jsonlines_dataset_classification_and_regression(tmp_path):
    records = [{"input": "foo", "label": 1}, {"input": "bar", "value": 2.5}]
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
    records = [{"text": "aaa"}, {"text": "bb"}]
    path = create_jsonlines_file(tmp_path, records)
    cfg = SimpleNamespace(
        train_path=path,
        val_path="",
        test_path="",
        batch_size=2,
        num_workers=0,
        task="language_modeling",
        shuffle_train=False,
    )
    dm = MiniFormerDataModule(cfg, tokenizer=DummyTokenizer())
    dm.setup()
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    # batch should be a dict with padded tensors
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape == batch["input_ids"].shape
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    # second sample is length 1, so padded positions should be -100
    assert (batch["labels"][1, 1:] == -100).all()


def test_datamodule_classification(tmp_path):
    records = [{"input": "x", "label": 0}, {"input": "y", "label": 1}]
    path = create_jsonlines_file(tmp_path, records)
    cfg = SimpleNamespace(
        train_path=path,
        val_path="",
        test_path="",
        batch_size=2,
        num_workers=0,
        task="classification",
        shuffle_train=False,
        model_config={"vocab_size": 20},
    )
    dm = MiniFormerDataModule(cfg, tokenizer=None)
    dm.setup()
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape == (2, 1)
    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].tolist() == [[True], [True]]
    assert batch["labels"].tolist() == [0, 1]


def test_datamodule_numeric_features_collate_dtype(tmp_path):
    records = [
        {"input": [{"a": 1.0, "b": 2.0}], "label": 1},
        {"input": [{"a": 3.0, "b": 4.0}, {"a": 5.0, "b": 6.0}], "label": 0},
    ]
    path = create_jsonlines_file(tmp_path, records)
    cfg = SimpleNamespace(
        train_path=path,
        val_path="",
        test_path="",
        batch_size=2,
        num_workers=0,
        task="classification",
        shuffle_train=False,
        model_config={"vocab_size": 20},
    )
    dm = MiniFormerDataModule(cfg, tokenizer=None)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))

    assert batch["input"].shape == (2, 2, 2)
    assert batch["input"].dtype == torch.float32
    assert batch["attention_mask"].tolist() == [[True, False], [True, True]]
    assert batch["labels"].dtype == torch.long


def test_datamodule_rejects_unknown_collation_task():
    cfg = SimpleNamespace(task="unsupported")
    dm = MiniFormerDataModule(cfg, tokenizer=None)

    with pytest.raises(ValueError, match="Unsupported task"):
        dm._collate_fn([{"input": "x", "labels": 0}])
