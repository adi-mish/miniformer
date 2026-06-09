import json

import pytest

from miniformer.data.validation import validate_jsonl


class DummyTokenizer:
    def encode(self, text, add_special_tokens=True):
        values = [ord(char) % 100 for char in text]
        return [1, *values, 2] if add_special_tokens else values


def write_jsonl(tmp_path, rows):
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    return path


def test_validate_jsonl_classification_reports_class_counts(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            {"input": "alpha beta", "label": 1},
            {"input": "gamma", "labels": 0},
        ],
    )

    report = validate_jsonl(path, task="classification", max_seq_len=1)

    assert report.records == 2
    assert report.class_counts == {"0": 1, "1": 1}
    assert report.max_sequence_length == 2
    assert report.ok
    assert len(report.warnings) == 1


def test_validate_jsonl_rejects_empty_supervised_input(tmp_path):
    path = write_jsonl(tmp_path, [{"input": "", "label": 0}])

    report = validate_jsonl(path, task="classification")

    assert not report.ok
    assert "input string" in report.errors[0].message
    with pytest.raises(ValueError, match="validation"):
        report.raise_for_errors()


def test_validate_jsonl_regression_requires_numeric_label(tmp_path):
    path = write_jsonl(tmp_path, [{"input": "alpha", "value": "bad"}])

    report = validate_jsonl(path, task="regression")

    assert not report.ok
    assert "regression labels" in report.errors[0].message


def test_validate_jsonl_language_modeling_uses_tokenizer_lengths(tmp_path):
    path = write_jsonl(tmp_path, [{"text": "ab"}])

    report = validate_jsonl(
        path,
        task="language_modeling",
        tokenizer=DummyTokenizer(),
        max_seq_len=3,
    )

    assert report.ok
    assert report.max_sequence_length == 4
    assert len(report.warnings) == 1


def test_validate_jsonl_language_modeling_can_require_tokenizer(tmp_path):
    path = write_jsonl(tmp_path, [{"text": "alpha"}])

    report = validate_jsonl(path, task="language_modeling", require_tokenizer=True)

    assert not report.ok
    assert "tokenizer" in report.errors[0].message
