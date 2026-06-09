from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from miniformer.data.preprocessing import TextTokenizer

ValidationLevel = Literal["error", "warning"]
TaskName = Literal["classification", "regression", "language_modeling"]


@dataclass(frozen=True)
class ValidationIssue:
    level: ValidationLevel
    line_number: int
    message: str


@dataclass(frozen=True)
class JsonlValidationReport:
    path: str
    task: str
    records: int
    max_sequence_length: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def raise_for_errors(self) -> None:
        if self.ok:
            return
        first = self.errors[0]
        raise ValueError(
            f"{self.path} failed {self.task} validation at line "
            f"{first.line_number}: {first.message}"
        )


def validate_jsonl(
    path: str | Path,
    *,
    task: TaskName,
    tokenizer: TextTokenizer | None = None,
    max_seq_len: int | None = None,
    require_tokenizer: bool = False,
) -> JsonlValidationReport:
    """Validate a small JSONL dataset before training starts."""
    dataset_path = Path(path)
    issues: list[ValidationIssue] = []
    class_counts: Counter[str] = Counter()
    records = 0
    max_observed_len = 0

    if require_tokenizer and tokenizer is None:
        issues.append(
            ValidationIssue(
                "error",
                0,
                "tokenizer is required for this validation mode",
            )
        )

    if task not in {"classification", "regression", "language_modeling"}:
        raise ValueError(f"Unsupported validation task: {task}")
    if not dataset_path.exists():
        issues.append(ValidationIssue("error", 0, "file does not exist"))
        return JsonlValidationReport(str(dataset_path), task, records, issues=issues)

    for line_number, line in enumerate(dataset_path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        records += 1
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            issues.append(ValidationIssue("error", line_number, f"invalid JSON: {exc.msg}"))
            continue
        if not isinstance(record, Mapping):
            issues.append(ValidationIssue("error", line_number, "record must be a JSON object"))
            continue

        length = _validate_record(
            record,
            task=task,
            line_number=line_number,
            issues=issues,
            class_counts=class_counts,
            tokenizer=tokenizer,
        )
        max_observed_len = max(max_observed_len, length)
        if max_seq_len is not None and length > max_seq_len:
            issues.append(
                ValidationIssue(
                    "warning",
                    line_number,
                    f"sequence length {length} exceeds max_seq_len {max_seq_len}",
                )
            )

    if records == 0:
        issues.append(ValidationIssue("error", 0, "dataset contains no records"))

    return JsonlValidationReport(
        path=str(dataset_path),
        task=task,
        records=records,
        max_sequence_length=max_observed_len,
        class_counts=dict(sorted(class_counts.items())),
        issues=issues,
    )


def _validate_record(
    record: Mapping[str, Any],
    *,
    task: TaskName,
    line_number: int,
    issues: list[ValidationIssue],
    class_counts: Counter[str],
    tokenizer: TextTokenizer | None,
) -> int:
    if task == "language_modeling":
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            issues.append(
                ValidationIssue(
                    "error",
                    line_number,
                    "language_modeling records require a non-empty text string",
                )
            )
            return 0
        return _text_length(text, tokenizer)

    if "input" not in record:
        issues.append(ValidationIssue("error", line_number, f"{task} records require input"))
        return 0

    length = _input_length(
        record["input"], line_number=line_number, issues=issues, tokenizer=tokenizer
    )
    label = _label_value(record, task=task)
    if task == "classification":
        if not _is_int_label(label):
            issues.append(
                ValidationIssue(
                    "error",
                    line_number,
                    "classification labels must be scalar integers",
                )
            )
        else:
            class_counts[str(label)] += 1
    elif not _is_number(label):
        issues.append(
            ValidationIssue(
                "error",
                line_number,
                "regression labels must be scalar numbers",
            )
        )
    return length


def _input_length(
    value: Any,
    *,
    line_number: int,
    issues: list[ValidationIssue],
    tokenizer: TextTokenizer | None,
) -> int:
    if isinstance(value, str):
        if not value.strip():
            issues.append(ValidationIssue("error", line_number, "input string must not be empty"))
            return 0
        return _text_length(value, tokenizer)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        issues.append(
            ValidationIssue(
                "error",
                line_number,
                "input must be a non-empty string or numeric feature sequence",
            )
        )
        return 0
    if len(value) == 0:
        issues.append(ValidationIssue("error", line_number, "input sequence must not be empty"))
        return 0
    if any(not isinstance(step, Mapping) for step in value):
        issues.append(ValidationIssue("error", line_number, "feature steps must be objects"))
        return len(value)
    keys = set(value[0].keys())
    if not keys:
        issues.append(ValidationIssue("error", line_number, "feature steps must contain features"))
    for step in value:
        if set(step.keys()) != keys:
            issues.append(
                ValidationIssue(
                    "error",
                    line_number,
                    "feature steps must share the same keys",
                )
            )
            break
        if any(not _is_number(feature_value) for feature_value in step.values()):
            issues.append(
                ValidationIssue(
                    "error",
                    line_number,
                    "feature values must be numbers",
                )
            )
            break
    return len(value)


def _text_length(text: str, tokenizer: TextTokenizer | None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=True))
    return len(text.split())


def _label_value(record: Mapping[str, Any], *, task: TaskName) -> Any:
    if "labels" in record:
        return record["labels"]
    if "label" in record:
        return record["label"]
    if task == "regression" and "value" in record:
        return record["value"]
    return None


def _is_int_label(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
