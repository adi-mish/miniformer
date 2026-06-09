from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

TaskName = Literal["classification", "regression", "language_modeling"]
TASKS: tuple[TaskName, ...] = ("classification", "regression", "language_modeling")


def _records(task: TaskName, split: str, rows: int) -> Iterable[dict[str, object]]:
    for index in range(rows):
        if task == "classification":
            sentiment = "great" if index % 2 == 0 else "rough"
            yield {
                "input": f"{split} sample {index} has a {sentiment} signal",
                "label": index % 2,
            }
        elif task == "regression":
            value = round((index * 0.5) + (0.25 if split == "val" else 0.0), 3)
            yield {
                "input": f"{split} feature row {index} value trend",
                "value": value,
            }
        else:
            yield {"text": f"{split} tiny transformer sequence {index} repeats context tokens"}


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def write_tiny_jsonl(
    output_dir: str | Path,
    *,
    task: TaskName,
    train_rows: int = 16,
    val_rows: int = 8,
) -> dict[str, Path]:
    """Write deterministic tiny train/validation JSONL files for one task."""
    if train_rows <= 0 or val_rows <= 0:
        raise ValueError("train_rows and val_rows must be positive")

    root = Path(output_dir)
    train_path = root / "train.jsonl"
    val_path = root / "val.jsonl"
    _write_jsonl(train_path, _records(task, "train", train_rows))
    _write_jsonl(val_path, _records(task, "val", val_rows))
    return {"train": train_path, "val": val_path}


def write_all_tiny_jsonl(
    output_dir: str | Path,
    *,
    train_rows: int = 16,
    val_rows: int = 8,
) -> dict[str, dict[str, Path]]:
    """Write deterministic tiny JSONL files for every supported task."""
    root = Path(output_dir)
    return {
        task: write_tiny_jsonl(
            root / task,
            task=task,
            train_rows=train_rows,
            val_rows=val_rows,
        )
        for task in TASKS
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create tiny Miniformer JSONL datasets")
    parser.add_argument("--output-dir", type=Path, default=Path("data/tiny"))
    parser.add_argument(
        "--task",
        choices=[*TASKS, "all"],
        default="classification",
        help="Task dataset to write",
    )
    parser.add_argument("--train-rows", type=int, default=16)
    parser.add_argument("--val-rows", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.task == "all":
        paths = write_all_tiny_jsonl(
            args.output_dir,
            train_rows=args.train_rows,
            val_rows=args.val_rows,
        )
        for task, task_paths in paths.items():
            print(f"{task}: train={task_paths['train']} val={task_paths['val']}")
        return 0

    dataset_paths = write_tiny_jsonl(
        args.output_dir,
        task=args.task,
        train_rows=args.train_rows,
        val_rows=args.val_rows,
    )
    print(f"train={dataset_paths['train']} val={dataset_paths['val']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
