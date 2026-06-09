from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from miniformer.data.validation import validate_jsonl


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Miniformer JSONL dataset")
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--task",
        required=True,
        choices=["classification", "regression", "language_modeling"],
    )
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = validate_jsonl(
        args.path,
        task=args.task,
        max_seq_len=args.max_seq_len,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        status = "ok" if report.ok else "failed"
        print(
            f"{status}: records={report.records} "
            f"errors={len(report.errors)} warnings={len(report.warnings)} "
            f"max_sequence_length={report.max_sequence_length}"
        )
        for issue in report.issues:
            print(f"{issue.level}: line {issue.line_number}: {issue.message}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
