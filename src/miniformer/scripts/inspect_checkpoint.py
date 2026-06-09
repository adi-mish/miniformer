from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from miniformer.train.checkpoints import checkpoint_summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a Miniformer checkpoint")
    parser.add_argument("path", type=Path)
    parser.add_argument("--json", action="store_true", help="Print full JSON summary")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = checkpoint_summary(args.path)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    print(f"path={summary['path']}")
    print(f"format_version={summary['format_version']}")
    print(f"epoch={summary['epoch']}")
    print(f"task={summary['task']} model={summary['model']} pooling={summary['pooling']}")
    print(f"optimizer_present={summary['optimizer_present']}")
    print(f"state_tensor_count={summary['state_tensor_count']}")
    print(f"metrics={summary['metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
