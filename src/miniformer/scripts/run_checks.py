from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

Check = tuple[str, tuple[str, ...]]

CHECKS: tuple[Check, ...] = (
    ("tests", ("uv", "run", "pytest", "-q")),
    ("black", ("uv", "run", "black", "--check", "src", "tests", "examples")),
    ("isort", ("uv", "run", "isort", "--check-only", "src", "tests", "examples")),
    ("flake8", ("uv", "run", "flake8", "src", "tests", "examples")),
    ("mypy", ("uv", "run", "mypy", "src/miniformer")),
    ("lock", ("uv", "lock", "--check")),
    (
        "compile",
        ("uv", "run", "python", "-m", "compileall", "-q", "src", "tests", "examples"),
    ),
    ("build", ("uv", "build")),
)


def _default_root() -> Path:
    source_root = Path(__file__).resolve().parents[3]
    if (source_root / "pyproject.toml").exists():
        return source_root
    return Path.cwd()


def run_checks(
    *,
    root: str | Path | None = None,
    selected: set[str] | None = None,
    skip_build: bool = False,
) -> int:
    """Run repository verification commands in order and stop on first failure."""
    repo_root = Path(root) if root is not None else _default_root()
    allowed = selected or {name for name, _ in CHECKS}
    if skip_build:
        allowed.discard("build")

    unknown = allowed - {name for name, _ in CHECKS}
    if unknown:
        raise ValueError(f"Unknown check names: {', '.join(sorted(unknown))}")

    for name, command in CHECKS:
        if name not in allowed:
            continue
        print(f"$ {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=repo_root, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Miniformer verification checks")
    parser.add_argument("--root", type=Path, default=None, help="Repository root")
    parser.add_argument(
        "--check",
        action="append",
        choices=[name for name, _ in CHECKS],
        help="Run one named check. Can be passed more than once.",
    )
    parser.add_argument("--skip-build", action="store_true", help="Skip uv build")
    parser.add_argument("--list", action="store_true", help="List available checks and exit")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list:
        for name, command in CHECKS:
            print(f"{name}: {' '.join(command)}")
        return 0
    try:
        return run_checks(
            root=args.root,
            selected=set(args.check) if args.check else None,
            skip_build=args.skip_build,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
