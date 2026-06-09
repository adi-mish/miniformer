from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from miniformer import __version__
from miniformer.train.train_config import TrainConfig


@dataclass(frozen=True)
class RunPaths:
    root: Path
    checkpoints: Path
    traces: Path
    config: Path
    metrics: Path
    manifest: Path


def create_run_paths(cfg: TrainConfig) -> RunPaths:
    root = Path(cfg.work_dir) / cfg.experiment_name
    paths = RunPaths(
        root=root,
        checkpoints=root / "checkpoints",
        traces=root / "traces",
        config=root / "config.json",
        metrics=root / "metrics.csv",
        manifest=root / "run_manifest.json",
    )
    paths.checkpoints.mkdir(parents=True, exist_ok=True)
    paths.traces.mkdir(parents=True, exist_ok=True)
    return paths


def write_train_config(cfg: TrainConfig, path: str | Path) -> None:
    _write_json(Path(path), asdict(cfg))


def write_run_manifest(
    path: str | Path,
    *,
    cfg: TrainConfig,
    status: str,
    device: torch.device,
    latest_metrics: Mapping[str, float] | None = None,
    best_metrics: Mapping[str, float] | None = None,
    epoch: int | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "git_sha": _git_sha(),
        "device": str(device),
        "epoch": epoch,
        "train_config": asdict(cfg),
        "latest_metrics": dict(latest_metrics or {}),
        "best_metrics": dict(best_metrics or {}),
    }
    _write_json(Path(path), payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
