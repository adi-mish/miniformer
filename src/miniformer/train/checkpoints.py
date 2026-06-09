from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from miniformer import __version__
from miniformer.train.artifacts import get_git_sha
from miniformer.train.train_config import TrainConfig

CHECKPOINT_FORMAT_VERSION = 2


def checkpoint_metadata(
    cfg: TrainConfig,
    *,
    epoch: int,
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "package_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "epoch": epoch,
        "metrics": dict(metrics),
        "task": cfg.task,
        "model": cfg.model,
        "pooling": cfg.pooling,
    }


def validate_checkpoint_compatibility(
    checkpoint: Mapping[str, Any],
    cfg: TrainConfig,
) -> None:
    saved_cfg = checkpoint.get("train_config") or checkpoint.get("cfg")
    if not isinstance(saved_cfg, Mapping):
        return

    mismatches: list[str] = []
    for key in ["task", "model", "pooling"]:
        if key in saved_cfg and saved_cfg[key] != getattr(cfg, key):
            mismatches.append(
                f"{key}: checkpoint={saved_cfg[key]!r}, requested={getattr(cfg, key)!r}"
            )

    saved_model_config = saved_cfg.get("model_config")
    if isinstance(saved_model_config, Mapping):
        for key, value in sorted(saved_model_config.items()):
            if key in cfg.model_config and cfg.model_config[key] != value:
                mismatches.append(
                    f"model_config.{key}: checkpoint={value!r}, requested={cfg.model_config[key]!r}"
                )

    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"Checkpoint is not compatible with the requested config: {details}")


def checkpoint_summary(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", {})
    metadata = checkpoint.get("metadata", {})
    train_config = checkpoint.get("train_config") or checkpoint.get("cfg") or {}
    return {
        "path": str(checkpoint_path),
        "format_version": checkpoint.get("format_version", metadata.get("format_version")),
        "epoch": checkpoint.get("epoch", metadata.get("epoch")),
        "metrics": checkpoint.get("metrics", metadata.get("metrics", {})),
        "metadata": metadata,
        "task": train_config.get("task") if isinstance(train_config, Mapping) else None,
        "model": train_config.get("model") if isinstance(train_config, Mapping) else None,
        "pooling": train_config.get("pooling") if isinstance(train_config, Mapping) else None,
        "model_config": (
            train_config.get("model_config") if isinstance(train_config, Mapping) else None
        ),
        "optimizer_present": "optimizer_state_dict" in checkpoint,
        "state_tensor_count": len(state_dict) if isinstance(state_dict, Mapping) else 0,
    }


def checkpoint_summary_json(path: str | Path, *, indent: int | None = 2) -> str:
    return json.dumps(checkpoint_summary(path), indent=indent, sort_keys=True)


def checkpoint_train_config(cfg: TrainConfig) -> dict[str, Any]:
    return asdict(cfg)
