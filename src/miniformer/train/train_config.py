from __future__ import annotations

import os
import math
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, Literal

@dataclass
class TrainConfig:
    # --- experiment ---------------------------------------------------------
    experiment_name: str = "miniformer-run"
    work_dir: str = "./runs"
    seed: int = 42
    deterministic: bool = False

    # --- data ---------------------------------------------------------------
    train_path: str = ""
    val_path: str = ""
    test_path: str = ""
    batch_size: int = 32
    num_workers: int = 4

    # --- optimisation -------------------------------------------------------
    lr: float = 5e-4
    weight_decay: float = 0.01
    scheduler: Literal["cosine", "onecycle", "linear", "none"] = "cosine"
    warmup_steps: int = 500
    max_epochs: int = 10
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1

    # --- hardware -----------------------------------------------------------
    gpus: int = 1  # 0 = CPU
    precision: Literal[16, 32, 64, "bf16"] = "bf16"

    # --- logging / callbacks -----------------------------------------------
    logger: Literal["tensorboard", "wandb", "csv", "none"] = "tensorboard"
    checkpoint_metric: str = "val_loss"
    early_stopping_patience: int = 3

    # --- model / task -------------------------------------------------------
    task: Literal["language_modeling", "classification", "regression"] = "language_modeling"
    model: Literal["seq2seq", "encoder"] = "seq2seq"
    model_config: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str):
        import json, pathlib
        pathlib.Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_cli(cls) -> "TrainConfig":
        import argparse, json
        parser = argparse.ArgumentParser(description="MiniFormer trainer")
        # Dynamically create args from dataclass fields
        for field_ in cls.__dataclass_fields__.values():  # type: ignore
            name = field_.name
            arg_type = field_.type
            default = field_.default if field_.default is not field_.default_factory else None  # type: ignore
            if arg_type is bool:
                parser.add_argument(f"--{name}", action="store_true" if default is False else "store_false")
            else:
                parser.add_argument(f"--{name}", type=type(default) if default is not None else str, default=default)
        parser.add_argument("--config_json", type=str, help="Path to JSON config that overrides args", default=None)
        args = parser.parse_args()
        cfg_dict = vars(args)
        # load external json overrides
        if args.config_json:
            cfg_dict.update(json.loads(open(args.config_json).read()))
        return cls(**cfg_dict)
