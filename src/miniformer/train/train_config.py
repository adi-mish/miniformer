from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any, Literal, get_args, get_origin, get_type_hints


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
    shuffle_train: bool = True

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
    logger: Literal["csv", "none"] = "csv"
    checkpoint_metric: str = "val_loss"
    early_stopping_patience: int = 3

    # --- model / task -------------------------------------------------------
    task: Literal["language_modeling", "classification", "regression"] = "language_modeling"
    model: Literal["seq2seq", "encoder"] = "seq2seq"
    pooling: Literal["first", "mean", "masked_mean"] = "masked_mean"
    model_config: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str):
        import json
        import pathlib

        pathlib.Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_cli(cls) -> "TrainConfig":
        import argparse
        import json

        def default_for(field_):
            if field_.default is not MISSING:
                return field_.default
            if field_.default_factory is not MISSING:  # type: ignore[attr-defined]
                return field_.default_factory()  # type: ignore[misc]
            return MISSING

        def parse_json_dict(value: str) -> dict[str, Any]:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise argparse.ArgumentTypeError(f"expected a JSON object: {exc}") from exc
            if not isinstance(parsed, dict):
                raise argparse.ArgumentTypeError("expected a JSON object")
            return parsed

        def literal_parser(choices):
            def parse(value: str):
                for choice in choices:
                    if value == str(choice):
                        return choice
                allowed = ", ".join(str(choice) for choice in choices)
                raise argparse.ArgumentTypeError(f"expected one of: {allowed}")

            return parse

        parser = argparse.ArgumentParser(description="MiniFormer trainer")
        type_hints = get_type_hints(cls)

        # Dynamically create args from dataclass fields
        for field_ in fields(cls):
            name = field_.name
            arg_type = type_hints[name]
            default = default_for(field_)
            kwargs: dict[str, Any] = {}

            if default is not MISSING:
                kwargs["default"] = default
            else:
                kwargs["required"] = True

            if arg_type is bool:
                parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, **kwargs)
            elif get_origin(arg_type) is Literal:
                choices = get_args(arg_type)
                parser.add_argument(
                    f"--{name}", type=literal_parser(choices), choices=choices, **kwargs
                )
            elif get_origin(arg_type) is dict or arg_type is dict or name == "model_config":
                parser.add_argument(f"--{name}", type=parse_json_dict, **kwargs)
            else:
                parser.add_argument(
                    f"--{name}", type=type(default) if default is not MISSING else str, **kwargs
                )

        parser.add_argument(
            "--config_json", type=str, help="Path to JSON config that overrides args", default=None
        )
        args = parser.parse_args()
        cfg_dict = vars(args)

        # load external json overrides
        if args.config_json:
            with open(args.config_json) as f:
                cfg_dict.update(json.load(f))

        # drop the helper key so __init__ only sees real fields
        cfg_dict.pop("config_json", None)
        if isinstance(cfg_dict.get("model_config"), str):
            cfg_dict["model_config"] = parse_json_dict(cfg_dict["model_config"])

        return cls(**cfg_dict)
