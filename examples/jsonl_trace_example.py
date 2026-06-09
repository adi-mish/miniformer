from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from miniformer.scripts.make_tiny_jsonl import write_tiny_jsonl
from miniformer.train.datamodule import MiniFormerDataModule
from miniformer.train.module import MiniFormerModule
from miniformer.train.train_config import TrainConfig
from miniformer.train.trainer import train_model


def run_example(
    output_dir: str | Path,
    *,
    max_epochs: int = 1,
    train_rows: int = 8,
    val_rows: int = 4,
) -> dict[str, Path]:
    """Train a tiny JSONL classification model and write a trace HTML report."""
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")

    root = Path(output_dir)
    data_paths = write_tiny_jsonl(
        root / "data",
        task="classification",
        train_rows=train_rows,
        val_rows=val_rows,
    )
    cfg = TrainConfig(
        task="classification",
        model="encoder",
        train_path=str(data_paths["train"]),
        val_path=str(data_paths["val"]),
        batch_size=4,
        num_workers=0,
        shuffle_train=False,
        max_epochs=max_epochs,
        lr=0.01,
        scheduler="none",
        logger="none",
        gpus=0,
        early_stopping_patience=0,
        work_dir=str(root),
        experiment_name="jsonl-trace",
        model_config={
            "vocab_size": 128,
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 1,
            "d_ff": 32,
            "dropout": 0.0,
            "causal": False,
            "output_mode": "projection",
            "output_dim": 2,
        },
    )

    module = MiniFormerModule(cfg)
    train_model(cfg, module=module)

    datamodule = MiniFormerDataModule(cfg)
    datamodule.setup()
    batch = next(iter(datamodule.val_dataloader()))
    input_ids = batch["input_ids"]
    assert isinstance(input_ids, torch.Tensor)

    module.eval()
    with torch.no_grad():
        trace = module.model.trace(input_ids, top_k=2, compare_cache=False)

    run_dir = root / cfg.experiment_name
    trace_html = run_dir / "traces" / "trace.html"
    trace_json = run_dir / "traces" / "trace.json"
    tokens = [str(token_id) for token_id in input_ids[0].tolist()]
    trace.to_html(trace_html, tokens=tokens)
    trace.save_json(trace_json)

    return {
        "train_jsonl": data_paths["train"],
        "val_jsonl": data_paths["val"],
        "trace_html": trace_html,
        "trace_json": trace_json,
        "metrics_csv": run_dir / "metrics.csv",
        "manifest": run_dir / "run_manifest.json",
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/jsonl-trace-example"))
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--train-rows", type=int, default=8)
    parser.add_argument("--val-rows", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = run_example(
        args.output_dir,
        max_epochs=args.max_epochs,
        train_rows=args.train_rows,
        val_rows=args.val_rows,
    )
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
