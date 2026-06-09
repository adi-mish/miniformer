from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from miniformer.config.model_config import TransformerConfig
from miniformer.inspect import capture_transformer_trace
from miniformer.model.transformer import Transformer


def write_trace_report(
    output_html: str | Path,
    *,
    output_json: str | Path | None = None,
    seed: int = 0,
    vocab_size: int = 64,
    seq_len: int = 8,
    d_model: int = 16,
    n_heads: int = 2,
    n_layers: int = 1,
    d_ff: int = 32,
    top_k: int = 3,
    compare_cache: bool = True,
) -> Path:
    """Write a deterministic static HTML report for a tiny causal transformer."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    torch.manual_seed(seed)
    model = Transformer(
        TransformerConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=0.0,
            output_mode="vocab",
            causal=True,
            use_sdpa=False,
        )
    ).eval()

    input_ids = (torch.arange(1, seq_len + 1, dtype=torch.long).unsqueeze(0) % vocab_size).clamp(
        min=1
    )
    trace = capture_transformer_trace(
        model,
        input_ids,
        top_k=top_k,
        compare_cache=compare_cache,
    )
    tokens = [str(token_id) for token_id in input_ids[0].tolist()]

    html_path = Path(output_html)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    trace.to_html(html_path, tokens=tokens)

    if output_json is not None:
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        trace.save_json(json_path)

    return html_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a Miniformer transformer trace report")
    parser.add_argument("--output-html", type=Path, default=Path("trace.html"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--d-ff", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-cache-compare", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    path = write_trace_report(
        args.output_html,
        output_json=args.output_json,
        seed=args.seed,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        top_k=args.top_k,
        compare_cache=not args.no_cache_compare,
    )
    print(f"trace_html={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
