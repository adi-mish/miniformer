"""Public package exports for miniformer.

Model classes are imported lazily so configuration and training utilities can
be imported in environments that have not installed PyTorch yet.
"""

from typing import TYPE_CHECKING

__all__ = [
    "Transformer",
    "Seq2SeqTransformer",
    "TransformerConfig",
    "get_logger",
    "setup_logging",
]
__version__ = "0.1.0"

if TYPE_CHECKING:
    from miniformer.config.model_config import TransformerConfig
    from miniformer.model.seq2seq_transformer import Seq2SeqTransformer
    from miniformer.model.transformer import Transformer
    from miniformer.utils.logging import get_logger, setup_logging


def __getattr__(name: str):
    if name == "TransformerConfig":
        from miniformer.config.model_config import TransformerConfig

        globals()[name] = TransformerConfig
        return TransformerConfig
    if name == "Transformer":
        from miniformer.model.transformer import Transformer

        globals()[name] = Transformer
        return Transformer
    if name == "Seq2SeqTransformer":
        from miniformer.model.seq2seq_transformer import Seq2SeqTransformer

        globals()[name] = Seq2SeqTransformer
        return Seq2SeqTransformer
    if name in {"get_logger", "setup_logging"}:
        from miniformer.utils.logging import get_logger, setup_logging

        globals()["get_logger"] = get_logger
        globals()["setup_logging"] = setup_logging
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
