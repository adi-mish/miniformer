"""Public package exports for miniformer.

Model classes are imported lazily so configuration and training utilities can
be imported in environments that have not installed PyTorch yet.
"""

from typing import TYPE_CHECKING

__all__ = [
    "Transformer",
    "Seq2SeqTransformer",
    "TransformerModelOutput",
    "Seq2SeqModelOutput",
    "TransformerConfig",
    "TransformerTrace",
    "capture_transformer_trace",
    "save_trace_html",
    "Batch",
    "BatchKind",
    "TextTokenizer",
    "TokenizerProtocol",
    "WhitespaceHashTokenizer",
    "attention_mask_from_lengths",
    "collate_records",
    "encode_text",
    "encode_text_batch",
    "pad_token_sequences",
    "ensure_tokenizer",
    "get_logger",
    "setup_logging",
]
__version__ = "0.1.0"

if TYPE_CHECKING:
    from miniformer.config.model_config import TransformerConfig
    from miniformer.data.batch import Batch, BatchKind
    from miniformer.data.preprocessing import (
        attention_mask_from_lengths,
        collate_records,
        encode_text,
        encode_text_batch,
        pad_token_sequences,
    )
    from miniformer.data.tokenizers import (
        TextTokenizer,
        TokenizerProtocol,
        WhitespaceHashTokenizer,
        ensure_tokenizer,
    )
    from miniformer.inspect import TransformerTrace, capture_transformer_trace, save_trace_html
    from miniformer.model.outputs import Seq2SeqModelOutput, TransformerModelOutput
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
    if name == "Seq2SeqModelOutput":
        from miniformer.model.outputs import Seq2SeqModelOutput

        globals()[name] = Seq2SeqModelOutput
        return Seq2SeqModelOutput
    if name == "TransformerModelOutput":
        from miniformer.model.outputs import TransformerModelOutput

        globals()[name] = TransformerModelOutput
        return TransformerModelOutput
    if name in {"TransformerTrace", "capture_transformer_trace", "save_trace_html"}:
        from miniformer.inspect import TransformerTrace, capture_transformer_trace, save_trace_html

        globals()["TransformerTrace"] = TransformerTrace
        globals()["capture_transformer_trace"] = capture_transformer_trace
        globals()["save_trace_html"] = save_trace_html
        return globals()[name]
    if name in {
        "Batch",
        "BatchKind",
        "TextTokenizer",
        "TokenizerProtocol",
        "WhitespaceHashTokenizer",
        "attention_mask_from_lengths",
        "collate_records",
        "encode_text",
        "encode_text_batch",
        "pad_token_sequences",
        "ensure_tokenizer",
    }:
        from miniformer.data.batch import Batch, BatchKind
        from miniformer.data.preprocessing import (
            attention_mask_from_lengths,
            collate_records,
            encode_text,
            encode_text_batch,
            pad_token_sequences,
        )
        from miniformer.data.tokenizers import (
            TextTokenizer,
            TokenizerProtocol,
            WhitespaceHashTokenizer,
            ensure_tokenizer,
        )

        globals()["Batch"] = Batch
        globals()["BatchKind"] = BatchKind
        globals()["TextTokenizer"] = TextTokenizer
        globals()["TokenizerProtocol"] = TokenizerProtocol
        globals()["WhitespaceHashTokenizer"] = WhitespaceHashTokenizer
        globals()["attention_mask_from_lengths"] = attention_mask_from_lengths
        globals()["collate_records"] = collate_records
        globals()["encode_text"] = encode_text
        globals()["encode_text_batch"] = encode_text_batch
        globals()["pad_token_sequences"] = pad_token_sequences
        globals()["ensure_tokenizer"] = ensure_tokenizer
        return globals()[name]
    if name in {"get_logger", "setup_logging"}:
        from miniformer.utils.logging import get_logger, setup_logging

        globals()["get_logger"] = get_logger
        globals()["setup_logging"] = setup_logging
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
