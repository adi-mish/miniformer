import matplotlib
import torch

from miniformer.config.model_config import TransformerConfig
from miniformer.model.seq2seq_transformer import Seq2SeqTransformer
from miniformer.model.transformer import Transformer
from miniformer.visualization import capture_transformer_trace, plot_trace_summary

matplotlib.use("Agg")


def test_capture_transformer_trace_encoder():
    model = Transformer(
        TransformerConfig(
            vocab_size=30,
            d_model=16,
            n_heads=2,
            n_layers=2,
            d_ff=32,
            dropout=0.0,
            output_dim=30,
        )
    )
    input_ids = torch.randint(1, 30, (2, 5))

    trace = capture_transformer_trace(model, input_ids)

    assert trace.output_shape == (2, 5, 30)
    assert [layer.name for layer in trace.layers] == ["encoder.layers.0", "encoder.layers.1"]
    assert len(trace.attentions) == 2
    assert all(attn.entropy >= 0 for attn in trace.attentions)


def test_capture_transformer_trace_seq2seq_and_plot():
    model = Seq2SeqTransformer(
        TransformerConfig(
            vocab_size=30,
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
            output_dim=30,
        )
    )
    src = torch.randint(1, 30, (2, 4))
    tgt = torch.randint(1, 30, (2, 3))

    trace = capture_transformer_trace(model, src, tgt)
    fig, ax = plot_trace_summary(trace)

    assert trace.output_shape == (2, 3, 30)
    assert {layer.name for layer in trace.layers} == {"encoder.layers.0", "decoder.layers.0"}
    assert len(trace.attentions) == 3
    assert ax.get_ylabel() == "Activation norm"
    assert fig is not None
