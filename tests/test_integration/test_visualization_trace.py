import json

import matplotlib
import torch

from miniformer.config.model_config import TransformerConfig
from miniformer.model.seq2seq_transformer import Seq2SeqTransformer
from miniformer.model.transformer import Transformer
from miniformer.visualization import capture_transformer_trace, plot_attention, plot_trace_summary

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

    model.train()
    with torch.no_grad():
        expected = model(input_ids)

    trace = model.trace(input_ids, top_k=3, compare_cache=True)

    with torch.no_grad():
        actual = model(input_ids)

    assert trace.output_shape == (2, 5, 30)
    assert [layer.name for layer in trace.layers] == ["encoder.layers.0", "encoder.layers.1"]
    assert all(layer.residual_delta_norm is not None for layer in trace.layers)
    assert all(layer.residual_delta_norm > 0 for layer in trace.layers)
    assert all(layer.self_attention_output_norm is not None for layer in trace.layers)
    assert all(layer.mlp_activation_norm is not None for layer in trace.layers)
    assert all(layer.mlp_output_norm is not None for layer in trace.layers)
    assert len(trace.attentions) == 2
    assert all(attn.entropy is not None and attn.entropy >= 0 for attn in trace.attentions)
    assert trace.logits is not None
    assert trace.logits.shape == (2, 5, 30)
    assert trace.logits.top_k == 3
    assert trace.cache.supported
    assert trace.cache.allclose
    assert trace.cache.max_abs_diff is not None
    assert trace.cache.max_abs_diff < 1e-6
    assert model.training
    assert torch.allclose(expected, actual)


def test_capture_transformer_trace_json_round_trip(tmp_path):
    model = Transformer(
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
    input_ids = torch.randint(1, 30, (1, 4))

    trace = capture_transformer_trace(model, input_ids, top_k=2)
    path = tmp_path / "trace.json"
    trace.save_json(path)

    loaded = json.loads(path.read_text())
    assert loaded["output_shape"] == [1, 4, 30]
    assert loaded["logits"]["top_k"] == 2
    assert loaded["layers"][0]["input_norm"] > 0
    assert json.loads(trace.to_json())["attentions"][0]["available"]


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

    trace = model.trace(src, tgt, compare_cache=True)
    fig, ax = plot_trace_summary(trace)

    assert trace.output_shape == (2, 3, 30)
    assert {layer.name for layer in trace.layers} == {"encoder.layers.0", "decoder.layers.0"}
    decoder_layer = next(layer for layer in trace.layers if layer.name == "decoder.layers.0")
    assert decoder_layer.self_attention_output_norm is not None
    assert decoder_layer.cross_attention_output_norm is not None
    assert len(trace.attentions) == 3
    assert trace.cache.supported
    assert trace.cache.allclose
    assert ax.get_ylabel() == "Activation norm"
    assert fig is not None


def test_trace_reports_unavailable_sdpa_attention():
    model = Transformer(
        TransformerConfig(
            vocab_size=30,
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
            output_dim=30,
            use_sdpa=True,
        )
    )
    input_ids = torch.randint(1, 30, (2, 5))

    trace = capture_transformer_trace(model, input_ids)

    assert len(trace.attentions) == 1
    assert not trace.attentions[0].available
    assert trace.attentions[0].entropy is None
    assert "use_sdpa=True" in trace.attentions[0].reason


def test_plot_attention_rejects_unavailable_attention_weights():
    try:
        plot_attention([None], layer=0, head=0)
    except ValueError as exc:
        assert "use_sdpa=True" in str(exc)
    else:
        raise AssertionError("plot_attention should reject unavailable attention weights")


def test_plot_attention_validates_token_length():
    attention = torch.ones(1, 1, 2, 2)

    try:
        plot_attention([attention], tokens=["only-one"])
    except ValueError as exc:
        assert "tokens length" in str(exc)
    else:
        raise AssertionError("plot_attention should reject mismatched token labels")
