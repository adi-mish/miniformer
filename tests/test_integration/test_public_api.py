import torch

import miniformer


def test_top_level_exports_data_helpers():
    batch = miniformer.collate_records(
        [{"input": "alpha beta", "label": 1}, {"input": "alpha", "label": 0}],
        task="classification",
        vocab_size=32,
    )

    assert batch["input_ids"].shape == (2, 2)
    assert batch["attention_mask"].tolist() == [[True, True], [True, False]]
    assert batch["labels"].tolist() == [1, 0]
    assert miniformer.attention_mask_from_lengths([1, 2]).tolist() == [
        [True, False],
        [True, True],
    ]
    assert miniformer.encode_text("alpha", vocab_size=32).dtype == torch.long
    assert callable(miniformer.encode_text_batch)
    assert callable(miniformer.pad_token_sequences)


def test_top_level_exports_models_and_trace_helpers(tmp_path):
    model = miniformer.Transformer(
        miniformer.TransformerConfig(
            vocab_size=32,
            d_model=8,
            n_heads=2,
            n_layers=1,
            d_ff=16,
            output_mode="vocab",
            dropout=0.0,
        )
    ).eval()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    trace = miniformer.capture_transformer_trace(model, input_ids, top_k=2)
    html_path = tmp_path / "trace.html"
    miniformer.save_trace_html(trace, html_path, tokens=["1", "2", "3"])

    assert trace.output_shape == (1, 3, 32)
    assert "Miniformer Trace" in html_path.read_text()
