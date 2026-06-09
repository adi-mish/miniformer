import pytest
import torch

from miniformer.config import CONFIG_SCHEMA_VERSION, migrate_config_dict
from miniformer.model.transformer import Transformer, TransformerConfig


@pytest.mark.parametrize("d_model,n_heads", [(32, 4), (48, 3), (64, 8), (128, 16)])
def test_valid_config_combinations(d_model, n_heads):
    """Test that valid d_model/n_heads combinations work correctly."""
    config = TransformerConfig(
        vocab_size=100, d_model=d_model, n_heads=n_heads, n_layers=2, d_ff=d_model * 4
    )
    model = Transformer(config)

    # Test forward pass works
    x = torch.randint(0, 100, (2, 10))
    output = model(x)
    assert output.hidden_states is not None
    assert output.output.shape == (2, 10, d_model)


def test_invalid_n_heads_not_divisible():
    """Test that n_heads must divide d_model evenly."""
    with pytest.raises(ValueError, match="d_model.*divisible.*n_heads"):
        TransformerConfig(
            vocab_size=100, d_model=64, n_heads=5, n_layers=2  # 64 not divisible by 5
        )


def test_invalid_activation():
    """Test that invalid activation functions are rejected."""
    with pytest.raises(ValueError, match="Unknown activation"):
        TransformerConfig(vocab_size=100, d_model=64, n_heads=4, activation="invalid_activation")


def test_invalid_rotary_pct():
    """Test that rotary percentage must stay in the supported range."""
    with pytest.raises(ValueError, match="rotary_pct"):
        TransformerConfig(rotary_pct=1.5)


def test_invalid_initializer_range():
    with pytest.raises(ValueError, match="initializer_range"):
        TransformerConfig(initializer_range=0.0)


def test_optional_attention_config_serializes(tmp_path):
    """Test optional attention fields are first-class config fields."""
    config = TransformerConfig(pre_norm=False, use_sdpa=True, position_mode="rope", rotary_pct=0.5)

    path = tmp_path / "config.json"
    config.save_json(path)
    loaded = TransformerConfig.from_json(path)

    assert loaded.pre_norm is False
    assert loaded.use_sdpa is True
    assert loaded.position_mode == "rope"
    assert loaded.rotary_pct == pytest.approx(0.5)
    assert loaded.schema_version == CONFIG_SCHEMA_VERSION


def test_config_serialization_includes_schema_version():
    config = TransformerConfig()

    assert config.to_dict()["schema_version"] == CONFIG_SCHEMA_VERSION


def test_from_dict_rejects_unknown_keys_by_default():
    payload = TransformerConfig().to_dict()
    payload["stale_field"] = True

    with pytest.raises(ValueError, match="stale_field"):
        TransformerConfig.from_dict(payload)


def test_from_dict_can_ignore_unknown_keys_explicitly():
    payload = TransformerConfig().to_dict()
    payload["stale_field"] = True

    loaded = TransformerConfig.from_dict(payload, allow_unknown=True)

    assert not hasattr(loaded, "stale_field")
    assert loaded.to_dict() == TransformerConfig().to_dict()


def test_migrate_legacy_config_without_schema_keeps_hidden_default():
    migrated = migrate_config_dict({"vocab_size": 128, "d_model": 16, "n_heads": 2})

    assert migrated["schema_version"] == CONFIG_SCHEMA_VERSION
    assert migrated["output_mode"] == "hidden"
    assert migrated["position_mode"] == "learned"
    loaded = TransformerConfig.from_dict(migrated)
    assert loaded.output_mode == "hidden"


def test_migrate_legacy_config_with_output_dim_uses_projection():
    migrated = migrate_config_dict(
        {"vocab_size": 128, "d_model": 16, "n_heads": 2, "output_dim": 3}
    )

    assert migrated["output_mode"] == "projection"
    loaded = TransformerConfig.from_dict(migrated)
    assert loaded.output_mode == "projection"
    assert loaded.output_dim == 3


def test_migrate_legacy_rope_config_preserves_position_policy():
    migrated = migrate_config_dict({"d_model": 16, "n_heads": 2, "rotary_pct": 0.5})

    assert migrated["position_mode"] == "rope"
    assert TransformerConfig.from_dict(migrated).position_mode == "rope"


def test_unsupported_config_schema_is_rejected():
    with pytest.raises(ValueError, match="schema_version"):
        TransformerConfig.from_dict({"schema_version": 999})


def test_input_dim_projection():
    """Test that input_dim != d_model creates proper projection layer."""
    config = TransformerConfig(
        input_dim=5,
        d_model=32,
        n_heads=4,
        n_layers=2,
        output_mode="projection",
        output_dim=1,
    )
    model = Transformer(config)

    # Should have input projection
    assert hasattr(model, "input_projection")
    assert isinstance(model.input_projection, torch.nn.Linear)
    assert model.input_projection.in_features == 5
    assert model.input_projection.out_features == 32

    # Test forward pass
    x = torch.randn(2, 10, 5)
    output = model(x)
    assert output.projection is not None
    assert output.output.shape == (2, 10, 1)


def test_output_mode_config_validation():
    with pytest.raises(ValueError, match="Unknown output_mode"):
        TransformerConfig(output_mode="guess")

    with pytest.raises(ValueError, match="hidden.*output_dim=None"):
        TransformerConfig(output_dim=2)

    with pytest.raises(ValueError, match="vocab.*token inputs"):
        TransformerConfig(input_dim=5, output_mode="vocab")

    with pytest.raises(ValueError, match="projection.*requires output_dim"):
        TransformerConfig(output_mode="projection")


def test_position_mode_config_validation():
    with pytest.raises(ValueError, match="Unknown position_mode"):
        TransformerConfig(position_mode="absolute")

    with pytest.raises(ValueError, match="learned.*rotary_pct=0"):
        TransformerConfig(position_mode="learned", rotary_pct=0.25)

    with pytest.raises(ValueError, match="rope.*rotary_pct > 0"):
        TransformerConfig(position_mode="rope", rotary_pct=0.0)

    with pytest.raises(ValueError, match="learned\\+rope.*rotary_pct > 0"):
        TransformerConfig(position_mode="learned+rope", rotary_pct=0.0)
