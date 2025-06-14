# Miniformer: A General-Purpose Transformer Implementation

A compact, versatile implementation of the Transformer architecture as described in the ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper by Vaswani et al. This implementation supports both NLP tasks and general sequence modeling for time series, sensor data, and more.

## Features

- Complete transformer architecture with scaled-down parameters
- Supports both token-based inputs and direct feature vector inputs
- Configurable for various tasks: language modeling, classification, regression
- Visualization tools for attention patterns
- Training utilities with PyTorch integration

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/miniformer.git
cd miniformer

# Install the package
pip install -e .
```

## Architecture

Miniformer implements the standard transformer architecture with the following components:

1. **Token/Feature Embeddings**: Convert input tokens to vectors or use input features directly
2. **Positional Encodings**: Add position information to embeddings
3. **Multi-Head Attention**: Allow the model to focus on different parts of the input
4. **Feed-Forward Networks**: Process the attention output
5. **Layer Normalization**: Stabilize the learning process
6. **Residual Connections**: Help with gradient flow during training

## Usage Examples

### NLP / Language Modeling

```python
import torch
from miniformer import Transformer

# Create a language model
model = Transformer(
    vocab_size=1000,
    d_model=64,
    n_heads=4,
    n_layers=3,
    d_ff=256,
    dropout=0.1
)

# Forward pass with token IDs
token_ids = torch.randint(0, 1000, (2, 10))  # Batch size 2, sequence length 10
output = model(token_ids)
```

### Time Series / Sensor Data

```python
import torch
from miniformer import Transformer, TransformerConfig

# Create a model for time series data
config = TransformerConfig(
    input_dim=5,     # 5 features per time step
    d_model=5,       # Same as input_dim
    output_dim=1,    # Predicting a single value
    n_heads=2,
    n_layers=2,
    d_ff=32
)

model = Transformer(config)

# Forward pass with feature vectors
features = torch.randn(2, 24, 5)  # Batch size 2, sequence length 24, 5 features
output = model(features)
```

## Model Configuration

You can easily configure the model for different tasks:

- `d_model`: Embedding dimension (default: 64)
- `n_heads`: Number of attention heads (default: 4)
- `n_layers`: Number of transformer layers (default: 3)
- `d_ff`: Dimension of feed-forward networks (default: 256)
- `dropout`: Dropout rate (default: 0.1)
- `input_dim`: Dimension of input features (if using feature vectors directly)
- `output_dim`: Dimension of output (number of classes or prediction dimensions)

## Training

Miniformer includes a Trainer class for easy model training with different task types:

```python
from miniformer import Transformer, TransformerTrainer, TransformerConfig

# For language modeling
model = Transformer(vocab_size=1000, d_model=64)
trainer = TransformerTrainer(
    model=model,
    task_type="language_modeling"
)

# For regression tasks
config = TransformerConfig(input_dim=5, d_model=5, output_dim=1)
model = Transformer(config)
trainer = TransformerTrainer(
    model=model,
    task_type="regression"
)

trainer.train(train_dataset, epochs=10)
```

## Visualization

Visualize attention patterns:

```python
from miniformer.visualization import plot_attention

attention_weights = model.get_attention_weights(input_data)
plot_attention(attention_weights, layer=0, head=0)
```

## License

MIT

## References

- Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Devlin, J., et al. (2018). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
