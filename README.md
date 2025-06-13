# Miniformer: A Tiny Transformer Implementation

A compact, educational implementation of the Transformer architecture as described in the ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper by Vaswani et al. This implementation maintains all the core components of modern transformers while being lightweight and easy to understand.

## Features

- Complete transformer architecture with scaled-down parameters
- Encoder-decoder structure with multi-head self-attention
- Position embeddings and layer normalization
- Flexible configuration for model size and hyperparameters
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

1. **Token Embeddings**: Convert input tokens to vectors
2. **Positional Encodings**: Add position information to embeddings
3. **Multi-Head Attention**: Allow the model to focus on different parts of the input
4. **Feed-Forward Networks**: Process the attention output
5. **Layer Normalization**: Stabilize the learning process
6. **Residual Connections**: Help with gradient flow during training

## Usage Example

```python
import torch
from miniformer import Transformer

# Create a small transformer model
model = Transformer(
    vocab_size=1000,
    d_model=64,
    n_heads=4,
    n_layers=3,
    d_ff=256,
    dropout=0.1
)

# Forward pass
x = torch.randint(0, 1000, (2, 10))  # Batch size 2, sequence length 10
output = model(x)
```

## Model Configuration

You can easily configure the model size:

- `d_model`: Embedding dimension (default: 64)
- `n_heads`: Number of attention heads (default: 4)
- `n_layers`: Number of transformer layers (default: 3)
- `d_ff`: Dimension of feed-forward networks (default: 256)
- `dropout`: Dropout rate (default: 0.1)

## Training

Miniformer includes a Trainer class for easy model training:

```python
from miniformer import Transformer, TransformerTrainer

model = Transformer(vocab_size=1000, d_model=64)
trainer = TransformerTrainer(
    model=model,
    learning_rate=5e-5,
    batch_size=32
)

trainer.train(train_dataset, epochs=10)
```

## Visualization

Visualize attention patterns:

```python
from miniformer.visualization import plot_attention

attention_weights = model.get_attention_weights(input_ids)
plot_attention(attention_weights, layer=0, head=0)
```

## License

MIT

## References

- Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Devlin, J., et al. (2018). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
