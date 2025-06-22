# Miniformer: Production-Grade Transformers for Local Development

A modern, production-grade Transformer implementation scaled down to run efficiently on local hardware. Miniformer balances state-of-the-art architecture with practical resource constraints while maintaining the versatility to handle various data types and tasks.

## 🔍 Overview

Miniformer implements the core principles from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) while incorporating modern improvements from research advances like:

- Memory-efficient attention mechanisms
- Rotary position embeddings
- Activation functions beyond GELU (SwiGLU)
- Pre-norm layer architecture
- Gradient and activation stabilizing techniques

## ✨ Features

### Current Features
- **Versatile Inputs**: Support for token-based NLP tasks and direct feature vector inputs (time series, sensor data, embeddings)
- **Architecture Innovations**:
  - Configurable pre/post norm layers
  - SwiGLU and other modern activation functions
  - RoPE (Rotary Position Embeddings) support
  - Efficient implementation of attention mechanisms
- **Task Adaptability**: Language modeling, classification, regression, and sequence-to-sequence tasks
- **Inference Optimization**: Key-value caching for efficient autoregressive generation
- **Training Utilities**:
  - Learning rate scheduling (linear, cosine)
  - Weight decay and gradient clipping
  - Checkpoint management
  - Early stopping
- **Model Serialization**: Save and load models with standardized interfaces
- **Testing & Validation**: Comprehensive test suite covering correctness, fuzzing, and performance

### Production-Grade Features
- **Numerical Stability**: Special attention to initialization, normalization techniques, and precision management
- **Memory Efficiency**: Optimized tensor operations and memory usage patterns
- **Developer Experience**: Clean APIs, comprehensive documentation, and helpful error messages
- **Extensibility**: Modular design for easy customization and extension
- **Test Coverage**: Comprehensive test suite ensuring code reliability and robustness

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/miniformer.git
cd miniformer

# Install the package
pip install -e .

# With optional development dependencies
pip install -e ".[dev]"
```

## 📋 Usage Examples

### NLP / Language Modeling

```python
import torch
from miniformer import Transformer

# Create a language model with modern architecture
model = Transformer(
    vocab_size=50257,  # GPT-2 vocabulary size
    d_model=384,       # Hidden dimension
    n_heads=6,         # Multi-head attention
    n_layers=6,        # Transformer layers
    activation="swiglu"  # Modern activation function
)

# Forward pass with token IDs
token_ids = torch.randint(0, 50_000, (2, 128))
output = model(token_ids)  # Shape: [2, 128, vocab_size]

# Generation with KV-caching for efficiency
model.eval()
with torch.no_grad():
    input_ids = torch.tensor([[464, 3290, 318]])  # "The quick fox"
    for _ in range(30):
        outputs = model(input_ids, use_cache=True)
        next_token = torch.argmax(outputs[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    
    # Decode generated text with your tokenizer
    # ...
```

### Time Series Forecasting

```python
import torch
from miniformer import Transformer, TransformerConfig

# Create a model for multivariate time series forecasting
config = TransformerConfig(
    input_dim=7,       # 7 features per time step
    d_model=64,        # Embedding dimension
    output_dim=7,      # Predicting same 7 features
    n_heads=4,
    n_layers=3,
    d_ff=128,
    dropout=0.1
)

model = Transformer(config)

# Forward pass with feature vectors
features = torch.randn(8, 96, 7)  # Batch size 8, sequence length 96, 7 features
predictions = model(features)  # Shape: [8, 96, 7]
```

### Sequence-to-Sequence Tasks

```python
from miniformer.model import Seq2SeqTransformer
from miniformer.config import TransformerConfig

# Create a translation model
config = TransformerConfig(
    vocab_size=32000,  # Shared vocabulary size
    d_model=256,
    n_heads=8,
    n_layers=3
)

model = Seq2SeqTransformer(config)

# Translation example
src_tokens = torch.tensor([[101, 2043, 2003, 1037, 13013, 2651, 102]])  # "This is a sample sentence"
tgt_tokens = torch.tensor([[101]])  # Start token

# Generate translation
translation = model.generate(
    src_tokens,
    max_new_tokens=20,
    temperature=0.7,
    top_p=0.9
)
```

## 🧠 Architecture

Miniformer implements a modern transformer architecture with these key components:

1. **Input Processing**: 
   - NLP: Token embeddings with learned positional information
   - Time Series: Linear projection for feature vectors

2. **Transformer Blocks**:
   - **Attention**: Multi-head attention with efficient implementation
   - **Position Encodings**: Options for:
     - Learned absolute position embeddings
     - Rotary position embeddings (RoPE) with configurable percentage
   - **Feed-Forward**: Configurable with modern activation functions
   - **Normalization**: Pre-norm architecture with layer normalization
   - **Residual Connections**: Throughout for gradient flow

3. **Output Projection**: Task-specific output projections

### Detailed Architecture
```
Input → [Embedding/Projection + Position Encoding]
       → N × [Attention → LayerNorm → FFN → LayerNorm] 
       → Output Projection
```

## 🔄 Current Status and Roadmap

### Current Status
Miniformer currently implements all the core features of modern transformer architectures with:
- ✅ Full encoder implementation with pre-norm architecture
- ✅ Multi-head attention with memory-efficient implementation
- ✅ Feature-based and token-based input support
- ✅ Seq2Seq encoder-decoder architecture
- ✅ Basic training pipeline with modern optimizers
- ✅ Test suite for model correctness
- ✅ Key-value caching for efficient inference

### Future Plans

#### Near-term (1-2 months)
- ⬜ FlashAttention 2 integration for faster training
- ⬜ Multi-query and grouped-query attention variants
- ⬜ Memory-efficient checkpointing for activation gradients
- ⬜ Quantization support (INT8, FP16) for inference
- ⬜ Advanced decoding strategies (beam search, diverse beam search)

#### Mid-term (3-6 months)
- ⬜ Distributed training support with DeepSpeed/FSDP
- ⬜ Parameter-efficient fine-tuning techniques (LoRA, Adapters)
- ⬜ Mixture-of-Experts integration for larger parameter capacity
- ⬜ Custom CUDA kernels for optimized operations
- ⬜ Streaming inference for memory-constrained environments

#### Long-term Vision
- ⬜ Cross-attention optimizations for multi-modal support
- ⬜ Retrieval-augmented generation capabilities
- ⬜ Model compression techniques (pruning, distillation)
- ⬜ Mobile-optimized variants for edge deployment
- ⬜ Native integration with popular data pipelines

## 🧩 Extensibility

Miniformer is designed for extensibility. Add your custom:
- Attention mechanisms
- Position encoding schemes
- Activation functions
- Layer implementations
- Task-specific heads

Example extension:

```python
from miniformer.model.attention import MultiHeadAttention

class LocalMultiHeadAttention(MultiHeadAttention):
    """Multi-head attention with a sliding window constraint."""
    
    def __init__(self, d_model, n_heads, window_size=128, **kwargs):
        super().__init__(d_model, n_heads, **kwargs)
        self.window_size = window_size
        
    def forward(self, q, k, v, mask=None):
        # Implement local attention with window_size constraint
        # ...
```

## 🔧 Configuration

Miniformer offers easy configuration through the TransformerConfig:

```python
from miniformer.config import TransformerConfig

# Create custom configuration
config = TransformerConfig(
    vocab_size=10000,
    d_model=256,
    n_heads=8,
    n_layers=4,
    d_ff=1024,
    dropout=0.1,
    activation="swiglu",
    rotary_pct=0.25,  # Use RoPE for 25% of dimensions
    pre_norm=True,    # Use pre-norm architecture
    max_seq_len=2048
)
```

## 📊 Testing

Miniformer includes:
- Unit tests for component correctness
- Integration tests for model behavior
- Property-based fuzzing for robustness
- Performance benchmarks

```bash
# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=miniformer
```

## 🔒 License

MIT

## 🔗 References

- Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Xiong, R., et al. (2020). [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- Su, J., et al. (2021). [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Shazeer, N. (2020). [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- Dao, T., et al. (2022). [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- Anthropic. (2023). [Claude Technical Report](https://www-cdn.anthropic.com/de2c9438-a790-4187-b533-82e28053df75/Model_Card_Claude.pdf)
