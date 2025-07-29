# Miniformer: A Lightweight Transformer Library

[![PyPI version](https://badge.fury.io/py/miniformer.svg)](https://badge.fury.io/py/miniformer)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

Miniformer is a compact transformer implementation that I built to understand and experiment with the attention mechanism without the complexity of larger frameworks. Unlike heavyweight libraries that can be intimidating to modify, this codebase prioritizes readability and hackability—you can actually follow what's happening in each layer.

The library started as a learning exercise but evolved into something useful for prototyping. If you want to quickly test transformer variants or understand how attention really works under the hood, this might save you some headaches.

## What You Can Build With It

- **Small-scale language models** when you don't need GPT-scale infrastructure
- **Custom transformer variants** for research without diving into complex codebases  
- **Educational projects** to understand attention, embeddings, and training loops
- **Proof-of-concept models** for classification, regression, or sequence tasks
- **Edge deployment experiments** where model size matters

## Table of Contents

- [What's Actually Here](#whats-actually-here)
- [Getting Started](#getting-started)
- [Project Layout](#project-layout)
- [Basic Usage](#basic-usage)
  - [Command Line Training](#command-line-training)
  - [Python API](#python-api)
- [Data Formats](#data-formats)
- [Architecture Details](#architecture-details)
- [Training and Configuration](#training-and-configuration)
- [Extending the Code](#extending-the-code)
- [Running Tests](#running-tests)
- [Current State and Limitations](#current-state-and-limitations)
- [License](#license)
- [References](#references)

---

## What's Actually Here

I built this around the standard transformer architecture from "Attention Is All You Need," but kept things simple. The core components are:

- **Encoder-only transformer**: Good for classification, regression, or feature extraction
- **Encoder-decoder (seq2seq)**: Handles translation, summarization, or generation tasks
- **Multi-head attention** with optional rotary position embeddings (RoPE)
- **Feed-forward networks** supporting GELU, ReLU, and SwiGLU activations
- **PyTorch Lightning integration** for training without boilerplate
- **KV-caching** for faster autoregressive generation

The code is modular—you can swap out attention mechanisms, activations, or position encodings through config files rather than rewriting classes.

## Getting Started

Clone and install dependencies. You'll need Python 3.9+ and PyTorch 2.0+:

```bash
git clone https://github.com/adi-mish/miniformer.git
cd miniformer
pip install -e .
```

For development with all testing tools:

```bash
pip install -e ".[dev]"
```

## Project Layout

The structure is pretty straightforward:

```
miniformer/
├── src/miniformer/
│   ├── config/              # Configuration classes
│   ├── model/               # Core transformer components
│   │   ├── attention.py     # Multi-head attention with RoPE
│   │   ├── embedding.py     # Token & positional embeddings
│   │   ├── feedforward.py   # MLP layers with different activations
│   │   ├── transformer.py   # Encoder-only model
│   │   ├── encoder.py       # Encoder stack for seq2seq
│   │   ├── decoder.py       # Decoder stack for seq2seq
│   │   └── seq2seq_transformer.py # Full encoder-decoder
│   ├── train/               # Training infrastructure
│   │   ├── datamodule.py    # Data loading (JSONL format)
│   │   ├── module.py        # Lightning wrapper
│   │   ├── trainer.py       # CLI entry point
│   │   └── train_config.py  # Training configuration
│   ├── utils/               # Utility functions
│   └── visualization/       # Attention plotting tools
├── tests/                   # Comprehensive test suite
├── examples/                # Usage examples
└── configs/                 # Example configurations
```

---

## Basic Usage

### Command Line Training

The simplest way to train a model is through the CLI. Here's a language modeling example:

```bash
python -m miniformer.train.trainer \
  --train_path data/train.jsonl \
  --val_path data/val.jsonl \
  --task language_modeling \
  --model seq2seq \
  --model_config '{"vocab_size":50257,"d_model":384,"n_heads":6,"n_layers":6,"activation":"swiglu"}' \
  --batch_size 16 \
  --max_epochs 5 \
  --lr 5e-4 \
  --scheduler cosine \
  --warmup_steps 100 \
  --gradient_clip_val 1.0 \
  --logger tensorboard \
  --work_dir "./runs" \
  --experiment_name "my_lm"
```

For classification tasks, swap the task and model config:

```bash
python -m miniformer.train.trainer \
  --train_path data/classification_train.jsonl \
  --val_path data/classification_val.jsonl \
  --task classification \
  --model encoder \
  --model_config '{"vocab_size":30000,"d_model":256,"n_heads":8,"n_layers":4,"output_dim":10}' \
  --batch_size 32 \
  --max_epochs 10 \
  --lr 3e-4 \
  --scheduler onecycle
```

### Python API

If you prefer code to command lines, the API is pretty clean:

**Building an encoder model:**

```python
from miniformer.config.model_config import TransformerConfig
from miniformer.model.transformer import Transformer
import torch

# Configure the model
config = TransformerConfig(
    vocab_size=10000,
    d_model=256,
    n_heads=8,
    n_layers=4,
    d_ff=1024,
    dropout=0.1,
    activation="gelu",
    output_dim=10,  # Number of classes
    max_seq_len=512
)

model = Transformer(config)

# Basic forward pass
input_ids = torch.randint(0, 10000, (2, 128))
outputs = model(input_ids)  # Shape: [2, 128, 10]

# For classification, use the first token
cls_output = outputs[:, 0, :]  # Shape: [2, 10]
```

**Using the seq2seq model:**

```python
from miniformer.model.seq2seq_transformer import Seq2SeqTransformer

config = TransformerConfig(
    vocab_size=32000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    dropout=0.1,
    activation="swiglu",
    max_seq_len=1024
)

model = Seq2SeqTransformer(config)

# Training mode: provide both source and target
src_ids = torch.randint(0, 32000, (2, 64))
tgt_ids = torch.randint(0, 32000, (2, 48))
logits, _, _ = model(src_ids, tgt_ids)  # Shape: [2, 48, 32000]

# Generation mode
src_ids = torch.randint(0, 32000, (1, 64))
generated = model.generate(
    src_ids,
    max_new_tokens=50,
    temperature=0.8,
    top_k=40
)
```

---

## Data Formats

The library expects JSONL files (one JSON object per line). The format depends on your task:

**Language modeling** needs a `"text"` field:
```json
{"text": "This is sample text for language modeling."}
{"text": "Each line should be a separate document or sequence."}
```

**Classification** needs `"input"` and `"label"`:
```json
{"input": "This movie was great!", "label": 1}
{"input": "Terrible plot and acting.", "label": 0}
```

**Regression** uses `"input"` and either `"value"` or `"label"`:
```json
{"input": "The house is 2000 square feet", "value": 2000.0}
{"input": "Temperature reading: 72F", "value": 72.0}
```

One thing to note: for language modeling, you'll need a tokenizer. The trainer tries to load GPT-2's tokenizer from HuggingFace by default, but you can provide your own.

---

## Architecture Details

### The Core Models

**Transformer (Encoder-only)**: This is your standard encoder stack—great for classification, regression, or when you need fixed-size representations. It supports both token inputs (with embeddings) and direct feature vectors.

**Seq2SeqTransformer (Encoder-Decoder)**: Full sequence-to-sequence model with cross-attention. Use this for translation, summarization, or any task where input and output lengths differ.

Both models share the same underlying components but wire them together differently.

### Attention Implementation

The `MultiHeadAttention` class handles the core attention mechanism. A few things I learned while building it:

- **Rotary Position Embeddings (RoPE)**: These work better than learned positional embeddings for longer sequences. You can enable them with `rotary_pct` in the config (0.0 = disabled, 1.0 = full RoPE).
- **KV-caching**: Essential for fast generation. The implementation caches key-value pairs across decoding steps.
- **SDPA integration**: I started adding PyTorch 2.0's scaled dot-product attention but disabled it for now since it caused some compatibility issues during testing.

### Position Encodings

Three options:
- **Fixed sinusoidal**: The original approach from "Attention Is All You Need"
- **Learned embeddings**: Standard trainable position embeddings
- **Rotary (RoPE)**: Position-dependent rotations applied to queries and keys

### Activations

The feed-forward layers support:
- **GELU**: Smooth approximation of ReLU, works well for most tasks
- **ReLU**: Classic and fast, though can cause dead neurons
- **SwiGLU**: Gated activation that often performs better than GELU, especially for larger models

---

## Training and Configuration

### Model Configuration

The `TransformerConfig` class handles all model settings. Here are the key parameters you'll probably want to adjust:

```python
from miniformer.config.model_config import TransformerConfig

config = TransformerConfig(
    vocab_size=30522,        # Size of your vocabulary
    d_model=768,             # Hidden dimension (should be divisible by n_heads)
    n_heads=12,              # Number of attention heads
    n_layers=12,             # Number of transformer layers
    d_ff=3072,               # Feed-forward dimension (typically 4x d_model)
    dropout=0.1,             # Dropout rate
    activation="swiglu",     # "gelu", "relu", or "swiglu"
    max_seq_len=2048,        # Maximum sequence length
    output_dim=None          # Custom output size (defaults to vocab_size)
)
```

### Training Configuration

Training settings live in `TrainConfig`. The CLI automatically generates arguments from these fields:

```python
from miniformer.train.train_config import TrainConfig

config = TrainConfig(
    train_path="data/train.jsonl",
    val_path="data/val.jsonl",
    task="language_modeling",          # "classification", "regression"
    model="seq2seq",                   # "encoder" for encoder-only
    batch_size=32,
    max_epochs=10,
    lr=3e-4,                           # Learning rate
    weight_decay=0.01,
    scheduler="cosine",                # "linear", "onecycle", "none"
    warmup_steps=1000,
    gradient_clip_val=1.0,
    logger="tensorboard",              # "wandb", "csv", "none"
    gpus=1,
    precision="bf16"                   # Use bfloat16 for better performance
)
```

### Learning Rate Schedules

I included the ones I actually use:
- **cosine**: Cosine annealing with warm restarts
- **onecycle**: One-cycle learning rate for super-convergence
- **linear**: Linear warmup then constant
- **none**: Just use the base learning rate

### What Actually Gets Logged

The trainer logs different metrics based on your task:
- **Language modeling**: Loss and perplexity
- **Classification**: Loss and accuracy  
- **Regression**: Loss and mean absolute error

Checkpoints save based on validation loss by default, but you can change that in the config.

---

## Extending the Code

The nice thing about keeping it simple is that extending the library is pretty straightforward. Here are the main extension points:

### Custom Attention

Want to try a different attention mechanism? Inherit from `MultiHeadAttention`:

```python
from miniformer.model.attention import MultiHeadAttention

class CustomAttention(MultiHeadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Your custom initialization
        
    def forward(self, q, k, v, mask=None, past_kv=None, use_cache=False):
        # Your attention implementation
        return output, attention_weights, new_kv_cache
```

### Custom Activations

The `FeedForward` class is modular too:

```python
from miniformer.model.feedforward import FeedForward

class CustomFeedForward(FeedForward):
    def __init__(self, d_model, d_ff, **kwargs):
        super().__init__(d_model, d_ff, **kwargs)
        # Add your custom layers
        
    def forward(self, x):
        # Your custom forward pass
        return output
```

### Task-Specific Heads

For specialized tasks, you can create custom output layers:

```python
import torch.nn as nn

class TokenClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, hidden_states):
        return self.classifier(hidden_states)
```

---

## Running Tests

I wrote a fairly comprehensive test suite to catch regressions. Run it with:

```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=miniformer

# Specific test groups
pytest tests/test_model/      # Model architecture tests
pytest tests/test_train/      # Training pipeline tests
pytest tests/test_integration/ # End-to-end tests

# Pattern matching
pytest tests/ -k "attention"  # Only attention-related tests
```

The tests cover:
- **Model architecture**: Shape correctness, initialization, forward passes
- **Training behavior**: Loss computation, gradient flow, metric tracking
- **Persistence**: Save/load functionality, checkpoint compatibility
- **Integration**: Full training loops for each task type
- **Edge cases**: Empty batches, extreme values, device transfers

---

## Current State and Limitations

### What Works Now

The library currently handles:
- ✅ **Full encoder and seq2seq architectures**
- ✅ **Multi-head attention with RoPE support**
- ✅ **SwiGLU and other gated activations**
- ✅ **PyTorch Lightning training pipeline**
- ✅ **KV-cache for generation**
- ✅ **Classification, regression, and language modeling tasks**
- ✅ **Proper initialization and numerical stability**

### Current Limitations

Some things I haven't gotten to yet:
- ❌ **FlashAttention integration**: Started this but disabled due to compatibility issues
- ❌ **Beam search**: Only greedy and sampling generation for now
- ❌ **Model parallelism**: Single-GPU training only
- ❌ **Quantization**: No INT8/FP16 optimization yet
- ❌ **Advanced features**: No mixture of experts, sparse attention, etc.

### What I'm Working On

Near-term improvements (next few months):
- **FlashAttention 2**: Proper integration without the current compatibility issues
- **Beam search decoding**: For better generation quality
- **Gradient checkpointing**: Memory-efficient training for larger models
- **Better examples**: More realistic use cases and tutorials

Longer-term ideas:
- **LoRA fine-tuning**: Parameter-efficient adaptation
- **Model parallelism**: Multi-GPU training support  
- **ONNX export**: For deployment to different runtimes
- **Custom CUDA kernels**: For specialized operations

### Known Issues

- The SDPA (Scaled Dot-Product Attention) integration is disabled because it caused some test failures on certain PyTorch versions
- Generation with very long sequences can be slow without FlashAttention
- The tokenizer integration assumes HuggingFace transformers—you'll need to adapt for custom tokenizers

---

## License

This project is licensed under the GNU General Public License v3.0. See the LICENSE file for details.

---

## References

The papers that actually helped me build this:

- Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The original transformer paper
- Su, J., et al. (2021). [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - RoPE implementation
- Shazeer, N. (2020). [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - SwiGLU and other gated activations
- Dao, T., et al. (2022). [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) - Memory-efficient attention (working on integrating this)
- Xiong, R., et al. (2020). [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) - Pre-norm vs post-norm
