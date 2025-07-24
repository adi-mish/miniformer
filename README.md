# Miniformer: Lightweight Transformer Models for Efficient Deep Learning

[![PyPI version](https://badge.fury.io/py/miniformer.svg)](https://badge.fury.io/py/miniformer)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

**Miniformer** is a production-grade Transformer implementation scaled down for efficient local development and edge deployment. It provides a flexible, lightweight foundation for transformer-based NLP models that can run efficiently on consumer hardware for both training and inference.

## 🚀 Key Features

- **Lightweight & Efficient**: Optimized transformer architecture that runs on consumer hardware
- **Production-Ready**: Stable, tested implementation with proper initialization and normalization
- **Flexible Architecture**: Modular design with swappable attention mechanisms and position encodings
- **PyTorch Lightning Integration**: Built-in training, logging, and optimization utilities
- **Edge-Deployment Ready**: Inference optimization with KV-cache support for fast generation
- **Educational**: Clean implementation for learning transformer architecture internals

## 🧠 Use Cases

- Run transformer models on edge devices or limited hardware
- Prototype NLP applications with minimal resources
- Learn transformer internals with clean, well-documented code
- Build foundation models for language modeling, classification, regression and seq2seq tasks

## ⚡ Quick Installation

```bash
git clone https://github.com/adi-mish/miniformer.git
cd miniformer
pip install -e .
# for all dependencies
pip install -e ".[all]"
```

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
  - [Training via CLI](#training-via-cli)
  - [Python API](#python-api)
- [Data Format](#data-format)
  - [Language Modeling](#language-modeling)
  - [Classification](#classification)
  - [Regression](#regression)
- [Components](#components)
  - [Data Module](#data-module)
  - [Lightning Module](#lightning-module)
  - [Trainer Script](#trainer-script)
- [Models](#models)
  - [Encoder-Only Transformer](#encoder-only-transformer)
  - [Seq2Seq Transformer](#seq2seq-transformer)
  - [Attention Mechanisms](#attention-mechanisms)
  - [Position Encodings](#position-encodings)
  - [Feed-Forward Networks](#feed-forward-networks)
- [Training](#training)
  - [Hyperparameters](#hyperparameters)
  - [Optimizers](#optimizers)
  - [Schedulers](#schedulers)
  - [Logging](#logging)
- [Inference](#inference)
  - [Text Generation](#text-generation)
  - [Classification/Regression](#classificationregression)
- [Configuration](#configuration)
  - [Model Configuration](#model-configuration)
  - [Training Configuration](#training-configuration)
- [Extensibility](#extensibility)
- [Testing](#testing)
- [Current Status & Roadmap](#current-status--roadmap)
- [Future Scope](#future-scope)
- [License](#license)
- [References](#references)

---

## Overview

Miniformer implements core Transformer architectures ("Attention Is All You Need") with modern improvements—efficient attention, rotary embeddings, gated activations—while remaining lightweight enough for local hardware. It provides a production-grade foundation for transformer-based models that can run efficiently on consumer hardware for both training and inference.

---

## Features

- **Modular Design**  
  Swap attention mechanisms, feed-forward layers, position encodings, and activation functions easily through configuration.
- **Task Flexibility**  
  Language modeling (causal and masked), sequence classification, token classification, regression, and seq2seq tasks.
- **Lightning Integration**  
  Built-in PyTorch Lightning modules for training, logging, checkpointing, early stopping, and gradient clipping.
- **Data Utilities**  
  JSON-lines dataset loader and efficient collation for variable-length sequences with proper padding masking.
- **Inference Optimization**  
  KV-cache support for fast autoregressive generation, greedy and beam search decoding.
- **Production-Grade**  
  Numerical stability through correct initialization, normalization, and attention scaling; memory efficiency via gradient checkpointing options; and clean APIs for extension.
- **Visualization Tools**  
  Attention pattern visualization, embedding space projections, and training progress tracking.

---

## Installation

```bash
git clone https://github.com/adi-mish/miniformer.git
cd miniformer
pip install -e .
# for development dependencies
pip install -e ".[dev]"
```

## Requirements

Miniformer requires:

- Python 3.8+
- PyTorch 1.10+
- PyTorch Lightning 2.0+
- TorchMetrics
- (Optional) Transformers library for tokenizers
- (Optional) Matplotlib for visualization
- (Optional) TensorBoard, WandB or CSV logger for logging

Install all dependencies with:

```bash
pip install -e ".[all]"
```

## Project Structure

```
miniformer/
├── src/miniformer/
│   ├── config/           # Configuration classes
│   ├── model/            # Core model components
│   │   ├── attention.py  # Attention mechanisms
│   │   ├── embedding.py  # Token & positional embeddings
│   │   ├── feedforward.py # Feed-forward networks
│   │   ├── transformer.py # Encoder-only transformer
│   │   └── seq2seq_transformer.py # Encoder-decoder model
│   ├── train/            # Training utilities
│   │   ├── datamodule.py # Data loading and processing
│   │   ├── module.py     # Lightning modules
│   │   └── trainer.py    # Training entrypoint
│   └── visualization/    # Visualization tools
├── tests/                # Test suite
├── data/                 # Data directory
│   ├── train.jsonl       # Training data
│   ├── val.jsonl         # Validation data
│   └── test.jsonl        # Test data
└── examples/             # Usage examples
```

---

## Quickstart

### Training via CLI

Train a language model on your dataset:

```bash
python -m miniformer.train.trainer \
  --train_path data/train.jsonl \
  --val_path data/val.jsonl \
  --task language_modeling \
  --model seq2seq \
  --model_config '{"vocab_size":50257,"d_model":384,"n_heads":6,"n_layers":6,"activation":"swiglu","rotary_pct":0.5}' \
  --batch_size 16 \
  --max_epochs 5 \
  --lr 5e-4 \
  --weight_decay 0.01 \
  --scheduler cosine \
  --warmup_steps 100 \
  --gradient_clip_val 1.0 \
  --accumulate_grad_batches 2 \
  --logger tensorboard \
  --work_dir "./runs" \
  --experiment_name "my_lm" \
  --seed 42
```

For a classification task:

```bash
python -m miniformer.train.trainer \
  --train_path data/classification_train.jsonl \
  --val_path data/classification_val.jsonl \
  --task classification \
  --model transformer \
  --model_config '{"vocab_size":30000,"d_model":256,"n_heads":8,"n_layers":4,"output_dim":10}' \
  --batch_size 32 \
  --max_epochs 10 \
  --lr 3e-4 \
  --scheduler onecycle \
  --logger wandb
```

### Python API

#### Building and using an encoder-only model:

```python
from miniformer.config.model_config import TransformerConfig
from miniformer.model.transformer import Transformer
import torch

# Build an encoder-only Transformer for classification
config = TransformerConfig(
    vocab_size=10000,
    d_model=256,
    n_heads=8,
    n_layers=4,
    d_ff=1024,
    dropout=0.1,
    activation="gelu",
    output_dim=10,  # For 10-class classification
    max_seq_len=512
)
model = Transformer(config)

# Forward pass with token IDs
input_ids = torch.randint(0, 10000, (2, 128))
logits = model(input_ids)  # shape: [2, 128, 10]

# If you need just the classification output (first token)
cls_logits = logits[:, 0, :]  # shape: [2, 10]
```

#### Using the seq2seq model:

```python
from miniformer.config.model_config import TransformerConfig
from miniformer.model.seq2seq_transformer import Seq2SeqTransformer
import torch

# Build a seq2seq Transformer for translation/summarization
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

# Training: forward pass with source and target
src_ids = torch.randint(0, 32000, (2, 64))  # [batch_size, src_seq_len]
tgt_ids = torch.randint(0, 32000, (2, 48))  # [batch_size, tgt_seq_len]

# Returns logits, encoder_hidden_states, and kv_cache (if requested)
logits, enc_states, _ = model(src_ids, tgt_ids)  # logits: [2, 48, 32000]

# Inference: generate output sequence
src_ids = torch.randint(0, 32000, (1, 64))
generated_ids = model.generate(
    src_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    repetition_penalty=1.2
)
```

---

## Data Format

Miniformer uses JSONL (JSON Lines) files for training data. Each line should be a valid JSON object with the appropriate fields for your task.

### Language Modeling

For language modeling tasks, each line should contain a `"text"` field:

```json
{"text": "This is a sample sentence for language modeling."}
{"text": "Another example with different length."}
```

### Classification

For classification tasks, each line should have `"input"` and `"label"` fields:

```json
{"input": "This movie was amazing!", "label": 1}
{"input": "I didn't enjoy this book at all.", "label": 0}
```

### Regression

For regression tasks, each line should have `"input"` and either `"label"` or `"value"` fields:

```json
{"input": "The temperature is 32 degrees.", "value": 32.0}
{"input": "The house is 1500 square feet.", "value": 1500.0}
```

---

## Components

### Data Module

- **`JSONLinesDataset`**  
  Loads line-separated JSON records; supports language modeling, classification, and regression tasks with proper tokenization.
  
  ```python
  dataset = JSONLinesDataset(
      path="data/train.jsonl",
      tokenizer=tokenizer,  # Optional, required for language modeling
      task="language_modeling"  # or "classification", "regression"
  )
  ```

- **`MiniFormerDataModule`**  
  Wraps datasets into Lightning `DataLoader`s with configurable batch size, collation, and workers.
  
  ```python
  datamodule = MiniFormerDataModule(
      config=train_config,
      tokenizer=tokenizer  # Optional, required for language modeling
  )
  ```

### Lightning Module

- **`MiniFormerLitModule`**  
  Lightning wrapper for encoder-only or seq2seq models; sets up metrics, optimizers, schedulers, training/validation steps, and checkpoint management.
  
  ```python
  model = MiniFormerLitModule(train_config)
  ```

### Trainer Script

- **`trainer.py`**  
  CLI entrypoint; parses `TrainConfig`, seeds RNGs, creates tokenizer, initializes data and model, sets up loggers and callbacks, then calls `pl.Trainer`.
  
  ```python
  # See --help for all available options
  python -m miniformer.train.trainer --help
  ```

---

## Models

### Encoder-Only Transformer

- **`Transformer`**  
  Stack of encoder layers for token/feature inputs with support for causal & padding masks, configurable output head with weight-tying.
  
  **Architecture details:**
  - Token embedding + optional positional encoding
  - Stack of N encoder layers (configurable)
  - Layer normalization (pre-norm or post-norm)
  - Optional output projection layer
  
  **Available configurations:**
  ```python
  config = TransformerConfig(
      vocab_size=50000,      # Vocabulary size
      d_model=768,           # Hidden dimension
      n_heads=12,            # Number of attention heads
      n_layers=12,           # Number of layers
      d_ff=3072,             # Feed-forward dimension
      dropout=0.1,           # Dropout rate
      activation="gelu",     # Activation function (gelu, relu, swiglu)
      output_dim=None,       # Output dimension (defaults to vocab_size)
      max_seq_len=1024,      # Maximum sequence length
      pre_norm=True,         # Pre-norm or post-norm architecture
      rotary_pct=0.0         # Rotary embedding percentage (0.0-1.0)
  )
  ```

### Seq2Seq Transformer

- **`Seq2SeqTransformer`**  
  Full encoder–decoder wrapper with cross-attention, causal masking, `generate()` utilities.
  
  **Architecture details:**
  - Separate encoder and decoder stacks
  - Cross-attention in decoder layers
  - Shared or separate token embeddings
  - Generation utilities with KV-cache
  
  **Generation parameters:**
  ```python
  outputs = model.generate(
      input_ids,
      max_new_tokens=100,    # Maximum generation length
      temperature=1.0,       # Sampling temperature (1.0 = no change)
      top_k=50,              # Top-k sampling (0 = disabled)
      top_p=0.9,             # Nucleus sampling threshold (1.0 = disabled)
      do_sample=True,        # Use sampling (False = greedy)
      repetition_penalty=1.0 # Penalize repeated tokens (1.0 = no penalty)
  )
  ```

### Attention Mechanisms

- **`MultiHeadAttention`**  
  Supports standard scaled dot-product attention with optional features:
  
  - **Rotary Position Embeddings (RoPE)**: Applies position-dependent rotation to keys and queries
  - **KV-Cache**: For efficient autoregressive generation
  - **Attention Masks**: Support for causal, padding, and cross-attention masks
  - **Optional SDPA**: PyTorch 2.0+ scaled dot-product attention for improved performance

### Position Encodings

- **Fixed sinusoidal embeddings**
- **Learned position embeddings**
- **Rotary position embeddings (RoPE)**

### Feed-Forward Networks

- **`FeedForward`**  
  Standard MLP with configurable activation functions:
  
  - **GELU**: Original GELU activation
  - **ReLU**: Standard rectified linear unit
  - **SwiGLU**: Swish-Gated Linear Unit for improved performance

---

## Training

### Hyperparameters

Key hyperparameters for training:

- **Learning rate**: Typically 1e-4 to 5e-4 for transformers
- **Batch size**: Adjust based on available memory
- **Weight decay**: 0.01 is a good default
- **Gradient clipping**: 1.0 helps with stability
- **Warmup steps**: 5-10% of total steps
- **Max epochs**: Task-dependent

### Optimizers

Miniformer uses AdamW by default with configurable learning rate and weight decay.

### Schedulers

Available learning rate schedulers:

- **`none`**: Constant learning rate
- **`linear`**: Linear warmup followed by constant LR
- **`cosine`**: Cosine annealing with warmup
- **`onecycle`**: OneCycleLR for super-convergence

### Logging

Supported loggers:

- **TensorBoard**: `--logger tensorboard`
- **Weights & Biases**: `--logger wandb`
- **CSV**: `--logger csv`

---

## Inference

### Text Generation

For language modeling/text generation:

```python
from miniformer.model.seq2seq_transformer import Seq2SeqTransformer
import torch

# Load model (assuming you have a trained model checkpoint)
model = Seq2SeqTransformer.from_pretrained("./runs/my_lm/checkpoints/best_model")
model.eval()

# Encode input text (using your tokenizer)
input_ids = tokenizer.encode("Once upon a time", return_tensors="pt")

# Generate text
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.8,
        top_k=50
    )

# Decode output tokens
generated_text = tokenizer.decode(output_ids[0])
print(generated_text)
```

### Classification/Regression

For classification or regression:

```python
from miniformer.model.transformer import Transformer
import torch

# Load model
model = Transformer.from_pretrained("./runs/my_classifier/checkpoints/best_model")
model.eval()

# Encode input
input_ids = tokenizer.encode("This is a test input", return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(input_ids)
    
# For classification
if task == "classification":
    logits = outputs[:, 0, :]  # Use [CLS] token
    probs = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    
# For regression
elif task == "regression":
    prediction = outputs[:, 0, 0].item()  # Use first dimension of [CLS]
```

---

## Configuration

### Model Configuration

Configure the model architecture using `TransformerConfig`:

```python
from miniformer.config.model_config import TransformerConfig

config = TransformerConfig(
    vocab_size=30522,        # Vocabulary size
    d_model=768,             # Model dimension
    n_heads=12,              # Attention heads
    n_layers=12,             # Transformer layers
    d_ff=3072,               # Feed-forward dimension (4x d_model is common)
    dropout=0.1,             # Dropout rate
    activation="swiglu",     # Activation: "gelu", "relu", "swiglu"
    rotary_pct=0.25,         # Apply RoPE to 25% of dimensions
    pre_norm=True,           # Pre-norm architecture (vs post-norm)
    max_seq_len=2048,        # Maximum sequence length
    input_dim=None,          # For feature inputs (not tokens)
    output_dim=None          # Custom output dimension
)
```

### Training Configuration

Configure training using `TrainConfig`:

```python
from miniformer.train.train_config import TrainConfig

config = TrainConfig(
    train_path="data/train.jsonl",          # Training data path
    val_path="data/val.jsonl",              # Validation data path
    test_path="data/test.jsonl",            # Test data path (optional)
    task="language_modeling",               # Task type
    model="seq2seq",                        # Model architecture
    model_config={},                        # Model configuration dict
    batch_size=32,                          # Batch size
    max_epochs=10,                          # Maximum epochs
    lr=3e-4,                                # Learning rate
    weight_decay=0.01,                      # Weight decay
    scheduler="cosine",                     # LR scheduler
    warmup_steps=1000,                      # Warmup steps
    gradient_clip_val=1.0,                  # Gradient clipping
    accumulate_grad_batches=1,              # Gradient accumulation steps
    work_dir="./runs",                      # Working directory
    experiment_name="experiment1",          # Experiment name
    logger="tensorboard",                   # Logger type
    gpus=1,                                 # Number of GPUs
    precision=32,                           # Precision (16, 32)
    seed=42,                                # Random seed
    num_workers=4,                          # DataLoader workers
    early_stopping_patience=3,              # Early stopping patience
    checkpoint_metric="val_loss"            # Metric for checkpointing
)
```

---

## Extensibility

Miniformer is designed to be extended with custom components. Here are some extension points:

### Custom Attention Mechanism

Inherit from `MultiHeadAttention` to implement a new attention mechanism:

```python
from miniformer.model.attention import MultiHeadAttention
import torch

class FlashAttention(MultiHeadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization
        
    def forward(self, q, k, v, mask=None, past_kv=None, use_cache=False):
        # Custom attention implementation
        # ...
        return output, attention_weights, new_kv_cache
```

### Custom Feed-Forward Network

Extend `FeedForward` to implement new activation functions or architectures:

```python
from miniformer.model.feedforward import FeedForward
import torch

class MoEFeedForward(FeedForward):
    def __init__(self, d_model, d_ff, num_experts=4, **kwargs):
        super().__init__(d_model, d_ff, **kwargs)
        # Implement Mixture of Experts
        
    def forward(self, x):
        # MoE routing and processing
        return output
```

### Task-Specific Heads

Implement custom output heads for specific tasks:

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

## Testing

Run the comprehensive test suite to ensure everything is working correctly:

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=miniformer

# Run specific test category
pytest tests/test_model/

# Run tests with a specific name pattern
pytest tests/ -k "attention"
```

Tests include:
- Model architecture tests
- Training behavior tests
- Persistence tests (save/load)
- Attention mechanism tests
- Integration tests

---

## Current Status & Roadmap

### Current Status

Miniformer currently implements:

- Full encoder and seq2seq architectures
- Efficient attention with rotary embeddings
- Gated activations (SwiGLU)
- Training utilities: schedulers, clipping, checkpointing
- KV-cache for fast inference
- Comprehensive test suite

### Roadmap

**Near-term (1–2 months)**

- FlashAttention 2 integration
- Activation checkpointing
- INT8/FP16 quantization workflows
- Beam search decoding
- Advanced logging options
- Example notebooks

**Mid-term (3–6 months)**

- DeepSpeed & FSDP support
- LoRA/Adapter fine-tuning
- Multi-query attention variants
- ONNX export & edge inference
- Dynamic sparsity & pruning
- Hyperparameter search integration

**Long-term (6–12 months)**

- Mixture-of-Experts layers
- Custom CUDA kernels
- Retrieval-Augmented Generation (RAG)
- Multi-modal support
- Streaming inference APIs
- Federated & privacy-preserving training

**Visionary (>12 months)**

- Automated NAS integration
- Self-optimizing hyperparameter loops
- Universal embedding spaces
- Synthetic data pipelines
- Differential privacy frameworks
- Continuous learning

---

## Future Scope

1. **Hardware-Aware Optimizations**

   - Auto-tuning kernels (GPU/TPU)
   - End-to-end compilation (XLA, TVM)

2. **Multi-Modal Fusion**

   - Unified text/image/audio architectures
   - Real-time cross-modal attention

3. **Adaptive Inference**

   - Early-exit or dynamic-depth Transformers
   - On-device profiling

4. **Scalable Collaboration**

   - Model hubs with versioning & plugins
   - MLOps CI/CD integration

5. **Advanced Training Paradigms**

   - Meta-learning loops
   - Curriculum learning & self-play

6. **Robustness & Safety**

   - Adversarial training
   - Bias detection & mitigation

7. **Ecosystem & Community**

   - Plugin architecture for community contributions
   - Workshops, tutorials, benchmarks

---

## License

GNU GENERAL PUBLIC LICENSE Version 3

---

## References

- Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Xiong, R., et al. (2020). [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- Su, J., et al. (2021). [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Shazeer, N. (2020). [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- Dao, T., et al. (2022). [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- Anthropic (2023). [Claude Technical Report](https://www-cdn.anthropic.com/de2c9438-a790-4187-b533-82e28053df75/Model_Card_Claude.pdf)
