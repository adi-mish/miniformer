# Miniformer

Production-grade Transformer implementations, scaled down for efficient local development.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
  - [Training via CLI](#training-via-cli)
  - [Python API](#python-api)
- [Components](#components)
  - [Data Module](#data-module)
  - [Lightning Module](#lightning-module)
  - [Trainer Script](#trainer-script)
- [Models](#models)
  - [Encoder-Only Transformer](#encoder-only-transformer)
  - [Seq2Seq Transformer](#seq2seq-transformer)
- [Configuration](#configuration)
- [Extensibility](#extensibility)
- [Testing](#testing)
- [Current Status & Roadmap](#current-status--roadmap)
- [Future Scope](#future-scope)
- [License](#license)
- [References](#references)

---

## Overview

Miniformer implements core Transformer architectures (“Attention Is All You Need”) with modern improvements—efficient attention, rotary embeddings, gated activations—while remaining lightweight enough for local hardware.

---

## Features

- **Modular Design**  
  Swap attention mechanisms, feed-forward layers, position encodings, and activation functions easily.
- **Task Flexibility**  
  Language modeling, classification, regression, and seq2seq tasks.
- **Lightning Integration**  
  Built-in PyTorch Lightning modules for training, logging, checkpointing, and early stopping.
- **Data Utilities**  
  JSON-lines dataset loader and collation for variable-length sequences.
- **Inference Optimization**  
  KV-cache support for fast autoregressive generation.
- **Production-Grade**  
  Numerical stability, memory efficiency, and clean APIs for extension.

---

## Installation

```bash
git clone https://github.com/yourusername/miniformer.git
cd miniformer
pip install -e .
# for development dependencies
pip install -e ".[dev]"
```

---

## Quickstart

### Training via CLI

```bash
python -m miniformer.trainer \
  --train_path data/train.jsonl \
  --val_path data/val.jsonl \
  --task language_modeling \
  --model seq2seq \
  --model_config '{"vocab_size":50257,"d_model":384,"n_heads":6,"n_layers":6,"activation":"swiglu"}' \
  --batch_size 16 \
  --max_epochs 5 \
  --lr 5e-4 \
  --scheduler cosine \
  --logger tensorboard
```

### Python API

```python
from miniformer.config.model_config import TransformerConfig
from miniformer.model.transformer import Transformer
import torch

# Build an encoder-only Transformer
config = TransformerConfig(
    vocab_size=10000,
    d_model=256,
    n_heads=8,
    n_layers=4
)
model = Transformer(config)

# Forward pass on token IDs
input_ids = torch.randint(0, 10000, (2, 128))
logits = model(input_ids)  # shape: [2, 128, 10000]
```

---

## Components

### Data Module

- **`JSONLinesDataset`**  
  Loads line-separated JSON records; supports LM, classification, regression.
- **`MiniFormerDataModule`**  
  Wraps datasets into Lightning `DataLoader`s with configurable batch size, collation, and workers.

### Lightning Module

- **`MiniFormerLitModule`**  
  Lightning wrapper for encoder-only or seq2seq models; sets up metrics, optimizers, schedulers, training/validation steps, and checkpoint management.

### Trainer Script

- **`trainer.py`**  
  CLI entrypoint; parses `TrainConfig`, seeds RNGs, creates tokenizer, initializes data and model, sets up loggers and callbacks, then calls `pl.Trainer`.

---

## Models

### Encoder-Only Transformer

- **`Transformer`**  
  Stack of encoder layers, token/feature inputs, causal & padding masks, configurable output head with weight-tying.
- **`EncoderLayer`**, **`MultiHeadAttention`**, **RoPE support**, **`FeedForward`**  
  Configurable attention, rotary embeddings, activation variants (GELU, ReLU, SwiGLU).

### Seq2Seq Transformer

- **`Seq2SeqTransformer`**  
  Full encoder–decoder wrapper with cross-attention, causal masking, `generate()` utilities.
- **`DecoderLayer`**  
  Self-attention, cross-attention, feed-forward, optional KV-cache.

---

## Configuration

Customize via dataclasses in `model_config.py` and `train_config.py`:

```python
from miniformer.config.model_config import TransformerConfig
from miniformer.train_config import TrainConfig

# Model configuration
model_cfg = TransformerConfig(
    vocab_size=30522,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    dropout=0.1,
    activation="swiglu",
    rotary_pct=0.5,
    pre_norm=True,
    max_seq_len=1024,
)

# Training configuration
train_cfg = TrainConfig(
    train_path="train.jsonl",
    val_path="val.jsonl",
    batch_size=32,
    lr=3e-4,
    max_epochs=20,
    scheduler="onecycle",
    work_dir="./runs",
    logger="wandb"
)
```

---

## Extensibility

Subclass or replace core components:

- **Attention**: Inherit from `MultiHeadAttention`.
- **Feed-Forward**: Extend `FeedForward` with new activations.
- **Layers**: Customize `EncoderLayer`/`DecoderLayer`.
- **Tasks**: Add new heads or pipelines.

---

## Testing

```bash
pytest tests/ --cov=miniformer
```

---

## 🔄 Current Status & Roadmap

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
