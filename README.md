# Miniformer

Production-grade Transformer implementations, scaled down for efficient local development.

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Installation](#installation)
* [Quickstart](#quickstart)

  * [Training via CLI](#training-via-cli)
  * [Python API](#python-api)
* [Components](#components)

  * [Data Module](#data-module)
  * [Lightning Module](#lightning-module)
  * [Trainer Script](#trainer-script)
* [Models](#models)

  * [Encoder-Only Transformer](#encoder-only-transformer)
  * [Seq2Seq Transformer](#seq2seq-transformer)
* [Configuration](#configuration)
* [Extensibility](#extensibility)
* [Testing](#testing)
* [🔄 Current Status & Roadmap](#current-status--roadmap)
* [Future Scope](#future-scope)
* [License](#license)
* [References](#references)

---

## Overview

Miniformer implements core Transformer architectures ("Attention Is All You Need") with modern improvements—efficient attention, rotary embeddings, gated activations—while remaining lightweight enough for local hardware.

---

## Features

* **Modular Design**: Easily swap attention, feed-forward, position encodings, and activation functions.
* **Task Flexibility**: Language modeling, classification, regression, and seq2seq tasks.
* **Lightning Integration**: Built-in PyTorch Lightning modules for training, logging, checkpointing, and early stopping.
* **Data Utilities**: JSON-lines dataset loader and collation for variable-length sequences.
* **Inference Optimization**: KV-cache support for fast autoregressive generation.
* **Production-Grade**: Numerical stability, memory efficiency, and clean APIs for extension.

---

## Installation

```bash
git clone https://github.com/yourusername/miniformer.git
cd miniformer
pip install -e .
# for dev dependencies
pip install -e ".[dev]"
```

---

## Quickstart

### Training via CLI

The entrypoint script `trainer.py` parses a rich set of arguments via `TrainConfig`, sets up data, model, logger, and callbacks, then runs training (and optional testing).

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

# Build a simple encoder-only model
config = TransformerConfig(vocab_size=10000, d_model=256, n_heads=8, n_layers=4)
model = Transformer(config)

# Forward pass on token IDs
input_ids = torch.randint(0, 10000, (2, 128))
logits = model(input_ids)  # [2,128,10000]
```

---

## Components

### Data Module

* **`JSONLinesDataset`**
  Loads line-separated JSON records, supports language modeling, classification, and regression tasks.
* **`MiniFormerDataModule`**
  Wraps datasets into PyTorch Lightning `DataLoader`s with configurable batch size, collation, and workers.

### Lightning Module

* **`MiniFormerLitModule`**
  Lightning wrapper for encoder-only or seq2seq models; sets up metrics (perplexity, accuracy, MAE), optimizers, schedulers, training/validation steps, and checkpoint management.

### Trainer Script

* **`trainer.py`**
  CLI entrypoint: parses `TrainConfig`, seeds, creates tokenizer for LM tasks, initializes data and LitModule, sets up logging (TensorBoard, WandB, CSV), and runs `pl.Trainer.fit()` (and `test()`).

---

## Models

### Encoder-Only Transformer

* **`Transformer`**
  Stack of `Encoder` layers with optional token or feature inputs, causal and padding masks, and configurable output head.
* **`EncoderLayer`**
  Pre-norm attention + feed-forward block with configurable RoPE and activation.
* **`MultiHeadAttention`** & **RoPE** support
  Efficient attention implementation with rotary embeddings.
* **`FeedForward`**
  GELU, ReLU, or SwiGLU variants.
* **Embedding & Position**
  Token embeddings and sinusoidal/learned positional encodings.

### Seq2Seq Transformer

* **`Seq2SeqTransformer`**
  Full encoder–decoder wrapper with cross-attention, causal masking, and `generate()` utilities for autoregressive decoding.
* **`DecoderLayer`**
  Self-attention, cross-attention, and feed-forward with optional KV-cache.

---

## Configuration

Customize via dataclasses in `model_config.py` and `train_config.py`:

```python
from miniformer.config.model_config import TransformerConfig
from miniformer.train_config import TrainConfig

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

Subclass or replace components:

* **Attention**: Inherit `MultiHeadAttention`.
* **Feed-Forward**: Extend `FeedForward` with new activations.
* **Layers**: Customize `EncoderLayer`/`DecoderLayer`.
* **Tasks**: Add new heads or pipelines.

---

## Testing

```bash
# Run all tests\ npytest tests/
# With coverage
pytest tests/ --cov=miniformer
```

---

## 🔄 Current Status & Roadmap

### Current Status

Miniformer currently implements all core modern Transformer features:

* Full encoder and seq2seq architectures
* Efficient attention with RoPE options
* Gated activations (SwiGLU) and numerical stability
* Training utilities: schedulers, clipping, checkpointing
* KV-cache for fast inference
* Comprehensive test suite

### Future Plans

#### Near-term (1–2 months)

* Integrate FlashAttention 2 for GPU speedups and memory savings
* Implement memory-efficient activation checkpointing
* Add INT8 and FP16 quantization workflows
* Provide beam search and diverse beam search decoding
* Extend CLI with advanced logging options (MLflow, Weights & Biases sweeps)
* Publish example notebooks for domain-specific tasks

#### Mid-term (3–6 months)

* DeepSpeed & FSDP integration for distributed training
* Parameter-efficient fine-tuning: LoRA, Adapters, Prefix Tuning
* Support multi-query and grouped-query attention variants
* ONNX export and inference for edge deployment
* Dynamic sparsity and pruning tools
* Enhanced metrics dashboard and hyperparameter search integration

#### Long-term (6–12 months)

* Mixture-of-Experts (MoE) layers for scalable capacity
* Custom CUDA kernels for specialized attention and FFN ops
* Retrieval-Augmented Generation (RAG) pipelines
* Multi-modal model support (vision, audio, text)
* Streaming inference and low-latency APIs
* Federated learning and privacy-preserving training

#### Visionary (>12 months)

* Automated architecture search (NAS) integration
* Self-optimizing hyperparameter control loops
* Universal embedding space for cross-domain models
* Synthetic data generation and augmentation frameworks
* Differential privacy and secure enclave training
* Continuous learning with drift detection and model updating

---

## Future Scope

The ongoing evolution of AI systems opens numerous avenues for Miniformer’s growth:

1. **Hardware-Aware Optimizations**

   * Auto-tuning kernels for various GPU/TPU architectures.
   * End-to-end compilation pipelines (XLA, TVM) for maximal throughput.

2. **Multi-Modal Fusion**

   * Unified architectures processing text, image, and audio streams.
   * Cross-attention blocks bridging modalities in real time.

3. **Adaptive Inference**

   * Early-exit and dynamic-depth Transformers to tailor compute per input.
   * On-device runtime profiling for resource-constrained settings.

4. **Scalable Collaboration**

   * Model hubs with versioning, peer reviews, and modular plugins.
   * Integration with MLOps platforms for CI/CD of model updates.

5. **Advanced Training Paradigms**

   * Meta-learning loops for rapid task adaptation.
   * Curriculum learning schedulers and self-play data generation.

6. **Robustness & Safety**

   * Adversarial training protocols and certified robustness.
   * Bias detection and mitigation toolkits.

7. **Ecosystem & Community**

   * Plugin architecture for community-contributed attention, layers, and utilities.
   * Regular workshops, tutorials, and benchmark challenges.

---

## License

GNU GENERAL PUBLIC LICENSE Version 3

---

## References

* Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* Xiong, R., et al. (2020). [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
* Su, J., et al. (2021). [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
* Shazeer, N. (2020). [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
* Dao, T., et al. (2022). [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
* Anthropic (2023). [Claude Technical Report](https://www-cdn.anthropic.com/de2c9438-a790-4187-b533-82e28053df75/Model_Card_Claude.pdf)
