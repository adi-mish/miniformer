import os
import tempfile
import torch
import pytest
import json
import lightning.pytorch as pl
from types import SimpleNamespace
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / 'src'))

from miniformer.train.train_config import TrainConfig
from miniformer.train.datamodule import MiniFormerDataModule
from miniformer.train.module import MiniFormerLitModule


class DummyTokenizer:
    """Simple tokenizer for testing language modeling."""
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
    
    def encode(self, text, add_special_tokens=True):
        # Simple character-level encoding for testing
        return [ord(c) % self.vocab_size for c in text]


class TestLanguageModelingE2E:
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory and sample data for language modeling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal language modeling dataset
            train_data = [
                {"text": f"This is sample text number {i}."} for i in range(20)
            ]
            val_data = [
                {"text": f"This is validation text {i}."} for i in range(10)
            ]
            
            # Write data files
            train_file = os.path.join(tmpdir, "train.jsonl")
            val_file = os.path.join(tmpdir, "val.jsonl")
            
            with open(train_file, 'w') as f:
                for item in train_data:
                    f.write(json.dumps(item) + "\n")
                    
            with open(val_file, 'w') as f:
                for item in val_data:
                    f.write(json.dumps(item) + "\n")
            
            yield SimpleNamespace(
                tmpdir=tmpdir,
                train_file=train_file,
                val_file=val_file
            )
    
    def test_language_modeling_training(self, temp_data_dir):
        """Test end-to-end language modeling training pipeline."""
        # Configure training
        cfg = TrainConfig()
        cfg.task = "language_modeling"
        cfg.train_path = temp_data_dir.train_file
        cfg.val_path = temp_data_dir.val_file
        cfg.test_path = ""
        cfg.batch_size = 4
        cfg.max_epochs = 2
        cfg.lr = 0.01
        cfg.model = "seq2seq"  # Use seq2seq model for LM
        cfg.model_config = {
            "vocab_size": 100,
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 1,
            "d_ff": 32,
            "dropout": 0.0
        }
        cfg.work_dir = temp_data_dir.tmpdir
        cfg.experiment_name = "test_lm"
        cfg.logger = "csv"
        cfg.gpus = 0
        cfg.early_stopping_patience = 0
        cfg.scheduler = "none"
        
        # Initialize tokenizer and components
        tokenizer = DummyTokenizer(vocab_size=100)
        datamodule = MiniFormerDataModule(cfg, tokenizer=tokenizer)
        model = MiniFormerLitModule(cfg)
        
        # Verify perplexity metric is set up
        assert hasattr(model, "val_ppl"), "Language model should have perplexity metric"
        
        # Extract initial parameters for later comparison
        initial_params = {}
        for name, param in model.named_parameters():
            initial_params[name] = param.clone().detach()
        
        # Setup trainer with minimal settings
        trainer = pl.Trainer(
            max_epochs=cfg.max_epochs,
            accelerator="cpu",
            devices=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            enable_model_summary=False,
        )
        
        # Train
        datamodule.prepare_data()
        datamodule.setup()
        trainer.fit(model, datamodule=datamodule)
        
        # Verify training occurred (parameters changed)
        param_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(initial_params[name], param):
                param_changed = True
                break
        
        assert param_changed, "Model parameters did not change during training"
        
        # Test language model inference
        model.eval()
        dataloader = datamodule.val_dataloader()
        batch = next(iter(dataloader))
        
        with torch.no_grad():
            # For language modeling, both source and target are input_ids
            src = batch["input_ids"]
            tgt = batch["input_ids"]
            
            # Test forward pass returns logits with expected shape
            outputs = model.model(src, tgt, use_causal_mask=True)[0]
            
            # Logits should have shape [batch_size, seq_len, vocab_size]
            assert outputs.dim() == 3, "Expected 3D output tensor for language model"
            assert outputs.size(0) == src.size(0), "Batch size mismatch"
            assert outputs.size(1) == src.size(1), "Sequence length mismatch"
            assert outputs.size(2) == cfg.model_config["vocab_size"], "Vocab size mismatch"
