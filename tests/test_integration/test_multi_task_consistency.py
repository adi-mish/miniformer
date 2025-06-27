import os
import tempfile
import torch
import pytest
import json
from types import SimpleNamespace
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / 'src'))

from miniformer.train.train_config import TrainConfig
from miniformer.train.module import MiniFormerLitModule
from miniformer.model.transformer import Transformer, TransformerConfig


class TestMultiTaskConsistency:
    """Test that models can be properly used for different tasks."""
    
    def test_model_task_switching(self):
        """Test that the same model architecture can be used for different tasks."""
        # Create a base transformer configuration that will be used for all tasks
        base_config = {
            "vocab_size": 100,
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 64,
            "dropout": 0.0
        }
        
        # Test for the three main tasks
        tasks = ["classification", "regression", "language_modeling"]
        models = {}
        
        for task in tasks:
            # Make a copy of the config and adjust for the task
            model_config = base_config.copy()
            
            if task == "classification":
                model_config["output_dim"] = 3  # 3 classes
            elif task == "regression":
                model_config["output_dim"] = 1  # Single output value
            
            # Create a TrainConfig for the task
            cfg = TrainConfig()
            # Use literal values instead of the variable
            if task == "classification":
                cfg.task = "classification"
            elif task == "regression":
                cfg.task = "regression"
            elif task == "language_modeling":
                cfg.task = "language_modeling"
            cfg.model_config = model_config
            cfg.model = "encoder" if task != "language_modeling" else "seq2seq"
            
            # Initialize model
            model = MiniFormerLitModule(cfg)
            models[task] = model
            
            # Verify model has proper metrics set up
            if task == "classification":
                assert hasattr(model, "val_acc"), "Classification model missing accuracy metric"
            elif task == "regression":
                assert hasattr(model, "val_mae"), "Regression model missing MAE metric"
            elif task == "language_modeling":
                assert hasattr(model, "val_ppl"), "Language model missing perplexity metric"
        
        # Create a regression model with the same architecture
        reg_cfg = TrainConfig()
        reg_cfg.task = "regression"  # This is already a literal, so it's fine
        reg_cfg.model_config = base_config.copy()
        clf_state_dict = models["classification"].model.state_dict()
        
        # Create a regression model with the same architecture
        reg_cfg = TrainConfig()
        reg_cfg.task = "regression"
        reg_cfg.model_config = base_config.copy()
        reg_cfg.model_config["output_dim"] = 1  # Regression output
        
        reg_model = MiniFormerLitModule(reg_cfg)
        
        # We can't directly load state dict because output layer is different size
        # But we can verify that all matching parameters can be copied
        
        matching_keys = [k for k in clf_state_dict.keys() 
                        if k in reg_model.model.state_dict() 
                        and clf_state_dict[k].shape == reg_model.model.state_dict()[k].shape]
        
        # There should be shared parameters between the models
        assert len(matching_keys) > 0, "No matching parameters between classification and regression models"
        
        # Create a partial state dict with only the matching keys
        partial_state_dict = {k: clf_state_dict[k] for k in matching_keys}
        
        # Load the partial state dict
        msg = reg_model.model.load_state_dict(partial_state_dict, strict=False)
        
        # Verify at least some parameters were loaded
        assert len(msg.missing_keys) < len(reg_model.model.state_dict()), "No parameters were loaded"
        
        # Ensure we can perform forward passes with all models
        with torch.no_grad():
            # For classification and regression, just need a batch of tokens
            if hasattr(models["classification"].model, "token_embedding"):
                # Token-based input
                batch = torch.randint(0, 100, (2, 5))
                clf_out = models["classification"].model(batch)
                assert clf_out.shape[-1] == 3, "Wrong classification output dimension"
                
                reg_out = models["regression"].model(batch)
                assert reg_out.shape[-1] == 1, "Wrong regression output dimension"
            
            # For language modeling with seq2seq
            if models["language_modeling"].model.__class__.__name__ == "Seq2SeqTransformer":
                src = torch.randint(0, 100, (2, 5))
                tgt = torch.randint(0, 100, (2, 5))
                lm_out = models["language_modeling"].model(src, tgt)[0]
                assert lm_out.shape[-1] == 100, "Wrong language model output dimension"
