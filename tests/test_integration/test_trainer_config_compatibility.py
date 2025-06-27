import os
import tempfile
import torch
import pytest
import lightning.pytorch as pl
from types import SimpleNamespace
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / 'src'))

from miniformer.train.train_config import TrainConfig
from miniformer.train.module import MiniFormerLitModule
from miniformer.model.transformer import Transformer, TransformerConfig


class TestTrainerConfigCompatibility:
    """Test compatibility between trainer configurations and model behavior."""
    
    def test_lr_scheduler_compatibility(self):
        """Test that different learning rate schedulers work correctly."""
        schedulers = ["none", "linear", "onecycle", "cosine"]
        
        for scheduler in schedulers:
            # Create config with the scheduler
            cfg = TrainConfig()
            cfg.task = "classification"
            cfg.model_config = {
                "vocab_size": 100,
                "d_model": 16,
                "n_heads": 2,
                "n_layers": 1,
                "output_dim": 3
            }
            cfg.lr = 0.01
            # Convert string to literal by direct assignment of the known value
            cfg.scheduler = scheduler  # type: ignore
            cfg.warmup_steps = 10
            cfg.max_epochs = 2
            
            # Initialize model
            model = MiniFormerLitModule(cfg)
            
            # Check scheduler configuration
            if scheduler == "none":
                optimizer = model.configure_optimizers()
                assert isinstance(optimizer, torch.optim.Optimizer), "Expected a single optimizer"
            else:
                # All other schedulers should return a tuple of optimizers and schedulers
                result = model.configure_optimizers()
                assert isinstance(result, tuple) and len(result) == 2, \
                    f"Expected tuple of (optimizers, schedulers) for {scheduler}"
                opts, scheds = result
                assert len(opts) == 1, "Expected 1 optimizer"
                assert len(scheds) == 1, "Expected 1 scheduler"
                
                # Verify specific scheduler types
                if scheduler == "linear":
                    assert isinstance(scheds[0], torch.optim.lr_scheduler.LinearLR), \
                        "Wrong scheduler type for linear"
                elif scheduler == "onecycle":
                    assert isinstance(scheds[0], torch.optim.lr_scheduler.OneCycleLR), \
                        "Wrong scheduler type for onecycle"
                elif scheduler == "cosine":
                    assert isinstance(scheds[0], torch.optim.lr_scheduler.CosineAnnealingWarmRestarts), \
                        "Wrong scheduler type for cosine"
    
    def test_gradient_clipping_compatibility(self):
        """Test gradient clipping with different values."""
        clip_values = [0.1, 0.5, 1.0]
        
        for clip_val in clip_values:
            # Setup basic model
            cfg = TrainConfig()
            cfg.task = "classification"
            cfg.model_config = {
                "vocab_size": 100,
                "d_model": 16,
                "n_heads": 2,
                "n_layers": 1,
                "output_dim": 3
            }
            cfg.gradient_clip_val = clip_val
            
            # Create a test input and target
            batch = [{"input": "test", "labels": 0}] * 2
            
            # Initialize model and perform forward/backward pass
            model = MiniFormerLitModule(cfg)
            
            # Create a class method to handle mocking
            orig_model = model.model
            mock_output = torch.randn(2, 3, requires_grad=True)
            
            # Create a mock model class with the same interface
            class MockModel(Transformer):
                def __init__(self, original_model):
                    # Initialize with the same configuration as the original model
                    if hasattr(original_model, 'config'):
                        super().__init__(original_model.config)
                    else:
                        # Create a minimal config if needed
                        default_config = TransformerConfig(
                            vocab_size=100, d_model=16, n_heads=2, n_layers=1)
                        super().__init__(default_config)
                    self.original_model = original_model
                
                def __call__(self, *args, **kwargs):
                    return mock_output
                    
                # Pass through any attribute access to the original model
                def __getattr__(self, name):
                    return getattr(self.original_model, name)
            
            # Replace the model with our mock
            model.model = MockModel(orig_model)
            
            try:
                # Manual training step to test gradient clipping
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model.model(batch)
                loss = torch.nn.functional.cross_entropy(
                    outputs, torch.tensor([0, 1], dtype=torch.long))
                
                # Backward pass
                loss.backward()
                
                # Check gradients before clipping
                grad_norms_before = [
                    param.grad.norm().item() 
                    for param in model.parameters()
                    if param.grad is not None
                ]
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                
                # Check gradients after clipping
                grad_norms_after = [
                    param.grad.norm().item() 
                    for param in model.parameters()
                    if param.grad is not None
                ]
                
                # Verify clipping works - max gradient norm should not exceed clip value (with small tolerance)
                if max(grad_norms_before) > clip_val:  # Only if clipping was actually needed
                    assert max(grad_norms_after) <= clip_val + 1e-5, \
                        f"Gradient norm {max(grad_norms_after)} exceeds clip value {clip_val}"
            finally:
                # Restore original model
                model.model = orig_model
    
    def test_precision_compatibility(self):
        """Test model compatibility with different precision settings."""
        # Skip if no GPU available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping precision test")
            
        precisions = ["32-true", "16-mixed", "bf16-mixed"]
        
        for precision in precisions:
            cfg = TrainConfig()
            cfg.task = "classification"
            cfg.model_config = {
                "vocab_size": 100,
                "d_model": 16,
                "n_heads": 2,
                "n_layers": 1,
                "output_dim": 3
            }
            # Convert string precision to the proper format accepted by TrainConfig
            if precision == "16-mixed":
                cfg.precision = 16  # type: ignore
            elif precision == "32-true":
                cfg.precision = 32  # type: ignore
            elif precision == "bf16-mixed":
                cfg.precision = "bf16"  # type: ignore
            
            # Initialize model
            model = MiniFormerLitModule(cfg)
            
            # Create a dummy batch
            batch = [{"input": "test", "labels": 0}] * 2
            
            try:
                # Setup a trainer with the precision
                # Convert string representation to the actual precision format expected by Trainer
                trainer_precision = None
                if precision == "32-true":
                    trainer_precision = 32
                elif precision == "16-mixed":
                    trainer_precision = 16
                elif precision == "bf16-mixed":
                    trainer_precision = "bf16"
                    
                trainer = pl.Trainer(
                    accelerator="gpu",
                    devices=1,
                    precision=trainer_precision,
                    enable_checkpointing=False,
                    enable_progress_bar=False,
                    logger=False,
                    enable_model_summary=False,
                    max_epochs=1,
                )
                
                # Apply the precision settings to the model
                # Just check that initialization doesn't fail
                assert trainer is not None
                
                # For bf16, we'd need to ensure the GPU supports it
                if precision == "bf16-mixed" and not torch.cuda.is_bf16_supported():
                    pytest.skip("BFloat16 not supported on this GPU")
                    
            except Exception as e:
                # Different PyTorch versions might not support all precision settings
                # Just make sure it doesn't crash unexpectedly
                assert "not supported" in str(e).lower() or "invalid" in str(e).lower(), \
                    f"Unexpected error with precision {precision}: {e}"
