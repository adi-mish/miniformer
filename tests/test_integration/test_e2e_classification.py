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
from miniformer.train.datamodule import MiniFormerDataModule, JSONLinesDataset
from miniformer.train.module import MiniFormerLitModule


class TestClassificationE2E:
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory and sample data for classification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal classification dataset
            train_data = [
                {"input": f"sample {i}", "label": i % 3} for i in range(20)
            ]
            val_data = [
                {"input": f"val {i}", "label": i % 3} for i in range(10)
            ]
            
            # Write data files
            train_file = os.path.join(tmpdir, "train.jsonl")
            val_file = os.path.join(tmpdir, "val.jsonl")
            
            with open(train_file, 'w') as f:
                for item in train_data:
                    f.write(f"{item}\n")
                    
            with open(val_file, 'w') as f:
                for item in val_data:
                    f.write(f"{item}\n")
            
            yield SimpleNamespace(
                tmpdir=tmpdir,
                train_file=train_file,
                val_file=val_file
            )
    
    def test_classification_training(self, temp_data_dir):
        """Test end-to-end classification training pipeline."""
        # Configure training
        cfg = TrainConfig()
        cfg.task = "classification"
        cfg.train_path = temp_data_dir.train_file
        cfg.val_path = temp_data_dir.val_file
        cfg.test_path = ""
        cfg.batch_size = 4
        cfg.max_epochs = 2
        cfg.lr = 0.01
        cfg.model_config = {
            "vocab_size": 100,
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 1,
            "d_ff": 32,
            "dropout": 0.0,
            "output_dim": 3  # 3 classes for classification
        }
        cfg.work_dir = temp_data_dir.tmpdir
        cfg.experiment_name = "test_clf"
        cfg.logger = "csv"
        cfg.gpus = 0
        cfg.early_stopping_patience = 0
        
        # Initialize components
        datamodule = MiniFormerDataModule(cfg, tokenizer=None)
        model = MiniFormerLitModule(cfg)
        
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
        trainer.fit(model, datamodule=datamodule)
        
        # Verify training occurred (parameters changed)
        param_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(initial_params[name], param):
                param_changed = True
                break
        
        assert param_changed, "Model parameters did not change during training"
        
        # Test inference
        model.eval()
        dataloader = datamodule.val_dataloader()
        batch = next(iter(dataloader))
        
        with torch.no_grad():
            outputs = model.model(batch)
            assert outputs.shape[0] == len(batch), "Output batch size mismatch"
            if outputs.dim() == 3:
                # If 3D tensor, first dimension is batch, second is sequence length
                assert outputs.dim() == 3
                # Classification output should be [batch_size, seq_len, num_classes]
                assert outputs.shape[-1] == 3
            else:
                # May be simplified to [batch_size, num_classes] for classification
                assert outputs.shape[-1] == 3
