import os
import tempfile
import torch
import pytest
import lightning.pytorch as pl
from types import SimpleNamespace
import glob
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / 'src'))

from miniformer.train.train_config import TrainConfig
from miniformer.train.datamodule import MiniFormerDataModule
from miniformer.train.module import MiniFormerLitModule
from lightning.pytorch.callbacks import ModelCheckpoint


class TestCheckpointSaveLoad:
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory and sample data."""
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
    
    def test_save_and_load_checkpoint(self, temp_data_dir):
        """Test saving and loading checkpoints during training."""
        # Configure training with checkpoint directory
        ckpt_dir = os.path.join(temp_data_dir.tmpdir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        
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
            "output_dim": 3
        }
        cfg.work_dir = temp_data_dir.tmpdir
        cfg.experiment_name = "test_ckpt"
        cfg.logger = "csv"
        cfg.gpus = 0
        cfg.checkpoint_metric = "val_loss"
        
        # Set up checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=False,
        )
        
        # Initialize components
        datamodule = MiniFormerDataModule(cfg, tokenizer=None)
        model = MiniFormerLitModule(cfg)
        
        # Setup trainer with checkpointing
        trainer = pl.Trainer(
            max_epochs=cfg.max_epochs,
            accelerator="cpu",
            devices=1,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False,
            logger=False,
            enable_model_summary=False,
        )
        
        # Train
        trainer.fit(model, datamodule=datamodule)
        
        # Verify checkpoint was saved
        checkpoint_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        assert len(checkpoint_files) > 0, "No checkpoint file was saved"
        
        # Save model state for comparison
        trained_state = {name: param.clone().detach() for name, param in model.named_parameters()}
        
        # Load from checkpoint and verify
        checkpoint_path = checkpoint_files[0]
        loaded_model = MiniFormerLitModule.load_from_checkpoint(
            checkpoint_path,
            cfg=cfg
        )
        
        # Verify loaded model has same architecture
        assert type(loaded_model.model) == type(model.model), "Loaded model has different architecture"
        
        # Verify parameters match the saved checkpoint
        for name, param in loaded_model.named_parameters():
            assert name in trained_state, f"Parameter {name} missing in loaded model"
            assert torch.allclose(param, trained_state[name]), \
                f"Parameter {name} has different values after loading"
        
        # Test inference with loaded model
        loaded_model.eval()
        dataloader = datamodule.val_dataloader()
        batch = next(iter(dataloader))
        
        with torch.no_grad():
            # Original model output
            original_output = model.model(batch)
            
            # Loaded model output
            loaded_output = loaded_model.model(batch)
            
            # Outputs should be identical
            assert torch.allclose(original_output, loaded_output, atol=1e-5), \
                "Loaded model produces different outputs"
    
    def test_resume_training_from_checkpoint(self, temp_data_dir):
        """Test resuming training from a checkpoint."""
        # Configure training with checkpoint directory
        ckpt_dir = os.path.join(temp_data_dir.tmpdir, "resume_checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        cfg = TrainConfig()
        cfg.task = "classification"
        cfg.train_path = temp_data_dir.train_file
        cfg.val_path = temp_data_dir.val_file
        cfg.test_path = ""
        cfg.batch_size = 4
        cfg.max_epochs = 1  # First we'll train for 1 epoch
        cfg.lr = 0.01
        cfg.model_config = {
            "vocab_size": 100,
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 1,
            "d_ff": 32,
            "dropout": 0.0,
            "output_dim": 3
        }
        cfg.work_dir = temp_data_dir.tmpdir
        cfg.experiment_name = "test_resume"
        cfg.logger = "csv"
        cfg.gpus = 0
        
        # Set up checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch}",
            monitor=None,  # Save all checkpoints
            save_top_k=-1,  # Save all checkpoints
            save_weights_only=False,
        )
        
        # Initialize components
        datamodule = MiniFormerDataModule(cfg, tokenizer=None)
        model = MiniFormerLitModule(cfg)
        
        # Setup trainer with checkpointing
        trainer = pl.Trainer(
            max_epochs=cfg.max_epochs,
            accelerator="cpu",
            devices=1,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False,
            logger=False,
            enable_model_summary=False,
        )
        
        # Train for 1 epoch
        trainer.fit(model, datamodule=datamodule)
        
        # Verify checkpoint was saved
        checkpoint_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        assert len(checkpoint_files) > 0, "No checkpoint file was saved"
        checkpoint_path = checkpoint_files[0]
        
        # Capture the state after first epoch
        first_epoch_state = {name: param.clone().detach() for name, param in model.named_parameters()}
        
        # Now create a new trainer and resume training for another epoch
        cfg.max_epochs = 2  # Train for one more epoch (total: 2)
        
        # Create new trainer to resume training
        new_trainer = pl.Trainer(
            max_epochs=cfg.max_epochs,
            accelerator="cpu",
            devices=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            enable_model_summary=False,
        )
        
        # Resume training from checkpoint
        new_trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)
        
        # Verify model parameters changed after resuming training
        param_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(first_epoch_state[name], param):
                param_changed = True
                break
        
        assert param_changed, "Model parameters did not change after resuming training"
