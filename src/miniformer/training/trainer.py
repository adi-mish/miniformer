import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union, Callable, Any, Tuple, Literal
from pathlib import Path
import os
import json
import time
import logging
from tqdm import tqdm

from miniformer.config import TransformerConfig
from miniformer.utils.logging import get_logger

logger = get_logger(__name__)


class TransformerTrainer:
    """Trainer for transformer models"""
    
    def __init__(
        self,
        model,
        config: Optional[TransformerConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        task_type: Literal["classification", "regression", "language_modeling"] = "language_modeling",
        criterion: Optional[nn.Module] = None
    ):
        """
        Args:
            model: Transformer model
            config: Model configuration
            optimizer: Optimizer for training
            lr_scheduler: Learning rate scheduler
            device: Device to run on (None for auto-detection)
            task_type: Type of task ("classification", "regression", or "language_modeling")
            criterion: Custom loss function
        """
        self.model = model
        self.config = config or (model.config if hasattr(model, 'config') else TransformerConfig())
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.task_type = task_type
        self.model.to(self.device)
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Task type: {task_type}")
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.98),
                eps=1e-8
            )
        else:
            self.optimizer = optimizer
            
        # Setup learning rate scheduler
        if lr_scheduler is None and self.config.lr_scheduler != "constant":
            if self.config.lr_scheduler == "linear":
                self.lr_scheduler = optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=0.0,
                    total_iters=self.config.warmup_steps
                )
            elif self.config.lr_scheduler == "cosine":
                self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    T_max=100000
                )
            else:
                self.lr_scheduler = None
        else:
            self.lr_scheduler = lr_scheduler
        
        # Setup loss function based on task type
        if criterion is not None:
            self.criterion = criterion
        else:
            if task_type == "regression":
                self.criterion = nn.MSELoss()
            elif task_type == "classification" or task_type == "language_modeling":
                self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def train(
        self,
        train_dataset,
        epochs: int = 10,
        validation_dataset = None,
        eval_steps: int = 500,
        save_steps: int = 1000,
        checkpoint_dir: str = "./checkpoints",
        callbacks: Optional[List[Callable]] = None
    ):
        """
        Train the model
        
        Args:
            train_dataset: Training dataset
            epochs: Number of epochs
            validation_dataset: Optional validation dataset
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            checkpoint_dir: Directory to save checkpoints
            callbacks: List of callback functions
            
        Returns:
            history: Training history
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True
        )
        
        val_dataloader = None  # Initialize val_dataloader
        if validation_dataset:
            val_dataloader = DataLoader(
                validation_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                pin_memory=True
            )
        
        history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
        
        # Save initial model and config
        self._save_checkpoint(checkpoint_dir, "initial")
        
        start_time = time.time()
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_loss = 0
            epoch_start_time = time.time()
            
            # Initialize current_lr with the learning rate at the start of the epoch.
            # This ensures current_lr is bound even if the dataloader is empty.
            # If the dataloader has items, current_lr will be updated in the loop.
            current_lr = self.optimizer.param_groups[0]['lr']
            
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            for step, batch in enumerate(progress_bar):
                input_ids, labels = batch
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(input_ids)
                
                # Calculate loss based on task type
                if self.task_type == "regression":
                    # For regression, both outputs and labels are expected to be 3D tensors
                    loss = self.criterion(outputs, labels)
                else:
                    # For classification/language modeling, reshape for cross entropy
                    loss = self.criterion(
                        outputs.contiguous().view(-1, outputs.size(-1)),
                        labels.contiguous().view(-1)
                    )
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                # Update progress bar
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.6f}'
                })
                
                total_loss += loss.item()
                self.global_step += 1
                
                # Evaluation
                if validation_dataset and self.global_step % eval_steps == 0:
                    val_loss = self.evaluate(val_dataloader)
                    logger.info(f"Step {self.global_step}: Validation loss: {val_loss:.4f}")
                    history['val_loss'].append((self.global_step, val_loss))
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        logger.info(f"New best validation loss: {val_loss:.4f}")
                        self.best_val_loss = val_loss
                        self._save_checkpoint(checkpoint_dir, "best")
                
                # Save checkpoint
                if self.global_step % save_steps == 0:
                    self._save_checkpoint(checkpoint_dir, f"step-{self.global_step}")
                
                # Run callbacks
                if callbacks:
                    for callback in callbacks:
                        callback(self, step, loss.item())
            
            # End of epoch
            avg_train_loss = total_loss / len(dataloader)
            history['train_loss'].append((self.global_step, avg_train_loss))
            history['learning_rate'].append((self.global_step, current_lr))
            
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1}/{epochs} - Train loss: {avg_train_loss:.4f} - Time: {epoch_time:.2f}s")
            
            # Validation at end of epoch
            if validation_dataset:
                val_loss = self.evaluate(val_dataloader)
                logger.info(f"Epoch {epoch+1}/{epochs} - Val loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    logger.info(f"New best validation loss: {val_loss:.4f}")
                    self.best_val_loss = val_loss
                    self._save_checkpoint(checkpoint_dir, "best")
            
            # Save epoch checkpoint
            self._save_checkpoint(checkpoint_dir, f"epoch-{epoch+1}")
                
        # End of training
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        # Save final model
        self._save_checkpoint(checkpoint_dir, "final")
        
        return history
    
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids, labels = batch
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(input_ids)
                
                # Calculate loss based on task type
                if self.task_type == "regression":
                    loss = self.criterion(outputs, labels)
                else:
                    loss = self.criterion(
                        outputs.contiguous().view(-1, outputs.size(-1)),
                        labels.contiguous().view(-1)
                    )
                
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
    
    def _save_checkpoint(self, checkpoint_dir, name):
        """Save a checkpoint of the model, optimizer, and training state"""
        checkpoint_path = os.path.join(checkpoint_dir, f"{name}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model
        if hasattr(self.model, 'module'):  # Handle distributed/parallel training
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
            
        # Save model weights
        torch.save(model_state_dict, os.path.join(checkpoint_path, "model.pt"))
        
        # Save optimizer and scheduler states
        training_state = {
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }
        
        if self.lr_scheduler is not None:
            training_state['lr_scheduler'] = self.lr_scheduler.state_dict()
            
        torch.save(training_state, os.path.join(checkpoint_path, "training_state.pt"))
        
        # Save configuration
        if hasattr(self.model, 'config'):
            self.model.config.save_json(os.path.join(checkpoint_path, "config.json"))
        
    def load_checkpoint(self, checkpoint_path, load_optimizer=True):
        """Load a checkpoint"""
        # Load model weights
        model_path = os.path.join(checkpoint_path, "model.pt")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        if load_optimizer:
            # Load training state
            training_state_path = os.path.join(checkpoint_path, "training_state.pt")
            if os.path.exists(training_state_path):
                training_state = torch.load(training_state_path, map_location=self.device)
                
                self.optimizer.load_state_dict(training_state['optimizer'])
                self.global_step = training_state['global_step']
                self.best_val_loss = training_state['best_val_loss']
                
                if 'lr_scheduler' in training_state and self.lr_scheduler:
                    self.lr_scheduler.load_state_dict(training_state['lr_scheduler'])
                    
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return self
