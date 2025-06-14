import torch
import argparse
import os
from pathlib import Path

from miniformer import Transformer, TransformerTrainer, TransformerConfig
from miniformer import setup_logging
from miniformer.config import TINY_CONFIG, SMALL_CONFIG, BASE_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer model")
    parser.add_argument("--config", choices=["tiny", "small", "base"], default="tiny",
                        help="Model size configuration")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO", 
                        help="Logging level")
    parser.add_argument("--log-file", type=str, default=None, help="Log file path")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level, log_file=args.log_file)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load configuration
    if args.config == "tiny":
        config = TINY_CONFIG
    elif args.config == "small":
        config = SMALL_CONFIG
    else:
        config = BASE_CONFIG
    
    # Override configuration with command line arguments
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    
    # Create a dummy dataset for demonstration
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=1000, seq_len=20):
            self.size = size
            self.seq_len = seq_len
            self.vocab_size = config.vocab_size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Create random input and target sequences
            input_ids = torch.randint(1, self.vocab_size, (self.seq_len,))
            labels = torch.randint(0, self.vocab_size, (self.seq_len,))
            return input_ids, labels
    
    train_dataset = DummyDataset(size=5000)
    val_dataset = DummyDataset(size=500)
    
    # Create model
    model = Transformer(config)
    
    # Create trainer
    trainer = TransformerTrainer(model)
    
    # Train model
    history = trainer.train(
        train_dataset=train_dataset,
        epochs=args.epochs,
        validation_dataset=val_dataset,
        eval_steps=100,
        save_steps=500,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Save final model
    output_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    import logging
    main()
