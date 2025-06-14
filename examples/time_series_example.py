import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from miniformer import Transformer, TransformerTrainer, TransformerConfig
from miniformer.visualization import plot_attention


# Create a simple time series dataset (synthetic data)
class TimeSeriesDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=24, feature_dim=5, targets=1):
        """
        Args:
            num_samples: Number of sequences
            seq_len: Length of each sequence
            feature_dim: Number of features per time step
            targets: Number of target values to predict
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.targets = targets
        
        # Generate random time series data
        np.random.seed(42)
        self.data = np.random.randn(num_samples, seq_len, feature_dim).astype(np.float32)
        
        # Generate target values (simple function of the input)
        # Here we'll use the mean across feature dimensions plus some noise
        self.labels = np.mean(self.data, axis=2, keepdims=True) + 0.1 * np.random.randn(num_samples, seq_len, targets).astype(np.float32)
    
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx])
        y = torch.tensor(self.labels[idx])
        return x, y


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create dataset
    train_dataset = TimeSeriesDataset(num_samples=800)
    val_dataset = TimeSeriesDataset(num_samples=200)
    
    # Get dataset parameters
    sample_x, sample_y = train_dataset[0]
    seq_len = sample_x.shape[0]
    input_dim = sample_x.shape[1]
    output_dim = sample_y.shape[1]
    
    print(f"Sequence length: {seq_len}")
    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    
    # Create model configuration
    config = TransformerConfig(
        input_dim=input_dim,  # Use input_dim since we're working with features, not tokens
        d_model=input_dim,    # Match d_model with input_dim for direct use
        output_dim=output_dim,
        n_heads=2,
        n_layers=2,
        d_ff=32,
        dropout=0.1,
        batch_size=16,
        learning_rate=1e-4
    )
    
    # Create transformer model
    model = Transformer(config)
    print(model)
    
    # Create trainer
    trainer = TransformerTrainer(
        model,
        config=config,
        task_type="regression"  # Important: set task type to regression
    )
    
    # Train model
    history = trainer.train(
        train_dataset=train_dataset,
        epochs=10,
        validation_dataset=val_dataset,
        eval_steps=50,
        save_steps=100,
        checkpoint_dir="./time_series_checkpoints"
    )
    
    # Visualize predictions
    model.eval()
    x, y_true = val_dataset[0]
    x = x.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        y_pred = model(x)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(y_true.numpy(), label='True')
    plt.plot(y_pred[0].numpy(), label='Predicted')
    plt.title('Time Series Prediction')
    plt.legend()
    plt.savefig('time_series_prediction.png')
    print("Plot saved to time_series_prediction.png")
    
    # Get attention weights and visualize
    attention_weights = model.get_attention_weights(x)
    fig, _ = plot_attention(attention_weights, layer=0, head=0)
    plt.savefig('time_series_attention.png')
    print("Attention visualization saved to time_series_attention.png")


if __name__ == "__main__":
    main()
