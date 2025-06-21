import torch
import matplotlib.pyplot as plt
from miniformer.model.transformer import Transformer
from miniformer.visualization import plot_attention

# Create a small transformer model
model = Transformer(
    vocab_size=1000,
    d_model=64,
    n_heads=4,
    n_layers=3,
    d_ff=256,
    dropout=0.1
)

# Create some sample data
batch_size = 2
seq_length = 10
input_ids = torch.randint(1, 1000, (batch_size, seq_length))

# Forward pass
outputs = model(input_ids)
print(f"Input shape: {input_ids.shape}")
print(f"Output shape: {outputs.shape}")

# Visualize attention patterns
attention_weights = model.get_attention_weights(input_ids)
fig, _ = plot_attention(attention_weights, layer=0, head=0)
plt.show()

print("Transformer model created and working!")
