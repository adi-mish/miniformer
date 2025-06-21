import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_attention(attention_weights, layer=0, head=0, tokens=None):
    """
    Plot attention patterns from transformer
    
    Args:
        attention_weights: List of attention weights from model (one per layer)
        layer: Layer to visualize
        head: Attention head to visualize
        tokens: Optional list of token strings for axis labels
    """
    # Get attention from specified layer and head
    attn = attention_weights[layer][0, head].cpu().detach().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn, cmap='viridis')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    
    # Set title and labels
    ax.set_title(f"Attention weights - Layer {layer+1}, Head {head+1}")
    
    # Set tick labels if tokens are provided
    if tokens is not None:
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens)
        ax.set_yticklabels(tokens)
        
        # Rotate x tick labels and set alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Show the plot
    plt.tight_layout()
    return fig, ax


def visualize_embeddings(model, vocab, method='pca'):
    """Visualize token embeddings using dimensionality reduction"""
    import sklearn.decomposition as decomposition
    import sklearn.manifold as manifold
    
    # Get embeddings from model
    embeddings = model.encoder.token_embedding.embedding.weight.cpu().detach().numpy()
    
    # Apply dimensionality reduction
    if method == 'pca':
        reducer = decomposition.PCA(n_components=2)
    else:  # default to t-SNE
        reducer = manifold.TSNE(n_components=2)
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Plot embeddings
    plt.figure(figsize=(12, 10))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
    
    # Add labels for some tokens
    for i, word in enumerate(vocab):
        if i < 100:  # Only show first N tokens for clarity
            plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
    
    plt.title(f'Token Embeddings visualized using {method.upper()}')
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()
