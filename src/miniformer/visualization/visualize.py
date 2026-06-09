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
    if not attention_weights:
        raise ValueError("attention_weights must contain at least one layer")
    if layer < 0 or layer >= len(attention_weights):
        raise IndexError(f"layer index {layer} out of range for {len(attention_weights)} layers")
    layer_weights = attention_weights[layer]
    if layer_weights is None:
        raise ValueError("attention weights are unavailable when use_sdpa=True")
    if not isinstance(layer_weights, torch.Tensor):
        raise TypeError("attention_weights entries must be torch.Tensor instances")
    if layer_weights.dim() != 4:
        raise ValueError("attention weights must have shape [batch, heads, query, key]")
    if head < 0 or head >= layer_weights.size(1):
        raise IndexError(f"head index {head} out of range for {layer_weights.size(1)} heads")

    attn = layer_weights[0, head].cpu().detach().numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn, cmap="viridis")

    # Add colorbar
    ax.figure.colorbar(im, ax=ax)

    # Set title and labels
    ax.set_title(f"Attention weights - Layer {layer+1}, Head {head+1}")

    # Set tick labels if tokens are provided
    if tokens is not None:
        if len(tokens) != attn.shape[-1]:
            raise ValueError("tokens length must match the attention key dimension")
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens)
        ax.set_yticklabels(tokens)

        # Rotate x tick labels and set alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Show the plot
    plt.tight_layout()
    return fig, ax


def visualize_embeddings(model, vocab, method="pca"):
    """Visualize token embeddings using dimensionality reduction"""
    import sklearn.decomposition as decomposition
    import sklearn.manifold as manifold

    # Get embeddings from model
    token_embedding = model.encoder.token_embedding
    if token_embedding is None:
        raise ValueError("visualize_embeddings requires a token-based model")
    embeddings = token_embedding.weight.cpu().detach().numpy()
    if len(vocab) > embeddings.shape[0]:
        raise ValueError("vocab contains more entries than the model embedding table")

    # Apply dimensionality reduction
    method = method.lower()
    if method == "pca":
        reducer = decomposition.PCA(n_components=2)
    elif method in {"tsne", "t-sne"}:
        reducer = manifold.TSNE(n_components=2)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    reduced_embeddings = reducer.fit_transform(embeddings)

    # Plot embeddings
    plt.figure(figsize=(12, 10))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)

    # Add labels for some tokens
    for i, word in enumerate(vocab):
        if i < 100:  # Only show first N tokens for clarity
            plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

    plt.title(f"Token Embeddings visualized using {method.upper()}")
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()
