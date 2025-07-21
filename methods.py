import numpy as np
from numpy.typing import NDArray
from typing import Literal, List, Optional, Tuple
from sklearn.linear_model import LinearRegression
from scipy.special import comb
import torch


def _nucleus_sampling(probs: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """
    Nucleus sampling function - you'll need to implement this based on your specific requirements.
    This is a placeholder implementation.
    """
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find the cutoff point where cumulative probability exceeds p
    cutoff_mask = cumulative_probs <= p
    
    # Create nucleus mask
    nucleus_mask = torch.zeros_like(probs)
    nucleus_mask.scatter_(-1, sorted_indices, cutoff_mask.float())
    
    # Apply nucleus sampling
    nucleus_probs = probs * nucleus_mask
    return nucleus_probs


def compute_nucleus_probs(exp_probs: List[torch.Tensor], p: float = 0.9) -> List[np.ndarray]:
    """Compute nucleus probabilities for comparison inputs."""
    return [_nucleus_sampling(t.float(), p=p).mean(dim=1).numpy() for t in exp_probs]


def compute_sequence_probs(exp_probs: List[torch.Tensor], gen_output: List[List[List[int]]]) -> List[np.ndarray]:
    """Compute sequence probabilities for comparison inputs."""
    return [np.prod([ 
        [float(t[i, j, id]) for j, id in enumerate(seq)]
        for i, seq in enumerate(gen_output)
    ], axis=-1) for t in exp_probs]


def compute_gen_sequence_prob(gen_probs: torch.Tensor, gen_output: List[List[List[int]]]) -> np.ndarray:
    """Compute sequence probability for generated output."""
    return np.prod([ 
        [float(gen_probs[i, j, id]) for j, id in enumerate(seq)]
        for i, seq in enumerate(gen_output)
    ], axis=-1)


def compute_gen_nucleus_probs(gen_probs: torch.Tensor, p: float = 0.9) -> np.ndarray:
    """Compute nucleus probabilities for generated output."""
    return _nucleus_sampling(gen_probs.float(), p=p).mean(dim=1).numpy()


def _get_shapley_values_precise(probs: List[np.ndarray], shap_cache: np.ndarray) -> NDArray[np.float_]:
    """Compute precise Shapley values using all permutations."""
    # Get the shape of the permutations matrix: (num_permutations, num_documents)
    num_permutations, num_docs = shap_cache.shape

    # Initialize array to store marginal contributions for each permutation step
    p_marginal = np.empty(shap_cache.shape + probs[0].shape, dtype=probs[0].dtype)

    # For each permutation, calculate the marginal contributions
    for i in range(num_permutations):
        # First document's contribution is its raw probability
        p_marginal[i, 0] = probs[shap_cache[i, 0]]

        for j in range(1, num_docs):
            # Difference in output probability when adding the j-th document
            prev = probs[shap_cache[i, j - 1]]
            curr = probs[shap_cache[i, j]]
            p_marginal[i, j] = curr - prev

    # Encode the permutation transitions using bitwise operations
    new_doc = np.empty(shap_cache.shape, dtype=np.int16)

    # First position is zeroed (i.e., no document yet included)
    new_doc[:, 0] = 0

    for j in range(1, num_docs):
        # Bitwise difference to capture which bit changed, then take log2
        prev = shap_cache[:, j - 1]
        curr = shap_cache[:, j]
        new_doc[:, j] = np.log2(curr & ~prev) + 1

    # Initialize SHAP value container: one entry per document
    p_shap = np.empty((num_docs,) + probs[0].shape, dtype=probs[0].dtype)

    # For each document, aggregate all matching marginal contributions
    for j in range(num_docs):
        # Mean over all marginal contributions that map to document j
        p_shap[j] = p_marginal[new_doc == j].mean(0)

    # Return SHAP values for all but the baseline (first one)
    return p_shap[1:]


def _get_shapley_values_kernel(probs: List[np.ndarray], shap_cache: np.ndarray) -> NDArray[np.float_]:
    """Compute approximate Shapley values using kernel method."""
    # fit a linear regressor using the SHAP kernel:
    lr = LinearRegression()
    x = shap_cache[1:-1]
    y = np.stack(probs[1:-1])
    w = [(len(z)-1) / (comb(len(z), sum(z)) * sum(z) * -sum(z-1))
         for z in x]
    lr.fit(x, y, w)

    # attributions are estimated SHAP values:
    attributions = lr.coef_.T

    # rescale attributions to fit prediction:
    return attributions / np.abs(attributions.sum(axis=0)) * (probs[-1] - probs[0])


def get_shapley_values(
    exp_probs: List[torch.Tensor],
    gen_probs: torch.Tensor,
    gen_output: List[List[List[int]]],
    shap_cache: np.ndarray,
    aggregation: Literal['token', 'sequence', 'bow', 'nucleus'] = 'token',
    shap_precise: bool = True,
    **kwargs
) -> NDArray[np.float_]:
    """
    Generates Shapley feature attribution values for the chosen aggregation method.

    Args:
        exp_probs: List of probability tensors from compared documents
        gen_probs: Probability tensor from generated output
        gen_output: Generated output token sequences
        shap_cache: Cached permutation matrix for SHAP computation
        aggregation: Aggregation method for probabilities (default: 'token')
        shap_precise: Whether to use precise or kernel-based SHAP computation
        **kwargs: Additional keyword arguments (e.g., 'p' for nucleus sampling)

    Returns:
        A numpy.ndarray containing the Shapley values.
    """
    
    # Validate inputs
    if len(exp_probs) == 0:
        raise ValueError("exp_probs cannot be empty. Run comparison first.")
    
    if gen_probs.numel() == 0:
        raise ValueError("gen_probs cannot be empty. Run generation first.")

    # Get the correct probabilities based on the aggregation parameter
    if aggregation == 'nucleus':
        # Get nucleus probabilities for compared documents
        cmp_probs = compute_nucleus_probs(exp_probs, **kwargs)
        
        # Get nucleus probabilities for generated output
        gen_prob = compute_gen_nucleus_probs(gen_probs, **kwargs)
        
        # Flatten each token probability array from compared documents
        probs = [p.flatten() for p in cmp_probs]
        
        # Add the generated token probabilities as the final "player" in the SHAP context
        probs.append(gen_prob.flatten())

    elif aggregation == 'sequence':
        # Get sequence probabilities for compared documents
        cmp_probs = compute_sequence_probs(exp_probs, gen_output)
        
        # Get sequence probability for generated output
        gen_prob = compute_gen_sequence_prob(gen_probs, gen_output)
        
        # Convert the scalar probability from compared documents to a ndarray
        probs = [np.array(p) for p in cmp_probs]
        
        # Add the generated token probabilities as the final "player" in the SHAP context
        probs.append(np.array(gen_prob))

    elif aggregation in ['token', 'bow']:
        # For token and bow aggregations, assume they work similarly to nucleus
        # You may need to implement specific functions for these if they differ
        if aggregation == 'token':
            # Flatten each token probability array from compared documents
            probs = [p.mean(dim=1).numpy().flatten() for p in exp_probs]
            # Add the generated token probabilities as the final "player"
            probs.append(gen_probs.mean(dim=1).numpy().flatten())
        elif aggregation == 'bow':
            # Bag of words aggregation - implement as needed
            probs = [p.sum(dim=1).numpy().flatten() for p in exp_probs]
            probs.append(gen_probs.sum(dim=1).numpy().flatten())

    else:
        raise ValueError(f'Unknown value for parameter `aggregation`: "{aggregation}"')

    # Call the appropriate SHAP computation method
    if shap_precise:
        return _get_shapley_values_precise(probs, shap_cache)
    else:
        return _get_shapley_values_kernel(probs, shap_cache)



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from typing import List, Optional, Union, Tuple
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


def load_shap_cache(pickle_path: str) -> np.ndarray:
    """Load SHAP cache from pickle file."""
    with open(pickle_path, 'rb') as f:
        shap_cache = pickle.load(f)
    return shap_cache


def plot_shap_attributions(
    shap_values: np.ndarray,
    document_names: Optional[List[str]] = None,
    aggregation: str = 'token',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show_values: bool = True,
    cmap: str = 'RdBu_r'
) -> plt.Figure:
    """
    Plot SHAP attributions with different visualization styles based on aggregation type.
    
    Args:
        shap_values: SHAP attribution values from get_shapley_values()
        document_names: Names of the documents being compared
        aggregation: Type of aggregation used ('token', 'sequence', 'bow', 'nucleus')
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
        show_values: Whether to show values on the plot
        cmap: Colormap for the plot
    
    Returns:
        matplotlib Figure object
    """
    
    # Create document names if not provided
    if document_names is None:
        document_names = [f'Document {i+1}' for i in range(len(shap_values))]
    
    # Set default title based on aggregation
    if title is None:
        title = f'SHAP Attributions - {aggregation.capitalize()} Aggregation'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if aggregation == 'sequence':
        # For sequence aggregation, shap_values should be 1D
        if shap_values.ndim == 1:
            # Bar plot for sequence-level attributions
            colors = ['red' if x < 0 else 'blue' for x in shap_values]
            bars = ax.bar(document_names, shap_values, color=colors, alpha=0.7)
            
            # Add value labels on bars
            if show_values:
                for bar, value in zip(bars, shap_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                           f'{value:.4f}', ha='center', va='bottom' if height >= 0 else 'top')
            
            ax.set_ylabel('SHAP Attribution')
            ax.set_xlabel('Documents')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
        else:
            # Multiple sequences - use heatmap
            im = ax.imshow(shap_values.T, cmap=cmap, aspect='auto')
            ax.set_xticks(range(len(document_names)))
            ax.set_xticklabels(document_names, rotation=45, ha='right')
            ax.set_ylabel('Sequence Index')
            plt.colorbar(im, ax=ax, label='SHAP Attribution')
    
    elif aggregation in ['token', 'bow', 'nucleus']:
        # For token-level aggregations, use heatmap
        if shap_values.ndim == 1:
            # Single dimension - convert to row vector for visualization
            shap_values = shap_values.reshape(1, -1)
            
        # Create heatmap
        im = ax.imshow(shap_values, cmap=cmap, aspect='auto')
        
        # Set labels
        ax.set_xticks(range(len(document_names)))
        ax.set_xticklabels(document_names, rotation=45, ha='right')
        
        if shap_values.shape[0] > 1:
            ax.set_ylabel('Token/Feature Index')
        else:
            ax.set_yticks([0])
            ax.set_yticklabels(['Attribution'])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='SHAP Attribution')
        
        # Add text annotations if requested and data is small enough
        if show_values and shap_values.size <= 100:
            for i in range(shap_values.shape[0]):
                for j in range(shap_values.shape[1]):
                    text = ax.text(j, i, f'{shap_values[i, j]:.3f}',
                                 ha="center", va="center", color="white" if abs(shap_values[i, j]) > 0.5 else "black")
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    document_names: Optional[List[str]] = None,
    aggregation: str = 'token',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a summary plot showing mean absolute SHAP values per document.
    
    Args:
        shap_values: SHAP attribution values
        document_names: Names of the documents
        aggregation: Type of aggregation used
        figsize: Figure size
        save_path: Path to save the plot
    
    Returns:
        matplotlib Figure object
    """
    
    if document_names is None:
        document_names = [f'Document {i+1}' for i in range(len(shap_values))]
    
    # Calculate mean absolute attributions per document
    if shap_values.ndim == 1:
        mean_abs_attributions = np.abs(shap_values)
    else:
        mean_abs_attributions = np.mean(np.abs(shap_values), axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    bars = ax.bar(document_names, mean_abs_attributions, 
                  color='skyblue', alpha=0.7, edgecolor='navy')
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_abs_attributions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
               f'{value:.4f}', ha='center', va='bottom')
    
    ax.set_ylabel('Mean Absolute SHAP Attribution')
    ax.set_xlabel('Documents')
    ax.set_title(f'SHAP Attribution Summary - {aggregation.capitalize()} Aggregation', 
                fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_shap_waterfall(
    shap_values: np.ndarray,
    document_names: Optional[List[str]] = None,
    base_value: float = 0.0,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a waterfall plot showing cumulative SHAP contributions.
    
    Args:
        shap_values: SHAP attribution values (1D array)
        document_names: Names of the documents
        base_value: Base value to start the waterfall from
        figsize: Figure size
        save_path: Path to save the plot
    
    Returns:
        matplotlib Figure object
    """
    
    if shap_values.ndim > 1:
        # If multidimensional, take mean across other dimensions
        shap_values = np.mean(shap_values, axis=tuple(range(1, shap_values.ndim)))
    
    if document_names is None:
        document_names = [f'Document {i+1}' for i in range(len(shap_values))]
    
    # Calculate cumulative values
    cumulative = np.cumsum(np.concatenate([[base_value], shap_values]))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot base value
    ax.bar(0, base_value, color='gray', alpha=0.7, label='Base Value')
    
    # Plot each contribution
    for i, (name, value) in enumerate(zip(document_names, shap_values)):
        color = 'green' if value > 0 else 'red'
        ax.bar(i + 1, value, bottom=cumulative[i], color=color, alpha=0.7)
        
        # Add value label
        ax.text(i + 1, cumulative[i] + value/2, f'{value:.4f}', 
               ha='center', va='center', fontweight='bold', color='white')
    
    # Plot final value
    ax.bar(len(document_names) + 1, cumulative[-1], color='blue', alpha=0.7, label='Final Value')
    
    # Connect bars with lines
    for i in range(len(document_names)):
        ax.plot([i + 0.4, i + 1.6], [cumulative[i+1], cumulative[i+1]], 'k--', alpha=0.5)
    
    # Set labels
    ax.set_xticks(range(len(document_names) + 2))
    ax.set_xticklabels(['Base'] + document_names + ['Final'], rotation=45, ha='right')
    ax.set_ylabel('Cumulative Attribution')
    ax.set_title('SHAP Waterfall Plot', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_multiple_aggregations(
    shap_results: dict,
    document_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot SHAP attributions for multiple aggregation methods in subplots.
    
    Args:
        shap_results: Dictionary with aggregation method as key and SHAP values as value
        document_names: Names of the documents
        figsize: Figure size
        save_path: Path to save the plot
    
    Returns:
        matplotlib Figure object
    """
    
    n_methods = len(shap_results)
    fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=figsize)
    
    if n_methods == 1:
        axes = [axes]
    elif n_methods <= 2:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, (method, shap_values) in enumerate(shap_results.items()):
        ax = axes[idx]
        
        if document_names is None:
            doc_names = [f'Doc {i+1}' for i in range(len(shap_values))]
        else:
            doc_names = document_names
        
        if method == 'sequence' and shap_values.ndim == 1:
            # Bar plot for sequence
            colors = ['red' if x < 0 else 'blue' for x in shap_values]
            ax.bar(doc_names, shap_values, color=colors, alpha=0.7)
            ax.set_ylabel('SHAP Attribution')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        else:
            # Heatmap for other methods
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)
            
            im = ax.imshow(shap_values, cmap='RdBu_r', aspect='auto')
            ax.set_xticks(range(len(doc_names)))
            ax.set_xticklabels(doc_names, rotation=45, ha='right')
            
            if shap_values.shape[0] > 1:
                ax.set_ylabel('Feature Index')
            else:
                ax.set_yticks([0])
                ax.set_yticklabels(['Attribution'])
        
        ax.set_title(f'{method.capitalize()} Aggregation', fontweight='bold')
    
    # Hide empty subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# Example usage
if __name__ == "__main__":
    # Load SHAP cache
    # shap_cache = load_shap_cache('path/to/your/shap_cache.pickle')
    
    # Example SHAP values (replace with your actual values)
    # shap_values = np.random.randn(5, 100)  # 5 documents, 100 features
    # document_names = ['Document A', 'Document B', 'Document C', 'Document D', 'Document E']
    
    # Create plots
    # fig1 = plot_shap_attributions(shap_values, document_names, aggregation='token')
    # fig2 = plot_shap_summary(shap_values, document_names, aggregation='token')
    # fig3 = plot_shap_waterfall(np.mean(shap_values, axis=1), document_names)
    
    # plt.show()
    
    pass