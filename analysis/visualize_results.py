#%%
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from analyze_results import load_pickle, analyze_retrieval_results, analyze_generation_results

def plot_retrieval_heatmap(analysis, output_path):
    """Generates a heatmap for retrieval importance scores."""
    if 'intGrad' not in analysis:
        print("No Integrated Gradients found for retrieval.")
        return
    
    # We'll plot query importance for now as an example
    # intGrad query score is likely [seq_len] or [batch, seq_len]
    # based on previous analysis it was 1D or 2D.
    
    scores = analysis['intGrad'].get('query_score')
    tokens = analysis.get('query_tokens')
    
    if scores is None or tokens is None:
        print("Missing scores or tokens for retrieval plot.")
        return
        
    # Ensure scores match tokens
    if len(scores) != len(tokens):
        print(f"Mismatch: {len(scores)} scores vs {len(tokens)} tokens. Truncating/Padding.")
        min_len = min(len(scores), len(tokens))
        scores = scores[:min_len]
        tokens = tokens[:min_len]
        
    plt.figure(figsize=(12, 2))
    plt.imshow([scores], cmap='RdBu_r', aspect='auto')
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    plt.yticks([])
    plt.colorbar(label='Importance')
    plt.title('Query Token Importance (Integrated Gradients)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved retrieval plot to {output_path}")

    # Plot Context Heatmap (Docs x Tokens)
    c_scores = analysis['intGrad'].get('context_score')
    # c_scores is [10, 147] (flattened in analyze_results? Let's check)
    # In analyze_results, I flattened it: analysis[key][f'{part}_score'] = val (if 2D) -> wait, I changed it to val.flatten() in one branch?
    # Let's check analyze_results.py content first to be sure.
    # If it is flattened, I can't easily plot heatmap [10, 147].
    # I should probably modify analyze_results to keep the shape for visualization.
    
    # Let's assume I need to fix analyze_results first if it flattens it.
    # But for now, let's write the plotting code assuming I have the 2D array.
    
    if c_scores is not None and len(c_scores.shape) == 2:
        plt.figure(figsize=(12, 6))
        # Plot heatmap
        plt.imshow(c_scores, cmap='RdBu_r', aspect='auto')
        plt.colorbar(label='Importance')
        plt.xlabel('Token Position')
        plt.ylabel('Document Index')
        plt.title('Context Token Importance (All Retrieved Documents)')
        plt.tight_layout()
        context_output_path = output_path.replace('retrieval_heatmap', 'context_heatmap')
        plt.savefig(context_output_path)
        plt.close()
        print(f"Saved context plot to {context_output_path}")

def plot_generation_heatmap(analysis, output_path):
    """Generates a heatmap for generation Shapley values."""
    if 'shapley_values_token' not in analysis:
        print("No Shapley values found for generation.")
        return
        
    # Plot query contribution to generated tokens
    # Shape: [num_gen_tokens, num_query_tokens]
    shap_values = analysis['shapley_values_token'].get('query')
    gen_tokens = analysis.get('gen_tokens')
    qry_tokens = analysis.get('qry_tokens')
    
    if shap_values is None or gen_tokens is None:
        print("Missing data for generation plot.")
        return
        
    # shap_values might be a list of arrays or a big array
    # If it's a dict (as seen in inspection sample), we need to check structure again.
    # Inspection said: Key: shapley_values_token, Type: <class 'dict'>, Sample: {'query': array...}
    # So it is an array.
    
    # Shape is (num_query, num_gen) based on check_shapes.py
    # We want rows to be generated tokens (64) and cols to be query tokens (4)
    # So we transpose it.
    plot_data = shap_values.T # Now (64, 4)
    
    # Subset for readability if too large
    max_gen = 64 # Show all generated tokens if possible, or truncate
    max_qry = 20
    
    plot_data = plot_data[:max_gen, :max_qry]
    y_labels = gen_tokens[:max_gen]
    x_labels = qry_tokens[:max_qry] if qry_tokens else range(plot_data.shape[1])
    
    plt.figure(figsize=(10, 8))
    plt.imshow(plot_data, cmap='RdBu_r', aspect='auto')
    plt.colorbar(label='Shapley Value')
    
    if qry_tokens:
        plt.xticks(range(len(x_labels)), x_labels, rotation=90)
    plt.yticks(range(len(y_labels)), y_labels)
    
    plt.xlabel('Query Tokens')
    plt.ylabel('Generated Tokens')
    plt.title('Token-level Shapley Values (Query Contribution)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved generation plot to {output_path}")

    plt.close()
    print(f"Saved generation plot to {output_path}")

def plot_token_comparison(analysis, output_path):
    """Plots a comparison of token importance between Query and Context."""
    if 'intGrad' not in analysis:
        return

    # Get aggregated scores
    query_agg = analysis['intGrad'].get('query_agg', {})
    context_agg = analysis['intGrad'].get('context_agg', {})
    
    if not query_agg or not context_agg:
        print("Missing aggregated scores for comparison.")
        return
        
    # Sort and take top N
    top_n = 10
    sorted_query = sorted(query_agg.items(), key=lambda x: x[1], reverse=True)[:top_n]
    sorted_context = sorted(context_agg.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Prepare data for plotting
    q_tokens, q_scores = zip(*sorted_query) if sorted_query else ([], [])
    c_tokens, c_scores = zip(*sorted_context) if sorted_context else ([], [])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Query Plot
    ax1.barh(range(len(q_tokens)), q_scores, color='skyblue')
    ax1.set_yticks(range(len(q_tokens)))
    ax1.set_yticklabels(q_tokens)
    ax1.invert_yaxis()
    ax1.set_title('Top Query Tokens (Importance)')
    
    # Context Plot
    # Highlight tokens that are also in top query
    c_colors = ['orange' if t in q_tokens else 'lightgray' for t in c_tokens]
    
    ax2.barh(range(len(c_tokens)), c_scores, color=c_colors)
    ax2.set_yticks(range(len(c_tokens)))
    ax2.set_yticklabels(c_tokens)
    ax2.invert_yaxis()
    ax2.set_title('Top Context Tokens (Importance)')
    
    plt.suptitle('Token Importance Comparison: Query vs Retrieved Context')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved comparison plot to {output_path}")
#%%
def main():
    parser = argparse.ArgumentParser(description="Visualize RAG pipeline results.")
    parser.add_argument("filepath", help="Path to the pickle file to visualize")
    parser.add_argument("--output_dir", default="analysis/plots", help="Directory to save plots")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    data = load_pickle(args.filepath)
    base_name = os.path.basename(args.filepath).replace('.pkl', '')
    
    # Determine type (reuse logic from analyze_results or just try both)
    # Simple heuristic:
    if 'gen_tokens' in data:
        print("Detected generation results.")
        analysis = analyze_generation_results(data)
        output_path = os.path.join(args.output_dir, f"{base_name}_generation_heatmap.png")
        plot_generation_heatmap(analysis, output_path)
    elif 'input' in data:
        print("Detected retrieval results.")
        analysis = analyze_retrieval_results(data)
        output_path = os.path.join(args.output_dir, f"{base_name}_retrieval_heatmap.png")
        plot_retrieval_heatmap(analysis, output_path)
        
        # Add comparison plot
        comp_output_path = os.path.join(args.output_dir, f"{base_name}_comparison.png")
        plot_token_comparison(analysis, comp_output_path)
    else:
        print("Unknown data format.")

if __name__ == "__main__":
    main()
