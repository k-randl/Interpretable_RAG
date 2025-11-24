import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import nltk
from nltk import pos_tag

def load_pickle_files(directory):
    """Loads all pickle files from a directory."""
    files = list(Path(directory).glob("*.pkl"))
    data_list = []
    for f in files:
        try:
            with open(f, 'rb') as pkl:
                data = pickle.load(pkl)
                data_list.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return data_list

def analyze_position_bias(data_list, label):
    """Calculates mean importance per document position."""
    doc_importances = []
    
    for data in data_list:
        if 'shapley_values_token' not in data or 'context' not in data['shapley_values_token']:
            continue
            
        # Shape: [num_gen_tokens, num_docs] (assuming context shapley is aggregated per doc or we need to aggregate)
        # Actually, context shapley is usually [num_gen_tokens, total_context_tokens]
        # We need to know which tokens belong to which document.
        # IF the pickle doesn't have doc boundaries, we might be in trouble.
        # BUT, generate_plots.py assumed: shapley_context = data['shapley_values_token']['context']
        # and did: np.sum(np.abs(shapley_context), axis=1) -> this was per generated token?
        # Wait, generate_plots.py line 97: document_importance_per_file = np.sum(np.abs(shapley_context), axis=1)
        # If shapley_context is [num_gen, num_docs], then axis=1 sum is total importance per gen token.
        # If shapley_context is [num_docs, num_gen] (transposed?), let's check.
        
        # In generate_plots.py:
        # plot_shapley_heatmap(shapley_context, gen_tokens, context_labels, ...)
        # context_labels = [f"Doc {i+1}" for i in range(shapley_context.shape[0])]
        # So shapley_context shape[0] is num_docs!
        # So it is [num_docs, num_gen_tokens] (or similar).
        
        shap_context = data['shapley_values_token']['context']
        
        # We want total importance of each document across all generated tokens.
        # Sum absolute values across generated tokens (axis=1)
        # Result: [num_docs]
        total_doc_imp = np.sum(np.abs(shap_context), axis=1)
        
        # Normalize by number of generated tokens to be comparable? Or just raw sum?
        # Let's use mean absolute importance per generated token to be safe against length.
        mean_doc_imp = np.mean(np.abs(shap_context), axis=1)
        
        doc_importances.append(mean_doc_imp)
        
    if not doc_importances:
        return None, None

    # Stack: [num_files, num_docs]
    # Pad if necessary (though usually fixed k=6)
    max_docs = max(len(x) for x in doc_importances)
    padded = np.zeros((len(doc_importances), max_docs))
    for i, x in enumerate(doc_importances):
        padded[i, :len(x)] = x
        
    mean_across_files = np.mean(padded, axis=0)
    std_across_files = np.std(padded, axis=0)
    
    return mean_across_files, std_across_files

def analyze_global_pos(data_list):
    """Aggregates POS importance across all files."""
    pos_importance = {}
    pos_counts = {}
    
    for data in data_list:
        if 'gen_tokens' not in data or 'shapley_values_token' not in data:
            continue
            
        gen_tokens = data['gen_tokens']
        # Total importance received by each generated token from Query + Context
        # Query: [num_gen, num_query] -> sum(axis=1) -> [num_gen]
        # Context: [num_docs, num_gen] -> sum(axis=0) -> [num_gen] (Wait, check shape again)
        
        # Re-check shape from generate_plots.py logic:
        # shapley_query is [num_gen, num_query] (transposed in plot? No, in plot it does imshow(shapley_values))
        # In plot_shapley_heatmap: x_labels (gen_tokens), y_labels (qry_tokens).
        # So shape is [num_qry, num_gen] ??
        # Let's look at visualize_results.py line 92: plot_data = shap_values.T # Now (64, 4) where rows=gen, cols=qry.
        # So original shap_values was [num_qry, num_gen].
        
        # Let's assume:
        # Query: [num_qry, num_gen]
        # Context: [num_docs, num_gen]
        
        s_query = data['shapley_values_token']['query']
        s_context = data['shapley_values_token']['context']
        
        # Total importance per generated token
        # Sum absolute values from all sources
        total_imp = np.sum(np.abs(s_query), axis=0) + np.sum(np.abs(s_context), axis=0)
        
        # Clean tokens for POS tagging
        clean_tokens = [t.replace('Ġ', '').strip() for t in gen_tokens]
        
        # POS Tag
        tags = pos_tag(clean_tokens, tagset='universal')
        
        for (token, tag), imp in zip(tags, total_imp):
            if tag not in pos_importance:
                pos_importance[tag] = 0.0
                pos_counts[tag] = 0
            pos_importance[tag] += imp
            pos_counts[tag] += 1
            
    # Average importance per tag
    avg_pos_importance = {k: v / pos_counts[k] for k, v in pos_importance.items()}
    return avg_pos_importance

def main():
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('universal_tagset', quiet=True)
    
    parser = argparse.ArgumentParser(description="Global analysis of generation results.")
    parser.add_argument("base_dir", help="Base directory containing 'original' and 'randomized' subfolders (e.g., results/generation/efra...)")
    parser.add_argument("--output_dir", default="analysis/plots/global", help="Directory to save plots")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    original_dir = os.path.join(args.base_dir, "original")
    randomized_dir = os.path.join(args.base_dir, "randomized")
    
    print(f"Loading original from {original_dir}...")
    orig_data = load_pickle_files(original_dir)
    print(f"Loaded {len(orig_data)} files.")
    
    print(f"Loading randomized from {randomized_dir}...")
    rand_data = load_pickle_files(randomized_dir)
    print(f"Loaded {len(rand_data)} files.")
    
    # --- 1. Position Bias Analysis ---
    print("Analyzing Position Bias...")
    orig_mean, orig_std = analyze_position_bias(orig_data, "Original")
    rand_mean, rand_std = analyze_position_bias(rand_data, "Randomized")
    
    if orig_mean is not None:
        plt.figure(figsize=(10, 6))
        x = range(1, len(orig_mean) + 1)
        
        plt.errorbar(x, orig_mean, yerr=orig_std, label='Original Order', capsize=5, marker='o')
        if rand_mean is not None:
             plt.errorbar(x, rand_mean, yerr=rand_std, label='Randomized Order', capsize=5, marker='x')
             
        plt.xlabel('Document Position')
        plt.ylabel('Mean Absolute Importance')
        plt.title('Document Importance by Position: Position Bias vs Content')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "position_bias_comparison.png"))
        plt.close()
        print("Saved position bias plot.")

    # --- 2. Global POS Analysis ---
    print("Analyzing Global POS Importance...")
    # Combine data for POS analysis (or just use original)
    pos_imp = analyze_global_pos(orig_data)
    
    if pos_imp:
        sorted_pos = sorted(pos_imp.items(), key=lambda x: x[1], reverse=True)
        tags, scores = zip(*sorted_pos)
        
        plt.figure(figsize=(10, 6))
        plt.bar(tags, scores, color='purple')
        plt.xlabel('POS Tag')
        plt.ylabel('Mean Absolute Importance')
        plt.title('Global POS Importance (Original Order)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "global_pos_importance.png"))
        plt.close()
        print("Saved global POS plot.")

if __name__ == "__main__":
    main()
