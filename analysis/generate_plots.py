import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
import nltk
from nltk import pos_tag

# --- Token Cleaning ---
def clean_token(token: str) -> str:
    """Cleans a single token by removing special characters and prefixes."""
    special_llama_tokens = ["<s>", "</s>", "<unk>", "[PAD]", "[CLS]", "[SEP]"]
    if token in special_llama_tokens:
        return ""
    if len(token) > 1 and token.startswith('G') and token[1:].islower():
        token = token[1:]
    token = token.replace('Ġ', '')
    token = token.strip()
    return token

# --- Plotting Functions ---
def plot_shapley_heatmap(shapley_values, x_labels, y_labels, title, output_path):
    """Plots a heatmap of Shapley values."""
    fig, ax = plt.subplots(figsize=(len(x_labels) * 0.4, len(y_labels) * 0.4))
    cbar_kws = {'label': 'Shapley Value'}
    x_labels = [str(label) for label in x_labels]
    y_labels = [str(label) for label in y_labels]
    im = ax.imshow(shapley_values, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=90, ha='right', fontsize=8)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, **cbar_kws)
    try:
        plt.tight_layout()
    except Exception as e:
        print(f"Could not apply tight_layout: {e}")
    plt.savefig(output_path)
    plt.close(fig)

def plot_mean_doc_importance(shapley_context, output_path):
    """Plots the mean importance of each document for a single file."""
    if shapley_context is None or shapley_context.size == 0:
        print("Shapley context is empty, cannot plot document importance.")
        return
    mean_importance = np.sum(np.abs(shapley_context), axis=1) / shapley_context.shape[1]
    doc_indices = np.arange(len(mean_importance)) + 1
    fig, ax = plt.subplots()
    ax.bar(doc_indices, mean_importance)
    ax.set_xlabel("Document")
    ax.set_ylabel("Mean Absolute Shapley Value")
    ax.set_title("Mean Importance per Document")
    ax.set_xticks(doc_indices)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def plot_pos_importance(gen_tokens, shapley_query, shapley_context, output_path):
    """Performs POS tagging and plots importance per POS tag."""
    # Sum over input dimensions to get importance per GENERATED token
    q_imp = np.sum(np.abs(shapley_query), axis=1)
    c_imp = np.sum(np.abs(shapley_context), axis=0)
    
    # Handle length mismatch
    min_len = min(len(q_imp), len(c_imp))
    q_imp = q_imp[:min_len]
    c_imp = c_imp[:min_len]
    
    total_incoming_importance = q_imp + c_imp
    pos_tagged_tokens = pos_tag(gen_tokens[:min_len], tagset='universal', lang='eng')
    pos_tags = [tag for token, tag in pos_tagged_tokens]
    min_len = min(len(pos_tags), len(total_incoming_importance))
    pos_tags = pos_tags[:min_len]
    total_incoming_importance = total_incoming_importance[:min_len]
    pos_importance = {}
    for pos, importance in zip(pos_tags, total_incoming_importance):
        if pos not in pos_importance:
            pos_importance[pos] = []
        pos_importance[pos].append(importance)
    mean_pos_importance = {pos: np.mean(values) for pos, values in pos_importance.items()}
    if not mean_pos_importance:
        print("No POS data to plot.")
        return
    sorted_pos = sorted(mean_pos_importance.items(), key=lambda item: item[1], reverse=True)
    pos_names = [item[0] for item in sorted_pos]
    mean_values = [item[1] for item in sorted_pos]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(pos_names, mean_values)
    ax.set_xlabel("Part of Speech (Universal Tagset)")
    ax.set_ylabel("Mean Absolute Shapley Value")
    ax.set_title("Mean Generation Importance by Part of Speech")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def calculate_mean_importance_across_files(pickle_files, output_path):
    """Calculates and plots the mean importance of documents across multiple pickle files."""
    all_document_importances = []
    for p_file in pickle_files:
        with open(p_file, 'rb') as f:
            data = pickle.load(f)
        shapley_context = data['shapley_values_token']['context']
        document_importance_per_file = np.sum(np.abs(shapley_context), axis=1)
        all_document_importances.append(document_importance_per_file)
    if not all_document_importances:
        print("No document importances to plot.")
        return
    max_docs = max(imp.shape[0] for imp in all_document_importances)
    padded_importances = [np.pad(imp, (0, max_docs - imp.shape[0]), 'constant', constant_values=np.nan) if imp.shape[0] < max_docs else imp for imp in all_document_importances]
    all_document_importances_array = np.array(padded_importances)
    mean_importances = np.nanmean(all_document_importances_array, axis=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    document_positions = np.arange(len(mean_importances)) + 1
    ax.bar(document_positions, mean_importances, color='skyblue')
    ax.set_xlabel("Document Position in Prompt")
    ax.set_ylabel("Mean Absolute Shapley Value (Importance)")
    ax.set_title("Mean Document Importance by Position Across Queries")
    ax.set_xticks(document_positions)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Mean document importance plot across all files saved to {output_path}")

# --- File Processing ---
def process_single_file(pickle_file, output_dir):
    """Processes a single pickle file to generate and save all plots."""
    print(f"Processing {pickle_file}...")
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    # Clean tokens
    qry_tokens = [clean_token(t) for t in data['qry_tokens']]
    gen_tokens = [clean_token(t) for t in data['gen_tokens']]
    
    # Get shapley values
    shapley_query = data['shapley_values_token']['query']
    shapley_context = data['shapley_values_token']['context']

    os.makedirs(output_dir, exist_ok=True)
    
    # --- Generate all plots for the single file ---
    # 1. Shapley Heatmaps
    query_heatmap_path = os.path.join(output_dir, "shapley_query_heatmap.png")
    plot_shapley_heatmap(shapley_query, gen_tokens, qry_tokens, "Shapley Values: Query -> Generation", query_heatmap_path)
    
    context_heatmap_path = os.path.join(output_dir, "shapley_context_heatmap.png")
    context_labels = [f"Doc {i+1}" for i in range(shapley_context.shape[0])]
    plot_shapley_heatmap(shapley_context, gen_tokens, context_labels, "Shapley Values: Context -> Generation", context_heatmap_path)

    # 2. Mean Document Importance (for this file)
    doc_importance_path = os.path.join(output_dir, "mean_document_importance.png")
    plot_mean_doc_importance(shapley_context, doc_importance_path)

    # 3. POS Importance
    pos_importance_path = os.path.join(output_dir, "pos_importance.png")
    plot_pos_importance(gen_tokens, shapley_query, shapley_context, pos_importance_path)

    print(f"All plots for {pickle_file} saved to {output_dir}")

# --- Main Execution ---
def main():
    """Main function to parse arguments and run the analysis."""
    # Download necessary NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('universal_tagset', quiet=True)

    parser = argparse.ArgumentParser(description="Generate all plots for RAG pipeline output analysis.")
    parser.add_argument("input_path", type=str, help="Path to a pickle file or a directory of pickle files.")
    parser.add_argument("--output-dir", type=str, default="analysis_plots", help="Main directory to save all plots.")
    args = parser.parse_args()

    # Unified output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isdir(args.input_path):
        pickle_files = list(Path(args.input_path).glob("*.pkl"))
        
        for pickle_file in pickle_files:
            file_specific_output_dir = os.path.join(args.output_dir, pickle_file.stem)
            process_single_file(pickle_file, file_specific_output_dir)
        
        # Calculate and plot mean importance across all files
        if pickle_files:
            mean_importance_plot_path = os.path.join(args.output_dir, "mean_document_importance_across_files.png")
            calculate_mean_importance_across_files(pickle_files, mean_importance_plot_path)

    elif os.path.isfile(args.input_path):
        file_specific_output_dir = os.path.join(args.output_dir, Path(args.input_path).stem)
        process_single_file(args.input_path, file_specific_output_dir)
    else:
        print(f"Error: {args.input_path} is not a valid file or directory.")

if __name__ == "__main__":
    main()
