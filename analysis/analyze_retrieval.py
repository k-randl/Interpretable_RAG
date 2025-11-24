import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from analyze_results import load_pickle, analyze_retrieval_results

def plot_query_heatmap(analysis, output_path):
    """Generates a heatmap for query token importance."""
    if 'intGrad' not in analysis or 'query_score' not in analysis['intGrad']:
        print("No query scores found for heatmap.")
        return

    scores = analysis['intGrad']['query_score']
    tokens = analysis.get('query_tokens', [])
    
    # Ensure 1D array
    if scores.ndim > 1:
        scores = scores.flatten()
        
    # Truncate/Pad if mismatch
    min_len = min(len(scores), len(tokens))
    scores = scores[:min_len]
    tokens = tokens[:min_len]
    
    plt.figure(figsize=(12, 2))
    # Use matplotlib imshow
    plt.imshow([scores], cmap='RdBu_r', aspect='auto')
    plt.colorbar(label='Integrated Gradients')
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    plt.yticks([])
    plt.title('Query Token Importance')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved query heatmap to {output_path}")

def plot_context_heatmap(analysis, output_path):
    """Generates a heatmap for context token importance (Documents x Tokens)."""
    if 'intGrad' not in analysis or 'context_score' not in analysis['intGrad']:
        print("No context scores found for heatmap.")
        return

    scores = analysis['intGrad']['context_score']
    # scores should be [num_docs, seq_len]
    
    if scores.ndim != 2:
        print(f"Context scores shape mismatch: {scores.shape}, expected 2D.")
        return

    plt.figure(figsize=(14, 8))
    plt.imshow(scores, cmap='RdBu_r', aspect='auto')
    plt.colorbar(label='Integrated Gradients')
    plt.xlabel('Token Position')
    plt.ylabel('Document Index')
    plt.title('Context Token Importance (All Retrieved Documents)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved context heatmap to {output_path}")

def plot_token_comparison(analysis, output_path):
    """Plots a comparison of top token importance between Query and Context."""
    if 'intGrad' not in analysis:
        return

    query_agg = analysis['intGrad'].get('query_agg', {})
    context_agg = analysis['intGrad'].get('context_agg', {})
    
    if not query_agg or not context_agg:
        print("Missing aggregated scores for comparison.")
        return

    # Sort by absolute importance sum
    top_n = 15
    sorted_query = sorted(query_agg.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    sorted_context = sorted(context_agg.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    
    q_tokens, q_scores = zip(*sorted_query) if sorted_query else ([], [])
    c_tokens, c_scores = zip(*sorted_context) if sorted_context else ([], [])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Query Plot
    ax1.barh(range(len(q_tokens)), q_scores, color='skyblue')
    ax1.set_yticks(range(len(q_tokens)))
    ax1.set_yticklabels(q_tokens)
    ax1.invert_yaxis()
    ax1.set_title('Top Query Tokens (Importance)')
    ax1.set_xlabel('Aggregated Integrated Gradients')
    
    # Context Plot
    # Highlight tokens present in query
    c_colors = ['orange' if t in q_tokens else 'lightgray' for t in c_tokens]
    
    ax2.barh(range(len(c_tokens)), c_scores, color=c_colors)
    ax2.set_yticks(range(len(c_tokens)))
    ax2.set_yticklabels(c_tokens)
    ax2.invert_yaxis()
    ax2.set_title('Top Context Tokens (Importance)')
    ax2.set_xlabel('Aggregated Integrated Gradients')
    
    plt.suptitle('Token Importance Comparison')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved comparison plot to {output_path}")

def plot_document_relevance(analysis, output_path):
    """Plots the total importance of each retrieved document."""
    if 'intGrad' not in analysis or 'context_score' not in analysis['intGrad']:
        print("No context scores found for document relevance.")
        return

    scores = analysis['intGrad']['context_score']
    # scores: [num_docs, seq_len]
    
    if scores.ndim != 2:
        return

    # Sum absolute importance per document
    doc_importance = np.sum(np.abs(scores), axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(doc_importance) + 1), doc_importance, color='teal')
    plt.xlabel('Document Rank (Retrieval)')
    plt.ylabel('Total Integrated Gradients (Absolute Sum)')
    plt.title('Retrieved Document Relevance')
    plt.xticks(range(1, len(doc_importance) + 1))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved document relevance plot to {output_path}")

def plot_weighted_token_overlap(analysis, output_path):
    """
    Analyzes the overlapping tokens between query and context, 
    weighted by their model-assigned importance.
    """
    if 'intGrad' not in analysis: return

    query_agg = analysis['intGrad'].get('query_agg', {})
    context_agg = analysis['intGrad'].get('context_agg', {})
    
    if not query_agg or not context_agg: return

    # Intersection
    common_tokens = set(query_agg.keys()) & set(context_agg.keys())
    if not common_tokens: return
        
    x_vals, y_vals, labels = [], [], []
    for token in common_tokens:
        x_vals.append(query_agg[token])
        y_vals.append(context_agg[token])
        labels.append(token)
        
    x_vals, y_vals = np.array(x_vals), np.array(y_vals)
    
    plt.figure(figsize=(10, 10))
    plt.scatter(x_vals, y_vals, alpha=0.7, edgecolors='b', s=100)
    
    # Identity line
    lims = [np.min([plt.xlim(), plt.ylim()]), np.max([plt.xlim(), plt.ylim()])]
    plt.plot(lims, lims, 'k--', alpha=0.5)
    
    # Annotate top tokens
    distances = np.sqrt(x_vals**2 + y_vals**2)
    top_indices = np.argsort(distances)[-15:]
    
    for i in top_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)
                     
    plt.xlabel('Query Importance')
    plt.ylabel('Context Importance')
    plt.title('Weighted Token Overlap')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze retrieval results.")
    parser.add_argument("filepath", help="Path to the retrieval pickle file")
    parser.add_argument("--output_dir", default="analysis/plots/retrieval", help="Directory to save plots")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    data = load_pickle(args.filepath)
    
    # Verify it's retrieval data
    if 'input' not in data:
        print("Error: This does not appear to be a retrieval results file (missing 'input' key).")
        return

    print("Analyzing retrieval results...")
    analysis = analyze_retrieval_results(data)
    
    base_name = os.path.basename(args.filepath).replace('.pkl', '')
    
    # 1. Query Heatmap
    plot_query_heatmap(analysis, os.path.join(args.output_dir, f"{base_name}_query_heatmap.png"))
    
    # 2. Context Heatmap
    plot_context_heatmap(analysis, os.path.join(args.output_dir, f"{base_name}_context_heatmap.png"))
    
    # 3. Comparison Plot
    plot_token_comparison(analysis, os.path.join(args.output_dir, f"{base_name}_comparison.png"))

    # 4. Document Relevance
    plot_document_relevance(analysis, os.path.join(args.output_dir, f"{base_name}_doc_relevance.png"))

if __name__ == "__main__":
    main()
