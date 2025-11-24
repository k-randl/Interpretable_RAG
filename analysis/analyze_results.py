import pickle
import torch
import numpy as np
import os
import argparse
from typing import Dict, Any, List, Optional

def load_pickle(filepath: str) -> Any:
    """Loads a pickle file."""
    print(f"Loading {filepath}...")
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def analyze_retrieval_results(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyzes retrieval results (gradients, integrated gradients)."""
    analysis = {}
    
    # Extract input tokens
    if 'input' in data:
        analysis['query_tokens'] = data['input']['query'][0] # Query is usually single [1, seq_len]
        # Context is [num_docs, seq_len]
        # Flatten context tokens for aggregation
        context_tokens_list = data['input']['context'] # List of lists
        analysis['context_tokens_list'] = context_tokens_list
        analysis['context_tokens'] = [token for doc in context_tokens_list for token in doc]
    
    # Analyze importance scores
    score_keys = ['grad', 'intGrad', 'gradIn', 'aGrad']
    for key in score_keys:
        if key in data:
            analysis[key] = {}
            for part in ['query', 'context']:
                if part in data[key]:
                    val = data[key][part]
                    if isinstance(val, torch.Tensor):
                        val = val.detach().cpu().numpy()
                    
                    scores_for_agg = None
                    
                    # If it's 3D [batch, seq, dim], aggregate over dim
                    if val.ndim == 3:
                        val_agg = np.linalg.norm(val, axis=-1) # L2 norm over embedding dim
                        # If query, take [0]. If context, keep all docs [num_docs, seq_len]
                        if part == 'query':
                             analysis[key][f'{part}_l2'] = val_agg[0]
                             scores_for_agg = val_agg[0]
                        else:
                             analysis[key][f'{part}_l2'] = val_agg
                             scores_for_agg = val_agg.flatten()
                             
                    elif val.ndim == 2:
                        # shape [batch/num_docs, seq_len]
                        if part == 'query':
                            analysis[key][f'{part}_score'] = val[0]
                            scores_for_agg = val[0]
                        else:
                            # Keep 2D shape for heatmap
                            analysis[key][f'{part}_score'] = val
                            scores_for_agg = val.flatten()
                            
                            # Also store per-document scores if needed
                            analysis[key][f'{part}_score_per_doc'] = val

                    # Aggregate by token string
                    if scores_for_agg is not None:
                        if part == 'query' and 'query_tokens' in analysis:
                            analysis[key][f'{part}_agg'] = aggregate_token_importance(analysis['query_tokens'], scores_for_agg)
                        elif part == 'context' and 'context_tokens' in analysis:
                            analysis[key][f'{part}_agg'] = aggregate_token_importance(analysis['context_tokens'], scores_for_agg)
                            
                            # Also aggregate per document
                            if part == 'context' and 'context_tokens_list' in analysis and f'{part}_score_per_doc' in analysis[key]:
                                analysis[key][f'{part}_agg_per_doc'] = []
                                doc_scores = analysis[key][f'{part}_score_per_doc']
                                doc_tokens = analysis['context_tokens_list']
                                for i in range(min(len(doc_scores), len(doc_tokens))):
                                    analysis[key][f'{part}_agg_per_doc'].append(
                                        aggregate_token_importance(doc_tokens[i], doc_scores[i])
                                    )
                    
    return analysis

def aggregate_token_importance(tokens: List[str], scores: np.ndarray) -> Dict[str, float]:
    """Aggregates scores for identical tokens."""
    agg = {}
    # Ensure lengths match
    min_len = min(len(tokens), len(scores))
    for i in range(min_len):
        token = tokens[i].strip().lower() # Normalize
        # Skip special tokens if needed, or keep them. Let's keep them but maybe clean up 'Ġ' etc if using roberta/llama tokenizer
        token = token.replace('Ġ', '').replace(' ', '') # Simple cleanup
        
        if not token: continue
        
        if token not in agg:
            agg[token] = 0.0
        agg[token] += abs(scores[i]) # Sum of absolute importance? Or just sum? Usually abs for magnitude.
        # Let's use sum of raw values if we want to see direction, but for "importance" usually magnitude.
        # User asked for "importance", so let's stick to original values but maybe sum them up.
        # If intGrad, positive means contributes to positive class.
        # Let's just sum them for now.
        # Actually, for "importance" magnitude is safer if we don't know the class direction.
        # But intGrad is attribution.
        # Let's use the raw sum.
        # agg[token] += scores[i] 
        
        # Re-reading: "importance of each token".
        # If I have "cancer" with +5 and "cancer" with -5 (unlikely in same context for same class), they cancel out.
        # Let's use max absolute value or sum of absolute values.
        # Let's go with sum of absolute values to capture total impact.
        # agg[token] += abs(scores[i])
        
        # Actually, let's just use the raw score sum for now, assuming positive correlation with retrieval score.
        agg[token] += scores[i]
        
    return agg

def analyze_generation_results(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyzes generation results (Shapley values)."""
    analysis = {}
    
    analysis['qry_tokens'] = data.get('qry_tokens', [])
    analysis['gen_tokens'] = data.get('gen_tokens', [])
    
    shap_keys = ['shapley_values_token', 'shapley_values_sequence', 'shapley_values_bow', 'shapley_values_nucleus']
    
    for key in shap_keys:
        if key in data:
            analysis[key] = {}
            # Structure: {'query': array, 'context': array}
            # Arrays are likely [gen_len, input_len]
            
            for part in ['query', 'context']: # Note: 'context' might be part of the input sequence in generation
                # In generation inspection: keys were 'query' and 'context' inside shapley_values dicts.
                if part in data[key]:
                    val = data[key][part]
                    analysis[key][part] = val
                    
                    # Compute summary stats
                    # e.g. max contribution per generated token
                    if isinstance(val, np.ndarray):
                        # val shape: [num_gen_tokens, num_input_tokens]
                        # Sum of contributions for each generated token
                        total_attribution = np.sum(val, axis=1)
                        analysis[key][f'{part}_total_attrib'] = total_attribution
                        
                        # Max contributor index for each generated token
                        max_attrib_idx = np.argmax(val, axis=1)
                        analysis[key][f'{part}_max_attrib_idx'] = max_attrib_idx

    return analysis

def main():
    parser = argparse.ArgumentParser(description="Analyze RAG pipeline results.")
    parser.add_argument("filepath", help="Path to the pickle file to analyze")
    parser.add_argument("--type", choices=['retrieval', 'generation', 'auto'], default='auto', help="Type of results file")
    args = parser.parse_args()
    
    data = load_pickle(args.filepath)
    
    file_type = args.type
    if file_type == 'auto':
        if 'gen_tokens' in data:
            file_type = 'generation'
        elif 'input' in data and 'intGrad' in data:
            file_type = 'retrieval'
        else:
            print("Could not automatically determine file type. Please specify --type.")
            return

    print(f"Analyzing as {file_type} results...")
    
    if file_type == 'retrieval':
        results = analyze_retrieval_results(data)
        print("\n--- Retrieval Analysis ---")
        if 'query_tokens' in results:
            print(f"Query Tokens: {results['query_tokens']}")
        if 'intGrad' in results:
            print("Integrated Gradients (Query) Stats:")
            print(f"  Mean: {np.mean(results['intGrad']['query_score'])}")
            print(f"  Max: {np.max(results['intGrad']['query_score'])}")
            
    elif file_type == 'generation':
        results = analyze_generation_results(data)
        print("\n--- Generation Analysis ---")
        print(f"Generated Tokens ({len(results['gen_tokens'])}): {results['gen_tokens'][:10]}...")
        if 'shapley_values_token' in results:
            print("Shapley Values (Token) - Query Contribution Stats:")
            print(f"  Mean Total Attrib: {np.mean(results['shapley_values_token']['query_total_attrib'])}")

    # Save analysis if needed? For now just printing or returning object if imported.
    # Maybe save to a processed file?
    
if __name__ == "__main__":
    main()
