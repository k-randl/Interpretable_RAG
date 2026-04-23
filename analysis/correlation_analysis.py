import argparse
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
from typing import List, Dict

# Assuming metrics.py is in src/Interpretable_RAG
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.Interpretable_RAG.metrics import calculate_metrics

def rbo(list1: List[int], list2: List[int], p: float) -> float:
    """
    Computes Rank Biased Overlap (RBO) between two lists of items.
    """
    if not list1 or not list2:
        return 0.0
    
    sl, ll = (list1, list2) if len(list1) < len(list2) else (list2, list1)
    
    overlap = 0
    rbo_score = 0.0
    
    seen_in_ll = set()
    seen_in_sl = set()
    
    for d in range(1, len(sl) + 1):
        item_sl = sl[d - 1]
        item_ll = ll[d - 1]
        
        seen_in_sl.add(item_sl)
        seen_in_ll.add(item_ll)
        
        overlap = len(seen_in_sl.intersection(seen_in_ll))
        agreement = overlap / d
        
        rbo_score += (p ** (d - 1)) * agreement
        
    return rbo_score * (1 - p)

def calculate_warg(ret_ranking: List[int], gen_ranking: List[int], p: float = 0.9) -> float:
    """
    Calculates Weighted Attribution-Relevance Gap (WARG)
    WARG = 1 - RBO(ret_ranking, gen_ranking)
    """
    rbo_score = rbo(ret_ranking, gen_ranking, p)
    return 1.0 - rbo_score

def calculate_failure_modes(ret_ranking: List[int], gen_ranking: List[int], k: int) -> dict:
    """
    ret_ranking: list of doc indices ordered by retriever importance (e.g. [0, 1, 2, ...])
    gen_ranking: list of doc indices ordered by generator importance
    """
    if not ret_ranking or not gen_ranking:
        return {f'wasted_retrieval_k{k}': 0, f'noise_distraction_k{k}': 0}
        
    ret_top = ret_ranking[0]
    gen_top = gen_ranking[0]
    
    try:
        gen_rank_of_ret_top = gen_ranking.index(ret_top)
    except ValueError:
        gen_rank_of_ret_top = float('inf')
        
    try:
        ret_rank_of_gen_top = ret_ranking.index(gen_top)
    except ValueError:
        ret_rank_of_gen_top = float('inf')
        
    wasted_retrieval = 1 if gen_rank_of_ret_top > k else 0
    noise_distraction = 1 if ret_rank_of_gen_top > k else 0
    
    return {
        f'wasted_retrieval_k{k}': wasted_retrieval,
        f'noise_distraction_k{k}': noise_distraction
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze correlations between downstream metrics and WARG.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing pickle files with generation outputs")
    parser.add_argument("--ground_truth_path", type=str, required=True, help="CSV containing 'query_id' and 'answers' columns")
    parser.add_argument("--output_path", type=str, default="analysis/correlation_results.csv", help="Path to save results DataFrame")
    args = parser.parse_args()

    # Load ground truths
    # We assume 'answers' is eval-able (e.g. list of strings as string representation or just string)
    gt_df = pd.read_csv(args.ground_truth_path)
    gt_map = {}
    for _, row in gt_df.iterrows():
        ans = row['answers']
        if isinstance(ans, str):
            if ans.startswith('[') and ans.endswith(']'):
                import ast
                try:
                    ans = ast.literal_eval(ans)
                except:
                    ans = [ans]
            else:
                ans = [ans]
        gt_map[str(row['query_id'])] = ans

    files = list(Path(args.results_dir).glob('**/*.pkl'))
    data_rows = []

    for f in files:
        # Assuming filename format includes query id: e.g. "model_qid_123.pkl"
        fname = f.name
        qid = None
        if '_qid_' in fname:
            qid = fname.split('_qid_')[-1].replace('.pkl', '')
            
        with open(f, 'rb') as pkl:
            data = pickle.load(pkl)
            
        gen_tokens = data.get('gen_tokens', [])
        prediction = "".join(gen_tokens).replace('Ġ', ' ').replace(' ', ' ')
        
        if qid not in gt_map:
            continue
            
        gt_answers = gt_map[qid]
        metrics = calculate_metrics(prediction, gt_answers)
        
        # Calculate generator importance
        # If shapley_values_token -> context is [num_gen_tokens, num_context_docs] ? Or [num_context_docs, num_gen_tokens]?
        # Usually from earlier scripts: [num_docs, num_gen_tokens] ?
        if 'shapley_values_token' in data and 'context' in data['shapley_values_token']:
            s_ctx = data['shapley_values_token']['context']
            # Based on generate_plots.py, sum across gen tokens to get doc importance
            if s_ctx.ndim == 2:
                # If shape is [num_docs, num_gen_tokens]
                if s_ctx.shape[1] == len(gen_tokens):
                    doc_importance = np.sum(np.abs(s_ctx), axis=1)
                else: # fallback, maybe [num_gen_tokens, num_docs]
                    doc_importance = np.sum(np.abs(s_ctx), axis=0)
            else:
                doc_importance = np.zeros(1)
        else:
            continue
            
        num_docs = len(doc_importance)
        ret_ranking = list(range(num_docs)) # Assuming documents are provided in retrieved order
        gen_ranking = np.argsort(-doc_importance).tolist()
        
        warg_score = calculate_warg(ret_ranking, gen_ranking)
        
        condition = "unknown"
        if f.parent.name:
            condition = f.parent.name
            
        row = {
            'query_id': qid,
            'condition': condition,
            'exact_match': metrics['exact_match'],
            'f1_score': metrics['f1_score'],
            'rougeL': metrics['rougeL'],
            'bert_score': metrics.get('bert_score', 0.0),
            'WARG': warg_score
        }
        
        # Calculate failure modes for k=1..5
        for k in range(1, 6):
            fm = calculate_failure_modes(ret_ranking, gen_ranking, k)
            row.update(fm)
            
        data_rows.append(row)

    df = pd.DataFrame(data_rows)
    
    if len(df) == 0:
        print("No valid data points found for correlation.")
        return

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df.to_csv(args.output_path, index=False)
    
    print("\n--- Correlation Results ---")
    
    for metric in ['exact_match', 'f1_score', 'rougeL', 'bert_score']:
        if metric in df.columns:
            pearson_corr, p_p = pearsonr(df[metric], df['WARG'])
            spearman_corr, p_s = spearmanr(df[metric], df['WARG'])
            print(f"WARG vs {metric}:")
            print(f"  Pearson:  {pearson_corr:.4f} (p={p_p:.4e})")
            print(f"  Spearman: {spearman_corr:.4f} (p={p_s:.4e})\n")
            
    # Calculate failure mode percentages
    print("--- Failure Mode Decay Percentages ---")
    for k in range(1, 6):
        w_ret_pct = df[f'wasted_retrieval_k{k}'].mean() * 100
        n_dist_pct = df[f'noise_distraction_k{k}'].mean() * 100
        print(f"k={k}: Wasted Retrieval = {w_ret_pct:.1f}%, Noise Distraction = {n_dist_pct:.1f}%")

if __name__ == "__main__":
    main()
