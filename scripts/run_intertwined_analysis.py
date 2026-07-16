import argparse
import os
import time
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
import pickle

# Add the parent directory to the path so we can import src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.experiments.eval_datasets import DatasetLoader
from src.Interpretable_RAG.retrieval_online import ExplainableAutoModelForRetrieval
from src.Interpretable_RAG.generation import ExplainableAutoModelForGeneration
from src.experiments.perturbations import PromptPerturbationModule
from src.experiments.metrics import calculate_metrics
from analysis.correlation_analysis import calculate_warg

QAMPARI_SYSTEM_PROMPT = (
    "Use the following retrieved documents to answer the user's query. "
    "The answer requires listing MULTIPLE entities. "
    "Respond with a comma-separated list of answers only. "
    "Do not explain, do not cite documents, do not use bullet points. "
    "Example format: Answer 1, Answer 2, Answer 3"
)

def setup_models(retriever_id: str, generator_id: str):
    print(f"INFO: Loading Generator {generator_id} with device_map='auto'...")
    generator = ExplainableAutoModelForGeneration.from_pretrained(
        generator_id,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )

    print(f"INFO: Loading Retriever {retriever_id}...")
    if 'dragon' in retriever_id.lower():
        retriever = ExplainableAutoModelForRetrieval.from_pretrained(
            query_encoder_name_or_path='facebook/dragon-plus-query-encoder',
            context_encoder_name_or_path='facebook/dragon-plus-context-encoder'
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        retriever = ExplainableAutoModelForRetrieval.from_pretrained(
            query_encoder_name_or_path=retriever_id,
            context_encoder_name_or_path=retriever_id
        ).to('cuda' if torch.cuda.is_available() else 'cpu')

    return retriever, generator


QUERY_PREFIXES = {
    'snowflake': 'query: ',
}

def main():
    print("DEBUG: Starting main...")
    parser = argparse.ArgumentParser(description="Runs Intertwined Analysis experiment (C1, C2, C3).")
    parser.add_argument("--dataset", type=str, default='musique', choices=['musique', 'nq', 'qampari'], help="Dataset to use.")
    parser.add_argument("--split", type=str, default=None, help="Dataset split to use.")
    parser.add_argument("--retriever_id", type=str, required=True, help="HF model ID or 'dragon'")
    parser.add_argument("--generator_id", type=str, required=True, help="HF generator ID")
    parser.add_argument("--output_path", type=str, required=True, help="Base folder where to save results.")
    parser.add_argument("--k", type=int, default=10, help="Number of documents for each group (Top-k and Low-k).")
    parser.add_argument("--num_docs_retrieved", type=int, default=100, help="Total docs to retrieve to find 'low' documents.")
    parser.add_argument("--max_gen_len", type=int, default=300, help="Max length of generated response.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for SHAP samples."); parser.add_argument("--retriever_batch_size", type=int, default=32, help="Batch size for retrieval encoding.")
    parser.add_argument("--num_queries", type=int, default=None, help="Limit number of queries for testing.")
    parser.add_argument("--max_samples_query", type=int, default=32, help="Max samples for query attribution (0 to skip).")
    parser.add_argument("--max_samples_context", type=int, default=32, help="Max samples for context attribution.")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="Override system prompt. Use 'qampari' for the QAMPARI list-format prompt.")
    parser.add_argument("--query_prefix", type=str, default=None,
                        help="Prefix prepended to queries before retrieval (e.g. for Snowflake Arctic Embed).")
    args = parser.parse_args()

    # Resolve query prefix
    if args.query_prefix is None:
        for key, prefix in QUERY_PREFIXES.items():
            if key in args.retriever_id.lower():
                args.query_prefix = prefix
                print(f"INFO: Auto-detected query prefix for {key}: '{prefix}'")

    retriever, generator = setup_models(args.retriever_id, args.generator_id)
    perturbator = PromptPerturbationModule()

    # Load data
    split_name = args.split if args.split else ('test' if args.dataset == 'qampari' else 'validation')
    loader = DatasetLoader(args.dataset, split=split_name)
    normalized_data = loader.load()
    
    if args.dataset == 'musique':
        from datasets import load_dataset
        raw_dataset = load_dataset('bdsaglam/musique', split='validation')
    else:
        raw_dataset = None # NQ doesn't need raw paragraphs for retrieval if we use the loader correctly? 
        # Actually DatasetLoader for NQ doesn't load documents. 
        # We might need a corpus for NQ.

    # Result logging
    gen_model_name = args.generator_id.split('/')[-1]
    ret_model_name = args.retriever_id.split('/')[-1] if 'dragon' not in args.retriever_id else 'dragon'

    base_out_dir = Path(args.output_path) / f"intertwined_{args.dataset}_{ret_model_name}_{gen_model_name}"
    base_out_dir.mkdir(parents=True, exist_ok=True)

    limit = args.num_queries if args.num_queries else len(normalized_data)
    
    gt_log = []
    processed_qids = set()
    gt_file = base_out_dir / "ground_truths.csv"
    if gt_file.exists():
        print(f"INFO: Loading existing ground truths from {gt_file}")
        df_gt = pd.read_csv(gt_file)
        gt_log = df_gt.to_dict('records')
        processed_qids = set(df_gt['query_id'].tolist())

    for i in tqdm(range(limit), desc=f"Processing {args.dataset}"):
        item = normalized_data[i]
        query_id = item['query_id']
        query = item['query']
        answers = item['answers']
        
        if query_id not in processed_qids:
            gt_log.append({'query_id': query_id, 'query': query, 'answers': answers})
            processed_qids.add(query_id)

        # Get paragraphs/corpus
        if 'paragraphs' in item:
            paragraphs = item['paragraphs']
        elif args.dataset == 'musique':
            paragraphs = [p['paragraph_text'] for p in raw_dataset[i]['paragraphs']]
        else:
            # For NQ, we might need a generic retrieval approach or a fixed set of documents.
            # If we don't have a corpus, we'll skip NQ for now or use a placeholder.
            print("WARNING: NQ corpus loading not fully implemented in this script. Skipping NQ.")
            continue

        if len(paragraphs) < args.k * 2:
            print(f"INFO: Skipping query {query_id} (not enough paragraphs: {len(paragraphs)})")
            continue

        # Check if all variations are already done for this query
        all_done = True
        for var_name in ['C1', 'C2', 'C3', 'C4_Random']:
            save_file = base_out_dir / var_name / f"{gen_model_name}_qid_{query_id}.pkl"
            if not save_file.exists():
                all_done = False
                break
        
        if all_done:
            if query_id not in processed_qids:
                gt_log.append({'query_id': query_id, 'query': query, 'answers': answers})
                processed_qids.add(query_id)
            continue

        # 1. Retrieve
        retrieval_query = (args.query_prefix + query) if args.query_prefix else query
        retrieved_ids, retrieved_sim = retriever(
            query=retrieval_query,
            contexts=paragraphs,
            k=min(args.num_docs_retrieved, len(paragraphs)),
            reorder=True, batch_size=args.retriever_batch_size, compute_grad=False
        )
        
        retrieved_docs = [paragraphs[idx] for idx in retrieved_ids]
        
        # 2. Create Intertwined variations (C1, C2, C3)
        variations = perturbator.generate_intertwined_setup(retrieved_docs, k=args.k)
        
        # Also add a Random baseline (C4)
        # Top-k + k Random from the rest of the corpus
        available_corpus = [p for p in paragraphs if p not in retrieved_docs[:args.k]]
        if len(available_corpus) >= args.k:
            random_docs = np.random.choice(available_corpus, args.k, replace=False).tolist()
            c4 = retrieved_docs[:args.k] + random_docs
            np.random.shuffle(c4)
            variations['C4_Random'] = c4

        for var_name, var_docs in variations.items():
            var_dir = base_out_dir / var_name
            var_dir.mkdir(parents=True, exist_ok=True)
            
            save_file = var_dir / f"{gen_model_name}_qid_{query_id}.pkl"
            if save_file.exists():
                continue 
                
            generator.explain_generate(
                query,
                var_docs,
                max_new_tokens=args.max_gen_len,
                batch_size=args.batch_size,
                max_samples_query=args.max_samples_query,
                max_samples_context=args.max_samples_context,
                do_sample=False,
                system=args.system_prompt
            )
            
            # Add metadata to save
            # We want to know the "ground truth" ranking for WARG.
            # For C1, C2, C3, the GT ranking of documents in var_docs is their rank in retrieved_docs.
            # Let's find the original ranks.
            orig_ranks = []
            for doc in var_docs:
                try:
                    orig_ranks.append(retrieved_docs.index(doc))
                except ValueError:
                    orig_ranks.append(999) # Very low rank if not in retrieved
            
            # Save metadata inside the pickle
            generator.save_values(str(save_file))
            
            # Append ranking info for easier offline analysis
            with open(save_file, 'rb') as f:
                data = pickle.load(f)
            data['ret_ranking_relative'] = orig_ranks
            data['ret_ranking_absolute'] = [retrieved_ids[r] if r < len(retrieved_ids) else -1 for r in orig_ranks]
            
            with open(save_file, 'wb') as f:
                pickle.dump(data, f)
        
        # Save ground truths incrementally
        pd.DataFrame(gt_log).to_csv(base_out_dir / "ground_truths.csv", index=False)
        
        # Save subset IDs for reproducibility
        subset_ids = [log['query_id'] for log in gt_log]
        import json
        with open(base_out_dir / "subset_qids.json", "w") as f:
            json.dump(subset_ids, f, indent=4)

    print(f"Finished processing {args.dataset}. Results in {base_out_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
