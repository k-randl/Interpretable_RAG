#%%
# run_pipeline.py
import argparse
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

# Add the 'resources' folder to the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.Interpretable_RAG.generation import ExplainableAutoModelForGeneration
from src.Interpretable_RAG.tools import *
def setup(model_id: str):
    """Loads the generative model and the tokenizer."""
    print(f"INFO: Loading model {model_id}...")
    model = ExplainableAutoModelForGeneration.from_pretrained(
        model_id,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )
    print("INFO: Model loaded successfully.")
    return model

def load_data(topics_path: Path, ranked_list_path: Path, collection_path: Path, top_k: int):
    """Loads the queries and the retrieved documents."""
    print("INFO: Loading input data...")
    sep, has_header = sniff_file_dialect(topics_path)
    queries_df = pd.read_csv(topics_path, names=['query_id', 'query'], sep=sep, header=0 if has_header else None)
    ranked_passages_df = pd.read_csv(ranked_list_path)
    # If the passage text is not already present, join it from the collection
    if 'retrieved_text' not in ranked_passages_df.columns:
        collection_df = pd.read_csv(collection_path, sep='\t', names=['id', 'text'])
        ranked_passages_df = ranked_passages_df.merge(collection_df, left_on='docno', right_on='id', how='left')
        ranked_passages_df = ranked_passages_df.rename(columns={'text': 'retrieved_text'})

    # Group the contexts by query and take the top 'top_k'
    contexts = ranked_passages_df.groupby('query_id')['retrieved_text'].apply(lambda x: x.head(top_k).tolist()).to_dict()
    
    # Filter the queries to have only those with contexts
    valid_qids = contexts.keys()
    queries_df = queries_df[queries_df['query_id'].isin(valid_qids)].reset_index(drop=True)
    
    print(f"INFO: Data loaded for {len(queries_df)} queries.")
    return queries_df, contexts

def create_context_variations(contexts: dict, num_docs: int) -> dict:
    """Creates context variations: original, randomized, and without duplicates."""
    print("INFO: Creating context variations...")
    variations = {
        'original': {k: v[:num_docs] for k, v in contexts.items()},
        'randomized': {k: np.random.choice(v, min(num_docs, len(v)), replace=False).tolist() for k, v in contexts.items()},
        'no_duplicates': {k: list(dict.fromkeys(v))[:num_docs] for k, v in contexts.items()}
    }
    print("INFO: Variations created.")
    return variations

def run_single_experiment(exp_type: str, context_dict: dict, queries_df: pd.DataFrame, model, args: argparse.Namespace):
    """Runs a set of generation experiments and saves the results."""
    print(f"--- Starting Experiment: {exp_type.upper()} ---")
    save_dir = args.output_path / exp_type
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results_log = []

    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc=f"Generation for '{exp_type}'"):
        query_id = row['query_id']
        query = row['query']
        
        # Skip if there are no contexts for this query
        if query_id not in context_dict or not context_dict[query_id]:
            continue
            
        context = context_dict[query_id]
        BATCH_SIZE = 8
        # Run generation and explainability
        model.explain_generate(
            query,
            context,
            max_new_tokens=args.max_gen_len,
            batch_size=BATCH_SIZE,
            do_sample=False,
            max_samples='inf' # For precise SHAP calculation
        )
        
        # Save explainability data
        save_file = save_dir / f"{args.model_id.split('/')[-1]}_qid_{query_id}.pkl"
        model.save_values(str(save_file))
        
        results_log.append({'query_id': query_id, 'query': query, 'contexts': context})

    # Save a CSV log with the inputs used
    log_df = pd.DataFrame(results_log)
    log_df.to_csv(save_dir / "_inputs_log.csv", index=False)
    print(f"--- Experiment {exp_type.upper()} completed. Results in: {save_dir} ---")

def main():
    parser = argparse.ArgumentParser(description="Runs an end-to-end explainable RAG pipeline.")
    
    # Arguments for paths
    parser.add_argument("--topics_path", type=Path, required=True, help="Path to the file with the queries (topics).")
    parser.add_argument("--ranked_list_path", type=Path, required=True, help="Path to the ranked list from the retrieval.")
    parser.add_argument("--collection_path", type=Path, required=True, help="Path to the collection of passages.")
    parser.add_argument("--output_path", type=Path, required=True, help="Base folder where to save all the results.")

    # Arguments for models
    parser.add_argument("--model_id", type=str, default='meta-llama/Llama-3.1-8B-Instruct', help="ID of the generative model from Hugging Face.")
    
    # Experiment parameters
    parser.add_argument("--num_docs_context", type=int, default=6, help="Number of documents to use as context.")
    parser.add_argument("--max_gen_len", type=int, default=300, help="Maximum length of the generated response.")
    
    # Flags to control which experiments to run
    parser.add_argument("--run_original", action='store_true', help="Run the experiment with the original contexts.")
    parser.add_argument("--run_randomized", action='store_true', help="Run with contexts in random order.")
    parser.add_argument("--run_no_duplicates", action='store_true', help="Run with contexts without duplicates.")

    args = parser.parse_args()

    # 1. Model setup
    model = setup(args.model_id)

    # 2. Data loading and preparation
    #sep, has_header = sniff_file_dialect(args.topics_path)
    #queries_df = pd.read_csv(args.topics_path,  sep=sep, header=0 if has_header else None, names=['query_id', 'query'])
    #contexts = pd.read_csv(args.ranked_list_path).groupby('query_id')['retrieved_text'].apply(list).to_dict()
    queries_df, contexts = load_data(args.topics_path, args.ranked_list_path, args.collection_path, args.num_docs_context * 2) # We load a few more documents for randomization
    context_variations = create_context_variations(contexts, args.num_docs_context)

    # 3. Execution of the selected experiments
    if args.run_original:
        run_single_experiment('original', context_variations['original'], queries_df, model, args)
    
    if args.run_randomized:
        run_single_experiment('randomized', context_variations['randomized'], queries_df, model, args)
        
    if args.run_no_duplicates:
        run_single_experiment('no_duplicates', context_variations['no_duplicates'], queries_df, model, args)
        
    print("\nPipeline completed.")
#%%
if __name__ == "__main__":
    main()
    
    '''python run_pipeline.py \
    --topics_path data/topics/topics_efra.tsv\
    --ranked_list_path data/retrieval_results/retrieval_results_efra_chunks.csv\
    --collection_path data/eval_datasets/aligned_validation_data.csv \
    --output_path results/generation/efra_chunks_10_11_2025/ \
    --num_docs_context 6 \
    --run_original \
    --run_randomized'''
    
    
'''
    topics_path = data/topics/topics_efra.tsv
    ranked_list_path = data/retrieval_results/retrieval_results_efra_chunks.csv
    collection_path = data/eval_datasets/aligned_validation_data.csv
    output_path = results/generation/efra_chunks_10_11_2025/
    num_docs_context = 6
    run_original = True
    run_randomized = True
    
'''
