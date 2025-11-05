import sys
import os
import argparse
from pathlib import Path
import torch
import pandas as pd
import pickle
from tqdm import tqdm

# Add the project root to the path to allow imports from src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.Interpretable_RAG.retrieval_online import ExplainableAutoModelForRetrieval

def setup_model(model_id: str, device: str):
    """Loads the retrieval model and tokenizer."""
    print(f"INFO: Loading model {model_id}...")
    model = ExplainableAutoModelForRetrieval.from_pretrained(
        model_id,
        add_pooling_layer=False
    ).to(device)
    print("INFO: Model loaded successfully.")
    return model

def load_data(topics_path: str, retrieval_results_path: str):
    """Loads topics and retrieval results."""
    print("INFO: Loading data...")
    try:
        topics = pd.read_csv(topics_path, sep='\t', header=None, names=['query_id', 'query'])
        retrieval_results = pd.read_csv(retrieval_results_path)
        print(f"INFO: Loaded {len(topics)} queries.")
        return topics, retrieval_results
    except FileNotFoundError as e:
        print(f"ERROR: Data file not found - {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run offline experiments for Snowflake model interpretability.")
    parser.add_argument("--topics_path", type=str, required=True, help="Path to the topics file (e.g., topics.tsv).")
    parser.add_argument("--retrieval_results_path", type=str, required=True, help="Path to the retrieval results file (e.g., ir_results.csv).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output .pkl files.")
    parser.add_argument("--model_id", type=str, default="Snowflake/snowflake-arctic-embed-l-v2.0", help="Model ID from Hugging Face.")
    parser.add_argument("--calculate_importance", action="store_true", help="Flag to enable importance score calculation (aGrad).")
    parser.add_argument("--importance_output_dir", type=str, help="Directory to save importance score pickles. Required if --calculate_importance is set.")

    args = parser.parse_args()

    if args.calculate_importance and not args.importance_output_dir:
        parser.error("--importance_output_dir is required when --calculate_importance is set.")

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"INFO: Using device: {device}")

    # Create output directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    if args.calculate_importance:
        os.makedirs(args.importance_output_dir, exist_ok=True)

    # Load model and data
    model = setup_model(args.model_id, device)
    topics, retrieval_results = load_data(args.topics_path, args.retrieval_results_path)
    queries = topics.set_index('query_id')['query'].to_dict()

    # Main processing loop
    for qid, query in tqdm(queries.items(), total=len(queries), desc='Processing queries'):
        # Get contexts for the current query
        contexts = retrieval_results[retrieval_results['query_id'] == qid]['retrieved_text'].tolist()
        
        if not contexts:
            print(f"WARNING: No contexts found for query_id {qid}. Skipping.")
            continue

        # Process with the model to get hidden states and attentions
        model('query: ' + query, contexts, output_attentions=True, output_hidden_states=True)
        
        # Save the intermediate values (attentions, hidden states)
        output_path = os.path.join(args.output_dir, f'query_{qid}.pkl')
        model.save_values(output_path, batch_size=16)

        # Optionally, calculate and save importance scores
        if args.calculate_importance:
            importance_score = model.aGrad()
            in_tokens = model.in_tokens
            importance_score['query_in_tokens'] = in_tokens['query']
            importance_score['context_in_tokens'] = in_tokens['context']
            
            importance_path = os.path.join(args.importance_output_dir, f'importance_scores_{qid}.pkl')
            with open(importance_path, 'wb') as f:
                pickle.dump(importance_score, f)

    print("\nProcessing complete.")
    print(f"Intermediate model values saved in: {args.output_dir}")
    if args.calculate_importance:
        print(f"Importance scores saved in: {args.importance_output_dir}")

if __name__ == "__main__":
    main()
