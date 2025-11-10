#%%
import sys
import os
import argparse
import numpy as np
import pandas as pd
import ir_measures
from pathlib import Path
from argparse import Namespace
from transformers import AutoTokenizer, AutoModel

# --- Assuming these are your custom functions from src ---
# Ensure sys.path is set correctly if this file isn't in the root
try:
    # Adds the parent directory of the script's parent (project root) to sys.path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.Interpretable_RAG.tools import *
    from src.Interpretable_RAG.search_tools import *
except ImportError:
    print("Warning: Could not import from src.Interpretable_RAG.")
    print("Please ensure the script is run from a location where 'src' is accessible")
    # Define placeholder functions for testing if imports fail
    def load_faiss_index(path):
        print(f"Mock: Load FAISS index from {path}")
        return None # Placeholder
    def search(args):
        print(f"Mock: Running search with args: {args}")
        return None # Placeholder
except NameError:
    # __file__ is not defined (e.g., in a notebook environment)
    print("Warning: __file__ not defined. Assuming 'src' is in sys.path")
    from src.Interpretable_RAG.tools import *
    from src.Interpretable_RAG.search_tools import *

# Set default CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,4,5,6,7"

def load_embedding_model(model_name: str) -> AutoModel:
    """
    Loads a transformer model based on a model_name.
    
    NOTE: You may need to adjust the paths/names below, especially for 'dragon'.
    """
    print(f"Loading embedding model: {model_name}")
    if model_name == 'snowflake':
        # Loads from Hugging Face
        return AutoModel.from_pretrained('Snowflake/snowflake-arctic-embed-l-v2.0')
    elif model_name == 'dragon':
        # *** This is an educated guess based on your original script's paths ***
        # Adjust this path to point to your actual local model directory
        model_path = '/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2019/passage_embeddings/dragon-plus-context-encoder'
        print(f"Warning: Loading 'dragon' model from hardcoded path: {model_path}")
        print("Please update 'load_embedding_model' if this path is incorrect.")
        return AutoModel.from_pretrained(model_path)
    else:
        # Fallback to load from Hugging Face by name
        print(f"Warning: Unknown model_name '{model_name}'. Attempting to load from Hugging Face.")
        return AutoModel.from_pretrained(model_name)

def load_queries_from_topics(topics_path: str) -> list[str]:
    """
    Loads queries from a TSV topics file.
    Assumes a two-column TSV [topic_id, query_text] with no header.
    """
    print(f"Loading queries from {topics_path}...")
    # Assuming topics are [topic_id, query_text]
    try:
        topics_df = pd.read_csv(topics_path, sep='\t', header=None, names=['id', 'query'])
        return topics_df['query'].tolist()
    except Exception as e:
        print(f"Error loading topics file: {e}")
        print("Please ensure it's a 2-column TSV file (topic_id, query_text) with no header.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run dense retrieval search with FAISS.")
    
    # Required Paths
    parser.add_argument('--topics_path', type=str, required=True, help='Path to the topics file (TSV)')
    parser.add_argument('--qrels_path', type=str, required=True, help='Path to the qrels file')
    parser.add_argument('--index_path', type=str, required=True, help='Path to the FAISS index file')
    parser.add_argument('--id_mapping_path', type=str, required=True, help='Path to the ID mapping file (TSV)')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the output results (CSV)')
    parser.add_argument('--model_name', type=str, required=True, help="Name of the model (e.g., 'dragon', 'snowflake')")

    # Optional Query Embeddings
    parser.add_argument('--query_embeddings_path', type=str, default=None,
                        help='(Optional) Path to pre-computed query embeddings .npy file. If not provided, embeddings will be calculated online.')
    
    # Search Parameters
    parser.add_argument('--top_k', type=int, default=1000, help='Number of documents to retrieve (k)')
    
    # Optional: Override CUDA devices
    parser.add_argument('--cuda_devices', type=str, default=os.environ["CUDA_VISIBLE_DEVICES"],
                        help='Comma-separated list of CUDA device IDs (e.g., "0,1")')

    #args = parser.parse_args()
    # --- ADD THIS BLOCK FOR INTERACTIVE TESTING ---
    test_args = [
        '--topics_path', '/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2022/data/topics.tsv',
        '--qrels_path', '/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2022/data/topics/qrels.qrel',
        '--index_path', '/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2022/passage_embeddings/snowflake-arctic-embed-l-v2.0/flat_index/index.faiss',
        '--id_mapping_path', '/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2022/data/CAST2022collection.cleaned.tsv',
        '--save_path', '/home/francomaria.nardini/raid/guidorocchietti/data/results/my_snowflake_run_interactive.csv',
        '--model_name', 'snowflake',
        '--top_k', '1000'
        # Add '--query_embeddings_path' here if you want to use it
    ]
    args = parser.parse_args(test_args)
    # --- END OF INTERACTIVE BLOCK ---

    # Update CUDA devices if specified
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    print(f"Using CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # --- Load persistent objects ---
    print(f"Loading FAISS index from {args.index_path}...")
    index = load_faiss_index(args.index_path)
    
    print(f"Loading ID mapping from {args.id_mapping_path}...")
    id_mapping = pd.read_csv(args.id_mapping_path, sep='\t')

    temp_path = None
    try:
        if args.query_embeddings_path:
            print(f"Loading pre-computed query embeddings from {args.query_embeddings_path}")
            query_embeddings_path_to_use = args.query_embeddings_path
        else:
            # --- Online Embedding Calculation ---
            print(f"Query embeddings path not provided. Calculating online using model: {args.model_name}")
            
            # 1. Load the model
            model = load_embedding_model(args.model_name)
            
            # 2. Load the queries
            queries = load_queries_from_topics(args.topics_path)
            
            # 3. Encode the queries
            print(f"Encoding {len(queries)} queries...")
            query_embeddings = model.encode(queries, show_progress_bar=True)
            
            # 4. Save to a temporary file
            temp_path = "temp_query_embeddings.npy"
            np.save(temp_path, query_embeddings)
            query_embeddings_path_to_use = temp_path
            print(f"Temporary embeddings saved to {temp_path}")

        # --- Define IR Metrics ---
        # Using a comprehensive set from your original script
        ir_metrics = [
            ir_measures.NDCG @ 3,
            ir_measures.NDCG @ 10,
            ir_measures.MRR @ 1000,
            ir_measures.P @ 1,
            ir_measures.P @ 3,
            ir_measures.P @ 10,
            ir_measures.R @ 10,
            ir_measures.R @ 100
        ]

        # --- Prepare arguments for the search function ---
        search_args = Namespace(
            queries_path=args.topics_path,
            qrels_path=args.qrels_path,
            index=index,
            query_embeddings_path=query_embeddings_path_to_use,
            id_mapping=id_mapping,
            model_name=args.model_name,
            top_k=args.top_k,
            save_path=args.save_path,
            sep=None,  # From your original script
            ir_metrics=ir_metrics,
            help=None  # From your original script
        )
        
        # --- Run Search ---
        print(f"Starting search... (top_k = {args.top_k})")
        print(f"Results will be saved to: {args.save_path}")
        search(search_args)
        print("Search complete.")

    finally:
        # --- Cleanup ---
        if temp_path:
            os.remove(temp_path)
            print(f"Removed temporary embeddings file: {temp_path}")

if __name__ == "__main__":
    main()

    
# %%
