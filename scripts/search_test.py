#%%
# --- Core Imports ---
import sys
import os
import numpy as np
import pandas as pd
import ir_measures
from pathlib import Path
from argparse import Namespace
from transformers import AutoTokenizer, AutoModel

# --- TQDM Warning ---
# This just silences the "IProgress not found" warning if you don't have
# ipywidgets installed. It's safe to ignore or remove.
from tqdm.autonotebook import tqdm as notebook_tqdm
#%%
# --- 1. SET YOUR PROJECT ROOT PATH ---
# !! IMPORTANT !!
# Change this path to the root of your project (the folder that contains the 'src' directory)
PROJECT_ROOT = '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/' # <-- CHANGE THIS

# Add project root to system path
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
#%%
# --- 2. Import Your Custom Modules ---
try:
    from src.Interpretable_RAG.tools import *
    from src.Interpretable_RAG.search_tools import *
    print("Successfully imported custom 'src' modules.")
except ImportError as e:
    print(f"ImportError: {e}")
    print(f"ERROR: Could not import from 'src.Interpretable_RAG'.")
    print(f"Please check that PROJECT_ROOT is set correctly: {PROJECT_ROOT}")
#%%
# --- 3. Define Parameters ---
# These replace the command-line arguments.
# Your provided arguments:
topics_path = '/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2022/data/topics.tsv'
qrels_path = '/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2022/data/topics/qrels.qrel'
index_path = '/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2022/passage_embeddings/snowflake-arctic-embed-l-v2.0/flat_index/index.faiss'
id_mapping_path = '/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2022/data/CAST2022collection.cleaned.tsv'
save_path = '/home/francomaria.nardini/raid/guidorocchietti/data/results/my_snowflake_run_interactive.csv'
model_name = 'snowflake'
top_k = 1000

# Set this to a path if you have pre-computed embeddings
# Set to None to calculate embeddings online
query_embeddings_path = None 

# CUDA device settings
cuda_devices = "1,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
#%%
# --- 4. Helper Functions (Copied from previous script) ---

def load_embedding_model(model_name: str) -> AutoModel:
    """
    Loads a transformer model based on a model_name.
    """
    print(f"Loading embedding model: {model_name}")
    if model_name == 'snowflake':
        # Loads from Hugging Face
        return AutoModel.from_pretrained('Snowflake/snowflake-arctic-embed-l-v2.0', add_pooling_layer=False, map_device='auto')
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
    try:
        topics_df = pd.read_csv(topics_path, sep='\t', header=None, names=['id', 'query'])
        return topics_df['query'].tolist()
    except Exception as e:
        print(f"Error loading topics file: {e}")
        print("Please ensure it's a 2-column TSV file (topic_id, query_text) with no header.")
        return []
#%%
# --- 5. Main Script Logic (was inside main()) ---

print(f"Using CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

# --- Load persistent objects ---
#print(f"Loading FAISS index from {index_path}...")
index = load_faiss_index(index_path)

print(f"Loading ID mapping from {id_mapping_path}...")
# Note: This loads the entire collection file as the mapping.
# This assumes the 'search' function can handle this structure,
# or that the file only contains ID mappings.
id_mapping = pd.read_csv(id_mapping_path, sep='\t', header=None)
id_mapping2 = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2022/data/CAST2022collection.aligned.tsv', sep='\t',header=None)
id_mapping3 = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2022/data/CAST2022collection.tsv', sep='\t',header=None)
temp_path = None
query_embeddings_path_to_use = None
query_embeddings = None # This will hold embeddings if calculated
#%%
try:
    if query_embeddings_path:
        print(f"Loading pre-computed query embeddings from {query_embeddings_path}")
        query_embeddings_path_to_use = query_embeddings_path
    else:
        # --- Online Embedding Calculation ---
        print(f"Query embeddings path not provided. Calculating online using model: {model_name}")
        
        # 1. Load the model
        model = load_embedding_model(model_name)
        
        # 2. Load the queries
        queries = load_queries_from_topics(topics_path)
        
        if queries:
            # 3. Encode the queries
            print(f"Encoding {len(queries)} queries...")
            query_embeddings = model.encode(queries, show_progress_bar=True)
            
            # 4. Save to a temporary file
            temp_path = "temp_query_embeddings.npy"
            np.save(temp_path, query_embeddings)
            query_embeddings_path_to_use = temp_path
            print(f"Temporary embeddings saved to {temp_path}")
            print(f"Embeddings shape: {query_embeddings.shape}")
        else:
            print("No queries loaded, stopping.")
            # This will skip the rest of the 'try' block
            raise Exception("Failed to load queries.")

    # --- Define IR Metrics ---
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
        queries_path=topics_path,
        qrels_path=qrels_path,
        index=index,
        query_embeddings_path=query_embeddings_path_to_use,
        id_mapping=id_mapping,
        model_name=model_name,
        top_k=top_k,
        save_path=save_path,
        sep=None,
        ir_metrics=ir_metrics,
        help=None
    )
    
    # --- Run Search ---
    print(f"Starting search... (top_k = {top_k})")
    print(f"All variables are now in scope. Inspect 'search_args' before proceeding if needed.")
    # You can pause here in a notebook by splitting the cell
    
    search(search_args)
    
    print("Search complete.")
    print(f"Results saved to: {save_path}")

finally:
    # --- Cleanup ---
    if temp_path:
        os.remove(temp_path)
        print(f"Removed temporary embeddings file: {temp_path}")