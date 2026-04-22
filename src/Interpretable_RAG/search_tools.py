#%%
import os
import argparse
try:
    import faiss
except ImportError:
    faiss = None
    print("WARNING: Could not import faiss in search_tools. Some functionality may be unavailable.")
import numpy as np
import pandas as pd
try:
    import pytrec_eval
    import ir_measures
except ImportError:
    pytrec_eval = None
    ir_measures = None
from time import time
from transformers import AutoModel, AutoTokenizer
from .tools import embed_passages
from argparse import Namespace

#%%
def generate_query_embeddings(queries, model_name, max_length=512, device='cuda'):
    """Generate embeddings for queries using a specified transformer model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, device_map='auto')
    model.eval()
    print(f"Generating embeddings using {model_name} model...")
    embeddings = embed_passages(queries, model, tokenizer, device=device, max_length=max_length)
    print(f"Generated embeddings for {len(queries)} queries.")
    return embeddings

def load_faiss_index(index_path):
    """Load a FAISS index from a file."""
    print(f"Loading FAISS index from: {index_path}")
    index = faiss.read_index(index_path)
    print(f"Index loaded successfully with {index.ntotal} vectors.")
    return index

def load_faiss_index_gpu(index_path):
    """Load a FAISS index from a file and move to GPU."""
    print(f"Loading FAISS index from: {index_path}")
    index = faiss.read_index(index_path)
    print("Moving index to GPU...")
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
    print(f"Index loaded successfully with {index.ntotal} vectors.")
    return index

def search_index(index, query_embeddings, top_k):
    """Perform FAISS search."""
    distances, indices = index.search(query_embeddings, top_k)
    return distances, indices

def load_data(file_path, column_names=None, sep=None):
    """Generic function to load a dataset from a given file path."""
    if sep is None:
        df = pd.read_csv(file_path, header=None)
    else:
        df = pd.read_csv(file_path, sep=sep, header=None)
    if column_names:
        df.columns = column_names
    return df

def map_results(indices, distances, qids, id_mapping, top_k):
    """Convert FAISS search results into a structured DataFrame for evaluation."""
    results = {'qid': [], 'docno': [], 'rank': [], 'score': []}
    
    for i in range(len(distances)):
        topic_id = [qids[i]] * top_k
        doc_indices = [x for x in indices[i]]
        dist = distances[i].tolist()
        doc_ids = id_mapping.loc[doc_indices]['id'].tolist()
        rank = [j+1 for j in range(top_k)]
        
        if len(topic_id) == len(doc_ids) == len(rank) == len(dist):
            results['qid'] += topic_id
            results['docno'] += doc_ids
            results['rank'] += rank
            results['score'] += dist
            
    return pd.DataFrame(results)


def search(args):
    """
    Main search function. Requires a Namespace object as input.
    
    Required attributes:
        queries_path: path to the queries file (two columns: qid and query)
        qrels_path: path to the qrels file (four columns: qid, 0, doc_id, relevance)
        index: FAISS index object OR index_path: path to FAISS index file
        id_mapping: DataFrame or path to ID mapping file
        top_k: number of results to retrieve per query
    
    Optional attributes:
        query_embeddings_path: path to pre-computed query embeddings
        model_name: transformer model for embedding generation
        save_path: path to save results
        sep: separator for input files
        ir_metrics: list of IR metrics to compute
        efsearch: efSearch parameter for HNSW indexes
    """
    if hasattr(args, 'help') and args.help:
        print(__doc__)
        return
    
    print("Loading query and qrels data...")
    sep = args.sep if hasattr(args, 'sep') and args.sep else None
    
    print('Loading queries from: ', args.queries_path)
    topics = load_data(args.queries_path, column_names=['qid', 'query'], sep=sep)
    print('Loading qrels from: ', args.qrels_path)
    qrels = load_data(args.qrels_path, column_names=['qid', '0', 'doc_id', 'relevance'], sep=sep)

    # Filter queries to match qrels
    qids = qrels.qid.unique()
    topics = topics[topics.qid.isin(qids)]

    # Load ID Mapping
    if isinstance(args.id_mapping, str):
        print("Loading ID mapping from path...")
        id_mapping = pd.read_csv(args.id_mapping, sep='\t')
        id_mapping.columns = ['index', 'id']
        print(id_mapping.head())
    elif isinstance(args.id_mapping, pd.DataFrame):
        print("Using provided ID mapping...")
        id_mapping = args.id_mapping
        print(id_mapping.head())
    
    # Generate or load embeddings
    if hasattr(args, 'query_embeddings_path') and args.query_embeddings_path:
        print(f"Loading query embeddings from: {args.query_embeddings_path}")
        query_embeddings = np.load(args.query_embeddings_path)
        print('QUERY EMBEDDINGS SHAPE: ', query_embeddings.shape)
    else:
        print(f"Generating query embeddings using {args.model_name} model...")
        queries = topics['query'].tolist()
        query_embeddings = generate_query_embeddings(queries, args.model_name)

    # Load FAISS Index
    print("Loading FAISS index...")
    if hasattr(args, 'index') and args.index is not None:
        index = args.index
        print("Using provided index")
    elif hasattr(args, 'index_path') and args.index_path:
        print("Loading index from path")
        index = load_faiss_index(args.index_path)
    else:
        print("No index provided. You must provide an index or an index path.")
        return
    
    # Set HNSW efSearch if applicable
    if hasattr(index, "hnsw") and hasattr(args, 'efsearch'):
        print("This index is HNSW-based.")
        index.hnsw.efSearch = args.efsearch
        print(f"efSearch set to {index.hnsw.efSearch}")

    # Perform Search
    print("Searching index...")
    start_time = time()
    distances, indices = index.search(query_embeddings, args.top_k)
    search_time = time() - start_time
    print('INDICES SHAPE: ', indices.shape)
    print('DISTANCES SHAPE: ', distances.shape)
    averaged_time = search_time / len(query_embeddings)
    print(f"Average search time per query: {averaged_time:.4f} seconds")
    
    # Save indices and distances
    if hasattr(args, 'save_path') and args.save_path:
        np.save(args.save_path.replace('.csv', '_indices.npy'), indices)
        np.save(args.save_path.replace('.csv', '_distances.npy'), distances)
    
    # Handle metric type for score conversion
    metric_type = index.metric_type
    if metric_type == faiss.METRIC_L2:
        print("The index is using L2 (Euclidean Distance).")
        distances = -1 * distances
    elif metric_type == faiss.METRIC_INNER_PRODUCT:
        print("The index is using Inner Product (Dot Product).")
    else:
        print(f"The index is using an unknown metric type: {metric_type}")
    
    # Process results
    results_df = map_results(indices, distances, qids, id_mapping, args.top_k)

    # Define evaluation metrics
    if hasattr(args, 'ir_metrics') and args.ir_metrics:
        ir_metrics = args.ir_metrics
        print("Using custom IR metrics")
        print(f"Custom IR metrics: {ir_metrics}")
    else:
        print("Using default IR metrics")
        ir_metrics = [
            ir_measures.AP @ 200,
            ir_measures.AP,
            ir_measures.NDCG @ 3,
            ir_measures.P @ 1,
            ir_measures.P @ 3,
            ir_measures.R @ 1,
            ir_measures.R @ 5,
            ir_measures.R @ 10,
            ir_measures.R @ 200,
            ir_measures.R @ 500,
            ir_measures.MRR,
        ]
    
    results_ir = calculate_results(qrels, results_df, metrics=ir_metrics)
    
    print('Average IR metrics:', results_ir)
    print('Results per query:', results_df.head())
    
    if hasattr(args, 'save_path') and args.save_path:
        results_df.to_csv(args.save_path, index=False)
        print(f"Results saved to: {args.save_path}")
        results_ir.to_csv(args.save_path.replace('.csv', '_mean.csv'), index=False)
        with open(args.save_path.replace('.csv', '_time.txt'), 'w') as f:
            f.write(f"Search time: {search_time:.2f} seconds\n")
            f.write(f"Average search time per query: {averaged_time:.4f} seconds")

    return results_df, results_ir


def generate_args(year, model, conversational_path, index_type='flat', 
                  index_name='flat_index_ip.faiss', 
                  passage_embeddings_name='passage_embeddings.npy', 
                  query_embeddings_name='query_embeddings.npy'):
    """Generate argument paths for search based on year and model."""
    indexes = {
        'flat': 'flat_index',
        'ivf': 'ivf_index',
        'hnsw': 'hnsw_index',
        'cosine': 'cosine_index'
    }
    base_paths = {
        2019: "CAST2019",
        2020: "CAST2020",
        2022: "CAST2022"
    }
    model_paths = {
        "dragon": "dragon-plus-context-encoder",
        "snowflake": "snowflake-arctic-embed-l-v2.0"
    }
    
    model_name = model_paths[model] if model == 'snowflake' else 'facebook/dragon-plus-query-encoder'
    base_path = os.path.join(conversational_path, base_paths[year])
    model_path = os.path.join(base_path, f"passage_embeddings/{model_paths[model]}")
    index_path = os.path.join(model_path, f"{indexes[index_type]}/{index_name}")
    embeddings_path = os.path.join(model_path, passage_embeddings_name)
    query_embeddings_path = os.path.join(model_path, query_embeddings_name)
    id_mapping_path = os.path.join(base_path, f"data/{base_paths[year]}_ID_Mapping.tsv")
    qrels_path = os.path.join(base_path, "data/topics/qrels.qrel")
    topics_path = os.path.join(base_path, "data/topics/topics.tsv")
    save_path = os.path.join(base_path, f"data/results/results_{model}_{index_type}.csv")
    
    return index_path, embeddings_path, query_embeddings_path, id_mapping_path, qrels_path, topics_path, save_path, model_name


def calculate_results(qrels, results_df, metrics=None):
    """Calculate IR evaluation metrics."""
    if metrics is None:
        print("No metrics specified, using default metrics")    
        metrics = [
            ir_measures.NDCG @ 3,
            ir_measures.NDCG @ 10,
            ir_measures.R @ 100,
            ir_measures.MRR @ 1000,
        ]
    
    qrels_ir = qrels[['qid', 'doc_id', 'relevance']]
    qrels_ir.columns = ['query_id', 'doc_id', 'relevance']
    ir_results_df = results_df[['qid', 'docno', 'score']]
    ir_results_df.columns = ['query_id', 'doc_id', 'score']
    
    results_ir = ir_measures.calc_aggregate(metrics, qrels_ir, ir_results_df)
    results_ir = pd.DataFrame(results_ir, index=[0])
    return results_ir
