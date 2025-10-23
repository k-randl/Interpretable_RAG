#%%
import os
import argparse
import faiss
import numpy as np
import pandas as pd
import pytrec_eval
import ir_measures
from time import time
from transformers import AutoModel, AutoTokenizer
from resources.tools import embed_passages
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
    """Load a FAISS index from a file."""
    print(f"Loading FAISS index from: {index_path}")
    index = faiss.read_index(index_path)
    print("Moving index to GPU...")
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(index)
    print(f"Index loaded successfully with {index.ntotal} vectors.")
    return index

def search_index(index, query_embeddings, top_k):

    distances, indices = index.search(query_embeddings, top_k)

    return distances, indices

def load_data(file_path,  column_names=None,sep =None):
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


### SEARCH FUNCTION ###
### args must be a Namespace object
### Example args = Namespace(queries_path = 'path/to/queries.csv', qrels_path = 'path/to/qrels.csv', index_path = 'path/to/index.faiss', query_embeddings_path = 'path/to/query_embeddings.npy', id_mapping_path = 'path/to/id_mapping.csv', model_name = 'facebook/dragon-plus-query-encoder', top_k = 500, save_path = 'path/to/save_results.csv')
def search_old_old(args):
    
    if args.help:
        print('The search function requires a Namespace object as input. The Namespace object must contain the following attributes:')
        print('queries_path: path to the queries file. The file must contain two columns: qid and query. Optional == False')
        print('qrels_path: path to the qrels file. The file must contain four columns: qid, 0, doc_id, relevance. Optional == False')
        print('index: FAISS index object. Optional == True. If not provided, index_path must be provided.')
        print('index_path: path to the FAISS index file. Optional == True. If not provided, index must be provided.')
        print('query_embeddings_path: path to the query embeddings file. Optional == True. If not provided, the query embeddings will be generated using the model_name.')
        print('id_mapping_path: path to the ID mapping file. The file must contain two columns: index and id. Optional == False')
        print('model_name: name of the transformer model used to generate the embeddings. Optional == True. If not provided, the query embeddings must be provided.')
        print('top_k: number of results to retrieve per query. Optional == False')
        print('save_path: path to save the search results. Optional == True. If not provided, the results will not be saved.')
        print('sep: separator used in the queries and qrels files. Optional == True. If not provided, the default separator will be used.')
        print('ir_metrics: list of IR metrics to compute. Optional == True. If not provided, the default metrics will be used.')
        return
    print("Loading query and qrels data...")
    if args.sep:
        sep = args.sep
    else:
        sep = None
    topics = load_data(args.queries_path,  column_names=['qid', 'query'], sep = sep)
    qrels = load_data(args.qrels_path,  column_names=['qid', '0', 'doc_id', 'relevance'],sep=sep)

    # Filter queries to match qrels
    qids = qrels.qid.unique()
    topics = topics[topics.qid.isin(qids)]

    # Load ID Mapping
    print("Loading ID mapping...")
    id_mapping = pd.read_csv(args.id_mapping_path, sep = '\t')
    id_mapping.columns = ['index', 'id']
    print(id_mapping.head())
    # Generate Embeddings
    
    if args.query_embeddings_path:
        print(f"Loading query embeddings from: {args.query_embeddings_path}")
        query_embeddings = np.load(args.query_embeddings_path)
        print('QUERY EMBEDDINGS SHAPE: ',query_embeddings.shape)
    else:
        print(f"Generating query embeddings using {args.model_name} model...")
        queries = topics['query'].tolist()
        query_embeddings = generate_query_embeddings(queries, args.model_name)

    # Load FAISS Index
    print("Loading FAISS index...")
    if args.index:
        index = args.index
        print("Using provided index")
    elif args.index_path:
        print("Loading index from path")
        index = load_faiss_index(args.index_path)
    else:
        print("No index provided. You must provide an index or an index path.")
        return

    # Perform Search
    print("Searching index...")
    start_time = time()
    distances, indices = search_index(index, query_embeddings, args.top_k)
    search_time = time() - start_time
    print('INDICES SHAPE: ',indices.shape)
    print('DISTANCES SHAPE: ',distances.shape)

    # Process and Save Results
    results_df = map_results(indices, distances, qids, id_mapping, args.top_k)

    # Define evaluation metrics
    if args.ir_metrics:
        ir_metrics = args.ir_metrics
        print("Using custom IR metrics")
        print(f"Custom IR metrics: {ir_metrics}")
    else:
        print("Using default IR metrics")
        print("Default IR metrics: MAP@200, MAP, NDCG@3, P@1, P@3, R@1, R@5, R@10, R@200, R@500, MRR")
        ir_metrics  = [
            
            #map cut 200
            ir_measures.AP @ 200,        # Mean Average Precision (MAP)
            ir_measures.AP,        # Mean Average Precision (MAP)
            ir_measures.NDCG @ 3,  # Normalized Discounted Cumulative Gain @3
            ir_measures.P @ 1,     # Precision at 1
            ir_measures.P @ 3,     # Precision at 3
            ir_measures.R @ 1,     # Recall at 1
            ir_measures.R @ 5,     # Recall at 5
            ir_measures.R @ 10,    # Recall at 10
            ir_measures.R @ 200,   # Recall at 200
            ir_measures.R @ 500,   # Recall at 500
            ir_measures.MRR,       # Mean Reciprocal Rank
        ]

    qrels_ir = qrels[['qid', 'doc_id', 'relevance']]
    qrels_ir.columns =['query_id', 'doc_id', 'relevance']
    ir_results_df = results_df[['qid', 'docno', 'score']]
    ir_results_df.columns = ['query_id', 'doc_id', 'score']
    # Compute IR metrics directly
    results_ir = ir_measures.calc_aggregate(ir_metrics, qrels_ir, ir_results_df)
    results_ir = pd.DataFrame(results_ir, index=[0])
#    for metric, value in results_ir.items():
#        print(f"{metric}: {value:.4f}")
    print(results_ir.head())
    
    print(results_df.head())
    if args.save_path:
        results_df.to_csv(args.save_path, index=False)
        print(f"Results saved to: {args.save_path}")
        ### Save IR metrics
        results_ir.to_csv(args.save_path.replace('.csv', '_mean.csv'), index=False)
        with open(args.save_path.replace('.csv', '_time.txt'), 'w') as f:
            f.write(f"Search time: {search_time:.2f} seconds")

    return results_df, results_ir

def search_old(args):
    
    if args.help:
        print('The search function requires a Namespace object as input. The Namespace object must contain the following attributes:')
        print('queries_path: path to the queries file. The file must contain two columns: qid and query. Optional == False')
        print('qrels_path: path to the qrels file. The file must contain four columns: qid, 0, doc_id, relevance. Optional == False')
        print('index: FAISS index object. Optional == True. If not provided, index_path must be provided.')
        print('index_path: path to the FAISS index file. Optional == True. If not provided, index must be provided.')
        print('query_embeddings_path: path to the query embeddings file. Optional == True. If not provided, the query embeddings will be generated using the model_name.')
        print('id_mapping: path to the ID mapping file. The file must contain two columns: index and id. Optional == False')
        print('model_name: name of the transformer model used to generate the embeddings. Optional == True. If not provided, the query embeddings must be provided.')
        print('top_k: number of results to retrieve per query. Optional == False')
        print('save_path: path to save the search results. Optional == True. If not provided, the results will not be saved.')
        print('sep: separator used in the queries and qrels files. Optional == True. If not provided, the default separator will be used.')
        print('ir_metrics: list of IR metrics to compute. Optional == True. If not provided, the default metrics will be used.')
        return
    print("Loading query and qrels data...")
    if args.sep:
        sep = args.sep
    else:
        sep = None
    print('Loading queries from: ', args.queries_path)
    topics = load_data(args.queries_path,  column_names=['qid', 'query'], sep = sep)
    print('Loading qrels from: ', args.qrels_path)
    qrels = load_data(args.qrels_path,  column_names=['qid', '0', 'doc_id', 'relevance'],sep=sep)

    # Filter queries to match qrels
    qids = qrels.qid.unique()
    topics = topics[topics.qid.isin(qids)]

    # Load ID Mapping
    ### check if id_mapping is a path or a dataframe
    if isinstance(args.id_mapping, str):
        print("Loading ID mapping from path...")
        id_mapping = pd.read_csv(args.id_mapping, sep = '\t')
        id_mapping.columns = ['index', 'id']
        print(id_mapping.head())
    elif isinstance(args.id_mapping, pd.DataFrame):
        print("Using provided ID mapping...")
        id_mapping = args.id_mapping
        print(id_mapping.head())
    # Generate Embeddings
    
    if args.query_embeddings_path:
        print(f"Loading query embeddings from: {args.query_embeddings_path}")
        query_embeddings = np.load(args.query_embeddings_path)
        print('QUERY EMBEDDINGS SHAPE: ',query_embeddings.shape)
    else:
        print(f"Generating query embeddings using {args.model_name} model...")
        queries = topics['query'].tolist()
        query_embeddings = generate_query_embeddings(queries, args.model_name)

    # Load FAISS Index
    print("Loading FAISS index...")
    if args.index:
        index = args.index
        print("Using provided index")
    elif args.index_path:
        print("Loading index from path")
        index = load_faiss_index(args.index_path)
    else:
        print("No index provided. You must provide an index or an index path.")
        return

    # Perform Search
    print("Searching index...")
    start_time = time()
    distances, indices = search_index(index, query_embeddings, args.top_k)
    search_time = time() - start_time
    print('INDICES SHAPE: ',indices.shape)
    print('DISTANCES SHAPE: ',distances.shape)

    # Process and Save Results
    results_df = map_results(indices, distances, qids, id_mapping, args.top_k)

    # Define evaluation metrics
    if args.ir_metrics:
        ir_metrics = args.ir_metrics
        print("Using custom IR metrics")
        print(f"Custom IR metrics: {ir_metrics}")
    else:
        print("Using default IR metrics")
        print("Default IR metrics: MAP@200, MAP, NDCG@3, P@1, P@3, R@1, R@5, R@10, R@200, R@500, MRR")
        ir_metrics  = [
            
            #map cut 200
            ir_measures.AP @ 200,        # Mean Average Precision (MAP)
            ir_measures.AP,        # Mean Average Precision (MAP)
            ir_measures.NDCG @ 3,  # Normalized Discounted Cumulative Gain @3
            ir_measures.P @ 1,     # Precision at 1
            ir_measures.P @ 3,     # Precision at 3
            ir_measures.R @ 1,     # Recall at 1
            ir_measures.R @ 5,     # Recall at 5
            ir_measures.R @ 10,    # Recall at 10
            ir_measures.R @ 200,   # Recall at 200
            ir_measures.R @ 500,   # Recall at 500
            ir_measures.MRR,       # Mean Reciprocal Rank
        ]
    results_ir = calculate_results(qrels, results_df, metrics=ir_metrics)
    
    print('Average IR metrics:', results_ir)
    
    print('Results per query:', results_df.head())
    if args.save_path:
        results_df.to_csv(args.save_path, index=False)
        print(f"Results saved to: {args.save_path}")
        ### Save IR metrics
        results_ir.to_csv(args.save_path.replace('.csv', '_mean.csv'), index=False)
        with open(args.save_path.replace('.csv', '_time.txt'), 'w') as f:
            f.write(f"Search time: {search_time:.2f} seconds")

    return results_df, results_ir


def search(args):
    
    if args.help:
        print('The search function requires a Namespace object as input. The Namespace object must contain the following attributes:')
        print('queries_path: path to the queries file. The file must contain two columns: qid and query. Optional == False')
        print('qrels_path: path to the qrels file. The file must contain four columns: qid, 0, doc_id, relevance. Optional == False')
        print('index: FAISS index object. Optional == True. If not provided, index_path must be provided.')
        print('index_path: path to the FAISS index file. Optional == True. If not provided, index must be provided.')
        print('query_embeddings_path: path to the query embeddings file. Optional == True. If not provided, the query embeddings will be generated using the model_name.')
        print('id_mapping: path to the ID mapping file. The file must contain two columns: index and id. Optional == False')
        print('model_name: name of the transformer model used to generate the embeddings. Optional == True. If not provided, the query embeddings must be provided.')
        print('top_k: number of results to retrieve per query. Optional == False')
        print('save_path: path to save the search results. Optional == True. If not provided, the results will not be saved.')
        print('sep: separator used in the queries and qrels files. Optional == True. If not provided, the default separator will be used.')
        print('ir_metrics: list of IR metrics to compute. Optional == True. If not provided, the default metrics will be used.')
        return
    print("Loading query and qrels data...")
    if args.sep:
        sep = args.sep
    else:
        sep = None
    print('Loading queries from: ', args.queries_path)
    topics = load_data(args.queries_path,  column_names=['qid', 'query'], sep = sep)
    print('Loading qrels from: ', args.qrels_path)
    qrels = load_data(args.qrels_path,  column_names=['qid', '0', 'doc_id', 'relevance'],sep=sep)

    # Filter queries to match qrels
    qids = qrels.qid.unique()
    topics = topics[topics.qid.isin(qids)]

    # Load ID Mapping
    ### check if id_mapping is a path or a dataframe
    if isinstance(args.id_mapping, str):
        print("Loading ID mapping from path...")
        id_mapping = pd.read_csv(args.id_mapping, sep = '\t')
        id_mapping.columns = ['index', 'id']
        print(id_mapping.head())
    elif isinstance(args.id_mapping, pd.DataFrame):
        print("Using provided ID mapping...")
        id_mapping = args.id_mapping
        print(id_mapping.head())
    # Generate Embeddings
    
    if args.query_embeddings_path:
        print(f"Loading query embeddings from: {args.query_embeddings_path}")
        query_embeddings = np.load(args.query_embeddings_path)
        print('QUERY EMBEDDINGS SHAPE: ',query_embeddings.shape)
    else:
        print(f"Generating query embeddings using {args.model_name} model...")
        queries = topics['query'].tolist()
        query_embeddings = generate_query_embeddings(queries, args.model_name)

    # Load FAISS Index
    print("Loading FAISS index...")
    if args.index:
        index = args.index
        print("Using provided index")
    elif args.index_path:
        print("Loading index from path")
        index = load_faiss_index(args.index_path)
    else:
        print("No index provided. You must provide an index or an index path.")
        return
    if hasattr(index, "hnsw"):
        print("This index is HNSW-based.")
    
        # Set efSearch
        index.hnsw.efSearch = args.efsearch  # or any value you prefer
        print(f"efSearch set to {index.hnsw.efSearch}")

    # Perform Search
    print("Searching index...")
    
    #distances, indices = search_index(index, query_embeddings, args.top_k)
    start_time = time()
    distances, indices = index.search(query_embeddings, args.top_k)
    search_time = time() - start_time
    print('INDICES SHAPE: ',indices.shape)
    print('DISTANCES SHAPE: ',distances.shape)
    averaged_time = search_time/len(query_embeddings)
    print(f"Average search time per query: {averaged_time:.4f} seconds")
    # Save indices and distances
    np.save(os.path.join(args.save_path.replace('.csv', '_indices.npy')), indices)
    np.save(os.path.join(args.save_path.replace('.csv', '_distances.npy')), distances)
    metric_type = index.metric_type

    # Print the metric used
    if metric_type == faiss.METRIC_L2:
        print("The index is using L2 (Euclidean Distance).")
        distances = -1*distances
    elif metric_type == faiss.METRIC_INNER_PRODUCT:
        print("The index is using Inner Product (Dot Product).")
    else:
        print(f"The index is using an unknown metric type: {metric_type}")
    # Process and Save Results
    results_df = map_results(indices, distances, qids, id_mapping, args.top_k)

    # Define evaluation metrics
    if args.ir_metrics:
        ir_metrics = args.ir_metrics
        print("Using custom IR metrics")
        print(f"Custom IR metrics: {ir_metrics}")
    else:
        print("Using default IR metrics")
        print("Default IR metrics: MAP@200, MAP, NDCG@3, P@1, P@3, R@1, R@5, R@10, R@200, R@500, MRR")
        ir_metrics  = [
            
            #map cut 200
            ir_measures.AP @ 200,        # Mean Average Precision (MAP)
            ir_measures.AP,        # Mean Average Precision (MAP)
            ir_measures.NDCG @ 3,  # Normalized Discounted Cumulative Gain @3
            ir_measures.P @ 1,     # Precision at 1
            ir_measures.P @ 3,     # Precision at 3
            ir_measures.R @ 1,     # Recall at 1
            ir_measures.R @ 5,     # Recall at 5
            ir_measures.R @ 10,    # Recall at 10
            ir_measures.R @ 200,   # Recall at 200
            ir_measures.R @ 500,   # Recall at 500
            ir_measures.MRR,       # Mean Reciprocal Rank
        ]
    results_ir = calculate_results(qrels, results_df, metrics=ir_metrics)
    
    print('Average IR metrics:', results_ir)
    
    print('Results per query:', results_df.head())
    if args.save_path:
        results_df.to_csv(args.save_path, index=False)
        print(f"Results saved to: {args.save_path}")
        ### Save IR metrics
        results_ir.to_csv(args.save_path.replace('.csv', '_mean.csv'), index=False)
        with open(args.save_path.replace('.csv', '_time.txt'), 'w') as f:
            f.write(f"Search time: {search_time:.2f} seconds")
            f.write(f"Average search time per query: {averaged_time:.4f} seconds")

    return results_df, results_ir



def generate_args_old(year,model,conversational_path,index_type='flat'):
    indexes = {
        'flat': 'flat_index',
        'hnsw': 'hnsw_index',
        'cosine': 'cosine_index'}
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
    if model == 'dragon':
        index_path = os.path.join(model_path, "flat_index/index_flat_ip.faiss")
    else:
        index_path = os.path.join(model_path, "flat_index/index_flat_cosine.faiss")
    embeddings_path = os.path.join(model_path, "passage_embeddings.npy")
    query_embeddings_path = os.path.join(model_path, "query_embeddings.npy")
    id_mapping_path = os.path.join(base_path, f"data/{base_paths[year]}_ID_Mapping.tsv")
    qrels_path = os.path.join(base_path, "data/topics/qrels.qrel")
    topics_path = os.path.join(base_path, "data/topics/topics.tsv")
    save_path = os.path.join(base_path, f"data/results/results_{model}.csv")
    return index_path,embeddings_path,query_embeddings_path,id_mapping_path,qrels_path,topics_path,save_path, model_name

def generate_args(year,model,conversational_path,index_type='flat',index_name='flat_index_ip.faiss', passage_embeddings_name='passage_embeddings.npy', query_embeddings_name='query_embeddings.npy'):
    indexes = {
        'flat': 'flat_index',
        'ivf': 'ivf_index',
        'hnsw': 'hnsw_index',
        'cosine': 'cosine_index'}
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
    return index_path,embeddings_path,query_embeddings_path,id_mapping_path,qrels_path,topics_path,save_path, model_name

def calculate_results(qrels, results_df, metrics= None):
    if metrics is None:
        print("No metrics specified, using default metrics")    
        metrics  = [
            ir_measures.NDCG @ 3,  # Normalized Discounted Cumulative Gain @3
            ir_measures.NDCG @ 10,  # Normalized Discounted Cumulative Gain @5
            ir_measures.R @ 100,    # Recall at 10
            ir_measures.MRR @ 1000,       # Mean Reciprocal Rank
        ]
    qrels_ir = qrels[['qid', 'doc_id', 'relevance']]
    qrels_ir.columns =['query_id', 'doc_id', 'relevance']
    ir_results_df = results_df[['qid', 'docno', 'score']]
    ir_results_df.columns = ['query_id', 'doc_id', 'score']
    # Compute IR metrics directly
    results_ir = ir_measures.calc_aggregate(metrics, qrels_ir, ir_results_df)
    results_ir = pd.DataFrame(results_ir, index=[0])
    return results_ir
#%%
'''
EXAMPLE OF USAGE
args = Namespace(
model_name = 'facebook/dragon-plus-query-encoder',
qrels_path = '/home/francomaria.nardini/raid/guidorocchietti/data/EFRA/dragon_embeddings/topics/qrels.qrel',
index_path = '/home/francomaria.nardini/raid/guidorocchietti/data/EFRA/dragon_embeddings/efra_manual_docs_flat.index',
top_k = 500)

llama_args = Namespace(
queries_path = '/home/francomaria.nardini/raid/guidorocchietti/data/EFRA/dragon_embeddings/topics/topics_llama70b.csv',
query_embeddings_path = '/home/francomaria.nardini/raid/guidorocchietti/data/EFRA/dragon_embeddings/llama_summary_embeddings.npy',
save_path = '/home/francomaria.nardini/raid/guidorocchietti/data/EFRA/dragon_embeddings/llama_results.csv',
id_mapping_path = '/home/francomaria.nardini/raid/guidorocchietti/data/EFRA/dragon_embeddings/ids_to_post_id.csv')
llama_args = argparse.Namespace(**{**vars(args), **vars(llama_args)})
search(llama_args)

'''