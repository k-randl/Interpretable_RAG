#%%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.Interpretable_RAG.tools import *
from src.Interpretable_RAG.search_tools import *

import os
from argparse import Namespace
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,4,5,6,7"

#%%
# Load the embeddings
conversational_path = '/home/francomaria.nardini/raid/guidorocchietti/data/conversational/'
dragon_index = load_faiss_index('/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2019/passage_embeddings/dragon-plus-context-encoder/flat_index/index_flat_ip.faiss')
snowflake_index = load_faiss_index('/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2019/passage_embeddings/snowflake-arctic-embed-l-v2.0/flat_index/index_flat_cosine.faiss')
id_mapping = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2019/data/CAST2019_ID_Mapping.tsv', sep = '\t')
#id_mapping_20 = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2020/data/CAST2020_ID_Mapping.tsv', sep = '\t')
#%%
#%%
for year in [2019,2020]:
    for model in ['dragon','snowflake']:
        for topic_type in ['manual']:
            index_path,embeddings_path,query_embeddings_path,id_mapping_path,qrels_path,topics_path,save_path,model_name  = generate_args(year,model,conversational_path,index_type='flat',index_name='index_flat_ip.faiss')
            ir_metrics  = [
                    ir_measures.NDCG @ 3,  # Normalized Discounted Cumulative Gain @3
                    ir_measures.NDCG @ 10,  # Normalized Discounted Cumulative Gain @5
                    ir_measures.MRR @ 10,       # Mean Reciprocal Rank
                    ir_measures.P @ 10,  # Precision at 10
                    ir_measures.P @ 3,     # Precision at 10
                    ir_measures.P @ 1,     # Precision at 10
                    ir_measures.R @ 10,    # Recall at 10
                    ]
            if topic_type == 'raw':
                topics_path = topics_path.replace('.tsv','_raw.tsv')
                save_path = save_path.replace('.csv','_raw.csv')
                query_embeddings_path = query_embeddings_path.replace('query_embeddings.npy','query_raw_embeddings.npy')
            if model == 'dragon':
                args = Namespace(queries_path = topics_path, qrels_path = qrels_path, index = dragon_index, query_embeddings_path = query_embeddings_path, id_mapping = id_mapping, model_name = model_name, top_k = 10, save_path = save_path, sep = None, ir_metrics =ir_metrics ,help=None)
            elif model == 'snowflake':
                args = Namespace(queries_path = topics_path, qrels_path = qrels_path, index = snowflake_index, query_embeddings_path = query_embeddings_path, id_mapping = id_mapping, model_name = model_name, top_k = 10, save_path = save_path, sep = None, ir_metrics =ir_metrics, help=None)
            search(args)
      
# %%
dragon_index = load_faiss_index('/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2019/passage_embeddings/dragon-plus-context-encoder/flat_index/index_transformed_flat_l2.faiss')
id_mapping = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2019/data/CAST2019_ID_Mapping.tsv', sep = '\t')
conversational_path = '/home/francomaria.nardini/raid/guidorocchietti/data/conversational/'
#%%
year = 2019
model = 'dragon'
ir_metrics  = [    
ir_measures.NDCG @ 3,  # Normalized Discounted Cumulative Gain @3
ir_measures.NDCG @ 10,    # Recall at 5
ir_measures.R @ 100,    # Recall at 10
ir_measures.MRR @ 1000,       # Mean Reciprocal Rank
]
index_path,embeddings_path,query_embeddings_path,id_mapping_path,qrels_path,topics_path,save_path,model_name  = generate_args(year,model,conversational_path)
query_embeddings_path = query_embeddings_path.replace('query_embeddings.npy','query_embeddings_transformed.npy')
save_path = save_path.replace('.csv','_transformed.csv')
args = Namespace(queries_path = topics_path, qrels_path = qrels_path, index = dragon_index, query_embeddings_path = query_embeddings_path, id_mapping = id_mapping, model_name = model_name, top_k = 1000, save_path = save_path, sep = None, ir_metrics =ir_metrics ,help=None)
search(args)
# %%
