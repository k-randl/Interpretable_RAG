#%%
import torch
from transformers import AutoTokenizer
from resources.retrieval_offline import ExplainableAutoModelForContextEncoding, ExplainableAutoModelForRetrieval
tokenizer = AutoTokenizer.from_pretrained('facebook/dragon-plus-query-encoder')
query_encoder = ExplainableAutoModelForContextEncoding.from_pretrained('facebook/dragon-plus-query-encoder')
context_encoder = ExplainableAutoModelForRetrieval.from_pretrained('facebook/dragon-plus-context-encoder')

# We use msmarco query and passages as an example
query =  "Where was Marie Curie born?"
contexts = [
    "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
]
# Apply tokenizer
query_input = tokenizer(query, return_tensors='pt')
ctx_input = tokenizer(contexts, padding=True, truncation=True, return_tensors='pt')
# Compute embeddings: take the last-layer hidden state of the [CLS] token
query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]
ctx_emb = context_encoder(**ctx_input).last_hidden_state[:, 0, :]
#%%
import matplotlib.pyplot as plt

def decode_tokens(input_ids):
    return [[tokenizer.decode(token) for token in text] for text in input_ids]

#%% attention rollout:
fig, axs = plt.subplots(3,1)

tokens = decode_tokens(query_input.input_ids)[0]
axs[0].imshow(query_encoder.attentionRollout()[:,0])
axs[0].set_xticks(range(len(tokens)), labels=tokens)
axs[0].set_yticks([])
axs[0].set_title('Query:')

for i in range(ctx_emb.shape[0]):
    tokens = decode_tokens(ctx_input.input_ids)[i]
    axs[i+1].imshow(context_encoder.attentionRollout()[i,0].reshape(1,-1))
    axs[i+1].set_xticks(range(len(tokens)), labels=tokens, rotation=90)
    axs[i+1].set_yticks([])
    axs[i+1].set_title(f'Context {i+1:d}:')

plt.show()

#%% aGrad:
fig, axs = plt.subplots(3,1)

tokens = decode_tokens(query_input.input_ids)[0]
axs[0].imshow(query_encoder.aGrad().mean(axis=1)[:,0])
axs[0].set_xticks(range(len(tokens)), labels=tokens)
axs[0].set_yticks([])
axs[0].set_title('Query:')

for i in range(ctx_emb.shape[0]):
    tokens = decode_tokens(ctx_input.input_ids)[i]
    axs[i+1].imshow(context_encoder.aGrad().mean(axis=1)[i,0].reshape(1,-1))
    axs[i+1].set_xticks(range(len(tokens)), labels=tokens, rotation=90)
    axs[i+1].set_yticks([])
    axs[i+1].set_title(f'Context {i+1:d}:')

plt.show()

#%% Grad x In:
fig, axs = plt.subplots(3,1)

tokens = decode_tokens(query_input.input_ids)[0]
axs[0].imshow(query_encoder.gradIn())
axs[0].set_xticks(range(len(tokens)), labels=tokens)
axs[0].set_yticks([])

for i in range(ctx_emb.shape[0]):
    tokens = decode_tokens(ctx_input.input_ids)[i]
    axs[i+1].imshow(context_encoder.gradIn()[i].reshape(1,-1))
    axs[i+1].set_xticks(range(len(tokens)), labels=tokens, rotation=90)
    axs[i+1].set_yticks([])
    axs[i+1].set_title(f'Context {i+1:d}:')

plt.show()
#%%
snowflake_embeddings_path = '/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2019/passage_embeddings/snowflake-arctic-embed-l-v2.0/passage_embeddings.npy'
import numpy as np
import os
if os.path.exists(snowflake_embeddings_path):
    snowflake_embeddings = np.load(snowflake_embeddings_path)
    
from resources.tools import *

create_faiss_index_flat(snowflake_embeddings, '/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2019/passage_embeddings/snowflake-arctic-embed-l-v2.0/flat_index/snowflake_index_flat.index')
# %%
from resources.search_tools import *
from argparse import Namespace
year = 2019
model = 'snowflake'
conversational_path = '/home/francomaria.nardini/raid/guidorocchietti/data/conversational/'
# Generate the arguments for the search


index_path,embeddings_path,query_embeddings_path,id_mapping_path,qrels_path,topics_path,save_path, model_name = generate_args(year,model,conversational_path,index_type='flat',index_name='flat_index_ip.faiss', passage_embeddings_name='passage_embeddings.npy', query_embeddings_name='query_embeddings.npy')
id_mapping = pd.read_csv(id_mapping_path, sep='\t')
snowflake_index = load_faiss_index('/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2019/passage_embeddings/snowflake-arctic-embed-l-v2.0/flat_index/snowflake_index_flat.index')
ir_metrics  = [
    ir_measures.NDCG @ 3,  # Normalized Discounted Cumulative Gain @3
    ir_measures.NDCG @ 10,  # Normalized Discounted Cumulative Gain @5
    ir_measures.MRR @ 10,       # Mean Reciprocal Rank
    ir_measures.P @ 10,  # Precision at 10
    ir_measures.P @ 3,     # Precision at 10
    ir_measures.P @ 1,     # Precision at 10    
    ir_measures.R @ 10,    # Recall at 10
]

args = Namespace(queries_path = topics_path, qrels_path = qrels_path, index = snowflake_index, query_embeddings_path = query_embeddings_path, id_mapping = id_mapping, model_name = model_name, top_k = 10, save_path = save_path, sep = None, ir_metrics =ir_metrics, help=None)


results_df, results_ir = search(args)
# %%
'''python generate_index.py \
    --embeddings_path /path/to/your/passage_embeddings.npy \
    --save_path /path/to/output/my_index.faiss \
    --metric IP'''