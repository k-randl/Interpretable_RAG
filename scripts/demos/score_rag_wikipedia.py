# %%
import os
import sys
sys.path.insert(0, "../..")

import torch
from src.Interpretable_RAG.rag import WARGScorer
from src.Interpretable_RAG.plotting import plot_document_importance_rag

from huggingface_hub import login
from getpass import getpass

# %%
TOKEN_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '.huggingface.token')
if os.path.exists(TOKEN_PATH):
    with open(TOKEN_PATH, 'r') as file:
        login(token=file.read())

else: login(token=getpass(prompt='Huggingface login  token: '))

#%%
INDEX_PATH = '.wiki_chunks'
MAX_SEQ_LEN = 1024
MAX_GEN_LEN = 256

# %% Load Pipeline:
scorer = WARGScorer(INDEX_PATH,
    query_encoder_name_or_path='Snowflake/snowflake-arctic-embed-l-v2.0',
    retriever_query_format='query: {query}',
    retriever_kwargs={'add_pooling_layer':False},

    generator_name_or_path='meta-llama/Llama-3.1-8B-Instruct',
    generator_kwargs={'device_map':'auto', 'dtype':torch.bfloat16}
)

# %% Score all queries based on the index documents:
queries = ["How did Marie Curie's husband die?", "Where was Marie Curie born?"]
scorer(queries, k=10, batch_size=32)

#%% All rag plotting functions can also be used with WARGScorer
plot_document_importance_rag(scorer)

# %% Saving:
scorer.dump('.scorer.json')

# %% Loading:
scorer = WARGScorer.load('.scorer.json')