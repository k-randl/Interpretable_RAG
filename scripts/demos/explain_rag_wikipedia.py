# %%
import os
import sys
sys.path.insert(0, "../..")

import torch
from src.Interpretable_RAG.rag import ExplainableAutoModelForRAG
from src.Interpretable_RAG.plotting import higlight_importance_rag, plot_document_importance_rag, visualize_importance_retriever, visualize_attribution_generator

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
model = ExplainableAutoModelForRAG(
    query_encoder_name_or_path='Snowflake/snowflake-arctic-embed-l-v2.0',
    index = INDEX_PATH,

    retriever_query_format='query: {query}',
    retriever_token_processor=lambda s: s.replace('▁', ' '),
    retriever_kwargs={'add_pooling_layer':False, 'dtype':torch.bfloat16},

    generator_name_or_path='meta-llama/Llama-3.1-8B-Instruct',
    generator_token_processor=lambda s: s.replace('Ġ', ' ').strip('Ċ'),
    generator_kwargs={'device_map':'auto', 'dtype':torch.bfloat16}
)

# We use msmarco query as an example
query =  "Where was Marie Curie born?"

# %% Load the Marie Curie Wikipedia page as context data:
if not os.path.exists(INDEX_PATH):
    import wikipedia
    from src.Interpretable_RAG.data import load_html

    # get page:
    wikipedia.set_lang('en')
    page = wikipedia.page('Marie Curiie', auto_suggest=True)

    # cut into chunks of 20 tokens:
    chunks = []
    for text, i, j in load_html(html=page.html(), window=20, handle_wiki_tags=True):
        if len(text) > 0:
            chunks.append(text)

    # compute index:
    model.retriever.compute_index(chunks, batch_size=16, save_folder=INDEX_PATH)

# %%
output = model(
    query=query,
    k=10,
    generator_kwargs={
        'max_new_tokens':MAX_GEN_LEN,
        'do_sample':False,
        'top_p':1,
        'num_beams':1,
        'batch_size':32,
        'max_samples_query':32,
        'max_samples_context':32,
        'conditional':True
    }
)
output

#%%
plot_document_importance_rag(model)

#%%
higlight_importance_rag(
    model,
    retriever_token_processor=lambda s: s.replace('▁', ' '),
    generator_token_processor=lambda s: s.replace('Ġ', ' ').strip('Ċ'),
    batch_size=8,
    output_format='latex'
)

# %%
model.retriever_query_importance
# %%
model.generator_query_importance
# %%
model.mean_query_importance