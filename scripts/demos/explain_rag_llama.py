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
MAX_SEQ_LEN = 1024
MAX_GEN_LEN = 256
# %% Load Pipeline:
model = ExplainableAutoModelForRAG(
    query_encoder_name_or_path='Snowflake/snowflake-arctic-embed-l-v2.0',
    dir='./demos/index_snowflake',

    retriever_query_format='query: {query}',
    retriever_token_processor=lambda s: s.replace('▁', ' '),
    retriever_kwargs={'add_pooling_layer':False},

    generator_name_or_path='meta-llama/Llama-3.1-8B-Instruct',
    generator_token_processor=lambda s: s.replace('Ġ', ' ').strip('Ċ'),
    generator_kwargs={'device_map':'auto', 'dtype':torch.bfloat16}
)

# We use msmarco query and passages as an example
query =  "Where was Marie Curie born?"
contexts = [
    "Maria Skłodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace.",
    "Maria Skłodowska was born in Warsaw, in Congress Poland in the Russian Empire, as the fifth and youngest child of well-known teachers Bronisława, née Boguska, and Władysław Skłodowski.",
    "While a French citizen, Marie Skłodowska Curie, who used both surnames, never lost her sense of Polish identity. She taught her daughters the Polish language and took them on visits to Poland.",
    "Marie Curie founded the Curie Institute in Paris in 1920, and the Curie Institute in Warsaw in 1932.",
]

# %%
output = model(
    query=query,
    contexts=contexts,
    k=5,
    generator_kwargs={
        'max_new_tokens':MAX_GEN_LEN,
        'do_sample':False,
        'top_p':1,
        'num_beams':1,
        'batch_size':64,
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
    generator_token_processor=lambda s: s.replace('Ġ', ' ').strip('Ċ')
)

# %%
visualize_importance_retriever(
    model.retriever,
    token_processor=lambda s: s.replace('▁', ' ')
)

# %%
visualize_attribution_generator(
    model.generator,
    token_processor=lambda s: s.replace('Ġ', ' ').strip('Ċ')
)

# %%
model.retriever_query_importance
# %%
model.generator_query_importance
# %%
model.mean_query_importance
# %%
model.query_agreement
# %%
model.document_agreement