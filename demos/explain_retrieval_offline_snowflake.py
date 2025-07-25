#%%
import sys
sys.path.insert(0, "..")

import torch
from resources.plotting import plot_importance_retriever, higlight_importance_retriever
from resources.retrieval_offline import ExplainableAutoModelForContextEncoding, ExplainableAutoModelForRetrieval

# %%
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
enc = ExplainableAutoModelForContextEncoding.from_pretrained(
    'Snowflake/snowflake-arctic-embed-l-v2.0',
    add_pooling_layer = False
).to('cuda' if torch.cuda.is_available() else 'cpu')

# Calculate similarity:
enc.save_index(contexts, 32, dir='index_snowflake', output_attentions=True, output_hidden_states=True)

#%%
rag = ExplainableAutoModelForRetrieval.from_pretrained(
    'Snowflake/snowflake-arctic-embed-l-v2.0',
    add_pooling_layer = False
).to('cuda' if torch.cuda.is_available() else 'cpu')

# Calculate similarity:
rag('query: ' + query, 2, dir='index_snowflake', output_attentions=True, output_hidden_states=True)

#%% aGrad:
plot_importance_retriever(rag, method='aGrad')
higlight_importance_retriever(rag, method='aGrad', token_processor=lambda s: s.replace('▁', ' '))

#%% Grad x In:
plot_importance_retriever(rag, method='gradIn')
higlight_importance_retriever(rag, method='gradIn', token_processor=lambda s: s.replace('▁', ' '))

#%% Grad:
plot_importance_retriever(rag, method='grad')
higlight_importance_retriever(rag, method='grad', token_processor=lambda s: s.replace('▁', ' '))