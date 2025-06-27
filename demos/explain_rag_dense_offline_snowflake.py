#%%
import sys
sys.path.insert(0, "..")

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from resources.retrieval_offline import ExplainableAutoModelForContextEncoding, ExplainableAutoModelForRetrieval

# %%
# We use msmarco query and passages as an example
query =  "Where was Marie Curie born?"
contexts = [
    "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
]

# %%
enc = ExplainableAutoModelForContextEncoding.from_pretrained(
    'Snowflake/snowflake-arctic-embed-l-v2.0',
    add_pooling_layer = False
).to('cuda' if torch.cuda.is_available() else 'cpu')

# Calculate similarity:
enc.save_index(contexts, 32, dir='index_snowflake', output_attentions=True, output_hidden_states=True, max_length=1024)

# %%
do_evaluate = False
if not do_evaluate: exit()
#%%
rag = ExplainableAutoModelForRetrieval.from_pretrained(
    'Snowflake/snowflake-arctic-embed-l-v2.0',
    add_pooling_layer = False
).to('cuda' if torch.cuda.is_available() else 'cpu')

# Calculate similarity:
rag('query: ' + query, 2, dir='index_snowflake', output_attentions=True, output_hidden_states=True)

#%%
def plot_importance(ax, scores, tokens, title):
    assert len(scores) == len(tokens)
    y = np.arange(len(scores))[::-1]
    ax.barh(y, scores)
    ax.set_yticks(y, labels=tokens)
    ax.set_title(title)

#%% aGrad:
fig, axs = plt.subplots(1, 3)

plot_importance(axs[0], 
    scores = rag.aGrad()['query'][0].mean(axis=0), 
    tokens = rag.in_tokens['query'][0],
    title  = 'Query:'
)

for i in range(len(contexts)):
    plot_importance(axs[1+i], 
        scores = rag.aGrad()['context'][i].mean(axis=0), 
        tokens = rag.in_tokens['context'][i],
        title  = f'Context {i+1:d}:'
    )

plt.tight_layout()
plt.show()

#%% Grad x In:
fig, axs = plt.subplots(1, 3)

plot_importance(axs[0], 
    scores = rag.gradIn()['query'][0], 
    tokens = rag.in_tokens['query'][0],
    title  = 'Query:'
)

for i in range(len(contexts)):
    plot_importance(axs[1+i], 
        scores = rag.gradIn()['context'][i], 
        tokens = rag.in_tokens['context'][i],
        title  = f'Context {i+1:d}:'
    )

plt.tight_layout()
plt.show()

#%% Grad:
fig, axs = plt.subplots(1, 3)

plot_importance(axs[0],
    scores = rag.grad()['query'][0].mean(axis=-1), 
    tokens = rag.in_tokens['query'][0],
    title  = 'Query:'
)

for i in range(len(contexts)):
    plot_importance(axs[1+i],
        scores = rag.grad()['context'][i].mean(axis=-1), 
        tokens = rag.in_tokens['context'][i],
        title  = f'Context {i+1:d}:'
    )

plt.tight_layout()
plt.show()
# %%
