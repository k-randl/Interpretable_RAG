#%%
import sys
sys.path.insert(0, "..")

import torch
import numpy as np
import matplotlib.pyplot as plt
<<<<<<<< HEAD:experiments_offline_snowflake.py
from resources.modelling_offline import ExplainableAutoModelForContextEncoding, ExplainableAutoModelForRAG
import pandas as pd
========
from resources.retrieval_offline import ExplainableAutoModelForContextEncoding, ExplainableAutoModelForRAG

>>>>>>>> 757d7d70de75a0c8cd8ceb93339eed0034b753a3:demos/explain_rag_dense_offline_snowflake.py
# %%
# We use msmarco query and passages as an example
df_for_testing = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/code/efra_retrieval/validation_Dataset_with_chunks_ids.csv')
contexts = df_for_testing['text'].unique().tolist()
df_for_testing_first_ten_queries = df_for_testing[df_for_testing['query'].isin(df_for_testing['query'].unique()[:10])]
unique_contexts = df_for_testing_first_ten_queries['text'].unique().tolist()
# %%
enc = ExplainableAutoModelForContextEncoding.from_pretrained(
    'Snowflake/snowflake-arctic-embed-l-v2.0',
<<<<<<<< HEAD:experiments_offline_snowflake.py
    add_pooling_layer = False,

)
enc.to('cuda' if torch.cuda.is_available() else 'cpu')
### use data parallelism
#%%
#from torch.nn.parallel import DataParallel
#enc = DataParallel(enc, device_ids=[0, 1, 2, 3])  # Adjust device_ids based on your setup

========
    add_pooling_layer = False
).to('cuda' if torch.cuda.is_available() else 'cpu')
>>>>>>>> 757d7d70de75a0c8cd8ceb93339eed0034b753a3:demos/explain_rag_dense_offline_snowflake.py

# Calculate similarity:
enc.save_index(unique_contexts, 32, dir='index_snowflake', output_attentions=True, output_hidden_states=True, max_length=1024)

# %%
do_evaluate = False
if not do_evaluate: exit()
#%%
rag = ExplainableAutoModelForRAG.from_pretrained(
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
