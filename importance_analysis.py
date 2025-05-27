#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import tqdm
import os

scores_path = './index_snowflake/importance_scores/'
scores_files = [f for f in os.listdir(scores_path) if f.endswith('.pkl')]
scores_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sort by query ID

# %%
scores = []
for f in tqdm.tqdm(scores_files):
    with open(os.path.join(scores_path, f), 'rb') as file:
        score = pickle.load(file)
        scores.append(score)

# %%
def plot_importance(ax, scores, tokens, title):
    assert len(scores) == len(tokens)
    y = np.arange(len(scores))[::-1]
    ax.barh(y, scores)
    ax.set_yticks(y, labels=tokens)
    ax.set_title(title)
    

# %%
for i in range(len(scores)):
    fig, axs = plt.subplots(1, 11, figsize=(30, 50))
    plot_importance(axs[0], 
        scores = scores[i]['query'][0].mean(axis=0)[np.where(scores[i]['query'][0].mean(axis=0) != 0)[0]], 
        tokens = [x for x in scores[i]['query_in_tokens'][0] if x not in ['<pad>', 'sep', '<s>', '</s>']],
        title  = f'Query {i:d}:'
    )
    for j in range(len(scores[i]['context'])):
        plot_importance(axs[1+j], 
            scores = scores[i]['context'][j].mean(axis=0)[np.where(scores[i]['context'][j].mean(axis=0) != 0)[0]], 
            tokens = [x for x in scores[i]['context_in_tokens'][j] if x not in ['<pad>', 'sep', '<s>', '</s>']],
            title  = f'Context {j+1:d}:'
        )
    plt.tight_layout()
    plt.savefig(f'plots/importance_scores_query_{i}.png', bbox_inches='tight')

#%%
fig, axs = plt.subplots(1, 11, figsize=(30, 50))


plot_importance(axs[0], 
    scores = scores[0]['query'][0].mean(axis=0)[np.where(scores[0]['query'][0].mean(axis=0) != 0)[0]], 
    tokens = [x for x in scores[0]['query_in_tokens'][0] if x not in ['<pad>', 'sep', '<s>', '</s>']],
    title  = 'Query:'
)

for i in range(len(scores[0]['context'])):
    plot_importance(axs[1+i], 
        scores = scores[0]['context'][i].mean(axis=0)[np.where(scores[0]['context'][i].mean(axis=0) != 0)[0]], 
        tokens = [x for x in scores[0]['context_in_tokens'][i] if x not in ['<pad>', 'sep', '<s>', '</s>']],
        title  = f'Context {i+1:d}:'
    )

plt.tight_layout()
### save the figure to a file
plt.savefig('importance_scores_query_0.png', bbox_inches='tight')
plt.show()

# %%

# %%
