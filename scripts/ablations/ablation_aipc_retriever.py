# %%
import os
import sys
sys.path.insert(0, "../..")

# paths:
TOKEN_PATH   = os.path.join(os.path.dirname(__file__), '..', '..', '.huggingface.token')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'results', 'aipc_retriever')
os.makedirs(RESULTS_PATH, exist_ok=True)

# parameters:
METHODS      = {'intGrad':{'num_steps':100, 'batch_size':32, 'verbose':False},
                'intGrad':{'num_steps':50,  'batch_size':32, 'verbose':False},
                'gradIn': {}}
STEP_SIZE    = 1
NUM_DOCS     = 5

# %%
from huggingface_hub import login
from getpass import getpass

if os.path.exists(TOKEN_PATH):
    with open(TOKEN_PATH, 'r') as file:
        login(token=file.read())

else: login(token=getpass(prompt='Huggingface login  token: '))

# %% Load the MS MARCO dataset:
from datasets import load_dataset

# Load the MS MARCO dataset version 2.1
dataset = load_dataset("ms_marco", "v2.1", split="train")

# Get a random sample of 200 documents
sample = dataset.shuffle(seed=42).select(range(200))

# Set dataset format:
sample = sample.map(lambda item: {'query':item['query'], 'context':item['passages']['passage_text']})

# %%% ===============================================================================================#
# Load Dragon Pipeline:                                                                              #
#====================================================================================================#

import torch
from src.Interpretable_RAG.retrieval_online import ExplainableAutoModelForRetrieval
from src.faithfullness.retrieval import AIPCForRetrieval

retriever = ExplainableAutoModelForRetrieval.from_pretrained(
    'facebook/dragon-plus-query-encoder',
    'facebook/dragon-plus-context-encoder'
).to('cuda' if torch.cuda.is_available() else 'cpu')
retriever.tokenizer.model_max_length = 256

aipc = AIPCForRetrieval(retriever, query_format='{query}')

# %%
import json
import pickle
import matplotlib.pyplot as plt

scores_dragon = {m:{} for m in METHODS}
curves_dragon = {m:{} for m in METHODS}

for method in METHODS:
    # get key:
    if method == 'intGrad': key = f'intGrad (n = {METHODS[method]["num_steps"]:d})'
    else:                   key = method

    # run faithfullness test:
    aipc(sample, method=method, method_args=METHODS[method], step=STEP_SIZE, k=NUM_DOCS)

    for k in range(NUM_DOCS):
        # get aipc:
        scores_dragon[key][f'k = {k+1:d}'] = aipc.get_aipc(k=k+1)
        curves_dragon[key] = {
            'xs':   aipc.xs,
            'morf': aipc.morf,
            'lerf': aipc.lerf
        }

        # plot pc:
        fig, ax = plt.subplots(1, 1)
        aipc.plot(ax, k=k+1)
        if method == 'intGrad': plt.savefig(os.path.join(RESULTS_PATH, f'dragon_{key}_n{METHODS[method]["num_steps"]:d}_k{k+1:d}.pdf'))
        else:                   plt.savefig(os.path.join(RESULTS_PATH, f'dragon_{key}_k{k+1:d}.pdf'))

with open(os.path.join(RESULTS_PATH, 'scores_dragon.json'), 'w') as file:
    json.dump(scores_dragon, file)

with open(os.path.join(RESULTS_PATH, 'curves_dragon.json'), 'wb') as file:
    pickle.dump(curves_dragon, file)


# %%% ===============================================================================================#
# Load Snowflake Pipeline:                                                                           #
#====================================================================================================#

import torch
from src.Interpretable_RAG.retrieval_online import ExplainableAutoModelForRetrieval
from src.faithfullness.retrieval import AIPCForRetrieval

retriever = ExplainableAutoModelForRetrieval.from_pretrained(
    'Snowflake/snowflake-arctic-embed-l-v2.0',
    add_pooling_layer = False
).to('cuda' if torch.cuda.is_available() else 'cpu')

aipc = AIPCForRetrieval(retriever, query_format='query: {query}')

# %%
import json
import pickle
import matplotlib.pyplot as plt

scores_snowflake = {m:{} for m in METHODS}
curves_snowflake = {m:{} for m in METHODS}

for method in METHODS:
    # get key:
    if method == 'intGrad': key = f'intGrad (n = {METHODS[method]["num_steps"]:d})'
    else:                   key = method

    # run faithfullness test:
    aipc(sample, method=method, method_args=METHODS[method], step=STEP_SIZE, k=NUM_DOCS)

    for k in range(NUM_DOCS):
        # get aipc:
        scores_snowflake[method][f'k = {k+1:d}'] = aipc.get_aipc(k=k+1)
        curves_snowflake[method] = {
            'xs':   aipc.xs,
            'morf': aipc.morf,
            'lerf': aipc.lerf
        }

        # plot pc:
        fig, ax = plt.subplots(1, 1)
        aipc.plot(ax, k=k+1)
        if method == 'intGrad': plt.savefig(os.path.join(RESULTS_PATH, f'snowflake_{key}_n{METHODS[method]["num_steps"]:d}_k{k+1:d}.pdf'))
        else:                   plt.savefig(os.path.join(RESULTS_PATH, f'snowflake_{key}_k{k+1:d}.pdf'))

with open(os.path.join(RESULTS_PATH, 'scores_snowflake.json'), 'w') as file:
    json.dump(scores_snowflake, file)

with open(os.path.join(RESULTS_PATH, 'curves_snowflake.json'), 'wb') as file:
    pickle.dump(curves_snowflake, file)

# %%
import json
import pickle
import matplotlib.pyplot as plt

with open(os.path.join(RESULTS_PATH, 'curves_dragon.json'), 'rb') as file:
    curves_dragon = pickle.load(file)

with open(os.path.join(RESULTS_PATH, 'curves_snowflake.json'), 'rb') as file:
    curves_snowflake = pickle.load(file)

fig, axs = plt.subplots(2, 2)
for method in METHODS:
    # Dragon:
    axs[0,0].plot(curves_dragon[method]['xs'] * 100.,    curves_dragon[method]['lerf'].mean(axis=0) * 100.,    label=method)
    axs[1,0].plot(curves_dragon[method]['xs'] * 100.,    curves_dragon[method]['morf'].mean(axis=0) * 100.,    label=method)

    # Snowflake:
    axs[0,1].plot(curves_snowflake[method]['xs'] * 100., curves_snowflake[method]['lerf'].mean(axis=0) * 100., label=method)
    axs[1,1].plot(curves_snowflake[method]['xs'] * 100., curves_snowflake[method]['morf'].mean(axis=0) * 100., label=method)

axs[0,0].set_aspect(1)
axs[0,1].set_aspect(1)
axs[1,0].set_aspect(1)
axs[1,1].set_aspect(1)

axs[0,0].set_title('Dragon')
axs[0,1].set_title('Snowflake')

axs[0,0].set_ylabel('Normalized $\Delta$ LeRF [%]')
axs[1,0].set_ylabel('Normalized $\Delta$ MoRF [%]')
axs[0,1].set_yticklabels([])
axs[1,1].set_yticklabels([])

axs[1,0].set_xlabel('Masked Tokens [%]')
axs[1,1].set_xlabel('Masked Tokens [%]')
axs[0,0].set_xticklabels([])
axs[0,1].set_xticklabels([])

axs[0,1].legend()
plt.tight_layout()
plt.savefig('aipc_retriever.pdf')