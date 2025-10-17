# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import sys
sys.path.insert(0, "..")

token_path = os.path.join(os.path.dirname(__file__), '..', '.huggingface.token')
results_path = os.path.join(os.path.dirname(__file__), 'results', 'aipc_retriever')

os.makedirs(results_path, exist_ok=True)

# %%
from huggingface_hub import login
from getpass import getpass

if os.path.exists(token_path ):
    with open(token_path , 'r') as file:
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
from resources.retrieval_online import ExplainableAutoModelForRetrieval
from resources.faithfulllness import AIPCForRetrieval

retriever = ExplainableAutoModelForRetrieval.from_pretrained(
    'facebook/dragon-plus-query-encoder',
    'facebook/dragon-plus-context-encoder'
).to('cuda' if torch.cuda.is_available() else 'cpu')

aipc = AIPCForRetrieval(retriever, query_format='{query}')

# %%
import json
import matplotlib.pyplot as plt

results_dragon = {'gradIn':{}, 'intGrad':{}}

for method in results_dragon:
    aipc(sample, method=method, step=1, k=5)

    for k in range(5):
        # get aipc:
        results_dragon[method][f'k = {k+1:d}'] = aipc.get_aipc(k=k+1)

        # plot pc:
        fig, ax = plt.subplots(1, 1)
        aipc.plot(ax, k=k+1)
        plt.safefig(os.path.join(results_path, f'dragon_{method}_k{k+1:d}.pdf'))

with open(os.path.join(results_path, 'aipc.json'), 'w') as file:
    json.dump(results_dragon, file)

# %%% ===============================================================================================#
# Load Snowflake Pipeline:                                                                           #
#====================================================================================================#

import torch
from resources.retrieval_online import ExplainableAutoModelForRetrieval
from resources.faithfulllness import AIPCForRetrieval

retriever = ExplainableAutoModelForRetrieval.from_pretrained(
    'Snowflake/snowflake-arctic-embed-l-v2.0',
    add_pooling_layer = False
).to('cuda' if torch.cuda.is_available() else 'cpu')

aipc = AIPCForRetrieval(retriever, query_format='query: {query}')

# %%
import json
import matplotlib.pyplot as plt

results_snowflake = {'gradIn':{}, 'intGrad':{}}

for method in results_snowflake:
    aipc(sample, method=method, step=1, k=5)

    for k in range(5):
        # get aipc:
        results_snowflake[method][f'k = {k+1:d}'] = aipc.get_aipc(k=k+1)

        # plot pc:
        fig, ax = plt.subplots(1, 1)
        aipc.plot(ax, k=k+1)
        plt.safefig(os.path.join(results_path, f'snowflake_{method}_k{k+1:d}.pdf'))

with open(os.path.join(results_path, 'aipc.json'), 'w') as file:
    json.dump(results_snowflake, file)