# %%% ===============================================================================================#
# Setup:                                                                                             #
#====================================================================================================#

import os
import sys
sys.path.insert(0, "../..")

# Paths:
TOKEN_PATH   = os.path.join(os.path.dirname(__file__), '..', '..', '.huggingface.token')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'results', 'additivity')
os.makedirs(RESULTS_PATH, exist_ok=True)

# Parameters:
PARAMS       = [{'num_steps':10,  'batch_size':16, 'verbose':False, 'base':'pad'},
                {'num_steps':50,  'batch_size':16, 'verbose':False, 'base':'pad'},
                {'num_steps':100, 'batch_size':16, 'verbose':False, 'base':'pad'},
                {'num_steps':200, 'batch_size':16, 'verbose':False, 'base':'pad'},
                {'num_steps':300, 'batch_size':16, 'verbose':False, 'base':'pad'},
                {'num_steps':400, 'batch_size':16, 'verbose':False, 'base':'pad'},
                {'num_steps':500, 'batch_size':16, 'verbose':False, 'base':'pad'}]
STEP_SIZE    = 1
NUM_DOCS     = 5
LONG_QUERIES = False

# %% Load data sample:
from utils import huggingface_login, load_ms_marco
huggingface_login(TOKEN_PATH)
sample = load_ms_marco(use_long_queries=LONG_QUERIES)

# %%
import os
import json
from tqdm.autonotebook import tqdm

def test(retriever, query_format, name):
    # Load file if it exists:
    values_path, values = os.path.join(RESULTS_PATH, f'values_{name}.json'), {}
    if os.path.exists(values_path):
        with open(values_path, 'r') as file:
            values = json.load(file)

    # Test methods:
    for kwargs in PARAMS:
        # get key:
        key = f'{kwargs["num_steps"]:d}'

        # skip method if already computed:
        if key in values:
            continue

        values[key] = {'query':[], 'context':[]}
        # calculate explanations:
        for qry, ctx in tqdm(zip(sample['query'], sample['context']), total=len(sample['query']), desc=f'Calcuclating additivity for n={key}'):
            # calculate similarity online:
            retrieved_ids, similarity = retriever.forward(
                query_format.format(query=qry), NUM_DOCS,
                contexts=ctx,
                reorder=True
            )

            # calculate explanation:
            _, coverage = retriever.intGrad(output_coverage=True, **kwargs)
            values[key]['query'].append(coverage['query'].tolist())
            values[key]['context'].append(coverage['context'].tolist())

        with open(values_path, 'w') as file:
            json.dump(values, file)

# %%% ===============================================================================================#
# Load Dragon Pipeline:                                                                              #
#====================================================================================================#

import torch
from src.Interpretable_RAG.retrieval_online import ExplainableAutoModelForRetrieval

retriever = ExplainableAutoModelForRetrieval.from_pretrained(
    'facebook/dragon-plus-query-encoder',
    'facebook/dragon-plus-context-encoder'
).to('cuda' if torch.cuda.is_available() else 'cpu')
retriever.tokenizer.model_max_length = 256

test(retriever, '{query}', 'dragon')

# %%% ===============================================================================================#
# Load Snowflake Pipeline:                                                                           #
#====================================================================================================#

import torch
from src.Interpretable_RAG.retrieval_online import ExplainableAutoModelForRetrieval

retriever = ExplainableAutoModelForRetrieval.from_pretrained(
    'Snowflake/snowflake-arctic-embed-l-v2.0',
    add_pooling_layer = False
).to('cuda' if torch.cuda.is_available() else 'cpu')

test(retriever, 'query: {query}', 'snowflake')

# %%% ===============================================================================================#
# Plots:                                                                                             #
#====================================================================================================#

import os
import json
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
from src.Interpretable_RAG.utils import bootstrap_ci

for name in ('dragon', 'snowflake'):

    with open(os.path.join(RESULTS_PATH, f'values_{name}.json'), 'r') as file:
        values = json.load(file)

    xs = np.array([int(x) for x in values.keys()])
    xs.sort()

    plt.figure(figsize=(4,4))
    for i, target in enumerate(('query', 'context')):
        ys = np.stack([np.array(values[str(x)][target]).flatten() for x in xs])
        ci = np.array([bootstrap_ci(y) for y in ys])
        plt.errorbar(xs-2.5+5*i, ys.mean(axis=1), np.abs(ci.T-ys.mean()), label=target)

        for x,y,c in zip(xs, ys.mean(axis=1), ci):
            print(f'{name}-{target} (n = {x}): {y:.2f} [{c[0]:.2f}, {c[1]:.2f}]')

    plt.ylabel('$\\left.{\\sum_{i=0}^n \\beta_{i,j}}~\\right/~{(\\bar{y}_j-y^0_j)}$')
    plt.xlabel('$L$')
    plt.legend()
    plt.savefig(f'additivity_{name}.pdf')