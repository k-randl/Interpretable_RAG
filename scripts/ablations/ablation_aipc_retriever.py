# %%
import os
import sys
sys.path.insert(0, "../..")

# paths:
TOKEN_PATH   = os.path.join(os.path.dirname(__file__), '..', '..', '.huggingface.token')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'results', 'aipc_retriever')
os.makedirs(RESULTS_PATH, exist_ok=True)

# parameters:
METHODS      = [('intGrad', {'num_steps':100, 'batch_size':16, 'verbose':False, 'base':None}),
                ('intGrad', {'num_steps':100, 'batch_size':16, 'verbose':False, 'base':'mask'}),
                ('intGrad', {'num_steps':100, 'batch_size':16, 'verbose':False, 'base':'unk'}),
                ('intGrad', {'num_steps':100, 'batch_size':16, 'verbose':False, 'base':'pad'}),
                ('intGrad', {'num_steps':50,  'batch_size':16, 'verbose':False, 'base':'pad'}),
                ('intGrad', {'num_steps':10,  'batch_size':16, 'verbose':False, 'base':'pad'}),
                ('gradIn',  {}),
                ('aGrad',   {}),
                ('grad',    {}),
                ('random',  {})]
STEP_SIZE    = 10
NUM_DOCS     = 5
LONG_QUERIES = False

# %%
from huggingface_hub import login
from getpass import getpass

if os.path.exists(TOKEN_PATH):
    with open(TOKEN_PATH, 'r') as file:
        login(token=file.read())

else: login(token=getpass(prompt='Huggingface login  token: '))

# %% Load the MS MARCO dataset:
import json
from datasets import load_dataset

# Load the MS MARCO dataset version 2.1
dataset = load_dataset("ms_marco", "v2.1", split="train")

# Get a random sample of 200 documents
sample = dataset.shuffle(seed=42).select(range(200))

# Set dataset format:
if LONG_QUERIES:
    with open('queries.json', 'r') as file:
        queries = json.load(file) 

    sample = sample.map(lambda item: {'query':queries[item['query']], 'context':item['passages']['passage_text']})

else: sample = sample.map(lambda item: {'query':item['query'], 'context':item['passages']['passage_text']})

# %%
import os
import json
import pickle
import matplotlib.pyplot as plt

def test(aipc, name, target):
    # Load curves file if it exists:
    curves_path, curves = os.path.join(RESULTS_PATH, f'curves_{name}_{target}_rephrased.pkl'), {}
    if os.path.exists(curves_path):
        with open(curves_path, 'rb') as file:
            curves = pickle.load(file)

    # Load scores file if it exists:
    scores_path, scores = os.path.join(RESULTS_PATH, f'scores_{name}_{target}_rephrased.json'), {}
    if os.path.exists(scores_path):
        with open(scores_path, 'r') as file:
            scores = json.load(file)

    # Test methods:
    for method, kwargs in METHODS:
        # get key:
        if method == 'intGrad': key = f'intGrad (n = {kwargs["num_steps"]:d}{", ["+kwargs["base"]+"]" if kwargs["base"] is not None else ""})'
        else:                   key = method

        # skip method if already computed:
        if (key in curves) and (key in scores):
            continue

        # run faithfullness test:
        aipc(sample,
            target=target,
            method=method, method_args=kwargs,
            step=STEP_SIZE, k=NUM_DOCS,
            output_attentions=True, output_hidden_states=True,
            max_length=128
        )

        # save curves:
        curves[key] = {
            'xs':   aipc.xs,
            'morf': aipc.morf,
            'lerf': aipc.lerf
        }
        with open(curves_path, 'wb') as file:
            pickle.dump(curves, file)

        # save scores:
        scores[key] = aipc.get_aipc()

        # plot pc:
        fig, ax = plt.subplots(1, 1)
        aipc.plot(ax)

        file_name = f'{name}_{method}_{target}'

        if method == 'intGrad':
            if kwargs['base'] is not None:
                file_name = file_name + f'_{kwargs["base"]}'

            file_name = file_name + f'_n{kwargs["num_steps"]:d}'

        plt.savefig(os.path.join(RESULTS_PATH, file_name + f'_rephrased.pdf'))

        with open(scores_path, 'w') as file:
            json.dump(scores, file)

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
test(aipc, 'dragon', 'query')
test(aipc, 'dragon', 'context')

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
test(aipc, 'snowflake', 'query')
test(aipc, 'snowflake', 'context')

# %%
import json
import pickle
import matplotlib.pyplot as plt

plots = {
    'Baseline': [
        ('random', 'Random'),
        ('intGrad (n = 100)', 'Zeros'),
        ('intGrad (n = 100, [mask])', '[mask]'),
        ('intGrad (n = 100, [pad])', '[pad]'),
        ('intGrad (n = 100, [unk])', '[unk]'),
    ],
    'Iterations': [
        ('random', 'Random'),
        ('intGrad (n = 10, [pad])',  'n = 10'),
        ('intGrad (n = 50, [pad])',  'n = 50'),
        ('intGrad (n = 100, [pad])', 'n = 100'),
    ],
    'Methods': [
        ('random', 'Random'),
        ('grad', 'Grad'),
        ('gradIn', 'GradIn'),
        ('aGrad', 'AGrad'),
        ('intGrad (n = 100, [unk])', 'IntGrad'),
    ],
}

for target in ('query', 'context'):
    for key in plots:
        with open(os.path.join(RESULTS_PATH, f'curves_dragon_{target}.pkl'), 'rb') as file:
            curves_dragon = pickle.load(file)

        with open(os.path.join(RESULTS_PATH, f'curves_snowflake_{target}.pkl'), 'rb') as file:
            curves_snowflake = pickle.load(file)

        fig, axs = plt.subplots(2, 3)
        for method, label in plots[key]:
            # Dragon:
            if method in curves_dragon:
                axs[0,0].plot(curves_dragon[method]['xs'] * 100.,    curves_dragon[method]['lerf'].mean(axis=0) * 100., label=label)
                axs[1,0].plot(curves_dragon[method]['xs'] * 100.,    curves_dragon[method]['morf'].mean(axis=0) * 100.)

            # Snowflake:
            if method in curves_snowflake:
                axs[0,1].plot(curves_snowflake[method]['xs'] * 100., curves_snowflake[method]['lerf'].mean(axis=0) * 100.)
                axs[1,1].plot(curves_snowflake[method]['xs'] * 100., curves_snowflake[method]['morf'].mean(axis=0) * 100.)


        for row in range(2):
            # Paint arrow for LeRF plot:
            axs[0,row].arrow(50, -50, 30, 30, width=5, head_length=10, ec='white', color='lightblue')
            axs[0,row].text(60, -40, 'better', ha='center', va='bottom', rotation=45, color='lightblue', zorder=0)

            # Paint arrow for MoRF plot:
            axs[1,row].arrow(50, -50, -30, -30 , width=5, head_length=10, ec='white', color='lightblue')
            axs[1,row].text(30, -70, 'better', ha='center', va='bottom', rotation=45, color='lightblue', zorder=0)

            # Set aspect ratio to 1. on all plots:
            axs[0,row].set_aspect(1)
            axs[1,row].set_aspect(1)

            # Set x-labels:
            axs[1,row].set_xlabel('Masked Tokens [%]')
            axs[0,row].set_xticklabels([])

        # Set y-labels:
        axs[0,0].set_ylabel('Normalized $\Delta$ LeRF [%]')
        axs[1,0].set_ylabel('Normalized $\Delta$ MoRF [%]')
        axs[0,1].set_yticklabels([])
        axs[1,1].set_yticklabels([])

        # Set titles:
        axs[0,0].set_title('Dragon')
        axs[0,1].set_title('Snowflake')

        # Deactivate third column:
        axs[0,-1].axis('off')
        axs[1,-1].axis('off')

        fig.legend(loc='center right', ncols=1)
        plt.tight_layout()
        plt.savefig(f'aipc_retriever_{target}_{key}.pdf')