# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7,8'
import sys
sys.path.insert(0, "../..")

# paths:
TOKEN_PATH   = os.path.join(os.path.dirname(__file__), '..', '..', '.huggingface.token')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'results', 'aipc_generator')
os.makedirs(RESULTS_PATH, exist_ok=True)

# parameters:
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
sample = sample.map(lambda item: {'query':item['query'], 'context':item['passages']['passage_text'][:NUM_DOCS]})

# %%
import os
import json
import pickle
import matplotlib.pyplot as plt

def test(aipc, name):
    # Load curves file if it exists:
    curves_path, curves = os.path.join(RESULTS_PATH, f'curves_{name}.pkl'), {}
    if os.path.exists(curves_path):
        with open(curves_path, 'rb') as file:
            curves = pickle.load(file)

    # Load scores file if it exists:
    scores_path, scores = os.path.join(RESULTS_PATH, f'scores_{name}.json'), {}
    if os.path.exists(scores_path):
        with open(scores_path, 'r') as file:
            scores = json.dump(file)

    # run faithfullness test:
    aipc(sample,
        step=STEP_SIZE,
        do_sample=False,
        top_p=1,
        num_beams=1,
        max_new_tokens=256
    )

    # Test methods:
    for method in aipc.morf:

        # save curves:
        curves[method] = {
            'xs':   aipc.xs,
            'morf': aipc.morf[method],
            'lerf': aipc.lerf[method]
        }
        with open(curves_path, 'wb') as file:
            pickle.dump(curves, file)

        # save scores:
        scores[method] = {}
        for k in range(NUM_DOCS):
            # get aipc:
            scores[method][f'k = {k+1:d}'] = aipc.get_aipc(method, k=k+1)

            # plot pc:
            fig, ax = plt.subplots(1, 1)
            aipc.plot(ax, method, k=k+1)
            plt.savefig(os.path.join(RESULTS_PATH, f'{name}_{method}_k{k+1:d}.pdf'))

        with open(scores_path, 'w') as file:
            json.dump(scores, file)

# %%% ===============================================================================================#
# Load Llama 3.1 8B Pipeline:                                                                        #
#====================================================================================================#

import torch
from src.Interpretable_RAG.generation import ExplainableAutoModelForGeneration
from src.faithfullness.generation import AIPCForGeneration

generator = ExplainableAutoModelForGeneration.from_pretrained(
    pretrained_model_name_or_path='meta-llama/Llama-3.1-8B-Instruct',
    device_map='auto',
    torch_dtype=torch.bfloat16
)

# %%
aipc = AIPCForGeneration(generator)
# run faithfullness test:
test(aipc, 'llama8b')

# %%
import json
import pickle
import matplotlib.pyplot as plt

with open(os.path.join(RESULTS_PATH, 'curves_dragon.pkl'), 'rb') as file:
    curves_dragon = pickle.load(file)

with open(os.path.join(RESULTS_PATH, 'curves_snowflake.pkl'), 'rb') as file:
    curves_snowflake = pickle.load(file)

fig, axs = plt.subplots(2, 2)
for method in set(list(curves_dragon.keys()) + list(curves_snowflake.keys())):
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
# %%
