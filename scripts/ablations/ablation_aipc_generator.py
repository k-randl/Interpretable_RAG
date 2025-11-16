# %%
import os
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
            scores = json.load(file)

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
        scores[method] = aipc.get_aipc(method)

        # plot pc:
        fig, ax = plt.subplots(1, 1)
        aipc.plot(ax, method)
        plt.savefig(os.path.join(RESULTS_PATH, f'{name}_{method}.pdf'))

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

methods = []
with open(os.path.join(RESULTS_PATH, 'curves_llama8b.pkl'), 'rb') as file:
    curves_llama8b = pickle.load(file)
    methods.extend(list(curves_llama8b.keys()))

fig, axs = plt.subplots(2, 2)
for method in set(methods):
    # Llama 8b:
    axs[0,0].plot(curves_llama8b[method]['xs'] * 100.,    curves_llama8b[method]['lerf'].mean(0) * 100.,    label=method)
    axs[1,0].plot(curves_llama8b[method]['xs'] * 100.,    curves_llama8b[method]['morf'].mean(0) * 100.)

    # Llama 8b:
    #axs[0,1].plot(curves_llama8b[method]['xs'] * 100.,    curves_llama8b[method]['lerf'].mean(0) * 100.)
    #axs[1,1].plot(curves_llama8b[method]['xs'] * 100.,    curves_llama8b[method]['morf'].mean(0) * 100.)

for row in range(1):
    # Paint arrow for LeRF plot:
    axs[0,row].arrow(50, 50, 30, -30, width=5, head_length=10, ec='white', color='lightblue')
    axs[0,row].text(65, 30, 'better', ha='center', va='bottom', rotation=-45, color='lightblue', zorder=0)

    # Paint arrow for MoRF plot:
    axs[1,row].arrow(50, 50, -30, 30 , width=5, head_length=10, ec='white', color='lightblue')
    axs[1,row].text(35, 60, 'better', ha='center', va='bottom', rotation=-45, color='lightblue', zorder=0)

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
axs[0,0].set_title('Llama-8B')
#axs[0,1].set_title('Snowflake')

# Deactivate third column:
axs[0,-1].axis('off')
axs[1,-1].axis('off')

fig.legend(loc='center right', ncols=1)
plt.tight_layout()
plt.savefig('aipc_generator.pdf')
