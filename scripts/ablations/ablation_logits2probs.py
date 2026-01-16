# %%% ===============================================================================================#
# Setup:                                                                                             #
#====================================================================================================#

import os
import sys
sys.path.insert(0, "../..")

# Paths:
TOKEN_PATH   = os.path.join(os.path.dirname(__file__), '..', '..', '.huggingface.token')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'results', 'logits2probs')
os.makedirs(RESULTS_PATH, exist_ok=True)

# %% Load data sample:
from utils import huggingface_login, load_ms_marco
huggingface_login(TOKEN_PATH)
sample = load_ms_marco()

# %% Load pipeline:
import torch
from src.Interpretable_RAG.generation import ExplainableAutoModelForGeneration, Focus

generator = ExplainableAutoModelForGeneration.from_pretrained(
    pretrained_model_name_or_path='meta-llama/Llama-3.1-8B-Instruct',
    device_map='auto',
    torch_dtype=torch.bfloat16
)

# %%
import numpy as np
from tqdm.autonotebook import tqdm

mae = []
for item in tqdm(sample):
    output = generator.explain_generate(
        query=item['query'],
        contexts=item['passages']['passage_text'][:5],
        max_new_tokens=256,
        do_sample=False,
        top_p=1,
        num_beams=1,
        max_samples_query='auto',
        max_samples_context='inf',
        batch_size=64,
        conditional=True
    )

    # Get the number of generated tokens:
    num_tokens = len(generator.gen_tokens)
    num_choices = max(num_tokens - 5, 1)
    num_samples = min(50, num_choices)

    ae = np.empty((num_samples, 5))
    for i, start in enumerate(np.random.choice(np.arange(num_choices), size=num_samples, replace=False)):
        # Focus a random 5-token sequence:
        with Focus(generator, (start, start + 5)) as focus:
            # Get the normalized mean attribution values for tokens:
            attribution_token = focus.get_shapley_values('context', 'token').sum(axis=-1)
            attribution_token /= np.abs(attribution_token).sum()

            # Get the normalized the attribution values for the sequence:
            attribution_sequence = focus.get_shapley_values('context', 'sequence').squeeze()
            attribution_sequence /= np.abs(attribution_sequence).sum()

            # Calculate absolute error:
            ae[i] = np.abs(attribution_token - attribution_sequence)

    # Print the mean absolute error:
    mae.append(ae.mean())
    print(np.mean(ae))

# %%% ===============================================================================================#
# Plots:                                                                                             #
#====================================================================================================#

import json
import matplotlib.pyplot as plt

with open(os.path.join(RESULTS_PATH, 'mae_offset.json'), 'r') as file:
    mae_offset = json.load(file)

with open(os.path.join(RESULTS_PATH, 'mae_relu.json'), 'r') as file:
    mae_relu = json.load(file)

_, ax = plt.subplots(1,1, figsize=(4,4))

ax.boxplot([mae_offset, mae_relu], labels=['Offset', 'ReLU'])
ax.set_ylim(bottom=0)

# clean frame (keep only left spine):
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# add horizontal grid lines at each tick:
for tick_loc in ax.get_yticks():
    ax.axhline(y=tick_loc, color='darkgray', linewidth=0.5, zorder=0)

ax.set_ylabel('Mean Absolute Error')

plt.tight_layout()
plt.savefig('logits2probs.pdf')