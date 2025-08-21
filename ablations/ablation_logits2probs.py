# %%
import os
import sys
sys.path.insert(0, "..")

token_path = os.path.join(os.path.dirname(__file__), '..', '.huggingface.token')
results_path = os.path.join(os.path.dirname(__file__), 'results', 'logits2probs')

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

# %% Load Pipeline:
import torch
from resources.generation import ExplainableAutoModelForGeneration, Focus

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
        max_samples=64,
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

# %%
import json
import matplotlib.pyplot as plt

with open(os.path.join(results_path, 'mae_offset.json'), 'r') as file:
    mae_offset = json.load(file)

with open(os.path.join(results_path, 'mae_relu.json'), 'r') as file:
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