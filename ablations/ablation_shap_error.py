# %%
import os
import sys
sys.path.insert(0, "..")

token_path = os.path.join(os.path.dirname(__file__), '..', '.huggingface.token')
results_path = os.path.join(os.path.dirname(__file__), 'results', 'shap')

#%%
import torch
from resources.generation import ExplainableAutoModelForGeneration
from resources.plotting import visualize_attribution_generator

# %%
from huggingface_hub import login
from getpass import getpass

if os.path.exists(token_path):
    with open(token_path, 'r') as file:
        login(token=file.read())

else: login(token=getpass(prompt='Huggingface login  token: '))

# %% Load the MS MARCO dataset:
from datasets import load_dataset

# Load the MS MARCO dataset version 2.1
dataset = load_dataset("ms_marco", "v2.1", split="train")

# Get a random sample of 200 documents
sample_texts = [(item['query'], item['passages']['passage_text']) for item in dataset.shuffle(seed=42).select(range(200))]

# %% Load Pipeline:
import torch
from resources.generation import ExplainableAutoModelForGeneration, Focus

generator = ExplainableAutoModelForGeneration.from_pretrained(
    pretrained_model_name_or_path='meta-llama/Llama-3.1-8B-Instruct',
    device_map='auto',
    torch_dtype=torch.bfloat16
)

# %%
import json
import numpy as np
from tqdm.autonotebook import tqdm

START = 101

if START > 0:
    with open(os.path.join(results_path, 'mae.json'), 'r') as file:
        mae = json.load(file)

    with open(os.path.join(results_path, 'mse.json'), 'r') as file:
        mse = json.load(file)

    with open(os.path.join(results_path, 'atr.json'), 'r') as file:
        atr = json.load(file)

else: mae, mse, atr = [], [], []

for query, contexts in tqdm(sample_texts[START:]):
    # Generate precise explanations:
    output = generator.explain_generate(
        query=query,
        contexts=contexts[:5],
        max_new_tokens=256,
        do_sample=False,
        top_p=1,
        num_beams=1,
        max_samples=64,
        batch_size=64,
        conditional=True
    )

    # Get shapley values for the generated tokens:
    precise_val = generator.get_shapley_values('context', 'token')
    precise_doc = precise_val.argmax(axis=0)

    # Generate approximated explanations:
    assert output == generator.explain_generate(
        query=query,
        contexts=contexts[:5],
        max_new_tokens=256,
        do_sample=False,
        top_p=1,
        num_beams=1,
        max_samples=60,
        batch_size=64,
        conditional=True
    )

    # Get the shap parameters:
    indices = generator._shap_cache['context']['indices'].copy()
    sets    = generator._shap_cache['context']['sets'].copy()

    # Get shapley values for the generated tokens:
    mae.append({})
    mse.append({})
    atr.append({})
    for max_samples in [30, 20, 10]:
        mae[-1][max_samples] = {}
        mse[-1][max_samples] = {}
        atr[-1][max_samples] = {}
        # Set the number of samples:
        sample_indices = np.random.choice(len(indices)-2, size=max_samples-2, replace=False) + 1 # skip the first and last indices
        generator._shap_cache['context']['indices'] = np.concatenate([indices[:1], indices[sample_indices], indices[-1:]])
        generator._shap_cache['context']['sets']    = np.concatenate([sets[:1], sets[sample_indices], sets[-1:]])

        # Get kernel shapley values:
        approx_val = generator.get_shapley_values('context', 'token', num_samples=1, sample_size=max_samples)
        approx_doc = approx_val.argmax(axis=0)
        mae[-1][max_samples]['kernel'] = np.abs(approx_val - precise_val).mean()
        mse[-1][max_samples]['kernel'] = ((approx_val - precise_val)**2).mean()
        atr[-1][max_samples]['kernel'] = np.mean(approx_doc == precise_doc)

        # Get Monte Carlo approximated shapley values:
        mae[-1][max_samples]['monte carlo'] = {}
        mse[-1][max_samples]['monte carlo'] = {}
        atr[-1][max_samples]['monte carlo'] = {}
        for num_mc_samples in range(20, 201, 20):
            # Get shapley values for the generated tokens:
            approx_val = generator.get_shapley_values('context', 'token', num_samples=num_mc_samples)
            approx_doc = approx_val.argmax(axis=0)
            mae[-1][max_samples]['monte carlo'][num_mc_samples] = np.abs(approx_val - precise_val).mean()
            mse[-1][max_samples]['monte carlo'][num_mc_samples] = ((approx_val - precise_val)**2).mean()
            atr[-1][max_samples]['monte carlo'][num_mc_samples] = np.mean(approx_doc == precise_doc)

    # save:
    with open(os.path.join(results_path, 'mae.json'), 'w') as file:
        json.dump(mae, file)

    with open(os.path.join(results_path, 'mse.json'), 'w') as file:
        json.dump(mse, file)

    with open(os.path.join(results_path, 'atr.json'), 'w') as file:
        json.dump(atr, file)

#%%
import json
import numpy as np
import matplotlib.pyplot as plt

# load:
with open(os.path.join(results_path, 'mae.json'), 'r') as file:
    mae = json.load(file)

with open(os.path.join(results_path, 'mse.json'), 'r') as file:
    mse = json.load(file)

with open(os.path.join(results_path, 'atr.json'), 'r') as file:
    atr = json.load(file)

mae_kernel, mae_mc = [], []
mse_kernel, mse_mc = [], []
atr_kernel, atr_mc = [], []

for mae_, mse_, atr_ in zip(mae, mse, atr):
    mae_kernel.append(mae_[str(30)]['kernel'])
    mse_kernel.append(mse_[str(30)]['kernel'])
    atr_kernel.append(atr_[str(30)]['kernel'])

    mae_mc.append(np.array([(float(k), v) for k,v in  mae_[str(30)]['monte carlo'].items()]))
    mse_mc.append(np.array([(float(k), v) for k,v in  mse_[str(30)]['monte carlo'].items()]))
    atr_mc.append(np.array([(float(k), v) for k,v in  atr_[str(30)]['monte carlo'].items()]))

_, axs = plt.subplots(3,1)

axs[0].axhline(np.mean(mae_kernel, axis=0), c='red', ls='--', label='Kernel Shap')
axs[0].plot(*np.mean(mae_mc, axis=0).T, c='tab:blue', label='Monte Carlo')
axs[0].set_ylabel('MAE')
axs[0].set_xlabel('Number of MC samples')
axs[0].legend()

axs[1].axhline(np.mean(mse_kernel, axis=0), c='red', ls='--', label='Kernel Shap')
axs[1].plot(*np.mean(mse_mc, axis=0).T, c='tab:blue', label='Monte Carlo')
axs[1].set_ylabel('MSE')
axs[1].set_xlabel('Number of MC samples')
axs[1].legend()

axs[2].axhline(np.mean(atr_kernel, axis=0), c='red', ls='--', label='Kernel Shap')
axs[2].plot(*np.mean(atr_mc, axis=0).T, c='tab:blue', label='Monte Carlo')
axs[2].set_ylabel('Attribution')
axs[2].set_xlabel('Number of MC samples')
axs[2].legend()

plt.tight_layout()
# %%
