# %%
import os
import sys
sys.path.insert(0, "../..")

# paths:
TOKEN_PATH   = os.path.join(os.path.dirname(__file__), '..', '..', '.huggingface.token')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'results', 'shap_stability')
os.makedirs(RESULTS_PATH, exist_ok=True)

#%%
import torch
from src.Interpretable_RAG.generation import ExplainableAutoModelForGeneration

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
sample_texts = [(item['query'], item['passages']['passage_text']) for item in dataset.shuffle(seed=42).select(range(200))]

# %% Load Pipeline:
import torch
from src.Interpretable_RAG.generation import ExplainableAutoModelForGeneration

generator = ExplainableAutoModelForGeneration.from_pretrained(
    pretrained_model_name_or_path='meta-llama/Llama-3.1-8B-Instruct',
    device_map='auto',
    torch_dtype=torch.bfloat16
)

# %%
import json
import numpy as np
from tqdm.autonotebook import tqdm

START = 0

if START > 0:
    with open(os.path.join(RESULTS_PATH, 'std.json'), 'r') as file:
        shap_std = json.load(file)

    with open(os.path.join(RESULTS_PATH, 'var.json'), 'r') as file:
        shap_var = json.load(file)

else: shap_std, shap_var = [], []

for query, contexts in tqdm(sample_texts[START:]):
    shap_std.append({})
    shap_var.append({})

    # Generate approximated explanations:
    generator.explain_generate(
        query=query,
        contexts=contexts[:5],
        max_new_tokens=256,
        do_sample=False,
        num_beams=1,
        max_samples=60,
        batch_size=64,
        conditional=True
    )

    # Get the shap parameters:
    indices = generator._shap_cache['context']['indices'].copy()
    sets    = generator._shap_cache['context']['sets'].copy()

    approx = {}
    # Get shapley values for the generated tokens:
    for max_samples in range(10, 31, 5):
        approx_kernel = []
        approx_monte_carlo = {num_mc_samples:[] for num_mc_samples in range(20, 201, 20)}

        for i in range(10):

            # Get subsample of size max_samples:
            sample_indices = np.random.choice(len(indices)-2, size=max_samples-2, replace=False) + 1 # skip the first and last indices
            sample_indices.sort()

            sample_indices = [0] + sample_indices.tolist() + [len(indices) - 1] 

            # Set the number of samples:
            generator._shap_cache['context']['indices'] = indices[sample_indices]
            generator._shap_cache['context']['sets']    = sets[sample_indices]

            # Get kernel shapley values:
            approx_kernel.append(
                generator.get_shapley_values('context', 'token', num_samples=1, sample_size=max_samples)
            )

            # Get Monte Carlo approximated shapley values:
            for num_mc_samples in approx_monte_carlo:
                approx_monte_carlo[num_mc_samples].append(
                    generator.get_shapley_values('context', 'token', num_samples=num_mc_samples)
                )

        # Calculate standard deviation:
        shap_std[-1][max_samples] = {
            'kernel':      np.std(approx_kernel, axis=0).tolist(),
            'monte carlo': {num_mc_samples: np.std(approx_monte_carlo[num_mc_samples], axis=0).tolist()
                            for num_mc_samples in approx_monte_carlo}
        }

        # Calculate variance:
        shap_var[-1][max_samples] = {
            'kernel':      np.var(approx_kernel, axis=0).tolist(),
            'monte carlo': {num_mc_samples: np.var(approx_monte_carlo[num_mc_samples], axis=0).tolist()
                            for num_mc_samples in approx_monte_carlo}
        }

    # save:
    with open(os.path.join(RESULTS_PATH, 'std.json'), 'w') as file:
        json.dump(shap_std, file)

    with open(os.path.join(RESULTS_PATH, 'var.json'), 'w') as file:
        json.dump(shap_var, file)

#%%
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import ttest_ind

MAX_SAMPLES = str(20)
NUM_MC_SAMPLES = str(200)

def plot(ax, l, x, v_kl, v_mc):
    # plot lines:
    ax.plot(x, v_kl.mean(axis=0), marker='.', c='red', ls='--', label='Kernel Shap')
    ax.plot(x, v_mc.mean(axis=0), marker='.', c='tab:blue', label='Monte Carlo')

    # get axis dimensions:
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # print t-test statistics:
    for _x, _kl, _mc in zip(x, v_kl.T, v_mc.T):
        t, p = ttest_ind(_mc, _kl, alternative='less')
        ax.axvline(_x, color='gray', ls='--', lw=.5, zorder=0.)
        ax.text(_x, y_min + .05*(y_max-y_min), f'$p={p:.4f}$',
                ha='right', va='bottom', c='gray', size='small', rotation=90, zorder=0.)

    ax.set_xlim(x_min - .05*(x_max-x_min), x_max)
        
    # print label:
    ax.text(x_min, y_max - .05*(y_max-y_min), l,
            size='large', ha='center', va='top')

def plot_var(axs, ls, file_path):
    # load var:
    with open(os.path.join(RESULTS_PATH, file_path), 'r') as file:
        var = json.load(file)

    # vs monte carlo samples @ sample size 20:
    x    = np.array([float(k) for k in var[0][MAX_SAMPLES]['monte carlo']])
    v_kl = np.array([[np.mean(var_[MAX_SAMPLES]['kernel'])] * len(x) for var_ in var])
    v_mc = np.array([[np.mean(var_[MAX_SAMPLES]['monte carlo'][f'{k:.0f}']) for k in x] for var_ in var])
    plot(axs[0], ls[0], x, v_kl, v_mc)

    # vs sample size @ 200 monte carlo samples:
    x    = np.array([float(k) for k in var[0]])
    v_kl = np.array([[np.mean(var_[f'{k:.0f}']['kernel']) for k in x] for var_ in var])
    v_mc = np.array([[np.mean(var_[f'{k:.0f}']['monte carlo'][NUM_MC_SAMPLES]) for k in  x] for var_ in var])
    plot(axs[1], ls[1], x, v_kl, v_mc)

    # get axis dimensions:
    dx = np.diff(axs[1].get_xlim())[0] * .01
    dy = np.diff(axs[1].get_ylim())[0] * .05

    # draw rectangle:
    x_rct = float(MAX_SAMPLES)
    y_rct = np.sort((v_kl.mean(axis=0)[x==x_rct][0], v_mc.mean(axis=0)[x==x_rct][0]))
    axs[1].add_patch(Rectangle((x_rct - dx, y_rct[0] - dy), 2*dx, 2*dy + np.diff(y_rct)[0], facecolor='none', edgecolor='black'))

    # draw text:
    axs[1].text(x_rct, y_rct[1] + dy, ls[0] + '\n↓', ha='center', va='bottom')

#%%
_, axs = plt.subplots(2,2, figsize=(8,4))

plot_var(axs[0], ['a)', 'b)'], 'std.json')
plot_var(axs[1], ['c)', 'd)'], 'var.json')

axs[0,1].legend()

axs[0,0].tick_params(axis='x', direction='out', bottom=True, top=True, labelbottom=False, labeltop=True)
axs[0,1].tick_params(axis='x', direction='out', bottom=True, top=True, labelbottom=False, labeltop=True)

axs[1,0].tick_params(axis='x', direction='out', bottom=True, top=True, labelbottom=True, labeltop=False)
axs[1,1].tick_params(axis='x', direction='out', bottom=True, top=True, labelbottom=True, labeltop=False)

axs[0,0].set_ylabel('$\sigma$')
axs[1,0].set_ylabel('$\sigma^2$')

axs[1,0].set_xlabel('Number of MC samples')
axs[1,1].set_xlabel('Sample size')

plt.tight_layout()
plt.savefig('shap_stability.pdf')