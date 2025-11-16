# %%
import os
import sys
sys.path.insert(0, "../..")

# paths:
TOKEN_PATH   = os.path.join(os.path.dirname(__file__), '..', '..', '.huggingface.token')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'results', 'shap_error')
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
from src.Interpretable_RAG.generation import ExplainableAutoModelForGeneration, Focus

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
    with open(os.path.join(RESULTS_PATH, 'val.json'), 'r') as file:
        val = json.load(file)

else: val = []

for query, contexts in tqdm(sample_texts[START:]):
    # Generate precise explanations:
    output = generator.explain_generate(
        query=query,
        contexts=contexts[:5],
        max_new_tokens=256,
        do_sample=False,
        top_p=1,
        num_beams=1,
        max_samples_query='auto',
        max_samples_context='inf',
        batch_size=64,
        conditional=True
    )

    # Get shapley values for the generated tokens:
    precise_val = generator.get_shapley_values('context', 'token')

    # Generate approximated explanations:
    assert output == generator.explain_generate(
        query=query,
        contexts=contexts[:5],
        max_new_tokens=256,
        do_sample=False,
        top_p=1,
        num_beams=1,
        max_samples_query='auto',
        max_samples_context=30,
        batch_size=64,
        conditional=True
    )

    # Get the shap parameters:
    indices = generator._shap_cache['context']['indices'].copy()
    sets    = generator._shap_cache['context']['sets'].copy()

    # Get shapley values for the generated tokens:
    val.append({})
    for max_samples in range(10, 31, 5):
        val[-1][max_samples] = {'precise': precise_val.tolist()}
    
        # Set the number of samples:
        sample_indices = np.random.choice(len(indices)-2, size=max_samples-2, replace=False) + 1 # skip the first and last indices
        generator._shap_cache['context']['indices'] = np.concatenate([indices[:1], indices[sample_indices], indices[-1:]])
        generator._shap_cache['context']['sets']    = np.concatenate([sets[:1], sets[sample_indices], sets[-1:]])

        # Get kernel shapley values:
        approx_val = generator.get_shapley_values('context', 'token', num_samples=1, sample_size=max_samples)
        val[-1][max_samples]['kernel'] = approx_val.tolist()

        # Get Monte Carlo approximated shapley values:
        val[-1][max_samples]['monte carlo'] = {}
        for num_mc_samples in range(20, 201, 20):
            # Get shapley values for the generated tokens:
            approx_val = generator.get_shapley_values('context', 'token', num_samples=num_mc_samples)
            val[-1][max_samples]['monte carlo'][num_mc_samples] = approx_val.tolist()

    # save:
    with open(os.path.join(RESULTS_PATH, 'val.json'), 'w') as file:
        json.dump(val, file)

#%%
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import wilcoxon, pearsonr

MAX_SAMPLES = str(20)
NUM_MC_SAMPLES = str(200)

def plot(ax, l, x, v_kl, v_mc, alternative):
    # plot lines:
    ax.plot(x, v_kl.mean(axis=0), marker='.', c='red', ls='--', label='Kernel Shap')
    ax.plot(x, v_mc.mean(axis=0), marker='.', c='tab:blue', label='Monte Carlo')

    # get axis dimensions:
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # print t-test statistics:
    for _x, _kl, _mc in zip(x, v_kl.T, v_mc.T):
        t, p = wilcoxon(_mc, _kl, alternative=alternative)
        ax.axvline(_x, color='gray', ls='--', lw=.5, zorder=0.)
        ax.text(_x, y_min + .05*(y_max-y_min), f'$p={p:.4f}$',
                ha='right', va='bottom', c='gray', size='small', rotation=90, zorder=0.)

    ax.set_xlim(x_min - .05*(x_max-x_min), x_max)

    # print label:
    ax.text(x_min, y_max - .05*(y_max-y_min), l,
            size='large', ha='center', va='top')

def plot_func(axs, ls, func, alternative):
    # load var:
    with open(os.path.join(RESULTS_PATH, 'val.json'), 'r') as file:
        var = json.load(file)

    # vs monte carlo samples @ sample size 20:
    x    = np.array([float(k) for k in var[0][MAX_SAMPLES]['monte carlo']])

    v_pr = [[var_[MAX_SAMPLES]['precise']] * len(x) for var_ in var]
    v_kl = [[var_[MAX_SAMPLES]['kernel']] * len(x) for var_ in var]
    v_mc = [[var_[MAX_SAMPLES]['monte carlo'][f'{k:.0f}'] for k in x] for var_ in var]

    y_kl = np.array(func(v_pr, v_kl))
    y_mc = np.array(func(v_pr, v_mc))
    plot(axs[0], ls[0], x, y_kl, y_mc, alternative)

    # vs sample size @ 200 monte carlo samples:
    x    = np.array([float(k) for k in var[0]])

    v_kl = [[var_[f'{k:.0f}']['kernel'] for k in x] for var_ in var]
    v_mc = [[var_[f'{k:.0f}']['monte carlo'][NUM_MC_SAMPLES] for k in  x] for var_ in var]

    y_kl = np.array(func(v_pr, v_kl))
    y_mc = np.array(func(v_pr, v_mc))
    plot(axs[1], ls[1], x, y_kl, y_mc, alternative)

    # get axis dimensions:
    dx = np.diff(axs[1].get_xlim())[0] * .01
    dy = np.diff(axs[1].get_ylim())[0] * .05

    # draw rectangle:
    x_rct = float(MAX_SAMPLES)
    y_rct = np.sort((y_kl.mean(axis=0)[x==x_rct][0], y_mc.mean(axis=0)[x==x_rct][0]))
    axs[1].add_patch(Rectangle((x_rct - dx, y_rct[0] - dy), 2*dx, 2*dy + np.diff(y_rct)[0], facecolor='none', edgecolor='black'))

    # draw text:
    axs[1].text(x_rct, y_rct[1] + dy, ls[0] + '\n↓', ha='center', va='bottom')

# %%
def get_doc_attribution(shap_values):
    # normalize:
    vals = shap_values / np.maximum(np.sum(shap_values, axis=0), 1e-9)

    # get most involved document:
    docs = vals.argmax(axis=0)

    # set documents with unclear attribution to -1:
    docs[vals[docs, np.arange(docs.shape[0])] < .5] = -1

    return docs

def attribution(precise, approx):
    r = []
    for item_prc, item_apr in zip(precise, approx):
        item = []
        for _prc, _apr in zip(item_prc, item_apr):
            _prc = get_doc_attribution(_prc)
            _apr = get_doc_attribution(_apr)

            item.append(np.mean(_prc == _apr))

        r.append(item)

    return r

def pearson(precise, approx):
    r = []
    for item_prc, item_apr in zip(precise, approx):
        item = []
        for _prc, _apr in zip(item_prc, item_apr):
            _prc = np.array(_prc)
            _apr = np.array(_apr)

            _prc = _prc / np.abs(_prc).sum(axis=0)
            _apr = _apr / np.abs(_prc).sum(axis=0)

            item.append(np.nanmean([pearsonr(_prc[i], _apr[i]).statistic for i in range(len(_prc))]))

        r.append(item)

    return r

def mse(precise, approx):
    r = []
    for item_prc, item_apr in zip(precise, approx):
        item = []
        for _prc, _apr in zip(item_prc, item_apr):
            _prc = np.array(_prc)
            _apr = np.array(_apr)

            _prc = _prc / np.abs(_prc).sum(axis=0)
            _apr = _apr / np.abs(_prc).sum(axis=0)

            diff = _prc-_apr
            #item.append(np.mean(diff@diff.T))
            item.append(np.mean(diff**2))

        r.append(item)

    return r

# %%
_, axs = plt.subplots(3,2, figsize=(8,6))

plot_func(axs[0], ['a)', 'b)'], pearson, alternative='greater')
plot_func(axs[1], ['c)', 'd)'], mse, alternative='less')
plot_func(axs[2], ['e)', 'f)'], attribution, alternative='greater')

axs[0,1].legend()

axs[0,0].tick_params(axis='x', direction='out', bottom=True, top=True, labelbottom=False, labeltop=True)
axs[0,1].tick_params(axis='x', direction='out', bottom=True, top=True, labelbottom=False, labeltop=True)

axs[1,0].tick_params(axis='x', direction='out', bottom=True, top=True, labelbottom=False, labeltop=False)
axs[1,1].tick_params(axis='x', direction='out', bottom=True, top=True, labelbottom=False, labeltop=False)

axs[2,0].tick_params(axis='x', direction='out', bottom=True, top=True, labelbottom=True, labeltop=False)
axs[2,1].tick_params(axis='x', direction='out', bottom=True, top=True, labelbottom=True, labeltop=False)

axs[0,0].set_ylabel('Pearson $r$')
axs[1,0].set_ylabel('MSE')
axs[2,0].set_ylabel('Document\nAttribution')

axs[2,0].set_xlabel('Number of MC samples')
axs[2,1].set_xlabel('Sample size')

plt.tight_layout()
plt.savefig('shap_error.pdf')
# %%
