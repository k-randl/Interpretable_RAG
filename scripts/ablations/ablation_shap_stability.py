# %%% ===============================================================================================#
# Setup:                                                                                             #
#====================================================================================================#

import os
import sys
sys.path.insert(0, "../..")

# Paths:
TOKEN_PATH    = os.path.join(os.path.dirname(__file__), '..', '..', '.huggingface.token')
RESULTS_PATH  = os.path.join(os.path.dirname(__file__), 'results', 'shap_stability')
COMPLEMENTARY = True
os.makedirs(RESULTS_PATH, exist_ok=True)

# %% Load data sample:
from utils import huggingface_login, load_ms_marco
huggingface_login(TOKEN_PATH)
sample = load_ms_marco()

# %% Load pipeline:
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

RESTART = False

if   COMPLEMENTARY == True:    suffix = '_complementary'
elif COMPLEMENTARY == 'no_mc': suffix = '_comp_no_mc'
elif COMPLEMENTARY == False:   suffix = '_uniform'

if not RESTART:
    with open(os.path.join(RESULTS_PATH, f'std{suffix}.json'), 'r') as file:
        shap_std = json.load(file)

    with open(os.path.join(RESULTS_PATH, f'var{suffix}.json'), 'r') as file:
        shap_var = json.load(file)

    n = min(len(shap_std), len(shap_var))
    shap_std = shap_std[:n]
    shap_var = shap_var[:n]

else: shap_std, shap_var = [], []

for query, contexts in tqdm(sample_texts[len(shap_std):]):
    shap_std.append({})
    shap_var.append({})

    # Generate approximated explanations:
    generator.explain_generate(
        query=query,
        contexts=contexts[:5],
        max_new_tokens=256,
        do_sample=False,
        num_beams=1,
        max_samples_query='auto',
        max_samples_context=30,
        batch_size=64,
        conditional=True,
        complementary=COMPLEMENTARY
    )

    # Get the shap parameters:
    indices = generator._shap_cache['context']['indices'].copy()
    sets    = generator._shap_cache['context']['sets'].copy()

    approx = {}
    # Get shapley values for the generated tokens:
    for max_samples in range(10, 31, 5):
        approx_kernel = []
        approx_monte_carlo = {num_mc_samples:[] for num_mc_samples in range(20, 201, 20)}

        # calculate sample size and population:
        if COMPLEMENTARY != False:
            size = (max_samples // 2) - 1
            population = (len(indices) // 2) - 1    # skip the first and last indices

        else:
            size = max_samples - 2
            population = len(indices) - 2           # skip the first and last indices

        
        for i in tqdm(np.arange(10), desc=f'`max_samples` = {max_samples:d}'):
            # Get subsample of size max_samples:
            sample_indices = np.random.choice(population, size=size, replace=False) + 1
            sample_indices.sort()

            # Add complementary examples if necessary:
            if COMPLEMENTARY != False:
                sample_indices = np.concatenate([sample_indices, (len(indices)-1)-sample_indices])

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
    with open(os.path.join(RESULTS_PATH, f'std{suffix}.json'), 'w') as file:
        json.dump(shap_std, file)

    with open(os.path.join(RESULTS_PATH, f'var{suffix}.json'), 'w') as file:
        json.dump(shap_var, file)

# %%% ===============================================================================================#
# Plots:                                                                                             #
#====================================================================================================#

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import wilcoxon

MAX_SAMPLES = str(20)
NUM_MC_SAMPLES = str(200)

def plot(ax, id, x, v_kl, *vs_mc, labels=['Kernel-SHAP', 'Monte Carlo'], alternative='two-sided'):

    cmap = plt.colormaps.get_cmap('tab10')
    mc = [(v, l, cmap(i)) for i, (v, l) in enumerate(zip(vs_mc, labels[1:], strict=True))]

    # plot lines:
    ax.plot(x, v_kl.mean(axis=0), marker='.', c='red', ls='--', label=labels[0])
    for v, l, c in mc: ax.plot(x, v.mean(axis=0), marker='.', c=c, label=l)

    # get axis dimensions:
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # print t-test statistics:
    for i, (_x, _kl) in enumerate(zip(x, v_kl.T)):
        ax.axvline(_x, color='gray', ls='--', lw=.5, zorder=0.)
        txt = ax.text(_x, y_min + .05*(y_max-y_min), '$p=$',
                    ha='right', va='bottom', c='gray', size='small', rotation=90, zorder=0.)
        _y = txt.get_window_extent().transformed(ax.transData.inverted()).ymax

        for j, (v, _, c) in enumerate(mc):
            if j > 0:
                txt = ax.text(_x, _y, ' $|$ ',
                    ha='right', va='bottom', c='gray', size='small', rotation=90, zorder=0.)
                _y = txt.get_window_extent().transformed(ax.transData.inverted()).ymax

            t, p = wilcoxon(v[:,i], _kl, alternative=alternative)
            txt = ax.text(_x, _y, f'${p:.4f}$',
                    ha='right', va='bottom', c=c, alpha=0.5, size='small', rotation=90, zorder=0.)
            _y = txt.get_window_extent().transformed(ax.transData.inverted()).ymax

    ax.set_xlim(x_min - .05*(x_max-x_min), x_max)
        
    # print label:
    ax.text(x_min, y_max - .05*(y_max-y_min), id,
            size='large', ha='center', va='top')

def plot_var(axs, ls, *file_path, labels=['Kernel-SHAP', 'Monte Carlo'], alternative='two-sided'):
    # load files:
    files = []
    for fp in file_path:
        with open(os.path.join(RESULTS_PATH, fp), 'r') as file:
            files.append(json.load(file))

    # vs monte carlo samples @ sample size 20:
    x    = np.array([float(k) for k in files[0][0][MAX_SAMPLES]['monte carlo']])
    v_kl = [np.array([[np.mean(var_[MAX_SAMPLES]['kernel'])] * len(x) for var_ in file]) for file in files] 
    v_mc = [np.array([[np.mean(var_[MAX_SAMPLES]['monte carlo'][f'{k:.0f}']) for k in x] for var_ in file]) for file in files]
    plot(axs[0], ls[0], x, np.mean(v_kl, axis=0), *v_mc, labels=labels, alternative=alternative)

    # vs sample size @ 200 monte carlo samples:
    x    = np.array([float(k) for k in files[0][0]])
    v_kl = [np.array([[np.mean(var_[f'{k:.0f}']['kernel']) for k in x] for var_ in file]) for file in files] 
    v_mc = [np.array([[np.mean(var_[f'{k:.0f}']['monte carlo'][NUM_MC_SAMPLES]) for k in  x] for var_ in file]) for file in files] 
    plot(axs[1], ls[1], x, np.mean(v_kl, axis=0), *v_mc, labels=labels, alternative=alternative)

    # get axis dimensions:
    dx = np.diff(axs[1].get_xlim())[0] * .01
    dy = np.diff(axs[1].get_ylim())[0] * .05

    # draw rectangle:
    x_rct = float(MAX_SAMPLES)
    y_rct = np.sort([np.mean(v_kl, axis=(0,1))[x==x_rct][0],] + [v.mean(axis=0)[x==x_rct][0] for v in v_mc])
    axs[1].add_patch(Rectangle((x_rct - dx, y_rct[0] - dy), 2*dx, 2*dy + np.diff(y_rct)[0], facecolor='none', edgecolor='black'))

    # draw text:
    axs[1].text(x_rct, y_rct[1] + dy, ls[0] + '\n↓', ha='center', va='bottom')

#%%
_, axs = plt.subplots(2,2, figsize=(8,4))

plot_var(axs[0], ['a)', 'b)'], 'std_uniform.json', alternative='less')
plot_var(axs[1], ['c)', 'd)'], 'var_uniform.json', alternative='less')

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
plt.savefig(f'shap_stability_uniform.pdf')

#%%
_, axs = plt.subplots(2,2, figsize=(8,4))

plot_var(axs[0], ['a)', 'b)'], 'std_comp_no_mc.json', 'std_complementary.json', labels=['Kernel-SHAP', 'Monte Carlo', 'Paired Monte Carlo'], alternative='two-sided')
plot_var(axs[1], ['c)', 'd)'], 'var_comp_no_mc.json', 'var_complementary.json', labels=['Kernel-SHAP', 'Monte Carlo', 'Paired Monte Carlo'], alternative='two-sided')

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
plt.savefig(f'shap_stability_complementary.pdf')