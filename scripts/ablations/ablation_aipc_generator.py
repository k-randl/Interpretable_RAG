# %%% ===============================================================================================#
# Setup:                                                                                             #
#====================================================================================================#

import os
import sys
sys.path.insert(0, "../..")

# Paths:
TOKEN_PATH   = os.path.join(os.path.dirname(__file__), '..', '..', '.huggingface.token')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'results', 'aipc_generator')
COMPLEMENTARY = True
os.makedirs(RESULTS_PATH, exist_ok=True)

# Parameters:
STEP_SIZE    = 1
NUM_DOCS     = 5

# %% Load data sample:
from utils import huggingface_login, load_ms_marco
huggingface_login(TOKEN_PATH)
sample = load_ms_marco()

# %%
import os
import json
import pickle
import matplotlib.pyplot as plt

def test(aipc, name, target):
    # get suffix:
    if   COMPLEMENTARY == True:    suffix = 'complementary'
    elif COMPLEMENTARY == 'no_mc': suffix = 'comp_no_mc'
    elif COMPLEMENTARY == False:   suffix = 'uniform'

    # Load curves file if it exists:
    curves_path, curves = os.path.join(RESULTS_PATH, f'curves_{name}_{target}_{suffix}.pkl'), {}
    if os.path.exists(curves_path): return

    # Load scores file if it exists:
    scores_path, scores = os.path.join(RESULTS_PATH, f'scores_{name}_{target}_{suffix}.json'), {}
    if os.path.exists(scores_path): return

    # run faithfullness test:
    aipc(sample,
        step=STEP_SIZE,
        do_sample=False,
        top_p=1,
        num_beams=1,
        max_new_tokens=256,
        complementary=COMPLEMENTARY,
        tmp_file=f'aipc_generator_{suffix}_tmp.pkl'
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
        plt.savefig(os.path.join(RESULTS_PATH, f'{name}_{target}_{method}_{suffix}.pdf'))

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
test(aipc, 'llama8b', 'context')

# %%% ===============================================================================================#
# Plots:                                                                                             #
#====================================================================================================#

import json
import pickle
import matplotlib.pyplot as plt

plots = {
    'comparison_n15': [
        ('Random', 'Random'),
        ('Kernel (n = 15)', 'Kernel'),
        ('Monte Carlo (n = 15)', 'Monte Carlo'),
        ('Precise', 'Precise'),
    ],
    'comparison_n20': [
        ('Random', 'Random'),
        ('Kernel (n = 20)', 'Kernel'),
        ('Monte Carlo (n = 20)', 'Monte Carlo'),
        ('Precise', 'Precise'),
    ],
    'comparison_n25': [
        ('Random', 'Random'),
        ('Kernel (n = 25)', 'Kernel'),
        ('Monte Carlo (n = 25)', 'Monte Carlo'),
        ('Precise', 'Precise'),
    ],
    'comparison_n30': [
        ('Random', 'Random'),
        ('Kernel (n = 30)', 'Kernel'),
        ('Monte Carlo (n = 30)', 'Monte Carlo'),
        ('Precise', 'Precise'),
    ],
    'kernel': [
        ('Kernel (n = 10)', 'n=10'),
        ('Kernel (n = 15)', 'n=15'),
        ('Kernel (n = 20)', 'n=20'),
        ('Kernel (n = 25)', 'n=25'),
        ('Kernel (n = 30)', 'n=30'),
    ],
    'mc': [
        ('Monte Carlo (n = 10)', 'n=10'),
        ('Monte Carlo (n = 15)', 'n=15'),
        ('Monte Carlo (n = 20)', 'n=20'),
        ('Monte Carlo (n = 25)', 'n=25'),
        ('Monte Carlo (n = 30)', 'n=30'),
    ],
}

model = 'llama8b'
for target in ('query', 'context'):
    for key in plots:
        suffixes = ['_'.join(fp[:-4].split('_')[3:]) for fp in os.listdir(RESULTS_PATH) if fp.startswith(f'curves_{model}_{target}')]
        if len(suffixes) == 0: continue

        fig, axs = plt.subplots(2, len(suffixes)+1)
        for method, label in plots[key]:
            set_label = True
            for col, suffix in enumerate(suffixes):
                with open(os.path.join(RESULTS_PATH, f'curves_{model}_{target}_{suffix}.pkl'), 'rb') as file:
                    curves = pickle.load(file)

                # Llama 8b:
                axs[0,col].plot(curves[method]['xs'] * 100.,    curves[method]['lerf'].mean(0) * 100.,    label=label if set_label else None)
                axs[1,col].plot(curves[method]['xs'] * 100.,    curves[method]['morf'].mean(0) * 100.)
                set_label = False

                # Set titles:
                axs[0,col].set_title(f'Llama-8B\n{suffix}')

                # Paint arrow for LeRF plot:
                axs[0,col].arrow(50, 50, 30, -30, width=5, head_length=10, ec='white', color='lightblue')
                axs[0,col].text(65, 30, 'better', ha='center', va='bottom', rotation=-45, color='lightblue', zorder=0)

                # Paint arrow for MoRF plot:
                axs[1,col].arrow(50, 50, -30, 30 , width=5, head_length=10, ec='white', color='lightblue')
                axs[1,col].text(35, 60, 'better', ha='center', va='bottom', rotation=-45, color='lightblue', zorder=0)

                # Set aspect ratio to 1. on all plots:
                axs[0,col].set_aspect(1)
                axs[1,col].set_aspect(1)

                # Set x-labels:
                axs[1,col].set_xlabel('Masked Tokens [%]')
                axs[0,col].set_xticklabels([])

                # Set y-labels:
                if col == 0:
                    axs[0,col].set_ylabel('Normalized $\Delta$ LeRF [%]')
                    axs[1,col].set_ylabel('Normalized $\Delta$ MoRF [%]')
                else:
                    axs[0,col].set_yticklabels([])
                    axs[1,col].set_yticklabels([])

        # Deactivate third column:
        axs[0,-1].axis('off')
        axs[1,-1].axis('off')

        fig.legend(loc='center right', ncols=1)
        plt.tight_layout()
        plt.savefig(f'aipc_generator_{target}_{key}.pdf')

#%%
import json
import numpy as np
import matplotlib.pyplot as plt

model = 'llama8b'
for target in ('query', 'context'):
    plt.figure(figsize=(4,7))
    # load scores:
    try:
        with open(os.path.join(RESULTS_PATH, f'scores_{model}_{target}_uniform.json'), 'r') as file:
            scores_uniform = json.load(file)

        with open(os.path.join(RESULTS_PATH, f'scores_{model}_{target}_complementary.json'), 'r') as file:
            scores_complementary = json.load(file)

        with open(os.path.join(RESULTS_PATH, f'scores_{model}_{target}_comp_no_mc.json'), 'r') as file:
            scores_comp_no_mc = json.load(file)
    except FileNotFoundError: continue

    # get x values:
    xs  = [10, 15, 20, 25, 30]

    # Precise and Random:
    prc = scores_uniform['Precise']
    rnd = scores_uniform['Random']
    
    plt.axhline(prc[0], ls='--', c='red', label="Precise")
    #plt.axhline(rnd[0], ls='--', c='grey', label="Random")

    # kernel methods:
    klu = np.array([scores_uniform[f'Kernel (n = {x:d})'] for x in xs])
    klc = np.array([scores_complementary[f'Kernel (n = {x:d})'] for x in xs])
    
    plt.plot(xs, klu[:,0], marker='o', label="Kernel-SHAP")
    plt.plot(xs, klc[:,0], marker='o', label="Kernel-SHAP (comp.)")

    # monte carlo methods:
    mcu = np.array([scores_uniform[f'Monte Carlo (n = {x:d})'] for x in xs])
    mcc = np.array([scores_comp_no_mc[f'Monte Carlo (n = {x:d})'] for x in xs])
    mcp = np.array([scores_complementary[f'Monte Carlo (n = {x:d})'] for x in xs])

    plt.plot(xs, mcu[:,0], marker='o', label="Monte Carlo")
    plt.plot(xs, mcc[:,0], marker='o', label="Monte Carlo (comp.)")
    plt.plot(xs, mcp[:,0], marker='o', label="Paired Monte Carlo (comp.)")
    
    plt.xticks(xs)

    plt.legend()
    plt.xlabel('$N$')
    plt.ylabel('AIPC')
    plt.tight_layout()
    plt.savefig(f'aipc_generator_{target}.pdf')
    plt.show()

#%%
import pandas as pd

model = 'llama8b'
for target in ('query', 'context'):
    # load scores:
    try:
        with open(os.path.join(RESULTS_PATH, f'scores_{model}_{target}_uniform.json'), 'r') as file:
            scores_uniform = json.load(file)

        with open(os.path.join(RESULTS_PATH, f'scores_{model}_{target}_complementary.json'), 'r') as file:
            scores_complementary = json.load(file)

        with open(os.path.join(RESULTS_PATH, f'scores_{model}_{target}_comp_no_mc.json'), 'r') as file:
            scores_comp_no_mc = json.load(file)
    except FileNotFoundError: continue

    def _tostr(val, min , max):
        return f'${val:.2f}~[{min:.2f}, {max:.2f}]$'


    # get x values:
    xs  = [10, 15, 20, 25, 30]

    # Precise and Random:
    prc = {f'n = {x:d}': _tostr(*scores_uniform['Precise']) for x in xs}
    rnd = {f'n = {x:d}': _tostr(*scores_uniform['Random']) for x in xs}

    # kernel methods:
    klu = {f'n = {x:d}': _tostr(*scores_uniform[f'Kernel (n = {x:d})']) for x in xs}
    klc = {f'n = {x:d}': _tostr(*scores_complementary[f'Kernel (n = {x:d})']) for x in xs}
    
    # monte carlo methods:
    mcu = {f'n = {x:d}': _tostr(*scores_uniform[f'Monte Carlo (n = {x:d})']) for x in xs}
    mcc = {f'n = {x:d}': _tostr(*scores_comp_no_mc[f'Monte Carlo (n = {x:d})']) for x in xs}
    mcp = {f'n = {x:d}': _tostr(*scores_complementary[f'Monte Carlo (n = {x:d})']) for x in xs}

    data = pd.DataFrame({
        'Precise': prc,
        'Random': rnd,
        'Kernel-SHAP (unif.)': klu,
        'Kernel-SHAP (compl.)': klc,
        'Monte Carlo (unif.)': mcu,
        'Monte Carlo (compl.)': mcc,
        'Paired Monte Carlo (compl.)': mcp
    }).T

    print(data.to_latex())