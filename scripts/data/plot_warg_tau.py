"""
WARG(B1) − WARG(B2) as a function of τ ∈ [1, 2], with Pearson and Spearman
B1−B2 as τ-independent baselines.

Retriever scores for WARG in the perturbation setting use the position-based
ground truth ret_scores = [9,8,…,0] (doc at context position 0 is "most
retrieved"), consistent with eval.py.  Pearson / Spearman use the same scores.
"""

import sys
import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.Interpretable_RAG.rag import RAGExplanation

# ── config ────────────────────────────────────────────────────────────────────
RET_FOLDERS = {
    'dragon':    'facebook-dragon-plus',
    'snowflake': 'snowflake-arctic-embed-l-v2.0',
}
GEN_FOLDERS = {
    'llama8B':    'llama-3.1-8b-instruct',
    'qwen7B':     'qwen2.5-7b-instruct',
    'gemma3_12B': 'gemma-3-12b-it',
}
DISPLAY_NAMES = {
    'dragon':     'DRAGON',
    'snowflake':  'Arctic Embed 2',
    'llama8B':    'Llama',
    'qwen7B':     'Qwen',
    'gemma3_12B': 'Gemma',
}

TAU_VALUES  = np.linspace(0.5, 4.0, 21)
CONDS       = ['B1', 'B2']
RET_SCORES  = np.arange(10, dtype=float)[::-1]   # [9,8,…,0]: pos 0 = top ranked

OUTPUT_DIR  = Path('analysis')
OUTPUT_DIR.mkdir(exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
# scores[ret][gen][cond] = {
#   'warg_per_tau': (n_queries, n_tau) array,
#   'spearman':     (n_queries,) array,
#   'pearson':      (n_queries,) array,
# }
scores = {}

all_rets = list(RET_FOLDERS)
all_gens = list(GEN_FOLDERS)

for ret, gen in itertools.product(all_rets, all_gens):
    res_dir    = Path(f'results/{RET_FOLDERS[ret]}/{GEN_FOLDERS[gen]}')
    perturb_dir = res_dir / 'perturbations_popqa'
    if not perturb_dir.exists():
        print(f"WARN: {perturb_dir} not found, skipping.")
        continue

    try:
        exps = RAGExplanation.load(str(perturb_dir))
    except Exception as e:
        print(f"WARN: could not load {perturb_dir}: {e}")
        continue

    cond_data = {c: {'warg_per_tau': [], 'spearman': [], 'pearson': []} for c in CONDS}

    for key, exp in exps.items():
        cond = key.split('_')[0].upper()
        if cond not in CONDS:
            continue

        gen_imp = exp.generator_document_importance   # (n_docs,)

        # WARG for every τ in one pass
        warg_row = [exp.warg(tau=t, _ret_scores=RET_SCORES) for t in TAU_VALUES]
        cond_data[cond]['warg_per_tau'].append(warg_row)
        cond_data[cond]['spearman'].append(
            stats.spearmanr(RET_SCORES, gen_imp).statistic)
        cond_data[cond]['pearson'].append(
            stats.pearsonr(RET_SCORES, gen_imp).statistic)

    # convert to arrays
    for c in CONDS:
        for k in ('warg_per_tau', 'spearman', 'pearson'):
            cond_data[c][k] = np.array(cond_data[c][k])

    scores.setdefault(ret, {})[gen] = cond_data

# ── plot ──────────────────────────────────────────────────────────────────────
n_rows = len(all_rets)
n_cols = len(all_gens)

fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(1.75 * n_cols, 2 * n_rows),
                          sharey=True, sharex=True)
axes = np.array(axes).reshape(n_rows, n_cols)

WARG_COLOR     = '#2ca02c'
SPEARMAN_COLOR = '#1f77b4'
PEARSON_COLOR  = '#ff7f0e'

for ri, ret in enumerate(all_rets):
    for ci, gen in enumerate(all_gens):
        ax = axes[ri, ci]

        if ret not in scores or gen not in scores[ret]:
            ax.set_title(f'{DISPLAY_NAMES[ret]} / {DISPLAY_NAMES[gen]}\n(missing)',
                         fontsize=9)
            ax.axis('off')
            continue

        cd = scores[ret][gen]

        # WARG B1 − B2 as a function of τ
        b1_warg = cd['B1']['warg_per_tau']   # (n_q, n_tau)
        b2_warg = cd['B2']['warg_per_tau']

        n_b1, n_b2 = len(b1_warg), len(b2_warg)
        diff_mean = b1_warg.mean(axis=0) - b2_warg.mean(axis=0)
        # SE of the difference (independent samples)
        diff_se = np.sqrt(
            b1_warg.std(axis=0) ** 2 / n_b1 +
            b2_warg.std(axis=0) ** 2 / n_b2
        )

        ax.semilogx(TAU_VALUES, diff_mean,
                color=WARG_COLOR, linewidth=2.0, label='WARG')
        ax.fill_between(TAU_VALUES,
                        diff_mean - diff_se,
                        diff_mean + diff_se,
                        color=WARG_COLOR, alpha=0.20)

        # Spearman baseline (τ-independent)
        sp_b1 = np.nanmean(cd['B1']['spearman'])
        sp_b2 = np.nanmean(cd['B2']['spearman'])
        sp_diff = sp_b1 - sp_b2
        ax.axhline(sp_diff, color=SPEARMAN_COLOR, linestyle='--', linewidth=1.4,
                   label='Spearman $r$')

        # Pearson baseline (τ-independent)
        pe_b1 = np.nanmean(cd['B1']['pearson'])
        pe_b2 = np.nanmean(cd['B2']['pearson'])
        pe_diff = pe_b1 - pe_b2
        ax.axhline(pe_diff, color=PEARSON_COLOR, linestyle=':', linewidth=1.4,
                   label='Pearson $r$')

        ax.axhline(0, color='gray', linestyle='-', linewidth=0.6, alpha=0.5)

        ax.set_title(f'{DISPLAY_NAMES[ret]} \n {DISPLAY_NAMES[gen]}',
                     fontsize=8, fontweight='bold')
        ax.set_ylim(0.4,1.2)
        if ri == 1: ax.set_xlabel('$\\tau$', fontsize=9)
        if ci == 0: ax.set_ylabel('$\\overline\u007bc2\u007d − \\overline\u007bc3\u007d$', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.set_xticks([0.5, 1., 2., 4.])
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(['0.5', '1', '2', '4']))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

# single figure-level legend, collected from the first populated axes
_handles, _labels = next(
    ax.get_legend_handles_labels()
    for ax in axes.flat
    if ax.get_legend_handles_labels()[0]
)
fig.legend(_handles, _labels, loc='upper center', ncol=3,
           fontsize=8, frameon=True, bbox_to_anchor=(0.5, 1.03))
fig.suptitle('')
plt.tight_layout()

for ext in ('pdf', 'png'):
    p = OUTPUT_DIR / f'popqa_warg_b1_b2_vs_tau.{ext}'
    fig.savefig(p, bbox_inches='tight', dpi=150)
    print(f"Saved {p}")
plt.close()
