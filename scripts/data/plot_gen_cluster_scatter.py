"""
Scatterplot of the tau-induced clusters in retriever-rank × generator-importance space.

Each point is one (query, document) pair.  Documents are split into two clusters
using the WARG rule:
    important   : normalised generator score > τ / n
    unimportant : normalised generator score ≤ τ / n

X-axis : retriever document importance (intGrad-based, abs-normalised)
Y-axis : normalised generator importance score
Colour : cluster membership (important / unimportant)
Marker : retriever model (circle = DRAGON, triangle = Arctic Embed 2)

One subplot per generator model.
"""

import sys
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
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

TAU = 1.5   # WARG default threshold

OUTPUT_DIR = Path('analysis')
OUTPUT_DIR.mkdir(exist_ok=True)

CLUSTER_COLORS = {1: '#d62728', 0: '#1f77b4'}      # important=red, unimportant=blue
CLUSTER_LABELS = {1: 'important', 0: 'unimportant'}
MARKERS        = {'dragon': 'o', 'snowflake': '^'}

# ── load data ─────────────────────────────────────────────────────────────────
# points[gen] = list of (ret_score, norm_gen_score, cluster_label, ret_key)
points = {gen: [] for gen in GEN_FOLDERS}

all_rets = list(RET_FOLDERS)
all_gens = list(GEN_FOLDERS)

for ret, gen in itertools.product(all_rets, all_gens):
    exp_dir = Path(f'results/{RET_FOLDERS[ret]}/{GEN_FOLDERS[gen]}/explanations_popqa')
    if not exp_dir.exists():
        print(f"WARN: {exp_dir} not found, skipping.")
        continue

    try:
        loaded = RAGExplanation.load(str(exp_dir))
    except Exception as e:
        print(f"WARN: could not load {exp_dir}: {e}")
        continue

    exps = loaded if isinstance(loaded, dict) else {'_': loaded}
    for exp in exps.values():
        gen_imp = exp.generator_document_importance   # (n_docs,)
        ret_imp = exp.retriever_document_importance   # (n_docs,) intGrad abs-norm
        n = len(gen_imp)

        pos_sum = float(np.sum([1e-9] + gen_imp[gen_imp > 0].tolist()))
        norm    = gen_imp / pos_sum

        labels = (norm > TAU / n).astype(int)

        for doc_idx in range(n):
            points[gen].append((ret_imp[doc_idx], norm[doc_idx], labels[doc_idx], ret))

# ── plot ──────────────────────────────────────────────────────────────────────
n_gens = len(all_gens)
fig = plt.figure(figsize=(1.5*n_gens, 2.), constrained_layout=True)
width_ratios = [4, 1] * n_gens
gs = fig.add_gridspec(1, 2 * n_gens, width_ratios=width_ratios, wspace=0.02)

# Build axes: each generator gets a scatter + a rotated-PDF column
ax_scatter_0 = None
axes_pairs = []  # list of (ax_scatter, ax_dist)
for ci in range(n_gens):
    ax_sc = fig.add_subplot(gs[0, 2 * ci], sharey=ax_scatter_0)
    if ax_scatter_0 is None:
        ax_scatter_0 = ax_sc
    ax_dt = fig.add_subplot(gs[0, 2 * ci + 1], sharey=ax_sc)
    axes_pairs.append((ax_sc, ax_dt))

print(f"\n{'─'*70}")
print(f"One-sided t-test: H0: μ_important = 0.05, H1: μ_important > 0.05")
print(f"Fixed null normal: μ = 0.05, σ = 0.1")
print(f"{'─'*70}")

for ci, gen in enumerate(all_gens):
    ax, ax_d = axes_pairs[ci]
    data = points[gen]
    if not data:
        ax.set_title(f'{DISPLAY_NAMES[gen]}\n(missing)', fontsize=9)
        ax.axis('off')
        ax_d.axis('off')
        continue

    arr = np.array([(x, y, lbl) for x, y, lbl, _ in data])
    ret_keys = [r for _, _, _, r in data]

    for cluster in (0, 1):
        for ret in all_rets:
            mask = (arr[:, 2] == cluster) & np.array([r == ret for r in ret_keys])
            if not mask.any():
                continue
            ax.scatter(
                arr[mask, 0] + np.random.default_rng(42).uniform(-0.15, 0.15, mask.sum()),
                arr[mask, 1],
                c=CLUSTER_COLORS[cluster],
                marker=MARKERS[ret],
                s=6, alpha=0.35, linewidths=0,
            )

    # ── fixed null normal: μ=0.05, σ=0.1 ─────────────────────────────────────
    y_unimp  = arr[arr[:, 2] == 0, 1]
    y_imp    = arr[arr[:, 2] == 1, 1]
    mu_unimp = 0.05
    sigma    = 0.1

    # ── rotated PDF + histogram panel ─────────────────────────────────────────
    y_grid = np.linspace(-1, 1, 400)
    pdf = stats.norm.pdf(y_grid, loc=mu_unimp, scale=sigma)

    bins = np.linspace(-1, 1, 40)
    ax_d.hist(y_unimp, bins=bins, orientation='horizontal', density=True,
              alpha=0.45, color=CLUSTER_COLORS[0])
    ax_d.hist(y_imp,   bins=bins, orientation='horizontal', density=True,
              alpha=0.45, color=CLUSTER_COLORS[1])
    ax_d.plot(pdf, y_grid, color='black', lw=1.5)
    ax_d.axhline(mu_unimp, color='black', lw=0.8, linestyle='--', alpha=0.6)

    ax_d.set_xlim(left=0)
    ax_d.set_xticks([])
    plt.setp(ax_d.get_yticklabels(), visible=False)
    ax_d.tick_params(left=False)
    ax_d.grid(False)

    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    ax_d.spines['bottom'].set_visible(False)

    ax.set_title(DISPLAY_NAMES[gen], fontsize=8, fontweight='bold')
    ax.set_xlabel('Retriever', fontsize=9)
    ax.set_yticks([-1, 0, 1])
    if ci == 0:
        ax.set_ylabel('Generator', fontsize=9)
        ax.tick_params(labelleft=True)
    else:
        ax.tick_params(labelleft=False)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(-.1, .3)
    ax.set_ylim(-1, 1)

print(f"{'─'*70}\n")
    

# ── shared legend ─────────────────────────────────────────────────────────────
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=CLUSTER_COLORS[1],
           markersize=7, label='Important'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=CLUSTER_COLORS[0],
           markersize=7, label='Unimportant'),
    Line2D([0], [0], marker='o', color='gray', markersize=7,
           linestyle='none', label=DISPLAY_NAMES['dragon']),
    Line2D([0], [0], marker='^', color='gray', markersize=7,
           linestyle='none', label=DISPLAY_NAMES['snowflake']),
]
fig.suptitle('', fontsize=10)
fig.legend(handles=legend_elements, loc='upper center', ncol=4,
           fontsize=8, frameon=True, bbox_to_anchor=(0.5, 1.1))

for ext in ('pdf', 'png'):
    p = OUTPUT_DIR / f'popqa_gen_cluster_scatter_tau{TAU}.{ext}'
    fig.savefig(p, bbox_inches='tight', dpi=150)
    print(f"Saved {p}")
plt.close()
