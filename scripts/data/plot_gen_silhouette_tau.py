"""
Mean silhouette score of the WARG tau-split clusters as a function of τ.

For each query and each τ, documents are split into two clusters using the
same rule as WARG:
    important   : normalised generator score > τ / n
    unimportant : normalised generator score ≤ τ / n

The silhouette score measures how well-separated these two groups are in the
1-D space of generator importance scores.  We plot the mean (± SE) silhouette
score over all queries, loading from explanations_popqa (original explanations,
not perturbations).
"""

import sys
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import silhouette_score

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

TAU_VALUES = np.linspace(0.0, 4.0, 21)

OUTPUT_DIR = Path('analysis')
OUTPUT_DIR.mkdir(exist_ok=True)


def _silhouette_for_tau(gen_scores: np.ndarray, tau: float) -> float:
    """Silhouette score for the tau-induced binary split on gen_scores.

    Replicates the WARG normalisation: divide by sum of positive values.
    Returns NaN when one cluster is empty (no valid split).
    """
    n = len(gen_scores)
    if n < 2:
        return np.nan

    pos_sum = float(np.sum([1e-9] + gen_scores[gen_scores > 0].tolist()))
    norm = gen_scores / pos_sum

    labels = (norm > tau / n).astype(int)

    n_imp = int(labels.sum())
    if n_imp == 0 or n_imp == n:
        return np.nan

    return float(silhouette_score(norm.reshape(-1, 1), labels))


# ── load data ─────────────────────────────────────────────────────────────────
# sil[ret][gen] = (n_queries, n_tau) array of silhouette scores
sil = {}

all_rets = list(RET_FOLDERS)
all_gens = list(GEN_FOLDERS)

for ret, gen in itertools.product(all_rets, all_gens):
    exp_dir = Path(f'results/{RET_FOLDERS[ret]}/{GEN_FOLDERS[gen]}/explanations_popqa')
    if not exp_dir.exists():
        print(f"WARN: {exp_dir} not found, skipping.")
        continue

    try:
        exps = RAGExplanation.load(str(exp_dir))
    except Exception as e:
        print(f"WARN: could not load {exp_dir}: {e}")
        continue

    rows = []
    for exp in exps.values():
        gen_imp = exp.generator_document_importance   # (n_docs,)
        rows.append([_silhouette_for_tau(gen_imp, t) for t in TAU_VALUES])

    sil.setdefault(ret, {})[gen] = np.array(rows)   # (n_queries, n_tau)

# ── plot ──────────────────────────────────────────────────────────────────────
COLORS     = {
    'llama8B':    '#1f77b4',
    'qwen7B':     '#ff7f0e',
    'gemma3_12B': '#2ca02c'
}
LINESTYLES = {
    'dragon':    '-',
    'snowflake': '--',
}

fig, ax = plt.subplots(figsize=(3, 3))

for ret, gen in itertools.product(all_rets, all_gens):
    if ret not in sil or gen not in sil[ret]:
        continue

    scores  = sil[ret][gen]
    mean    = np.nanmean(scores, axis=0)
    n_valid = (~np.isnan(scores)).sum(axis=0)
    se      = np.nanstd(scores, axis=0) / np.sqrt(np.maximum(n_valid, 1))

    label = f'{DISPLAY_NAMES[ret]} / {DISPLAY_NAMES[gen]}'
    color = COLORS[gen]
    ls    = LINESTYLES[ret]

    ax.plot(TAU_VALUES, mean, color=color, linestyle=ls, linewidth=2.0, label=label)
    ax.fill_between(TAU_VALUES, mean - se, mean + se, color=color, alpha=0.15)

ax.axhline(0, color='gray', linestyle='-', linewidth=0.6, alpha=0.5)
ax.set_xlabel('$\\tau$', fontsize=11)
ax.set_ylabel('Mean silhouette score', fontsize=10)
ax.legend(fontsize=8, frameon=True)
ax.grid(True, linestyle='--', alpha=0.35)
plt.tight_layout()

for ext in ('pdf', 'png'):
    p = OUTPUT_DIR / f'popqa_gen_silhouette_vs_tau.{ext}'
    fig.savefig(p, bbox_inches='tight', dpi=150)
    print(f"Saved {p}")
plt.close()
