"""
Mean |intGrad| attribution by context position (0-9), one line per condition.
Faceted: retriever (rows) × model (columns).
Attribution is normalized per-query (sum=1) before averaging,
so curves are comparable across models and conditions.
"""

import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

EXPERIMENT_DIRS = {
    'DRAGON+':   'results/facebook-dragon-plus',
    'Snowflake': 'results/snowflake-arctic-embed-l-v2.0',
}
MODEL_LABELS = {
    'llama-3.1-8b-instruct': 'Llama',
    'qwen2.5-7b-instruct':   'Qwen',
    'gemma-3-12b-it':        'Gemma',
}
CONDITION_ORDER  = ['B1', 'B2', 'B3']
CONDITION_COLORS = {'B1': '#2196F3', 'B2': '#4CAF50', 'B3': '#FF9800'}
CONDITION_LABELS = {
    'B1': 'B1 – top-k first, random last',
    'B2': 'B2 – random first, top-k last',
    'B3': 'B3 – shuffled',
}
OUTPUT_DIR = Path('analysis')
OUTPUT_DIR.mkdir(exist_ok=True)

# attr_data[retriever][model][condition] = list of 1-D arrays (one per query, length=K)
attr_data = {}

for retriever, base_dir in EXPERIMENT_DIRS.items():
    base = Path(base_dir)
    if not base.exists():
        print(f"WARN: {base_dir} not found, skipping.")
        continue
    attr_data[retriever] = {}

    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        model_raw = model_dir.name
        model = MODEL_LABELS.get(model_raw, model_raw)

        pkl_dir = model_dir / 'perturbations_popqa' / 'retrieval'
        if not pkl_dir.exists():
            continue

        attr_data[retriever][model] = {c: [] for c in CONDITION_ORDER}

        for cond in CONDITION_ORDER:
            for pkl_path in sorted(pkl_dir.glob(f'{cond.lower()}_*.pkl')):
                with open(pkl_path, 'rb') as fh:
                    data = pickle.load(fh)
                if 'intGrad' not in data or 'context' not in data['intGrad']:
                    continue
                sv = data['intGrad']['context']
                if hasattr(sv, 'numpy'):
                    sv = sv.numpy()
                attr = np.sum(np.abs(sv), axis=1) if sv.ndim == 2 else np.abs(sv)
                total = attr.sum()
                if total > 0:
                    attr = attr / total
                attr_data[retriever][model][cond].append(attr)

# ── plot ─────────────────────────────────────────────────────────────────────

retrievers  = [r for r in EXPERIMENT_DIRS if r in attr_data and attr_data[r]]
model_order = ['Llama', 'Qwen', 'Gemma']
models      = [m for m in model_order if any(m in attr_data[r] for r in retrievers)]

n_rows, n_cols = len(retrievers), len(models)
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(4.5 * n_cols, 3.8 * n_rows),
                         sharey=False, sharex=True)
axes = np.array(axes).reshape(n_rows, n_cols)

K = 10  # default; overridden per-cell
for ri, retriever in enumerate(retrievers):
    for ci, model in enumerate(models):
        ax = axes[ri, ci]
        cond_data = attr_data.get(retriever, {}).get(model, {})

        for cond in CONDITION_ORDER:
            arrays = cond_data.get(cond, [])
            if not arrays:
                continue
            K = arrays[0].shape[0]
            mat = np.vstack([a[:K] for a in arrays])   # (n_queries, K)
            mean_attr = mat.mean(axis=0)
            se_attr   = mat.std(axis=0) / np.sqrt(len(arrays))
            pos = np.arange(K)

            ax.plot(pos, mean_attr,
                    marker='o', markersize=4,
                    label=CONDITION_LABELS[cond],
                    color=CONDITION_COLORS[cond],
                    linewidth=1.8)
            ax.fill_between(pos,
                            mean_attr - se_attr,
                            mean_attr + se_attr,
                            color=CONDITION_COLORS[cond], alpha=0.15)

        # mark the midpoint between the two halves
        ax.axvline(x=(K / 2) - 0.5, color='gray', linestyle=':', linewidth=1, alpha=0.6)

        ax.set_title(model, fontsize=10, fontweight='bold')
        ax.set_xlabel('Document position in context', fontsize=8)
        ax.set_ylabel(f'Mean norm. attribution\n({retriever})', fontsize=8)
        ax.set_xticks(range(K))
        ax.grid(True, linestyle='--', alpha=0.35)

        if ri == 0 and ci == n_cols - 1:
            ax.legend(fontsize=7, title='Condition', title_fontsize=8,
                      loc='upper right')

fig.suptitle(
    'Mean intGrad attribution by document position\n(PopQA, normalised per query)',
    fontsize=12, fontweight='bold', y=1.01,
)
plt.tight_layout()

for ext in ('pdf', 'png'):
    p = OUTPUT_DIR / f'popqa_attribution_by_position.{ext}'
    fig.savefig(p, bbox_inches='tight', dpi=150)
    print(f"Saved {p}")
plt.close()
