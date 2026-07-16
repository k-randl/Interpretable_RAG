"""
Wasted Retrieval (WR) and Noise Distraction (ND) vs context-position threshold τ.

For threshold τ, the split is purely by context position:
  first_τ  = documents at context positions 0 .. τ-1  (early in context)
  last_Kτ  = documents at context positions τ .. K-1  (late in context)

  WR(τ) = |first_τ  ∩ NOT gen_top_{τ}|     / τ
           fraction of early-placed docs the generator ignores in its top-τ
  ND(τ) = |last_Kτ  ∩ gen_top_{K-τ}|       / (K-τ)
           fraction of late-placed docs the generator uses in its top-(K-τ)
"""

import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
TAU_RANGE  = list(range(0, 11))
OUTPUT_DIR = Path('analysis')
OUTPUT_DIR.mkdir(exist_ok=True)


def doc_attributions(data):
    sv = data['intGrad']['context']
    if hasattr(sv, 'numpy'):
        sv = sv.numpy()
    return np.sum(np.abs(sv), axis=1) if sv.ndim == 2 else np.abs(sv)


def compute_wr_nd(doc_attrs, tau):
    K = len(doc_attrs)
    n_good  = tau
    n_noise = K - tau

    gen_rank_of = {int(d): r for r, d in enumerate(np.argsort(-doc_attrs))}

    wr = sum(1 for i in range(n_good)    if gen_rank_of[i] >= n_good)   / n_good   if n_good  > 0 else np.nan
    nd = sum(1 for i in range(n_good, K) if gen_rank_of[i] < n_noise)   / n_noise  if n_noise > 0 else np.nan
    return wr, nd


rows = []

for retriever, base_dir in EXPERIMENT_DIRS.items():
    base = Path(base_dir)
    if not base.exists():
        print(f"WARN: {base_dir} not found, skipping.")
        continue

    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        model_label = MODEL_LABELS.get(model_dir.name, model_dir.name)

        pkl_dir = model_dir / 'perturbations_popqa' / 'retrieval'
        if not pkl_dir.exists():
            continue

        for cond in CONDITION_ORDER:
            for pkl_path in sorted(pkl_dir.glob(f'{cond.lower()}_*.pkl')):
                with open(pkl_path, 'rb') as fh:
                    data = pickle.load(fh)

                if 'intGrad' not in data or 'context' not in data['intGrad']:
                    continue

                d_attr = doc_attributions(data)

                for tau in TAU_RANGE:
                    wr, nd = compute_wr_nd(d_attr, tau)
                    rows.append({
                        'retriever': retriever,
                        'model':     model_label,
                        'condition': cond,
                        'tau':       tau,
                        'WR':        wr,
                        'ND':        nd,
                    })

df = pd.DataFrame(rows)
summary = (df.groupby(['retriever', 'model', 'condition', 'tau'])[['WR', 'ND']]
             .mean()
             .reset_index())

csv_path = OUTPUT_DIR / 'popqa_wr_nd_by_tau.csv'
summary.to_csv(csv_path, index=False)
print(f"Summary saved to {csv_path}")

retrievers  = [r for r in EXPERIMENT_DIRS if r in df['retriever'].values]
model_order = ['Llama', 'Qwen', 'Gemma']
models      = [m for m in model_order if m in df['model'].values]

n_rows = len(retrievers)
n_cols = len(models)

METRIC_INFO = [
    ('WR', 'Wasted Retrieval',  'wr'),
    ('ND', 'Noise Distraction', 'nd'),
]

for metric, metric_label, slug in METRIC_INFO:
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.8 * n_rows),
                             sharey=True, sharex=True)
    axes = np.array(axes).reshape(n_rows, n_cols)

    for ri, retriever in enumerate(retrievers):
        for ci, model in enumerate(models):
            ax = axes[ri, ci]
            sub = summary[(summary['retriever'] == retriever) & (summary['model'] == model)]

            for cond in CONDITION_ORDER:
                csub = sub[sub['condition'] == cond].sort_values('tau')
                vals = csub[metric].values
                taus = csub['tau'].values
                mask = ~np.isnan(vals)
                if mask.sum() == 0:
                    continue
                ax.plot(taus[mask], vals[mask], marker='o', markersize=4,
                        label=cond, color=CONDITION_COLORS[cond], linewidth=1.8)

            ax.set_title(model, fontsize=10, fontweight='bold')
            ax.set_xlabel('Rank Threshold τ', fontsize=8)
            ax.set_ylabel(f'{metric_label}\n({retriever})', fontsize=8)
            ax.set_xlim(-0.3, 10.3)
            ax.set_ylim(-0.05, 1.05)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.grid(True, linestyle='--', alpha=0.4)
            if ri == 0 and ci == n_cols - 1:
                ax.legend(title='Condition', fontsize=7, title_fontsize=8)

    fig.suptitle(
        f'{metric_label} vs Rank Threshold τ\n(PopQA, all models & retrievers)',
        fontsize=12, fontweight='bold', y=1.01,
    )
    plt.tight_layout()

    for ext in ('pdf', 'png'):
        p = OUTPUT_DIR / f'popqa_{slug}_vs_threshold.{ext}'
        fig.savefig(p, bbox_inches='tight', dpi=150)
        print(f"Saved {p}")
    plt.close()
