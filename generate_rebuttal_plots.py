#!/usr/bin/env python3
"""Generate a single Figure-4-style rebuttal plot from wandb bilinear_rl project.

Layout matches Figure 4 from arxiv:2412.00944:
  - Rows = environments (Maze Hard, Coinrun Hard)
  - Columns = metrics (Expected Return, Average Entropy, Unexplained Variance)
  - Each subplot: Bilinear (dark blue) vs ReLU baseline (lighter orange)
"""

import wandb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── Style (compact, paper-quality) ───────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 300,
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.titleweight': 'bold',
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'legend.framealpha': 0.8,
    'lines.linewidth': 1.5,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'bilinear': '#1565C0',   # dark blue
    'relu':     '#E65100',   # dark orange
}
LABELS = {
    'bilinear': 'Bilinear IMPALA',
    'relu':     'ReLU IMPALA',
}

OUT_DIR = Path("procegen hard training")
OUT_DIR.mkdir(exist_ok=True)

# ── Fetch & stitch runs ─────────────────────────────────────────────────
api = wandb.Api()
ALL_METRICS = ['eprewmean', 'eval_eprewmean', 'average_value_loss',
               'average_policy_loss', 'average_entropy',
               'explained_variance', 'num_ppo_update']

runs = api.runs("nlp_and_interpretability/bilinear_rl")

grouped = defaultdict(list)
for r in runs:
    grouped[r.name].append(r)
for name in grouped:
    grouped[name].sort(key=lambda r: r.created_at)


def stitch_histories(run_list, keys):
    frames = []
    offset = 0
    for r in run_list:
        h = r.history(samples=10000, keys=keys)
        if len(h) == 0:
            continue
        h = h.sort_values('num_ppo_update').reset_index(drop=True)
        h['global_step'] = h['num_ppo_update'] + offset
        offset = h['global_step'].max()
        frames.append(h)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


stitched = {}
for name, rlist in grouped.items():
    # Only maze_hard and coinrun_hard (skip dodgeball)
    if 'dodgeball' in name:
        continue
    print(f"Stitching {name}: {len(rlist)} run(s)")
    stitched[name] = stitch_histories(rlist, ALL_METRICS)


def smooth(y, window=50):
    return pd.Series(y).rolling(window, min_periods=1, center=True).mean()


# ── Grid: rows=environments, cols=metrics ────────────────────────────────
ENVS = ['maze_hard', 'coinrun_hard']
ENV_LABELS = {'maze_hard': 'Maze (Hard)', 'coinrun_hard': 'Coinrun (Hard)'}

# Columns: (metric_key, column_title, y_label)
METRICS = [
    ('eprewmean',          'Expected Return',       'Mean Episode Return'),
    ('average_entropy',    'Average Entropy',        'Entropy'),
    ('explained_variance', 'Unexplained Variance',   'Unexplained Var.'),
]

n_rows = len(ENVS)
n_cols = len(METRICS)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.8 * n_rows),
                          squeeze=False)

for row, env in enumerate(ENVS):
    for col, (metric, col_title, ylabel) in enumerate(METRICS):
        ax = axes[row][col]

        for model in ['relu', 'bilinear']:
            key = f"{model}_{env}"
            if key not in stitched or stitched[key].empty:
                continue
            df = stitched[key]

            if metric == 'explained_variance':
                # Unexplained variance = 1 - explained_variance
                raw = df['explained_variance'].dropna()
                vals = 1.0 - raw
                # Drop outliers (likely system glitches) outside 1st-99th percentile
                lo, hi = vals.quantile(0.01), vals.quantile(0.99)
                mask = vals.between(lo, hi)
                vals = vals[mask]
            else:
                vals = df[metric].dropna()

            steps = df.loc[vals.index, 'global_step']
            color = COLORS[model]
            label = LABELS[model]

            smoothed = smooth(vals)
            ax.plot(steps, smoothed, color=color, label=label, alpha=0.9)
            # Shaded std band
            std = vals.rolling(100, min_periods=1, center=True).std().fillna(0)
            ax.fill_between(steps, smoothed - std, smoothed + std,
                            color=color, alpha=0.1)

        # Column titles on top row only
        if row == 0:
            ax.set_title(col_title)

        # Row labels on left column only
        if col == 0:
            ax.set_ylabel(f"{ENV_LABELS[env]}\n{ylabel}")
        else:
            ax.set_ylabel(ylabel)

        # X label on bottom row only
        if row == n_rows - 1:
            ax.set_xlabel('PPO Updates')
        else:
            ax.set_xticklabels([])

        # Legend only in first subplot
        if row == 0 and col == 0:
            ax.legend(loc='lower right')

fig.suptitle(
    'Performance comparison: ReLU and Bilinear architectures\n'
    'across hard ProcGen environments',
    fontsize=11, fontweight='bold', y=1.02
)
fig.tight_layout()

path = OUT_DIR / "fig4_procgen_hard_comparison.png"
fig.savefig(path, bbox_inches='tight')
print(f"Saved {path}")
plt.close(fig)

print("\nDone!")
