#!/usr/bin/env python3
"""Generate rebuttal training plots from wandb bilinear_rl project."""

import wandb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 200,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'bilinear': '#2196F3',  # blue
    'relu': '#FF5722',      # red-orange
}

OUT_DIR = Path("rebuttal_figures")
OUT_DIR.mkdir(exist_ok=True)

# ── Fetch & stitch runs ─────────────────────────────────────────────────
api = wandb.Api()
ALL_METRICS = ['eprewmean', 'eval_eprewmean', 'average_value_loss',
               'average_policy_loss', 'average_entropy', 'num_ppo_update']

runs = api.runs("nlp_and_interpretability/bilinear_rl")

# Group runs by name; sort by creation time so earlier run comes first
from collections import defaultdict
grouped = defaultdict(list)
for r in runs:
    grouped[r.name].append(r)

# Sort each group by created_at
for name in grouped:
    grouped[name].sort(key=lambda r: r.created_at)

def stitch_histories(run_list, keys):
    """Concatenate histories of runs with the same name, continuing the x-axis."""
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
    print(f"Stitching {name}: {len(rlist)} run(s)")
    stitched[name] = stitch_histories(rlist, ALL_METRICS)

# ── Helper: smoothed line ────────────────────────────────────────────────
def smooth(y, window=50):
    return pd.Series(y).rolling(window, min_periods=1, center=True).mean()

# ── Identify environments ────────────────────────────────────────────────
envs = set()
for name in stitched:
    # strip bilinear_ or relu_ prefix
    for prefix in ['bilinear_', 'relu_']:
        if name.startswith(prefix):
            envs.add(name[len(prefix):])
envs = sorted(envs)
print(f"Environments: {envs}")

# ── Figure 1 & 2: Training & Eval reward per environment ────────────────
for metric, label, fig_prefix in [
    ('eprewmean', 'Training Reward (mean episode return)', 'fig_train_reward'),
    ('eval_eprewmean', 'Eval Reward (mean episode return)', 'fig_eval_reward'),
]:
    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5), squeeze=False)
    axes = axes[0]

    for i, env in enumerate(envs):
        ax = axes[i]
        for model, color in COLORS.items():
            key = f"{model}_{env}"
            if key not in stitched or stitched[key].empty:
                continue
            df = stitched[key]
            if metric not in df.columns:
                continue
            vals = df[metric].dropna()
            steps = df.loc[vals.index, 'global_step']
            ax.plot(steps, smooth(vals), color=color, label=model.capitalize(), alpha=0.9)
            ax.fill_between(steps, smooth(vals, 100) - vals.rolling(100, min_periods=1).std().fillna(0),
                            smooth(vals, 100) + vals.rolling(100, min_periods=1).std().fillna(0),
                            color=color, alpha=0.1)

        ax.set_title(env.replace('_', ' ').title())
        ax.set_xlabel('PPO Updates')
        ax.set_ylabel(label if i == 0 else '')
        ax.legend()

    fig.suptitle(label, fontsize=15, y=1.02)
    fig.tight_layout()
    path = OUT_DIR / f"{fig_prefix}.png"
    fig.savefig(path, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close(fig)

# ── Figure 3: Value loss comparison ──────────────────────────────────────
fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5), squeeze=False)
axes = axes[0]
for i, env in enumerate(envs):
    ax = axes[i]
    for model, color in COLORS.items():
        key = f"{model}_{env}"
        if key not in stitched or stitched[key].empty:
            continue
        df = stitched[key]
        vals = df['average_value_loss'].dropna()
        steps = df.loc[vals.index, 'global_step']
        ax.plot(steps, smooth(vals), color=color, label=model.capitalize(), alpha=0.9)
    ax.set_title(env.replace('_', ' ').title())
    ax.set_xlabel('PPO Updates')
    ax.set_ylabel('Value Loss' if i == 0 else '')
    ax.legend()

fig.suptitle('Value Loss During Training', fontsize=15, y=1.02)
fig.tight_layout()
path = OUT_DIR / "fig_value_loss.png"
fig.savefig(path, bbox_inches='tight')
print(f"Saved {path}")
plt.close(fig)

# ── Figure 4: Policy loss + Entropy (2-row subplot) ─────────────────────
fig, axes = plt.subplots(2, len(envs), figsize=(6 * len(envs), 9), squeeze=False)
for row, (metric, ylabel) in enumerate([
    ('average_policy_loss', 'Policy Loss'),
    ('average_entropy', 'Entropy'),
]):
    for i, env in enumerate(envs):
        ax = axes[row][i]
        for model, color in COLORS.items():
            key = f"{model}_{env}"
            if key not in stitched or stitched[key].empty:
                continue
            df = stitched[key]
            vals = df[metric].dropna()
            steps = df.loc[vals.index, 'global_step']
            ax.plot(steps, smooth(vals), color=color, label=model.capitalize(), alpha=0.9)
        ax.set_title(env.replace('_', ' ').title() if row == 0 else '')
        ax.set_xlabel('PPO Updates' if row == 1 else '')
        ax.set_ylabel(ylabel if i == 0 else '')
        ax.legend()

fig.suptitle('Policy Loss & Entropy During Training', fontsize=15, y=1.01)
fig.tight_layout()
path = OUT_DIR / "fig_policy_entropy.png"
fig.savefig(path, bbox_inches='tight')
print(f"Saved {path}")
plt.close(fig)

# ── Figure 5: Dodgeball standalone (if exists) ──────────────────────────
dodge_keys = [k for k in stitched if 'dodgeball' in k]
if dodge_keys:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for dk in dodge_keys:
        df = stitched[dk]
        model = dk.split('_')[0]
        color = COLORS.get(model, '#333')

        for ax_idx, (metric, ylabel) in enumerate([
            ('eprewmean', 'Training Reward'),
            ('eval_eprewmean', 'Eval Reward'),
        ]):
            vals = df[metric].dropna()
            steps = df.loc[vals.index, 'global_step']
            axes[ax_idx].plot(steps, smooth(vals), color=color, label=f"{model.capitalize()}", alpha=0.9)
            axes[ax_idx].fill_between(steps,
                smooth(vals, 100) - vals.rolling(100, min_periods=1).std().fillna(0),
                smooth(vals, 100) + vals.rolling(100, min_periods=1).std().fillna(0),
                color=color, alpha=0.1)
            axes[ax_idx].set_xlabel('PPO Updates')
            axes[ax_idx].set_ylabel(ylabel)
            axes[ax_idx].set_title(ylabel)
            axes[ax_idx].legend()

    fig.suptitle('Dodgeball Easy — Bilinear', fontsize=15, y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig_dodgeball.png"
    fig.savefig(path, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close(fig)

print("\nDone! All figures saved to rebuttal_figures/")
