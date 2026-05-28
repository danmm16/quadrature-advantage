"""
plot_benchmark.py

Reads benchmark_results.csv produced by ppo_benchmark.py and generates
four figures saved to plots/:

  1. learning_curves.png    smoothed reward over training steps, per environment
  2. advantage_variance.png advantage estimate variance over training, per environment
  3. final_reward.png       bar chart of mean final reward per (env, estimator)
  4. noisy_vs_clean.png     side-by-side adv_var comparison: clean vs noisy CartPole

Usage:
    python plot_benchmark.py
    python plot_benchmark.py --csv my_results.csv
    python plot_benchmark.py --no-show   # save only, do not display
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    import pandas as pd
except ImportError:
    print("pandas is required: pip install pandas")
    sys.exit(1)


# ── Style ─────────────────────────────────────────────────────────────────────

ESTIMATOR_COLORS = {
    'gae':           '#4C72B0',   # blue
    'quadrature_h7': '#DD8452',   # orange
    'sobolev_h4':    '#55A868',   # green
    'sobolev_h7':    '#C44E52',   # red
}

ESTIMATOR_LABELS = {
    'gae':           'GAE  (O(h¹))',
    'quadrature_h7': 'Quadrature  (O(h⁷))',
    'sobolev_h4':    'Sobolev h4  (O(h⁴))',
    'sobolev_h7':    'Sobolev h7  (O(h⁷))',
}

SMOOTH_WINDOW = 5   # rolling mean window over updates for display


def smooth(series: np.ndarray, window: int) -> np.ndarray:
    """Apply a causal rolling mean with min_periods=1."""
    out = np.full_like(series, np.nan, dtype=np.float64)
    for i in range(len(series)):
        start = max(0, i - window + 1)
        out[i] = np.nanmean(series[start:i + 1])
    return out


def estimator_style(name: str) -> dict:
    return {
        'color':     ESTIMATOR_COLORS.get(name, '#666666'),
        'label':     ESTIMATOR_LABELS.get(name, name),
        'linewidth': 1.8,
    }


def setup_ax(ax, xlabel='', ylabel='', title=''):
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x/1000:.0f}k' if x >= 1000 else str(int(x))))


def plot_with_ci(ax, df_group, x_col, y_col, estimator, smooth_w=SMOOTH_WINDOW):
    """
    Plot mean ± 1 std across seeds for a given estimator on an axis.
    Aligns seeds to a common x grid via interpolation.
    """
    seeds = df_group['seed'].unique()
    # Build common x grid from the estimator with most updates
    x_ref = df_group[df_group['estimator'] == estimator][x_col].values
    if len(x_ref) == 0:
        return

    ys = []
    for seed in seeds:
        sub = df_group[(df_group['estimator'] == estimator) &
                       (df_group['seed'] == seed)].sort_values(x_col)
        if len(sub) == 0:
            continue
        y = smooth(sub[y_col].values, smooth_w)
        y_interp = np.interp(x_ref, sub[x_col].values, y)
        ys.append(y_interp)

    if not ys:
        return

    ys    = np.array(ys)
    mean  = ys.mean(axis=0)
    std   = ys.std(axis=0)
    style = estimator_style(estimator)

    ax.plot(x_ref, mean, **style)
    ax.fill_between(x_ref, mean - std, mean + std,
                    color=style['color'], alpha=0.15)


# ── Figure 1: Learning curves ─────────────────────────────────────────────────

def fig_learning_curves(df: pd.DataFrame, out_dir: str):
    envs       = df['env'].unique()
    estimators = df['estimator'].unique()
    n          = len(envs)
    ncols      = min(n, 2)
    nrows      = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 4.5 * nrows),
                             squeeze=False)
    fig.suptitle('Learning Curves — Smoothed Reward', fontsize=13,
                 fontweight='bold', y=1.01)

    for i, env in enumerate(envs):
        ax  = axes[i // ncols][i % ncols]
        sub = df[df['env'] == env]
        for est in estimators:
            plot_with_ci(ax, sub, 'global_step', 'smoothed_reward', est)
        setup_ax(ax, xlabel='Environment steps', ylabel='Smoothed reward', title=env)
        ax.legend(fontsize=8, framealpha=0.7)

    # Hide unused axes
    for j in range(i + 1, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, 'learning_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    return fig


# ── Figure 2: Advantage variance ──────────────────────────────────────────────

def fig_advantage_variance(df: pd.DataFrame, out_dir: str):
    envs       = df['env'].unique()
    estimators = df['estimator'].unique()
    n          = len(envs)
    ncols      = min(n, 2)
    nrows      = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 4.5 * nrows),
                             squeeze=False)
    fig.suptitle('Advantage Estimate Variance', fontsize=13,
                 fontweight='bold', y=1.01)

    for i, env in enumerate(envs):
        ax  = axes[i // ncols][i % ncols]
        sub = df[df['env'] == env]
        for est in estimators:
            plot_with_ci(ax, sub, 'global_step', 'adv_var', est,
                         smooth_w=SMOOTH_WINDOW * 2)
        setup_ax(ax, xlabel='Environment steps',
                 ylabel='Advantage variance', title=env)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.set_yscale('log')

    for j in range(i + 1, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, 'advantage_variance.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    return fig


# ── Figure 3: Final reward bar chart ──────────────────────────────────────────

def fig_final_reward(df: pd.DataFrame, out_dir: str):
    """
    For each environment, compute mean final reward per estimator across seeds,
    then plot as grouped bars with error bars (±1 std).
    'Final reward' is the mean smoothed reward in the last 10% of training.
    """
    envs       = df['env'].unique()
    estimators = df['estimator'].unique()

    # Compute final reward: mean smoothed_reward in last 10% of steps per run
    records = []
    for env in envs:
        for est in estimators:
            for seed in df['seed'].unique():
                sub = df[(df['env'] == env) &
                         (df['estimator'] == est) &
                         (df['seed'] == seed)].sort_values('global_step')
                if len(sub) == 0:
                    continue
                cutoff = sub['global_step'].max() * 0.9
                tail   = sub[sub['global_step'] >= cutoff]['smoothed_reward']
                if len(tail) == 0:
                    tail = sub['smoothed_reward'].tail(5)
                records.append({
                    'env': env, 'estimator': est, 'seed': seed,
                    'final_reward': tail.mean()
                })

    summary = pd.DataFrame(records)
    grouped = summary.groupby(['env', 'estimator'])['final_reward'].agg(
        ['mean', 'std']).reset_index()

    n_est  = len(estimators)
    width  = 0.8 / n_est
    x      = np.arange(len(envs))
    colors = [ESTIMATOR_COLORS.get(e, '#666') for e in estimators]

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(envs)), 5))

    for j, est in enumerate(estimators):
        sub    = grouped[grouped['estimator'] == est]
        means  = [sub[sub['env'] == e]['mean'].values[0]
                  if len(sub[sub['env'] == e]) > 0 else 0.0
                  for e in envs]
        stds   = [sub[sub['env'] == e]['std'].values[0]
                  if len(sub[sub['env'] == e]) > 0 else 0.0
                  for e in envs]
        offset = (j - n_est / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4,
               label=ESTIMATOR_LABELS.get(est, est),
               color=colors[j], alpha=0.85, error_kw={'linewidth': 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels(envs, fontsize=9, rotation=15, ha='right')
    ax.set_ylabel('Mean final reward (±1 std across seeds)', fontsize=10)
    ax.set_title('Final Reward by Environment and Estimator',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, 'final_reward.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    return fig


# ── Figure 4: Noisy vs clean CartPole ─────────────────────────────────────────

def fig_noisy_vs_clean(df: pd.DataFrame, out_dir: str):
    """
    Side-by-side advantage variance comparison between clean and noisy CartPole.
    This is the primary plot for testing the theoretical prediction that
    Sobolev smoothing reduces variance under noisy TD residuals.
    """
    clean = 'CartPole-v1'
    noisy = 'CartPole-v1 (noisy)'

    if clean not in df['env'].values and noisy not in df['env'].values:
        print("  Skipping noisy_vs_clean.png: CartPole data not found in CSV")
        return None

    estimators = df['estimator'].unique()
    fig, axes  = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle('Advantage Variance: Clean vs Noisy Reward (CartPole-v1)',
                 fontsize=12, fontweight='bold')

    for ax, env, title in [
        (axes[0], clean, 'Clean rewards'),
        (axes[1], noisy, 'Noisy rewards (σ=1.0)'),
    ]:
        sub = df[df['env'] == env]
        if len(sub) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title, fontsize=11, fontweight='bold')
            continue
        for est in estimators:
            plot_with_ci(ax, sub, 'global_step', 'adv_var', est,
                         smooth_w=SMOOTH_WINDOW * 2)
        setup_ax(ax, xlabel='Environment steps',
                 ylabel='Advantage variance', title=title)
        ax.legend(fontsize=9, framealpha=0.7)
        ax.set_yscale('log')

    # Add annotation explaining the key comparison
    fig.text(0.5, -0.03,
             'Lower variance is better. Sobolev smoothing should have most impact '
             'in the noisy regime (right panel).',
             ha='center', fontsize=9, style='italic', color='#555555')

    fig.tight_layout()
    path = os.path.join(out_dir, 'noisy_vs_clean.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',      default='benchmark_results.csv')
    parser.add_argument('--out-dir',  default='plots')
    parser.add_argument('--no-show',  action='store_true',
                        help='Save figures without displaying them')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}")
        print("Run ppo_benchmark.py first to generate results.")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"  Environments: {sorted(df['env'].unique())}")
    print(f"  Estimators:   {sorted(df['estimator'].unique())}")
    print(f"  Seeds:        {sorted(df['seed'].unique())}")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nSaving figures to {args.out_dir}/")

    figs = [
        fig_learning_curves(df, args.out_dir),
        fig_advantage_variance(df, args.out_dir),
        fig_final_reward(df, args.out_dir),
        fig_noisy_vs_clean(df, args.out_dir),
    ]

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()