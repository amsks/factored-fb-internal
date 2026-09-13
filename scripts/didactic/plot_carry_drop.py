"""Render the carry/drop graph and saved experiment results (no training)."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

COLORS = {'fb': '#c34a36', 'density': '#087da0', 'normalized_squared': '#6f7785',
          'fb_no_ortho': '#ca9d37', 'direct': '#6c55a0'}
LABELS = {'fb': 'FB', 'density': 'Density FB', 'normalized_squared': 'Normalized squared',
          'fb_no_ortho': 'FB, no orthogonality', 'direct': 'Direct squared table'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def save(fig, out, stem):
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(out/f'{stem}.{suffix}', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def graph(out):
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.set(xlim=(-1, 10.5), ylim=(-3.25, 3.2))
    ax.axis('off')
    pos = {'H0': (0, 1), 'H1': (3, 1), 'H2': (6, 1), 'G': (9, 1), 'R': (3, -1.7)}
    labels = {'H0': 'Held at start', 'H1': 'Carried halfway', 'H2': 'Near target',
              'G': 'At target', 'R': 'Resting on table'}
    for name, (x, y) in pos.items():
        color = '#d9eef4' if name.startswith('H') else ('#dcefdc' if name == 'G' else '#f4e4d5')
        ax.add_patch(FancyBboxPatch((x-.83, y-.43), 1.66, .86,
                     boxstyle='round,pad=.05,rounding_size=.13', facecolor=color, edgecolor='#34404c', zorder=3))
        ax.text(x, y+.13, name, ha='center', va='center', weight='bold', fontsize=12, zorder=4)
        ax.text(x, y-.18, labels[name], ha='center', va='center', fontsize=9, zorder=4)
    def arrow(start, end, color='#34404c', rad=0, style='-'):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=13,
                     linewidth=1.6, color=color, connectionstyle=f'arc3,rad={rad}', linestyle=style, zorder=2))
    for x in (0, 3, 6):
        arrow((x+.9, 1), (x+2.1, 1), COLORS['density'])
        ax.text(x+1.5, 1.2, 'Advance: 0.9', ha='center', color=COLORS['density'], fontsize=9)
        arrow((x, .53), (3+(x-3)*.15, -1.21), COLORS['fb'])
    ax.text(5.7, -.65, 'From each held state to R:\nDrop: 1\nAdvance failure: 0.1',
            ha='left', va='center', color=COLORS['fb'], fontsize=10,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 3})
    arrow((2.1, -1.7), (-.4, .52), rad=-.45)
    ax.text(-.55, -.75, 'Pickup\n1/k', ha='center', fontsize=10)
    arrow((3.4, -2.16), (2.6, -2.16), rad=-1.5)
    ax.text(3, -3.03, 'Stay: 1 − 1/k', ha='center', fontsize=10)
    arrow((8.6, 1.48), (.4, 1.48), rad=.22, style='--')
    ax.text(4.5, 2.55, 'After reaching G: reset to H0 with probability 1', ha='center', fontsize=10)
    ax.text(9, -.05, 'Reward +1\non arrival', color='#317440', ha='center', fontsize=10)
    fig.suptitle('Carry or drop: consequential decisions followed by a long wait',
                 x=.5, y=1.01, fontsize=15, weight='bold')
    fig.text(.5, .96, 'Shown: 3 transport decisions. Experiments vary depth L ∈ {1, 3, 6} and mean rest duration k ∈ {1, 10, 50}.',
             ha='center', fontsize=10)
    fig.text(.5, -.015, 'Continuing MDP · γ = 0.98 · Both actions have identical effects at G and R · All other arrival rewards are zero',
             ha='center', fontsize=9)
    save(fig, out, 'carry_drop_mdp')


def read_rows(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for k in row:
            if k != 'method':
                row[k] = float(row[k])
    return rows


def result_plots(source, out):
    rows = read_rows(source/'summary.csv')
    depths = sorted(set(r['depth'] for r in rows))
    rates = sorted(set(r['lr'] for r in rows))
    fig, axes = plt.subplots(2, len(depths), figsize=(4*len(depths), 6.1), squeeze=False)
    all_gaps = [r['start_gap_ratio'] for r in rows if r['method'] in ('fb', 'density')]
    gap_limits = (min(0, min(all_gaps)-.05), max(1.1, max(all_gaps)+.05))
    for col, depth in enumerate(depths):
        for method in ('fb', 'density'):
            for ri, rate in enumerate(rates):
                data = [r for r in rows if r['depth'] == depth and r['method'] == method and r['lr'] == rate]
                if not data:
                    continue
                ks = sorted(set(r['persistence'] for r in data))
                for row_idx, metric in enumerate(('start_gap_ratio', 'greedy_return_ratio')):
                    vals = np.array([[r[metric] for r in data if r['persistence'] == k] for k in ks])
                    ax = axes[row_idx, col]
                    ax.plot(ks, vals.mean(1), color=COLORS[method], marker='o' if ri == 0 else 's',
                            linestyle='-' if ri == 0 else '--', markersize=4,
                            label=f'{LABELS[method]}, lr {rate:g}')
                    ax.fill_between(ks, vals.min(1), vals.max(1), color=COLORS[method], alpha=.09)
                    ax.scatter(np.repeat(ks, vals.shape[1]), vals.ravel(), color=COLORS[method], s=8, alpha=.4)
        axes[0, col].set_title(f'{int(depth)} transport decision'+('' if depth == 1 else 's'))
        for row_idx in range(2):
            ax = axes[row_idx, col]
            ax.axhline(1, color='black', linestyle=':', linewidth=1, zorder=0)
            ax.set_xscale('log')
            ax.set_xticks([1, 10, 50], ['1', '10', '50'])
            ax.set_xlabel('Mean resting time k')
            ax.grid(alpha=.15)
        axes[1, col].set_ylim(-.03, 1.06)
        axes[0, col].set_ylim(*gap_limits)
    axes[0, 0].set_ylabel('Initial action gap\n(predicted / exact)')
    axes[1, 0].set_ylabel('Greedy policy return\n(relative to optimal)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(.5, -.02), frameon=False)
    mode = 'Expected TD updates' if source.name == 'expected' else 'Balanced sampled TD updates'
    fig.suptitle(f'{mode} · full representation capacity · 100,000 updates', fontsize=13)
    fig.text(.5, .93, 'Lines: means over 3 seeds; shading: seed ranges. Dotted line: exact gap or optimal return.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .91))
    save(fig, out, f'{source.name}_results')

    # Plot all declared learning rates, with no post-hoc best-rate selection.
    curve_rows = []
    for path in source.glob('depth*_*.csv'):
        curve_rows.extend(read_rows(path))
    selected_depths = [d for d in depths if d >= 3]
    fig, axes = plt.subplots(len(selected_depths), len(rates), figsize=(10, 3.2*len(selected_depths)), squeeze=False)
    for i, depth in enumerate(selected_depths):
        for j, rate in enumerate(rates):
            ax = axes[i, j]
            for method in COLORS:
                data = [r for r in curve_rows if r['depth'] == depth and r['persistence'] == 50 and r['lr'] == rate and r['method'] == method]
                if not data:
                    continue
                steps = sorted(set(r['step'] for r in data))
                errors = np.array([[max(abs(r['start_gap_ratio']-1), 1e-8) for r in data if r['step'] == step] for step in steps])
                ax.plot(steps, errors.mean(1), color=COLORS[method], label=LABELS[method], linewidth=1.3)
                ax.fill_between(steps, errors.min(1), errors.max(1), color=COLORS[method], alpha=.08)
            ax.set(xlabel='TD updates', ylabel='Absolute relative action-gap error', yscale='log',
                   title=f'L = {int(depth)}, k = 50, learning rate = {rate:g}')
            ax.grid(alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(.5, -.02), frameon=False)
    fig.suptitle(f'{mode}: learning the first transport decision', fontsize=13)
    fig.tight_layout(rect=(0, .09, 1, .95))
    save(fig, out, f'{source.name}_learning')

    # Control consequences throughout training, including both learning rates
    # and all persistence settings. No cherry-picked checkpoint is hidden.
    if 6 in depths:
        methods = ('fb', 'density', 'normalized_squared')
        fig, axes = plt.subplots(len(rates), len(methods), figsize=(12, 5.8), squeeze=False)
        k_colors = {1: '#7d8995', 10: '#bd8642', 50: '#5e3d85'}
        for i, rate in enumerate(rates):
            for j, method in enumerate(methods):
                ax = axes[i, j]
                for k, color in k_colors.items():
                    data = [r for r in curve_rows if r['depth'] == 6 and r['lr'] == rate and r['method'] == method and r['persistence'] == k and r['step'] > 0]
                    if not data:
                        continue
                    steps = sorted(set(r['step'] for r in data))
                    vals = np.array([[r['greedy_return_ratio'] for r in data if r['step'] == step] for step in steps])
                    ax.plot(steps, vals.mean(1), color=color, label=f'k = {k}', linewidth=1.5)
                    ax.fill_between(steps, vals.min(1), vals.max(1), color=color, alpha=.07)
                ax.axvline(10000, color='black', linestyle=':', linewidth=.8)
                ax.set(xscale='log', ylim=(-.05, 1.06), xlabel='TD updates',
                       title=f'{LABELS[method]}, lr = {rate:g}')
                ax.grid(alpha=.12)
            axes[i, 0].set_ylabel('Greedy return / optimal return')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(.5, -.01), frameon=False)
        fig.suptitle(f'{mode}: temporary control errors with six transport decisions', fontsize=13)
        fig.text(.5, .93, 'Mean and range across 3 seeds. Vertical line marks 10,000 updates; training continues to 100,000.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .06, 1, .91))
        save(fig, out, f'{source.name}_control_learning')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    parser.add_argument('--out', type=Path, default=Path('PAPER/ICLR/figures/didactic/carry_drop'))
    args = parser.parse_args()
    graph(args.out)
    if args.source:
        result_plots(args.source, args.out)
