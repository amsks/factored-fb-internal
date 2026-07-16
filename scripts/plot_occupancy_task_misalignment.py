"""Occupancy-vs-task misalignment figure (cube vs antmaze), model-free.

Three dataset-only panels explaining why density/occupancy objectives (FB
successor measure, SF vector TD) support control on antmaze but not on cube:

  A  The gate: P(task variable displaced within K steps | interaction mode).
     Cube futures are a binary switch on the observed grasp bit (obs[18]);
     antmaze futures are one diffusion no observable splits.
  B  The temporal misalignment: mode-informativeness (ratio of conditional
     mean K-step displacements) vs horizon, against the discount kernel's
     mass. The gate matters exactly where gamma=0.99 occupancy mass isn't.
  C  The missing boundary: rho-mass of grasp events on a log axis. Mid-air
     drops (the eval failure mode) have ~1.7e-5 mass: 17 transitions in 1M.

Usage: .venv/bin/python scripts/plot_occupancy_task_misalignment.py
Writes analysis/misalignment/occupancy_task_misalignment.png (+ printed stats).
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA = os.path.expanduser('~/.ogbench/data')
OUT = os.path.join(os.path.dirname(__file__), '..', 'analysis', 'misalignment')

BLUE, BLUE_D = '#2a78d6', '#104281'
AQUA, AQUA_D = '#1baf7a', '#0a6b4a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e1'
GAMMA = 0.99
KS = np.array([1, 2, 3, 4, 5, 7, 10, 14, 20, 30, 50, 70, 100, 140, 200])


def load(name):
    d = np.load(os.path.join(DATA, f'{name}.npz'))
    return d['observations'], d['terminals']


def k_step(pos, gate, term, K, thr):
    """P(||pos_{t+K}-pos_t|| > thr | gate_t) and E[disp | gate_t], same-episode."""
    n = len(pos)
    same = np.ones(n - K, bool)
    for t in np.where(term == 1)[0]:
        same[max(0, t - K + 1):t + 1] = False
    d = np.linalg.norm(pos[K:] - pos[:-K], axis=1)
    g = gate[:n - K]
    on, off = d[same & g], d[same & ~g]
    return ((on > thr).mean(), (off > thr).mean(),
            on.mean() / max(off.mean(), 1e-9))


def main():
    os.makedirs(OUT, exist_ok=True)

    obs, term = load('cube-single-play-v0')
    cube, grasp = obs[:, 19:22], obs[:, 18] > 0.5   # units: 1 = 10 cm
    aobs, aterm = load('antmaze-medium-navigate-v0')
    xy = aobs[:, :2]
    spd = np.append(np.linalg.norm(np.diff(xy, axis=0), axis=1), 0)
    fast = spd > np.median(spd)                      # best observable split

    cub = np.array([k_step(cube, grasp, term, K, 0.2) for K in KS])   # >2 cm
    ant = np.array([k_step(xy, fast, aterm, K, 0.5) for K in KS])     # >0.5 m

    # grasp-event masses
    rel = grasp[:-1] & ~grasp[1:]
    acq = ~grasp[:-1] & grasp[1:]
    drops = rel & (cube[1:, 2] > 0.35)               # released above the table
    n = len(obs) - 1
    masses = [('cube held (any step)', grasp.mean()),
              ('grasp acquisitions', acq.sum() / n),
              ('deliberate places', (rel & ~drops).sum() / n),
              ('mid-air drops', drops.sum() / n)]
    print(f'held={grasp.mean():.3f} acq={acq.sum()} places={(rel & ~drops).sum()} '
          f'drops={drops.sum()} (rho-mass {drops.sum() / n:.1e})')

    plt.rcParams.update({
        'font.size': 9.5, 'axes.edgecolor': GRID, 'axes.labelcolor': INK,
        'text.color': INK, 'xtick.color': INK2, 'ytick.color': INK2,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
        'axes.axisbelow': True, 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'axes.titlesize': 10.5,
        'axes.titleweight': 'bold', 'legend.frameon': False,
    })
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.5, 4.1))

    # -- A: the gate ------------------------------------------------------
    axA.plot(KS, cub[:, 0], color=BLUE, lw=2, label='cube | grasped now')
    axA.plot(KS, cub[:, 1], color=BLUE, lw=2, ls='--', label='cube | free now')
    axA.plot(KS, ant[:, 0], color=AQUA, lw=2, label='antmaze | fast now')
    axA.plot(KS, ant[:, 1], color=AQUA, lw=2, ls='--', label='antmaze | slow now')
    axA.set(xscale='log', xlabel='horizon K (steps)',
            ylabel='P(task variable displaced within K)',
            title='A · Task-variable futures are gated on cube,\nnot on antmaze')
    axA.annotate('one bit (obs[18])\nseparates these', xy=(3, 0.45),
                 color=BLUE_D, fontsize=8.5, ha='center')
    axA.annotate('no observable\nsplits these', xy=(40, 0.55), color=AQUA_D,
                 fontsize=8.5, ha='center')
    axA.legend(loc='lower right', fontsize=8)

    # -- B: informativeness vs the discount kernel ------------------------
    axB.plot(KS, cub[:, 2], color=BLUE, lw=2, label='cube: grasp bit')
    axB.plot(KS, ant[:, 2], color=AQUA, lw=2, label='antmaze: best observable')
    axB.axhline(1.0, color=INK2, lw=1, ls=':')
    axB.set(xscale='log', yscale='log', xlabel='horizon K (steps)',
            ylabel='E[disp | mode on] / E[disp | mode off]',
            title='B · The gate lives at horizons the\ndiscounted occupancy barely weighs')
    axB.axvspan(1, 5, color=BLUE, alpha=0.10)
    axB.annotate('action\nselection\n(K ≤ 5)', xy=(2.1, 3.2), fontsize=8.5,
                 color=BLUE_D, ha='center')
    kmass = 1 - GAMMA ** 20
    axB.axvspan(20, KS[-1], color=INK2, alpha=0.08)
    axB.annotate(f'{1 - kmass:.0%} of γ=0.99 occupancy\nmass is at K > 20,\n'
                 'where the gate ratio = 1', xy=(58, 14), fontsize=8.5,
                 color=INK2, ha='center')
    axB.legend(loc='upper right', fontsize=8)

    # -- C: the missing boundary ------------------------------------------
    labels = [m[0] for m in masses][::-1]
    vals = [m[1] for m in masses][::-1]
    bars = axC.barh(labels, vals, color=BLUE, height=0.55)
    bars[0].set_color('#e34948')  # mid-air drops = the eval failure mode
    axC.set(xscale='log', xlabel='share of dataset transitions (ρ-mass, log)',
            title='C · The grasp-failure boundary is\nnearly absent from the data')
    for b, v in zip(bars, vals):
        axC.annotate(f'{v:.1e}' if v < 1e-3 else f'{v:.3f}',
                     (v, b.get_y() + b.get_height() / 2), xytext=(4, 0),
                     textcoords='offset points', va='center', fontsize=8.5,
                     color=INK)
    axC.annotate('17 transitions in 1M —\nthe eval failure mode',
                 xy=(3e-5, 0.35), fontsize=8.5, color='#a33',
                 ha='left')

    fig.suptitle('Why occupancy objectives fail on cube: the interaction gate is '
                 'short-horizon, and its failure boundary is unsupported',
                 fontsize=11.5, fontweight='bold', y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(OUT, 'occupancy_task_misalignment.png')
    fig.savefig(path, dpi=170, bbox_inches='tight')
    print('wrote', path)


if __name__ == '__main__':
    main()
