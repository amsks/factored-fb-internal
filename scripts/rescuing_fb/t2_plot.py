"""T2 figure: DR3 co-adaptation (coad) and q_mean vs training step, one column per arch.

Inputs: analysis/rescuing-fb/t2_dr3_stats.json (probe) + t2_train_curves.json
(training logs, for the bilinear_sg per-step q_mean — sg has only final-500k
checkpoints, so its coad appears as per-seed points at 500k).
Run with the repo torch venv: .venv/bin/python scripts/rescuing_fb/t2_plot.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
A = REPO / 'analysis/rescuing-fb'
stats = json.loads((A / 't2_dr3_stats.json').read_text())
curves = json.loads((A / 't2_train_curves.json').read_text())

# validated default categorical order (dataviz reference palette, slots 1-4)
C_BIL, C_MLP, C_SGSTABLE, C_SGDIV = '#2a78d6', '#008300', '#e87ba4', '#eda100'
INK, MUTED = '#0b0b0b', '#52514e'


def series(tag):
    rows = sorted((v for v in stats.values() if v['tag'] == tag), key=lambda v: v['step'])
    return ([r['step'] for r in rows], [r['coad_mean'] for r in rows],
            [r['q_mean'] for r in rows])


fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4), sharex='col')
fig.patch.set_facecolor('#fcfcfb')
cols = [('bilinear  ψ(g)  (exp5 sd0 + ref sd000)', 'bilinear'),
        ('mlp  (exp5 sd0)', 'mlp'),
        ('bilinear_sg  ψ(s,g)  (3 seeds, final ckpt only)', 'sg')]

for ax in axes.flat:
    ax.set_facecolor('#fcfcfb')
    ax.grid(True, color='#e8e7e3', lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c9c8c3')
    ax.tick_params(colors=MUTED, labelsize=8.5)

# --- top row: coad ---
st, cd, _ = series('exp5_bilinear_sd0')
axes[0, 0].plot(st, cd, '-o', color=C_BIL, lw=2, ms=4)
rst, rcd, _ = series('ref_bilinear_sd000')
axes[0, 0].plot(rst, rcd, 's', color=C_BIL, ms=7, mfc='none', mew=2)
axes[0, 0].annotate('ref sd000 (500k, 1M)', (rst[0], rcd[0]), xytext=(-8, -14),
                    textcoords='offset points', fontsize=8, color=MUTED, ha='right')

st, cd, _ = series('exp5_mlp_sd0')
axes[0, 1].plot(st, cd, '-o', color=C_MLP, lw=2, ms=4)

sg_style = {'sg_s0': (C_SGSTABLE, 'o', 's0 stable (0.69)'),
            'sg_s1': (C_SGDIV, '^', 's1 diverged (0.28)'),
            'sg_s2': (C_SGDIV, 'v', 's2 diverged (0.00)')}
sg_lbl_dy = {'sg_s0': -3, 'sg_s1': 6, 'sg_s2': -10}
for tag, (c, m, lbl) in sg_style.items():
    st, cd, _ = series(tag)
    axes[0, 2].plot(st, cd, m, color=c, ms=9, mec=INK, mew=0.6)
    axes[0, 2].annotate(f'{lbl}: {cd[0]:.3f}', (st[0], cd[0]),
                        xytext=(-12, sg_lbl_dy[tag]), textcoords='offset points',
                        fontsize=8.5, color=INK, ha='right', va='center')

for j in range(3):
    axes[0, j].set_ylim(0.925, 1.0)
    axes[0, j].set_title(cols[j][0], fontsize=9.5, color=INK)
axes[0, 0].set_ylabel('coad  E[φ·φ′] / √(E‖φ‖²·E‖φ′‖²)', fontsize=9, color=INK)

# --- bottom row: q_mean ---
_, _, qm = series('exp5_bilinear_sd0')
st = series('exp5_bilinear_sd0')[0]
axes[1, 0].plot(st, qm, '-o', color=C_BIL, lw=2, ms=4)
rst, _, rqm = series('ref_bilinear_sd000')
axes[1, 0].plot(rst, rqm, 's', color=C_BIL, ms=7, mfc='none', mew=2)

st, _, qm = series('exp5_mlp_sd0')
axes[1, 1].plot(st, qm, '-o', color=C_MLP, lw=2, ms=4)

for tag, (c, m, lbl) in sg_style.items():
    cv = curves[tag]
    axes[1, 2].plot(cv['step'], cv['q_mean'], color=c, lw=2,
                    ls='-' if tag != 'sg_s2' else '--')
    _, _, qm = series(tag)
    axes[1, 2].plot(series(tag)[0], qm, m, color=c, ms=9, mec=INK, mew=0.6)
axes[1, 2].axhline(0, color=MUTED, lw=1, ls=':')
axes[1, 2].annotate('q=0 (value is ≤0 by construction)', (15000, 2.5), fontsize=8,
                    color=MUTED)
axes[1, 2].annotate('s2', (455000, 4), fontsize=8.5, color=INK)
axes[1, 2].annotate('s1', (355000, -7), fontsize=8.5, color=INK)
axes[1, 2].annotate('s0', (200000, -30), fontsize=8.5, color=INK)

for j in range(3):
    axes[1, j].set_ylim(-40, 15)
    axes[1, j].set_xlabel('training step', fontsize=9, color=INK)
    axes[1, j].xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
axes[1, 0].set_xlim(0, 1050000)
axes[1, 1].set_xlim(0, 540000)
axes[1, 2].set_xlim(0, 540000)
axes[1, 0].set_ylabel('q_mean', fontsize=9, color=INK)
axes[1, 2].text(0.02, 0.02, 'lines: training-log TD-target q_mean\nmarkers: probe Q(s,a,g) @500k',
                transform=axes[1, 2].transAxes, fontsize=7.5, color=MUTED, va='bottom')
axes[1, 0].text(0.02, 0.02, 'probe Q(s,a,g), min over heads', transform=axes[1, 0].transAxes,
                fontsize=7.5, color=MUTED, va='bottom')

fig.suptitle('T2 — DR3 feature co-adaptation: coad (top) and q_mean (bottom) vs training step\n'
             'fixed batch B=4096, φ = bilinear φ-tower / MLP penultimate LayerNorm; head-mean',
             fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = A / 't2_coad_vs_step.png'
fig.savefig(out, dpi=160)
print(out)
