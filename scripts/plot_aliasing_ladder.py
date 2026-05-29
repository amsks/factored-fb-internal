#!/usr/bin/env python
"""scripts/plot_aliasing_ladder.py — the representation aliasing ladder.

One robust figure for rlbrew P4. Per-task linear-probe AUC (train grasps,
independent states) for predicting success-bound vs fail-bound at every level of
representation, from raw state to the learned forward map:

  raw gripper  ->  raw gripper+joints  ->  raw full state (ceiling)
              ->  phi(s)  ->  F(s, pi, z)   [per method]

If even the raw full state caps at ~0.64 and the forward map F -- which is built
to predict successor occupancy -- does no better than phi, then no static or
predicted-future representation separates the grasps: the discriminator is
post-contact dynamics, the interaction-gap signature. Eval AUC is omitted on
purpose (high-variance over ~50 episodes); the per-task train probe is stable.

Reads analysis/repsep/grasp_feature.json and forward_probe_{fb,rldp}.json.
Run under .venv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO = Path(__file__).resolve().parent.parent
REPSEP = REPO / "analysis" / "repsep"
OUT = REPO / "PAPER" / "rlbrew" / "figures"

sns.set_palette("colorblind")
_CB = sns.color_palette("colorblind").as_hex()
RAW, FB, RLDP = "0.55", _CB[0], _CB[3]


def main() -> int:
    gf = json.loads((REPSEP / "grasp_feature.json").read_text())
    fb = json.loads((REPSEP / "forward_probe_fb.json").read_text())
    rl = json.loads((REPSEP / "forward_probe_rldp.json").read_text())

    # (label, AUC, color, group) left -> right ladder
    bars = [
        ("Gripper\n(raw)",        gf["gripper (open+contact)"]["mean"], RAW,  "Raw state"),
        ("Gripper+joints\n(raw)", gf["gripper + joints"]["mean"],       RAW,  "Raw state"),
        ("Full state\n(raw)",     gf["full state"]["mean"],             RAW,  "Raw state"),
        (r"$\phi(s)$",            fb["train"]["phi_mean"],              FB,   "FB"),
        (r"$F(s,\pi,z)$",         fb["train"]["F_mean"],                FB,   "FB"),
        (r"$Q{=}F{\cdot}z$",      fb["train"]["V_mean"],                FB,   "FB"),
        (r"$\phi(s)$",            rl["train"]["phi_mean"],              RLDP, "RLDP"),
        (r"$F(s,\pi,z)$",         rl["train"]["F_mean"],                RLDP, "RLDP"),
        (r"$Q{=}F{\cdot}z$",      rl["train"]["V_mean"],                RLDP, "RLDP"),
    ]
    labels = [b[0] for b in bars]
    vals = np.array([b[1] for b in bars])
    cols = [b[2] for b in bars]
    x = np.arange(len(bars))

    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    ax.bar(x, vals, color=cols, edgecolor="black", linewidth=0.5, width=0.74)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, color="black", ls="--", lw=1.0)
    ax.text(len(bars) - 0.4, 0.505, "chance", fontsize=8, va="bottom", ha="right")
    ax.axhline(1.0, color="black", ls=":", lw=1.0)
    ax.text(len(bars) - 0.4, 0.99, "perfect separation", fontsize=8,
            va="top", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("Linear-probe AUC\n(success- vs fail-bound, per-task)", fontsize=10)
    # group labels below the (two-line) tick labels
    for xpos, name, col in ((1.0, "Raw state", RAW), (4.0, "FB", FB),
                            (7.0, "RLDP", RLDP)):
        ax.text(xpos, -0.20, name, ha="center", fontsize=10, fontweight="bold",
                color=col, transform=ax.get_xaxis_transform(), clip_on=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = OUT / "fig_repsep_ladder.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")
    print("ladder:", {l.replace(chr(10), ' '): round(v, 3) for l, v in zip(labels, vals)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
