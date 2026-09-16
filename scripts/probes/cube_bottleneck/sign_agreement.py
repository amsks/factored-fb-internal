"""Figure-5 action-gap metric: does the critic preserve the SIGN of the control-critical
action gap the theory (Eq. action-gap, Lemma rare-bottleneck) is about?

Reduction of the CACHED per-candidate readings only -- no rollouts. For each restored state
we have, per candidate action, the predicted critic value q_pred and the Monte-Carlo return
return_true (from analysis/failure_figs/<env>/actions/actions_long_seed*_<epoch>_{approach,
inhand,carrying}.csv, written by failure_figs.py:810-813).

Per state we score whether the critic gets the DECISIVE pair right:
  sign_top2         a1,a2 = the two highest-return candidates (the closest decision the policy
                    must win to pick the best action). agree = 1[ sign(q[a1]-q[a2]) ==
                    sign(ret[a1]-ret[a2]) ].
  sign_secure_rel   a1,a2 = the 'secure' vs 'release' candidates (the semantic carry decision:
                    keep the cube on course vs let it go). Same sign test.
  top1              1[ argmax_a q == argmax_a ret ].
Theory reading: the lemma's d.Pi_k^0 = 0 case sends the reconstructed gap to 0, i.e. the sign
becomes a coin flip. So 0.5 = gap fully lost, 1.0 = fully preserved -- a cleaner axis than the
rank correlation currently in fig_failure_why. Prediction: FB collapses toward 0.5 at carrying,
Density-FB stays high.

    python scripts/probes/cube_bottleneck/sign_agreement.py --env cube-single-play-v0 --epoch 1000000
"""
import argparse, glob, os
import numpy as np
import pandas as pd

PHASES = ["approach", "inhand", "carrying"]


def _sign_agree(q, r):
    # agreement in sign of the two gaps; ties (r-gap == 0) are dropped by the caller
    return float(np.sign(q) == np.sign(r))


def per_state_metrics(g):
    """g: candidates of ONE (method, phase, seed, state) with q_pred, return_true."""
    q = g["q_pred"].to_numpy(float)
    r = g["return_true"].to_numpy(float)
    cand = g["candidate"].to_numpy(object)
    out = {}
    # PRIMARY: sign on the widest control-critical gap -- the true-best vs true-worst action.
    # This is the a1 (reaches the goal) / a2 (does not) pair the theory is about. Its sign is
    # preserved iff the reconstructed gap keeps the sign of the true gap; when the gap is lost
    # (Lemma: d.Pi_k^0 = 0) it becomes a coin flip at 0.5.
    ib, iw = int(np.argmax(r)), int(np.argmin(r))
    out["sign_bestworst"] = _sign_agree(q[ib] - q[iw], r[ib] - r[iw]) if r[ib] != r[iw] else np.nan
    # sign on the top-2 by true return (the closest decision)
    order = np.argsort(-r)
    a1, a2 = order[0], order[1]
    out["sign_top2"] = _sign_agree(q[a1] - q[a2], r[a1] - r[a2]) if r[a1] != r[a2] else np.nan
    # sign on secure vs release (the semantic carry decision), when both are present
    idx = {c: i for i, c in enumerate(cand)}
    if "secure" in idx and "release" in idx:
        s, l = idx["secure"], idx["release"]
        out["sign_secure_rel"] = _sign_agree(q[s] - q[l], r[s] - r[l]) if r[s] != r[l] else np.nan
    else:
        out["sign_secure_rel"] = np.nan
    # top-1 action agreement
    out["top1"] = float(int(np.argmax(q)) == int(np.argmax(r)))
    return pd.Series(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="cube-single-play-v0")
    ap.add_argument("--epoch", type=int, default=1000000)
    ap.add_argument("--root", default="analysis/failure_figs")
    ap.add_argument("--out", default="PAPER/ICLR/figures/failure")
    args = ap.parse_args()

    fs = [f for f in sorted(glob.glob(
        f"{args.root}/{args.env}/actions/actions_long_seed*_{args.epoch}_*.csv"))
        if "smoke" not in f]
    assert fs, f"no cached long CSVs under {args.root}/{args.env}/actions/"
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df = df[df["bank"].isin(PHASES)]

    keys = ["method", "bank", "seed", "state"]
    per = df.groupby(keys, sort=False).apply(per_state_metrics).reset_index()

    # aggregate over states and seeds -> one number per (method, phase, metric)
    agg = (per.groupby(["method", "bank"])[["sign_bestworst", "sign_top2", "sign_secure_rel", "top1"]]
              .mean().reset_index())
    agg["bank"] = pd.Categorical(agg["bank"], PHASES, ordered=True)
    agg = agg.sort_values(["method", "bank"])
    os.makedirs(args.out, exist_ok=True)
    csv = os.path.join(args.out, "fig_failure_signgap.csv")
    agg.to_csv(csv, index=False)
    print("=== sign-agreement (1.0 = gap preserved, 0.5 = coin flip / gap lost) ===")
    print(agg.to_string(index=False))

    # ---- plot: sign_top2 per phase, FB vs Density-FB, 0.5 reference ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2.75, 2.3))
    methods = ["FB", "Density-FB"]
    colors = {"FB": "#c44e52", "Density-FB": "#4c72b0"}
    x = np.arange(len(PHASES)); w = 0.38
    for j, m in enumerate(methods):
        sub = agg[agg.method == m].set_index("bank").reindex(PHASES)
        ax.bar(x + (j - 0.5) * w, sub["sign_bestworst"].to_numpy(), w,
               label=m, color=colors[m])
    ax.axhline(0.5, ls="--", lw=0.8, color="0.4")
    ax.text(len(PHASES) - 1 + 0.15, 0.51, "gap lost", fontsize=6, color="0.4", va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels([p.capitalize() for p in PHASES], fontsize=7)
    ax.set_ylim(0.3, 1.02); ax.set_ylabel("action-gap sign agreement", fontsize=7)
    ax.tick_params(labelsize=7); ax.legend(fontsize=6, frameon=False, loc="lower left")
    fig.tight_layout(pad=0.3)
    pdf = os.path.join(args.out, "fig_failure_signgap.pdf")
    fig.savefig(pdf, bbox_inches="tight")
    print("wrote", csv, "and", pdf)


if __name__ == "__main__":
    main()
