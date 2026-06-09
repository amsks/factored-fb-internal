#!/usr/bin/env python
"""scripts/ablation_progress.py — STEP-MATCHED, seed-matched progress check for the
PSM readout ablation (pessimism=0 + ortho-high) vs the three references.

For each in-progress ablation group we take its current eval step S (the last
completed 50k eval, per seed) and compare against PSM-tuned, PSM-default and FB
*at that same step S* — NOT at their final value — restricted to the same seeds
{0,1,2} we are currently evaluating. References' values at step S come from their
wandb history (eval logged every 50k).

References:
  - PSM tuned   (psm_state_tuned,  ortho_coef=1, lr_phi=1e-5)  — bar to beat (~0.31 final)
  - FB          (fb_state_plain  = fb_flowbc)                  — ceiling (~0.49 final)
  - PSM default (psm_flowbc_state, lr_phi=1e-4)                — older baseline (~0.22 final)

Usage: .venv/bin/python scripts/ablation_progress.py
"""
import wandb

ENTITY_PROJECT = "amsks/factored-fb"
SUCCESS = "eval/reward/eval/success"
SEEDS = [0, 1, 2]
EVAL_EVERY = 50_000

ABLATIONS = [  # (label, group-prefix)
    ("pess0", "psm_state_pess0"),
    ("ortho=100", "psm_state_orthohi__ortho_coef100"),
    ("ortho=1000", "psm_state_orthohi__ortho_coef1000"),
]
REFERENCES = [  # (label, group-prefix)
    ("PSM tuned 1e-5", "psm_state_tuned"),
    ("FB (fb_flowbc)", "fb_state_plain"),
    ("PSM default 1e-4", "psm_flowbc_state"),
]


def runs_by_seed(all_runs, prefix):
    """seed -> run (the most-advanced run per seed for that group prefix)."""
    out = {}
    for r in all_runs:
        g = r.group or ""
        if not g.startswith(prefix):
            continue
        s = r.config.get("seed")
        if s not in SEEDS:
            continue
        st = r.summary.get("_step") or 0
        if s not in out or st > (out[s].summary.get("_step") or 0):
            out[s] = r
    return out


def success_series(run):
    """[(step, success)] from a run's history where success is logged (every 50k)."""
    pts = []
    for row in run.scan_history(keys=[SUCCESS, "_step"], page_size=10000):
        v, st = row.get(SUCCESS), row.get("_step")
        if isinstance(v, (int, float)) and st is not None:
            pts.append((st, v))
    return sorted(pts)


def value_at_step(series, target):
    """Success at the eval step nearest to (and not far past) target; None if none."""
    if not series:
        return None, None
    # nearest by absolute step distance
    st, v = min(series, key=lambda p: abs(p[0] - target))
    return v, st


def main():
    api = wandb.Api(timeout=30)
    all_runs = list(api.runs(ENTITY_PROJECT, per_page=400))

    # Cache reference series once (seed -> [(step,val)]).
    ref_series = {}
    for label, prefix in REFERENCES:
        rbs = runs_by_seed(all_runs, prefix)
        ref_series[label] = {s: success_series(r) for s, r in rbs.items()}

    print(f"\n=== PSM readout ablation — STEP-matched, seeds {SEEDS} ===")
    print("(each ablation at its current eval step S; references read AT step S, same seeds)\n")

    for ab_label, ab_prefix in ABLATIONS:
        ab_runs = runs_by_seed(all_runs, ab_prefix)
        if not ab_runs:
            print(f"### {ab_label}: no runs found\n")
            continue
        # current eval step per seed = floor(step / 50k) * 50k
        ab_vals, ab_steps = {}, {}
        for s, r in ab_runs.items():
            cur = r.summary.get("_step") or 0
            ab_steps[s] = (cur // EVAL_EVERY) * EVAL_EVERY
            ab_vals[s] = r.summary.get(SUCCESS)
        # common step for the group = min completed eval step across its seeds
        S = min(v for v in ab_steps.values() if v) if any(ab_steps.values()) else 0
        present = [s for s in SEEDS if s in ab_vals and isinstance(ab_vals[s], (int, float))]
        ab_mean = sum(ab_vals[s] for s in present) / len(present) if present else float("nan")

        print(f"### {ab_label}  @ step ~{S:,}  (seeds present: {present})")
        ab_ps = ", ".join(f"s{s}={ab_vals[s]:.2f}" for s in present)
        print(f"  {'OUR ' + ab_label:18} {ab_mean:>6.3f}   {ab_ps}")
        for ref_label, prefix in REFERENCES:
            series = ref_series[ref_label]
            vals, ps = [], []
            for s in present:
                v, at = value_at_step(series.get(s, []), S)
                if v is not None:
                    vals.append(v)
                    ps.append(f"s{s}={v:.2f}@{at//1000}k")
            if vals:
                mean = sum(vals) / len(vals)
                delta = ab_mean - mean
                sign = "+" if delta >= 0 else ""
                print(f"  {ref_label:18} {mean:>6.3f}   {', '.join(ps)}   (ours {sign}{delta:.3f})")
            else:
                print(f"  {ref_label:18} {'—':>6}   (no eval at ~{S//1000}k)")
        print()


if __name__ == "__main__":
    main()
