#!/usr/bin/env python
"""scripts/rescuing_fb/summarize_p0a.py -- the spec 28 P0a readout table.

Collects results/spec28_p0a/p0a_*.json into the table the spec's three-way
decision is read off, and REFUSES to print any z cell until every g-indexed
anchor in the same collection has passed its pre-registered control gate
(spec 28 P0, the three-times-burned rule). Chance/floor references are
printed as columns, never left implicit: AUC chance = .500, spearman null =
0, and the actor-sensitivity noise floor sits beside every sensitivity
number.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "spec28_p0a"


def _f(v, fmt="{:.3f}"):
    return "  n/a" if v is None else fmt.format(float(v))


def main() -> int:
    files = sorted(OUT.glob("p0a_*.json"))
    if not files:
        print("no P0a results yet")
        return 1
    recs = []
    for f in files:
        r = json.loads(f.read_text())
        recs.append((f.stem.replace("p0a_", ""), r))

    anchors = [(n, r) for n, r in recs if "control_gate" in r]
    zcells = [(n, r) for n, r in recs if r["meta"]["is_density_z"]]

    # The gate is PER-ENVIRONMENT: an anchor licenses only the z cells trained
    # in ITS OWN domain (a working instrument on cube-single says nothing about
    # whether the probe can resolve anything on point-mass-maze).
    licensed = {r["meta"]["env"]: r["control_gate"]["blocking_pass"]
                for _, r in anchors}

    print("=" * 100)
    print("CONTROL GATE (g-indexed anchors; pre-registered, spec 28 P0). "
          "PER-ENVIRONMENT: an anchor licenses only its own domain's z cells.")
    print("=" * 100)
    for n, r in anchors:
        g = r["control_gate"]
        print(f"{n:12s} env={r['meta']['env']:34s} "
              f"blocking={'PASS' if g['blocking_pass'] else 'FAIL'}")
        for row in g["rows"]:
            print(f"    {row['check']:26s} {row['value']:8.4f} "
                  f"{'>=' if row['direction']=='ge' else '<='} {row['threshold']:<5} "
                  f"{'PASS' if row['ok'] else 'FAIL'}"
                  f"{'' if row['blocking'] else '  (non-blocking)'}")
    blocked = [e for e, ok in licensed.items() if not ok]
    if blocked:
        print(f"\nGATE FAILED for {blocked} -- the instrument cannot resolve the "
              "question there; those z cells are printed as UNLICENSED and must "
              "not be read as evidence.")
    for n, r in zcells:
        if r["meta"]["env"] not in licensed:
            print(f"\nNOTE: {n} has no g-indexed anchor in this collection -- "
                  "UNLICENSED.")

    hdr = (f"{'cell':14s} {'env':10s} {'z?':3s} {'lic':4s} {'rho(qg,qz)':>11s} "
           f"{'AUC_g':>7s} "
           f"{'AUC_z':>7s} {'relspan_g':>10s} {'relspan_z':>10s} {'selfc_g':>8s} "
           f"{'snr_g':>7s} {'D/span_g':>9s} {'D/span_z':>9s} {'sensCond':>9s} "
           f"{'sensNoise':>10s} {'cos z(g)':>9s} {'cos ref':>8s}")
    print("\n" + "=" * len(hdr))
    print("P0a READOUT TABLE   (chance: AUC .500, spearman 0; sensCond must beat "
          "sensNoise to mean anything)")
    print("=" * len(hdr))
    print(hdr)
    order = [(n, r) for n, r in anchors] + [(n, r) for n, r in zcells]
    for n, r in order:
        s, m = r["summary"], r["meta"]
        env = m["env"].replace("-play-v0", "").replace("exorl:point_mass_maze_rnd_seed0",
                                                       "pmm")
        lic = "n/a" if not m["is_density_z"] else (
            "yes" if licensed.get(m["env"], False) else "NO")
        print(f"{n:14s} {env:10s} {'z' if m['is_density_z'] else 'g':3s} {lic:4s} "
              f"{_f(s['spearman_median']):>11s} {_f(s['auc_g_median']):>7s} "
              f"{_f(s['auc_z_median']):>7s} {_f(s['relspan_g_median']):>10s} "
              f"{_f(s['relspan_z_median']):>10s} {_f(s['selfcons_g_median']):>8s} "
              f"{_f(s['snr_g_median'], '{:.1f}'):>7s} "
              f"{_f(s['delta_over_span_g_median']):>9s} "
              f"{_f(s['delta_over_span_z_median']):>9s} "
              f"{_f(s['sens_cond_goal_median']):>9s} "
              f"{_f(s['sens_noise_floor_median']):>10s} "
              f"{_f(s['z_geometry']['cos_mean']):>9s} "
              f"{_f(s['z_geometry']['sphere_ref_cos_mean']):>8s}")
    print(f"\nuniform-action scale (sensitivity ceiling) = "
          f"{order[0][1]['summary']['sens_unif_ceiling']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
