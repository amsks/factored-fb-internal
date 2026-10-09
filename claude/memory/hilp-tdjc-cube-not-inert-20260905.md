---
name: hilp-tdjc-cube-not-inert-20260905
description: "2026-09-05 HILP is NOT inert on cube — under td_jepa's cube config (hilp_tdjc: batch 256, phi lr 1e-4, 512x4 phi, left encoder) and td_jepa's reward-inference eval (eval_relabel.csv, 10k relabelled samples, shift 1.0), HILP gets 0.83±0.03 cube-single @1M (3 seeds) vs 0.14 under our goal-Dirac eval of the SAME checkpoints. Density-HILP 0.74±0.08 (worse). fb_tdj same protocol 0.378 vs 0.197 goal-Dirac. Cube-double: everything ≤0.08. The privcoord table's HILP row (.08/.09) was config+protocol, not the method."
metadata:
  type: project
---

Runs: `exp/cube-{single,double}-play-v0_hilp{,_density}_tdjc_flowbc_seed_{0,1,2}` (1M, launched
09-04 23:58 via `scripts/launch_hilp_tdjc.sh`, tmux `hilptdjc`). Two eval routes logged per run:
`eval.csv` = our goal-conditioned z=φ(goal) Dirac; `eval_relabel.csv` = td_jepa's protocol.

| cube-single @1M, 3 seeds | goal-Dirac eval | td_jepa relabel eval |
|---|---|---|
| hilp_tdjc | 0.14±0.04 | **0.83±0.03** (.80/.84/.86) |
| hilp_density_tdjc | 0.15±0.04 | 0.74±0.08 (.74/.84/.64) |
| fb_tdj | 0.20±0.04 | 0.38±0.07 |
cube-double: hilp_tdjc relabel 0.08±0.04, density 0.01; goal-Dirac ≤0.02.

**Why it matters:** (1) "HILP inert to everything" (privcoord table, 09-02) is WRONG — it was
official-ExoRL hyperparameters + the goal-Dirac readout, which HILP's φ (temporal distance) does
not support; its native LS reward-inference index works. (2) The density retrofit HURTS HILP on
cube (.83→.74), consistent with walker (768→730⚠). (3) Eval route matters as much as config:
the same checkpoint reads .14 or .83. Any table must state the route. (4) Comparison caveat from
the launcher: td_jepa reports best-of-2x2 sweep over 10 seeds; we pin BASE_CFG at 3 seeds.
Density-FB has NO relabel-protocol eval on cube yet (its z path .82 @500k is the centered
reward-sample encoder on a frozen pool — a different route). Extra-seed HILP family at 500k
(hilp_cube/oc/density_cube/density_oc, goal-Dirac): all .11–.15 cube-single, ≤.05 cube-double.
Related: [[density-fb-privileged-coords-20260902]], [[tdjepa-vs-rldp-published-20260831]].
