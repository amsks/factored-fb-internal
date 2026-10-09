---
name: handoff-sf-featuretd-20260704
description: "READ FIRST (2026-07-04, session end) — SF-feature-TD campaign on bilinear GCIQL: 8-cell ablation {agnostic,conditioned}×{identity,frozen}×{no-ortho,ortho1000} RUNNING; agnostic baseline DONE @500k (~0.08-0.10 = reproduces the KNOWN canonical-SF collapse per raymondchua ref); code UNCOMMITTED since 663a3f5. Next: collect 6 remaining cells @500k, decide vs scalar-reward-TD."
metadata:
  node_type: memory
  type: project
  originSessionId: 2ad16711-4f75-4873-a0f0-27a559213354
---

**READ FIRST. 2026-07-04 session end.** Supersedes [[handoff-identity-psi-20260704]] (which holds the
ISFB-null + the reverted measure mis-build history). Full living map: `PAPER/PROJECT-MAP.md` §5.

## What this is
Successor-**FEATURE** vector TD on the bilinear GCIQL-flow agent (`tools/wandb_mode_shim/gciql_flow.py`):
train ONLY φ (the state-action embedding) with a **per-sample** TD `φ(s,a) ← b(s) + γ·φ_target(s',a')`,
readout `Q=⟨φ,ψ(g)⟩`; the goal/base leg ψ=b is NEVER trained. NOT the successor-MEASURE (B×B matrix).

## Code (in gciql_flow.py) — UNCOMMITTED since commit 663a3f5
- Modules `GCIdentityBValue` (ψ(g)=g identity, d==obs dim=28) and `GCFrozenBValue` (ψ=frozen random
  MLP, d free=50, matches FB/GCIQL rank; stop_grad ⇒ ψ params stay at init).
- Method `sf_loss` (the per-sample vector TD). Knobs: `sf_feature_td`, `sf_base∈{identity,frozen}`,
  `fb_forward_z` (TASK-CONDITION φ: concat z=ψ(g) into φ input, à la the ref SF net cat([obs,task])),
  `sf_ortho_coef` (aux `‖E[φφᵀ]−I‖` ortho reg **on φ**; ψ ortho would be inert), `sf_coef`. All default-off.
- Tests `tests/test_gciql_sf_feature_td.py` (11, green): γ=0 semantic proofs (identity L=mean((φ−obs)²),
  frozen L=mean((φ−ψ(s))²)), ψ-stays-frozen, φ-depends-on-goal (conditioning), ortho present+finite.
- **NOTE: base SF (agnostic identity+frozen) IS committed (663a3f5); task-conditioning + ortho are NOT.**
  Commit the increment next session (push blocked on box auth).

## The 8-cell ablation matrix (cube-single state, flow actor, 3 seeds @500k, wandb amsks/factored-fb)
{agnostic, task-conditioned} × {identity d28, frozen d50} × {no-ortho, ortho1000}. wandb groups:
`GCIQL-State-Flow-SF-FeatureTD-<X>` where X ∈ {IdentityB, Frozen50, IdentityCond, FrozenCond,
Identity-Ortho1k, Frozen-Ortho1k, IdentityCond-Ortho1k, FrozenCond-Ortho1k}. Metric `evaluation/overall_success`.

## Results so far
- **AGNOSTIC baseline DONE @500k: IdentityB 0.081, Frozen50 ~0.10.** Both bases ~0.08-0.10 — reproduces
  the KNOWN canonical-SF collapse (see §ref). d=50 frozen ≈ d=28 identity ⇒ NOT a rank artifact.
- Conditioned + ortho cells: RUNNING/QUEUED at session end (early evals @100-200k also ~0.08-0.11, flat).

## Live campaigns at session close (check `tmux ls`)
- **tmux `sfcond`** — 6 conditioned no-ortho runs (IdentityCond+FrozenCond ×3), ~40-90k/500k at close.
- **tmux `sforthoall`** — UNIFIED watcher, all 12 ortho1000 runs (4 cells ×3), 2 launched (GPU2/4), rest
  queue. Script `/dev/shm/sf_ortho_all_launch.sh`, logs `/dev/shm/sf_ortho_all_logs/`.
- Watchers use the >1500MB block (launch on <500MB GPU, then wait until it exceeds 1500MB before the
  next launch). LESSON: NEVER run 2 independent watchers — they race on the same free GPU → co-init
  collision (hit + fixed this session; jax logs metrics to WANDB not stdout, so verify via wandb).

## The reference-fidelity finding (systematic debug vs raymondchua/simple_successor_features)
Repo at `…/c1a38cc3…/scratchpad/simple_successor_features`. Read in full. **We replicated `sf_canonical.py`
= their vector SF-TD, which is the paper's COLLAPSING baseline.** Their WORKING method (`sf_simple.py:342`)
is NOT a representation TD — it's **scalar reward-Q-TD** `Q=r+γ·min(Q1',Q2')` + reward-prediction basis,
≈ our **`fb_cube_zg_iql` 0.44**. So our ~0.09 is the EXPECTED collapse, not a bug. 3 fidelity gaps in our
canonical replication, being tested in order: (1) **φ task-conditioning** — ref SF net = cat([obs,
normalize(task)]) (`sf_canonical.py:226`); we test via `fb_forward_z`. (2) min-L2-norm double-SF target
(`:122-139`) — NOT done. (3) task L2-normalization (`:223`) — NOT done.

## NEXT SESSION
1. **Collect all 8 cells @500k** (`evaluation/overall_success`, mean over 3 seeds). Key comparisons:
   does task-conditioning and/or ortho1000 lift SF off the ~0.09 collapse toward `zg_iql` 0.44?
2. If NONE lift it → clean result: SF *representation* TD collapses on manipulation; only reward-in-loop
   (scalar TD ≈ zg_iql) works → the project thesis. Consider building the faithful "Simple SF" (scalar
   reward-Q-TD + reward-prediction basis) as the positive control, and/or gaps #2/#3.
3. Commit the task-conditioning + ortho increment (headline: sf task-cond + sf_ortho_coef).
4. Fold results into PROJECT-MAP §5 + isfb.tex if writing up.

Collect snippet (jax venv or .venv both have wandb):
`.venv/bin/python -c "import wandb;api=wandb.Api();[print(g,[round(r.summary.get('evaluation/overall_success',-1),3) for r in api.runs('amsks/factored-fb',{'group':g})]) for g in ['GCIQL-State-Flow-SF-FeatureTD-IdentityCond', ...]]"`

[[handoff-identity-psi-20260704]] [[handoff-torch-isfb-20260702]] [[jax-venv-location]]
[[jax-gpu-packing-preallocate]] [[always-run-in-tmux]] [[wandb-entity-amsks]] [[host-identity-you-are-aws-gpu]] [[commit-no-claude-coauthor]]
