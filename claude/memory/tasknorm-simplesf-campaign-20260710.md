---
name: tasknorm-simplesf-campaign-20260710
description: "12 runs RUNNING in parallel (tmux tasknorm + simplesf, launched 2026-07-10 ~07:12): N1/N2 task-norm measure cells and GC-SimpleSF rungs 2/3. Both plans audited pre-launch; 3 real bugs caught. Next after these: C = shared-frozen-psi control, D = N2 disambiguation (contingent)."
metadata: 
  node_type: memory
  type: project
  originSessionId: f75106fc-71c3-4a35-b220-0550320b953f
---

**Bridge track.** Two campaigns launched in parallel on the freed GPUs, 12 runs total.

## RUNNING (launched 2026-07-10 ~07:12)
- tmux **`tasknorm`**, GPUs 0–5, logs `/dev/shm/tasknorm_logs/`
  - **N1 `GCIQL-State-Flow-Bilinear-d50-C-TaskNorm`** — measure residual tested under ν
    (`value_goals`) instead of ρ (`next_observations`). Matched to
    `C-fbz-projz-ortho1000` = **0.231 ± 0.030** in EVERY respect except the test set.
    ≈0.90 vindicates the research note's §6; ≈0.23 refutes it.
  - **N2 `...-C-TaskProj`** — + rank-1 test set. Its *stationary point* (not its loss) is GCIQL's
    backup with the learned kernel reward `⟨ψ(s′),ψ(g)⟩/√d`. ≈0.90 ⇒ the RANK was the failure;
    ≈0.23 ⇒ the KERNEL REWARD was; ≈0.12 ⇒ the diagonal estimator dominates (cell-A effect) and
    N2 says nothing about the reward.
- tmux **`simplesf`**, GPUs 6,7,4,5,2,3, logs `/dev/shm/simplesf_logs/`
  - **rung 2 `GCIQL-State-Flow-SimpleSF-Arch`** — trunk split, ψ learned end-to-end. Isolates the
    ARCHITECTURE vs bilinear GCIQL **0.904 ± 0.019**. Expect 0.84–0.90 (B1 = 0.844).
  - **rung 3 `...-SimpleSF-Grounded`** — Chua et al.'s cell: ψ's ONLY gradient is the reward
    regression. Predicted **0.0–0.2**; near 0.90 falsifies "grounding is a repair, not a virtue".

Monitor: `.venv/bin/python scripts/campaign_monitor.py` (all four cells registered, seed-paired).

## PRE-REGISTERED CHECK (rung 3) — read BEFORE the success number
`simplesf/reward_loss` must fall well below the **class prior 0.166** (`r ∈ {−1,0}`, ~79% at −1 ⇒
predicting the mean scores `0.79×0.21`). A plateau at the prior means the reward is NOT linearly
readable in `b` — the reference's load-bearing assumption failing — and rung 3 says nothing about
grounding. **@20–25k it reads 0.237/0.270, i.e. still ABOVE the prior.** Watch it fall.
Architecture invariants confirmed live: `|b| = 1.0000`, `|z| = 7.0711 = √50`.

## Audit before launch — 3 real bugs caught (all fixed, none reached a run)
1. **Float32 reduction order (would have confounded the experiment).** The task-norm split path
   recomputed the Dirac term with a fresh einsum while the legacy path extracts it from the
   already-built `M − γtM`. Same math, different rounding — and the Dirac is a small number from
   cancellation (~0.0167), so they disagreed by **~1e-3 relative**. N1/N2 would have differed from
   the anchor in the REWARD term as well as the test set. Fixed (`231541b`): Dirac + ortho are now
   **bit-identical** across anchor/N1/N2; only `offdiag` moves.
2. `measure_scale_norm` would silently break the same invariant (latent; knob off in all configs).
   Each term now normalizes by its own matrix (`b0eaad3`).
3. `iql_ortho_coef` + `value_arch='simplesf'` crashes on a 3-tuple-vs-4-tuple unpack. Guarded
   (`4defb44`).

Also verified numerically against commit `fb83d68`: stock GCIQL, bilinear d50, B1, the C anchor and
psi_base-frozen all produce **bit-identical `total_loss`** — nothing existing moved. 49 tests green.
Both plans' test suites never called `agent.update()`; I added a 5-step smoke for all cells.

## QUEUED after these (user's order: A,B parallel → then C,D)
- **C — shared-frozen-ψ control** (3 runs, NOT implemented). The `psi_base` frozen arm gives the
  value net and critic INDEPENDENT random bases; that asymmetry is an untested alternative
  explanation for the underfitting attributed to "a random basis can't express Q". The frozen arm
  carries the headline (rank-matched cell), so this control matters for the writeup.
- **D — N2 disambiguation** (3 runs, contingent). Only if N2 ≈ 0.12, to separate rank from
  estimator.

Specs: `docs/superpowers/specs/2026-07-10-{gc-simple-sf,task-weighted-moments}-design.md`
Plans: `docs/superpowers/plans/2026-07-10-{gc-simple-sf,task-weighted-moments}.md`
Living doc: `PAPER/fb-gciql-bridge/sf-design-space.md` §§1d, 1e.

[[sf-joint-psi-ortho-campaign-20260710]] [[e3-sf-rewreg-campaign-20260709]]
[[results-csv-pipeline]] [[always-run-in-tmux]] [[wandb-entity-amsks]]
[[commit-no-claude-coauthor]] [[jax-venv-location]]
