---
name: z-index-cube-double-conjunction-20260907
description: "2026-09-07 WHY the z-conditioned Density-FB is .2-.4 on cube-double vs .7 goal-conditioned: the two-cube reward is a CONJUNCTION (both cubes at goal) — (1) 50x sparser in play data than single-cube (both-placed .04-.14% of transitions → 4-23 positives per 10k relabel, ESS ≈3 of the 256 frozen pool vs 15-30 on cube-single); (2) a linear index in B cannot represent a product kernel: on the (old-reward) 1M cube-double ckpt the encoder's z_both has cos .6-.8 to EACH single-cube marginal index while the two marginals are orthogonal (cos ~.04) ⇒ z ≈ sum of marginals = 'cube0 OR cube1', linear-fit R2 .2-.3 vs .92-.95 on cube-single (fixed model). The goal-conditioned variant hands the critic the exact product kernel, hence .7. Same limit as FB/SF (reward linear in features)."
metadata:
  type: project
---

Diagnostic script: scratchpad zfit_diag.py (index-fit R2 on the frozen pool, kernel ESS, relabel
positives, cos between z_both and single-cube marginal z's). ⚠ The HF folder exp/partb-tier1/
holds the OLD-reward Part B runs (config lacks measure_density_z_actor_reward); the matrix Tier-1
checkpoints (status/ files, 64371eb) were requested from the cluster on 2026-09-07 16:06 UTC
(mailbox). Re-run the diagnostic on them when they land; the DATA facts (sparsity, ESS) are
model-independent. Options if the paper needs z on cube-double: nonlinear/compositional index
(outside the linear-in-B family), larger reward-sample pools (fixes noise only, not linearity),
or state the limit and keep goal-conditioning for conjunctive object tasks.
Related: [[density-fb-z-port-gap-20260906]], [[partb-campaign-handoff-20260906]].

**CONFIRMED on the FIXED (latent_centered) 1M cube-double checkpoints (matrix tier 1, seed 0,
downloaded 2026-09-07 16:30 UTC to /tmp/claude-1001/matrix_ckpts):** obj-coords z: linfit R2
.26-.51, pool ESS 2.6-12/256, cos(z_both, marginals) .6-.9 with marginals orthogonal, and
cos(z_own, z_relabel) ≈ 0 on 4/5 tasks (two noisy estimates of an ill-posed index); per-task
own-route success .95/.50/.20/.10/.20 (task 1 is the one well-posed case, R2 .30 but consistent
routes .74). Full-state z: R2 .22-.40, same marginal pattern, success .70/.35/.05/.05/.15.
