---
name: route-a-goal-blind-bug-20260908
description: 2026-09-08 Route A is GOAL-BLIND for the HILP FAMILY ONLY (hilp, hilp_density) — no map_eval_cond hook, actor rolls on z=0; explains the old "HILP 0.83 vs 0.14 same ckpts"; eval-only, fixable by re-score
metadata:
  type: project
---

**`hilp` and `hilp_density` have no `map_eval_cond` hook, so their Route-A eval rolls the actor on
z = 0 while the env goal sits unused in `cond["goal"]`.** Found by luis 2026-09-08, verified here.

⚠ **`rldp_density` is NOT affected** — an early report said it was, from counting `def map_eval_cond`
per FILE, which misses inheritance. Checked at runtime through the MRO:
`RLDPDensityCritic(RLDPCritic)` **inherits** a working hook (`rldp_density.py:90`).
**Blast radius = HILP family only, 24 runs of 84, not 36.** Always resolve this through the MRO,
never by grepping files.

    impls/agents/__init__.py:367   cond = {**self.cond_eval, "goal": goals}   goal IS supplied
    impls/agents/__init__.py:372   if hasattr(self.critic, "map_eval_cond")   silent skip, no base default
    impls/critics/hilp.py:369      eval_cond0 -> {"z": zeros(z_dim)}
    impls/critics/hilp.py:366      cond_vec(cond) -> cond["z"]                actor sees z = 0

    hook resolved via MRO:  fb, density_fb, rldp, rldp_density(inherited), psm, hiql, iql_bilinear
    MISSING:                hilp, hilp_density        <- the bug
    correctly hookless:     crl, gciql, gcivl, qrl -- their cond_vec reads cond["goal"] directly

**The discriminator is what `cond_vec` READS, not whether a hook exists.** A raising base-class
default (my first proposal) is BOTH unimplementable — `base.py` is `class Critic(Protocol)`, which
no critic inherits, so it carries no runtime behaviour — AND wrong, since it would fire on the four
goal-consuming critics. Correct guard: at the dispatch site, plus a test over `CRITICS` asserting
every critic whose `cond_vec` reads `cond["z"]` resolves a `map_eval_cond` through its MRO.

`fb.py:89-93`'s own docstring names it — *"the policy is goal-blind (2026-08-17 FB-zero bug)"* — it was
fixed for FB in Aug 2026 and never propagated; the `hasattr` dispatch let it fail silently for a month.

**This RETROSPECTIVELY EXPLAINS [[hilp-tdjc-cube-not-inert-20260905]]**: the "0.83 reward-inference vs
0.14 under our goal-Dirac eval of the SAME checkpoints" — the 0.14 was a goal-blind policy, not a bad
goal readout. The standing "state the eval route in every table" rule is still right, but the mechanism
is a defect, not a property of HILP.

**RETRACTED (posted 16:41, withdrawn 19:23):** "HILP has a systematic 5-8x eval-route gap." That gap is
the goal never being delivered. luis's tell was better than the code read: HILP Route A pinned at
0.02-0.11 across four envs of very different difficulty while `fb_tdj` ranged 0.14-0.71 and tracked
published FB — a number that does not move with env difficulty is not measuring the env.

**SURVIVES UNTOUCHED** (all four resolve to critics WITH the hook): `fb_tdj`, `rldp`, `density_fb`,
`density_fb_z`. So Density-FB **93.1 Route A / 11.1 Route B** on cube-single stands, and so does Route B
collapsing on CONJUNCTIVE rewards (`fb_tdj` puzzle 13.7→2.7, `density_fb` cube-double 68.6→8.9).
**Affected cells: `hilp_tdjc` on all 6 envs + `hilp_density_tdjc` on the 2 cube envs. These are not results at all** and must not appear as goal-conditioned numbers.

**RULING (evalbox, 19:23) — estimate `cov` at eval time from a fixed dataset sample.** HILP trains the
actor on `project_z(phi @ pinv(cov))` (`hilp.py:400-402`), so the eval hook must match. Decided by
evidence: unpickling a checkpoint shows `hilp_stats` holds ONLY `params/mean` and `params/std`, shape
(50,) — **no covariance is persisted**, so "persist it" cannot be applied to the 36 existing
checkpoints without retraining; and "drop the whitening" would feed the actor a z it never trained on,
replacing a goal-blind number with a differently-wrong one. The map ALREADY EXISTS at `hilp_density.py:143-151` under `goal_kernel`
(`cov = pool.T@pool/N`; `z = project_z(phi_g @ pinv(cov), norm_z)`, pool from `next_observations`) —
mirror it rather than re-derive, which also guarantees train/eval consistency for the goal_kernel arm.
Constraints: fixed 100k `next_observations` sample + recorded seed; pass `pinv(cov)` via the `cond`
dict NOT `net`, so no checkpoint structure changes; re-score via `scripts/reeval_checkpoint.py`, never
retrain. **Implement on `HILPCritic` ONLY** — `HILPDensityCritic` inherits it, `rldp_density` needs
nothing. Keep it on a branch; landing mid-campaign would give later runs a repaired Route A and
earlier ones the broken one.

**Nothing was cancelled**: training never calls `map_eval_cond` (losses use `sample_cond`, correct for
all three), so every checkpoint is VALID and Route B is unaffected throughout.
⚠ Outstanding falsification test (direction FLIPPED by the inheritance correction, and called in
advance both ways): `rldp_density` inherits the hook, so its Route A should read LIKE plain `rldp`
(0.32-0.42 on antmaze-medium), NOT floor. If it comes back at floor, inheritance is not the whole
story — check before re-scoring 24 runs. See [[tdjtab-first-results-20260908]].
