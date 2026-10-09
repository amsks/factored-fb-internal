---
name: fb-state-run-name-map
description: cube-single STATE FB wandb groups are misleadingly named — what each actually is (verify via .hydra/overrides.yaml); + audited FB↔GCIQL bridge/morph run→success map
metadata: 
  node_type: memory
  type: project
  originSessionId: 6787d218-60cd-4a6d-ad25-71a042116338
  modified: 2026-08-13T19:38:59.231Z
---

The cube-single **state** FB wandb run-group names (entity amsks, project factored-fb)
do not match their real configs. Always confirm a run's identity from its
`.hydra/overrides.yaml`, not the group name.

- **`fb_state_L256`** — NOT L_dim=256. The `L_dim=256` override was never passed; it
  ran at `L_dim=50`, no goal_cond, no fixed_b, no onestep → it is **plain FB**
  (config-identical to `cube-single-ortho1000-lrb1e-4`, success ≈0.50). The only local
  10-seed plain-FB *checkpoints*.
- **`fb_state_v1`** — goal-conditioned FB (`goal_cond=true`), the V1 from
  `feat/goal-conditioned-fb`. Success ≈0.22.
- **`fb_state_v2`** — privileged FB (`fixed_b=cube_xyz`, backward map B = cube xyz),
  the V2. Success ≈0.06.
- **`cube-single-ortho1000-lrb1e-4`** — the *canonical* plain FB (launcher defaults),
  success 0.47, BUT saved **no checkpoints** (not in /dev/shm, no wandb .pt artifacts).
- **`fb_state_plain`** — fresh clean plain-FB re-run launched 2026-06-07 to get real
  checkpoints (see [[tmux-sessions-registry]]).

True plain FB = no `onestep`, no `goal_cond`, no `fixed_b`, no `covbal*` reweight,
`L_dim=50`, ortho_coef=1000, lr_b=1e-4. There was never an actual `L_dim=256` state run
until `fb_state_encmatch` (2026-06-07), which sets L_dim=256 + MLP encoders.

**⚠ `agent=` STRING → WHICH ACTOR (2026-08-13; cost a wrong-row number in the paper).**
In the torch FB codebase `agent: fb` is the **Gaussian + BC** actor (the DDPG+BC row),
NOT plain-flow FB; `agent: fb_flowbc` is the **flow** actor. Run *display names* embed
this (`scene-play-v0__fb__s0` vs `__fb_flowbc__s0`), so a bare `__fb__` run is ddpgbc.
This put scene's ddpgbc number (.047±.057) into the paper's FQL+BC plain-FB cell; the
flow value is .076±.057. Also beware `fb_hitdirac_flowbc` (`exp_missbase/fbregion_*`) —
a DIFFERENT method sitting right beside the reference runs (`fbref_*` = real fb_flowbc).
`tunedfb_*` = fb_flowbc with ortho_coef 1.0 (vs fbref's 1000). Only 2 state-based
fb_flowbc 500k ckpts exist per env; the `bridge_ckpt_backup/fb_outputs` cube-single ones
are `obs_type: pixels`. **Resolve every FB checkpoint from `.hydra/config.yaml`.**

**FB↔GCIQL bridge / morph run map (audited 2026-07-01 vs wandb, n=3 each unless noted).**
Two metric keys: FB/torch groups log `success/overall`; GCIQL/jax groups log
`evaluation/overall_success`. FB-side morph (start from FB, import GCIQL value objective):
`fb_state_plain` **48.0** (plain-FB anchor, n=10) · `fb_state_iqlcritic_fix` **23.6** (=the
Part-2 "FB-IQL critic=24", old lr1e-4/ortho1000) · `fb_bridge_r0_base` **58.7** (=ladder
"FB-IQL 59", sane incidentals) · `r1_decond` **58.0** (+de-cond F→φ) · `r2_dnorm` **65.3**
(+√d) · `r3_allon` **56.7** (+both) · `r3_bilinearv` **54.0** (+bilinear-V head — REFUTES it
as the 59→90 gap-closer, <57) · `a0_cubexyz` **62.7** (privileged cube-xyz reward) ·
`idB_dnorm` **65.3** (z=g identity-B +√d) · `idB_decond_dnorm` **2.7** (collapse) ·
`fb_cube_zg_iql` **44.0** · `fb_cube_iql_learnedB_r1` **20.7** · `fb_cube_decouple{,_iql,
_zher_traj}` **39.3/20.7/18.0**. Takeaway: on the FB-IQL base (~59), every structural knob
stays **54–65, none reach 90** — the lever is the objective, not FB's structure.
GCIQL-side morph (start from GCIQL-bilinear=90, impose FB restrictions, @500k):
`GCIQL-State-Flow-Bilinear-d50` **90.7** · `-d128` **91.5** · `-d5` **9.6** (DEAD floor —
never cross knobs here) · `-d50-TwoTower` **86.4** (FB two-tower ~free) · `-d50-Ortho1000`
**7.2** (FB ortho@1000 catastrophic) · `GCIQL-Bil-FBz-O1000` **25.5** (measure + z-fwd) ·
`GCIQL-Bil-MeasureOnly{,-O1000}` **9.9/8.8** · `GCIQL-Bil-MeasureSA-*` **0.0** (dead).
Full catalog of untried combos + this map = `PAPER/fb-gciql-bridge/untried-design-space.md`.
