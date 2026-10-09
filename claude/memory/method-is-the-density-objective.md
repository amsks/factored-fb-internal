---
name: method-is-the-density-objective
description: THE PAPER'S THESIS — our method is the DENSITY OBJECTIVE, applied to FB/HILP/RLDP; it must beat those same three without it. ⚠ Currently confounded: only the FB arm gets the hit kernel.
metadata:
  type: project
---

**OUR METHOD IS THE DENSITY OBJECTIVE.** Not Density-FB specifically. The contribution is the
categorical/density successor estimator, applied as a *drop-in* to three representation learners:

    FB   -> density_fb / density_fb_z      vs baseline  fb_tdj
    HILP -> hilp_density_tdjc              vs baseline  hilp_tdjc
    RLDP -> rldp_density                   vs baseline  rldp

**The claim the paper needs: the density objective beats the same learner without it, in all three
families.** A win only on FB is a much weaker paper — it reads as one lucky pairing, not an objective.

## ⚠ THE COMPARISON IS CURRENTLY CONFOUNDED — the three arms are NOT the same treatment

`measure_hit_spec` is a hand-written per-env string naming which observation columns are the task and
what radius counts as success (cube `19:22:0.4` = cube xyz within 0.4; antmaze `0:2:0.5` = ant xy
within 0.5 m). It is **transcribed from each env's own success rule**, i.e. the evaluation criterion
handed to the learner at training time. Verified in code 2026-09-08:

    density_fb / density_fb_z   parse_hit_spec called UNCONDITIONALLY (density_fb.py:1128)
                                -> kernel ALWAYS ON. B eats the scaled hit columns; the actor
                                   ascends a reward on the raw hit columns (u_raw vs u_sc, :432)
    rldp_density                goal_kernel: bool = False (rldp_density.py:99), gate at :231
                                rldp_density.yaml has NO goal_kernel key -> kernel OFF
    hilp_density_tdjc           goal_kernel: bool = False (hilp_density.py:69), gate at :186
                                hilp_density_tdjc.yaml has NO goal_kernel key -> kernel OFF

**So the FB arm gets a hand-specified success kernel and the HILP and RLDP arms get none.** The
campaign passes `measure_hit_spec` to all three, but on two of them it is INERT — carried only so
`config.json` self-evidences the cell.

**And the results track the kernel, not the objective:** density_fb 93 vs fb_tdj 18 on cube-single;
rldp_density 23 vs rldp 15; hilp_density ~= hilp_tdjc. The one arm with the kernel wins big; the two
without it are flat. As it stands the data supports "kernel helps", NOT "density objective helps".

Under `measure_density_hard=true` (the whole campaign) the kernel is NOT the TD target
(`density_fb.py:911`, `k = jnp.eye(B)`); it enters via B's input, the actor's reward, and the eval
goal->z map.

## The two controls that decide it (neither has ever run off cube)

    density_fb_state    B eats FULL STATE, kernel intact. Isolates B's INPUT.
                        cube-single: 0.91 vs density_fb 0.93 => the hit COLUMNS are not the win.
                        ZERO manifest lines; cube-only on the hub.
    rldp_density_gk     rldp_density + goal_kernel=true. Isolates the KERNEL from FB's estimator.
                        `rldp_density_gk.yaml` exists and its own header calls it "the
                        discriminating variant for the density_fb (0.97) vs rldp_density (0.24) gap".
                        NEVER run on non-cube.

⚠ **Run `rldp_density_gk` via `--critic rldp_density_gk`, NEVER `--critic rldp_density --override
goal_kernel=true`** — the run dir is composed from the critic FLAG (`log_utils.py:7-11`), so the
override form writes into the SAME dirs as the kernel-off campaign cells. That is the fb_tdj
contamination again. Kernel-on and kernel-off are DIFFERENT METHODS and need different rows.

`density_fb.py:430-435` says separating `u_raw` (kernel operand) from `u_sc` (B's input) is "the ONLY
way to ask what Density-FB's cube number owes to the object-coordinate outcome space rather than to
the categorical estimator." The repo's own authors flagged this as the question.
See [[tdjtab-first-results-20260908]], [[route-a-goal-blind-bug-20260908]].
