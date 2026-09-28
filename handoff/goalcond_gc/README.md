# Route-3 (goal-conditioned) baselines — handoff for Luis

Prepared and validated on evalbox 2026-09-16. This is the FB / RLDP / HILP side of the
route-3 comparison: each baseline trained as a goal-conditioned RL algorithm, to sit next to
the `density_fb` goal arm (which already has cube-single / cube-double, 3 seeds).

## What route 3 is
Every training transition draws its conditioning from a hindsight goal (a future state on the
same trajectory), so the actor is extracted as a pure goal-reacher; at eval the single test
goal is mapped the same way and the policy is rolled out. The critic loss is unchanged — only
the policy's conditioning distribution collapses onto reachable goals. What the goal is passed
through before it conditions the actor differs by method:
- fb_gc:   z = B(goal)         (backward embedding), hindsight-goal fraction = 1.0
- rldp_gc: z = B(goal)         (RLDP backward map),  hindsight-goal fraction = 1.0
- hilp_gc: phi(goal), whitened (HILP feature),       mix ratio = 1.0
All three use FULL STATE (no privileged object coordinates), flowbc actor, discount 0.99,
matching the density_fb goal arm's raw-goal conditioning.

## Configs
- `impls/configs/critic/fb_gc.yaml`  — NEW (fb.yaml with the hindsight-goal fraction set to 1.0).
- `impls/configs/critic/rldp_gc.yaml`, `hilp_gc.yaml` — already in the repo (2026-09-06), unused until now.
- `impls/agents/__init__.py` — added the alias `fb_gc -> fb` (without it the run dies with `KeyError: 'fb_gc'`).
Grab these from git (branch `density-fb`, commit below) or from the files attached to the HF message.

## What to run
`scripts/launch_goalcond_gc.sh` — fb_gc, rldp_gc, hilp_gc on cube_single + cube_double, 3 seeds
= 18 runs, 1M steps, native goal-conditioned eval (eval.csv; NO reward-inference / eval_relabel
flags — the goal-conditioned checkpoints have no reward-inference path). Idempotent
(`--auto_resume`, final-checkpoint skip). Save dir `exp/goalcond_gc`.

    DRY=1 bash scripts/launch_goalcond_gc.sh          # print the 18-job list
    JOBS_ONLY=1 bash scripts/launch_goalcond_gc.sh    # one shell command per line (feed your scheduler/slurm)
    GPUS="0 1 2 3 4 5 6 7" bash scripts/launch_goalcond_gc.sh   # local tmux, one job per GPU

scene_play / puzzle_3x3 are valid (`ENVS="... scene_play puzzle_3x3"`), but their density_fb
goal-arm counterpart has not been run, so the four-method comparison is only complete on cube.

## Validated on evalbox before handoff
2k-step smokes on cube_single: all three build, train (finite loss), and roll the native
goal-conditioned eval across tasks. HILP's `map_eval_cond` is present (`impls/critics/hilp.py:375`),
so its eval conditions on the goal rather than the old goal-blind z=0.

## Report back
Per-run status via the mailbox (`python scripts/mailbox.py post --from luis --text "..."`), or
the final numbers however is easiest. The headline is `eval.csv` goal-conditioned success at 1M.
Expected reference points: density_fb goal arm beat GCIQL on cube-double; these say whether
FB / RLDP / HILP as GCRL match or trail it.
