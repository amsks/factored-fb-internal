# Handoff to luis: route-3 (goal-conditioned) baselines FB / RLDP / HILP

Prepared and validated on evalbox 2026-09-16. Runs to launch: the three baselines trained as
goal-conditioned RL, to sit next to the density_fb goal arm on cube.

## Code delivery (this box has no GitHub push access, so it comes as a git bundle on this repo)
File in this HF repo: `handoff/goalcond_gc/factored-fb-goalcond-handoff.bundle` (357K).
Base = 6cc2952 (the shared push point you already have). It carries 6cc2952..3de10b8, i.e. the
route-3 pieces you are missing:
  * 305c5d4 — HILP `map_eval_cond` (goal-conditioned eval; without it HILP eval is goal-blind z=0)
  * 3de10b8 — `fb_gc` config + the `fb_gc -> fb` alias in impls/agents/__init__.py + launch_goalcond_gc.sh
(rldp_gc / hilp_gc configs are at 64371eb, already behind 6cc2952 — you have them.)

Apply, e.g.:
    huggingface-cli download amsks/factored-fb-checkpoints handoff/goalcond_gc/factored-fb-goalcond-handoff.bundle --repo-type model --local-dir /tmp/gc
    cd <your Factored-FB checkout>
    git fetch /tmp/gc/handoff/goalcond_gc/factored-fb-goalcond-handoff.bundle density-fb:gc-handoff
    git merge gc-handoff        # or: git checkout 3de10b8   (or cherry-pick 305c5d4 3de10b8)
It also brings the rest of 6cc2952..HEAD (td_jepa port, squared-loss knob, density-HILP levers,
k-hot) — all committed density-fb work; merge or cherry-pick as you prefer.

## What to run
    JOBS_ONLY=1 bash scripts/launch_goalcond_gc.sh     # 18 commands, feed your scheduler/slurm
    # or locally: GPUS="0 1 2 3 4 5 6 7" bash scripts/launch_goalcond_gc.sh
fb_gc, rldp_gc, hilp_gc on cube_single + cube_double, 3 seeds = 18 runs, 1M steps, flowbc actor,
discount 0.99, full state (no object coordinates). Eval is native goal-conditioned success
(eval.csv); NO eval_relabel / reward-inference flags. Idempotent (--auto_resume, final-ckpt skip).
Save dir exp/goalcond_gc. Details: docs/notes/2026-09-16-goalcond-gc-handoff-luis.md (in the bundle).

## Validated here before handoff
2k-step smokes on cube_single: all three build, train (finite loss), and roll the native
goal-conditioned eval across tasks; HILP eval conditions on the goal (not z=0).

## Report back
Mailbox post from luis, or status/*.json per run. Headline = eval.csv goal-conditioned success at
1M. Reference: density_fb goal arm beat GCIQL on cube-double; these say whether FB/RLDP/HILP as
GCRL match or trail it.
