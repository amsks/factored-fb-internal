# Spec 27 — Implementation queue (2026-08-14): what remains to build

Status: WRITTEN 2026-08-14, for a fresh implementing agent. Each item is a
self-contained contract; do them in order. House rules apply everywhere:
TDD, default-off bit-identical knobs, thin wrappers over audited eval code
(the v1-E2 lesson: a reimplemented protocol produced a broken instrument —
every new eval must pass a control gate against archived numbers before
its new cells are read), tests via `.venv/bin/python -m pytest`, evals on
CPU jax (`/dev/shm/.venv-jax/bin/python`, `CUDA_VISIBLE_DEVICES=` empty),
long runs in tmux, wandb entity amsks, no co-author lines in commits.

## Q1 — FB baseline seed parity (training + probe rerun)

The paper's plain-FB zero-shot column (density-fb.tex tab:zeroshot) rests
on 1 checkpoint (cube-single) and 2 per multi-object env; every density
cell has 3. Contract:
- Enumerate existing state-based flow-actor torch FB checkpoints
  (agents/fb stack; audit by recorded config, groups lie). Train the
  MISSING seeds to reach 3 per env (cube-single, cube-double, scene),
  stock FB recipe, same steps as the existing ckpts (read their configs;
  do not invent HPs).
- Rerun the fbzs probes (the script behind `results/fbzs_*.json` — find
  it via git log f63abb9; it already implements the z_r protocol and the
  native-z control) on the new ckpts, all three reward families.
- Deliverable: updated 3-seed FB cells as mean ± 95% t-CI, ready to paste
  into tab:zeroshot, plus the native-z control per env (the scene
  z_r-vs-native observation should be checked at 3 seeds too).

## Q2 — E2 readout on the revived-sensitivity z checkpoint (eval-only)

Spec 25's C3 arm revived measure index-sensitivity (density_w_ztv .594)
without reviving control. Determine whether that measure supports reward
readout: run `eval_density_z_rewards.py` (v2, the fixed harness) E2a+E2b
on `z25_cs_actor` (and `z25_cs_full`) at 500k — cube-single, so anchors
are the density soft/hard cs zero-shot cells. MANDATORY: re-run the
harness's bit-exact control gate first (g-indexed cd control must
reproduce its archived JSON) — no z number is read before the gate
passes. Deliverable: E2a/E2b family means vs the cs anchors + a one-
paragraph verdict: does index-sensitive-but-unextractable also mean
unreadable, or is extraction now the sole z blocker?

## Q3 — R0a-v2 synthetic redesign

Already contracted: PAPER/Multistep-FB/specs/26-bfm-roadmap.md, section
"R0a verdict and the R0a-v2 redesign". Delta on committed code (6d5c4cc).
Key requirements restated: multiplicity as sampling WEIGHT with an
empirical-kernel-invariance test; rank-constrained arms (d ∈ {2,4},
shared trunk) + fixed budget + transient logging at 1k/3k/10k; aliased
distractor region so success does not saturate; converged-tabular column
kept as the predicted-null control. Gate P1-v2: separation appears under
constraint AND vanishes in the tabular control.

## Q4 — offline supervised topline (the missing yardstick; optional, flag before starting)

The FB literature reports zero-shot performance as a percentage of
offline TD3 trained with the true task reward on the same data. We have
no such topline. Contract (only if the user green-lights the compute):
offline TD3 (or the repo's stock single-task baseline if one exists —
check first) on cube-single/double/scene with the native task rewards,
3 seeds each, standing recipe budget; deliverable = a topline row so the
paper can quote both absolute success and %-of-supervised. This is a
reporting upgrade, not a method experiment; skip if compute is tight.

## Q5 — bookkeeping (cheap, do alongside)

- results_map.py entries + CSV refresh for every group since d17 (d17*,
  d18*, d18x*, d19*, d21*, a22*, dbx*, hx*, gap*, p0*, z24*, z25*,
  fbzs) — the standing debt, now large.
- The paper's `pending` antmaze plain-FB ddpgbc cell: the launch failed
  on a missing dataset path (caption notes it); fix the path and run it
  (1 seed screen parity with the other antmaze cells), or delete the row
  cell if antmaze FB is descoped — ask the user which.

Priorities: Q2 (hours, decides the z fork) > Q3 (gates R1 and the causal
chapter) > Q1 (paper hygiene, medium compute) > Q5 > Q4 (gated on user).
