# Spec: MW dwell-free 1M regeneration + rerun, and 30M FB-CPR watch guide

2026-08-29 handoff. Written for a fresh agent/session taking over. Load these memory files
first: `hf-play-regeneration-comparison`, `only-1m-datasets-citable`,
`default-discount-cells-only`, `density-fb-hpo-spec`, `density-pickplace-collapse-fixed`,
`eval-reseeds-global-rng`, `humenv-cpr-arms`, `plain-language-only`, `banned-words`,
`no-claude-coauthor-trailer`. Verify every number here against the named raw files before
spending compute (docs drift; files do not).

## Part 1 — regenerate the dwell-free Metaworld datasets at 1M

**Why.** The teammate's regenerated set (HF `TommyShen/factored-fb-play`, local snapshot
`$SCRATCH/fb-data/hf-play/`) fixed the property our MW data lacked — no dwell-at-goal
(success-step fraction 0.16/0.075 vs our 0.86/0.85), ~55 goal switches per episode, ~2x
state coverage — but at 120,120 transitions/task it fails the hard 1M rule and cannot be
cited. Full comparison: `$SCRATCH/fb-data/hf-play/COMPARISON.md` (+ `_stats_{old,new}.json`).

**Target generation settings** (from `hf-play/play-resample/Metaworld__mw-push/meta.json`):
policy `scripted-goal-resample`, horizon 1001, radius 0.06, `two_goal: false`, no
stop-at-success, per-episode goal chains. Tasks: `mw-push`, `mw-pick-place`.

**Steps.**
1. Diff our collector (`data_collection/collect_mw_play_resample.py`, launcher
   `scripts/collect_play_resample_v2.slurm`) against those settings. The critical deltas to
   reproduce: goal-splice cadence (no long hold at reached goals) and radius 0.06. If our
   collector cannot express it, port the missing logic; do not approximate silently.
2. Collect a SMALL probe (120 episodes x 1001, matching the HF shape) and verify the
   distribution matches the HF set on the COMPARISON.md stats: success-step fraction
   (~0.16 push / ~0.075 pick-place), goals/episode (~55 / ~39), per-dim state std, action
   stats. This is the gate; do not scale a probe that does not match.
3. Scale to >= 1,000,000 transitions/task (=1000 episodes x 1001). Pack to the layout the
   domain yamls load (`data_path` packed npz — see `env/domains/` MW yamls and
   `load_packed`; the HF per-episode layout needs packing regardless).
   Output: `$SCRATCH/fb-data/play-splice-1m/data/metaworld/{mw-push,mw-pick-place}.npz`.
4. Record transition counts in the run manifest. The 1M rule is checked at launch AND at
   table time.

## Part 2 — rerun baselines + density_fb (hard) on the new data

**Launcher.** `scripts/launch_density_table.sh` is the existing baselines+density matrix
(read its header: `DENSITY_ACTOR`/`DENSITY_OV` must be set explicitly; the `pr/prh/pnh1m`
regime loop is where a new `psplice1m` regime slots in). Add the new regime pointing at the
Part-1 npz files rather than editing the old regimes.

**Arms.** Whatever baseline set the launcher already runs (do not invent a new one), plus
`density_fb` hard mode (`measure_density_hard` is the yaml default — verify in the emitted
config) with `ACTOR=ddpgbc_density` (never stock ddpgbc — it discards the centered
normalizer). Known per-task HP: `actor.q_coeff=0.3` on mw-pick-place (fixed a 0.133→0.900
collapse; see memory). HP source of truth: the density-fb-hpo-spec uniform config.

**Rules that bind these runs.**
- gamma=0.99 default cells only for anything table-bound.
- 3 seeds; smoke each distinct code path ~200 steps first; re-read each launched run's own
  `config.json` and confirm the intended values; state expected outcomes before results.
- Error bars understate (eval reseeds the global RNG — known repo-wide issue).
- `AUTO_RESUME=1` is now supported by `scripts/vista_train.slurm` (commit 887ad60 lineage);
  pass it so walltime kills resume instead of restarting.
- Comparison target: the same arms' numbers on the OLD 1M data (roots under
  `$SCRATCH/factored-fb-density-table/`; classify runs by their config.json, not names).
  The question this answers: does the dwell-free distribution close the MW gap to
  density_fb's OGBench-level standing, holding size at 1M?
- ManiSkill: nothing to rerun — the HF drop is byte-identical to the old 48k upstream; the
  local 1M MS regeneration remains the citable set.

## Part 3 — the 30M FB-CPR density run: what to watch

**What it is.** Meta Motivo's exact online FB-CPR recipe with only the objective swapped to
density (`impls/torch_density/fbcpr_density.py`, trainer
`scripts/fbcpr_density_train_humenv.py`, launcher `scripts/fbcpr_density_full.slurm`).
Job chain name `fbcpr-density-full`, started 2026-08-29 as 951537, work_dir
`$SCRATCH/factored-fb-humenv/fbcpr-full/`. ~293 env-steps/s → ~29h training; one 47h leg
should finish it.

**Chain mechanics.**
- Full-state checkpoint (model+5 optimizers+buffers+RNG, ~35 GB, ~45 s) every 2M steps and
  on the pre-walltime USR1 signal; latest 2 kept (~70 GB steady disk).
- Exit code 99 = deliberate resubmit marker; those legs show FAILED in sacct — NOT a crash.
  The next leg self-resubmits via ssh to the submit host, `--resume`, max 8 legs
  (`resubmit_count` file in work_dir).
- A leg that dies without the signal window (node crash) loses at most 2M steps.
- Env quirk: `CC=gcc` export is load-bearing (CC=nvc breaks triton; symptom: crash at the
  first agent update in `bmm_outer_product`).

**Health signals** (in the leg's slurm log, `logs/fbcpr-density-full.o<jobid>`; wandb is
disabled — the printed metrics dicts are the record):
- `density_ce`: was ~2.88 at 3M (pilot) and still falling; should continue down. A jump to
  ~6.1 means a from-scratch restart happened — check that resume actually fired.
- `density_ess` / `density_ess_median`: pilot ~19 / ~8.9; a collapse toward 1 means
  one-hot softmax degeneracy.
- Discriminator: `disc_loss` ~0.05, `disc_wgan_gp_loss` ~0.0007 at pilot end; explosion or
  a race to 0 both mean the WGAN broke.
- `Q_fb` finite (pilot ~9.3); NaN anywhere = stop the chain, keep the last checkpoint.

**At completion** (monitor in the launching session fires when no `fbcpr-density-full` job
has been queued for 10+ min): goal-eval the final checkpoint with the scratch eval script
`$SCRATCH/factored-fb-humenv/eval-scratch/humenv_goal_eval_local.py` (one-line deviation
from `scripts/humenv_goal_eval.py`: `from_pretrained(..., device="cpu")`), protocol
`--agent motivo --motivo_model <work_dir>/checkpoint/model --goals buffer --num_goals 4
--episodes_per_goal 3 --seed 0`, out_jsonl
`$SCRATCH/factored-fb-humenv/goal_eval_fbcpr_full.jsonl`.
References on the identical protocol: Meta Motivo released model 0.750 succ / 0.818 prox;
density pilot @3M 0.250 / 0.439; best offline arm 0.250 / 0.243. The full-budget question:
does density-in-their-recipe approach the oracle, and by how much.
Attribution rule: FB-CPR, its discriminator, AMASS regularization, and the training recipe
are Meta's work; a good number here is "density objective inside FB-CPR's recipe", never a
standalone win. The Motivo 0.750 is the reference model's score, not a baseline we beat.

**Also parked:** exact 3M rows for the XP maze fb cells (timed out ~2.7-2.8M, dead at 0.02
throughout — a continuation is possible with AUTO_RESUME but the cells are dead; only run
it if a table needs the exact 3M row).
