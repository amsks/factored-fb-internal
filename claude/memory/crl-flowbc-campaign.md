---
name: crl-flowbc-campaign
description: "CRL+FlowBC large-run campaign launched 2026-05-24 (status, layout, throughput)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 538b0dc6-3753-482b-a260-1fb03bbdd2b0
---

## ⛔ CRITICAL BUG — all CRL+FlowBC results before fix f7b623d are INVALID (2026-05-25)
CRL+FlowBC scored 0% success on EVERY cube-single task, all seeds, the whole 1M run
(state). Root cause: `crl_flowbc.py actor_loss` evaluated the Q-max critic with
`params=grad_params`, so `-q.mean()` backpropped into the critic; the single
optimizer inflated Q instead of training the actor → contrastive critic COLLAPSED
(q_mean→2.4e6, logits_pos==logits_neg, categorical_accuracy pinned at 1/batch
chance, value head 0/inf) → no goal signal → 0% eval. Fix `f7b623d`: call
`critic(...)` WITHOUT grad_params (matches OGBench DDPG+BC, agents/crl.py).
Verified by jax.grad probe (critic grad from actor_loss 2941→0.0). See
[[crl-actor-loss-critic-gradient]]. Whole campaign STOPPED 2026-05-25, all 8 GPUs
idle; resumed state seeds 0-5 (1M) + everything are buggy 0% and must be re-run.
Fix verified mechanically but NOT yet validated by a training run (accuracy rising /
eval>0) — do a short fixed-code sanity run before any full relaunch. User to decide
what to relaunch.

CRL+FlowBC large-run campaign launched 2026-05-24 on branch `gciql-pixel-integration`
via `scripts/run_crl_flowbc_campaign.sh` (detached). Two tracks, all at 1M steps,
wandb online (entity amsks, project factored-fb):

- **State**: `cube_single_state_crl_flowbc`, seeds 0-9, GPUs 5-7 (2/GPU, queued).
  ~67 it/s → ~4h/seed training.
- **Pixel/DrQ**: `cube_single_visual_crl_flowbc_drq`, seeds 0-4, GPUs 0-4 (1/GPU,
  ~17GB each). ~10 it/s → ~28h training/seed + eval overhead.

Run groups: `factored-fb-crl-flowbc` (state), `factored-fb-crl-flowbc-pixel-drq` (pixel).
Logs: `logs/crl_flowbc_campaign_<ts>/`. Checkpoints every 100k to `/dev/shm/gciql_outputs`,
rsynced every 10min to durable `~/gciql_runs/crl_flowbc_<ts>/`.

Enabled by two fixes committed this session: eval `sample_actions` noise-shape fix
(b47fdf2) and the NumPy `random_shifts` perf fix (349749d) — see [[ogbench-drq-augment-host-bound]].

Eval kept at eval_episodes=50 / eval_interval=100k / eval_on_cpu=0 (config defaults,
UNCHANGED) so eval is consistent across GCIQL/FB/CRL for fair comparison (user
decision 2026-05-24). Do NOT reduce eval_episodes for pixel.

## RESUME (2026-05-25): campaign died ~3h in, now resumable
Killed 2026-05-24 ~10:17 (all 11 jobs at once): SSH disconnect -> SIGHUP, which the
old launcher (`scripts/run_crl_flowbc_campaign.sh`) trapped only INT/TERM, so it died
AND its final durable rsync never ran. No crash/reboot/OOM. State seeds 0-5 reached
~690k (params_600000 in /dev/shm); seeds 6-9 never started; pixel 0-4 reached <=12%
with no usable ckpts. Durable `~/gciql_runs` is STALE (only 100k) — the live data is
in volatile /dev/shm. NOTE: durable disk `/` is 100% full (the 465G
`~/git/offline-rl-aditya` repo), so everything now runs in /dev/shm (user accepts
volatility); copy out before any reboot.

Resume plumbing added (vendored OGBench never edited — all via the shim):
- `run_gciql.py` passes `--restore_path/--restore_epoch`; sets env GCIQL_EXP_NAME /
  GCIQL_STEP_OFFSET / GCIQL_CSV_APPEND.
- `tools/wandb_mode_shim/sitecustomize.py` resume block (gated, inert unless env set):
  pins `get_exp_name`, offsets `save_agent` epoch + wandb/CSV step by restore_epoch,
  appends CSVs. So a run resumed from N writes params_{N+i} into the SAME sd00X dir =
  single-go on-disk layout. Config keys `restore_path/restore_epoch/exp_name` (null).
- main.py loop counts i=1..train_steps regardless of restore, so resume = pass
  train_steps=REMAINING (400000) + restore_epoch=600000. wandb_mode MUST be online
  (save_dir uses wandb.run.project, which the shim sets to factored-fb only online;
  offline/disabled -> 'dummy' -> wrong dir).
- Caveat: wandb shows TWO runs/seed (orig 0-600k + resumed 600k-1M, same name,
  contiguous x-axis via offset); flags.json in resumed dir reflects the resume cmd.
Verified: 16 shim tests pass; patches checked in JAX env; 2-step smoke restored a
real params_600000 and wrote params_600001/2.

Relaunch via NEW `scripts/resume_crl_flowbc_campaign.sh` (run in tmux): STATE 0-5
resume +400k, STATE 6-9 fresh 1M (2/GPU on 5-7), PIXEL 0-4 fresh 1M (1/GPU on 0-4),
all in /dev/shm, TMPDIR/WANDB_DIR/logs pinned to /dev/shm. Decision (user 2026-05-25):
all 10 state + 5 pixel @1M. Code changes UNCOMMITTED as of handoff.

Pixel eval cost notes: OGBench evals at step 1 and every 100k (`main.py:113`); eval is
5 tasks × 51 episodes (50+1 video). Cost is dominated by EPISODE LENGTH (env physics +
image render), NOT CRL compute — `sample_actions` is ~4.65ms/call. Untrained policies
that don't terminate episodes early run to max horizon → ~20s/episode (~85min/eval);
policies that terminate early run ~1.2s/episode (GCIQL-like). This is seed/policy
dependent (at launch, pixel s3 evaled at 1.2s/ep while s0-s2/s4 were 20s/ep) and is
expected to self-heal as policies learn. State eval is CPU and fast. Realistic pixel
per-seed wall: ~28h train + variable eval (likely small once trained).
