# fb_tdj parity arm — launch note (2026-09-03)

Implements the minimal parity arm from `docs/notes/2026-09-03-fb-vs-tdjepa.md` §7: the
seam's plain FB / flow-BC on cube-single is a faithful port of td_jepa's training stack (that
doc's §1-4), but the seam had **no OGBench reward-relabel eval path**, so the one genuine,
unmeasured divergence — td_jepa's number is a reward-inference zero-shot readout, not the
seam's `z = B(goal_obs)` Dirac — could not even be tested. This lands that path and launches
the arm.

## What was built

1. **`env/ogbench.py`**: `cube_relabel_reward(qpos, goal_xyzs, threshold=0.04)`, transcribed
   bit-for-bit from td_jepa's `cube_reward_fn`
   (`~/git/baselines/td_jepa/metamotivo/envs/ogbench.py:30-39`) — count of cubes within 4cm
   of their goal, minus the cube count. `OGBenchAdapter.relabel_infer_batch(train_ds, env,
   task_id, num_samples, shift_reward=1.0)` draws `num_samples` random TRAIN rows, relabels
   each by its **next observation's** stored simulator qpos (never the dataset's own
   `rewards` column, which `FBDataset` zeroes for unlabelled play data — datasets.py:125-128)
   against `env.unwrapped.task_infos[task_id-1]['goal_xyzs']`. `make_env_and_datasets` now
   forces `add_info=True` (loads the `qpos` column) whenever `cfg['eval_relabel_size']` is
   set, alongside the existing `measure_qpos_slices` gate.
2. **`impls/utils/evaluation.py`**: `evaluate(..., any_step_success=False)` — when True,
   scores an episode's success as ANY step's `info['success']`, not just the final step's
   (td_jepa's accounting). Default False is bit-identical to before the parameter existed
   (this env build's `terminate_at_goal=True` default means the two usually coincide anyway,
   so the flag is what makes that non-load-bearing rather than assumed).
3. **`impls/main.py`**: a SECOND eval route, gated on `cfg['eval_relabel_size']`, that runs
   **alongside** — never instead of — the existing goal-conditioned per-task loop. For every
   real task_id (`env.unwrapped.task_infos`, independent of `eval_aggregate`): relabel via
   `adapter.relabel_infer_batch`, `agent.infer_eval_cond(next_obs, rewards)`, wrap in the
   existing `InferredCondAgent` (so the goal frame is never forwarded — reuses the
   `--eval_on_inferred_cond` machinery's wrapper, not a new one), roll
   `cfg['eval_relabel_episodes']` (default 10) any-step episodes, log to
   `eval_relabel/task<N>`. Both routes are logged to **separate files** —
   `eval.csv` (unchanged) and a new `eval_relabel.csv` — because `CsvLogger` freezes its
   header on the first row logged; sharing one file would silently drop the relabel route's
   `inferred_cond_*` witness columns (and its rows share the same `task_id` as the
   goal-conditioned route's, so they'd also be indistinguishable). An adapter without
   `relabel_infer_batch` (every non-cube domain) raises `NotImplementedError` naming itself,
   rather than silently skipping the requested relabel eval.
4. **`impls/configs/critic/fb_tdj.yaml`** = `fb.yaml` at `ortho_coef: 1000` (same value as
   `fb_ref.yaml`, kept as its own config/alias so this arm's runs and CSV/wandb group aren't
   tangled with `fb_ref`'s earlier audit-arm history). `fb_tdj` added to
   `impls/agents/__init__.py::CRITIC_ALIASES` -> `FBCritic`.

## Protocol (per the doc's §7)

- `eval_relabel_size=10000` rows/task, `eval_relabel_shift=1.0` (turns the raw
  `successes - num_cubes` signal into a 0/1 cube-at-goal indicator on cube-single),
  `eval_relabel_episodes=10`, any-step success, 5 tasks — matching td_jepa's protocol and
  the bridge's `origin/gciql-fb-bridge` 0.420 cell's `configs/train.yaml` knobs
  (`eval_relabel_size 10_000`, `eval_shift_reward 1.0`, `eval_n_episodes 10`).
- The **existing** goal-conditioned route is untouched (`--eval_episodes` stays at its
  default 20), so both numbers are reportable for every run: `eval.csv` for the existing
  route, `eval_relabel.csv` for the new one.

## Verification before launch

- `tests/impls/test_ogbench_relabel.py`: 18 new tests — hand-computed `cube_relabel_reward`
  (single AND double cube), the any-step reducer (a synthetic flicker trajectory: final-step
  success False, any-step True), `relabel_infer_batch` reading the NEXT observation's qpos
  (recovered by row-matching, not by trusting internal RNG draws) against the right task's
  goal, an end-to-end `impls/main.py` test proving the two eval routes run for every task,
  infer genuinely DIFFERENT z per task, and land in separate files with the goal-conditioned
  route's z untouched (`eval_cond0`) — plus a regression pair proving the relabel route is
  **absent** (no file, `relabel_infer_batch` never called) when `eval_relabel_size` is unset
  or 0, and that an adapter without the method fails loudly.
- `tests/impls/` regression: the suite is documented to core-dump when run as one process
  (`tests/impls/test_eval_cond_route.py`'s own docstring; confirmed again here — a
  `mujoco.renderer` EGL abort inside `test_density_fb_pixels.py`, unrelated to this change).
  Ran per-file instead: `for f in tests/impls/test_*.py; do pytest "$f" -q; done`.
- **Real-GPU smoke test**: `--critic fb_tdj --domain cube_single --actor flowbc
  --train_steps 30 --eval_interval 15 --eval_episodes 2 --override
  eval_relabel_size=200 --override eval_relabel_episodes=1` against the real
  `cube-single-play-v0` dataset on GPU. Both routes ran at every eval point (steps 1/15/30),
  `eval_relabel` rows carried `inferred_cond=1.0`, `inferred_cond_goals_dropped=200` (proof
  the goal frame was live and intercepted every step), and `inferred_cond_cos_neutral`
  differed per task (proof the inferred z is genuinely task-specific, not a shared
  constant-reward fallback). No errors.

## Launched matrix

`scripts/launch_fb_tdj_parity.sh` — tmux session **`fbtdj`**, 6 windows, GPUs 0-5:

| window | lr_b | seed | GPU |
|---|---|---|---|
| tdj-lrb4-s0 | 1e-4 | 0 | 0 |
| tdj-lrb4-s1 | 1e-4 | 1 | 1 |
| tdj-lrb4-s2 | 1e-4 | 2 | 2 |
| tdj-lrb5-s0 | 1e-5 | 0 | 3 |
| tdj-lrb5-s1 | 1e-5 | 1 | 4 |
| tdj-lrb5-s2 | 1e-5 | 2 | 5 |

All: `--critic fb_tdj --actor flowbc --domain cube_single --train_steps 1000000
--eval_interval 100000 --save_interval 250000 --auto_resume --override
eval_relabel_size=10000 --override eval_relabel_shift=1.0 --override
eval_relabel_episodes=10`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
`XLA_FLAGS=--xla_gpu_enable_command_buffer=` (CUDA graphs off — see the launch script's own
comment on `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`), `WANDB_MODE=offline`. GPUs 6-7 were free
and left idle (no other campaign was running at launch time; 6 GPUs is what the matrix needs).

Runs land at `exp/cube-single-play-v0_fb_tdj_flowbc_seed_<N>/` (relative to the repo,
`impls/main.py`'s default `--save_dir`) with `train.csv`, `eval.csv` (existing
goal-conditioned route), and `eval_relabel.csv` (the new route, this arm's headline number).

**How to poll**: `tmux attach -t fbtdj` (detach `C-b d`), or
`tail -f /dev/shm/fbtdj_logs/tdj-lrb4-s0.log`. Per-run status:
`tail -n1 exp/cube-single-play-v0_fb_tdj_flowbc_seed_0/eval_relabel.csv`. All 6 windows
confirmed alive and past their first eval point (step 1) at launch +~2min; a real-GPU smoke
test at the same config confirmed both eval.csv/eval_relabel.csv rows populate correctly
before the full campaign was started.

## Open questions for whoever reads the results

- This only tests divergence #1/#2 from the audit doc (eval protocol / missing relabel
  path). #3 (ortho sweep) is partially addressed (`ortho_coef=1000`, not swept against 1.0
  here — `fb_ref` already showed ortho alone doesn't close the OLD-eval gap); #5 (500k vs
  1M) is addressed (this arm runs 1M). If `eval_relabel.csv`'s success lands near 0.42-0.53,
  the eval protocol was the whole story; if it stays near the old 0.11-0.23, the divergence
  is elsewhere and #3/#5 need their own controlled cells.
- `eval_relabel_size=10000` re-draws fresh random rows from the train buffer at EVERY eval
  point (no fixed relabel sample across steps) — matches td_jepa's protocol (which redraws
  at eval time too), but means step-to-step relabel noise is a second source of variance
  beyond the policy's own drift; not deconfounded here.
