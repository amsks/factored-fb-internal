# Density-FB z-branch on ExoRL: implementation + hyperparameter audit (2026-09-18)

Read-only audit of the configuration running in `scripts/launch_exorl_1mrecipe.sh` (critic
`density_fb_z`, actor `ddpgbc`, 20 jobs) against the FB reference in the same pipeline
(`scripts/launch_walker_fbref.sh`, walker 777 / cheetah 366 at 1M). Code read:
`impls/critics/density_fb.py`, `impls/critics/fb.py`, `impls/actors/ddpgbc.py`,
`impls/agents/__init__.py`, `impls/utils/eval_cond.py`, `impls/utils/config_utils.py`,
`impls/utils/datasets.py`, `scripts/reeval_checkpoint.py`, `scripts/exorl_paper_eval.py`,
`env/exorl_env.py`. Checkpoint diagnostics run on CPU (script kept at
`docs/notes/2026-09-18-density-fb-exorl-impl-audit/bspread.py`). Items the 2026-09-17 audit
(`docs/notes/2026-09-17-exorl-audit/audit_dfb_exorl.md`) and the 2026-09-08 code audit
already closed are not repeated: flat training kernel on locomotion, linear-proxy actor
reward, maze actor saturation, 2M-vs-1M recipe, hit-spec parsing, inference stability.

## A. Confirmed bugs

None in the training or eval math of the z branch. Two wiring facts that are bugs in the
sense of "the launch script does not do what its comment says":

**A1. The running batch is NOT actor-lr-matched to the FB reference it is compared against.**
`scripts/launch_exorl_1mrecipe.sh:23-29` sets `lr_actor=1e-4` on every domain, citing
"td_jepa launch_fb_dmc.py / rldp_train_dmc.py / Touati all run 1e-4". That is true of
td_jepa (`metamotivo/agents/fb/agent.py:25`, `lr_actor: float = 1e-4`) but false of the FB
reference run in this pipeline: `exp/walker_fbref/*/config.json` records `lr_actor = 3e-4`,
because `impls/utils/config_utils.py:170-174` merges domain < critic < actor, so
`impls/configs/actor/ddpgbc.yaml:1` (`lr_actor: 3.0e-4`) silently overrides
`impls/configs/critic/fb.yaml` (`lr_actor: 1.0e-4`) and `launch_walker_fbref.sh` never
overrides it. Every Density-FB walker cell on disk that scored 681-717 (`exp/walker_arch`,
`exp/exorl_dfb_sup86`, `exp/exorl_dfb` 2M) also ran at 3e-4 (`config.json` of each). So the
20 running jobs are the first Density-FB ExoRL runs at actor lr 1e-4, and the arm they will
be compared to (fbref 777/366) ran at 3x that. The 681 "control" in the script comment is in
fact the matched arm; the new batch is the unmatched one. Also a landmine for the FB side:
the walker 777 number is not at td_jepa's published actor lr either.

**A2. `fb.yaml`'s `lr_actor` is dead.** `impls/configs/critic/fb.yaml` carries
`lr_actor: 1.0e-4` with the comment "mirrors get_config()", but under the merge order it can
never take effect against any actor yaml that declares `lr_actor` (all of them do). Only
`flowbc.yaml:4-9` documents this. Not a Density-FB bug, but it is why A1 happened.

## B. Hyperparameter / structural mismatches vs the FB reference, ranked by expected impact on walker / cheetah

**B1. The backward features collapse onto a cap of the sphere during training; nothing in
the objective opposes it, whereas FB's orthonormality term does.** Measured on
`exp/walker_arch/dmc-walker_density_fb_z_ddpgbc_seed_0` (the 681 recipe) on 20k walker rows,
slow index tower (online tower identical to 3 digits):

| | init | 250k | 1M | FB ref 1M (`exp/walker_fbref` s0) |
|---|---|---|---|---|
| mean pairwise cos(B(u_i), B(u_j)) | 0.38 | 0.87 | **0.95** | 0.03 (rms 0.15) |
| ‖mu‖² of 50 (B on the sqrt(50) sphere) | 19.0 | 43.7 | **47.4** | 1.3 |
| tr C (total variance of B − mu) | 31.0 | 6.3 | **2.6** | 48.7 |
| effective rank of C | 13 | 26 | 26 | 45 |
| std of the actor's per-candidate reward (B−mu)·z/√50 at the eval z | – | 0.48 | **0.25** | 0.90 |
| corr(proxy, true reward) / R² of reward on B | – | .78 / .61 | .82 / .68 | .61 / .61 |

Mechanism (`density_fb.py:1001-1004`, `stage_loss`): the logits are F·B and the softmax over
the candidate axis cancels any component of B common to all candidates, so the loss is
indifferent to B drifting toward one direction; `measure_density_b_norm=true` (`_B`,
density_fb.py:566-577) pins ‖B‖ = √50 so the drift shows up as a shrinking residual. The
ridge index is centered and normalized (density_fb.py:708-770), so the eval z is unaffected
in direction (its z^T C z sits inside the sphere-z distribution: 3.17 vs median 2.42, so
there is no train/eval mismatch in where z lives), and the reward is fit *better* than by
FB's features (R² .68 vs .61). What shrinks is the actor's training signal: q_mean falls
0.33 → 0.19 from 150k to 1M in every walker/cheetah run (`train.csv`) while FB's rises 25 →
768. With `q_normalize=false` (`ddpgbc.py:101-103`) nothing rescales it. This is the one
structural difference between the two critics that the 09-17 audit did not measure, and it
is the strongest candidate for "matched everything, still 100 short". It is not a code bug:
the design note records the ortho term as harmful on the cube goal arm, and `b_norm` as
"separately harmful on the bridge" (density_fb.py:571-575), yet the ExoRL recipe turned
`b_norm` on to copy Motivo without an ablation (`exp/walker_dfb` is the only b_norm=off
walker arm and it is confounded: critic lr 3e-4, 500k, no motivo shapes).

**B2. Actor learning rate 1e-4 vs 3e-4 (A1).** Same actor code path; a 3x lower step on a
TD3 actor whose Q signal is already 0.2-0.3 (B1). Expected effect on walker: slower policy
improvement within 1M; the 09-17 audit found walker flat from 250k at 3e-4, so this could
go either way, but it is unmatched.

**B3. min-over-heads vs mean-over-heads in the actor's Q.** Density: `q = min(q1, q2)`
(`density_fb.py:1131`). FB ref: `actor_pessimism_penalty = 0.0` in
`exp/walker_fbref/config.json` → mean of heads (`fb.py:175-181`). td_jepa's FB uses min, so
Density matches the published reference and the fbref run does not. Small on walker.

**B4. F tower 512×3 (`value_hidden_dims=[512,512,512]`, read as width 512 / 3 trunk layers
under motivo, density_fb.py:1433-1436) vs FB ref 1024×1; tau 0.005 vs 0.001.** Both were
flipped together in the 2M "reference recipe" run and scored lower (09-17 audit §3), so the
pair is tested, but never one at a time.

**B5. No left encoder.** FB ref feeds F a 256-d LayerNorm-tanh embedding of the observation
(`left_encoder 256×0 norm`, `PhiMap` with 0 hidden layers, `networks.py:56-75`); Density's
F reads the raw 24-d observation (`density_fb.py:996`). Motivo's own FB has no left
encoder, so this is a difference from the in-pipeline reference, not from the published one.

**B6. Cube defaults silently applied on DMC (`impls/configs/critic/density_fb_z.yaml`).**
Read the code for each; none is harmful-by-construction, all are unablated on ExoRL:
- `value_p_curgoal .2 / trajgoal .5 (geometric at 1−γ) / randomgoal .3`, `actor_p_trajgoal
  1.0 uniform-future` (`datasets.py:484-502`): decide which state's bump the encoded z rows
  come from. Under the flat walker kernel (09-17 §2) the goal identity barely matters.
  FB's goal rows use a random batch state instead (`fb.py:207-214`).
- `measure_density_z_rand_frac 0.5`: matches FB's `train_goal_ratio 0.5`.
- `measure_density_z_ridge_lambda_mult 0.01`, `cov_eig_floor 1e-6`: relative to tr(C)/d, so
  invariant to B1's shrinkage; floor never binds (0 eigenvalues below it at 250k or 1M).
- `measure_density_z_enc_tau 0.0005` (time constant 2k steps), `z_sigma_tau 0.005` (200
  steps): both converge in <1% of the run; slow tower equals the online tower to 3 digits
  on every statistic above, so the lag is not a train/eval mismatch.
- `measure_density_z_eval_pool_m 256`, `const_eps 1e-6`, `khot_frac 0`: not on the
  inferred-cond eval path (`map_eval_cond` returns early without a goal,
  density_fb.py:948-949).
- `gc_negative true`, dataset `rewards` column: not read by `stage_loss`; only `infer_cond`
  reads rewards and gets the relabelled task reward (`datasets.py:441-474`).
- `layer_norm true`: only the mlp archi reads it; motivo towers ignore it.
- `norm_obs` absent → off in both arms.
- `measure_density_temp 1.0`, `actor_temp None`: bit-identical training/actor softmax.

## C. Verified clean

- **Sphere draws vs encoded rows** (`_z_mix_masked`, density_fb.py:772-782): Bernoulli(0.5)
  per row, fresh √50-sphere draw, stop-gradient; both legs (actor z, value z) drawn from the
  same family with independent keys (`sample_cond`, 1163-1199).
- **B-tower input scaling** (`_density_queries`, 320-325): divides every block by eps
  (walker /15.9, quadruped /21.087). Applied identically in `stage_loss` (999),
  `update_stats` (853), `sample_cond` (1157), `infer_cond` (1288) and `map_eval_cond`
  (972), so training and eval feed B the same object. Motivo's B starts Dense→LayerNorm→
  tanh, which makes a scalar input scale nearly invisible. Not intended as a feature; not a
  mismatch.
- **stage_loss target** (996-1043): t = (1−γ)·I + γ·softmax(F_tgt(s',a⁺,z_v)·B_tgt/τ),
  stop-gradient, a⁺ = actor at the value-leg z with TD3 smoothing noise, discount 0.98 from
  the domain override, hard one-hot immediate term, both towers' targets at tau 0.005.
- **Actor reward rows** (`z_actor_reward_rows`, 1065-1093): (B_zenc(u) − mu_hat)·z/√50 on the
  batch's own 1024 next states, stop-gradient, with z = the actor-leg z; `q_fn` (1095-1136)
  uses online F/B (stop-gradient) for the weights; `actor_q_center` only touches the second
  return, which `ddpgbc` never reads, so with `q_normalize=false` the actor loss is exactly
  −mean(min(q1,q2)).
- **infer_cond** (1266-1289): 50k `next_observations` rows (`INFER_COND_MARGINAL`,
  `impls/critics/base.py:63-66`), rewards = relabelled task reward at the next state
  (`env/exorl_env.py:478`, `[1:]` shift), centered ridge on the slow tower with the stored
  EMA mu/cov (the same tower and mean the actor's reward was built from), √50 projection.
  `InferredCondAgent` forwards `goals=None`, `map_eval_cond` then returns the cond unchanged.
- **Restore**: `restore_agent(params_only=True)` restores `density_b_zenc` and
  `density_zstats` (they are `net` groups, `density_fb.py:1464-1470`); `reeval_checkpoint.py`
  rebuilds from the run's own `config.json` (`_recorded_config_overrides`).
- **Actor coupling** (`ddpgbc.py`): bootstrap noise 0.2/0.3 on `next_action_vgoal` (same
  key as FB's `next_action`, 118-133), q-noise 0.2/0.3 on the actor-loss action (94-97),
  cond vector = z on the √50 sphere for both critics, `stop_encoder_grad` is a no-op for
  both. Identical across the two critics.
- **Optimizers**: `optax.adam` per group, `lr` 1e-4 for both density towers
  (`agents/__init__.py:194`), `lr_f`/`lr_b` 1e-4 for FB; no weight decay, no clipping.
- **Running jobs** (`logs/exorl_1mrecipe/*.log`): no NaN / traceback / noop warning; the
  step-1 eval prints `[infer-cond] ... env goal is NOT supplied`; `eval.csv` carries
  `inferred_cond_norm 7.071`.
- **Tests**: see the count below.

## D. The single next experiment

One lever, on walker only, 3 seeds, 1M, everything else exactly the `exp/walker_arch`
recipe (actor lr 3e-4, so it is matched to fbref): `measure_density_b_norm=false`. It is the
knob the design note already recorded as harmful, it is the only thing that turns B1's drift
into a shrinking signal, and it has never been run under the motivo shapes. Read-out is
cheap and does not need the return: if the mean pairwise cosine of B at 250k stays near the
init value (0.4) instead of 0.87, the collapse is the b_norm interaction; the walk return
then says whether it mattered. If the cosine still climbs, the second lever is a B-spread
term (FB's ortho term on B, coef 1, on the z arm only), which is the exact term that keeps
FB's features at cos 0.03.

## Tests

`JAX_PLATFORMS=cpu MUJOCO_GL=egl python -m pytest tests/impls/test_density_fb*.py -q -x`:
**168 passed, 5 warnings, 335 s** (7 files, CPU). Diagnostic log for the table in B1:
`docs/notes/2026-09-18-density-fb-exorl-impl-audit/bspread_walk.log`.

## Probe prepared, NOT launched (2026-09-18, coordinator request)

The coordinator asked for experiment D to be launched now (walker seeds 0-2 on GPUs 0-2,
cheetah seeds 0-1 on GPUs 3-4, `exp/walker_arch` recipe with the single change
`measure_density_b_norm=false`, actor lr left at the ddpgbc default 3e-4 so it is matched to
`exp/walker_fbref`). The launcher is written and syntax-checked:
`scripts/launch_exorl_bnorm0.sh` (`bash scripts/launch_exorl_bnorm0.sh smoke DIR` runs the
300-step walker smoke on GPU 0; `bash scripts/launch_exorl_bnorm0.sh` opens tmux session
`exorl_bnorm0` with one window per job, save dir `exp/exorl_dfb_bnorm0/`, logs
`logs/exorl_bnorm0/<domain>_s<k>.log`, `JOB_DONE rc=` appended on exit). The smoke itself was
refused by this session's permission system (training launch), so no job is running, no
tmux session exists, and nothing was written under `exp/` or `logs/exorl_bnorm0/`.

Baseline for the read-out, already measured (`bspread_walk.log`): `exp/walker_arch` seed 0
at 250k has mean pairwise B cosine **0.874** (rms 0.877), effective rank **26.0**, ‖mu‖² 43.7,
and its in-run `eval.csv` at 250k is the goal-blind eval (that run predates
`--eval_on_inferred_cond`; its paper-protocol walk score is in
`exp/paper_eval_exorl_1m.jsonl`). Once `exp/exorl_dfb_bnorm0/dmc-walker_density_fb_z_ddpgbc_seed_0/params_250000.pkl`
exists, run on CPU:
`JAX_PLATFORMS=cpu MUJOCO_GL=egl FB_MUJOCO_SITE_PACKAGES=... EXORL_DATA_ROOT=... EXORL_CACHE_ROOT=... python docs/notes/2026-09-18-density-fb-exorl-impl-audit/bspread.py walk`
after pointing `dfb_dir` at the new run dir (the script hard-codes `exp/walker_arch` seed 0
and epochs 250k/1M). Decision rule: cosine near the init value 0.4 at 250k = the sphere pin
is the collapse mechanism; cosine still ~0.87 = it is not, and the next lever is a B-spread
term on the z arm.

## Probe RESULT at 250k (2026-09-18 19:30Z, exp/exorl_dfb_bnorm0, b_norm=false, walker_arch recipe)
B-spread on walker s0 (logs/exorl_bnorm0/readout_250k.log): ||mu||^2 108 vs tr C 3.1, eff. rank 32.7 -- the common
component still dominates (97% of feature energy; with b_norm on it was 95%), proxy reward std 0.35 (FB 0.90).
In-run inferred-cond returns at 250k: walker 622 / 669 / 643 vs the 1M-recipe+lr1e-4 batch 636 / 764 / 681 / 762 / 593
and the 2M runs 728 / 674 / 778 / 621 / 686; cheetah 48 / 37 vs batch 67 / 69 / 83 / 106 / 49.
=> Removing the norm does NOT stop the collapse (the candidate-softmax objective leaves B's common component
unconstrained either way) and buys nothing at 250k. NOT a lever; probe left to finish for a 1M paper score only.
Remaining unmatched knobs vs fbref: F 1024x1 + tau 0.001 (flipped together in the 2M run), the 256-d left encoder,
and (method change, not tried) an ortho term on B, which the cube results convicted (0.967 -> 0.225).

## Probe RESULT at 1M (2026-09-19 01:20Z, exp/paper_eval_exorl_bnorm0.jsonl): b_norm=false is HARMFUL
walker 518.7 (14.8; 3 seeds) vs 681 with the norm on at the same recipe; cheetah 158.6 (28.8; 2 seeds) vs 197-301.
Keep measure_density_b_norm=true. The B-collapse the audit measured is a property of the candidate-softmax objective,
not of the norm, and the norm is what keeps the actor's proxy reward usable. Closed.
