# ExoRL cheetah: model-based task-vector inference does not close the ridge-vs-oracle gap (2026-09-20)

## Question

On the round-3 cheetah checkpoints the task vector inferred by the paper protocol (ridge
regression of 50k reward samples on the centered backward features) reaches walk returns of
763 (seed 2) and 512 (seed 4), while a search over task vectors scored by environment
rollouts finds 911 and 795 (scripts/exorl_zsearch.py, exp/zsearch/cheetah.jsonl). Ridge
strength and sample count do not help. The hypothesis tested here: the linear regression
cannot express a speed-threshold reward, but the critic can evaluate ANY reward exactly, so
a search over task vectors scored by the critic's own value of the true reward should find
the better task vector with no environment access.

## Objective

For a task vector z on the sqrt(d) sphere (d = 50), the critic's value of the true reward
r at a state s and action a is the reward-weighted successor distribution over a pool of
reward-labelled next states u_j:

    q_r(s, a, z) = min over the two heads h of  sum_j p_h(u_j | s, a, z) r(u_j),
    p_h(. | s, a, z) = softmax_j( F_h(s, a, z) . B(u_j) / tau ).

The search objective is the critic's estimate of how well the frozen actor does at z under
the true reward,

    J(z) = mean over 512 dataset states s of  q_r(s, pi(s, z), z),

with pi(s, z) the actor's mode action. The pool is 4096 of the same 50k reward-labelled
next states the ridge regression sees (fixed subsample). The readout calls the critic's own
forward and backward towers, softmax temperature and head reduction
(impls/critics/density_fb.py:1113-1150), nothing re-implemented. Check: with the latent
proxy reward r_z(u) = (B_zenc(u) - mu)^T z / sqrt(d), which is what the actor ascends in
training, the readout reproduces the critic's `q_fn` to max |dq| = 9e-8 on |q| <= 0.57 at
exact-f32 matmul precision (it is 5e-4 under the GPU's default TF32 matmuls, which is
matmul rounding, not a bug; the check forces exact precision). Then the true rewards are
swapped in.

Two searches, both scored by J only:
- CEM on the sphere seeded at the ridge z (pop 64 + 16 uniform sphere draws in round 1,
  4 rounds, 8 elites, per-coordinate noise 0.6 -> 0.15).
- Adam ascent of J through the actor and the forward tower from the ridge z (200 steps,
  lr 0.05 on the pre-projection vector, best J along the trajectory kept).

Each z is then scored on the paper protocol: 10 episodes, the inferred task vector rolled
through `InferredCondAgent` (scripts/exorl_zsearch.py's scoring path).

## Results (exp/zsearch/value_infer.jsonl)

Return = environment return over 10 episodes; J = the critic's own value of the true reward
at that z (per-step reward units; walk reward in [0, 1]).

| seed / task | ridge z        | CEM z          | gradient z     | oracle z (env search) |
|-------------|----------------|----------------|----------------|-----------------------|
| 2 / walk    | 720.6 (J .452) | 656.4 (J .494) | 782.5 (J .517) | 911.0 (J .411)        |
| 4 / walk    | 504.2 (J .449) | 617.9 (J .476) | 446.1 (J .510) | 795.4 (J .348)        |
| 2 / run     | 240.5 (J .111) | 222.2 (J .120) | 222.8 (J .127) | 266.2 (J .107)        |
| 4 / run     | 115.7 (J .104) | 121.2 (J .113) | 129.9 (J .125) | 179.7 (J .091)        |

(The ridge-z returns differ from the earlier paper-protocol cells, 763/512 walk and
243/105 run, only by 10-episode evaluation noise; same z, same protocol.)

Walk gain over the ridge z, mean of the two seeds: CEM +25, gradient +2. Neither clears the
+50 extension threshold, so the 20-cell (5 seeds x 4 tasks) domain mean was not run; the
paper-protocol domain mean stays 431.0.

## What the numbers say

1. The critic's value does not rank task vectors the way the environment does. On all four
   cells the oracle z, which the environment scores highest, gets a LOWER J than the ridge z
   (walk seed 4: .348 vs .449, environment 795 vs 504). A search that maximises J therefore
   cannot find the oracle z; it moves away from it.
2. Both searches raise J monotonically (ridge < CEM < gradient on every cell, +9% to +20%)
   while the environment return does not follow: gradient ascent gains 62 on seed 2 walk and
   loses 58 on seed 4 walk; CEM loses 64 on seed 2 walk and gains 114 on seed 4 walk. Re-scoring
   the same ridge z under the paper protocol lands within about 40 of its earlier cell
   (seed 2 walk 720.6 here vs 743-770 across exp/paper_eval_exorl_sqt*.jsonl), so these
   gains are within evaluation noise. The extra J is model error being exploited, the same failure
   spec 31's z-search found on cube.
3. The gradient z drifts farther from the ridge z (cos .59-.78) than CEM (cos .94-.95) or
   the oracle (cos .80-.89), and reaches the highest J, i.e. the model's value is easiest
   to inflate in directions the environment does not reward.
4. The hypothesis's premise held: the exact-reward readout does express the threshold
   reward (J at the ridge z is 0.45 for walk against a pool reward mean of 0.23, so the
   critic does credit the policy with reaching high-reward states). What fails is the
   accuracy of the critic's value across task vectors, not the reward interface.

So the cheetah gap is not fixable by replacing the ridge readout with the critic's own value
at inference time. The oracle search needs environment rollouts because the critic's value
error across z is larger than the return differences between the ridge z and the oracle z.

## Caveats

- One pool subsample (4096 of 50k) and one state set (512) per run, fixed by the rng seed;
  J is a plug-in estimate with the pool's own sampling error, which was not measured across
  subsamples.
- 10 episodes per z, one search seed. The sign of the gains is inconsistent across seeds,
  which is the finding, not a precision problem.
- Only cheetah, seeds 2 and 4, walk and run; the 20-cell extension was gated on a walk gain
  it did not reach.
- J is the critic at the eval-time greedy action of the frozen actor; a critic error at
  off-policy z (task vectors the actor was trained on only through uniform sphere draws,
  since this arm trains at sphere-only z) is exactly the regime the search enters.

## Runtime

Per (seed, task) process on one L4: ~30 s load and ridge inference, CEM 0.2 s, gradient
ascent 7 s (jitted, 512 states x 4096 pool), then 3 x ~210 s for the three 10-episode
environment scorings (MuJoCo, CPU-bound). ~11 min per cell, two cells in parallel on
GPUs 6 and 7; 45 min wall for the four cells.

## Reproduce

    # one cell
    CUDA_VISIBLE_DEVICES=6 FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages \
      EXORL_DATA_ROOT=$HOME/exorl_data/exorl/datasets EXORL_CACHE_ROOT=$HOME/exorl_data/relabel_cache \
      XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl \
      /dev/shm/.venv-jax/bin/python -m scripts.exorl_value_infer \
        --restore_path exp/exorl_sqt/dmc-cheetah/dmc-cheetah_density_fb_z_ddpgbc_seed_2 --restore_epoch 1000000 \
        --seed 2 --override exorl_task=walk --tag sqt_s2_walk --out_jsonl exp/zsearch/value_infer.jsonl
    # the four cells as run (two tmux sessions)
    tmux new -d -s vi6 'bash scripts/launch_value_infer.sh 6 "2 walk" "2 run"'
    tmux new -d -s vi7 'bash scripts/launch_value_infer.sh 7 "4 walk" "4 run"'
    # objective and searches only, no rollouts (parity check included)
    ... --skip_env

Files: scripts/exorl_value_infer.py (objective, parity check, CEM, gradient ascent,
scoring), scripts/launch_value_infer.sh, exp/zsearch/value_infer.jsonl (one row per cell
with J and return per z, the search histories, and the z vectors), logs in
exp/zsearch/logs/value_infer_s{2,4}_{walk,run}.log.
