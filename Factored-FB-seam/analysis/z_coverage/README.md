# z coverage: where the actor's training task vectors lie vs where evaluation task vectors land

Question (Ben): FB-style methods assume the training distribution of task vectors z covers the
z's that real rewards produce. Our reward-sample-index Density-FB (`density_fb_z`) trains the
actor on z's built from hindsight goals (plus, in the campaign config, uniform sphere draws).
Does the Route-B evaluation z (10k relabelled reward samples) land inside that region?

## Files
- `compute_z_coverage.py`  GPU driver (forward passes only). Loads each checkpoint exactly as
  `impls/main.py` does (`make_config` from the run's recorded config, `Agent.create`,
  `restore_agent(params_only=True)`), then builds and saves the z's. Nothing in `impls/` or
  `env/` is patched.
- `plot_z_coverage.py`     the schematic figure (CPU). Writes
  `PAPER/ICLR/figures/z_coverage/fig_z_coverage{,_exact}.{pdf,svg,png}`; SVG text is editable
  (`svg.fonttype='none'`).
- `z_<env>.npz`            campaign checkpoints (1M, `density_fb_z`, latent_centered actor reward,
  z_rand_frac 0.5). `z_<env>_exact.npz`: 500k exact-reward checkpoints (kernel actor reward,
  z_rand_frac 0, `exp/kz_conf`). Keys: `z_goal` [4096,50], `z_sphere` [4096,50], `z_eval` [5,50],
  `z_khot` [5,50], `z_dense` [5,50], `labels_*`, `meta` (json: run dir, step, seed, hit spec,
  reward-sample stats per z, summary metrics).
- `coverage.csv`           one row per (set, env, family, task); columns below.
- `log_campaign.txt`, `log_exact.txt`  driver logs.

## Checkpoints
| set | env | run | step | seed |
|---|---|---|---|---|
| campaign | cube-single | HF `exp/cube-single-play-v0_density_fb_z_flowbc_seed_0` | 1M | 0 |
| campaign | cube-double | HF `exp/cube-double-play-v0_density_fb_z_flowbc_seed_0` | 1M | 0 |
| campaign | scene | HF `exp/scene-play-v0_density_fb_z_flowbc_seed_0` | 1M | 0 |
| campaign | puzzle-3x3 | HF `exp/puzzle-3x3-play-v0_density_fb_z_flowbc_seed_0` | 1M | 0 |
| campaign | antmaze-medium | HF `exp/antmaze-medium-navigate-v0_density_fb_z_flowbc_seed_2` (only seed on the hub) | 1M | 2 |
| exact | cube-single | `exp/kz_conf/cs_s1/...seed_1` | 500k | 1 |
| exact | cube-double | `exp/kz_conf/cd_s1/...seed_1` | 500k | 1 |
| exact | scene | `exp/kz_conf/scene_s0/...seed_0` | 500k | 0 |
| exact | puzzle-3x3 | `exp/kz_conf/puzzle_s0/...seed_0` | 500k | 0 |

HF = `amsks/factored-fb-checkpoints`, pulled to the session scratchpad (`zcov_ckpt/`).
ExoRL (walker/cheetah): SKIPPED. The only Density-FB ExoRL runs with the reward-sample index
(`scripts/launch_exorl_paper.sh`, `density_fb` arm) were saved to the cluster's
`$SCRATCH/factored-fb-exorl-paper`; no `density_fb` walker/cheetah checkpoint exists locally
under `exp/exorl-*` (only rldp/hilp) or on the HF repo.

## How each z is built
- goal z (a): `critic.sample_cond` on real `GCDataset` batches of 1024 (actor goals =
  same-trajectory futures, `actor_p_trajgoal=1`), with `z_rand_frac` forced to 0 through
  `dataclasses.replace` so every row keeps the encoded index; `cond["z"]` read back. This is the
  training path: kernel reward `_delta_phi(goal, next-state queries)` -> centered ridge solve ->
  sqrt(d) normalization. 4 batches -> 4096.
- sphere z (b): `_sphere(normal)` draws, 4096. (The other half of the campaign training mix.)
- eval z (c): `OGBenchAdapter.relabel_infer_batch(train_ds, env, tid, 10000, shift_reward=1.0)`
  -> `critic.infer_cond`, i.e. the `impls/main.py` eval_relabel block verbatim. RNG note: main.py
  draws the 10k rows from the global NumPy stream at whatever state training left it in; here
  `np.random.seed(seed)` is set right before the draws, so the rows are a different sample of
  the same distribution.
- k-hot z (d): one 10k next-state pool drawn like Route B; goal = a random dataset state;
  subset = a random non-empty subset of the hit-spec blocks; r = 1 if every block in the subset
  is within its eps of the goal's block coords, else 0 (resample the goal until >= 5 positives).
  cube-single has one block, so its five draws differ only in the goal.
- dense z (d): same pool; r = -clip(d_eps / median_pool(d_eps), 0, 1), with d_eps the
  eps-scaled Euclidean distance of all block coords to a random goal state (bounded in [-1, 0],
  dense on the nearer half of the pool).
  Both go through `critic.z_from_reward_samples` on `critic._b_query(pool)`.

## coverage.csv columns
- `d_goal_rms`: distance from the z to its nearest goal z, in units of the goal cloud's RMS
  radius R = sqrt(mean ||z_g - mean||^2) (R ~ 7.05 on every env: the cloud is spread over
  20-35 effective dimensions of the sqrt(50) sphere, centered near the origin).
- `cos_goal`: cosine to that nearest goal z. `d_sphere_rms`, `cos_sphere`: the same to the
  sphere cloud.
- `inside_goal_region`: 1 if `d_goal_rms` < the goal cloud's median leave-one-out
  nearest-neighbour distance (`goal_nn_median_rms`). Strict in 50-d; on puzzle the goal cloud's
  median NN distance is 0 (discrete button goals give duplicate z's), so it is degenerate there.
- `centroid_pct`: percentile of the z's distance-to-goal-centroid within the goal cloud's own
  distribution. `knn10_pct`: percentile of its mean distance to the 10 nearest goal z's within
  the goal cloud's own leave-one-out values (100 = farther out than every goal z). The figure's
  strip row plots this 10-NN distance against the goal band and the random-sphere band.
- `z_norm` (sqrt(50) = 7.07 always), `is_neutral` (constant-reward fallback fired; never did),
  `meta` (reward-sample stats).

## Figure
`fig_z_coverage`: one panel per env, 2-D PCA fitted on the env's goal z's (explained variance in
the corner: 11-18%, which is why the plane alone cannot show the separation). Goal region =
kernel density of the 4096 goal z's at the levels enclosing 50% and 90% of the samples
(Figure-1 orange-red family); sphere region = the same for the sphere draws, drawn beneath with
a dashed 90% outline (its projection sits inside the goal blob in every PCA plane). Eval z's =
dark markers with a thin convex hull; k-hot = blue triangles; dense = violet hollow diamonds.
Strip under each panel = the 50-d 10-NN distance readout. Palette CVD check (Python stand-in
for the dataviz validator, node absent): all adjacent pairs >= 8 OKLab dE under deuteranopia and
protanopia except goal-fill vs dense-marker under protanopia (6.3), covered by marker shape.

Run (tmux `zcov`, GPU 5):
```
CUDA_VISIBLE_DEVICES=5 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl WANDB_MODE=offline \
  /dev/shm/.venv-jax/bin/python -m analysis.z_coverage.compute_z_coverage --set campaign
... --set exact
/dev/shm/.venv-jax/bin/python -m analysis.z_coverage.plot_z_coverage [--set exact]
```
