# z coverage: training task vectors vs evaluation task vectors (2026-09-10)

Question (Ben): FB-style methods assume the training distribution p(z) covers the z's that real
rewards produce. Our reward-sample-index Density-FB trains the actor on z's built from hindsight
goals, and under reward inference it reaches 0.96 on cube-single but 0.22 on scene (500k,
exact-reward checkpoints). Hypothesis: scene's reward-inference z lands outside the goal-task
region.

## Setup
- Code: `analysis/z_coverage/compute_z_coverage.py` (GPU forward passes only) and
  `plot_z_coverage.py`. Figures: `PAPER/ICLR/figures/z_coverage/fig_z_coverage{,_exact}.{pdf,svg,png}`.
  Data: `analysis/z_coverage/z_<env>[_exact].npz`, `coverage.csv`. Details in the README there.
- Checkpoints: campaign = 1M `density_fb_z` (latent_centered actor reward, z_rand_frac 0.5,
  seed 0; antmaze seed 2, the only one on the hub) for cube-single, cube-double, scene,
  puzzle-3x3, antmaze-medium-navigate. Exact = the 500k `exp/kz_conf` checkpoints (kernel actor
  reward, z_rand_frac 0: the actor trained on hindsight-goal z only) for cube-single (seed 1),
  cube-double (seed 1), scene, puzzle. ExoRL skipped: no Density-FB walker/cheetah checkpoint
  with the reward-sample index exists locally or on the HF repo (they were saved on the cluster).
- Per env: 4096 hindsight-goal z's from the critic's own `sample_cond` on real batches with
  z_rand_frac forced to 0; 4096 uniform sqrt(d)-sphere z's; the 5 Route-B eval z's
  (10k relabelled samples, shift 1.0, `infer_cond`); 5 k-hot z's (reward 1 when a random subset
  of the hit-spec blocks is within eps of a random dataset state) and 5 dense bounded z's
  (negative normalized block distance to a random state, clipped), both through the same
  reward-sample index.
- Distances are in units of the goal cloud's RMS radius R (about 7.05 on every env; the goal
  cloud spans 20-35 effective dimensions of the sqrt(50) sphere). "10-NN" is the mean distance
  to the 10 nearest goal z's; its percentile is taken within the goal cloud's own leave-one-out
  values, so 100 means farther out than every goal z.

## Coverage table (eval z's, median over the 5 tasks)

| set | env | Route B success | nearest goal z [R] | cos to nearest goal | 10-NN to goal [R] | goal cloud 10-NN 5-95% [R] | random z 10-NN 5-95% [R] | 10-NN percentile | strict inside frac |
|---|---|---|---|---|---|---|---|---|---|
| exact 500k | cube-single | 0.96 | 0.45 | 0.90 | 0.50 | 0.19-0.52 | 1.03-1.18 | 93 | 0/5 |
| exact 500k | cube-double | 0.08 | 0.91 | 0.59 | 0.99 | 0.37-0.75 | 1.02-1.16 | 100 | 0/5 |
| exact 500k | scene | 0.22 | 1.02 | 0.48 | 1.05 | 0.04-0.69 | 1.03-1.17 | 100 | 0/5 |
| exact 500k | puzzle-3x3 | 0.02 | 0.96 | 0.54 | 0.99 | 0.03-0.59 | 1.02-1.16 | 100 | 0/5 |
| campaign 1M | cube-single | 0.68 | 0.41 | 0.92 | 0.47 | 0.20-0.50 | 1.03-1.19 | 93 | 0/5 |
| campaign 1M | cube-double | 0.46 | 0.78 | 0.70 | 0.91 | 0.37-0.75 | 1.02-1.16 | 100 | 0/5 |
| campaign 1M | scene | 0.52 | 1.05 | 0.45 | 1.08 | 0.05-0.63 | 1.03-1.18 | 100 | 0/5 |
| campaign 1M | puzzle-3x3 | 0.04 | 1.02 | 0.48 | 1.06 | 0.03-0.62 | 1.02-1.16 | 100 | 0/5 |
| campaign 1M | antmaze-medium | 0.50 | 0.05 | 1.00 | 0.12 | 0.10-0.43 | 1.04-1.17 | 10 | 5/5 |

Route B success = mean over the 5 tasks at the checkpoint's step (from eval_relabel.csv). The
strict "inside" criterion (nearest-goal distance below the goal cloud's median nearest-neighbour
distance) is very tight in 50-d; on puzzle the goal cloud's median NN distance is 0 because
discrete button goals give duplicate z's. The 10-NN percentile is the usable yardstick. Every
eval z has norm sqrt(50); the constant-reward fallback never fired. The distance to the nearest
sphere z is about 1.0-1.05 R for every family on every env, and the cosine to it about 0.45-0.5,
so the sphere cloud gives no z a close neighbour anywhere.

k-hot and dense families (median 10-NN to goal, R): cube-single 0.41 / 0.86, cube-double
0.74 / 0.86, scene 0.74 / 0.65, puzzle 0.88 / 0.70, antmaze 0.28 / 1.13 (campaign). Single-block
k-hot rewards on cube-single and antmaze land near the goal region; multi-block k-hot and dense
rewards land outside it on every manipulation env.

## Reading
Antmaze's five eval z's sit deep inside the goal-task region (nearest goal z 0.05 R, 10-NN
distance below the 12th percentile of the goal cloud's own), because the relabel reward for a
maze cell is the same kind of reward the hindsight goals produce. Cube-single's eval z's sit at
the edge of the goal-task region (nearest goal z 0.41-0.45 R, 10-NN at the 85-97th percentile of
the goal cloud, far from the random-sphere band at 1.0-1.2 R), so the actor was trained on z's
close to them. Scene's, puzzle's and cube-double's eval z's sit outside the goal-task region:
their 10-NN distance to the goal cloud (0.9-1.15 R) is beyond every goal z's own and equals a
random sphere draw's, so for these rewards the inferred z is as far from any training goal z as a
random z would be. This matches the 500k exact-reward results: cube-single, whose eval z's are
covered, reaches 0.96, while scene (0.22), cube-double (0.08) and puzzle (0.02), whose eval z's
are not covered, fail, and the same ordering holds at 1M. The mechanism is the reward class: the
Route-B reward on scene and puzzle is a graded count over several conditions (shift 1.0 makes it
#satisfied minus k plus 1), which the reward-sample index maps to a direction the single-goal
kernel rewards never produce, and even a k-hot conjunction over a random subset of blocks lands
closer (0.74 R) than the real eval reward (1.05 R). Coverage is necessary rather than sufficient:
antmaze rolls an in-region z and still scores 0 on tasks 3 and 4 under both routes, and scene's
1M campaign actor succeeds on tasks 1-3 with z's just as far out as tasks 4-5, so within an env the
distance does not rank tasks. The sphere half of the campaign training mix is what makes those
out-of-region z's reachable at all: scene rises from 0.22 (goal-only actor, 500k) to 0.52
(goal plus sphere, 1M) with eval z's at the same 1.05 R, although that comparison also changes the
step, the seed and the actor reward, so it is suggestive rather than controlled. The next test is
a training arm whose z prior samples the eval reward class itself (k-hot and count rewards over
the hit-spec blocks, built through the same index) and a check that its eval z's move inside the
trained region and that Route B on scene and puzzle moves with them.
