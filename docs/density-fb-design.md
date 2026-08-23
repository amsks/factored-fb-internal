# density_fb design notes

Long-form rationale for `impls/critics/density_fb.py`. The file's own docstring keeps
only the contract; everything that is background rather than contract lives here.

Reference implementation: `tools/wandb_mode_shim/gciql_flow.py` on
`origin/gciql-fb-bridge` @ `542c058` — `density_loss` (~1975), the density branch of
`actor_loss` (~1089), the tower registration in `create` (~3147), the target soft-update
in `update` (~2587). Ported per `docs/specs/2026-08-10-measure-objectives-on-mw-play.md`
P1; the reward-sample index per spec 30 P1. Pixel path:
`docs/specs/2026-08-21-density-fb-pixels-drq.md`. Narrative writeup:
`PAPER/Multistep-FB/DENSITY-FB-EXPLAINED.md`.

## Provenance of the objective

One-stage, from-scratch, two-tower softmax density over a query support `U` resampled
every update from the batch's own next-state object coordinates — `U` is literally the
batch's `next_observations` block columns. There is no prototype file and no fixed
support. Cube-single 0.929 on the bridge.

No orthonormalization, no mass anchor, no scale normalization on this path. Each was
measured separately harmful on the bridge; orthonormalization took cube-single from
0.967 to 0.225.

## The bootstrap goal

`F(s, a, g)` is g-indexed, so the TD target must bootstrap at that same `g`. The bridge
draws it as `actor(nobs, batch['value_goals'], noise)` (`gciql_flow.py` ~2030), and every
other bootstrap in that file is at `vgoals` too.

`boot["next_action"]` is conditioned on the ACTOR goal via `_cvec(cond)`, which is a
different goal distribution (`p_trajgoal=1.0` against the 0.2/0.5/0.3 value mix). Reading
it makes `w_bar` the occupancy of a policy heading to a different goal, i.e. a
goal-averaged behaviour-like density: measured `density_ess_median` ~310 against the
bridge's healthy ~89 at M=B=1024.

The counterpart for forward maps with no goal leg: `td_fb`'s forward map is `phi(s, a)`,
one vector per row serving every column, with readout `<phi(s,a), psi(actor_goal)>`, so
it reads `next_action` instead. The measured triple on the cube-single phase-1 anchor is
0.920 actor-goal against 0.076 value-goal.

## The reward-sample index (`measure_density_z`)

Shim functions: `_density_z_moments`, `_density_z_centered_coef`,
`_density_z_centered_index`, `_density_z_centered_moment`,
`density_z_from_reward_samples` (~1176-1341), plus `_sphere_or_neutral` (~34-50).

Only the shipped spec-30 centered dual encoder is ported. The spec-24 sphere index, the
spec-29 uncentered dual, the analytic point/Dirac prompt and every orthogonality term are
deliberately absent. The point prompt is what made spec 29's 0.812 uninterpretable: it
let a goal shortcut read as general-encoder evidence.

One pinned base measure `rho_phi` = the next-state query marginal, used for `mu_hat`,
`C_hat`, `rbar` and `m_c` consistently. Reward samples are expected from `rho_phi`; there
is no arbitrary-proposal importance correction.

### Invariances

Population exact; the plug-in moment estimate approximates them at finite M.

* `B -> B + b`: `m_c` is literally independent of `mu` because `sum_j (r_j - rbar) = 0`,
  and `C_hat` is a covariance, so both are unchanged.
* `r -> r + beta`: exact, `r - rbar` is unchanged.
* `r -> alpha r` for `alpha > 0`: exact after the exact `sqrt(d)` normalization.

The FULL linear gauge `B -> A B`, `F -> A^-T F` is NOT fixed: an isotropic ridge and a
sphere normalization are not invariant under a general `A`. The claim is scoped to the
additive gauge (spec 30 P1, Terminology, binding).

There is no one-sample / Dirac limit: `m_c` is identically zero at M=1, so M=1 raises
rather than silently returning a zero index.

### Why both centerings are kept

Because `mu_hat` is the mean of `B` over the same pool the reward samples live on,
`sum_j (B(u_j) - mu_hat)` is exactly zero (measured 9.6e-14). The reward-centering
`(r - rbar)` and the B-centering `(B - mu)` therefore each kill both gauges on their own
and mask each other: deleting either leaves every end-to-end invariance test green
(verified by mutation). Both are kept because both are in the ported formula, and because
the redundancy disappears the moment the moments stop being the plug-in ones — which is
exactly what a future EMA port would do. The discriminating tests are the formula-level
ones in `tests/impls/test_density_fb_reward_index.py` section 3b; do not read the
end-to-end invariance tests as evidence about which centering is present.

### Persistent state

Batch statistics are forbidden by the contract: they reintroduce the train/eval mismatch
that broke the first z build (8/8 unit tests green while the eval path was wrong, because
no test drove the eval entry). So three objects persist across updates and serialize with
the checkpoint.

* `density_b_zenc`, the index tower. A slow-EMA copy of `density_b`, initialized as a
  copy of it (bridge ~5729-5733) and mixed at `measure_density_z_enc_tau` inside the
  update (~3849-3859). It is neither the online tower nor `target_density_b`: the shipped
  centered arm asserts `measure_density_z_enc_tau > 0` (~4837-4841), which selects
  `density_b_zenc` over `target_density_b` in `_density_z_enc_module` (~1125-1137),
  because running the solve through the fast-drifting target tower is the known
  non-stationarity risk.
* `density_zstats.mu` / `.cov`, i.e. `mu_hat` and `C_hat` (bridge `ZCenteredMomentsState`
  ~189), EMA'd at `measure_density_z_sigma_tau` over each batch's query set at the
  pre-step index tower (~3903-3921). Init `mu = 0`, `C = I`, so the first steps solve
  `(I + lam I)^-1 m_c ~ m_c` and the EMA walks onto the real moments with no cold-start
  blow-up. The ridge solve is only a stable, comparable object if `C_hat` is
  quasi-stationary; a per-batch d x d covariance is a noisy, rank-deficient estimate that
  would make the index jitter step to step.
* `density_zstats.pool` / `.filled`, the frozen M-row eval pool (bridge `ZEvalPoolState`
  ~218), a fixed draw of raw hit-spec block coords from `rho_phi` written once on the
  first update (~3921-3939). It is what lets a native-goal eval go through the
  reward-sample API (M = `measure_density_z_eval_pool_m` = 256) instead of silently
  substituting a point prompt.

All three live in `net` rather than `target`, because `net` is the only container every
consumer can reach — `sample_cond` and `infer_cond`/`map_eval_cond` are handed `net`, and
`q_fn` sees them through its `params` argument — and because `net` is what `save_agent`
serializes. They are written by the additive `update_stats` hook in
`impls/agents/__init__.py`, a no-op for every critic that does not define it.

Structural difference from the shim, semantically neutral: the shim keeps `mu`/`C`
(`ZCenteredMomentsState`) and the pool (`ZEvalPoolState`) as two registered modules; here
they are four leaves of one `density_zstats` group. Nothing reads the param tree layout,
and the checkpoints were never interchangeable.

### Ridge conditioning

`C_hat` is symmetrized before the solve and its spectrum is floored: eigenvalues below
`z_cov_eig_floor * tr(C)/d` are lifted to that floor. The centered covariance of a d=50
tower estimated from finite batches genuinely has near-zero directions (bridge
measurement: effective rank 6-21 of 50), and an unfloored inverse would amplify whatever
numerical noise sits in them.

`lam = z_ridge_lambda_mult * tr(C_hat)/d` is the mean-eigenvalue convention, computed
from the unfloored trace so the number is the one the bridge's preflight probes scored.
It is dimensionless, hence comparable across checkpoints and towers.

## `actor_q_center`

The bridge's actor centers the Q-loss normalizer, `sg[mean|q - mean(q)|]` instead of
`sg[mean|q|]` (`gciql_flow.py` ~1168). Our `FlowBCActor` has no such knob, so per spec §2
it is implemented inside `q_fn`: only the second (normalizer) return is centered.
Centering the first would make `-Q.mean()` the batch mean of a batch-centered quantity —
identically ~0, no greedy signal, a pure-BC actor. Whether this composition reproduces
the bridge's numbers is what the cube anchor (P5) gates.

## Pixel path

Two encoders, one per tower, mirroring FB's `_fw_encoder` / `_bw_encoder`
(`agents/fb/model.py`) and TD-JEPA upstream. They are not a shared trunk: FB's optimizer
split puts `_bw_encoder` on the backward optimizer and `_fw_encoder` on the forward one.
Here the two towers are the two seam groups, `density_b` and `density_f`, so declaring
one encoder inside each reproduces that split exactly, with no new group type and no new
lr key.

The actor owns neither encoder. It reads features through `cond` (published
stop-gradient'd by `sample_cond`) and takes its Q-term through `q_fn`, which already
stop-gradients the critic params — the seam's analogue of FB's `with torch.no_grad():`
around the actor's encode.

`_PixelDensityF` uses one encoder for both the observation and the goal leg, not two:
this is `_fw_encoder`, the F side's single view of the pixel world, and the goal is a
state of that same world.

### The silent-failure guard

The measure is defined by column slices. `_raw_block_cols` takes `obs_rows[:, 19:22]` and
`_delta_phi` slices both arguments the same way. Hand either one a `(B, 64, 64, 3)` image
and jax slices the last axis without complaint: `[:, 19:22]` of a rank-4 array slices the
HEIGHT axis and returns a `(B, 3, 64, 3)` tensor that broadcasts its way through the
softmax; a flattened `(B, 12288)` image returns three arbitrary pixel intensities. Either
way the kernel returns numbers, training proceeds against a measure computed from pixel
values, and nothing downstream — not a shape error, not a nan, not a loss spike — can
tell you.

So every array entering the kernel is checked at the boundary on rank AND on exact width.
Exact width is what catches the flattened-image case, which a rank check alone lets
through. Shapes are static under jax tracing, so the checks run once at trace time and add
nothing to the compiled graph; the state path stays byte-identical
(`tests/impls/test_boot_contract.py`).

On pixels, each of `u` (the query support) and `g` (the goal) exists twice: as an image,
which the towers encode, and as a privileged object-coordinate row, which the hit kernel
slices. The image channel keeps the existing key names (`_density_u_sc`, `cond["goal"]`,
`cond["value_goal"]`); the qpos channel rides alongside under `_density_u_raw`,
`cond["goal_qpos"]` and `cond["value_goal_qpos"]`. `qpos_dim` is the exact width of that
channel and is what `_check_rows` enforces at every kernel boundary.

The eval loop hands `map_eval_cond` `info['goal']`, which on a `visual-*` OGBench env is
the goal IMAGE — verified on this box. The pixel spec's section 2 claim that
`info['goal']` is "the goal in STATE form" is false for visual envs: both `goal` and
`goal_rendered` come back `(64, 64, 3)` uint8, because the env's `compute_observation`
returns pixels. So the image channel is `cond["goal"]` and the object-coordinate channel
must be supplied separately as `cond["goal_qpos"]`, which
`impls/utils/evaluation.py` builds from the env's task info.

`measure_density_z` on pixels is refused rather than faked: the frozen eval pool stores
raw object-coordinate columns and re-scales them for the B tower, but on pixels the B
tower consumes images. Making the pool store images instead is a design change (~256
frames carried in the checkpoint), not a wiring fix — pixel spec section 10.

## `map_eval_cond` precedence

`Agent.sample_actions` merges `goals` into `cond` whenever it is given, so a goal always
wins over an index previously inferred by `infer_cond`. That is the bridge's split too
(`sample_actions(obs, goal)` maps the goal, `sample_actions_with_z` rolls a precomputed
z), and `FBCritic` behaves the same way here. Consequence: `impls/main.py`'s eval loop
always passes a goal, so `--infer_cond_episodes` sets a `cond_eval` that the very next
`sample_actions` overwrites. Roll a reward-inferred index by calling `sample_actions` with
`goals=None`.

## `infer_cond` caller obligations

* WHICH ROWS. The pinned base measure `rho_phi` is the next-state query marginal. This is
  declared as `INFER_COND_MARGINAL = "next_observations"` and `impls/main.py` slices the
  batch by that declaration, so the caller no longer picks. It used to pass
  `rb["observations"]`, i.e. the current states: on a play dataset the two marginals are
  close, but they are not the same object, and nothing downstream complains.
* WHICH REWARDS. `rewards` must be the task reward on those rows. `GCDataset`'s reward
  column is the goal-reaching reward, which is near-constant on most samples; a
  near-constant reward legitimately carries no task information and the encoder returns
  the neutral `1_d` index. That is designed behaviour, not a failure — but it is also not
  a zero-shot result, and `impls/main.py::infer_cond_noop_reason` will not flag it,
  because `cond` did change.

## `_sphere` vs `_sphere_or_neutral`

`_sphere`'s `+1e-6` denominator guard makes it only approximately scale-invariant: under
`x -> alpha x` the returned norm is `sqrt(d)(1 - 1e-6/(alpha||x||))`, so a small alpha
visibly moves the result (bridge measurement: a 100x downscale moved z by 5e-3). The
centered encoder's whole claim is exact invariance to `r -> alpha r` and to the additive
gauge, so it normalizes exactly and handles `||x|| ~ 0` explicitly — which it must do
anyway, since a near-constant reward is a real input, not a numerical accident.
`_sphere` is kept for the task-prior sphere draws, where the argument is a fresh gaussian
and the guard is irrelevant.

## Self-containment

The tower primitives (`_MLP`, `_ensemblize`) are copied from the vendored OGBench
`impls/utils/networks.py` the bridge itself imports `MLP`/`ensemblize` from, rather than
imported, so this file's isolation from other critics is structural and not just a
registry line (spec §2). The only symbol shared with `td_fb.py` is
`impls.critics._measure_common.parse_hit_spec`.

## Edge-attribution knobs (`measure_density_bootstrap`, `measure_density_actor_temp`)

W1 `bootstrap`: `'actor'` (default, bit-identical) bootstraps the TD target at the actor's
action, so the actor sits inside the critic's fixed point — mechanism M1, the confirmed
collapse cause. `'sarsa'` bootstraps at the dataset's next action instead, giving GCIQL's
in-sample-backup property while keeping the kernel readout, which is what separates M1
from M2 and makes the arm a discriminator rather than another mitigation.

W2 `actor_temp`: temperature for the actor readout only (`q_fn`'s softmax). Raising it
flattens the softmax the actor ascends (M2: a spiky readout is the residual suspect for
the precision gap, ESS ~13 under qc03) without touching what the measure is trained to be
— `stage_loss` always keeps `temp`. A knob that moved both would confound readout
smoothness with the measure being trained.
