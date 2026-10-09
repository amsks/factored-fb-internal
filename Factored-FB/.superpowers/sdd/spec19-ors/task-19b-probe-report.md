# Task 19b: transcription note (sec.0.1) + calibration probe (sec.1)

Scope: spec 19b's sec.0.1 (transcription duty) and sec.1 (P1.2 hard-gate
calibration probe) only. Sec.2 (the extractor in
`tools/wandb_mode_shim/gciql_flow.py`) is explicitly out of scope — that file
was read for context/reuse but not edited.

## (a)-(e), one line each

- **(a) reward composition**: r_W **REPLACES** `batch['rewards']` for every
  non-terminal transition, but the exact success-index transition is force-set
  to a fixed `1.0` regardless of r_W (a three-way convention, not pure
  replace/add) — verified from Appendix B.3's TD-target equation AND the
  released `datasets.py`/`gciql.py` source directly (`batch['rewards'] =
  compute_shaped_rewards(...)`, then `jnp.where(successes, 1.0, ...)`).
- **(b) normalization/scale**: NOT unscaled. Appendix B.7 Table 10 has a
  per-task divisor column ("reward scaling for ORS, `r/x`"): cd=0.25,
  scene=0.75 (i.e. 4× and 1.33× multipliers respectively) — no closed-form
  derivation exists; the released code has only a single hardcoded `0.75×`
  with no per-task CLI hook, a real gap between the paper's table and the
  public repo, flagged not resolved.
- **(c) discount γ**: same symbol/definition throughout, and matches
  numerically for our target envs (occupancy γ=0.99, policy γ=0.99 per Tables
  8/10) — but the paper's own antmaze cells actually mismatch (occ 0.99 vs
  policy 0.995), so "same γ" is not a hard invariant of the paper's own
  practice, only true where we need it.
- **(d) optimality condition**: Theorem 1 requires A.1 (deterministic
  dynamics), A.2 (one-step reachability), A.3 (strong-monotonic potential gap
  ΔΦ>0), A.4 (layer-monotone behavior policy) — none audited empirically by
  the paper on its own OGBench data, and A.1 is not literally true for cube
  contact dynamics.
- **(e) numbers**: Table 1 confirms our spec's quoted cd 45±7 / triple 37±8 /
  scene 80±4 vs GCIQL 40/7/51 exactly (5 test-time goals, 8 seeds, 95%
  bootstrapped CI, trained to convergence — cd/scene 1M iterations,
  triple 2M).

Full citations, quotes, and every conflict with spec 19b's stated defaults are
in `PAPER/Multistep-FB/specs/19b-ors-transcription.md`.

## Sources

- Paper: `arXiv:2604.20627v1`, fetched as PDF and extracted to text with
  `pymupdf` (27 pages, read in full — main text, Appendix A proofs, Appendix
  B hyperparameters). OpenReview's own PDF endpoint (`EW8DskWQ1K`) blocked
  automated fetches (bot wall); the arXiv PDF is the same paper.
- Code: `github.com/aravindvenu7/occupancy_reward_shaping`, fetched directly
  with `curl` (raw file bytes, not an LLM-summarized proxy) for `main.py`,
  `agents/gciql.py`, `agents/rew.py`, `utils/datasets.py`, `occupancy.py`,
  `reward_distillation.py`. All code quotes in the transcription note are
  copy-pasted from these fetches, not paraphrased.

## Files created

- `PAPER/Multistep-FB/specs/19b-ors-transcription.md` — the transcription
  note (deliverable 1).
- `scripts/rescuing_fb/ors_calibration_probe.py` — the eval-only P1.2 gate
  probe (deliverable 2).
- `tests/test_ors_calibration_probe.py` — 34 tests.

## The probe: CLI and method

```
taskset -c 40-47 env OMP_NUM_THREADS=4 \
  JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/ors_calibration_probe.py \
  --run-dir '<19a run dir glob>' --checkpoint-step 500000 \
  --n-samples 2000 --n-random-goals 20 --min-remaining 500 --horizon 500 \
  --pool-size 4096
```

Method: samples N=2000 dataset rows (seeded, uniform over the union of rows
with ≥500 same-trajectory steps remaining — spread across trajectories in
proportion to each trajectory's own eligible-row count) × the env's 5 task
goals + 20 random dataset goals. For each (row, goal): `r_W = ors_reward_scale
* (1-γ) * Σ_j mean_h softmax_j(F_D(s,a)_h·B_D(u_j)/τ) * (-d²_eps(u_j,g))` over
a fixed pool drawn once from the dataset, reusing `GCIQLFlowAgent.
_raw_block_cols`/`_density_queries`/`density_f`/`density_b` (the checkpoint's
own code path) and `eval_density_zeroshot.softmax_np` for normalization —
**not** that script's `density_weights` (which does an elementwise MIN over
the two F-heads; spec 19b sec.2.3 calls for a MEAN for r_W since it's a
reward, not a pessimistic value estimate — a different algorithm on the same
towers, not a duplicate). Ground truth is the empirical
`Σ_{k≥1} γ^{k-1}(-d²_eps(x_{i+k}, g))` within the trajectory, truncated at
horizon 500 (always non-binding under the ≥500-remaining sampling filter, so
the renormalization constant is identical across rows for a given env — the
code still implements it generally and a unit test exercises the truncating
branch directly). Distance is the per-block ε-scaled squared distance,
reusing `env_geometry.HitSpec.scaled` (the repo's one hit-spec definition).
Spearman reuses `evals._profile_core._spearman` (the repo's one rank-corr
helper). GATE: state-axis pooled Spearman ≥ 0.65 (`GATE_SPEARMAN`), reported
per-goal too. Action-axis: near-duplicate-state groups (KD-tree radius query
on ε-scaled coords, filtered to groups whose actions differ enough) are
reported with explicit coverage (`n_groups`, `coverage_fraction`) — expected
to be near-empty on continuous manipulation data, and an empty result is
correctly reported as such, not fabricated. `--ors-reward-scale` and `(1-γ)`
are documented as **not** affecting the gate statistic (Spearman is invariant
to positive monotonic rescaling).

Consistency refusal (`check_behavior_density_consistency` /
`OrsCalibrationMismatchError`): refuses, both directions, a checkpoint whose
recorded config is not `measure_density=True AND
measure_density_behavior=True`; `--force` overrides with a loud warning.

## Test summary

`.venv/bin/python -m pytest tests/test_ors_calibration_probe.py -q` →
**34 passed** (0.6s for the 30 pure-numpy tests; ~39s total with the 4
jax-subprocess tests against `/dev/shm/.venv-jax`).

- Pure-numpy (30, no jax/agent/dataset): eps-scaled distance (incl.
  heterogeneous per-block eps and a dim-mismatch refusal), ground truth on a
  **hand-computable synthetic trajectory** (both a genuinely truncated case
  and an episode-end-capped case), the vectorized batch path cross-checked
  against the scalar reference on mixed full/truncated rows, index sampling
  (min-remaining respected, deterministic given seed, spreads across
  trajectories, empty when nothing qualifies), the r_W formula (hand-computed,
  scale/γ linearity), the mean-of-heads weight convention (pinned to differ
  from the min convention), the Spearman gate (pass/fail/degenerate/threshold
  boundary), the action-axis grouping (planted duplicates found; honest empty
  result on well-separated data; near-identical-action duplicates dropped),
  the consistency refusal (all 4 branches), and a CLI-defaults smoke test.
- jax-dependent (4, via `tests.jax_subprocess.run_jax`, skips if no jax venv):
  built a structurally-valid but **untrained** `measure_density_behavior=True`
  agent in-process (same harness `tests/test_gciql_measure_density.py`
  already uses for the sibling spec) and proved (1) `density_f`/`density_b`
  register with the correct no-goal-leg input shape, (2) the probe's
  `compute_f_emb` matches a **direct module call** numpy reference bit-for-bit
  (proves no accidental goal leg), (3) a **full-pipeline repeat run is exactly
  bit-identical** (the spec-18-lesson determinism check — "use a repeat run
  as the noise floor" — with a companion check that a *different* seed does
  NOT reproduce the same output, so the determinism check isn't vacuous), (4)
  the default-off path registers no density modules.

No 19a checkpoint exists yet, so nothing past checkpoint restore (`main`,
`build_ors_pool`/`compute_f_emb` against a *trained* agent) has been run
end-to-end — this mirrors `eval_density_zeroshot.py`'s own precedent
(explicitly UNVALIDATED in its module docstring before spec 18's checkpoint
existed) and is stated plainly in `ors_calibration_probe.py`'s own module
docstring, not hidden.

## Concerns / unverified

- Table 10's per-task reward-scale provenance is not reproducible from the
  public code (only one hardcoded constant, no config flag) — flagged in the
  transcription note, not resolved by guessing.
- Proposition 2's constant `C` (W₂² ≤ C·E[flow MSE]) is never estimated in
  the paper or code; the released code implicitly uses `C=1`.
- `tests/test_gciql_measure_density.py::test_default_off_bit_identical`
  fails — **pre-existing, unrelated to this task**: that test diffs the
  current knob-off agent against `git show HEAD:tools/wandb_mode_shim/
  gciql_flow.py` under the assumption HEAD predates spec 18, but HEAD
  (`74e861d`) already contains `measure_density` (spec 18 is committed), so
  the test's own baseline assumption is stale. `gciql_flow.py` was not
  touched by this task. All 19 tests in `test_eval_density_zeroshot.py` and
  the other 10 tests in `test_gciql_measure_density.py` pass.
- The extractor (spec 19b sec.2, `tools/wandb_mode_shim/gciql_flow.py`) is
  out of scope for this task and was not implemented or edited, per the hard
  rule.
