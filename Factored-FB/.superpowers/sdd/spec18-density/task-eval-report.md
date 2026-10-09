# Spec 18 §6 — density zero-shot eval: task report

## Files
- `scripts/rescuing_fb/eval_density_zeroshot.py` (new)
- `tests/test_eval_density_zeroshot.py` (new, 19 tests, all passing)
- `.superpowers/sdd/spec18-density/task-eval-report.md` (this file)

Neither `eval_partition_zeroshot.py`, `eval_zeroshot_rewards.py`, nor
`tools/wandb_mode_shim/gciql_flow.py` was edited (`git status` confirms only
the parallel agent's changes are in `gciql_flow.py`).

## CLI
`--run-dir --checkpoint-step --env --tasks --families --regions
--region-min-pos --retrieval-k --episodes --horizon --relabel-size
--pool_size (4096) --pool_sweep (256,1024,4096,16384)
--pool_sweep_full_rollout --ridge (1e-4) --seed --force --out`.
γ, `double_q`, `measure_density_temp`, `measure_hit_spec` are all read from
the checkpoint's recorded config, never CLI defaults.

## Both readouts (report separately, never conflated)
- **PRIMARY** `Qhat_r = (r(x0) + (w-kappa(x0|U_P))^T r(U_P)) / (1-gamma)` —
  17b's equation, prototypes → pool. Pure dot product of an r-independent
  vector against `r(U_P)`.
- **SECONDARY** `Qhat_r ~= (r(x0) + (mu-mu0)^T w_r) / (1-gamma)`, `mu = w^T
  B(U~P)`, `w_r` = ridge-fit of `r` on `B(U~P)`.
- `w`: per-F-head softmax over the pool, **elementwise min of the two
  probability vectors** (spec §3 "min over the two w's" — literal, not a min
  of a downstream scalar). This is what keeps `w`/`kappa` reward-independent
  and gives PRIMARY its exact linearity.
- `kappa(x0|U_P) = GCIQLFlowAgent._delta_phi(x0, U_raw, blocks)` — live
  import, raw block rows passed (the spec's called-out double-scaling trap
  avoided).

## Shared with the partition harness (imported, not reimplemented)
`draw_region_centers_with_obs`, `select_retrieval_candidate` from
`eval_partition_zeroshot.py`; `reward_conj_dirac/dense`,
`region_success_from_obs`, `eval_tasks_for`, `print_table` from
`eval_zeroshot_rewards.py`; `geometry_for`, `episode_bounds` from
`env_geometry.py`.

**Scope narrowing, stated in the module docstring:** only the *retrieval*
decode is implemented (task brief said "the retrieval policy decode,"
singular) — no `qr_greedy` analogue.

## Pool sweep (G3 evidence)
Default `--pool_sweep` recomputes the **t=0 PRIMARY Q-estimate** at each pool
size via prefix+renormalize of one shared draw (exact — restricting a
softmax to a leading subset and renormalizing by that subset's own sum is
mathematically identical to computing the softmax over just that subset;
proved and used, not assumed). This is cheap and is the direct "query count
is a Monte-Carlo sample size" evidence. `--pool_sweep_full_rollout` (off by
default) additionally re-runs retrieval decode + a full episode batch at
every swept size, matching G3's literal success-rate-stability language —
expensive, so gated behind its own flag per the "no full eval sweep" rule.

## Linearity investigation (worth surfacing)
I derived and empirically verified that under a straightforward fixed-λ
closed-form ridge, SECONDARY is *also* exactly linear in `r` to ~1e-14 on a
well-conditioned pool — same order as PRIMARY. The spec's "=" vs "≈" split
is real but shows up specifically as **numerical conditioning**, not a
structural difference in the estimator: with an ill-conditioned `B` (near-
collinear columns) and small ridge, SECONDARY's combined-vs-separate
discrepancy grows to ~1e3× PRIMARY's (demonstrated in
`test_secondary_linearity_holds_only_approximately`, not merely asserted).
The test suite documents this honestly rather than asserting a nonlinearity
that isn't structurally there.

## Tests — `.venv/bin/python -m pytest tests/test_eval_density_zeroshot.py -q`
**19 passed.** Covers: numpy reference for both readouts (incl. a degenerate
constant-reward pin, checked separately for double_q on/off since elementwise
min doesn't preserve the simplex); pool-sweep smoke (finite + drift shrinks
P=64→20000 vs an independent 300k-pool reference, seed 7 pinned after
checking it's non-flaky); consistency refusals (both directions of
`check_density_consistency`, mirroring `check_delta_consistency`); exact
PRIMARY linearity (10 random trials) + the SECONDARY conditioning
demonstration above; `--help`/parser defaults; and a regression test for a
real bug I caught and fixed — `density_weights`'s original ndim-based
"does f_emb have a head axis" auto-detection was ambiguous (`[H,d]` with
H=2 heads is indistinguishable by shape from `[N,d]`, N=2 headless queries)
and silently mis-shaped the H=2/no-batch case; fixed to require an explicit
leading head axis always, no guessing.

`tests/test_partition_zeroshot.py` re-run (22 passed) to confirm the
read-only imports from that module didn't disturb it.

## Unverified until a checkpoint lands
Everything past checkpoint restore — `build_pool`, `density_query`,
`decode_goal_retrieval_density`, `batched_goal_rollout_density`, and `main`
itself — is jax-dependent and untested end-to-end; no `measure_density`
checkpoint exists yet. I checked (read-only) that the parallel agent's
landed code in `gciql_flow.py` (`_density_queries`, `density_f`/`density_b`
registration, `[2,B,d]` F-tower shape) matches every convention this script
assumes, so it should wire up once a checkpoint exists, but this is
unconfirmed by an actual run — no fabricated numbers anywhere in this report.
