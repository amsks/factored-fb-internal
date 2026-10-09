# Spec 17b zero-shot partition readout — implementation report

Files created:
- `scripts/rescuing_fb/eval_partition_zeroshot.py` — the harness.
- `tests/test_partition_zeroshot.py` — 13 tests, all green under `.venv`.

No files edited (per hard rules): `tools/wandb_mode_shim/gciql_flow.py` and
`scripts/rescuing_fb/eval_zeroshot_rewards.py` were only Read, never Edited.

## CLI

Same shape as `eval_zeroshot_rewards.py` / `run_spec14_jax_table.sh`:
`--run-dir --checkpoint-step --env --tasks --families --regions
--region-min-pos --n-candidates --episodes --horizon --relabel-size --seed
--protos-path --expected-protos-sha256 --out`. `--families` defaults to
`dirac,dense,region` (`FAMILIES` constant) — the same restricted set
`eval_zeroshot_rewards.py` uses off cube-single. `mixture`/`reachavoid` were
left out even though they're trivially expressible via
`reward_conj_dirac`/`reward_conj_dense` composition (max / difference), to
keep the (unvalidated) rollout's scope small; `linear` is out of scope
entirely (needs a psi feature-space z-manifold with no analogue in a K-simplex).

## Readout implementation

`compute_Qr(r_x0, p, phi_x0, z_r, gamma)` implements spec 17b sec.3 exactly:
`F^Delta = p - phi(x0)`, `Q_r = (r(x0) + F^Delta·z_r) / (1-gamma)`. A second
function `compute_Qr_from_dots(r_x0, p_dot_zr, phi_dot_zr, gamma)` is the
algebraic shortcut the rollout loop actually uses (it already has the dot
products from candidate scoring); a test proves the two are exactly
equivalent on 20 random draws.

`z_r,k = r(proto_k)` — reward evaluated **directly** at each prototype, no
relabel-sample z-inference (unlike the bilinear harness). `phi(x)` is computed
by importing `GCIQLFlowAgent._delta_phi` from `gciql_flow` at call time — never
reimplemented. Reward families reuse `reward_conj_dirac` / `reward_conj_dense`
and `region_success_from_obs` / `eval_tasks_for` imported straight from
`eval_zeroshot_rewards.py`; `episode_bounds` / `geometry_for` / `hindsight_rows`
from `env_geometry.py`. The only new helper is
`draw_region_centers_with_obs`, a ~15-line fork of
`eval_zeroshot_rewards.draw_region_centers` that additionally returns the
source dataset **observation** (not just the coordinate) at the hindsight
row, because 17b's actor/delta_head need a full goal observation to condition
on and the existing helper discards the row index.

Output JSON mirrors `eval_zeroshot_rewards.py`'s multi-env structure
(`families["dirac"]["t1"]["success"]`, `families["region"]["c0"]...`,
`"mean"` / `"mean_final"`) so `spec14_jax_table.py` can be pointed at this
script's `--out` directory too. `meta.readout` records the Q_r formula,
`meta.prototypes` records the verified path+digest, `meta.gamma`/`hit_spec`
come from the checkpoint's own config, never CLI defaults.

## Prototype verification

`verify_protos_file(path, expected_sha256)`: computes a streaming sha256,
prints it unconditionally, raises `FileNotFoundError` (loud, no silent
regeneration) if missing, and — when a digest is known, either via
`--expected-protos-sha256` or the built-in `CANONICAL_PROTOS_SHA256` registry
keyed by repo-relative path — raises `AssertionError` on mismatch.
`results/delta_protos/cube_single_K256_s0.npy` is registered with
`b4e13e9d9e0a0b0b18deedf5fdce66c1d3f9a17e02175608dd1679064494cbf5`, verified
by hand against the real file (`sha256sum`) before pinning it — matches. In
`main()`, after `load_agent_dims` restores the agent, the script additionally
asserts `agent.delta_protos` (baked in at `create()` from the same
`measure_delta_protos` path) equals the independently-loaded array — catches
a path-resolution divergence between this script and the agent.

## Rollout (Q_r-greedy) — the unvalidated part

The trained flow actor is only ever conditioned on real goal observations
(`actor_cond='goal'` is required by the 17b lattice). Deployment therefore
picks one real observation `g_hat` per episode — the task's own `goal_obs`
for dirac/dense, the hindsight-drawn dataset row for region — and at each
step draws `--n-candidates` noise-sampled actions from the actor at `g_hat`,
scores each via `p(s,a,g_hat)·z_r` (min over the 2 delta_head heads,
matching `actor_loss`'s own reduction, since `r(x0)` and `phi(x0)·z_r` are
candidate-independent at a fixed step), and executes the argmax. This
mechanism is **not** specified by spec 17/17b — the spec explicitly deferred
"the zero-shot readout script" as post-wave — so it has no golden reference.
It is documented prominently in the module docstring as a new design choice.

## Tests (`.venv/bin/python -m pytest tests/test_partition_zeroshot.py -q`)

13 passed. Covers: (a) `compute_Qr` vs a hand-derived numpy reference
(K=3, worked example) and a batched version; (b) the degenerate
constant-reward pin (`Q_r == c/(1-gamma)` for K in {2,3,8,256}, several `c`,
plus a non-simplex counter-example proving the pin is discriminating, not
vacuous); (c) `compute_Qr_from_dots` ≡ `compute_Qr` on 20 random draws;
(d) the real canonical prototype file matches the pinned digest;
(e) a permuted copy is detected (different digest, refused when checked
against the canonical digest, reported as `matched=None` — not silently
"OK" — when no digest is known); (f) missing-file → loud `FileNotFoundError`;
(g) `draw_region_centers_with_obs` hindsight/min_pos semantics + empty-case
shapes; (h) `--help` runs jax-free and the parser's own defaults.
Full-suite regression check: `test_eval_zeroshot_rewards.py` (21),
`test_spec14_jax_port.py` (6), `test_env_geometry.py` (15) all still green
alongside the new 13 — no interference.

## What remains unverified until a checkpoint exists

No 17b checkpoint exists yet (wave at ~30k/500k steps; first save at 100k).
Unverified end-to-end:
1. `load_agent_dims` actually restoring a real 17b checkpoint through this
   script's path (only exercised structurally, via the existing
   `test_gciql_measure_delta_partition.py` suite, not by this script).
2. Whether the Q_r-greedy rollout produces sane behavior at all — no
   baseline/anchor to compare against; `--n-candidates` was not tuned.
3. Whether `agent.delta_protos` (nonpytree, tuple-of-tuples) round-trips
   through a REAL restore identically to the independently-loaded array —
   the cross-check assertion exists in `main()` but has never fired against
   real weights.
4. The `qr_pred_t0` diagnostic (readout's own Q_r estimate at the executed
   t=0 action) has never been compared against an actual achieved
   discounted return.

## Addendum — two decode modes (coordinator follow-up)

The spec-14 numbers this script sits beside were produced with
`--z-inference retrieval --retrieval-k 1` (`run_spec14_jax_table.sh`), a
retrieval decode, not the Q_r-greedy candidate rollout originally shipped.
Added `--decode {retrieval,qr_greedy}` (`DECODES` constant), **default
`retrieval`**; the original mechanism is preserved as `qr_greedy`.

- **`retrieval`** (default): mirrors `eval_zeroshot_rewards.py`'s
  `actor_cond='goal'` deployment exactly — decode ONE goal-proxy observation
  `g_hat` per family/task from a fixed dataset candidate pool (`S_obs`/
  `S_coords`, built with the identical `np.random.default_rng(args.seed)`
  pool/rng convention as the harness's own relabel sample `S`), once,
  state-independently, then roll out the goal-conditioned flow actor at that
  single `g_hat` for the whole episode (`batched_goal_rollout`, no per-step
  re-scoring — the same shape as `batched_rollout_record`). Candidates are
  scored by this readout's own Q_r algebra, not raw reward, per the
  coordinator's instruction: `decode_goal_retrieval` computes
  `p(x0, a_j, g_j).z_r` (min over heads) for every candidate `g_j`, then
  `select_retrieval_candidate` (jax-free, unit-tested) applies
  `--retrieval-k` (k=1: top-1 directly; k>1: nearest dataset row to the
  top-k's coordinate centroid — the discrete analogue of the harness's
  average-then-nearest-neighbor decode).
  **Action fed for scoring**: `a_j = actor(x0, g_j)`, temperature=0, one
  shared rng key across all candidates in a single batched call — the
  trained policy's own one-shot proposal at the canonical start state
  `x0` (a seed-0 reset of the family/task's own env). No dataset behavior
  action exists at a fresh `x0`, and using the actor's own proposal keeps
  the score consistent with "what deployment at that goal would actually
  do."
- **`qr_greedy`**: unchanged — `batched_qr_rollout`, re-scoring
  `--n-candidates` action proposals by Q_r every step at the "natural" goal
  (task `goal_obs` / region hindsight center).

`main()`'s new `rollout_for` closure is the single dispatch point (`if
args.decode == "qr_greedy": ... else: assert args.decode == "retrieval"`);
`--decode` is `argparse choices=DECODES`-restricted so an unknown value is a
hard parse error, never a silent fallthrough. `meta.decode`, `meta.
retrieval_k`, `meta.n_candidates`, and `meta.readout.policy`/`.decode` record
the mode; every family cell additionally carries its own `"decode"` info
dict (selection scores, x0/action provenance for retrieval; a plain note for
qr_greedy).

**New tests** (`tests/test_partition_zeroshot.py`, now 22 total, all green):
`select_retrieval_candidate` top-1, top-k-nearest-centroid (constructed so
the centroid-nearest point is NOT one of the top-k by score, pinning the
rule), k-clamping, shape-mismatch guard; `--decode` choices rejects an
unknown value (subprocess); `DECODES`/parser-default checks; a direct
object-identity + source-diff check that `decode_goal_retrieval`,
`batched_goal_rollout`, `batched_qr_rollout` are three distinct callables;
a source-inspection test of `rollout_for`'s two branches proving the
qr_greedy branch calls only `batched_qr_rollout` (never the retrieval
functions) and the retrieval branch calls only
`decode_goal_retrieval`+`batched_goal_rollout` (never `batched_qr_rollout`);
a static check that `meta.decode` and per-cell `decode` are wired into the
JSON. Full regression sweep (`test_eval_zeroshot_rewards.py` 21,
`test_spec14_jax_port.py` 6, `test_env_geometry.py` 15,
`test_gciql_measure_delta_partition.py` jax suite) still green.

**Still unverified** (unchanged reason: no checkpoint): whether the
retrieval decode's Q_r-based candidate scoring actually picks a sensible
goal in practice, and whether `--retrieval-k 1` on a real checkpoint
reproduces the qualitative behavior the spec-14 numbers show for the
bilinear cells.
