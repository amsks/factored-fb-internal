# Spec 18 (`measure_density`) — correctness-gate review

Reviewed scope: spec §4 (agent code in `tools/wandb_mode_shim/gciql_flow.py`), §5
(`tests/test_gciql_measure_density.py`), and the two `configs/gciql/cube_single_state_measure_density*_d50.yaml`.
Read-only; nothing edited, committed or launched.

**Verdict: no functional defect found. The implementation matches §1/§3/§4, the
anti-scope of §8 is respected, and the off path is bit-identical to HEAD.
7 lower-severity items below — 2 are guard gaps worth closing before follow-on
cells, the rest are gate/test/provenance notes.**

---

## What was verified independently (not by trusting the shipped tests)

All numbers below come from my own scripts run CPU-only under `/dev/shm/.venv-jax`
(`JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false`), not from the task agent's suite.

**§1 object / estimator (priority 1)** — on a 2-block hit-spec `19:22:0.4,28:31:0.9`:
- `k` from `gciql_flow.py:1914` reproduces the spec's `softmax_j(−½Σ_b‖(x′_b−u_b)/ε_b‖²)`
  to 1.2e-7 (analytic numpy, written from §1, not from the impl).
- CE at `gciql_flow.py:1916` equals an independent reference that applies the
  log-normalizer with an explicit `logsumexp` over the **query** axis j — 6.7107972 vs
  6.7107978. The batch-axis alternative gives 7.7310, i.e. the axis check is live and
  the shipped code is on the right axis.
- rows of `t` sum to 1 (both hard and soft immediate).
- Substituting the *online* towers for the target towers changes the CE ⇒ the bootstrap
  genuinely reads `target_density_f/target_density_b` at `(s′,a⁺,g)` (1907-1909) while the
  online term reads `(s,a,g)` (1898-1900).
- `density_ess` == `mean_{h,i} 1/Σ_j w²` on the query axis; `density_w_entropy` == mean row
  entropy. Both correct (1918-1919).

**§4.2 ε-scaling trap (priority 2)** — `_delta_phi` receives RAW rows
(`_raw_block_cols`, 1372-1379) at 1914 and 1059; the B tower and both target-B calls receive
ε-scaled rows (`_density_queries`, 1360-1369) at 1897, 1906, 1060 and at registration
`ex_b` (3336). No path double-scales, none forgets to scale. A deliberately
double-scaled variant produces a materially different `k`, so the trap is not vacuous.

**Grad hygiene (priority 3)** — independent `jax.grad`, exact zeros:
`actor_loss` → 0.0 on `modules_density_f/_b`, both targets, `modules_critic`, `modules_value`;
> 0 on `modules_actor`. `density_loss` → exactly 0.0 on `modules_actor` and both targets;
> 0 on `density_f` and `density_b`; 0 on critic/value. `total_loss` → > 0 on
`density_f`/`density_b`/`actor`, 0 on critic/value (measure_only preserved).

**Default-off bit-identity (priority 4)** — strongest available check: I extracted
`HEAD:tools/wandb_mode_shim/gciql_flow.py` into a parallel shim dir and compared
working-tree vs HEAD on three off-path configs (stock GCIQL default; 17b
`measure_delta_partition` + K256 protos; plain `measure_only` + `measure_coef=1.0`).
`total_loss` value, **every** info key, param-tree leaf count, a post-`update()` weighted
param hash and the post-update rng state are byte-for-byte identical in all three.
17a/17b behaviour is therefore unperturbed. Mutual exclusion is asserted on all three
sides (2903, 2947, 2957) and all are reachable from `create()`.

**17-trap (a): `measure_coef` gating** — `total_loss:2305` ungates via
`measure_delta_partition or measure_density`; `loss = m_loss + actor_loss`. Confirmed by
gradient: with the shipped `measure_coef=0.0`, `total_loss` still puts > 0 gradient on both
towers. Not silently zeroed.

**17-trap (b): pytree field** — no new agent dataclass field (only `delta_protos:764`
exists). `flax.serialization.to_state_dict` → `from_state_dict` into a *differently seeded*
fresh agent round-trips all four new modules exactly (`density_loss` 4.928012371 both sides).
Pre-existing checkpoints are unaffected because the modules are registered only under the knob.

**Anti-scope §8** — `density_loss` contains a CE and nothing else: no entropy term, no τ
schedule, no learned τ, no extra query draws, no ortho/anchor/hinge/scale_norm. `actor_loss`
uses only the stock 2-head `min`. Clean.

**Test quality (priority 8) — mutation-tested.** I built six mutants of the shipped file and
re-ran the suite against each with `tests.jax_subprocess.SHIM_DIR` repointed:

| mutant | killed by |
|---|---|
| `log_softmax(logits, -1)` → `-2` (the silent-fatal axis error) | `test_ce_matches_numpy_reference` |
| kernel fed ε-scaled rows (double-scale) | `test_ce_matches_numpy_reference` |
| B tower fed RAW rows (missing scale) | `test_ce_matches_numpy_reference` |
| `params=grad_params` on the actor-side `density_b` call | `test_actor_grad_hygiene` |
| `update()` density target branch removed | `test_target_towers_init_and_soft_update` |
| target-init mirror removed | `test_target_towers_init_and_soft_update` |
| `stop_gradient` around `t` removed | **nothing** (equivalent mutant, see L3) |

Full suite re-run by me: `tests/test_gciql_measure_density.py` +
`test_gciql_measure_delta.py` + `test_gciql_measure_delta_partition.py` → **30 passed**.

**Launch-blocker sweep (priority 9)** — `log_softmax` is the numerically stable form;
`log(t+1e-8)` and `w²` denominators are guarded/bounded (`Σ_j w² ≥ 1/M`); `M == B` holds by
construction (queries are the batch's own `next_observations`) and a future M≠B would raise a
shape error at the `k[None] + w_bar` broadcast rather than fail silently; save/restore verified
above; ESS/entropy on the correct axis. Working-set sizes at `batch_size=1024`: `logits`
[2,1024,1024] ≈ 8 MB, `_delta_phi` diffs [1024,1024,3] ≈ 12.6 MB — fine.

---

## Findings

### M1 — lattice does not refuse `value_arch='dueling'` (and `task_prior_critic_mix`)
`tools/wandb_mode_shim/gciql_flow.py:2955-2973` (lattice) vs `:1027` (dueling branch) and `:2315`.
Spec §4.6 (assert-lattice: "refuse anything that changes … the bootstrap policy, or composes
with the region-matrix / 17a / 17b paths").
The `if value_arch == 'dueling'` branch in `actor_loss` sits **before** the `elif measure_density`
branch at `:1050`. A config with `measure_density=True, value_arch='dueling'` passes the whole
lattice, trains the density CE normally, and the actor silently maximizes `A_bil` of a critic
that `measure_only` never trains — a run that looks healthy in every logged key (`density_ce`
falls, ESS behaves) but whose policy is trained against noise. Same shape for
`task_prior_critic_mix=True`, which would add a scalar FP TD on top of the CE-only path.
Not triggered by the shipped wave (both yamls set `value_arch: bilinear`, neither sets
`task_prior_critic_mix`), so **this is not a blocker for the 7-job launch**; it is a gap for the
§9 cube-double/scene follow-ons. 17b has the identical hole.

### M2 — G2 gate reads a statistic the code does not log
`tools/wandb_mode_shim/gciql_flow.py:1918` vs spec §7 gate G2.
`density_ess = jnp.mean(1/Σ_j w²)` is a **mean over rows × heads**. G2 pre-registers
"median `density_ess` ≥ 4 at 500k". If "median" was meant over rows (the natural reading for a
degeneration diagnostic, since a mean hides a bimodal collapse where half the rows are one-hot),
the logged key cannot answer the gate and the arithmetic mean will read healthy through a partial
collapse. Also: the mean is taken across both F heads, so one degenerate head is masked by the
other. Decide the reading **before** collection, not when the number lands.

### L3 — `test_actor_grad_hygiene`'s density-side target/actor assertions are tautological
`tests/test_gciql_measure_density.py:324-328`.
`target_density_f/_b` and `actor` are invoked without `params=grad_params`
(`gciql_flow.py:1905-1908`), so under `jax.grad` they close over concrete arrays and their
gradient is structurally zero — with or without the two `jax.lax.stop_gradient` calls. Confirmed:
deleting the `stop_gradient` around `t` (`:1915`) leaves all ten tests green. The two sg's are
inert belt-and-braces (correct to keep), but those three assertions are **not** evidence that
they work; they are guards against a future `params=grad_params` edit. The actor-side assertions
(`:315-318`) are genuinely live — the `actorgrad` mutant kills them.

### L4 — tautological tail in `test_hard_immediate_ablation`
`tests/test_gciql_measure_density.py:428-433`. `blocks`, `nobs`, `gamma` are computed and unused;
`assert np.allclose(k, k.T) and np.allclose(np.diagonal(k), 1.0)` asserts properties of the
test's own locally-built `np.eye(B)`, never touching the agent. The test's only load-bearing
line is the `density_k_self == 1.0` check at `:426` (which does catch the knob being ignored).

### L5 — `test_default_off_bit_identical` cannot detect an off-path regression
`tests/test_gciql_measure_density.py:122-144`. Both `cfg0` and `cfg1` have the knob off and both
run the *new* code, so the test proves "explicit False == default", not "spec 18 did not change
the off path". Spec §0/§4.1's bit-identity requirement needs a cross-version comparison. I closed
this gap out-of-band against HEAD (see above) and it passes — but the suite alone does not cover it.

### L6 — stale provenance comment in the K-sweep control configs
`configs/gciql/cube_single_state_measure_delta_partition_K{64,512,1024}_d50.yaml`, the
`make_delta_prototypes.py` comment line. Each still reads `--k 256` while pointing `--out` at
`cube_single_K{64,512,1024}_s0.npy`. The launcher (`scripts/rescuing_fb/launch_spec18_cs.sh:57`)
uses `--k $K` correctly, so the runs are right; the comment is wrong. Under the standing
"group names lie, audit by config" rule the config comment is the audit trail, so it should say
what actually generated the file.

### L7 — `density_k_self` baseline: pre-register the number before reading it as coverage
Spec §4.3 calls `density_k_self` a coverage diagnostic and §9 gates "extra iid queries" on it.
On the real `cube-single-play-v0` object marginal (cube xyz, per-dim std 0.71/1.68/0.84) with
ε = 0.4 and M = 1024 batch-drawn queries, the kernel row is intrinsically diffuse:
`k_self ≈ 0.057`, kernel row entropy 4.33 vs log M 6.93, kernel ESS ≈ 63. At M = 256:
`k_self ≈ 0.19`, ESS ≈ 16. So `density_k_self ≈ 0.06` is the **healthy** baseline at the shipped
batch size, not a coverage failure — and it is batch-size dependent, so it is not comparable
across M. Corollary for the hard ablation: soft vs hard is a 63-effective-query target vs a
1-query target, i.e. as much a target-sharpness contrast as a "smoothing" one; §7's
"ε-kernel smoothing is load-bearing" reading should be phrased accordingly.

### L8 — τ = 1.0 starts the softmax fairly peaked at d = 50 (calibration note, not a defect)
`configs/gciql/cube_single_state_measure_density_d50.yaml:45`, spec §3 "rank/temperature FIRST".
`MLP` applies `layer_norm` to hidden layers only (`activate_final=False`), so the towers emit
unnormalized `d=50` vectors: at init ‖F‖ ≈ 9.5, ‖B‖ ≈ 9.9 and the logits have std ≈ 12.9
(range ±60). With τ = 1 and M = 1024 that gives step-0 `density_ess` ≈ 24 and
`density_w_entropy` ≈ 3.43 vs log M = 6.93. Comfortably above G2's floor of 4 and it moves the
right way (ESS ≈ 65, logit_max 23.7 → 8.7 after 20 CPU updates), so **not a launch blocker** —
but it confirms temperature/rank as the first suspects if G1 misses, and it is worth logging
`density_logit_max` early rather than only at 500k.

---

## Out of reviewed scope (flagged, not reviewed)

`scripts/rescuing_fb/eval_density_zeroshot.py`, `tests/test_eval_density_zeroshot.py`,
`scripts/rescuing_fb/launch_spec18_cs.sh`, `scripts/rescuing_fb/measure_region_density.py` and
`tests/test_smb_density.py` are also untracked and about to be committed, but the brief scoped
this review to §4/§5 + the two configs. Gates **G3 (pool sweep)** and **G4 (Spearman ordering)**
are answered entirely by the first two files, so they need their own gate pass before collection.

## Config check (priority 7) — clean

Both new yamls diffed against `cube_single_state_measure_delta_partition_d50.yaml`. Every §7 slot
setting matches (`bc_coeff 3.0`, `train_steps 500000`, `eval_interval 50000`, `save_interval
100000`, `eval_episodes 50`, `eval_on_cpu 1`, `storage: shm`, `wandb_mode online` ⇒ entity
defaults to `amsks` in `run_gciql.build_child_env`). Only intended knobs differ:
`measure_delta_partition`+`measure_delta_protos` → `measure_density`+`_dim 50`+`_temp 1.0`
(+`_hard: true` in the ablation), and the `run_group`. `measure_coef` and `measure_delta_protos`
are absent from both (lattice would refuse them anyway). Run groups
`GCIQL-CubeSingle-Flow-Measure-Density-d50` / `-DensityHard-d50` do not collide with any group
in `scripts/results_map.py` or any existing `results/` directory. Both configs construct an agent
without tripping the lattice and register exactly the four new modules.
