# Density-FB as a first-class agent against a complete baseline set

**Date:** 2026-08-21
**Status:** approved design, not yet implemented
**Base:** `origin/feat/metaworld-runs` @ `e35a542`
**First branch:** `feat/seam-boot-contract`

---

## 1. Goal

Make Density-FB a method that can be run head-to-head against a complete set of
baselines — GCIQL, bilinear GCIQL, GCIVL, HIQL, FB, RLDP, PSM — on one training
loop, one environment layer, one venv, with the actor held fixed so a win is
attributable to the value representation rather than to the extraction.

Two further axes are in scope for this branch: **ExoRL** (reward-specified DMC
tasks, which is what actually tests a general reward interface) and **online
Density-FB**.

## 2. Where things stand

Three codebases hold pieces of this, and Density-FB exists in a different form in
each.

| codebase | Density-FB form | baselines present |
|---|---|---|
| `Factored-FB` @ `gciql-fb-bridge` | ~60 flags on the 6,207-line research shim `tools/wandb_mode_shim/gciql_flow.py`; **plus** the reward→`z` index from specs 24–30 | torch `agents/{fb,psm,rldp,tdmpc2}`; jax vendored `third_party/ogbench/impls/agents/{gciql,gcivl,hiql,crl,qrl,gcbc,sac}`. Split across two venvs and two entrypoints. |
| `origin/feat/jaxify` | absent | clean critic×actor seam, 8 critics × 4 actors |
| `origin/feat/metaworld-runs` | `impls/critics/density_fb.py` (330 lines), a documented near-verbatim port | 13 critics: `fb, td_fb, density_fb, gciql, isfb, iql_bilinear, centered_bilinear, fb_forward_bilinear, simple_sf, sf_feature, crl, qrl, vip`. 6 actors. `env/` with 7 MetaWorld + 2 ManiSkill + OGBench + box2d domains. |

There is also an untracked torch rewrite on `gciql-fb-bridge` (`agents/dfb/`,
17/17 tests passing). It is **not** used by this branch and is not deleted by it;
it remains that track's artifact.

`feat/metaworld-runs` is the base because it already has the port, the env layer
and 4 of the 8 requested baselines, on one entrypoint.

### 2.1 What is missing

| baseline | status | source for the port |
|---|---|---|
| GCIQL | present (`impls/critics/gciql.py`) | — |
| bilinear GCIQL | present (`impls/critics/iql_bilinear.py`) | — |
| FB | present (`impls/critics/fb.py`, `td_fb.py`) | — |
| GCIVL | **missing** | vendored `third_party/ogbench/impls/agents/gcivl.py` |
| HIQL | **missing** | vendored `third_party/ogbench/impls/agents/hiql.py` |
| PSM | **missing** | torch `agents/psm/` (1,342 lines) on `gciql-fb-bridge` |
| RLDP | **missing** | torch `agents/rldp/` (277 lines) on `gciql-fb-bridge` |

### 2.2 The blocking finding

`impls/critics/density_fb.py:299` is:

```python
def infer_cond(self, net, obs, rewards):
    return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}
```

and `PROVIDES_COND = frozenset({"value_goal", "goal"})` (line 142). **The ported
Density-FB is goal-conditioned only and has no reward interface** — `infer_cond`
ignores `rewards` entirely and returns a zero placeholder. `impls/main.py` will
warn that `--infer_cond_episodes` had no effect.

The reward→`z` index — specs 24→30, ending at the spec-30 centered dual encoder
that scored .816 through the real M=256 reward-sample API — exists **only** in
the shim on `gciql-fb-bridge`. It was never carried to `metaworld-runs`.

Consequence: ExoRL is not a domain-porting task. It requires porting that index
(workstream 5). This is the workstream carrying the standing requirement that a
goal-conditioned-only Density-FB is not an acceptable end state.

## 3. Decisions

| decision | choice |
|---|---|
| host codebase | branch off `origin/feat/metaworld-runs` |
| domains | OGBench cube/scene **and** MetaWorld, full cross-product |
| vendored agents | **transcribe** into the seam, parity-pinned; import nothing from `third_party` |
| PSM / RLDP | transcribe torch→jax in this branch |
| actor policy | every method composes with all 4 actors; Density-FB's actor difference is isolated **in the critic**, not forked into its own actor |
| branch structure | contract change lands alone, then the ports fan out in parallel, campaign last |
| ExoRL | second table, successor-measure family only; GC baselines marked `N/A — no reward interface` |
| online Density-FB | in scope for this branch |
| validation | bit-identity gate + parity gates + cube anchor blocks the campaign |

## 4. Architecture: the seam contract change

`boot` is produced by `actor.sample_bootstrap` (`impls/agents/__init__.py:115`),
but *which action the TD target bootstraps at* is a **critic** property, and the
two answers genuinely differ per critic:

- `density_fb`'s forward tower takes the goal as an input, so its row is
  `g`-indexed and the target must bootstrap at the **value** goal.
- `td_fb`'s forward map is `phi(s,a)` with no goal leg, so it bootstraps at the
  **actor** goal. `impls/critics/td_fb.py::stage_loss` records the measured
  triple: **0.920 actor-goal vs 0.076 value-goal** on the cube-single phase-1
  anchor. The choice is not cosmetic and there is no universally right default.

`impls/actors/flowbc.py:104` already handles this correctly: it emits **both**
keys — `next_action` at the actor goal, and `next_action_vgoal` whenever
`"value_goal" in cond` — reusing `na_noise` for the second draw so no extra
randomness is spent and every critic reading `next_action` stays bit-identical.
`DDPGBCActor`, `AWRActor` and `FQLActor` emit only `next_action`. That is why
`density_fb` cannot compose with `awr` or `fql` at all today, and it is one of
the three reasons `impls/actors/ddpgbc_density.py` was forked — see below for
the other two, which are why that file survives this change.

**Change, in two parts:**

1. **Every actor emits both keys**, following the `flowbc` pattern exactly —
   same reused-noise discipline, same `"value_goal" in cond` guard. This is what
   makes all 17 critics × 4 actors compose, and it preserves bit-identity by
   construction: existing critics keep reading `next_action`, which is drawn
   from the same key as before.
2. **`BOOT_COND` on the `Critic` protocol** (`impls/critics/base.py`) declares
   which key the critic reads — `"actor_goal"` (default, current behaviour) or
   `"value_goal"` (`density_fb`). It is a *declaration*, checked by `check_combo`
   and asserted by the contract test; it does **not** steer the draw. Keeping it
   out of the actor's control flow is deliberate: it means the change cannot move
   an existing number, only add keys that were previously missing.

Note `sample_bootstrap(self, net, batch, cond, rng)` receives no critic, and
`impls/actors/base.py` is a `Protocol` with a stub — each actor implements its
own. Part 1 is therefore four parallel edits, not one shared implementation.

**`impls/actors/ddpgbc_density.py` is KEPT.** An earlier draft of this design
proposed deleting it. That was wrong, and the reason is worth recording:
`DDPGBCDensityActor` differs from `DDPGBCActor` in three ways, not one.

1. `sample_bootstrap` emits `next_action_vgoal` — the part that part 1
   above generalizes to every actor.
2. `loss` computes `q_scale = |qs|.mean()` off the critic's **second** `q_fn`
   return — the centered normalizer under `density_fb`'s `actor_q_center`.
   Stock `DDPGBCActor` uses `|q|.mean()` off the first return. These are not the
   same quantity for any critic in the tree: `gciql.q_fn` returns
   `(min(q1,q2), stack([q1,q2]))`, `sf_feature` returns `(min(q1,q2), v)`,
   `td_fb` returns `(q, (q - q.mean())[None])`. Folding the centered normalizer
   into stock `ddpgbc` would therefore move every existing `x ddpgbc` cell.
3. It supports `const_std=false` via `boot["actor_sample_rng"]`, and logs
   `q_scale`, `mse` and `action_gap_l2`. Stock `ddpgbc` always takes `dist.mode()`.

It is the published centered-normalizer arm — `hard-ddpgbc 0.920/0.868/0.844`
against `hard-flow 0.772/0.580/0.656`. Deleting it would delete that arm.

The consequence is that **workstream 0 is purely additive**: actors gain a
bootstrap key they never emitted, and no existing key, loss or info value
changes. Bit-identity holds by construction, and the gate in section 8 confirms
it rather than hoping for it. `density_fb` ends up with five usable actors
(`flowbc`, `ddpgbc`, `awr`, `fql`, `ddpgbc_density`) instead of the one it has
today.

The unused second key is returned in `boot` and, under the jitted `Agent.update`,
dropped by XLA dead-code elimination in any critic that does not read it. This is
not separately asserted — a compiled-cost assertion is brittle across jax versions,
and the quantity at stake is one extra actor forward pass per update on critics
whose cond carries `value_goal`. What *is* asserted is the thing that matters: the
bit-identity gate proves the emission changes no existing value.

### 4.1 Files

```
impls/critics/base.py              +BOOT_COND on the Critic protocol; check_combo validates
impls/critics/density_fb.py        declares BOOT_COND = "value_goal"
impls/actors/{ddpgbc,awr,fql}.py   emit next_action_vgoal, flowbc-style
impls/actors/base.py               UNCHANGED (actors stay unaware of BOOT_COND by design)
impls/actors/flowbc.py             UNCHANGED (it already emits both keys)
impls/actors/ddpgbc_density.py     UNCHANGED (see above - it is not redundant)
tests/impls/test_boot_contract.py  NEW - bit-identity gate
```

`ACTORS` keeps its six entries: the four actor families (`flowbc`, `ddpgbc`,
`awr`, `fql`), plus `bcflow` (`FlowBCActor` with `q_coeff: 0`) and
`ddpgbc_density`, both of which are variants rather than families. "17 x 4"
throughout this document means the four families.

## 5. Workstreams

Workstream 0 merges first because everything else depends on the frozen contract;
porting four critics against a moving `sample_bootstrap` signature would mean
redoing all four. Workstreams 1–6 are mutually independent and run in parallel
worktrees. Workstream 7 lands last.

| # | workstream | ~files | depends on |
|---|---|---|---|
| 0 | seam boot contract | 7 | — |
| 1 | `gcivl` critic | 3 | 0 |
| 2 | `hiql` critic | 4 | 0 |
| 3 | `psm` critic (torch→jax) | 5 | 0 |
| 4 | `rldp` critic (torch→jax) | 4 | 0 |
| 5 | reward index for `density_fb` | 4 | 0 |
| 6 | ExoRL family in `env/` | 4 | — |
| 7 | online training loop | 5 | 0 |

### 5.1 Notes per workstream

**1 — GCIVL.** GCIQL minus the twin-Q: expectile value only, `PROVIDES_V = True`,
`PROVIDES_COND = {"value_goal", "goal"}`. Composes with `awr` natively and with
the other three through the standard `q_fn`.

**2 — HIQL.** The critic owns the subgoal representation *and* the high-level
policy; `sample_cond` emits a subgoal and `cond_vec` hands it to the actor. This
is what keeps the 4-actor cross-product intact for HIQL. `tests/impls/fixtures/hiql_reference.npz`
already exists on this branch but no test references it — it is an abandoned
start, and must be regenerated rather than trusted.

**3 — PSM.** The largest port: `agents/psm/{agent,model,psm_nets,proto_sampler}.py`
plus `_truncated_normal.py`. `proto_sampler.py` is stochastic; parity requires it
seeded and pinned or the fixture is meaningless.

**4 — RLDP.** `agents/rldp/{agent,model}.py`, 277 lines. Smallest of the four.

**5 — reward index for `density_fb`.** Port the spec-30 centered dual encoder as
shipped: `PROVIDES_COND` gains `"z"`, `infer_cond` becomes a real reward→`z` map,
`eval_cond0` gains the matching zero-`z`. Pinned against the shim's own numbers on
`gciql-fb-bridge`, not re-derived. Pin the **shipped** variant only — the index
moved four times across specs 27→30, and the older variants are not the target.

**6 — ExoRL.** `env/registry.py` resolves family, horizon and data_dir from a
task-slug prefix (`_PREFIX` currently holds `b2d`, `p2d`, `mw`, `ms`), so ExoRL is
one new prefix plus a backend and a loader. Conversion already exists on
`gciql-fb-bridge` as `scripts/exorl/convert_exorl.py`, with
`third_party/exorl_dmc_tasks/` and `tools/wandb_mode_shim/exorl_env.py`.

**7 — online.** `impls/main.py` (248 lines) is offline-only: `make_dataset`, then a
fixed `train_ds.sample` loop. Online adds env stepping inside training, a replay
buffer, and an exploration policy, behind a flag so the offline path stays
byte-identical.

## 6. Parity discipline

Every port follows the precedent already set on this branch by `gciql`, `crl`,
`qrl` and `vip`, whose docstrings all say *"transcribed from OGBench …; nothing
vendored is imported."*

1. Re-express the agent in the critic×actor protocol. Import nothing from
   `third_party/`.
2. Generate `tests/impls/fixtures/<name>_reference.npz` from the reference
   implementation.
3. Add an equivalence test asserting the seam critic matches it.

GCIVL and HIQL fixtures come from vendored OGBench. PSM and RLDP fixtures come
from the torch agents on `gciql-fb-bridge`, mapped through
`impls/utils/torch_to_flax.py` — 143 lines of `dense_params` / `edense_params` /
`eln_params` / `layernorm_params` helpers, already used by five existing tests, so
the torch→jax parity harness is proven rather than new.

## 7. Campaign scope

The seam supports the full method × actor cross-product. The **launch plan** is a
named subset, because the full product is roughly 1,250 runs at 500k–1M steps.

| table | cells | runs |
|---|---|---|
| **T1** goal-conditioned manipulation | 8 methods × `flowbc` × 12 domains × 3 seeds | 288 |
| **T2** actor study | 8 methods × 4 actors × {cube-single, mw-pick-place} × 3 seeds | 192 |
| **T3** ExoRL, reward-specified | 5 methods × 3 DMC domains × 3 seeds | 45 |
| **T4** online Density-FB | `density_fb` online × 3 DMC domains × 3 seeds | 9 |
| | | **≈534** |

T1 methods: `density_fb, fb, gciql, iql_bilinear, gcivl, hiql, psm, rldp`.
T1 domains (12): `cube-single, cube-double, scene, mw-{pick-place, push,
assembly, basketball, bin-picking, box-close, button-press}, ms-{pick-cube,
push-cube}`. `env/domains/` also holds `antmaze`, `pointmaze` and five `b2d-*`
domains; they are excluded from T1, which is manipulation-only.
T3 methods: `density_fb, fb, td_fb, psm, rldp` — the ones with a reward
interface. T3 domains: `point_mass_maze`, `walker`, `cheetah`, each on the **RND**
exploration dataset (one dataset, so the table varies method and domain only).
T4 uses the same three domains.

T3 is deliberately small: with a reward interface the tasks within a domain are
**zero-shot evals inside a single run**, not separate runs. That property is the
thing being tested. GC baselines appear in T3 as explicit
`N/A — no reward interface` cells rather than silent omissions.

`campaign/gen_impls.py` gains `--critics/--actors/--domains/--seeds`, so each
table is one invocation and anything outside them is a deliberate extra launch.

## 8. Gating

1. **Bit-identity** (workstream 0). Every pre-existing critic×actor pair produces
   byte-identical losses before and after the contract change, against fixtures
   generated on `metaworld-runs` @ `e35a542`. Blocks the merge of workstream 0,
   and therefore blocks everything.
2. **Parity** (workstreams 1–5). Each ported critic matches its reference within
   fixture tolerance. Blocks that critic's merge only — a failing PSM port does
   not hold up GCIVL.
3. **Cube anchor** (the port's own P5). One `density_fb × flowbc × cube-single`
   run to 500k must land near the bridge's 0.929. Launches as soon as workstream 0
   merges and runs concurrently with 1–6. **Blocks the T1–T4 launch.** If it
   misses, the branch stops and the port is diagnosed rather than spending ~534
   runs on an unvalidated critic.

## 9. Testing

- `tests/impls/test_boot_contract.py` — the bit-identity gate.
- One parity test per ported critic, with its fixture.
- `check_combo` coverage asserting all 17 x 4 pairs build, including the four
  pairs that cannot exist today: `density_fb` x `{awr, fql}`.
- A `td_fb` regression specifically: it must still read `next_action` at the
  actor goal. Silently flipping it to the value goal is the 0.920 -> 0.076 cliff.
- A noise-reuse test for `fql`, the only actor whose extra draw could move an
  existing value: its four random arrays are compared against the baseline so a
  future refactor that adds an rng split fails with the cause named.
- A cube train-smoke per new critic, following `tests/impls/test_cube_train_smoke.py`.
- An ExoRL dataset-loader test.
- An online-loop smoke, plus a test asserting the offline path is unchanged when
  the online flag is off.

## 10. Risks

- **HIQL-as-critic is a design bet.** Putting the subgoal representation and
  high-level policy inside the critic is what preserves the 4-actor cross-product.
  If the hierarchy will not factor that way, the fallback is a `hiql` actor plus a
  `gcivl`-on-rep critic — which costs clean actor control for that one row.
- **PSM parity is the hardest port**, for the stochastic-sampler reason above.
- **`measure_density_hard` disagrees with itself.** `impls/critics/density_fb.py`
  falls back to `False`; `impls/configs/critic/density_fb.yaml` sets `true` and
  documents `true` as the default since the bridge-port spec's item A2. Harmless
  while the yaml always supplies it, but a live trap for any path that builds the
  critic from a partial config. Fix the code default to match the documented one.
- **Workstream 5 is the highest-value and least-specified**, for the spec-churn
  reason above.
- **The cube anchor may legitimately fail.** The port's own spec flags
  `actor_q_center` as composed differently from the bridge — implemented inside
  `q_fn` rather than in the actor's normalizer. That is the most likely cause and
  the first thing to check.
- **Online has no baseline counterpart.** T4 is a Density-FB-only table; it
  demonstrates the capability, it does not compare methods.

## 11. Out of scope

- The torch `agents/dfb/` rewrite on `gciql-fb-bridge`. Untouched by this branch.
- The pixel/DrQ path and TD-MPC2.
- Any change to `tools/wandb_mode_shim/gciql_flow.py`; it is read as a reference
  for workstream 5, never edited.
- Retiring the torch stack. Out of scope even once every baseline has a jax twin.
