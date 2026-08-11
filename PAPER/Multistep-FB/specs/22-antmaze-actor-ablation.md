# Spec 22 — `measure_density` on locomotion (antmaze-medium) + actor ablation

Status: WRITTEN 2026-08-11. Parents: spec 18 result (density = 0.929/3s on
cube-single, `GCIQL-CubeSingle-Flow-Measure-Density-d50`, gates passed) + the
sanity-antmaze campaign (the FLOW actor is the antmaze killer, not the
bilinear value: bilinear+DDPGBC 0.837 vs bilinear+flow 0.239 — the oft-cited
"0.88 vs 0.22" is imprecise, `results/summary_agents.csv:64` carries the
correction) + the domain-flip precedent (L2 measure-only + DDPGBC on antmaze
= 0.784, `results/bridge_grafts.csv:163`).

Two loaded questions, both pre-registered before any number lands:

1. **Locomotion**: does the normalized successor density transfer to
   antmaze-medium-navigate, a domain with NO dwell asymmetry (the agent IS
   the object — there is no freely-moving/held/slipping/dropped interaction
   variable for the measure to resolve)?
2. **Actor attribution, both directions**: on antmaze the flow actor is a
   known killer independent of the critic; on cube the flow actor beats
   DDPGBC (0.870 vs 0.563). So (a) a density+flow failure on antmaze must
   NOT be read as a measure failure, and (b) the cube density result 0.929
   may itself be actor-carried — the cube density+DDPGBC cell tests whether
   OUR headline number needs the flow actor.

TDD; default-off must be BIT-identical (all existing suites stay green).
Self-contained for the implementing agent; sibling contracts: 18 (density),
19a (behavior mode, untouched), 19 P0.2 (`actor_bc_only`, untouched).

## 0. One paragraph

Spec 18's `measure_density` currently hard-pins `actor_kind='flow'` in the
assert-lattice, and its actor q-term `q_h = Σ_j w_h(s,a,g,u_j)·kernel(g,u_j)`
(frozen towers, batch-resampled query support) lives only in the flow
branch of `actor_loss`. This spec relaxes the lattice to
`actor_kind ∈ {'flow','ddpgbc','awr_flow'}` and plugs the SAME density
q-readout into the stock gaussian DDPG+BC extraction and the AWR-flow
extraction — mirroring the agent's three existing actor paths
(`tools/wandb_mode_shim/gciql_flow.py:959` awr_flow, `:981` ddpgbc, flow
otherwise; the stock reference is
`third_party/ogbench/impls/agents/gciql.py:61` with its `awr`/`ddpgbc`
options). The critical trap: under `measure_only` the bilinear critic is
registered dead weight, and the ddpgbc branch today DELEGATES to
`GCIQLAgent.actor_loss`, which would silently maximize that never-trained
critic while the density CE trains beside it — a fake-NULL of exactly the
17b `measure_coef` species. The old L2 MeasureOnly-DDPGBC cell dodged this
only because its measure loss trained the critic modules themselves
(`measure_legs: critic`); density's towers are separate, so ddpgbc/awr get
explicit density branches, never delegation. The wave then runs the density
column of a {flow, ddpgbc, awr} × {density, stock-GCIQL} grid on
antmaze-medium-navigate (the stock column is ALREADY COLLECTED — reuse, do
not relaunch) plus cube-single density+ddpgbc/awr as the manipulation-side
mirror. hit spec for antmaze: the agent's own xy, `'0:2:0.5'` (ant obs =
[qpos(15); qvel(14)], xy = qpos[:2] = obs cols 0–1; ε = OGBench's own ant
goal tolerance 0.5, `ogbench/locomaze/maze.py:86`).

## 1. What each outcome means (pre-registered BEFORE launch)

Per-actor comparison, antmaze (density vs the collected stock control):

- **NULL (density ≈ stock within ~0.05, per actor)** — the expected result
  under the program's thesis. Locomotion has no dwell asymmetry: arrival ≈
  occupancy when the agent is the object, so a normalized occupancy density
  has nothing extra to resolve over a scalar TD critic. Parity certifies
  the density objective as a valid drop-in critic outside manipulation;
  it is NOT a locomotion win and must not be written up as one.
- **WIN (density+ddpgbc > 0.89 ≈ stock+0.05)** — would suggest the
  simplex-CE parameterization (structural non-negativity + unit mass +
  dense vector supervision) is a generically better TD target, not a
  manipulation-specific fix. Big if true; demand 3 seeds before claiming.
- **LOSS (density+ddpgbc ≪ 0.784 while stock holds 0.837)** — a
  locomotion-specific failure of the kernel/occupancy design. Ordered
  suspects: (1) kernel bandwidth — ε=0.5 on a ~20×20-unit maze is a far
  sharper relative kernel than ε=0.4 on cube coords; check `density_k_self`
  and ESS first; (2) xy-marginal query coverage (batch next-states cluster
  along corridors); (3) γ-horizon vs maze diameter. Bandwidth sweep is
  round 2, not a round-1 patch.
- **density+flow low (~0.24–0.38) with density+ddpgbc high** — replicates
  the stock actor ordering ⇒ the flow failure is actor-borne and
  measure-independent. This is the pre-registered EXPECTED shape; do not
  read density+flow alone as "density fails on locomotion".

Cube-single mirror (density+ddpgbc vs density+flow 0.929):

- **density+ddpgbc ≈ 0.93** ⇒ the cube headline is actor-robust; the
  measure carries it (stock ddpgbc sits at 0.563 bilinear / 0.703 mlp, so
  parity here would even mean density RESCUES the ddpgbc extraction).
- **density+ddpgbc ≈ 0.56–0.70 (stock-like)** ⇒ the flow actor is
  load-bearing on manipulation and the honest claim is
  "density measure + flow extraction", never the measure alone. Report
  either reading as a RESULT; neither is a bug.

## 2. Learnings applied

1. **Never delegate to a dead critic** (17b measure_coef fake-NULL): every
   density actor branch computes q from the density towers explicitly;
   tests assert zero gradient AND zero dependence on the `critic` modules.
2. **Actor never backprops into any measure tower** (crl f7b623d): frozen
   params + stop-grad on the goal kernel, enforced by grad-hygiene tests —
   now for all three actor kinds.
3. **Refuse inert knobs** (19a lesson): `actor_q_center` is meaningless in
   the AWR path (no q-scale normalizer exists there) — the lattice refuses
   it under awr_flow rather than letting it silently no-op.
4. **Domain-correct BC strength** (td_jepa HP audit 2026-07-07): antmaze
   runs `bc_coeff 0.3` (flow) / `alpha 0.3` (ddpgbc/awr), NOT the
   cube-tuned 3.0/1.0 — the original antmaze flow collapse was partly a
   bc_coeff transplant.
5. **Group names lie**: distinct run_group per cell; when matching a new
   cell to a collected control, copy the control's RECORDED config, not
   its group name.
6. **Controls that exist are not relaunched**: the stock column
   (flow/ddpgbc/awr × GCIQL) is in `results/summary_agents.csv` at 3 seeds
   each; burning GPU on re-collection buys nothing.
7. **ml_collections list trap**: any new knob scalar/string/bool (none are
   planned; the change is a lattice relaxation + two branches).
8. **`/dev/shm` datasets get wiped**: dataset path defaults to
   `~/.ogbench/data/antmaze-medium-navigate-v0.npz`.

## 3. Decisions pinned

- **No new config knob.** The relaxation is
  `assert actor_kind in ('flow','ddpgbc','awr_flow')` inside the density
  lattice block; the actor is selected by the existing `actor_kind`, and
  the density q plugs into whichever branch runs. A dedicated
  "density_actor" knob would just duplicate `actor_kind` and desynchronize
  from `create()`'s module registration (`gciql_flow.py:3683/:3693` already
  register the gaussian actor / actor_vf+value per kind).
- **One shared q helper.** Factor the flow branch's density q computation
  (`gciql_flow.py:1098–1112`) into
  `_density_actor_q(self, batch, obs, actions, goals) -> (q1, q2)` and call
  it from all three branches. The flow path must stay BIT-identical after
  the refactor (test §5.1).
- **ddpgbc branch mirrors the stock DDPG+BC loss** (gaussian dist,
  `const_std` mode/sample, `q = min(q1,q2)`,
  `q_loss = -q.mean()/sg(scale)`, `bc = -alpha·log_prob`) with q from
  `_density_actor_q`. `actor_q_center=True` REQUIRED for flow AND ddpgbc:
  density q carries an action-independent kernel level exactly like the
  flow case, so the centered scale `sg[mean|q - mean(q)|]` replaces the
  stock `sg[mean|q|]` in this branch.
- **awr_flow branch**: advantage `adv = q(s, a_data, g) − v̂(s, g)` with the
  baseline `v̂ = min-head density q at ā`, where ā = the Euler rollout of
  `actor_vf` at STOP-GRADDED current params (`_flow_rollout`,
  `flow_steps`) — a one-sample estimate of E_π[q] under the current BC-flow
  policy. The stock awr_flow's `value` network stays registered dead
  weight (it is IQL-trained, i.e. never trained under `measure_only`) and
  is NOT read. Weighting `exp(min(awr_alpha·adv, cap))` and the
  flow-matching BC term stay exactly as in the existing awr_flow branch
  (`gciql_flow.py:959–978`). Alternative baselines (K-sample mean, a
  trained expectile head on density q) are anti-scope round 1.
- **behavior mode (19a) stays flow-only** — it has no actor q-term at all;
  nothing to ablate.
- **`actor_bc_only` stays flow-only** — its lattice comment
  (`gciql_flow.py:3198–3210`) predates this spec and remains true for the
  stock kinds; extending it to the new density-ddpgbc/awr branches is
  anti-scope.
- **hit spec = `'0:2:0.5'`**, the env's own success rule (xy = obs cols
  0–1; ant `goal_tol = 0.5`). ε is NOT tuned in round 1; a bandwidth sweep
  is a pre-registered round-2 response to a specific failure signature
  (§1 LOSS suspects), not a launch-time degree of freedom.
- **Metric = native GC success** (main.py's standard eval), matching every
  anchor row quoted here. The zero-shot reward-readout harness on antmaze
  is a non-gating stretch item (§6), not part of the wave's gates.
- **Recipe splits by domain, matching the collected controls**: antmaze
  cells copy `configs/gciql/antmaze_medium_gciql_flow_bilinear_d50_bc03.yaml`
  scaffolding (eval_interval 100000, bc_coeff/alpha 0.3); cube cells copy
  `configs/gciql/cube_single_state_measure_density_d50.yaml` and change
  ONLY `actor_kind` (+ `alpha: 1.0` for ddpgbc, the OGBench cube value).

## 4. Implementation contract — `tools/wandb_mode_shim/gciql_flow.py`

### 4.1 Helper (next to the flow branch's density block)

```python
def _density_actor_q(self, batch, obs, actions, goals):
    """(q1, q2) = sum_j w_h(s,a,g,u_j) * kernel(goals, u_j) at FROZEN density
    towers, queries = the batch's own next-state block coords (train-eval z
    match). Extracted verbatim from the flow branch; shared by all three
    actor kinds. No params=grad_params anywhere; kg stop-gradded."""
```

Body = current lines 1103–1112 verbatim (`_parse_hit_spec`,
`_raw_block_cols`, `_density_queries`, kg via `_delta_phi`, frozen
`density_b`/`density_f`, softmax over queries, `(w * kg[None]).sum(-1)`).
The flow branch's `else:` arm calls it; behavior mode keeps `q1 = q2 = None`.

### 4.2 ddpgbc branch (inside `actor_kind == 'ddpgbc'`, BEFORE the
`GCIQLAgent.actor_loss` delegation at `gciql_flow.py:983`)

```python
if self.config.get('measure_density', False):
    # spec 22: density q-term under the stock gaussian DDPG+BC extraction.
    # NEVER delegate to GCIQLAgent.actor_loss here -- under measure_only the
    # bilinear critic is registered dead weight and delegation would silently
    # maximize a never-trained critic (17b fake-NULL species).
    obs, goals, actions = batch['observations'], batch['actor_goals'], batch['actions']
    rng = rng if rng is not None else self.rng
    dist = self.network.select('actor')(obs, goals, params=grad_params)
    q_actions = jnp.clip(dist.mode() if self.config['const_std']
                         else dist.sample(seed=rng), -1, 1)
    q1, q2 = self._density_actor_q(batch, obs, q_actions, goals)
    q = jnp.minimum(q1, q2) if self.config['double_q'] else q1
    q_scale = jnp.abs(q - q.mean()).mean()        # actor_q_center, lattice-required
    q_loss = -q.mean() / jax.lax.stop_gradient(q_scale + 1e-6)
    log_prob = dist.log_prob(actions)
    bc_loss = -(self.config['alpha'] * log_prob).mean()
    actor_loss = q_loss + bc_loss
    # info: actor_loss, q_loss, bc_loss, q_mean, bc_log_prob, mse (stock keys)
```

### 4.3 awr_flow branch (inside `actor_kind == 'awr_flow'`, replacing the
critic/value reads when density is on)

```python
if self.config.get('measure_density', False):
    # spec 22: AWR advantage from the density q. Baseline = one-sample
    # E_pi[q] at the BC-flow rollout action (stop-gradded params); the stock
    # 'value' net is IQL-trained and untrained under measure_only -- unread.
    q1, q2 = self._density_actor_q(batch, obs, actions, goals)
    q = jnp.minimum(q1, q2)
    vf_obs = jnp.concatenate([obs, goals], axis=-1)
    rng, v_rng = jax.random.split(rng)
    a_bar = self._flow_rollout(vf_obs, jax.random.normal(v_rng, actions.shape),
                               jax.lax.stop_gradient(grad_params))
    v1, v2 = self._density_actor_q(batch, obs, jnp.clip(a_bar, -1, 1), goals)
    adv = q - jnp.minimum(v1, v2)
else:
    <existing critic/value reads>
exp_a = jnp.minimum(jnp.exp(adv * self.config['awr_alpha']), 100.0)
<existing weighted flow-matching BC, unchanged>
```

(Whatever `_flow_rollout`'s exact signature is — mirror the flow branch's
distillation call at `gciql_flow.py:1160`; if the awr_flow path's vf input
convention differs, follow ITS convention. rng threading must not disturb
the existing awr_flow draw order when density is off — bit-identity test.)

### 4.4 Assert-lattice edits (the spec-18 block, `gciql_flow.py:3147`)

Replace line 3151's pin with:

```python
assert config.get('actor_kind', 'flow') in ('flow', 'ddpgbc', 'awr_flow'), \
    "spec 22: density supports flow / ddpgbc / awr_flow extractions only"
assert config.get('actor_cond', 'goal') == 'goal'
if not config.get('measure_density_behavior', False):
    if config.get('actor_kind', 'flow') == 'awr_flow':
        assert not config.get('actor_q_center', False), \
            "AWR has no q-scale normalizer; actor_q_center would be inert (19a rule)"
    else:
        assert config.get('actor_q_center', False)
if config.get('measure_density_behavior', False):
    assert config.get('actor_kind', 'flow') == 'flow'   # 19a stays flow-only
```

Everything else in the block (measure_only, mutual exclusions, no dueling,
no critic-mix, temp/dim/ortho/protos refusals) unchanged. The 19a block's
own `actor_kind == 'flow'` assert (line 3183) stays. `sample_actions`,
`create()` registration, and the `update` target-mix need NO changes — the
ddpgbc/awr_flow paths already exist there (`:2612–:2625`, `:3683/:3693`);
verify, don't re-implement.

### 4.5 Configs (new files, `configs/gciql/`)

- `antmaze_medium_state_measure_density_d50.yaml` — flow cell. Scaffold
  from the antmaze bc03 flow config; agent_overrides = the cube density
  cell's overrides with `measure_hit_spec: '0:2:0.5'`, `bc_coeff: 0.3`.
  run_group `GCIQL-AntMaze-Medium-Flow-Measure-Density-d50`.
- `antmaze_medium_state_measure_density_d50_ddpgbc.yaml` — + `actor_kind:
  ddpgbc`, `alpha: 0.3`, drop bc_coeff (flow-only knob), keep
  `actor_q_center: true`. run_group `...-DDPGBC-Measure-Density-d50`.
- `antmaze_medium_state_measure_density_d50_awrflow.yaml` — + `actor_kind:
  awr_flow`, `actor_q_center: false`, `awr_alpha: 10.0` (the collected
  awr_flow control's value). run_group `...-AWRFlow-Measure-Density-d50`.
- `cube_single_state_measure_density_d50_ddpgbc.yaml` /
  `..._awrflow.yaml` — the cube density config with only the actor knobs
  changed (`alpha: 1.0` for ddpgbc). run_groups
  `GCIQL-CubeSingle-DDPGBC-Measure-Density-d50` /
  `GCIQL-CubeSingle-AWRFlow-Measure-Density-d50`.

## 5. Tests — `tests/test_gciql_density_actors.py`

Same harness as the spec-18 suite (`tests.jax_subprocess.run_jax`; OBS_D=40
for cube-shaped cases, OBS_D=29 with hit_spec `'0:2:0.5'` for the antmaze
shape; d=8, B=16).

1. `test_flow_path_bit_identical_after_refactor` — density+flow
   `total_loss` on a fixed batch is bit-identical before/after the
   `_density_actor_q` extraction (golden value captured pre-change).
2. `test_stock_kinds_bit_identical_without_density` — plain GCIQL-flow
   configs with `actor_kind='ddpgbc'` / `'awr_flow'` and NO density knob:
   param trees and `total_loss` bit-identical to HEAD (delegation and rng
   draw order untouched).
3. `test_ddpgbc_density_numpy_reference` — rebuild dist-mode actions, w,
   kg, q, centered scale, q_loss + alpha·BC in numpy from direct module
   calls (atol 1e-5).
4. `test_ddpgbc_density_grad_hygiene` — `jax.grad(actor_loss)`: ≈0 over
   density_f/density_b/targets AND over the dead `critic`/`value` leaves;
   >0 over the gaussian actor. `jax.grad(density_loss)`: ≈0 over the actor.
5. `test_ddpgbc_density_never_reads_critic` — perturb the dead critic
   params by +1e3: `actor_loss` value unchanged to the last bit (the
   anti-delegation guarantee, stronger than a zero-grad check).
6. `test_awr_density_numpy_reference` — adv = q(a_data) − q(ā) with ā from
   a hand-rolled Euler integration of actor_vf; exp_a cap; weighted
   flow-matching loss matches numpy (atol 1e-5).
7. `test_awr_density_grad_hygiene` — no grads into density towers, targets,
   dead critic, dead value; grads into actor_vf only.
8. `test_lattice_spec22` — accepts flow(+center), ddpgbc(+center),
   awr_flow(−center); refuses ddpgbc without actor_q_center, awr_flow WITH
   actor_q_center, behavior+ddpgbc, actor_cond='z' with any kind,
   actor_bc_only+ddpgbc; all pre-existing spec-18 refusals still fire.
9. `test_antmaze_hit_spec_shape` — OBS_D=29, `'0:2:0.5'`:
   `_density_queries`/`_delta_phi` slice cols 0–2, scale by 0.5; kernel
   row i peaks at j=i (self-coverage) on synthetic xy data.
10. `test_end_to_end_updates_both_kinds` — 30 `update()` steps for
    density+ddpgbc and density+awr_flow: finite losses, w on the simplex,
    no NaN; target towers τ-mix.

Runner: `.venv/bin/python -m pytest tests/test_gciql_density_actors.py -x -q`,
then the spec-18/19a/19 suites (lattice touched) — all green before launch.

## 6. Eval

Native GC success from main.py's standard loop — nothing new. Pre-launch
sanity (one-off, scratchpad): load the antmaze dataset, report the fraction
of next-state xy's within ε=0.5 of a uniformly-relabelled goal xy and the
mean `k_self` of `_delta_phi` on a 1k-row sample — if hit coverage is an
order of magnitude below cube-single's, say so in the launch log (it
sharpens the §1 LOSS suspects; it does not block the launch).

Stretch (non-gating, after the wave lands): add
`"antmaze-medium-navigate-v0": EnvGeometry(..., HitSpec.parse("0:2:0.5"), ...)`
to `scripts/rescuing_fb/env_geometry.py` and run
`eval_density_zeroshot.py` on the density+ddpgbc checkpoints (dirac family
only). Directional read, no gate.

## 7. Pre-registered wave (500k, standing recipe)

Standing recipe as `launch_spec18_cs.sh`: `.venv/bin/python` launcher, wandb
entity **amsks** (verify before launch), tmux session `s22am`, dispatcher
log under `/dev/shm/spec22_logs/`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
outroot `~/exp_spec22`, datasets from `~/.ogbench/data/` (never /dev/shm).
eval_episodes 50; eval_interval 100000 (antmaze) / 50000 (cube).

Launched cells (the density column + cube mirror; controls are reused):

| cell            | env             | actor    | seeds | note |
|-----------------|-----------------|----------|-------|------|
| a22_am_dflow    | antmaze-medium  | flow     | 3     | discriminating |
| a22_am_dddpg    | antmaze-medium  | ddpgbc   | 3     | discriminating |
| a22_am_dawr     | antmaze-medium  | awr_flow | 1     | screen |
| a22_cs_dddpg    | cube-single     | ddpgbc   | 3     | manipulation mirror (loaded) |
| a22_cs_dawr     | cube-single     | awr_flow | 1     | screen |

11 jobs total. Reused controls (NO relaunch; quote from the CSVs):
gciql+ddpgbc antmaze 0.837/3s (`summary_agents.csv:64`), gciql+flow antmaze
0.383/3s bc0.3 (`:67`; 0.239 at the wrong bc3.0, `:66`), gciql+awr_flow
antmaze 0.617/3s (`:63`); cube: gciql+flow 0.870 (`:14`), gciql+ddpgbc
bilinear 0.563 / mlp 0.703 (`:9`/`:10`), density+flow 0.929/3s
(`bridge_grafts.csv:97`).

Gates:

- **G1 (locomotion viability)**: density+ddpgbc mean ≥ 0.70 at 500k over 3
  seeds. Anchors bracketing it: stock ddpgbc 0.837, L2 measure-only+ddpgbc
  0.784.
- **G2 (actor attribution, antmaze)**: the stock actor ordering replicates
  under density — density+ddpgbc − density+flow ≥ 0.2. If instead
  density+flow ≈ density+ddpgbc ≥ 0.7, the density q-term RESCUES the flow
  extraction on antmaze — report as a standalone finding (it would
  implicate the stock critic's q-landscape, not the flow machinery, in the
  original 0.239).
- **G3 (health)**: median `density_ess` ≥ 4 at 500k and `density_w_entropy`
  not → 0 while success holds; `density_k_self` reported per env (the
  bandwidth diagnostic feeding §1's LOSS suspects).
- **G4 (manipulation mirror)**: no threshold — the two §1 readings are the
  deliverable. Quote density+ddpgbc(cube) against BOTH density+flow 0.929
  and stock ddpgbc 0.563/0.703.

Interpretations are pinned in §1; read them BEFORE reacting to a number.
Further anchors for context only: FB flowbc antmaze 0.593/3s (`:62`), HIQL
awr antmaze 0.958/8s (domain ceiling), GCIVL-flow antmaze 0.829 (V-only
critics tolerate the flow actor). The OGBench paper's own GCIQL
antmaze-medium-navigate number (~0.7, from memory) is UNVERIFIED — the
repo's 0.837 (bilinear-d50) / 0.695 (stock mlp) rows take precedence.

Post-collection: add all five run_groups to `scripts/results_map.py`, then
`.venv/bin/python scripts/compile_results_csvs.py --refresh` (standing rule;
also still owed for any stale d17/d18 groups found at that point). Record
gate verdicts in this file under a `## 10. RESULTS` section.

## 8. Anti-scope (do NOT add in round 1)

No ε/bandwidth sweep on antmaze (round-2 response to a G3/LOSS signature
only); no antmaze-large/giant, no point/humanoid mazes; no stump/teleport
variants; no trained value baseline for the AWR branch (expectile head on
density q = round 2); no `actor_bc_only` extension to ddpgbc/awr; no
behavior-mode (19a) actor ablation; no changes to `sample_actions`,
`create()` registration, or GCDataset; no relaunch of any stock control; no
zero-shot gate on antmaze (§6 stretch is directional only); no pixel path;
no new knobs of any kind — if a knob feels needed, the design is wrong.

## 9. Open questions carried forward

- If G2's rescue reading fires (density fixes flow on antmaze), the
  spec-18 program gains a locomotion leg — would need its own deconfound
  (is it the simplex CE, the kernel goal-relevance, or the q centering?).
- If the cube mirror shows density+ddpgbc ≈ 0.93, the flow actor exits the
  cube story entirely and every future manipulation cell gets cheaper
  (gaussian actor, no distillation). Pre-register that follow-up wave
  separately before acting on it.
- AWR screens (1 seed) promote to 3 seeds only if they land within 0.05 of
  the same-env ddpgbc density cell — otherwise AWR stays a footnote.
- Dwell-asymmetry claim: the §1 NULL reading asserts locomotion lacks it;
  the honest version of that claim eventually wants the spec-15 dwell
  statistics computed on the antmaze dataset (arrival×dwell decomposition
  on xy cells). Eval-only, existing scratchpad tooling; not this wave.
