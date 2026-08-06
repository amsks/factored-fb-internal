# Spec 17b — `measure_delta_partition` implementation contract (variant B)

Audience: the implementing agent. Self-contained; science context in
`specs/17-delta-measure.md`, sibling knob in `specs/17a-delta-measure-impl.md`
(mutually exclusive, independent code paths — 17a needs nothing from 17b).
TDD; default-off must be BIT-identical.

## 0. What is being built

A one-stage simplex successor measure on `tools/wandb_mode_shim/gciql_flow.py`
(class `GCIQLFlowAgent`). A fixed partition of the object-configuration
space (K prototypes) turns the measure into a probability vector; a new head
learns it by cross-entropy TD; the flow actor reads its goal's cell mass.
No goal-region queries, no hindsight columns, no phase switch, no bilinear
measure matrix.

Objects:

    φ_k(x)      = softmax_k( −½ Σ_blocks ‖(x_blk − proto_k,blk)/ε_blk‖² )   ∈ Δ^K  (fixed)
    p(s,a,g)    = softmax( delta_head([s, a, g]) )                          ∈ Δ^K  (learned, 2 heads)
    fixed point p*(s,a,g) = (1−γ)·E^{π_g}[ Σ_{t≥0} γ^t φ(x_{t+1}) | s,a ]   (normalized occupancy)

TD (per sample, per head; g = value_goals; a⁺ ~ π(s', g) stop-gradded):

    p_tgt = sg[ (1−γ)·φ(x') + γ·p̄(s', a⁺, g) ]      (a probability vector)
    L     = CE(p_tgt, p) = −E[ Σ_k p_tgt,k · log(p_k + 1e−8) ]

Actor (flow actor, q-term only changes):

    z_g = sg[ φ(goal_obs) ]        q_h = Σ_k p_h(s, a_actor, g)_k · z_g,k

`p·z_g = (1−γ)M·z_g` contains the action-independent level `φ(x₀)·z_g`, so
`actor_q_center=True` is REQUIRED (lattice) — the q normalizer must be the
centered magnitude. Validity (p ≥ 0, Σp = 1) holds by construction: NO
ortho, NO hinge, NO mass anchor, NO scale_norm machinery anywhere in this
path (T10-b's runaway is structurally impossible; do not re-add its
anchors).

## 1. New files

### 1.1 `scripts/rescuing_fb/make_delta_prototypes.py`

CLI: `--dataset <npz path> --hit_spec '<start:stop:eps,...>' --k 256
--seed 0 --out <npy path>`. Loads the OGBench npz (`observations` key
suffices), parses the hit-spec blocks (same grammar as
`_parse_hit_spec`: comma-separated `start:stop:eps`), extracts the
concatenated block columns `[N, d_c]` (d_c = Σ(stop−start)), draws K
distinct row indices uniformly with the seeded rng, saves float32 `[K, d_c]`
via `np.save`, and prints K, d_c, per-dim min/max, and the mean
nearest-neighbor distance between prototypes (coverage sanity). Prototypes
are stored in RAW obs units — ε scaling happens at use.

Protos live somewhere durable (`results/delta_protos/<env>_K<k>_s<seed>.npy`
— NOT `/dev/shm`, which gets wiped; the checkpoint's config records this
path and restore needs the file present).

### 1.2 `tests/test_gciql_measure_delta_partition.py` — see §5.

## 2. Changes in `tools/wandb_mode_shim/gciql_flow.py`

### 2.1 `get_config()`

```python
config.measure_delta_partition = False  # spec 17b: one-stage simplex successor measure over a
  # fixed K-prototype partition of the hit-spec object blocks. delta_head([s,a,g]) -> K logits,
  # p=softmax; CE fit to sg[(1-gamma)*phi(x') + gamma*p_target(s',a'~pi,g)]; actor q = p . phi(goal)
  # at frozen head params. Requires measure_only + flow actor + actor_q_center + state obs;
  # mutually exclusive with measure_delta (17a). False = off, bit-identical.
config.measure_delta_protos = ''   # 17b: path to the [K, d_c] .npy prototype file (raw obs units,
  # rows = dataset draws of the concatenated measure_hit_spec blocks). Required when the knob is on.
config.measure_delta_coef = 1.0    # 17b: weight on the CE loss (keep 1.0; no dose sweeps round 1)
```

### 2.2 Agent field for the prototypes

> **AMENDED 2026-08-06 (post-audit).** The original text below specced the
> protos as a pytree leaf. Round-1 review convicted that design: a new
> pytree field breaks checkpoint restore for EVERY pre-existing ckpt
> (tree-structure mismatch). The SHIPPED design — do not regress to the
> struck text — is a `nonpytree_field` static hashable nested tuple
> (see gciql_flow.py, `delta_protos` definition, with the full rationale
> inline): never serialized, never a jit-traced leaf, rebuilt at
> `create()` from `measure_delta_protos` + the .npy on disk. Consequence
> (accepted, documented in the launcher): restoring a 17b checkpoint
> requires the .npy to still exist at the recorded path — hence mode 444,
> the `~/delta_protos_backup/` copy, and the sha256 registry in the eval
> script.

`GCIQLFlowAgent(GCIQLAgent)` gains one dataclass field, declared as a
STATIC non-pytree field holding a hashable nested tuple (converted from
the loaded [K, d_c] array), `None` when off. `create()` returns
`cls(rng, network=..., config=..., delta_protos=protos_tuple_or_None)`.
Old checkpoints keep loading: restore only touches
`['agent']['network']`, and the field is not part of the serialized tree
— pinned by `test_delta_protos_not_serialized_and_old_checkpoint_restores`.

### 2.3 φ helper (staticmethod, next to `_hit_matrix`)

```python
@staticmethod
def _delta_phi(obs_rows, protos, blocks):
    """phi(x) in Delta^K: softmax over K of -0.5 * sum_blocks ||(x_blk - proto_blk)/eps_blk||^2.
    obs_rows [B, obs_dim]; protos [K, d_c] raw obs units, columns = the blocks concatenated in
    _parse_hit_spec order; per-block eps scaling makes heterogeneous tolerances (scene) one knob."""
```

Implementation: slice each block from `obs_rows`, divide by that block's ε,
concat → `xc [B, d_c]`; scale `protos` columns the same way (block column
offsets = cumulative widths) → `pc [K, d_c]`; `logits = −0.5·‖xc[:,None]−pc[None]‖²`
summed over the last axis; return `jax.nn.softmax(logits, axis=-1)` `[B, K]`.

### 2.4 The loss — `delta_partition_loss(self, batch, grad_params, rng)`

```python
obs, nobs, acts, g = batch['observations'], batch['next_observations'], batch['actions'], batch['value_goals']
blocks = _parse_hit_spec(self.config.get('measure_hit_spec', '19:22:0.4'))
gamma = self.config['discount']
logits = self.network.select('delta_head')(jnp.concatenate([obs, acts, g], -1), params=grad_params)  # [2,B,K]
p = jax.nn.softmax(logits, -1)
# bootstrap action: one-shot flow actor at (s', g), stop-gradded -- the same machinery
# measure_loss uses (actor_kind='flow' is pinned by the lattice)
noise = jax.random.normal(rng, acts.shape)
a_plus = jax.lax.stop_gradient(self.network.select('actor')(nobs, g, noise))
t_logits = self.network.select('target_delta_head')(jnp.concatenate([nobs, a_plus, g], -1))       # [2,B,K]
p_bar = jax.nn.softmax(t_logits, -1)
phi_n = self._delta_phi(nobs, self.delta_protos, blocks)                                          # [B,K]
p_tgt = jax.lax.stop_gradient((1.0 - gamma) * phi_n[None] + gamma * p_bar)
ce = -jnp.mean(jnp.sum(p_tgt * jnp.log(p + 1e-8), axis=-1))
loss = self.config.get('measure_delta_coef', 1.0) * ce
```

Info keys: `delta_ce`, `delta_p_entropy` (−E Σ p log p, online), 
`delta_tgt_entropy`, `delta_p_max` (E max_k p), `delta_cell_change`
(= 0.5·E Σ_k |φ(x')−φ(x)| — the dense-supervision diagnostic; needs
`phi_c = _delta_phi(obs, ...)`, metrics only).

### 2.5 `total_loss` wiring

In the `measure_only` branch (grep `measure_only` in `total_loss`): when
`measure_delta_partition`, call `delta_partition_loss` INSTEAD of
`measure_loss` (IQL value/critic losses stay dropped; actor loss unchanged
call; prefix its info as the measure info is prefixed).

> **AMENDED 2026-08-06 (post-audit).** Round-1 review found that the base
> config's `measure_coef` (default 0.0) silently multiplied the CE to
> zero — a FAKE-NULL hazard on gate G3. The shipped code does NOT gate
> the CE on `measure_coef` at all, and the assert-lattice (§2.7) refuses
> `measure_coef > 0` outright so the inert knob cannot record misleading
> provenance. Any re-implementation must reproduce both halves. The bilinear
critic/value modules remain registered but receive zero gradients — that is
accepted dead weight in round 1 (do NOT try to deregister them; the
warm-start/restore machinery assumes the base tree).

### 2.6 `actor_loss` q-term

In the flow-actor path, where the stock branch computes
`q1, q2 = self.network.select('critic')(obs, goals, actor_actions, *zarg)`
(grep for that line), add a branch BEFORE it:

```python
if self.config.get('measure_delta_partition', False):
    blocks = _parse_hit_spec(self.config.get('measure_hit_spec', '19:22:0.4'))
    z_g = jax.lax.stop_gradient(self._delta_phi(goals, self.delta_protos, blocks))   # [B,K]
    logits_a = self.network.select('delta_head')(
        jnp.concatenate([obs, actor_actions, goals], -1))       # FROZEN params -- never grad_params
    q1, q2 = (jax.nn.softmax(logits_a, -1) * z_g[None]).sum(-1)
```

`goals = batch['actor_goals']` (already bound in that scope). The
BC-flow/distillation terms are untouched; the existing q normalizer applies
(centered, by the lattice). CRITICAL (crl f7b623d): no `params=grad_params`
on the delta_head call — enforced by test §5.6.

### 2.7 `create()` — registration, protos, lattice, target init

Registration (next to the `center_b`/`nhat_head` conditional blocks, so the
default tree stays byte-identical):

```python
if config.get('measure_delta_partition', False):
    protos_np = np.load(config['measure_delta_protos'])            # [K, d_c]
    blocks = _parse_hit_spec(config.get('measure_hit_spec', '19:22:0.4'))
    d_c = sum(b - a for (a, b, e) in blocks)
    assert protos_np.ndim == 2 and protos_np.shape[1] == d_c and protos_np.shape[0] >= 2, \
        f"measure_delta_protos must be [K>=2, {d_c}], got {protos_np.shape}"
    K = int(protos_np.shape[0])
    dh_def = ensemblize(MLP, 2)((*config['value_hidden_dims'], K),
                                activate_final=False, layer_norm=config['layer_norm'])
    ex_dh = (jnp.concatenate([ex_observations, ex_actions, ex_goals], -1),)
    network_info['delta_head'] = (dh_def, ex_dh)
    network_info['target_delta_head'] = (copy.deepcopy(dh_def), ex_dh)
```

After `params['modules_target_critic'] = params['modules_critic']` (end of
`create`), mirror it:

```python
if config.get('measure_delta_partition', False):
    params['modules_target_delta_head'] = params['modules_delta_head']
```

and return `cls(..., delta_protos=jnp.asarray(protos_np, jnp.float32))`
(None otherwise — keep the default-off return exactly as today).

Assert-lattice (one block, mirror the others' style):

```python
if config.get('measure_delta_partition', False):
    assert config.get('measure_only', False), "17b replaces the measure loss; set measure_only=True"
    assert not config.get('measure_delta', False), "measure_delta (17a) and 17b are mutually exclusive"
    assert not config.get('measure_first_passage', False), \
        "17b does not use the region-matrix path; set measure_first_passage=False"
    assert config.get('actor_kind', 'flow') == 'flow' and config.get('actor_cond', 'goal') == 'goal'
    assert config.get('actor_q_center', False), \
        "17b requires actor_q_center=True (p.z_g carries the action-independent level phi(x0).z_g)"
    assert config['encoder'] is None, "17b supports state obs only"
    assert config.get('measure_delta_protos', ''), "17b needs measure_delta_protos (.npy path)"
    assert config.get('measure_ortho_coef', 0.0) == 0.0, "no ortho anywhere near 17b (fp-noortho lesson)"
    for bad in ('fb_forward_z', 'sf_feature_td', 'measure_expectile', 'measure_nhat_head',
                'actor_z_infer', 'fwd_two_tower', 'isfb', 'freeze_critic_ckpt', 'warm_start_ckpt'):
        assert not config.get(bad, False), f"measure_delta_partition is incompatible with {bad}"
    assert int(config.get('measure_nstep', 0) or 0) == 0 and not config.get('measure_eps_multiscale', '')
    assert config.get('centered_measure', 'off') == 'off' \
        and config.get('task_prior', 'dirac') == 'dirac' \
        and config.get('task_prior_actor', 'dirac') == 'dirac'
```

### 2.8 `update` override (target soft-update for the head)

`GCIQLFlowAgent` currently INHERITS `update` from `GCIQLAgent` (which
soft-updates only `'critic'`). Add an override, identical to the base plus
the conditional extra target update (config is a static field, so the
branch resolves at trace time — bit-identical when off):

```python
@jax.jit
def update(self, batch):
    new_rng, rng = jax.random.split(self.rng)
    def loss_fn(grad_params):
        return self.total_loss(batch, grad_params, rng=rng)
    new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
    self.target_update(new_network, 'critic')
    if self.config.get('measure_delta_partition', False):
        self.target_update(new_network, 'delta_head')
    return self.replace(network=new_network, rng=new_rng), info
```

(The actual train loop is `third_party/ogbench/impls/main.py` —
`run_gciql.py` is pure orchestration and never touches the agent. Verify
main.py calls `agent.update(batch)` (it does in the vendored version) and
that no `full_update`/other entry point is used; override whichever is,
same pattern.)

## 3. Conventions the tests must pin

- φ and p live on the simplex: Σ_k = 1, ≥ 0, everywhere.
- Fixed point identity (tabular): p* = (1−γ)·E[Σ γ^t φ(x_{t+1})] — counting
  from the NEXT state, matching the repo's measure convention. The J-form
  readout used later is `F^Δ = p − φ(x₀)`; `Q_r = (r(x₀) + F^Δ·z_r)/(1−γ)`
  with `z_r,k = r(proto_k)`.
- ε scaling: block b's coordinates are divided by that block's ε from
  `measure_hit_spec` BEFORE the distance — scene's mixed tolerances
  (0.4/0.5/0.72/0.6) must produce comparable per-block contributions.

## 4. Anti-scope (learnings; do NOT add)

No ortho, no mass/nonneg penalties (simplex already conserves mass — §9.15c
made the anchor the convicted ingredient, so if ordering damage appears
here that is a RESULT to report, not a thing to patch), no pessimism beyond
the stock 2-head min, no scale_norm, no z-conditioning, no hindsight
columns, no warm start. One knob dose (coef 1.0).

## 5. Tests — `tests/test_gciql_measure_delta_partition.py`

Same harness as `tests/test_gciql_measure_nstep.py`
(`tests.jax_subprocess.run_jax`; OBS_D=40, action dim 5, d=8, B=16;
protos written to a tmp .npy inside the subprocess body). Base config:
`measure_only=True, measure_first_passage=False, actor_q_center=True,
measure_delta_partition=True, measure_delta_protos=<tmp>`, hit_spec
`'19:22:0.4'`, K=8 toy protos.

1. `test_default_off_bit_identical` — knob-off config: param tree byte-
   identical to no-knob config (no delta_head modules registered), and
   `total_loss` bit-identical on a fixed batch.
2. `test_phi_simplex_and_scaling` — `_delta_phi`: rows sum to 1, ≥ 0;
   argmax = nearest prototype on constructed points; 2-block spec
   `'19:22:0.4,28:31:0.8'` with a point off by 0.4 in block 1 vs another
   off by 0.8 in block 2 → equal logits (per-block ε scaling verified).
3. `test_ce_matches_numpy_reference` — small batch, compute p, p̄, φ(x')
   by direct module calls with the agent's own params, rebuild
   p_tgt/CE in numpy, compare to `delta_partition_loss` output (atol 1e-5);
   verify the (1−γ)/γ composition by asserting rows of p_tgt sum to 1.
4. `test_target_head_init_and_soft_update` — at create,
   target_delta_head params == delta_head params; after one `update()` on
   a synthetic batch, new_target == tau·online + (1−tau)·old_target
   (elementwise, the repo's `target_update` formula) — this catches a
   missing `update` override (target frozen forever) AND a missing init
   mirror.
5. `test_measure_only_wiring` — with the knob on, total_loss info contains
   `delta_ce` and NO `measure_loss` keys; IQL value/critic loss keys absent
   (measure_only semantics preserved).
6. `test_actor_grad_hygiene` — `jax.grad(actor_loss)`: grad-norm over
   `modules_delta_head` + `modules_target_delta_head` + `modules_critic`
   leaves ≈ 0; over `modules_actor` > 0. AND `jax.grad(delta_partition_loss)`:
   grad over `modules_actor` ≈ 0 (bootstrap stop-gradded), over
   `modules_delta_head` > 0, over `modules_target_delta_head` == 0.
7. `test_actor_q_reads_goal_cell` — construct goals sitting exactly on
   proto j*: z_g must be peaked at j* (max component); q equals
   Σ p·z_g for a hand-computed p (stub via direct module call).
8. `test_assert_lattice` — refusals: no measure_only; with measure_delta;
   with measure_first_passage; empty/missing protos path; wrong-width
   protos; actor_q_center=False; with warm_start_ckpt; with
   measure_ortho_coef>0. Accepts the base config.
9. `test_prototype_script` — run `make_delta_prototypes.py` as a subprocess
   on a synthetic npz (200 rows), K=16: output shape [16, d_c], float32,
   every row appears among the dataset's block rows.
10. `test_end_to_end_updates` — 30 `update()` steps on synthetic data:
    finite losses, entropy finite, no NaN in params; p stays on the simplex.

Runner: `.venv/bin/python -m pytest tests/test_gciql_measure_delta_partition.py -x -q`,
then the full measure-test neighborhood (see 17a §3) — all green.

## 6. Config (write after green; do NOT launch)

`configs/gciql/cube_single_state_measure_delta_partition_d50.yaml` — copy
`cube_single_state_measure_ballvisit_d50.yaml` for the launcher scaffolding
(env, steps, intervals, storage), then set the agent_overrides to EXACTLY:
`bc_coeff: 3.0`, `value_arch: bilinear`, `value_latent_dim: 50`,
`measure_only: true`, `measure_delta_partition: true`,
`measure_hit_spec: "19:22:0.4"`, `actor_q_center: true`,
`measure_delta_protos: /home/adityamohan/git/Factored-FB/results/delta_protos/cube_single_K256_s0.npy`
— and DROP the measure-matrix knobs the base carries (`measure_coef`,
`measure_legs`, `measure_scale_norm`, `measure_first_passage`,
`measure_fp_absorbing`, `measure_stopgrad_psi`, `measure_ortho_coef`):
none of them touch the 17b path (the lattice refuses fp; the rest would
just record misleading provenance). This is a FROM-SCRATCH one-stage cell —
no warm start (lattice-refused).
run_group `GCIQL-CubeSingle-Flow-Measure-DeltaPartition-d50`. The protos file itself is generated at launch
time by the wave script (spec 17 §6):
`make_delta_prototypes.py --dataset <cube-single npz> --hit_spec '19:22:0.4' --k 256 --seed 0`.

## 7. Out of scope for 17b

- cube-double/scene partition configs (gated on the cs result, spec 17 §7).
- The zero-shot readout script (z_r = r(protos)) — post-wave.
- Any change to GCDataset, the bilinear measure path, or 17a.
