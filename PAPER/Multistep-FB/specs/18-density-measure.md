# Spec 18 — `measure_density`: continuous normalized successor density

Status: WRITTEN 2026-08-06. Parents: spec 17b result (part = 0.884/0.884 on
cube-single, gate G3 passed both seeds; delta = zero-flux collapse as
pre-registered) + external proposal 2026-08-06 #2 ("keep the simplex
constraint, replace the finite simplex with a continuous probability
measure") + 2026-08-05 stack-divergence audit (z-conditioning is the
obstruction ⇒ round 1 stays g-conditioned).

Naming: "density" = the learned object is a normalized successor
probability density over continuous object configurations, w.r.t. the
dataset object marginal ρ_x. We avoid "flow" (FlowBC collision), "energy"
(invites EBM-sampler readings; nothing is sampled from the model), and
"contrastive" (collides with the CRL cells, which are Monte-Carlo,
full-state, and not goal-conditioned TD — this is none of those).

TDD; default-off must be BIT-identical. Self-contained for the
implementing agent; sibling contracts: 17a/17b (all three knobs mutually
exclusive).

## 0. One paragraph

Part (17b) wins while Delta (17a) collapses, and the audit + result read
attribute the win to the *parameterization*: non-negativity and unit mass
structural (softmax), a proper cross-entropy projection of one probability
vector onto another, dense per-transition vector supervision. The 256
fixed prototypes are NOT on that list — they exist only because
`delta_head` is an MLP with K output logits. Spec 18 removes the fixed
support: replace `MLP([s,a,g]) → K logits` by a two-tower score
`f(s,a,g,u) = F(s,a,g)ᵀB(u)/τ` over arbitrary object-coordinate queries
`u`, softmax-normalized over a query set **resampled from ρ_x every
update** (round 1: the batch's own next-state object coords). The CE
target keeps part's exact structure `sg[(1−γ)·k + γ·w̄⁺]`, where the
immediate term `k` is literally today's `_delta_phi` evaluated with
`protos := queries`. The fixed point is the ε-mollified normalized
occupancy density w.r.t. ρ_x — FB's continuous object `M(s,a,du) =
m(u)ρ_x(du)` — obtained WITHOUT FB's convicted ingredients (unnormalized
signed L2 measure loss, ortho: 45% negativity; fp-noortho 0.967 vs
with-ortho 0.225). The query count M becomes a Monte-Carlo sample size,
not a spatial resolution: queries change every update, eval can use a
larger pool than training, and increasing M reduces integration error
rather than changing the represented space. No prototype .npy, no
checkpoint↔file coupling (audit finding #2 disappears structurally).

## 1. Object and estimator

Normalize the successor measure: ν(du|s,a,g) = (1−γ)·M(du|s,a,g). Since
M(X|s,a) = 1/(1−γ), ν is a probability measure over object
configurations. Bellman equation (counting from the NEXT state, repo
convention):

    ν(du|s,a,g) = (1−γ)·κ_ε(du|x′) + γ·E[ν(du|s′,a⁺,g)],  a⁺ ~ π(·|s′,g)

with κ_ε(du|x′) ∝ K_ε(u,x′)·ρ_x(du), the ε-kernel mollifier
K_ε(u,x′) = exp(−½ Σ_blocks ‖(u_blk − x′_blk)/ε_blk‖²) — the continuous
analogue of φ(x′); it does NOT partition the space.

Parameterization (exponential family w.r.t. ρ_x; F = natural parameter,
B = sufficient statistic):

    f(s,a,g,u)  = F(s,a,g)ᵀB(ũ)/τ              ũ = per-block ε-scaled coords of u
    m(u|s,a,g)  = exp f(s,a,g,u) / E_{v~ρ_x}[exp f(s,a,g,v)]
    ν(du|s,a,g) = m(u|s,a,g)·ρ_x(du)            m ≥ 0, E_ρ[m] = 1 by construction

Estimator (per batch, per head; U = {u_1..u_M} ~ ρ_x resampled each
update; g = value_goals; a⁺ stop-gradded):

    w_ij  = softmax_j f(s_i, a_i, g_i, u_j)                       (online, [B,M])
    k_ij  = softmax_j( −½ Σ_b ‖(x′_i,b − u_j,b)/ε_b‖² )           (= _delta_phi(nobs, U); [B,M])
    w̄⁺_ij = softmax_j f̄(s′_i, a⁺_i, g_i, u_j)                     (target towers, [B,M])
    t_ij  = sg[ (1−γ)·k_ij + γ·w̄⁺_ij ]                            (rows on the simplex)
    L     = −E_i[ Σ_j t_ij · log(w_ij + 1e−8) ]

As M→∞ the self-normalized empirical measure Σ_j w_ij δ_{u_j} converges
to ν; self-normalization bias is O(1/M) (standard; stated, accepted).
With U = batch next-obs coords, every row's immediate target has its own
successor in the support (row i's k peaks at j=i), which is the §7
coverage guarantee of the proposal, for free.

Readout for any reward r at test time, over an eval pool U_P ~ ρ_x
(P ≫ M_train allowed — integration error only):

    Q̂_r(s,a,g) = ( r(x₀) + (w − κ(x₀|U_P))ᵀ r(U_P) ) / (1−γ)      (PRIMARY; part's eq. with protos→pool)
    μ(s,a,g)   = Σ_j w_j B(ũ_j);  r ≈ w_rᵀB fit by ridge under ρ_x;
    Q̂_r ≈ ( r(x₀) + (μ − μ₀)ᵀw_r ) / (1−γ)                        (SECONDARY: compact reward code)

where κ(x₀|U_P) = _delta_phi(obs, U_P) and μ₀ = κ(x₀|U_P)ᵀB(Ũ_P) — the
same centered de-biasing part uses (action-independent; exact-r at x₀
replaces the smoothed estimate). The dot-product theorem of classical FB
is lost to the partition function; the SECONDARY readout is its repair
and a genuine expectation under a valid measure. Linearity in r holds
exactly for the PRIMARY readout.

## 2. Learnings applied

1. **z-conditioning is the obstruction** (stack-divergence 2026-08-05):
   f is conditioned on g = value_goals, exactly like part. w_r/μ live
   eval-side ONLY; feeding w_r to the actor is a later, separately-gated
   step (§9), not round 1.
2. **No ortho, no anchors, no hinges** (fp-noortho; §9.15c): the softmax
   supplies validity; nothing is bolted on. The lattice refuses ortho.
3. **Actor never backprops into any critic head** (crl f7b623d): frozen
   params + sg on goal weights, enforced by grad-hygiene TESTS.
4. **No pessimism beyond the stock 2-head min** (T7).
5. **No penalties as patches**: ESS collapse / premature sharpening is a
   pre-registered RESULT (§7), not a thing to regularize away in round 1.
   The entropy regularizer of the proposal's §7 is anti-scope.
6. **Train-eval z match** (rz law analogue): the actor consumes goal
   weights built by the SAME `_delta_phi` kernel eval uses.
7. **Scale identities tested up front**: rows of t on the simplex is a
   unit test; readout formula has a numpy reference test.
8. **ml_collections list trap**: every new knob scalar/string/bool.
9. **Group names lie**: distinct run_group per arm; audit by config.

## 3. Decisions pinned (deviations from the raw proposal, with reasons)

- **g-conditioned, not z-conditioned** (§2.1). The proposal writes
  f(s,a,z,u); round 1 is f(s,a,g,u), drop-in for the 17b slot.
- **U = batch next-state object coords, M = batch size.** No GCDataset
  change (17b precedent), ρ_x-distributed, coverage guarantee free.
  Extra iid draws = §9.
- **Rank-d expressiveness regression pre-registered.** Part's head emits
  K unconstrained logits; the two-tower log-density is rank-d (default
  d = 50, the slot's latent dim). If density underperforms part at
  matched budget, suspects are rank/temperature FIRST — fallback is a
  joint MLP score f = MLP([s,a,g,ũ]) (loses precomputable readouts) —
  and "continuity is harmful" may only be claimed after both are swept.
- **actor_q_center REQUIRED**: the action-independent level
  (the κ(x₀)ᵀz analogue) survives the reparameterization.
- **Temperature fixed, not learned** (round 1): `measure_density_temp`,
  default 1.0.
- **B input is ε-scaled block coords** so scene's heterogeneous
  tolerances (0.4/0.5/0.72/0.6) are normalized before the MLP, matching
  the kernel's geometry.
- **Ensembling: F 2-headed, B single.** double_q = min over the two w's
  at actor/eval; CE trains both heads. B is a feature map of x alone; a
  second B buys nothing the min needs.

## 4. Changes in `tools/wandb_mode_shim/gciql_flow.py`

### 4.1 `get_config()` (next to the 17b knobs)

```python
config.measure_density = False   # spec 18: continuous normalized successor density.
  # f(s,a,g,u) = F(s,a,g).B(u)/temp scored against the batch's own next-state object
  # coords as queries, softmax-normalized; CE fit to sg[(1-gamma)*k + gamma*w_bar+];
  # k = _delta_phi(nobs, queries). Actor q = sum_j w_j * kernel(goal, u_j) at frozen
  # params. Requires measure_only + flow actor + actor_q_center + state obs; mutually
  # exclusive with measure_delta (17a) and measure_delta_partition (17b).
  # False = off, bit-identical.
config.measure_density_dim = 50        # d of the F/B towers (the slot's latent dim)
config.measure_density_temp = 1.0      # softmax temperature tau (fixed, not learned)
config.measure_density_hard = False    # ablation: immediate term = one-hot at the row's own
  # successor (k = I) instead of the eps-kernel softmax. Attributes smoothing vs simplex-TD.
```

No protos knob, no protos field, no .npy. Nothing static beyond config.

### 4.2 Query helper (staticmethod, next to `_delta_phi`)

```python
@staticmethod
def _density_queries(nobs, blocks):
    """[B, d_c] eps-scaled block coords of next_observations -- the resampled
    query support U-tilde AND the B-tower input rows. Raw-unit twin for the
    kernel term is the plain block slice (no scaling); _delta_phi does its
    own scaling internally, so the kernel path passes RAW rows."""
```

Slice each block from `nobs`, divide by that block's ε, concat. (The
kernel term reuses `_delta_phi(nobs, protos=raw_block_rows, blocks)` —
`_delta_phi` already handles scaling; do NOT double-scale.)

### 4.3 The loss — `density_loss(self, batch, grad_params, rng)`

```python
obs, nobs, acts, g = batch['observations'], batch['next_observations'], batch['actions'], batch['value_goals']
blocks = _parse_hit_spec(self.config.get('measure_hit_spec', '19:22:0.4'))
gamma, tau = self.config['discount'], self.config['measure_density_temp']
u_raw = <concatenated raw block columns of nobs>          # [M, d_c], M == B
u_sc  = self._density_queries(nobs, blocks)               # [M, d_c] eps-scaled
b_emb = self.network.select('density_b')(u_sc, params=grad_params)          # [M, d]
f_emb = self.network.select('density_f')(jnp.concatenate([obs, acts, g], -1),
                                         params=grad_params)                # [2, B, d]
logits = jnp.einsum('hbd,md->hbm', f_emb, b_emb) / tau
logp = jax.nn.log_softmax(logits, -1)                                       # [2, B, M]
noise = jax.random.normal(rng, acts.shape)
a_plus = jax.lax.stop_gradient(self.network.select('actor')(nobs, g, noise))
tb = self.network.select('target_density_b')(u_sc)
tf = self.network.select('target_density_f')(jnp.concatenate([nobs, a_plus, g], -1))
w_bar = jax.nn.softmax(jnp.einsum('hbd,md->hbm', tf, tb) / tau, -1)
if self.config.get('measure_density_hard', False):
    k = jnp.eye(nobs.shape[0])                                              # [B, M], M == B
else:
    k = self._delta_phi(nobs, u_raw, blocks)                                # [B, M]
t = jax.lax.stop_gradient((1.0 - gamma) * k[None] + gamma * w_bar)
ce = -jnp.mean(jnp.sum(t * logp, axis=-1))
```

Same no-split rng convention as `delta_partition_loss`. No coef knob —
the 17b `measure_coef` lesson: one loss, weight 1, refuse anything else.

Info keys: `density_ce`, `density_ess` (E[1/Σ_j w²] over rows/heads —
the proposal's effective-support monitor), `density_w_entropy`,
`density_tgt_entropy`, `density_logit_max`, `density_k_self` (E[k_ii],
coverage diagnostic; ≡ 1.0 when hard).

### 4.4 `total_loss` wiring

In the `measure_only` branch: when `measure_density`, call `density_loss`
INSTEAD of `measure_loss` (IQL value/critic losses stay dropped; actor
loss call unchanged; prefix info as the measure info is prefixed). The
bilinear critic/value modules remain registered dead weight (17b
precedent — do not deregister).

### 4.5 `actor_loss` q-term (branch before the stock critic call, next to
the 17b branch)

```python
if self.config.get('measure_density', False):
    blocks = _parse_hit_spec(self.config.get('measure_hit_spec', '19:22:0.4'))
    u_raw = <raw block columns of batch['next_observations']>
    u_sc  = self._density_queries(batch['next_observations'], blocks)
    kg = jax.lax.stop_gradient(self._delta_phi(goals, u_raw, blocks))       # [B, M]
    b_emb = self.network.select('density_b')(u_sc)                          # FROZEN
    f_emb = self.network.select('density_f')(
        jnp.concatenate([obs, actor_actions, goals], -1))                   # FROZEN
    w = jax.nn.softmax(jnp.einsum('hbd,md->hbm', f_emb, b_emb)
                       / self.config['measure_density_temp'], -1)
    q1, q2 = (w * kg[None]).sum(-1)
```

`goals = batch['actor_goals']`. No `params=grad_params` on either tower
call (crl f7b623d) — enforced by test §5.6. Existing centered q
normalizer applies (`actor_q_center=True`, lattice-required).

### 4.6 `create()` — registration, lattice, target init

Registration (next to the 17b conditional block):

```python
if config.get('measure_density', False):
    blocks = _parse_hit_spec(config.get('measure_hit_spec', '19:22:0.4'))
    d_c = sum(b - a for (a, b, e) in blocks)
    d = int(config['measure_density_dim'])
    f_def = ensemblize(MLP, 2)((*config['value_hidden_dims'], d),
                               activate_final=False, layer_norm=config['layer_norm'])
    b_def = MLP((*config['value_hidden_dims'], d),
                activate_final=False, layer_norm=config['layer_norm'])
    ex_f = (jnp.concatenate([ex_observations, ex_actions, ex_goals], -1),)
    ex_b = (<eps-scaled block slice of ex_observations>,)   # [1, d_c]
    network_info['density_f'] = (f_def, ex_f)
    network_info['density_b'] = (b_def, ex_b)
    network_info['target_density_f'] = (copy.deepcopy(f_def), ex_f)
    network_info['target_density_b'] = (copy.deepcopy(b_def), ex_b)
```

Target init mirror (next to the 17b mirror):

```python
params['modules_target_density_f'] = params['modules_density_f']
params['modules_target_density_b'] = params['modules_density_b']
```

Assert-lattice (one block, mirror 17b's style):

```python
if config.get('measure_density', False):
    assert config.get('measure_only', False)
    assert not config.get('measure_delta', False) and not config.get('measure_delta_partition', False)
    assert not config.get('measure_first_passage', False)
    assert config.get('actor_kind', 'flow') == 'flow' and config.get('actor_cond', 'goal') == 'goal'
    assert config.get('actor_q_center', False)
    assert config['encoder'] is None, "state obs only"
    assert float(config.get('measure_density_temp', 1.0)) > 0.0
    assert int(config.get('measure_density_dim', 50)) >= 2
    assert config.get('measure_ortho_coef', 0.0) == 0.0
    assert float(config.get('measure_coef', 0.0)) <= 0.0, "17b lesson: no inert coef knobs"
    for bad in ('fb_forward_z', 'sf_feature_td', 'measure_expectile', 'measure_nhat_head',
                'actor_z_infer', 'fwd_two_tower', 'isfb', 'freeze_critic_ckpt', 'warm_start_ckpt'):
        assert not config.get(bad, False)
    assert int(config.get('measure_nstep', 0) or 0) == 0 and not config.get('measure_eps_multiscale', '')
    assert config.get('centered_measure', 'off') == 'off' \
        and config.get('task_prior', 'dirac') == 'dirac' \
        and config.get('task_prior_actor', 'dirac') == 'dirac'
    assert not config.get('measure_delta_protos', ''), "density has no prototype file"
```

### 4.7 `update` override

Extend the existing 17b override (trace-time branch, bit-identical off):

```python
if self.config.get('measure_density', False):
    self.target_update(new_network, 'density_f')
    self.target_update(new_network, 'density_b')
```

## 5. Tests — `tests/test_gciql_measure_density.py`

Same harness as the 17b suite (`tests.jax_subprocess.run_jax`; OBS_D=40,
action dim 5, d=8, B=16; hit_spec `'19:22:0.4'`). Base config:
`measure_only=True, actor_q_center=True, measure_density=True,
measure_density_dim=8`.

1. `test_default_off_bit_identical` — param tree byte-identical to
   no-knob config (no density modules); `total_loss` bit-identical on a
   fixed batch.
2. `test_query_scaling_matches_delta_phi` — 2-block spec
   `'19:22:0.4,28:31:0.8'`: `_density_queries` scaling agrees with
   `_delta_phi`'s internal scaling (equal-logit construction), and the
   kernel term `k` row i peaks at j=i (self-coverage).
3. `test_ce_matches_numpy_reference` — rebuild w, w̄⁺, k, t, CE in numpy
   from direct module calls with the agent's own params (atol 1e-5);
   assert rows of t sum to 1 (both hard and soft immediate).
4. `test_target_towers_init_and_soft_update` — at create, target == online
   for BOTH towers; after one `update()`, τ-mix formula elementwise for
   both (catches a missing override branch or init mirror).
5. `test_measure_only_wiring` — knob on: info has `density_ce`, no
   `measure_loss`/`delta_ce` keys; IQL value/critic keys absent.
6. `test_actor_grad_hygiene` — `jax.grad(actor_loss)`: ≈0 over
   density_f/density_b/targets/critic leaves, >0 over actor.
   `jax.grad(density_loss)`: ≈0 over actor, >0 over density_f AND
   density_b, ==0 over both targets.
7. `test_actor_q_reads_goal_kernel` — goals placed exactly on one query
   row: kg peaks there; q equals Σ w·kg for hand-computed w via direct
   module calls.
8. `test_assert_lattice` — refusals: no measure_only; with measure_delta;
   with measure_delta_partition; with measure_first_passage;
   actor_q_center=False; temp=0; with measure_delta_protos set; with
   warm_start_ckpt; measure_ortho_coef>0. Accepts the base config.
9. `test_hard_immediate_ablation` — `measure_density_hard=True`: k == I,
   t rows still on simplex, CE finite, `density_k_self` == 1.
10. `test_end_to_end_updates` — 30 `update()` steps: finite losses, w on
    the simplex, ESS ∈ [1, M], no NaN in params.

Runner: `.venv/bin/python -m pytest tests/test_gciql_measure_density.py -x -q`,
then the 17a/17b suites (mutual-exclusion asserts touched) — all green.

## 6. Eval — `scripts/rescuing_fb/eval_density_zeroshot.py`

Generalize `eval_partition_zeroshot.py` (import the agent's own
`_delta_phi`/`_density_queries`/tower modules — SAME code path as
training, the audit-verified pattern):

- Pool: P draws of object block coords from the relabel dataset
  (default P=4096, `--pool_size`), fresh per seed with a seeded rng;
  B-embeddings precomputed once.
- w over the pool, min over the 2 F-heads (`double_q` read from the
  checkpoint's config, not hard-coded).
- PRIMARY readout: `Q̂_r = (r(x₀) + (w − κ(x₀|U_P))ᵀ r(U_P))/(1−γ)`.
- SECONDARY readout (report both, quote separately): ridge-fit w_r of r
  on B(Ũ_P) (`--ridge 1e-4`), `Q̂_r = (r(x₀) + (μ − μ₀)ᵀw_r)/(1−γ)`.
- Families: dirac / dense / region — the spec-14-comparable retrieval
  policy decode, same as the part harness.
- `--pool_sweep 256,1024,4096,16384`: recompute the PRIMARY readout per
  pool size on the same checkpoint and report drift (the "M is a sample
  count" claim, tested).
- Consistency refusal: loud error when pointed at a non-density
  checkpoint (both directions, `--force` override) — the
  `check_delta_consistency` pattern.

Tests (`tests/test_eval_density_zeroshot.py`): numpy reference for both
readouts on a random net; pool-sweep smoke (drift finite, shrinking in
P on a synthetic ν); consistency refusals; linearity check
Q̂_{αr1+βr2} = αQ̂_{r1} + βQ̂_{r2} exact for PRIMARY.

## 7. Pre-registered wave (cube-single, from scratch, 500k, standing recipe)

All d50 slot settings as `cube_single_state_measure_delta_partition_d50.yaml`
(bc_coeff 3.0, eval_interval 50000, eval_episodes 50, wandb entity amsks,
tmux, XLA_PYTHON_CLIENT_PREALLOCATE=false):

- `d18_cs_dens`   seeds 0/1/2 — density (PRIMARY).
  Config `configs/gciql/cube_single_state_measure_density_d50.yaml`,
  run_group `GCIQL-CubeSingle-Flow-Measure-Density-d50`.
- `d18_cs_hard`   seed 0 — `measure_density_hard=True` (smoothing
  ablation). Distinct run_group `...-DensityHard-d50`.
- `d18_cs_partK{64,512,1024}` seed 0 — 17b part K-sweep, existing code,
  new protos via `make_delta_prototypes.py --k {64,512,1024}` (files
  registered by sha256 in the eval registry like K256). The
  "prototypes-are-not-load-bearing" control. Distinct run_groups.

Gates:
- G1 (density): success ≥ 0.5 at 500k; the real target is ≈ part's 0.884.
- G2 (health): ESS bounded away from 1 (pre-register: median
  `density_ess` ≥ 4 at 500k) and `density_w_entropy` not → 0 while
  success holds.
- G3 (continuity claim): PRIMARY readout stable across the pool sweep —
  per-task success change ≤ 0.05 from P=4096 to P=16384.
- G4 (ordering, carried from spec 17): Spearman of MC region masses vs
  read-out masses ≥ 0.65. Run it for part K256 too — it is still owed.

Pre-registered interpretations:
- density ≈ part, hard floors ⇒ the ε-kernel smoothing of the immediate
  term is load-bearing (dense supervision verdict).
- density floors while part holds at every K ⇒ the finite fixed support
  itself is load-bearing — report as a RESULT (discretization is a
  feature), do not patch; rank/temperature sweep (§3) before that claim.
- K-sweep flat in {64, 256, 512, 1024} ⇒ prototype count is not
  load-bearing (the paper's answer to "arbitrary discretization").
- ESS → 1 with success collapse ⇒ nearest-neighbour degeneration — the
  density analogue of part's entropy→log K signature; report, round-2
  territory for temperature/bandwidth, NOT a round-1 regularizer.
- Region-family zero-shot: expect improvement over part's 0.17–0.20 (the
  prototype interpolation error is the standing suspect there); no gate,
  directional read.

Post-collection: add every run_group to `scripts/results_map.py`, refresh
CSVs (`.venv/bin/python scripts/compile_results_csvs.py --refresh`) —
this is ALSO still owed for the d17_* groups (audit finding).

## 8. Anti-scope (do NOT add in round 1)

No entropy regularizer / temperature schedules / learned τ; no extra iid
query draws beyond the batch (knob later if coverage diagnostics demand);
no ortho, anchors, hinges, scale_norm; no pessimism beyond the 2-head
min; no z-conditioned actor (w_r stays eval-side); no GCDataset changes;
no warm start; no pixel path; no cube-double/scene configs (gated on G1,
§9); no generative/particle head (proposal §8 — reserve; that territory
is TD-Flows/Bellman-Diffusion adjacent and practically heavier).

## 9. Open questions carried forward

- w_r → actor (task-embedding conditioning): the true-zero-shot FB
  endgame, blocked on the stack-divergence z-conditioning verdict; would
  need its own deconfounded wave.
- Extra iid queries / replay queue of recent object coords (proposal §7)
  — only if `density_k_self`/ESS show coverage failure on conjunctive
  envs.
- cube-double/scene: conjunctive d_c=6/joint blocks — gated on G1; the
  batch-successor query set is expected to matter most there (0.046%
  conjunctive hit rate).
- Multiple kernel bandwidths / temperature sweep — round 2, only with a
  round-1 failure signature in hand.
- Joint-MLP score fallback (§3) — only on a rank/temperature-implicating
  failure.
