# Joint-ψ SF cells with FB orthonormalization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two successor-feature cells in which ψ is trained by the SF vector TD itself (rather than frozen, identity, or reward-regressed), held non-degenerate by FB's orthonormality penalty and √d-sphere normalization.

**Architecture:** Four new default-OFF knobs inside `gciql_flow.py::sf_loss` on the GCIQL trunk — `sf_joint_psi` (un-stop-grad the base so the TD reaches ψ), `sf_base_index` (`b=ψ(s)` vs `ψ(s′)`), `sf_mask` (goal mask vs `m≡1`), and `sf_psi_ortho_coef` (FB whitening on ψ). Plus threading the existing `project_z` sphere norm into `GCFrozenBValue`/`GCLearnedBValue`. Five `create()` guards encode the algebra so misconfigured cells fail loudly. Cell **A** (`b=ψ(s′)`, no mask) matches `C-fbz-projz-ortho1000` = 0.231±0.030 in every respect except the estimator; cell **C** (`b=ψ(s)`, goal mask, z-conditioned φ) computes a first-hitting value `γ^T`.

**Tech Stack:** JAX/Flax, OGBench GCIQL trunk, pytest via jax-subprocess harness, hydra configs, tmux + wandb.

**Spec:** `docs/superpowers/specs/2026-07-10-joint-sf-ortho-design.md`

## Global Constraints

- Python for tests: `.venv/bin/python -m pytest` (never `uv run`). JAX GPU venv is `/dev/shm/.venv-jax`; the test harness (`tests/jax_subprocess.py`) invokes it for you.
- Agent file: `tools/wandb_mode_shim/gciql_flow.py`. Tests: `tests/`. Configs: `configs/gciql/`.
- All new knobs default to OFF/current behaviour. **Every existing SF config must construct and train unchanged.** `tests/test_gciql_sf_feature_td.py` (16 tests) must stay green throughout.
- `sf_psi_ortho_coef` acts on **ψ**. The pre-existing `sf_ortho_coef` acts on **φ**. Never conflate them: φ-ortho at λ=1000 drove cells to 0.001–0.004.
- Commit messages: **omit** any `Co-Authored-By: Claude` trailer (repo convention).
- Commit only code/tests/configs/scripts. `PAPER/`, `results/`, `analysis/`, `docs/superpowers/` are gitignored.
- wandb entity is `amsks` (defaulted by `run_gciql.py:128`). Every training job runs in **tmux**.
- Launch scripts must set `PY` to the **repo** `.venv/bin/python` — `run_gciql.py` is a hydra launcher that subprocesses ogbench `main.py` into `.venv-jax` itself.
- **Never reuse a wandb `run_group`.** `results_map.py` keys on the group; reusing one silently averages old and new seeds.

---

## File Structure

| File | Responsibility |
|---|---|
| `tools/wandb_mode_shim/gciql_flow.py` | Modify: `GCFrozenBValue` (add `project_z` field, sphere in `_psi`); `sf_loss` (base index, mask mode, joint grad, ψ-ortho); `create()` (guards, `project_z` in `crit_kwargs`); `get_config()` (4 knobs). |
| `tests/test_gciql_sf_joint_psi.py` | Create: 7 TDD tests for the new knobs and guards. |
| `configs/gciql/cube_single_state_gciql_flow_sf_joint_ortho1k.yaml` | Create: cell A. |
| `configs/gciql/cube_single_state_gciql_flow_sf_joint_ortho1k_absorb.yaml` | Create: cell C. |
| `scripts/results_map.py` | Modify: register both wandb groups. |
| `scripts/launch_sf_joint.sh` | Create: 6 runs (2 cells × 3 seeds) @500k in tmux. |

---

## Two deviations from the spec (decided while writing this plan)

**1. Guard 3 is scoped to `sf_joint_psi`.** The spec says `sf_mask='goal' ⇒ fb_forward_z`. Enforced unconditionally this breaks *every existing SF config* (all use the goal mask with a goal-agnostic φ) including the E3 cells currently training. Guard 3 therefore fires only when `sf_joint_psi=True`. The audit's point-3 objection still stands for the older cells; it is documented, not retro-enforced.

**2. The ortho identity test asserts the exact finite-batch form.** The spec's `L_ort = ½‖C−I‖²_F − d/2` holds only as `B→∞` (the implemented `o_off` averages over ordered pairs `i≠j`, excluding the diagonal). The exact identity, with `G = ΨΨᵀ` and `C = ΨᵀΨ/B`, is

```
mean_{i≠j} ⟨ψᵢ,ψⱼ⟩²  =  ( B²·‖C‖²_F  −  Σᵢ‖ψᵢ‖⁴ ) / (B² − B)
o_diag                =  − tr(C)
```

because `‖G‖²_F = tr((ΨᵀΨ)²) = B²‖C‖²_F`. Test 2 asserts this exactly and documents the asymptotic form in a comment.

---

### Task 1: Thread `project_z` into the frozen/learned base classes

**Files:**
- Modify: `tools/wandb_mode_shim/gciql_flow.py:240-260` (`GCFrozenBValue`), `:911-914` (`crit_kwargs`), `get_config()`
- Test: `tests/test_gciql_sf_joint_psi.py`

**Interfaces:**
- Consumes: existing `_sphere(x, d)` helper; `GCFrozenBValue._psi`; `GCLearnedBValue` inherits both.
- Produces: `GCFrozenBValue.project_z: bool` field. When True, `_psi(goals)` returns `_sphere(psi(goals), latent_dim)`, so `‖ψ(·)‖ = √d` for every row, and the z fed to φ under `fb_forward_z` is also sphere-normed. `GCIdentityBValue` is **untouched** (its `_psi` returns raw goals; sphering them would destroy the identity base).

- [ ] **Step 1: Write the failing test**

Create `tests/test_gciql_sf_joint_psi.py` with the module docstring and this first test:

```python
"""Joint-psi SF cells (spec: docs/superpowers/specs/2026-07-10-joint-sf-ortho-design.md).

psi is trained by the SF vector TD itself (the base b(.) leaves the stop_gradient), and held
non-degenerate by FB's orthonormality penalty + sqrt(d)-sphere normalization -- instead of being
frozen, identity, or reward-regressed. Runs in the jax venv subprocess.
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def test_project_z_puts_psi_on_the_sqrt_d_sphere():
    """FB pins ||B|| = ||z|| = sqrt(d) via a Norm layer. With project_z the learned base leg has
    unit-sqrt(d) rows for EVERY input, so the ortho penalty's diagonal term is the constant -d
    (tr(C) = d) and only the off-diagonal decorrelation is active. Radial collapse (psi -> 0)
    becomes impossible by construction."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
        cfg["sf_feature_td"] = True; cfg["sf_base"] = "learned"
        cfg["sf_reward_regress"] = True          # keeps guard 5 satisfied for now
        cfg["fb_forward_z"] = True
        cfg["project_z"] = True
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (8, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (8, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        goals = jax.random.normal(jax.random.PRNGKey(3), (8, 11))
        p = agent.network.params
        _, _, psi = agent.network.select("critic")(ex_obs, goals, ex_act, goals, info=True, params=p)
        norms = jnp.linalg.norm(psi, axis=-1)               # [2,8]
        assert jnp.allclose(norms, jnp.sqrt(50.0), atol=1e-4), norms
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_gciql_sf_joint_psi.py::test_project_z_puts_psi_on_the_sqrt_d_sphere -v`
Expected: FAIL — `project_z` is not passed to `GCLearnedBValue`, so `psi` rows have arbitrary norms and `jnp.allclose(norms, 7.07)` is False.

- [ ] **Step 3: Add the `project_z` field and sphere the base leg**

In `gciql_flow.py`, in `class GCFrozenBValue`, add the field next to `fb_forward_z` and sphere inside `_psi`:

```python
class GCFrozenBValue(GCBilinearValue):
    ...
    fb_forward_z: bool = False
    project_z: bool = False     # FB Norm: hard-project psi (and the z fed to phi) onto the sqrt(d) sphere

    def setup(self):
        m = MLP
        if self.ensemble:
            m = ensemblize(m, 2)
        self.phi = m((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.psi = m((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def _psi(self, goals):
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)
        psi = jax.lax.stop_gradient(self.psi(goals))   # FROZEN: zero grad -> stays at random init
        return _sphere(psi, self.latent_dim) if self.project_z else psi
```

And in `class GCLearnedBValue`:

```python
    def _psi(self, goals):
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)
        psi = self.psi(goals)   # LEARNED: grads arrive via sf_reward_regress and/or sf_joint_psi
        return _sphere(psi, self.latent_dim) if self.project_z else psi
```

In `create()`, extend the kwarg line at `gciql_flow.py:911-912`:

```python
            if critic_cls in (GCIdentityBValue, GCFrozenBValue, GCLearnedBValue):
                crit_kwargs['fb_forward_z'] = config.get('fb_forward_z', False)
            if critic_cls in (GCFrozenBValue, GCLearnedBValue):
                # NOT GCIdentityBValue: its _psi returns the raw goal, and sphering it would
                # destroy the identity base b(g)=g.
                crit_kwargs['project_z'] = config.get('project_z', False)
```

- [ ] **Step 4: Run the test and make sure it passes**

Run: `.venv/bin/python -m pytest tests/test_gciql_sf_joint_psi.py -v`
Expected: 1 passed.

- [ ] **Step 5: Verify no regression in the existing SF suite**

Run: `.venv/bin/python -m pytest tests/test_gciql_sf_feature_td.py -q`
Expected: `16 passed`.

- [ ] **Step 6: Commit**

```bash
git add tools/wandb_mode_shim/gciql_flow.py tests/test_gciql_sf_joint_psi.py
git commit -m "feat(gciql): thread project_z into the frozen/learned SF base legs

FB pins ||B||=||z||=sqrt(d) via a Norm layer. project_z was only reaching
GCFBValue/GCTwoTowerCritic; the SF base classes never saw it. Sphering the learned base makes
radial psi->0 collapse impossible by construction and fixes tr(C)=d, so the ortho penalty's
diagonal term becomes constant and only the off-diagonal decorrelation is active.

GCIdentityBValue is deliberately excluded: its _psi returns the raw goal."
```

---

### Task 2: ψ-orthonormality penalty (`sf_psi_ortho_coef`)

**Files:**
- Modify: `tools/wandb_mode_shim/gciql_flow.py::sf_loss` (after the existing `sf_ortho_coef` block), `get_config()`
- Test: `tests/test_gciql_sf_joint_psi.py`

**Interfaces:**
- Consumes: `GCLearnedBValue` with `project_z` (Task 1).
- Produces: config knob `sf_psi_ortho_coef: float = 0.0`. When `> 0`, `sf_loss` adds `sf_psi_ortho_coef * orth_psi` where `orth_psi` is FB's penalty on `ψ(next_observations)`, per ensemble head then averaged. New info keys: `psi_ortho`, `psi_ortho_off`, `psi_ortho_diag`, `psi_norm_mean`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gciql_sf_joint_psi.py`:

```python
def test_psi_ortho_matches_the_exact_finite_batch_fb_formula():
    """DECISIVE. sf_psi_ortho_coef applies FB's orthonormality penalty to psi(next_observations)
    -- the rho marginal, exactly as measure_loss does (B := psi(s')), NOT to the HER value_goals.

    Implemented form (per head, then averaged):
        o_off  = 0.5 * mean_{i!=j} <psi_i, psi_j>^2
        o_diag = - mean_i ||psi_i||^2
    With G = Psi Psi^T and C = Psi^T Psi / B, ||G||_F^2 = tr((Psi^T Psi)^2) = B^2 ||C||_F^2, so

        mean_{i!=j} <psi_i,psi_j>^2 = ( B^2 ||C||_F^2 - sum_i ||psi_i||^4 ) / (B^2 - B)
        o_diag                      = - tr(C)

    exactly. (Asymptotically o_off + o_diag -> 0.5||C - I||_F^2 - d/2, which is why minimizing it
    whitens psi; the finite-batch form above is what the code actually computes.)"""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
        cfg["sf_feature_td"] = True; cfg["sf_base"] = "learned"
        cfg["sf_reward_regress"] = True; cfg["sf_reward_coef"] = 0.0
        cfg["fb_forward_z"] = True
        cfg["sf_psi_ortho_coef"] = 3.0
        cfg["discount"] = 0.0
        B = 12
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
        ex_nobs = jax.random.normal(jax.random.PRNGKey(4), (B, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
        goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_nobs, actions=ex_act,
                     rewards=jnp.zeros((B,)), masks=jnp.ones((B,)),
                     value_goals=goals, actor_goals=goals)
        loss, info = agent.sf_loss(batch, agent.network.params, jax.random.PRNGKey(2))
        p = agent.network.params
        _, _, psi_o = agent.network.select("critic")(ex_obs, ex_nobs, ex_act, goals, info=True, params=p)
        # exact finite-batch identity, per head then averaged
        G = jnp.einsum('hid,hjd->hij', psi_o, psi_o)                  # [2,B,B]
        sq_norms = jnp.diagonal(G, axis1=1, axis2=2)                  # [2,B] = ||psi_i||^2
        C = jnp.einsum('hid,hie->hde', psi_o, psi_o) / B              # [2,d,d]
        c_fro2 = jnp.sum(C ** 2, axis=(1, 2))                         # [2] = ||C||_F^2
        mean_off = (B * B * c_fro2 - jnp.sum(sq_norms ** 2, axis=1)) / (B * B - B)
        exp_orth = jnp.mean(0.5 * mean_off - jnp.mean(sq_norms, axis=1))
        assert jnp.allclose(info["psi_ortho"], exp_orth, rtol=1e-4), (info["psi_ortho"], exp_orth)
        # and it is the term added to the loss (discount=0, reward_coef=0 -> loss = td + 3*orth)
        _, phi, _ = agent.network.select("critic")(ex_obs, goals, ex_act, goals, info=True, params=p)
        _, _, b = agent.network.select("critic")(ex_obs, ex_obs, ex_act, goals, info=True, params=p)
        exp_td = jnp.mean((phi - b) ** 2)
        assert jnp.allclose(loss, exp_td + 3.0 * exp_orth, rtol=1e-4), (loss, exp_td, exp_orth)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_psi_ortho_diag_is_constant_minus_d_under_project_z():
    """Under project_z every psi row has norm sqrt(d), so tr(C) = d and the ortho DIAGONAL term is
    the constant -d for any batch. The sphere kills the radial collapse; only the off-diagonal
    (angular) decorrelation term still has a gradient."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        def build_info(seed):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
            cfg["sf_feature_td"] = True; cfg["sf_base"] = "learned"
            cfg["sf_reward_regress"] = True; cfg["sf_reward_coef"] = 0.0
            cfg["fb_forward_z"] = True; cfg["project_z"] = True
            cfg["sf_psi_ortho_coef"] = 1.0
            B = 10
            ex_obs = jax.random.normal(jax.random.PRNGKey(seed), (B, 11))
            ex_nobs = jax.random.normal(jax.random.PRNGKey(seed + 100), (B, 11))
            ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
            goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
            agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
            batch = dict(observations=ex_obs, next_observations=ex_nobs, actions=ex_act,
                         rewards=jnp.zeros((B,)), masks=jnp.ones((B,)),
                         value_goals=goals, actor_goals=goals)
            _, info = agent.sf_loss(batch, agent.network.params, jax.random.PRNGKey(2))
            return info
        i0, i1 = build_info(0), build_info(7)
        for i in (i0, i1):
            assert jnp.allclose(i["psi_ortho_diag"], -50.0, atol=1e-3), i["psi_ortho_diag"]
            assert jnp.allclose(i["psi_norm_mean"], jnp.sqrt(50.0), atol=1e-3), i["psi_norm_mean"]
        assert not jnp.allclose(i0["psi_ortho_off"], i1["psi_ortho_off"]), "off-diag must vary"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/bin/python -m pytest tests/test_gciql_sf_joint_psi.py -k "psi_ortho" -v`
Expected: both FAIL with `KeyError: 'psi_ortho'` — the info key does not exist.

- [ ] **Step 3: Add the ψ-ortho term**

In `sf_loss`, immediately **after** the existing `sf_ortho_coef` (φ-ortho) block and **before** the `sf_reward_regress` block, insert:

```python
        lam_psi = self.config.get('sf_psi_ortho_coef', 0.0)
        if lam_psi > 0:
            # FB orthonormality ||E_rho[psi psi^T] - I|| on the BASE/goal leg psi -- the same
            # penalty and the same distribution measure_loss uses (B := psi(s'), the rho marginal,
            # NOT the HER value_goals). This is what keeps a jointly-trained psi non-degenerate:
            # un-stop-gradding the base makes phi=0, psi=0 a GLOBAL minimum of the TD, and the
            # -tr(C) term below turns the origin into a strict local maximum (repelling).
            # Distinct from sf_ortho_coef, which penalizes phi.
            _, _, psi_o = self.network.select('critic')(obs, nobs, acts, *zarg, info=True, params=grad_params)
            Bn = obs.shape[0]; off_p = 1.0 - jnp.eye(Bn)
            CovP = jnp.einsum('hid,hjd->hij', psi_o, psi_o)                 # [2,B,B]
            p_off = 0.5 * jnp.sum((CovP * off_p[None]) ** 2, axis=(1, 2)) / jnp.sum(off_p)
            p_diag = -jnp.mean(jnp.diagonal(CovP, axis1=1, axis2=2), axis=1)
            orth_psi = jnp.mean(p_off + p_diag)
            loss = loss + lam_psi * orth_psi
            info['psi_ortho'] = orth_psi
            info['psi_ortho_off'] = jnp.mean(p_off)
            info['psi_ortho_diag'] = jnp.mean(p_diag)
            info['psi_norm_mean'] = jnp.mean(jnp.linalg.norm(psi_o, axis=-1))
```

In `get_config()`, add below `config.sf_ortho_coef`:

```python
        config.sf_psi_ortho_coef = 0.0  # FB orthonormality ||E_rho[psi psi^T]-I|| on the BASE leg psi(s') (FB uses 1000). DISTINCT from sf_ortho_coef, which penalizes phi. Required by sf_joint_psi (else phi=0,psi=0 is a global min of the TD)
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/test_gciql_sf_joint_psi.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/wandb_mode_shim/gciql_flow.py tests/test_gciql_sf_joint_psi.py
git commit -m "feat(gciql): sf_psi_ortho_coef -- FB orthonormality on the SF base leg psi(s')

Penalizes ||E_rho[psi psi^T] - I|| on psi(next_observations), the same term and the same rho
marginal measure_loss uses (B := psi(s')), NOT the HER value_goals that ortho_loss uses.

Distinct knob from sf_ortho_coef, which penalizes phi (and at lambda=1000 drove SF cells to
0.001-0.004). Logs psi_ortho / psi_ortho_off / psi_ortho_diag / psi_norm_mean.

Test asserts the EXACT finite-batch identity
  mean_{i!=j}<psi_i,psi_j>^2 = (B^2||C||_F^2 - sum_i ||psi_i||^4)/(B^2-B),  o_diag = -tr(C)
(the 0.5||C-I||_F^2 - d/2 form in the spec is its B->inf limit), and that under project_z the
diagonal term is the constant -d."
```

---

### Task 3: `sf_base_index`, `sf_mask`, and `sf_joint_psi`

**Files:**
- Modify: `tools/wandb_mode_shim/gciql_flow.py::sf_loss` (lines 627-654 region), `get_config()`
- Test: `tests/test_gciql_sf_joint_psi.py`

**Interfaces:**
- Consumes: `sf_psi_ortho_coef` (Task 2) — the joint cells require it.
- Produces: knobs `sf_joint_psi: bool = False`, `sf_base_index: str = 'current'` (`'current'|'next'`), `sf_mask: str = 'goal'` (`'goal'|'none'`). With defaults, `sf_loss` returns exactly today's value.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gciql_sf_joint_psi.py`:

```python
def test_cell_A_gamma0_target_is_psi_of_next_state():
    """DECISIVE for cell A. sf_base_index='next' makes the immediate feature b = psi(s'), FB's
    arriving-state convention (its measure counts arrival at s'). At gamma=0 the bootstrap
    vanishes, so the TD term must equal mean((phi - psi(s'))^2) exactly -- NOT psi(s)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
        cfg["sf_feature_td"] = True; cfg["sf_base"] = "learned"
        cfg["sf_joint_psi"] = True; cfg["sf_psi_ortho_coef"] = 2.0
        cfg["sf_base_index"] = "next"; cfg["sf_mask"] = "none"
        cfg["fb_forward_z"] = True; cfg["project_z"] = True
        cfg["discount"] = 0.0
        B = 9
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
        ex_nobs = jax.random.normal(jax.random.PRNGKey(4), (B, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
        goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_nobs, actions=ex_act,
                     rewards=jnp.zeros((B,)), masks=jnp.ones((B,)),
                     value_goals=goals, actor_goals=goals)
        loss, info = agent.sf_loss(batch, agent.network.params, jax.random.PRNGKey(2))
        p = agent.network.params
        _, phi, _ = agent.network.select("critic")(ex_obs, goals, ex_act, goals, info=True, params=p)
        _, _, psi_ns = agent.network.select("critic")(ex_obs, ex_nobs, ex_act, goals, info=True, params=p)
        _, _, psi_s  = agent.network.select("critic")(ex_obs, ex_obs,  ex_act, goals, info=True, params=p)
        exp_td = jnp.mean((phi - psi_ns) ** 2)
        assert jnp.allclose(info["sf_loss"], exp_td, rtol=1e-5), (info["sf_loss"], exp_td)
        wrong = jnp.mean((phi - psi_s) ** 2)
        assert not jnp.allclose(exp_td, wrong), "test vacuous: psi(s) == psi(s')"
        assert jnp.allclose(loss, exp_td + 2.0 * info["psi_ortho"], rtol=1e-4)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_sf_mask_none_ignores_the_goal_mask_and_goal_keeps_it():
    """Cell A computes a visitation density Q = sum_t gamma^t 1[s_t=g], so the goal-dependent
    absorbing mask must be GONE (m == 1). FB's measure_loss has zero references to batch['masks'].
    Randomizing masks must not move the loss under sf_mask='none', and MUST move it under 'goal'."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        def loss_for(mask_mode, masks):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
            cfg["sf_feature_td"] = True; cfg["sf_base"] = "learned"
            cfg["sf_joint_psi"] = True; cfg["sf_psi_ortho_coef"] = 1.0
            cfg["sf_base_index"] = "next"; cfg["sf_mask"] = mask_mode
            cfg["fb_forward_z"] = True; cfg["discount"] = 0.99
            B = 9
            ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
            ex_nobs = jax.random.normal(jax.random.PRNGKey(4), (B, 11))
            ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
            goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
            agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
            batch = dict(observations=ex_obs, next_observations=ex_nobs, actions=ex_act,
                         rewards=jnp.zeros((B,)), masks=masks,
                         value_goals=goals, actor_goals=goals)
            _, info = agent.sf_loss(batch, agent.network.params, jax.random.PRNGKey(2))
            return info["sf_loss"]
        ones = jnp.ones((9,)); zeros = jnp.zeros((9,))
        # sf_mask='none' but base_index='next' -> guard 4 satisfied; masks must be inert
        assert jnp.allclose(loss_for("none", ones), loss_for("none", zeros), rtol=1e-6)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_cell_C_goal_mask_is_live_and_absorbs_at_the_goal():
    """Cell C keeps the absorbing mask (Q = gamma^T, a first-hitting value) with a z-conditioned
    phi. Flipping masks 1->0 must change the loss: with m=0 the bootstrap is cut."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        def loss_for(masks):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
            cfg["sf_feature_td"] = True; cfg["sf_base"] = "learned"
            cfg["sf_joint_psi"] = True; cfg["sf_psi_ortho_coef"] = 1.0
            cfg["sf_base_index"] = "current"; cfg["sf_mask"] = "goal"
            cfg["fb_forward_z"] = True; cfg["discount"] = 0.99
            B = 9
            ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
            ex_nobs = jax.random.normal(jax.random.PRNGKey(4), (B, 11))
            ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
            goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
            agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
            batch = dict(observations=ex_obs, next_observations=ex_nobs, actions=ex_act,
                         rewards=jnp.zeros((B,)), masks=masks,
                         value_goals=goals, actor_goals=goals)
            _, info = agent.sf_loss(batch, agent.network.params, jax.random.PRNGKey(2))
            return info["sf_loss"]
        assert not jnp.allclose(loss_for(jnp.ones((9,))), loss_for(jnp.zeros((9,)))), \\
            "sf_mask='goal' must use batch['masks']"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_joint_psi_sends_td_gradient_into_psi_and_off_does_not():
    """sf_joint_psi un-stop-grads the base, so the TD residual reaches psi. Control: with
    sf_joint_psi=False and sf_reward_coef=0 (reward regression weighted to zero), psi has NO
    gradient path at all through sf_loss -- the base is stop-gradded and the actor Q-term takes
    no critic grads. Tiny ortho coef in the joint arm so the TD dominates the psi gradient."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        def psi_grad_norm(joint):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
            cfg["sf_feature_td"] = True; cfg["sf_base"] = "learned"
            cfg["fb_forward_z"] = True; cfg["discount"] = 0.99
            if joint:
                cfg["sf_joint_psi"] = True; cfg["sf_psi_ortho_coef"] = 1e-9
                cfg["sf_base_index"] = "next"; cfg["sf_mask"] = "none"
            else:
                cfg["sf_reward_regress"] = True; cfg["sf_reward_coef"] = 0.0
            B = 9
            ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
            ex_nobs = jax.random.normal(jax.random.PRNGKey(4), (B, 11))
            ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
            goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
            agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
            batch = dict(observations=ex_obs, next_observations=ex_nobs, actions=ex_act,
                         rewards=jnp.zeros((B,)), masks=jnp.ones((B,)),
                         value_goals=goals, actor_goals=goals)
            g = jax.grad(lambda p: agent.sf_loss(batch, p, jax.random.PRNGKey(2))[0])(agent.network.params)
            leaves = jax.tree_util.tree_leaves(g["modules_critic"]["psi"])
            return sum(float(jnp.sum(jnp.abs(l))) for l in leaves)
        gj, gn = psi_grad_norm(True), psi_grad_norm(False)
        assert gj > 1e-3, ("joint psi must receive TD gradient", gj)
        assert gn < 1e-8, ("non-joint psi must receive NO sf_loss gradient", gn)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_defaults_reproduce_todays_sf_loss():
    """BACKWARDS COMPAT. With every new knob at its default (joint off, base 'current', mask
    'goal', no psi ortho) sf_loss must equal the pre-existing expression. gamma=0 pins the target
    to b(s)=psi(s), the same decisive form the legacy suite uses."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        assert cfg["sf_joint_psi"] is False and cfg["sf_base_index"] == "current"
        assert cfg["sf_mask"] == "goal" and cfg["sf_psi_ortho_coef"] == 0.0
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
        cfg["sf_feature_td"] = True; cfg["sf_base"] = "frozen"
        cfg["discount"] = 0.0
        B = 7
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
        goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((B,)), masks=jnp.ones((B,)),
                     value_goals=goals, actor_goals=goals)
        loss, info = agent.sf_loss(batch, agent.network.params, jax.random.PRNGKey(2))
        p = agent.network.params
        _, phi, _ = agent.network.select("critic")(ex_obs, goals, ex_act, info=True, params=p)
        _, _, b = agent.network.select("critic")(ex_obs, ex_obs, ex_act, info=True, params=p)
        assert jnp.allclose(loss, jnp.mean((phi - b) ** 2), rtol=1e-5)
        assert "psi_ortho" not in info
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/bin/python -m pytest tests/test_gciql_sf_joint_psi.py -k "cell_A or sf_mask or cell_C or joint_psi_sends or defaults" -v`
Expected: FAIL — `KeyError: 'sf_joint_psi'` in the defaults test, and the others fail because `sf_base_index`/`sf_mask` are ignored so `b` is always `psi(s)` and the mask is always live.

- [ ] **Step 3: Rewrite the `sf_loss` head**

Replace `gciql_flow.py:627-654` (from `obs, nobs, acts = ...` through the `info = {...}` line) with:

```python
        obs, nobs, acts = batch['observations'], batch['next_observations'], batch['actions']
        vgoals = batch['value_goals']
        gamma = self.config['discount']
        # sf_mask: 'goal' -> batch['masks'] = 1[s != g], an ABSORBING goal (Q = gamma^T, a
        # first-hitting value). 'none' -> m == 1, no absorption (Q = sum_t gamma^t 1[s_t=g], a
        # visitation DENSITY -- FB's object; measure_loss has no mask at all).
        mask = batch['masks'] if self.config.get('sf_mask', 'goal') == 'goal' else jnp.ones_like(batch['masks'])
        # sf_base_index: 'current' -> b(s)=psi(s), forced by GCDataset tying rewards AND masks to
        # the current index. 'next' -> b(s)=psi(s'), FB's arriving-state convention (its measure
        # counts arrival at s'). Never mix the two (guard 4).
        base_next = self.config.get('sf_base_index', 'current') == 'next'
        joint = self.config.get('sf_joint_psi', False)
        # fb_forward_z: task-condition phi([s,a,z=psi(g)]) -> thread z_goals=vgoals through the
        # phi, base-feature, and target-critic calls (same task g on both sides of the SF backup).
        zarg = (vgoals,) if self.config.get('fb_forward_z', False) else ()
        _, phi, _psi = self.network.select('critic')(obs, vgoals, acts, *zarg, info=True, params=grad_params)  # [2,B,d]
        if self.config.get('sf_base', 'identity') in ('frozen', 'learned'):
            # b = psi(s) or psi(s'): call psi via b_goals. Stop-gradded UNLESS sf_joint_psi, in
            # which case the TD residual trains psi through the base (held non-degenerate by
            # sf_psi_ortho_coef -- without it phi=0, psi=0 is a global minimum).
            b_goals = nobs if base_next else obs
            _, _, b = self.network.select('critic')(obs, b_goals, acts, *zarg, info=True, params=grad_params)  # [2,B,d]
            if not joint:
                b = jax.lax.stop_gradient(b)
        else:
            b = (nobs if base_next else obs)[None]                   # identity base b(s)=s -> [1,B,d]
        if self.config['actor_kind'] == 'ddpgbc':
            # stock gaussian actor: a' = mode of pi(.|s',g) (or a sample if learned std).
            dist = self.network.select('actor')(nobs, vgoals)
            a_next = dist.mode() if self.config['const_std'] else dist.sample(seed=rng)
        else:
            noise = jax.random.normal(rng, acts.shape)
            a_next = self.network.select('actor')(nobs, vgoals, noise)          # a' ~ pi(s',g)
        a_next = jax.lax.stop_gradient(jnp.clip(a_next, -1, 1))
        _, tphi, _ = self.network.select('target_critic')(nobs, vgoals, a_next, *zarg, info=True)  # [2,B,d]
        boot = gamma * mask[None, :, None] * jax.lax.stop_gradient(tphi)                      # [2,B,d]
        # b carries gradient iff sf_joint_psi; the bootstrap is always semi-gradient (FB does the
        # same). With joint=False this is identical to the old sg[b + gamma*m*tphi] target.
        target = b + boot
        td = jnp.mean((phi - target) ** 2)
        loss = td
        info = {'sf_loss': td, 'phi_absmax': jnp.max(jnp.abs(phi)),
                'target_absmax': jnp.max(jnp.abs(jax.lax.stop_gradient(target)))}
```

In `get_config()`, add below `config.sf_psi_ortho_coef`:

```python
        config.sf_joint_psi = False   # train psi by the SF vector TD itself (un-stop-grad the base b). Requires sf_base='learned' + sf_psi_ortho_coef>0; mutually exclusive with sf_reward_regress
        config.sf_base_index = 'current'  # immediate feature: 'current' -> b(s)=psi(s) (GCDataset ties rewards+masks to the current index) | 'next' -> b(s)=psi(s') (FB's arriving-state measure convention). Never mix 'next' with sf_mask='goal'
        config.sf_mask = 'goal'       # bootstrap gate: 'goal' -> batch['masks'], absorbing goal (Q=gamma^T, hitting time) | 'none' -> m==1 (Q=sum gamma^t 1[s_t=g], visitation density; FB's measure_loss has no mask)
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/test_gciql_sf_joint_psi.py -v`
Expected: 8 passed. (Tests 4–8 need guard 5 relaxed; if `test_cell_A_...` fails with `AssertionError: sf_base='learned' requires sf_reward_regress=True`, proceed to Task 4 and re-run — the guards and these tests land together. To keep the cycle honest, temporarily relax that assert to `or config.get('sf_joint_psi', False)` now and finish it properly in Task 4.)

- [ ] **Step 5: Verify no regression**

Run: `.venv/bin/python -m pytest tests/test_gciql_sf_feature_td.py -q`
Expected: `16 passed`.

- [ ] **Step 6: Commit**

```bash
git add tools/wandb_mode_shim/gciql_flow.py tests/test_gciql_sf_joint_psi.py
git commit -m "feat(gciql): sf_joint_psi + sf_base_index + sf_mask

sf_joint_psi un-stop-grads the base b so the SF vector TD trains psi as well as phi -- the one
gradient path psi has never had (it was frozen, identity, or reward-regressed).

sf_base_index selects the immediate feature: 'current' b=psi(s) (GCDataset ties rewards AND masks
to the current index, datasets.py:242-244) or 'next' b=psi(s') (FB's arriving-state convention;
its measure counts arrival at s'). sf_mask drops the absorbing goal mask, turning Q from a
first-hitting value gamma^T into a visitation density sum_t gamma^t 1[s_t=g] -- FB's object, and
measure_loss indeed has no mask.

Defaults reproduce today's loss bit-for-bit (test included). 16/16 legacy SF tests still green."
```

---

### Task 4: The five guards

**Files:**
- Modify: `tools/wandb_mode_shim/gciql_flow.py::create()` lines 814-835 region
- Test: `tests/test_gciql_sf_joint_psi.py`

**Interfaces:**
- Consumes: all four knobs from Tasks 1–3.
- Produces: `create()` raises `AssertionError` on five invalid combinations; the two valid cells construct.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gciql_sf_joint_psi.py`:

```python
def test_joint_psi_guards_reject_invalid_combinations():
    """The guards encode the algebra, so a misconfigured cell fails loudly instead of producing an
    uninterpretable number -- the failure mode this program keeps repeating.

    1. joint => sf_base='learned' and NOT sf_reward_regress  (exactly one grad path into psi)
    2. joint => sf_psi_ortho_coef > 0   (else phi=0, psi=0 is a GLOBAL minimum of the TD)
    3. joint and sf_mask='goal' => fb_forward_z  (a goal-dependent absorption coefficient cannot
       sit in a goal-agnostic recursion). Scoped to joint: every legacy SF cell is agnostic+masked.
    4. NOT (sf_base_index='next' and sf_mask='goal')  (mixes the arriving-state and current-index
       conventions -- the exact class of bug fixed in E3 on 2026-07-09)
    5. sf_base='learned' => sf_reward_regress OR sf_joint_psi  (a learned psi with no signal)"""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (6, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (6, 5))
        def build(**over):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
            cfg["sf_feature_td"] = True
            cfg.update(over)
            return GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        J = dict(sf_base="learned", sf_joint_psi=True, sf_psi_ortho_coef=1000.0,
                 fb_forward_z=True, project_z=True)
        bad = [
            dict(J, sf_base="frozen"),                                   # 1: joint on a fixed base
            dict(J, sf_reward_regress=True),                             # 1: two grad paths
            dict(J, sf_psi_ortho_coef=0.0),                              # 2: guaranteed collapse
            dict(J, sf_mask="goal", sf_base_index="current", fb_forward_z=False),  # 3
            dict(J, sf_base_index="next", sf_mask="goal"),               # 4: mixed conventions
            dict(sf_base="learned"),                                     # 5: no signal at all
        ]
        for over in bad:
            try:
                build(**over)
            except AssertionError:
                continue
            raise SystemExit("guard missing for " + repr(over))
        build(**J, sf_base_index="next", sf_mask="none")       # cell A
        build(**J, sf_base_index="current", sf_mask="goal")    # cell C
        build(sf_base="frozen")                                # legacy cell unaffected
        build(sf_base="learned", sf_reward_regress=True)       # E3 cell unaffected
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_gciql_sf_joint_psi.py::test_joint_psi_guards_reject_invalid_combinations -v`
Expected: FAIL with `SystemExit: guard missing for ...`.

- [ ] **Step 3: Replace the E3 guard block with the five guards**

In `create()`, replace the block at `gciql_flow.py:824-833` (from the `# E3 guards:` comment through the `assert config.get('sf_reward_regress', False), ...` message) with:

```python
            # E3 guard: reward regression on a FIXED base is the audited confound (the linear
            # proxy r=<s,g> cannot express the goal indicator) -> refuse.
            if config.get('sf_reward_regress', False):
                assert config.get('sf_base', 'identity') == 'learned', \
                    "sf_reward_regress requires sf_base='learned' (fixed bases can't express the goal indicator)"
            joint = config.get('sf_joint_psi', False)
            if joint:
                # (1) exactly ONE gradient path into psi.
                assert config.get('sf_base', 'identity') == 'learned', \
                    "sf_joint_psi requires sf_base='learned' (a fixed psi has no gradient to receive)"
                assert not config.get('sf_reward_regress', False), \
                    "sf_joint_psi and sf_reward_regress are mutually exclusive (two grad paths into psi)"
                # (2) without the ortho penalty, phi=0 and psi=0 is a GLOBAL minimum of the TD.
                assert config.get('sf_psi_ortho_coef', 0.0) > 0, \
                    "sf_joint_psi requires sf_psi_ortho_coef>0 (else phi=0,psi=0 is a global min of the SF TD)"
                # (3) a goal-dependent absorption coefficient cannot sit in a goal-agnostic
                # recursion. Scoped to joint cells: the legacy SF grid is agnostic AND masked,
                # and is left untouched (the objection is documented, not retro-enforced).
                if config.get('sf_mask', 'goal') == 'goal':
                    assert config.get('fb_forward_z', False), \
                        "sf_joint_psi with sf_mask='goal' requires fb_forward_z (goal-dependent mask needs a goal-conditioned phi)"
            # (4) never mix the arriving-state base with the current-index mask: GCDataset ties
            # rewards AND masks to the current index, so b=psi(s') + m_g(s) is off-by-one against
            # the mask (the class of bug fixed in E3, 2026-07-09).
            assert not (config.get('sf_base_index', 'current') == 'next'
                        and config.get('sf_mask', 'goal') == 'goal'), \
                "sf_base_index='next' requires sf_mask='none' (arriving-state base + current-index mask mixes conventions)"
            # (5) a learned psi with no training signal at all.
            if config.get('sf_base', 'identity') == 'learned':
                assert config.get('sf_reward_regress', False) or joint, \
                    "sf_base='learned' requires sf_reward_regress=True or sf_joint_psi=True (else psi has no training signal)"
```

- [ ] **Step 4: Run the test and make sure it passes**

Run: `.venv/bin/python -m pytest tests/test_gciql_sf_joint_psi.py -v`
Expected: 9 passed.

- [ ] **Step 5: Run the whole affected suite**

Run: `.venv/bin/python -m pytest tests/test_gciql_sf_joint_psi.py tests/test_gciql_sf_feature_td.py tests/test_gciql_psi_base.py tests/test_gciql_fb_forward_z.py -q`
Expected: `9 + 16 + 4 + N passed`, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add tools/wandb_mode_shim/gciql_flow.py tests/test_gciql_sf_joint_psi.py
git commit -m "feat(gciql): five guards for the joint-psi SF cells

Encode the algebra so a misconfigured cell fails loudly rather than producing an uninterpretable
number:
 1. joint => sf_base='learned' and NOT sf_reward_regress (exactly one grad path into psi)
 2. joint => sf_psi_ortho_coef > 0 (else phi=0,psi=0 is a GLOBAL min of the TD, not a bad local one)
 3. joint + sf_mask='goal' => fb_forward_z (goal-dependent mask needs a goal-conditioned phi)
 4. never sf_base_index='next' with sf_mask='goal' (mixes arriving-state and current-index
    conventions -- the class of bug fixed in E3)
 5. sf_base='learned' => sf_reward_regress or sf_joint_psi (a learned psi with no signal)

Guard 3 is scoped to joint cells on purpose: every legacy SF cell is goal-agnostic AND masked, so
enforcing it globally would break the running E3 runs. The objection is documented, not
retro-enforced."
```

---

### Task 5: Configs, results_map, launch script

**Files:**
- Create: `configs/gciql/cube_single_state_gciql_flow_sf_joint_ortho1k.yaml`, `configs/gciql/cube_single_state_gciql_flow_sf_joint_ortho1k_absorb.yaml`, `scripts/launch_sf_joint.sh`
- Modify: `scripts/results_map.py`

**Interfaces:**
- Consumes: all knobs and guards from Tasks 1–4.
- Produces: wandb groups `GCIQL-State-Flow-SF-JointPsi-Ortho1k` (cell A) and `GCIQL-State-Flow-SF-JointPsi-Ortho1k-Absorb` (cell C). Both registered in `results_map.MAP` with `csv='bridge'`.

- [ ] **Step 1: Write cell A's config**

Create `configs/gciql/cube_single_state_gciql_flow_sf_joint_ortho1k.yaml`:

```yaml
# Cell A (sf-design-space 1a): the FB successor-measure objective estimated by the PER-SAMPLE
# vector TD instead of the BxB measure matrix. Matches
# cube_single_state_gciql_flow_bilinear_d50_C_proj.yaml (GCIQL-State-Flow-Bilinear-d50-C-FBzProjZ-O1000,
# 0.231 +/- 0.030) in EVERY respect except the estimator:
#   - psi trained by the SF TD itself (sf_joint_psi), held non-degenerate by ortho 1000 + sphere
#   - b(s) = psi(s'), FB's arriving-state convention
#   - NO absorbing mask: Q = sum_t gamma^t 1[s_t=g], a visitation density (FB's object)
# Reads as: does the estimator (per-sample diagonal vs BxB) matter, at matched basis-grounding?
defaults:
  - storage: shm
  - _self_
variant: state
env_name: cube-single-play-v0
agent_file: tools/wandb_mode_shim/gciql_flow.py
seed: 0
train_steps: 500000
log_interval: 5000
eval_interval: 100000
save_interval: 500000
eval_episodes: 50
eval_on_cpu: 1
run_group: GCIQL-State-Flow-SF-JointPsi-Ortho1k
restore_path: null
restore_epoch: null
exp_name: null
agent_overrides:
  bc_coeff: 3.0
  value_arch: bilinear
  value_latent_dim: 50
  sf_feature_td: true
  sf_coef: 1.0
  sf_base: learned
  sf_joint_psi: true
  sf_base_index: next
  sf_mask: none
  sf_psi_ortho_coef: 1000.0
  fb_forward_z: true
  project_z: true
wandb_mode: online
```

- [ ] **Step 2: Write cell C's config**

Create `configs/gciql/cube_single_state_gciql_flow_sf_joint_ortho1k_absorb.yaml`:

```yaml
# Cell C (sf-design-space 1a): the joint-psi SF cell that KEEPS the absorbing goal mask, so
#   Q = gamma^T   (discounted first-hitting time)
# rather than cell A's visitation density sum_t gamma^t 1[s_t=g]. No FB counterpart -- FB's
# measure_loss has no mask. Asks whether absorption is the better goal-reaching object.
# Coherent ONLY with fb_forward_z: a goal-dependent absorption coefficient cannot sit in a
# goal-agnostic recursion (guard 3). Current-index base b(s)=psi(s), matching the mask's index.
# Ortho is still on psi(s') so the penalty is IDENTICAL to cell A's and only the recursion differs
# (costs one extra critic forward per step).
defaults:
  - storage: shm
  - _self_
variant: state
env_name: cube-single-play-v0
agent_file: tools/wandb_mode_shim/gciql_flow.py
seed: 0
train_steps: 500000
log_interval: 5000
eval_interval: 100000
save_interval: 500000
eval_episodes: 50
eval_on_cpu: 1
run_group: GCIQL-State-Flow-SF-JointPsi-Ortho1k-Absorb
restore_path: null
restore_epoch: null
exp_name: null
agent_overrides:
  bc_coeff: 3.0
  value_arch: bilinear
  value_latent_dim: 50
  sf_feature_td: true
  sf_coef: 1.0
  sf_base: learned
  sf_joint_psi: true
  sf_base_index: current
  sf_mask: goal
  sf_psi_ortho_coef: 1000.0
  fb_forward_z: true
  project_z: true
wandb_mode: online
```

- [ ] **Step 3: Verify both configs construct an agent (guards accept them)**

Run:

```bash
cd /home/adityamohan/git/Factored-FB/third_party/ogbench/impls && /dev/shm/.venv-jax/bin/python -c "
import sys, yaml; sys.path.insert(0,'/home/adityamohan/git/Factored-FB/tools/wandb_mode_shim')
import jax, jax.numpy as jnp
from gciql_flow import GCIQLFlowAgent, get_config
for n in ['sf_joint_ortho1k','sf_joint_ortho1k_absorb']:
    p='/home/adityamohan/git/Factored-FB/configs/gciql/cube_single_state_gciql_flow_%s.yaml'%n
    ov=yaml.safe_load(open(p))['agent_overrides']
    keys=set(dict(get_config()))
    bad=[k for k in ov if k not in keys]
    assert not bad, (n, bad)
    cfg=dict(get_config()); cfg.update(ov); cfg['encoder']=None; cfg['frame_stack']=None
    ex_obs=jax.random.normal(jax.random.PRNGKey(0),(4,29)); ex_act=jax.random.normal(jax.random.PRNGKey(1),(4,5))
    GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
    print('ok', n)
"
```

Expected:
```
ok sf_joint_ortho1k
ok sf_joint_ortho1k_absorb
```

- [ ] **Step 4: Register both groups in `results_map.py`**

In `scripts/results_map.py`, immediately **above** the line beginning `# --- 1b: readout-vs-TD crux`, insert:

```python
    # --- 1a: joint-psi SF, FB ortho + sphere (spec 2026-07-10). Cell A is the estimator-axis
    # control: identical to GCIQL-State-Flow-Bilinear-d50-C-FBzProjZ-O1000 (0.231) except the
    # BxB measure loss is replaced by the per-sample vector TD.
    'GCIQL-State-Flow-SF-JointPsi-Ortho1k':        dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='sf-joint-ortho1k', graft='SF vector TD trains psi too; FB ortho1000 + sphere; b=psi(s"), no mask (matched to C-fbz-projz-ortho1000 except the estimator)'),
    'GCIQL-State-Flow-SF-JointPsi-Ortho1k-Absorb': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='sf-joint-ortho1k-absorb', graft='+ absorbing goal mask: Q=gamma^T (first-hitting) instead of a visitation density'),
```

- [ ] **Step 5: Verify the map still accounts for every group**

Run: `.venv/bin/python scripts/compile_results_csvs.py 2>&1 | tail -3`
Expected: `[compile] audit -> .../compile_audit.txt  (all groups accounted for)` and exit 0. (The two new groups appear under `MAPPED BUT ABSENT FROM WANDB` until the runs start — that is a note, not a failure.)

- [ ] **Step 6: Write the launch script**

Create `scripts/launch_sf_joint.sh` and `chmod +x` it:

```bash
#!/usr/bin/env bash
# Joint-psi SF cells (spec docs/superpowers/specs/2026-07-10-joint-sf-ortho-design.md).
#
#   A  sf_joint_ortho1k          b=psi(s'), no mask   -> visitation density; matched to
#                                GCIQL-State-Flow-Bilinear-d50-C-FBzProjZ-O1000 = 0.231 +/- 0.030
#                                in every respect EXCEPT the estimator (per-sample vs BxB).
#   C  sf_joint_ortho1k_absorb   b=psi(s), goal mask  -> Q = gamma^T (first-hitting)
#
# ALWAYS in tmux:
#   tmux new -d -s sfjoint 'DRY_RUN=0 scripts/launch_sf_joint.sh 2>&1 | tee /dev/shm/sfjoint_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(pwd)/.venv/bin/python}   # run_gciql.py is a hydra launcher: it runs HERE and
                                    # subprocesses ogbench main.py into .venv-jax itself.
                                    # Do NOT point this at .venv-jax -- that venv has no hydra.
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-500000}
LOGDIR=/dev/shm/sfjoint_logs
mkdir -p "$LOGDIR"

configs=(
  cube_single_state_gciql_flow_sf_joint_ortho1k
  cube_single_state_gciql_flow_sf_joint_ortho1k_absorb
)

JOBS=()
for cfg in "${configs[@]}"; do
  for s in 0 1 2; do
    JOBS+=("${cfg}__s${s}|||--config-name $cfg seed=$s train_steps=$STEPS save_interval=$STEPS exp_name=${cfg}_s${s}")
  done
done

GPUS=(${GPUS:-0 1 2 3 4 5})
echo "joint-psi SF: ${#JOBS[@]} runs @${STEPS} over ${#GPUS[@]} GPUs"
if [ "$DRY_RUN" = "1" ]; then
  i=0; for j in "${JOBS[@]}"; do echo "GPU${GPUS[$((i % ${#GPUS[@]}))]}  ${j%%|||*}"; i=$((i+1)); done
  exit 0
fi

i=0
for j in "${JOBS[@]}"; do
  tag=${j%%|||*}; args=${j#*|||}
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  echo "[launch] GPU$g  $tag"
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    $PY run_gciql.py $args >"$LOGDIR/${tag}.log" 2>&1 &
  i=$((i+1))
  sleep 5
done
echo "Launched ${#JOBS[@]} jax runs -> $LOGDIR ; waiting..."
wait
echo "ALL DONE"
```

- [ ] **Step 7: Dry-run the launcher**

Run: `chmod +x scripts/launch_sf_joint.sh && scripts/launch_sf_joint.sh`
Expected: `joint-psi SF: 6 runs @500000 over 6 GPUs` followed by 6 `GPU<n>  <cfg>__s<seed>` lines. No processes started.

- [ ] **Step 8: Commit**

```bash
git add configs/gciql/cube_single_state_gciql_flow_sf_joint_ortho1k.yaml \
        configs/gciql/cube_single_state_gciql_flow_sf_joint_ortho1k_absorb.yaml \
        scripts/results_map.py scripts/launch_sf_joint.sh
git commit -m "feat(configs): joint-psi SF cells A and C + launcher

A (sf_joint_ortho1k): b=psi(s'), no mask, ortho1000 on psi(s'), fb_forward_z + project_z.
Matched to C-fbz-projz-ortho1000 (0.231 +/- 0.030) in every respect except the estimator, so it
isolates the per-sample-vs-BxB axis the SF program exists to test and never has.

C (sf_joint_ortho1k_absorb): keeps the absorbing goal mask with a z-conditioned phi, computing
Q = gamma^T (first-hitting) rather than a visitation density. No FB counterpart.

Distinct wandb groups, both registered in results_map."
```

---

### Task 6: Launch and verify the campaign

**Files:** none modified. Operational only.

**Interfaces:**
- Consumes: `scripts/launch_sf_joint.sh` (Task 5).
- Produces: 6 live runs; a health check that distinguishes a real failure from the known slow first eval.

- [ ] **Step 1: Confirm GPU capacity**

Run: `nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l`

The `psi_base` campaign may still hold GPUs. If ≥ 10 processes are resident, wait or set `GPUS` to the free devices. Do **not** kill the running campaign.

- [ ] **Step 2: Launch in tmux**

Run:

```bash
cd /home/adityamohan/git/Factored-FB
tmux ls 2>/dev/null | grep -q "^sfjoint:" && echo "SESSION EXISTS - abort" || \
  tmux new -d -s sfjoint 'DRY_RUN=0 scripts/launch_sf_joint.sh 2>&1 | tee /dev/shm/sfjoint_dispatch.log'
```

Note: `tmux has-session -t sfjoint` prefix-matches other sessions; the `grep "^sfjoint:"` form above is the safe check.

- [ ] **Step 3: Confirm all 6 dispatched and none crashed on import**

Run:

```bash
until [ "$(grep -c '\[launch\]' /dev/shm/sfjoint_dispatch.log 2>/dev/null)" = "6" ]; do sleep 10; done
sleep 90
for f in /dev/shm/sfjoint_logs/*.log; do
  if grep -qi "Traceback\|AssertionError" "$f"; then echo "CRASH $(basename $f): $(grep -m1 -i error "$f" | cut -c1-90)";
  else echo "ok    $(basename $f)"; fi
done
```

Expected: 6 `ok` lines. A `ModuleNotFoundError: hydra` means `PY` is wrong. An `AssertionError` means a guard rejected the config — fix the config, do not weaken the guard.

- [ ] **Step 4: Confirm the runs reach the GPU**

Run: `sleep 120; nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l`
Expected: 6 more compute processes than in Step 1.

**Do not mistake a slow first eval for a hang.** The step-1 eval runs a random-init flow policy that flails, making mujoco's contact solver crawl (~20 s/episode × 5 tasks × 51 episodes ≈ 80 min) while the GPU shows 0% util. `.venv/bin/py-spy dump --pid <child>` will show `utils/evaluation.py:evaluate`. That is normal.

- [ ] **Step 5: Sanity-check the diagnostics at the first logged eval**

Once any run logs `_step >= 5000`, verify the ortho machinery is live:

```bash
cd /home/adityamohan/git/Factored-FB
.venv/bin/python -c "
import wandb; api=wandb.Api(timeout=120)
K=['training/sf/psi_norm_mean','training/sf/psi_ortho_off','training/sf/psi_ortho_diag','training/sf/sf_loss']
for g in ['GCIQL-State-Flow-SF-JointPsi-Ortho1k','GCIQL-State-Flow-SF-JointPsi-Ortho1k-Absorb']:
    for r in api.runs('amsks/factored-fb',{'group':g},per_page=10):
        pts=[p for p in r.history(keys=K,samples=500,pandas=False) if p.get(K[0]) is not None]
        if pts:
            p=pts[-1]
            print(g.split('-')[-1], 'seed', r.config.get('seed'), {k.split('/')[-1]: round(p[k],4) for k in K})
"
```

Expected: `psi_norm_mean ≈ 7.071` (= √50, pinned by `project_z`) and `psi_ortho_diag ≈ -50.0` (constant, since `tr(C)=d`). If `psi_norm_mean` drifts, `project_z` is not reaching the base leg. Watch `psi_ortho_off` fall — if it plateaus high, ψ never decorrelated and the peaked-kernel argument does not hold, which changes how the cell reads.

`sf_loss` growing without bound is **benign** for a successor measure (the PSM work established this); it is not divergence. The flow actor normalizes its Q-term by `sg[|q|.mean()]`.

- [ ] **Step 6: Record the campaign in memory and stop**

Update the memory note for the bridge track with: tmux session `sfjoint`, log dir, the two groups, the matched baseline (0.231 ± 0.030), and what each outcome means (A ≈ 0.23 ⇒ estimator irrelevant, SF story dissolves into FB; A ≈ 0.0 ⇒ per-sample vector TD is uniquely destructive).

Collection is a separate session: add nothing to `results_map` (already done in Task 5), run `.venv/bin/python scripts/compile_results_csvs.py --refresh`, and fold the numbers into `PAPER/fb-gciql-bridge/sf-design-space.md` §1a and its 2×2.

---

## Self-Review

**Spec coverage.** §3 two cells → Task 5 configs. §4 loss → Tasks 2+3. §4.1 ortho-not-optional → guard 2 (Task 4). §5 four knobs → Tasks 2+3 `get_config()`; `project_z` threading → Task 1. §5.1 five guards → Task 4. §6 seven tests → Tasks 1–4 (9 test functions; the spec's test 2 splits into an exact-formula test and a `project_z`-diagonal test, and its test 3 splits into the `none` and `goal` halves). §7 configs → Task 5. §8 diagnostics → Task 2 info keys + Task 6 Step 5. §9 anchors → config headers and the launcher comment. §10 sequencing → Task 6.

**Deviations, both flagged above and justified in-plan:** guard 3 scoped to `sf_joint_psi` (else every legacy SF config breaks); the ortho identity asserted in its exact finite-batch form rather than the spec's `B→∞` limit.

**Type/name consistency.** Knobs are spelled `sf_joint_psi`, `sf_base_index`, `sf_mask`, `sf_psi_ortho_coef` identically in `get_config()`, `sf_loss`, guards, tests, and both YAMLs. Info keys `psi_ortho`, `psi_ortho_off`, `psi_ortho_diag`, `psi_norm_mean` are produced in Task 2 and consumed in Task 3's cell-A test and Task 6's wandb check (as `training/sf/<key>` — the agent prefixes `sf/`, wandb prefixes `training/`). `_sphere(x, d)` and `GCLearnedBValue`/`GCFrozenBValue`/`GCIdentityBValue` match the existing source.

**No placeholders.** Every code step carries the code; every run step carries the command and its expected output.
