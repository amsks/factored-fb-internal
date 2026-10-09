# Task-Weighted Successor Moments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether FB's measure loss fails on cube because it minimizes density error in the **replay norm** (ρ) rather than the **task norm** (ν), by moving the B×B test states from `next_observations` to `value_goals` — and separately, by collapsing the test set to rank 1.

**Architecture:** Two default-OFF knobs inside `gciql_flow.py::measure_loss`'s critic branch. `measure_test_states ∈ {'rho','goals'}` selects the distribution the residual is tested against; `measure_test_rank ∈ {'full','task'}` selects B×B cross-pairs versus a per-sample projection onto `zᵢ = ψ(gᵢ)`. The **Dirac term stays on `ψ(s′ᵢ)`** and the **ortho stays on `ψ(next_observations)`** in every cell — those two facts are what make the task norm non-degenerate, and both are asserted by tests.

**Tech Stack:** JAX/Flax, OGBench GCIQL trunk, pytest via jax-subprocess harness, hydra configs, tmux + wandb.

**Spec:** `docs/superpowers/specs/2026-07-10-task-weighted-moments-design.md`

## Global Constraints

- **DO NOT LAUNCH ANY RUNS.** This plan ends at a dry-run. The user requires a deep code audit and a run manifest first.
- Python for tests: `.venv/bin/python -m pytest` (never `uv run`). The jax GPU venv is `/dev/shm/.venv-jax`; `tests/jax_subprocess.py` invokes it for you.
- Agent file: `tools/wandb_mode_shim/gciql_flow.py`. Tests: `tests/`. Configs: `configs/gciql/`.
- Both new knobs default to today's behaviour. **`('rho','full')` must be bit-identical to the current `measure_loss`** — that is why the legacy path keeps calling `_measure_terms` untouched.
- Existing suites must stay green: `tests/test_gciql_sf_feature_td.py` (16), `tests/test_gciql_psi_base.py` (4), `tests/test_gciql_sf_joint_psi.py` (9), `tests/test_gciql_fb_forward_z.py` (2), `tests/test_gciql_measure_scale.py`.
- Commit messages: **omit** any `Co-Authored-By: Claude` trailer (repo convention).
- Commit only code/tests/configs/scripts. `PAPER/`, `results/`, `analysis/`, `docs/superpowers/` are gitignored.
- wandb entity is `amsks` (defaulted by `run_gciql.py:128`). Every training job runs in **tmux**.
- Launch scripts set `PY` to the **repo** `.venv/bin/python` — `run_gciql.py` is a hydra launcher that subprocesses ogbench `main.py` into `.venv-jax` itself. Pointing `PY` at `.venv-jax` dies on missing hydra, silently, per-run.
- **Never reuse a wandb `run_group`.** `results_map.py` keys on the group; reusing one silently averages old and new seeds.
- `d = value_latent_dim = 50`, batch size 1024 (`gciql.py:282`).

---

## File Structure

| File | Responsibility |
|---|---|
| `tools/wandb_mode_shim/gciql_flow.py` | Modify: `measure_loss` critic branch (split test set from Dirac term); `create()` (4 guards); `get_config()` (2 knobs). `_measure_terms` is **not** touched. |
| `tests/test_gciql_task_norm.py` | Create: 6 TDD tests. |
| `configs/gciql/cube_single_state_gciql_flow_bilinear_d50_C_tasknorm.yaml` | Create: N1 (ν test states, full B×B). |
| `configs/gciql/cube_single_state_gciql_flow_bilinear_d50_C_taskproj.yaml` | Create: N2 (ν test states, rank-1 diagonal). |
| `scripts/results_map.py` | Modify: register both wandb groups. |
| `scripts/launch_tasknorm.sh` | Create: 6 runs (2 cells × 3 seeds) @500k in tmux. **Dry-run only in this plan.** |

---

## Key facts (read before Task 1)

Current critic branch of `measure_loss` (`gciql_flow.py:596-626`):

```python
_, phi, psi = self.network.select('critic')(obs, nobs, acts, *zarg, info=True, params=grad_params, **zkw)
M  = jnp.einsum('hid,hjd->hij', phi, psi) / sqd            # psi = psi(nobs)  -> test states ~ rho
_, tphi, tpsi = self.network.select('target_critic')(nobs, nobs, a_next, *zarg, info=True, **zkw)
tM = jax.lax.stop_gradient(jnp.einsum('hid,hjd->hij', tphi, tpsi) / sqd)
offdiag, diagt = self._measure_terms(M, tM, gamma, scale_norm)
Cov = jnp.einsum('hid,hjd->hij', psi, psi)                 # ortho on psi(nobs)
```

- `_measure_terms(M, tM, ...)` extracts **both** the off-diagonal (test) and diagonal (Dirac) terms from **one** matrix. That only works because the test states *are* the next-states. It must keep working untouched for the legacy path.
- `phi` depends on `(observations, actions, z_goals)` only; `b_goals` feeds **`psi` alone**. So `critic(obs, vgoals, acts, *zarg)` returns the *same* `phi` and `psi(vgoals)`. Same for `target_critic`. We therefore take only the `psi` from the extra calls and reuse `phi`/`tphi`.
- `measure_loss` has **zero** references to `batch['masks']` (verified). Do not add one.
- With `fb_forward_z=True` the critic class is `GCFBValue`, whose `__call__(observations, b_goals, actions, z_goals, info=False, z_vec=None)` spheres `psi` when `project_z`.
- Ensembled shapes: `phi`, `psi` are `[2, B, d]`; `M` is `[2, B, B]`; `offdiag`, `diagt`, `orth` are `[2]` and are averaged at the end.

---

### Task 1: The two knobs and the split loss path

**Files:**
- Modify: `tools/wandb_mode_shim/gciql_flow.py` (`measure_loss` critic branch, `gciql_flow.py:596-626`; `get_config()`)
- Test: `tests/test_gciql_task_norm.py`

**Interfaces:**
- Consumes: `_measure_terms` (untouched), `GCFBValue`, existing `measure_*` knobs.
- Produces: config knobs `measure_test_states: str = 'rho'` and `measure_test_rank: str = 'full'`. When either is off-default, `measure_loss` computes `offdiag` from `ψ(value_goals)` and `diagt` from `ψ(next_observations)` separately, and adds info key `measure_qz_mean`.

- [ ] **Step 1: Write the five failing tests**

Create `tests/test_gciql_task_norm.py`:

```python
"""Task-weighted successor moments (spec: docs/superpowers/specs/2026-07-10-task-weighted-moments-design.md).

FB's measure loss tests its residual against psi(s'_j), s' ~ rho (the replay marginal). The research
note argues control needs the residual weighted by the TASK norm instead. For goal indicators the
moment map collapses (z_g ∝ psi(g)), so the task norm is exactly: draw the B x B test states from
value_goals (nu) rather than next_observations (rho).

Two invariants make the cell non-degenerate, and both are asserted here:
  * the ORTHO stays on psi(next_observations). If it followed the test set, C_nu -> I at the ortho
    optimum and the objective would silently become canonical SF (measured: 0.120).
  * the DIRAC term stays on psi(s'_i). It is the reward. If it followed the test set, the diagonal
    becomes Q - gamma*Qbar and the reward vanishes.

Runs in the jax venv subprocess.
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS

# Shared config + batch preamble for a measure_only FB cell (the C-fbz-projz-ortho1000 shape).
PREAMBLE = """
import jax, jax.numpy as jnp
from gciql_flow import GCIQLFlowAgent, get_config

def build(test_states="rho", test_rank="full", lam=2.0, gamma=0.99, B=9, gseed=3):
    cfg = dict(get_config())
    cfg["encoder"] = None; cfg["frame_stack"] = None
    cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
    cfg["measure_only"] = True; cfg["measure_coef"] = 1.0; cfg["measure_legs"] = "critic"
    cfg["fb_forward_z"] = True; cfg["project_z"] = True
    cfg["measure_ortho_coef"] = lam
    cfg["measure_test_states"] = test_states; cfg["measure_test_rank"] = test_rank
    cfg["discount"] = gamma
    ex_obs  = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
    ex_nobs = jax.random.normal(jax.random.PRNGKey(4), (B, 11))
    ex_act  = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
    goals   = jax.random.normal(jax.random.PRNGKey(gseed), (B, 11))
    agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
    batch = dict(observations=ex_obs, next_observations=ex_nobs, actions=ex_act,
                 rewards=-jnp.ones((B,)), masks=jnp.ones((B,)),
                 value_goals=goals, actor_goals=goals)
    return agent, batch, ex_obs, ex_nobs, ex_act, goals

def legs(agent, obs, b_goals, acts, zg, params):
    # phi depends on (obs, acts, z_goals); b_goals feeds psi alone.
    _, phi, psi = agent.network.select("critic")(obs, b_goals, acts, zg, info=True, params=params)
    return phi, psi
"""


def test_rho_full_is_bit_identical_to_the_legacy_measure_loss():
    """BACKWARD COMPAT, decisive. The default ('rho','full') must reproduce the single-matrix
    _measure_terms formula exactly: offdiag and diag are the two halves of ONE B x B matrix built
    from psi(next_observations)."""
    code = PREAMBLE + """
        agent, batch, obs, nobs, acts, goals = build("rho", "full")
        p = agent.network.params
        loss, info = agent.measure_loss(batch, p, jax.random.PRNGKey(2))
        # hand-compute the legacy formula
        phi, psi = legs(agent, obs, nobs, acts, goals, p)
        noise = jax.random.normal(jax.random.PRNGKey(2), acts.shape)
        a_next = jax.lax.stop_gradient(agent.network.select("actor")(nobs, goals, noise))
        _, tphi, tpsi = agent.network.select("target_critic")(nobs, nobs, a_next, goals, info=True)
        sqd = jnp.sqrt(50.0); B = obs.shape[0]; off = 1.0 - jnp.eye(B)
        M  = jnp.einsum('hid,hjd->hij', phi, psi) / sqd
        tM = jnp.einsum('hid,hjd->hij', tphi, tpsi) / sqd
        diff = M - 0.99 * tM
        exp_off = 0.5 * jnp.sum((diff * off[None]) ** 2, axis=(1, 2)) / jnp.sum(off)
        exp_dia = -jnp.mean(jnp.diagonal(diff, axis1=1, axis2=2), axis=1)
        assert jnp.allclose(info["measure_offdiag"], jnp.mean(exp_off), rtol=1e-5), (info["measure_offdiag"], jnp.mean(exp_off))
        assert jnp.allclose(info["measure_diag"],    jnp.mean(exp_dia), rtol=1e-5)
        assert "measure_qz_mean" not in info
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_goals_full_tests_against_psi_of_value_goals():
    """DECISIVE for N1. offdiag is built from psi(value_goals) (the task distribution nu), while the
    Dirac term stays on the psi(next_observations) diagonal and the ortho stays on
    psi(next_observations)."""
    code = PREAMBLE + """
        agent, batch, obs, nobs, acts, goals = build("goals", "full")
        p = agent.network.params
        loss, info = agent.measure_loss(batch, p, jax.random.PRNGKey(2))
        phi, psi_o = legs(agent, obs, nobs,  acts, goals, p)      # rho  leg
        _,   psi_t = legs(agent, obs, goals, acts, goals, p)      # task leg
        noise = jax.random.normal(jax.random.PRNGKey(2), acts.shape)
        a_next = jax.lax.stop_gradient(agent.network.select("actor")(nobs, goals, noise))
        _, tphi, tpsi_o = agent.network.select("target_critic")(nobs, nobs,  a_next, goals, info=True)
        _, _,    tpsi_t = agent.network.select("target_critic")(nobs, goals, a_next, goals, info=True)
        sqd = jnp.sqrt(50.0); B = obs.shape[0]; off = 1.0 - jnp.eye(B)
        diff_t = (jnp.einsum('hid,hjd->hij', phi, psi_t) - 0.99 * jnp.einsum('hid,hjd->hij', tphi, tpsi_t)) / sqd
        exp_off = 0.5 * jnp.sum((diff_t * off[None]) ** 2, axis=(1, 2)) / jnp.sum(off)
        dir_i = (jnp.einsum('hid,hid->hi', phi, psi_o) - 0.99 * jnp.einsum('hid,hid->hi', tphi, tpsi_o)) / sqd
        exp_dia = -jnp.mean(dir_i, axis=1)
        Cov = jnp.einsum('hid,hjd->hij', psi_o, psi_o)
        o_off = 0.5 * jnp.sum((Cov * off[None]) ** 2, axis=(1, 2)) / jnp.sum(off)
        o_dia = -jnp.mean(jnp.diagonal(Cov, axis1=1, axis2=2), axis=1)
        assert jnp.allclose(info["measure_offdiag"], jnp.mean(exp_off), rtol=1e-4), (info["measure_offdiag"], jnp.mean(exp_off))
        assert jnp.allclose(info["measure_diag"],    jnp.mean(exp_dia), rtol=1e-4)
        assert jnp.allclose(info["measure_orth"],    jnp.mean(o_off + o_dia), rtol=1e-4)
        # test vacuity guard: psi(goals) must actually differ from psi(nobs)
        assert not jnp.allclose(psi_t, psi_o)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_perturbing_goals_moves_offdiag_but_never_the_ortho():
    """THE INVARIANT THAT MAKES THE CELL NON-DEGENERATE. We whiten under rho and test under nu.
    If the ortho followed the test set, C_nu -> I at its optimum and the objective would silently
    collapse to canonical SF (0.120).

    NOTE: we do NOT assert the Dirac term is goal-invariant. Under fb_forward_z the forward map is
    phi(s,a,z) with z = psi(g), so perturbing the goals legitimately changes phi and hence
    dir_i = <phi_i, psi(s'_i)>. The ortho reads psi(next_observations) alone and is the only term
    that must be goal-independent. That the Dirac uses psi(s'_i) rather than psi(g_i) is asserted
    separately, by test_dirac_term_never_follows_the_test_set."""
    code = PREAMBLE + """
        a1, b1, *_ = build("goals", "full", gseed=3)
        a2, b2, *_ = build("goals", "full", gseed=77)   # different value_goals, same obs/nobs/acts
        i1 = a1.measure_loss(b1, a1.network.params, jax.random.PRNGKey(2))[1]
        # same params: rebuild agent 2 with agent 1's params so only the goals differ
        i2 = a1.measure_loss(dict(b1, value_goals=b2["value_goals"], actor_goals=b2["actor_goals"]),
                             a1.network.params, jax.random.PRNGKey(2))[1]
        assert not jnp.allclose(i1["measure_offdiag"], i2["measure_offdiag"]), "offdiag must depend on the goals"
        assert jnp.allclose(i1["measure_orth"], i2["measure_orth"], rtol=1e-6), "ortho must NOT follow the test set"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_goals_task_gamma0_is_the_rank_one_projection():
    """DECISIVE for N2. At gamma=0 the rank-1 cell is
        L = 0.5*mean_i (Q_i)^2  -  mean_i <phi_i, psi(s'_i)>/sqrt(d)  +  lam*ortho
    with Q_i = <phi_i, psi(g_i)>/sqrt(d): each residual is projected onto its OWN task vector."""
    code = PREAMBLE + """
        agent, batch, obs, nobs, acts, goals = build("goals", "task", lam=2.0, gamma=0.0)
        p = agent.network.params
        loss, info = agent.measure_loss(batch, p, jax.random.PRNGKey(2))
        phi, psi_o = legs(agent, obs, nobs,  acts, goals, p)
        _,   psi_t = legs(agent, obs, goals, acts, goals, p)
        sqd = jnp.sqrt(50.0); B = obs.shape[0]; off = 1.0 - jnp.eye(B)
        q_i   = jnp.einsum('hid,hid->hi', phi, psi_t) / sqd        # [2,B]
        dir_i = jnp.einsum('hid,hid->hi', phi, psi_o) / sqd        # [2,B]
        exp_off = 0.5 * jnp.mean(q_i ** 2, axis=1)
        exp_dia = -jnp.mean(dir_i, axis=1)
        Cov = jnp.einsum('hid,hjd->hij', psi_o, psi_o)
        o_off = 0.5 * jnp.sum((Cov * off[None]) ** 2, axis=(1, 2)) / jnp.sum(off)
        o_dia = -jnp.mean(jnp.diagonal(Cov, axis1=1, axis2=2), axis=1)
        assert jnp.allclose(info["measure_offdiag"], jnp.mean(exp_off), rtol=1e-4), (info["measure_offdiag"], jnp.mean(exp_off))
        assert jnp.allclose(info["measure_diag"],    jnp.mean(exp_dia), rtol=1e-4)
        assert jnp.allclose(loss, jnp.mean(exp_off + exp_dia + 2.0 * (o_off + o_dia)), rtol=1e-4)
        assert jnp.allclose(info["measure_qz_mean"], jnp.mean(q_i), rtol=1e-4)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_dirac_term_never_follows_the_test_set():
    """The Dirac term IS the reward. If it were computed from psi(g_i) instead of psi(s'_i), the
    diagonal would be Q_i - gamma*Qbar_i and the reward would vanish -- silently, and the cell's
    number would be uninterpretable. Assert the two candidates differ and that we use psi(s')."""
    code = PREAMBLE + """
        agent, batch, obs, nobs, acts, goals = build("goals", "task", gamma=0.0)
        p = agent.network.params
        _, info = agent.measure_loss(batch, p, jax.random.PRNGKey(2))
        phi, psi_o = legs(agent, obs, nobs,  acts, goals, p)
        _,   psi_t = legs(agent, obs, goals, acts, goals, p)
        sqd = jnp.sqrt(50.0)
        right = -jnp.mean(jnp.einsum('hid,hid->hi', phi, psi_o) / sqd)     # psi(s') -- correct
        wrong = -jnp.mean(jnp.einsum('hid,hid->hi', phi, psi_t) / sqd)     # psi(g)  -- the bug
        assert not jnp.allclose(right, wrong), "test vacuous: psi(s') == psi(g)"
        assert jnp.allclose(info["measure_diag"], right, rtol=1e-4), (info["measure_diag"], right, wrong)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_gciql_task_norm.py -v`
Expected: `test_rho_full_is_bit_identical...` FAILS with `KeyError: 'measure_test_states'` (the config key does not exist, so `dict(get_config())` has no such entry and `cfg[...] = ...` succeeds but `create()` ignores it — the assertion `"measure_qz_mean" not in info` passes, but the other four FAIL because the knobs are ignored and `offdiag` is still built from `ψ(nobs)`).

*(If the first test unexpectedly passes, that is correct — it asserts today's behaviour. The other four must fail.)*

- [ ] **Step 3: Add the two config knobs**

In `get_config()`, immediately after `config.measure_z_random_frac = 0.0`, insert:

```python
        config.measure_test_states = 'rho'   # distribution the measure residual is TESTED against: 'rho' -> psi(next_observations), FB-faithful | 'goals' -> psi(value_goals), the task distribution nu (the research note's task norm). The ortho and the Dirac term ALWAYS stay on psi(next_observations)
        config.measure_test_rank = 'full'    # test set rank: 'full' -> B x B cross-pairs (i != j) | 'task' -> diagonal, project each residual onto its OWN task vector z_i = psi(g_i) (rank 1). 'task' requires measure_test_states='goals'
```

- [ ] **Step 4: Split the test set from the Dirac term in `measure_loss`**

In `gciql_flow.py`, replace the block from

```python
            _, tphi, tpsi = self.network.select('target_critic')(nobs, nobs, a_next, *zarg, info=True, **zkw)
            tM = jax.lax.stop_gradient(jnp.einsum('hid,hjd->hij', tphi, tpsi) / sqd)            # [2, B, B]
            offdiag, diagt = self._measure_terms(M, tM, gamma, scale_norm)                      # [2]
```

with:

```python
            _, tphi, tpsi = self.network.select('target_critic')(nobs, nobs, a_next, *zarg, info=True, **zkw)
            tM = jax.lax.stop_gradient(jnp.einsum('hid,hjd->hij', tphi, tpsi) / sqd)            # [2, B, B]
            test_states = self.config.get('measure_test_states', 'rho')
            test_rank = self.config.get('measure_test_rank', 'full')
            qz_mean = None
            if test_states == 'rho' and test_rank == 'full':
                # Legacy FB path, untouched: offdiag and diag are the two halves of ONE matrix,
                # which only works because the test states ARE the next-states.
                offdiag, diagt = self._measure_terms(M, tM, gamma, scale_norm)                  # [2]
            else:
                # Task norm. The test set moves to nu=psi(value_goals); the DIRAC term (the reward,
                # -E_i M(s_i,a_i,s'_i)) stays on psi(s'_i), and the ortho below stays on psi(nobs).
                # Letting the Dirac follow the test set would make the diagonal Q_i - gamma*Qbar_i
                # and delete the reward. phi depends on (obs,acts,z) only -- b_goals feeds psi
                # alone -- so we reuse phi/tphi and take only the psi legs from the extra calls.
                _, _, psi_t = self.network.select('critic')(obs, vgoals, acts, *zarg, info=True,
                                                            params=grad_params, **zkw)
                _, _, tpsi_t = self.network.select('target_critic')(nobs, vgoals, a_next, *zarg,
                                                                    info=True, **zkw)
                M_t = jnp.einsum('hid,hjd->hij', phi, psi_t) / sqd                              # [2,B,B]
                tM_t = jax.lax.stop_gradient(jnp.einsum('hid,hjd->hij', tphi, tpsi_t) / sqd)
                diff_t = M_t - gamma * tM_t
                if test_rank == 'full':
                    offdiag = 0.5 * jnp.sum((diff_t * off[None]) ** 2, axis=(1, 2)) / jnp.sum(off)
                else:  # 'task': rank-1 projection of each residual onto its OWN z_i = psi(g_i)
                    qz = jnp.diagonal(diff_t, axis1=1, axis2=2)                                 # [2,B]
                    offdiag = 0.5 * jnp.mean(qz ** 2, axis=1)                                   # [2]
                    qz_mean = jnp.mean(qz)
                # Dirac term: the rho diagonal, computed directly (never via the test matrix).
                dir_i = (jnp.einsum('hid,hid->hi', phi, psi)
                         - gamma * jax.lax.stop_gradient(jnp.einsum('hid,hid->hi', tphi, tpsi))) / sqd
                diagt = -jnp.mean(dir_i, axis=1)                                                # [2]
                if scale_norm:
                    msq = jax.lax.stop_gradient(jnp.mean(M_t ** 2))
                    offdiag = offdiag / (msq + 1e-6)
                    diagt = diagt / (jnp.sqrt(msq) + 1e-6)
```

Then, in the same branch, replace

```python
            offdiag, diagt, orth = jnp.mean(offdiag), jnp.mean(diagt), jnp.mean(orth)
        return loss, {'measure_loss': loss, 'measure_offdiag': offdiag, 'measure_diag': diagt,
                      'measure_orth': orth}
```

with

```python
            offdiag, diagt, orth = jnp.mean(offdiag), jnp.mean(diagt), jnp.mean(orth)
        info = {'measure_loss': loss, 'measure_offdiag': offdiag, 'measure_diag': diagt,
                'measure_orth': orth}
        if self.config.get('measure_test_rank', 'full') == 'task':
            info['measure_qz_mean'] = qz_mean
        return loss, info
```

**Note for the implementer:** `qz_mean` is defined only inside the critic branch. The `measure_legs='value'` branch never sets `measure_test_rank='task'` (guard 3 in Task 2 forbids it), so the `info` line above is safe — but define `qz_mean = None` at the top of `measure_loss` as well, so a future value-legs cell cannot raise `UnboundLocalError`.

Concretely, add `qz_mean = None` immediately after the line `scale_norm = self.config.get('measure_scale_norm', False)`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_gciql_task_norm.py -v`
Expected: 5 passed.

- [ ] **Step 6: Verify the legacy path is untouched**

Run: `.venv/bin/python -m pytest tests/test_gciql_measure_scale.py tests/test_gciql_fb_forward_z.py -q`
Expected: all passed (these exercise `measure_loss` with the default `('rho','full')`).

- [ ] **Step 7: Commit**

```bash
git add tools/wandb_mode_shim/gciql_flow.py tests/test_gciql_task_norm.py
git commit -m "feat(gciql): measure_test_states + measure_test_rank -- the task norm

FB tests its measure residual against psi(s'_j), s' ~ rho. The research note argues control needs
the residual weighted by the TASK norm. For goal indicators the moment map collapses
(z_g ∝ psi(g)), so the task norm is exactly: draw the B x B test states from value_goals (nu).

measure_test_states='goals' does that. measure_test_rank='task' collapses the test set to rank 1 by
projecting each residual onto its OWN z_i = psi(g_i); its stationary point is GCIQL's backup with
the learned kernel <psi(s'),psi(g)>/sqrt(d) in place of the true indicator.

Two invariants keep the cell non-degenerate, both asserted by tests:
 * the ORTHO stays on psi(next_observations). If it followed the test set, C_nu -> I at the ortho
   optimum and the objective would silently become canonical SF (measured 0.120).
 * the DIRAC term stays on psi(s'_i). It is the reward; letting it follow the test set turns the
   diagonal into Q - gamma*Qbar and deletes it.

('rho','full') keeps calling _measure_terms untouched and is bit-identical to today's loss."
```

---

### Task 2: The four guards

**Files:**
- Modify: `tools/wandb_mode_shim/gciql_flow.py` (`create()`, after `config['action_dim'] = action_dim`)
- Test: `tests/test_gciql_task_norm.py`

**Interfaces:**
- Consumes: the two knobs from Task 1.
- Produces: `create()` raises `AssertionError` on four invalid combinations; both valid cells construct.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gciql_task_norm.py`:

```python
def test_task_norm_guards_reject_invalid_combinations():
    """The guards encode the design so a misconfigured cell fails loudly instead of producing an
    uninterpretable number.

    1. measure_test_rank='task' => measure_test_states='goals'. The rho diagonal IS the Dirac term;
       projecting there would double-count the reward.
    2. either knob off-default => measure_only or measure_coef>0 (they only affect measure_loss).
    3. either knob off-default => measure_legs='critic' (the value-legs branch has no goal leg).
    4. either knob off-default => not sf_feature_td, and value_arch != 'simplesf' (each replaces
       the loss entirely)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (6, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (6, 5))
        def build(**over):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
            cfg.update(over)
            return GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        M = dict(measure_only=True, measure_coef=1.0, measure_legs="critic",
                 fb_forward_z=True, project_z=True, measure_ortho_coef=1000.0)
        bad = [
            dict(M, measure_test_rank="task", measure_test_states="rho"),   # 1
            dict(M, measure_test_states="goals", measure_only=False, measure_coef=0.0),  # 2
            dict(M, measure_test_states="goals", measure_legs="value"),     # 3
            dict(M, measure_test_states="goals", sf_feature_td=True, sf_base="frozen"),  # 4
        ]
        for over in bad:
            try:
                build(**over)
            except AssertionError:
                continue
            raise SystemExit("guard missing for " + repr(over))
        build(**M, measure_test_states="goals", measure_test_rank="full")   # N1
        build(**M, measure_test_states="goals", measure_test_rank="task")   # N2
        build(**M)                                                          # the 0.231 anchor
        build()                                                             # stock GCIQL
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gciql_task_norm.py::test_task_norm_guards_reject_invalid_combinations -v`
Expected: FAIL with `SystemExit: guard missing for ...`.

- [ ] **Step 3: Add the guards**

In `create()`, immediately after `config['action_dim'] = action_dim`, insert:

```python
        if (config.get('measure_test_states', 'rho') != 'rho'
                or config.get('measure_test_rank', 'full') != 'full'):
            # (1) the rho diagonal IS the Dirac term; projecting there would double-count the reward.
            if config.get('measure_test_rank', 'full') == 'task':
                assert config.get('measure_test_states', 'rho') == 'goals', \
                    "measure_test_rank='task' requires measure_test_states='goals'"
            # (2) both knobs only affect measure_loss.
            assert config.get('measure_only', False) or config.get('measure_coef', 0.0) > 0, \
                "measure_test_states/measure_test_rank require measure_only=True or measure_coef>0"
            # (3) the value-legs branch has no goal leg to test against.
            assert config.get('measure_legs', 'critic') == 'critic', \
                "measure_test_states/measure_test_rank require measure_legs='critic'"
            # (4) these replace the loss entirely and would silently override the measure objective.
            assert not config.get('sf_feature_td', False), \
                "measure_test_states/measure_test_rank are incompatible with sf_feature_td"
            assert config.get('value_arch', 'mlp') != 'simplesf', \
                "measure_test_states/measure_test_rank are incompatible with value_arch='simplesf'"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gciql_task_norm.py -v`
Expected: 6 passed.

- [ ] **Step 5: Run every affected suite**

Run: `.venv/bin/python -m pytest tests/test_gciql_task_norm.py tests/test_gciql_measure_scale.py tests/test_gciql_fb_forward_z.py tests/test_gciql_sf_feature_td.py tests/test_gciql_psi_base.py tests/test_gciql_sf_joint_psi.py -q`
Expected: all passed, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add tools/wandb_mode_shim/gciql_flow.py tests/test_gciql_task_norm.py
git commit -m "feat(gciql): four guards for the task-norm knobs

 1. measure_test_rank='task' => measure_test_states='goals' (the rho diagonal IS the Dirac term)
 2. either knob off-default => measure_only or measure_coef>0
 3. either knob off-default => measure_legs='critic'
 4. either knob off-default => not sf_feature_td and value_arch != 'simplesf'

Misconfigured cells now fail loudly rather than producing an uninterpretable number."
```

---

### Task 3: Configs, results_map, launcher (DRY-RUN ONLY)

**Files:**
- Create: `configs/gciql/cube_single_state_gciql_flow_bilinear_d50_C_tasknorm.yaml`, `configs/gciql/cube_single_state_gciql_flow_bilinear_d50_C_taskproj.yaml`, `scripts/launch_tasknorm.sh`
- Modify: `scripts/results_map.py`

**Interfaces:**
- Consumes: knobs and guards from Tasks 1–2.
- Produces: wandb groups `GCIQL-State-Flow-Bilinear-d50-C-TaskNorm` (N1) and `GCIQL-State-Flow-Bilinear-d50-C-TaskProj` (N2), both registered in `results_map.MAP` with `csv='bridge'`.

**DO NOT LAUNCH.** This task ends at a dry-run.

- [ ] **Step 1: Write N1's config**

Create `configs/gciql/cube_single_state_gciql_flow_bilinear_d50_C_tasknorm.yaml`:

```yaml
# N1 (sf-design-space 1e): the FB measure objective with its B x B test states drawn from the TASK
# distribution nu (value_goals) instead of the replay marginal rho (next_observations).
# Identical to cube_single_state_gciql_flow_bilinear_d50_C_proj.yaml
# (GCIQL-State-Flow-Bilinear-d50-C-FBzProjZ-O1000, 0.231 +/- 0.030) EXCEPT measure_test_states.
#
# The ortho stays on psi(next_observations) and the Dirac term stays on psi(s'_i). That asymmetry
# is the point: FB whitens under rho AND tests under rho, so at the ortho optimum C_rho -> I and
# its own norm degenerates to the identity. Whitening under rho while testing under nu keeps
# C_nu != I, so the residual is genuinely weighted by task-relevant directions.
#
# Reads as: does the replay-vs-task norm explain the measure objective's collapse (0.231 vs 0.904)?
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
run_group: GCIQL-State-Flow-Bilinear-d50-C-TaskNorm
restore_path: null
restore_epoch: null
exp_name: null
agent_overrides:
  bc_coeff: 3.0
  value_arch: bilinear
  value_latent_dim: 50
  measure_only: true
  measure_coef: 1.0
  measure_legs: critic
  fb_forward_z: true
  project_z: true
  measure_ortho_coef: 1000.0
  measure_test_states: goals
wandb_mode: online
```

- [ ] **Step 2: Write N2's config**

Create `configs/gciql/cube_single_state_gciql_flow_bilinear_d50_C_taskproj.yaml`:

```yaml
# N2 (sf-design-space 1e): the same task-norm cell, but with the test set collapsed to RANK 1 --
# each residual is projected onto its OWN task vector z_i = psi(g_i) instead of against all B
# test directions.
#
# Its STATIONARY POINT (not its loss) is a scalar TD. Minimizing
#   0.5*(Q - gamma*Qbar)^2 - <F, psi(s')>/sqrt(d)
# over F and projecting the gradient onto z (with ||z||^2 = d under project_z) gives
#   Q = <psi(s'),psi(g)>/d + gamma*Qbar
# i.e. GCIQL's backup with the LEARNED KERNEL reward in place of the true indicator 1[s=g]-1.
# The exact midpoint between the FB measure (0.231) and bilinear GCIQL (0.904).
#
# Reading it:  ~0.90 => the RANK of the test matrix was the failure (fitting the full density)
#              ~0.23 => the KERNEL reward was the failure, not the rank
#              ~0.12 => the DIAGONAL ESTIMATOR dominates (the cell-A effect) and N2 says nothing
#                       about the reward; a fourth cell would be needed to separate them.
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
run_group: GCIQL-State-Flow-Bilinear-d50-C-TaskProj
restore_path: null
restore_epoch: null
exp_name: null
agent_overrides:
  bc_coeff: 3.0
  value_arch: bilinear
  value_latent_dim: 50
  measure_only: true
  measure_coef: 1.0
  measure_legs: critic
  fb_forward_z: true
  project_z: true
  measure_ortho_coef: 1000.0
  measure_test_states: goals
  measure_test_rank: task
wandb_mode: online
```

- [ ] **Step 3: Verify both configs construct an agent (guards accept them)**

Run:

```bash
cd /home/adityamohan/git/Factored-FB/third_party/ogbench/impls && /dev/shm/.venv-jax/bin/python -c "
import sys, yaml; sys.path.insert(0,'/home/adityamohan/git/Factored-FB/tools/wandb_mode_shim')
import jax
from gciql_flow import GCIQLFlowAgent, get_config
for n in ['C_tasknorm','C_taskproj']:
    p='/home/adityamohan/git/Factored-FB/configs/gciql/cube_single_state_gciql_flow_bilinear_d50_%s.yaml'%n
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
ok C_tasknorm
ok C_taskproj
```

- [ ] **Step 4: Register both groups in `results_map.py`**

In `scripts/results_map.py`, immediately **above** the line beginning
`# --- 1a: joint-psi SF, FB ortho + sphere`, insert:

```python
    # --- 1e: task-weighted successor moments (research note). N1 swaps the measure loss's B x B
    # test states from rho (next_observations) to nu (value_goals); N2 additionally collapses the
    # test set to rank 1. Ortho and Dirac stay on psi(next_observations) in both.
    'GCIQL-State-Flow-Bilinear-d50-C-TaskNorm': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='C-tasknorm', graft='FB measure, residual tested in the TASK norm (test states ~ value_goals) instead of the replay norm'),
    'GCIQL-State-Flow-Bilinear-d50-C-TaskProj': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='C-taskproj', graft='+ rank-1 test set: project each residual onto its own z_i=psi(g_i); fixed point = scalar TD with a kernel reward'),
```

- [ ] **Step 5: Verify the map still accounts for every group**

Run: `.venv/bin/python scripts/compile_results_csvs.py 2>&1 | tail -3`
Expected: `[compile] audit -> .../compile_audit.txt  (all groups accounted for)` and exit 0. (The two new groups appear under `MAPPED BUT ABSENT FROM WANDB` until the runs start — a note, not a failure.)

- [ ] **Step 6: Write the launcher**

Create `scripts/launch_tasknorm.sh` and `chmod +x` it:

```bash
#!/usr/bin/env bash
# Task-weighted successor moments (spec docs/superpowers/specs/2026-07-10-task-weighted-moments-design.md).
#
#   N1  C_tasknorm   measure residual tested against psi(value_goals) (the task norm nu)
#                    -> vs C-fbz-projz-ortho1000 = 0.231 +/- 0.030 (same cell, test states = rho)
#   N2  C_taskproj   + rank-1: project each residual onto its own z_i = psi(g_i)
#                    -> fixed point is GCIQL's backup with a learned kernel reward
#
# ALWAYS in tmux:
#   tmux new -d -s tasknorm 'DRY_RUN=0 scripts/launch_tasknorm.sh 2>&1 | tee /dev/shm/tasknorm_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(pwd)/.venv/bin/python}   # run_gciql.py is a hydra launcher: it runs HERE and
                                    # subprocesses ogbench main.py into .venv-jax itself.
                                    # Do NOT point this at .venv-jax -- that venv has no hydra.
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-500000}
LOGDIR=/dev/shm/tasknorm_logs
mkdir -p "$LOGDIR"

configs=(
  cube_single_state_gciql_flow_bilinear_d50_C_tasknorm
  cube_single_state_gciql_flow_bilinear_d50_C_taskproj
)

JOBS=()
for cfg in "${configs[@]}"; do
  for s in 0 1 2; do
    JOBS+=("${cfg}__s${s}|||--config-name $cfg seed=$s train_steps=$STEPS save_interval=$STEPS exp_name=${cfg}_s${s}")
  done
done

GPUS=(${GPUS:-0 1 2 3 4 5})
echo "task-norm moments: ${#JOBS[@]} runs @${STEPS} over ${#GPUS[@]} GPUs"
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

- [ ] **Step 7: Dry-run the launcher (and stop)**

Run: `chmod +x scripts/launch_tasknorm.sh && scripts/launch_tasknorm.sh`
Expected: `task-norm moments: 6 runs @500000 over 6 GPUs` followed by 6 `GPU<n>  <cfg>__s<seed>` lines. **No processes started.**

- [ ] **Step 8: Commit**

```bash
git add configs/gciql/cube_single_state_gciql_flow_bilinear_d50_C_tasknorm.yaml \
        configs/gciql/cube_single_state_gciql_flow_bilinear_d50_C_taskproj.yaml \
        scripts/results_map.py scripts/launch_tasknorm.sh
git commit -m "feat(configs): task-norm measure cells N1 (C-TaskNorm) and N2 (C-TaskProj) + launcher

N1: FB measure with its B x B test states drawn from the task distribution nu (value_goals)
instead of the replay marginal rho. Identical to C-fbz-projz-ortho1000 (0.231 +/- 0.030) except
measure_test_states. The ortho stays on psi(next_observations), which is what keeps C_nu != I.

N2: + rank-1 test set. Its stationary point is GCIQL's backup with the learned kernel reward
<psi(s'),psi(g)>/sqrt(d) in place of the true indicator -- the midpoint between the measure (0.231)
and bilinear GCIQL (0.904).

Distinct wandb groups, both registered in results_map. NOT LAUNCHED: a code audit and a run
manifest come first."
```

---

## Self-Review

**Spec coverage.** §1 the reduction → the config-knob docstrings and both config headers.
§1.2 the ortho must not move → Task 1 Step 4 (ortho untouched) + Task 1 test 3.
§2 the decoupled loss → Task 1 Step 4. §2.1 N2's fixed point → N2's config header and the commit
message. §3 the two cells → Task 3 configs. §3.1 the two deliberate choices (Dirac on ψ(s′);
exclude `j=i`) → Task 1 Step 4 (`off[None]` mask retains the `i≠j` exclusion) + Task 1 test 5.
§4 knobs → Task 1 Step 3; guards → Task 2. §5 tests 1–6 → Task 1 (tests 1,2,3,4,5) and Task 2
(test 6); test 7 "existing suites green" → Task 1 Step 6 and Task 2 Step 5.
§6 configs → Task 3 Steps 1–2. §7 diagnostics → `measure_qz_mean` in Task 1 Step 4.
§8 anchors → both config headers. §9 sequencing → Task 3 ends at a dry-run, per the user's
audit-first requirement.

**Deviation from the spec, flagged here:** §7 asks that `measure_orth` stay at the ρ-whitened
floor. That is a *runtime* observation, not a code change; it is asserted structurally instead, by
Task 1 test 3 (perturbing `value_goals` must not move `measure_orth`). Also, under `scale_norm`
the split path normalizes both terms by `mean(M_t²)` (the test matrix) rather than by `mean(M²)`;
`scale_norm` is `False` in both cells, so this is inert, and it is the only self-consistent choice
once the two matrices differ.

**Type/name consistency.** Knobs spelled `measure_test_states` and `measure_test_rank` identically
in `get_config()`, `measure_loss`, the guards, all six tests, both YAMLs, and `results_map`.
Info keys `measure_loss`, `measure_offdiag`, `measure_diag`, `measure_orth` are the existing four;
`measure_qz_mean` is added only under `test_rank='task'`. Local names `psi_t`, `tpsi_t`, `M_t`,
`tM_t`, `diff_t`, `dir_i`, `qz`, `qz_mean` are introduced in Task 1 Step 4 and used nowhere else.
`_measure_terms(M, tM, gamma, scale_norm)` keeps its existing signature and is called only on the
legacy path. `off = 1.0 - jnp.eye(B)` and `sqd = jnp.sqrt(d)` already exist in `measure_loss`.

**No placeholders.** Every code step carries the code; every run step carries the command and its
expected output.
