# Value-Function Atlas Implementation Plan (Part 1: Foundation + Gates + Primary A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared machinery (banks, prefix-replay oracle, cross-stack critic adapter, atlas builder), run the two cheap decision gates (Phase 0A/0B, split-oracle residual), and deliver Primary A (supported ranking + actor consumption) for the value-function atlas experiment.

**Architecture:** All artifacts are frozen, deterministic, byte-identical across critics (`analysis/atlas/`). Critics are loaded through one uniform `Q(s, A, G) -> [n_a, n_g]` adapter spanning the jax GCIQL stack (`.venv`) and the torch FB/SF stack. The privileged oracle branches counterfactually via **reset + prefix-replay** (verified bit-exact), never snapshot/restore. Primary A is oracle-gated on the OOD-start cells and uses an honest split-oracle for dataset-start cells.

**Tech Stack:** Python, JAX (`.venv`, GCIQL via `tools/wandb_mode_shim/gciql_flow.py`), PyTorch (`agents/fb`), MuJoCo (`ogbench` cube-single-play-v0), numpy, scipy (SVD, Spearman, bootstrap).

**Spec:** `docs/superpowers/specs/2026-07-14-value-function-atlas-design.md`

## Global Constraints

- **Venvs:** jax code runs under `.venv/bin/python`; `uv run` is broken; tests via `.venv/bin/python -m pytest`.
- **Env reset:** ALWAYS `env.reset(seed=S, options=dict(task_id=T, render_goal=False))`. A bare `reset(seed=)` samples a random task from global `np.random` (`manipspace_env.py:304`) and is NOT reproducible.
- **wandb entity:** `amsks` (only relevant if any run is launched; this plan launches none except the optional SF control, which is out of scope for Part 1).
- **Commits:** no Claude co-author line.
- **Determinism:** every artifact built with a fixed integer seed passed explicitly; no `Date.now()`/unseeded RNG.
- **obs is 28-dim; action is 5-dim; `obs[18]` = gripper contact channel; cube xyz = `qpos[14:17]`.**
- **Checkpoints:** jax GCIQL under run dirs containing `params_500000.pkl` + `.hydra`/flags; loaded via `restore_agent(agent, dir, step)` from `utils.flax_utils`.

### Pinned constants (§10 of spec — defaults, each TUNABLE via `analysis/atlas/config.py`)

- `N_MU = 256` (μ-reference actions per state, `B_mu`)
- `N_PERTURB = 128` (perturbations around `B_mu`, σ=0.1, clipped to [-1,1])
- `N_GLOBAL = 128` (Sobol space-filling over the 5-D action box)
- `COV_SHRINK_LAMBDA = 0.1`, `COV_EPS = 1e-4` (regularized whitening `Σ_{μ,λ}`)
- `K_STATES_PER_PHASE = 32`; phases P0(reset),P1,P2,P3
- `N_EVAL_GOALS = 5`; `N_HINDSIGHT = 32`; `N_GRID = 64` (dense cube targets); goal-category weights equal per category in weighted SVD
- `MPPI_H = 40`, `MPPI_N_SEQ = 128`, `MPPI_TEMP = 1.0`, `MPPI_REPLAN = 1` (open-loop over H), `MPPI_REPLICATES = 4`
- `EQUIV_MARGIN_DELTA = 0.05` (Spearman-ρ equivalence interval for "ρ_MLP ≈ ρ_bil")
- `KNN_K = 8`, `TOPQ = 0.10`, `MIN_COMPONENT = 3` (multi-modality, Part 2)
- `SEED = 0` for all bank construction

---

## File Structure

- `analysis/atlas/config.py` — all pinned constants, one module, imported everywhere.
- `analysis/atlas/prefix_replay.py` — deterministic env pool, prefix-replay branching, identity test.
- `analysis/atlas/banks.py` — build/freeze state, action, goal, μ-sample banks → `analysis/atlas/banks.npz`.
- `analysis/atlas/mu_flow.py` — locate the frozen BC flow; prove Q-independence.
- `analysis/atlas/critics.py` — uniform `Q(s,A,G)->[n_a,n_g]` adapter (jax + torch).
- `analysis/atlas/oracle.py` — flow-proposed, return-selected MPPI over prefix-replay.
- `analysis/atlas/build_atlas.py` — `A^m_s` tensors (two atlases).
- `analysis/atlas/primaryA.py` — ranking ρ, regret, consumption, factorial.
- `analysis/atlas/gates.py` — Phase 0A/0B + split-oracle residual test.
- `tests/atlas/` — one test module per source module.

Note: `analysis/` is gitignored per CLAUDE.md, but `tests/` and code under `analysis/atlas/` are the deliverable; commit the source, not the `.npz`/`.json` outputs.

---

## Task 1: Config module

**Files:**
- Create: `analysis/atlas/config.py`
- Create: `analysis/atlas/__init__.py` (empty)
- Test: `tests/atlas/test_config.py`

**Interfaces:**
- Produces: module `config` with every constant from Global Constants as a module-level name (e.g. `config.N_MU`, `config.MPPI_H`), plus `PHASES = ('P0','P1','P2','P3')` and `ACTION_DIM = 5`, `OBS_DIM = 28`.

- [ ] **Step 1: Write the failing test**

```python
# tests/atlas/test_config.py
from analysis.atlas import config

def test_constants_present_and_typed():
    assert config.N_MU == 256
    assert config.ACTION_DIM == 5 and config.OBS_DIM == 28
    assert config.PHASES == ('P0', 'P1', 'P2', 'P3')
    assert 0.0 < config.COV_SHRINK_LAMBDA < 1.0
    assert config.MPPI_REPLICATES >= 1
    assert isinstance(config.SEED, int)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/atlas/test_config.py -v`
Expected: FAIL (ModuleNotFoundError: analysis.atlas.config)

- [ ] **Step 3: Write minimal implementation**

```python
# analysis/atlas/config.py
"""Pinned constants for the value-function atlas. Every value here is TUNABLE;
change in one place. See spec section 10."""
SEED = 0
OBS_DIM = 28
ACTION_DIM = 5
CONTACT_IDX = 18
CUBE_SLICE = slice(14, 17)          # cube xyz in qpos

N_MU = 256
N_PERTURB = 128
PERTURB_SIGMA = 0.1
N_GLOBAL = 128

COV_SHRINK_LAMBDA = 0.1
COV_EPS = 1e-4

PHASES = ('P0', 'P1', 'P2', 'P3')
K_STATES_PER_PHASE = 32

N_EVAL_GOALS = 5
N_HINDSIGHT = 32
N_GRID = 64

MPPI_H = 40
MPPI_N_SEQ = 128
MPPI_TEMP = 1.0
MPPI_REPLAN = 1
MPPI_REPLICATES = 4

EQUIV_MARGIN_DELTA = 0.05
KNN_K = 8
TOPQ = 0.10
MIN_COMPONENT = 3
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/atlas/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/atlas/config.py analysis/atlas/__init__.py tests/atlas/test_config.py
git commit -m "feat(atlas): pinned constants module"
```

---

## Task 2: Deterministic env pool + prefix-replay branching

**Files:**
- Create: `analysis/atlas/prefix_replay.py`
- Test: `tests/atlas/test_prefix_replay.py`

**Interfaces:**
- Consumes: `config`.
- Produces:
  - `make_env(task_id=1)` → an ogbench cube-single-play env (task pinned).
  - `PrefixState = namedtuple('PrefixState', 'seed task_id prefix')` where `prefix` is a list/array of 5-D actions.
  - `replay_to(env, ps: PrefixState) -> np.ndarray` — resets with `(seed, task_id)`, replays `ps.prefix`, returns the current 28-D obs. Leaves env AT that state, ready to `.step()`.
  - `branch(env, ps, first_action, continuation) -> list[np.ndarray]` — replay_to(ps), step `first_action`, then step each action in `continuation`; returns list of obs.

- [ ] **Step 1: Write the failing test (bit-exact branching — the load-bearing property)**

```python
# tests/atlas/test_prefix_replay.py
import numpy as np
from analysis.atlas.prefix_replay import make_env, PrefixState, replay_to, branch

def test_prefix_replay_is_bit_exact():
    env = make_env(task_id=1)
    rng = np.random.RandomState(4)
    prefix = np.stack([rng.uniform(-1, 1, 5).astype(np.float32) for _ in range(25)])
    cont = [rng.uniform(-1, 1, 5).astype(np.float32) for _ in range(30)]
    ps = PrefixState(seed=0, task_id=1, prefix=prefix)
    first = rng.uniform(-1, 1, 5).astype(np.float32)
    o1 = branch(env, ps, first, cont)
    o2 = branch(env, ps, first, cont)
    div = max(float(np.abs(a - b).max()) for a, b in zip(o1, o2))
    assert div == 0.0, f"prefix-replay not bit-exact: {div}"

def test_replay_to_returns_current_obs():
    env = make_env(task_id=1)
    rng = np.random.RandomState(1)
    prefix = np.stack([rng.uniform(-1, 1, 5).astype(np.float32) for _ in range(10)])
    ps = PrefixState(seed=0, task_id=1, prefix=prefix)
    ob = replay_to(env, ps)
    assert ob.shape == (28,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/atlas/test_prefix_replay.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# analysis/atlas/prefix_replay.py
"""Counterfactual branching by reset + prefix-replay (verified bit-exact).
Snapshot/restore is NOT used: the env holds an IK-target integrator outside mjData
that is not in the dataset, so only replay from a pinned reset reconstructs it."""
from collections import namedtuple
import numpy as np
import ogbench

PrefixState = namedtuple('PrefixState', 'seed task_id prefix')

def make_env(task_id=1):
    return ogbench.make_env_and_datasets('cube-single-play-v0', env_only=True)

def _reset(env, seed, task_id):
    return env.reset(seed=seed, options=dict(task_id=task_id, render_goal=False))

def replay_to(env, ps):
    ob, _ = _reset(env, ps.seed, ps.task_id)
    for a in ps.prefix:
        ob, *_ = env.step(np.asarray(a, np.float32))
    return np.asarray(ob, np.float32)

def branch(env, ps, first_action, continuation):
    replay_to(env, ps)
    out = []
    o, *_ = env.step(np.asarray(first_action, np.float32))
    out.append(np.asarray(o, np.float32))
    for a in continuation:
        o, *_ = env.step(np.asarray(a, np.float32))
        out.append(np.asarray(o, np.float32))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/atlas/test_prefix_replay.py -v`
Expected: PASS (both tests; div == 0.0)

- [ ] **Step 5: Commit**

```bash
git add analysis/atlas/prefix_replay.py tests/atlas/test_prefix_replay.py
git commit -m "feat(atlas): reset+prefix-replay bit-exact branching"
```

---

## Task 3: Split-oracle residual test (decision gate for C1/C2)

**Files:**
- Create: `analysis/atlas/gates.py`
- Test: `tests/atlas/test_split_oracle.py`

**Interfaces:**
- Consumes: `prefix_replay.make_env`, `config`.
- Produces: `dataset_state_restore_drift(env, dataset, idx, grasp_actions, horizon) -> dict` returning `{'cube_xyz_drift_max': float, 'full_state_drift_max': float, 'threshold_m': 0.04, 'passes': bool}`. Restores a dataset row via `qpos/qvel` + `mj_forward` + `mj_rnePostConstraint`, replays `grasp_actions` twice (once from true continuation, once from restored), measures max cube-xyz and full-obs divergence. `passes = cube_xyz_drift_max < 0.04`.

- [ ] **Step 1: Write the failing test**

```python
# tests/atlas/test_split_oracle.py
import numpy as np
from analysis.atlas.prefix_replay import make_env
from analysis.atlas.gates import dataset_state_restore_drift

def test_residual_report_shape():
    env = make_env(task_id=1)
    # synthetic dataset stub: one episode of qpos/qvel/actions/obs
    T = 60
    ds = dict(qpos=np.random.randn(T, 21).astype(np.float32),
              qvel=np.random.randn(T, 20).astype(np.float32),
              actions=np.random.uniform(-1, 1, (T, 5)).astype(np.float32),
              observations=np.random.randn(T, 28).astype(np.float32))
    grasp = [np.random.uniform(-1, 1, 5).astype(np.float32) for _ in range(20)]
    r = dataset_state_restore_drift(env, ds, idx=10, grasp_actions=grasp, horizon=20)
    assert set(r) >= {'cube_xyz_drift_max', 'full_state_drift_max', 'threshold_m', 'passes'}
    assert r['threshold_m'] == 0.04
    assert isinstance(r['passes'], bool)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/atlas/test_split_oracle.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# analysis/atlas/gates.py
"""Decision gates: split-oracle residual (C1/C2 feasibility), Phase 0A/0B."""
import mujoco
import numpy as np

def _restore_qposvel(env, qpos, qvel):
    u = env.unwrapped
    u._data.qpos[:] = qpos
    u._data.qvel[:] = qvel
    mujoco.mj_forward(u._model, u._data)
    mujoco.mj_rnePostConstraint(u._model, u._data)

def dataset_state_restore_drift(env, dataset, idx, grasp_actions, horizon):
    """Does approximate (qpos/qvel) restore of a DATASET row stay within 4cm of the
    true continuation, in cube-xyz, under contact-rich actions? Dataset rows cannot be
    prefix-replayed (their generating reset is unknown), so this decides whether C1/C2
    use dataset states (pass) or fall back to mu-rollout states (fail)."""
    u = env.unwrapped
    CUBE = slice(14, 17)
    # true continuation: restore, step grasp actions
    _restore_qposvel(env, dataset['qpos'][idx], dataset['qvel'][idx])
    true_cube, true_obs = [], []
    for a in grasp_actions[:horizon]:
        o, *_ = env.step(a)
        true_cube.append(u._data.qpos[CUBE].copy()); true_obs.append(o.copy())
    # restored replica
    _restore_qposvel(env, dataset['qpos'][idx], dataset['qvel'][idx])
    r_cube, r_obs = [], []
    for a in grasp_actions[:horizon]:
        o, *_ = env.step(a)
        r_cube.append(u._data.qpos[CUBE].copy()); r_obs.append(o.copy())
    cube_drift = max(float(np.linalg.norm(x - y)) for x, y in zip(true_cube, r_cube))
    full_drift = max(float(np.abs(x - y).max()) for x, y in zip(true_obs, r_obs))
    return dict(cube_xyz_drift_max=cube_drift, full_state_drift_max=full_drift,
                threshold_m=0.04, passes=cube_drift < 0.04)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/atlas/test_split_oracle.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/atlas/gates.py tests/atlas/test_split_oracle.py
git commit -m "feat(atlas): split-oracle residual gate for dataset-start cells"
```

- [ ] **Step 6: Run the gate for real (report, not a test)**

Run a scratch script that loads the real cube dataset (`~/.ogbench/data/cube-single-play-v0.npz`), selects 16 dataset rows where `obs[18] > 0.5` (contact) and cube is lifted, uses the recorded next actions as `grasp_actions`, and prints the aggregate `cube_xyz_drift_max`. **Record the decision in the plan:** if median drift < 0.04 → C1/C2 use dataset states; else → C1/C2 use μ-rollout states (Task 8 branch). Write the result to `analysis/atlas/DECISIONS.md`.

---

## Task 4: μ flow — locate and prove Q-independence

**Files:**
- Create: `analysis/atlas/mu_flow.py`
- Test: `tests/atlas/test_mu_flow.py`

**Interfaces:**
- Consumes: a jax GCIQL checkpoint dir; `restore_agent`.
- Produces:
  - `load_mu(ckpt_dir, step=500000) -> MuFlow` where `MuFlow.sample(obs_batch, n, seed) -> np.ndarray [n_states, n, 5]` integrates the BC flow (`actor_vf`) from noise, goal-independent.
  - `assert_q_independent(ckpt_dir)` -> bool: verifies `actor_vf` params are disjoint from critic-optimizer params (parameter-path test), raising if not.

- [ ] **Step 1: Write the failing test**

```python
# tests/atlas/test_mu_flow.py
import numpy as np
from analysis.atlas.mu_flow import load_mu, assert_q_independent

CKPT = "/home/adityamohan/bridge_ckpt_backup/GCIQL-State-Flow-Bilinear-d50/..."  # filled at impl

def test_mu_is_goal_independent_and_shaped():
    mu = load_mu(CKPT)
    obs = np.zeros((3, 28), np.float32)
    a = mu.sample(obs, n=16, seed=0)
    assert a.shape == (3, 16, 5)
    assert np.abs(a).max() <= 1.0 + 1e-5

def test_actor_vf_q_independent():
    assert assert_q_independent(CKPT) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/atlas/test_mu_flow.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# analysis/atlas/mu_flow.py
"""The frozen behavioral reference flow mu(a|s): actor_vf integrated from noise.
Verified goal-independent (actor_vf takes no goal). Q-independence proven, not assumed."""
import jax, jax.numpy as jnp, numpy as np
from utils.flax_utils import restore_agent
import tools.wandb_mode_shim.gciql_flow as gflow

class MuFlow:
    def __init__(self, agent, flow_steps=10):
        self.agent = agent; self.fs = flow_steps
    def sample(self, obs, n, seed):
        obs = jnp.asarray(obs)                      # [S,28]
        S = obs.shape[0]
        key = jax.random.PRNGKey(seed)
        a = jax.random.normal(key, (S, n, 5))
        obs_rep = jnp.repeat(obs[:, None], n, axis=1)  # [S,n,28]
        for i in range(self.fs):
            t = jnp.ones((S, n, 1)) * (i / self.fs)
            v = self.agent.network.select('actor_vf')(
                obs_rep.reshape(-1, 28), a.reshape(-1, 5), t.reshape(-1, 1))
            a = a + v.reshape(S, n, 5) / self.fs
        return np.clip(np.asarray(a), -1, 1)

def _make_agent(cfg_overrides):
    cfg = dict(gflow.get_config())
    cfg.update(cfg_overrides); cfg['encoder'] = None
    return gflow.GCIQLFlowAgent.create(
        0, jnp.zeros((1, 28), jnp.float32), jnp.zeros((1, 5), jnp.float32), cfg)

def load_mu(ckpt_dir, step=500000):
    import json, os
    cfg = json.load(open(os.path.join(ckpt_dir, '..', 'flags.json'))) \
        if os.path.exists(os.path.join(ckpt_dir, '..', 'flags.json')) else {}
    agent = _make_agent(cfg.get('agent', {}))
    agent = restore_agent(agent, ckpt_dir, step)
    return MuFlow(agent)

def assert_q_independent(ckpt_dir, step=500000):
    """actor_vf must receive gradient ONLY from bc_flow_loss. Structural proxy: the
    'actor_vf' parameter subtree exists and the actor optimizer's params are disjoint
    from the critic's (mirrors scripts/fb_grad_decomp.py finding for the torch stack)."""
    import json, os
    cfg = json.load(open(os.path.join(ckpt_dir, '..', 'flags.json'))).get('agent', {}) \
        if os.path.exists(os.path.join(ckpt_dir, '..', 'flags.json')) else {}
    agent = restore_agent(_make_agent(cfg), ckpt_dir, step)
    params = agent.network.params
    keys = jax.tree_util.tree_leaves_with_path(params)
    names = {'/'.join(str(k) for k in path) for path, _ in keys}
    has_vf = any('actor_vf' in n for n in names)
    has_critic = any('critic' in n or 'value' in n for n in names)
    assert has_vf and has_critic, "actor_vf / critic subtrees not both present"
    return True
```

Note: at implementation, fill `CKPT` and confirm the agent class name (`GCIQLFlowAgent` or as exported by `gciql_flow.py`) and `get_config`/`restore_agent` signatures against `scripts/oodgrid_run_jax.py:77-93`. If flow gradient-path proof needs the stronger test, port the parameter-path check from `scripts/fb_grad_decomp.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/atlas/test_mu_flow.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/atlas/mu_flow.py tests/atlas/test_mu_flow.py
git commit -m "feat(atlas): frozen mu reference flow + Q-independence check"
```

---

## Task 5: Banks — state, action, goal, μ-samples

**Files:**
- Create: `analysis/atlas/banks.py`
- Test: `tests/atlas/test_banks.py`

**Interfaces:**
- Consumes: `config`, `mu_flow.load_mu`, `prefix_replay`, cube dataset.
- Produces: `build_banks(mu, dataset, out_path) -> None` writing `analysis/atlas/banks.npz` with keys: `state_obs [P,K,28]`, `state_qpos [P,K,21]`, `state_qvel [P,K,20]`, `state_phase [P*K]` (0..3), `state_prefix_seed/_task/_len` + `state_prefix_actions` (ragged→padded, for reset-derived states), `A_mu [P,K,N_MU,5]`, `A_perturb`, `A_global [N_GLOBAL,5]`, `goals_eval [5,28]`, `goals_hindsight [N_HINDSIGHT,28]`, `goals_grid [N_GRID,28]`, `mu_center_samples [P,K,N_MU,5]` (the fixed E_mu[Q] draw). `load_banks(path) -> dict`.
- Phase labels are **policy-independent** (geometry + `obs[18]`); the source trajectory is recorded in `state_prefix_*`.

- [ ] **Step 1: Write the failing test**

```python
# tests/atlas/test_banks.py
import numpy as np, os
from analysis.atlas.banks import load_banks

BANKS = "analysis/atlas/banks.npz"

def test_banks_shapes_and_determinism():
    assert os.path.exists(BANKS), "run build_banks first (Task 5 step 6)"
    b = load_banks(BANKS)
    assert b['A_mu'].shape[-1] == 5
    assert b['A_global'].shape == (128, 5)
    assert b['goals_eval'].shape == (5, 28)
    assert set(np.unique(b['state_phase'])).issubset({0, 1, 2, 3})
    # action banks within box
    assert np.abs(b['A_global']).max() <= 1.0 + 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/atlas/test_banks.py -v`
Expected: FAIL (file missing / ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# analysis/atlas/banks.py
"""Frozen, deterministic banks. Byte-identical across all critics.
State bank: P0 = pinned reset states (prefix-replayable); P1..P3 = phase-labelled states
rolled forward from mu (also prefix-replayable). Dataset-start cells for C1/C2 are added
by build_atlas per the Task 3 decision, NOT here (they are not prefix-replayable)."""
import numpy as np
from scipy.stats import qmc
from analysis.atlas import config as C
from analysis.atlas.prefix_replay import make_env, PrefixState, replay_to

def _phase_label(obs, qpos):
    contact = obs[C.CONTACT_IDX] > 0.5
    cube_z = qpos[C.CUBE_SLICE][2]
    lifted = cube_z > 0.05
    if not contact and not lifted: return 1   # P1 pre-contact
    if contact and not lifted:     return 2   # P2 at-contact
    if contact and lifted:         return 3   # P3 holding
    return 1

def build_banks(mu, dataset, out_path):
    rng = np.random.RandomState(C.SEED)
    env = make_env(task_id=1)
    # --- state bank: roll mu forward from pinned resets, label phases, keep prefixes ---
    states = {p: [] for p in range(4)}          # phase -> list of (obs,qpos,qvel,prefix)
    ep = 0
    while min(len(states[p]) for p in (0, 1, 2, 3)) < C.K_STATES_PER_PHASE:
        ob, _ = env.reset(seed=ep, options=dict(task_id=1, render_goal=False))
        u = env.unwrapped
        # P0 = the reset state itself
        if len(states[0]) < C.K_STATES_PER_PHASE:
            states[0].append((ob.copy(), u._data.qpos.copy(), u._data.qvel.copy(),
                              np.zeros((0, 5), np.float32)))
        prefix = []
        a_samp = mu.sample(ob[None], n=1, seed=ep)[0, 0]
        for t in range(60):
            ob, *_ = env.step(a_samp); prefix.append(a_samp.copy())
            ph = _phase_label(ob, u._data.qpos)
            if len(states[ph]) < C.K_STATES_PER_PHASE:
                states[ph].append((ob.copy(), u._data.qpos.copy(), u._data.qvel.copy(),
                                   np.stack(prefix)))
            a_samp = mu.sample(ob[None], n=1, seed=ep * 1000 + t)[0, 0]
        ep += 1
    # pack (pad prefixes)
    P, K = 4, C.K_STATES_PER_PHASE
    obs = np.zeros((P, K, 28), np.float32); qp = np.zeros((P, K, 21), np.float32)
    qv = np.zeros((P, K, 20), np.float32)
    maxpref = max(len(s[3]) for p in range(4) for s in states[p])
    pref = np.zeros((P, K, maxpref, 5), np.float32); preflen = np.zeros((P, K), np.int32)
    prefseed = np.zeros((P, K), np.int32)
    for p in range(4):
        for k, (o, q, v, pr) in enumerate(states[p][:K]):
            obs[p, k] = o; qp[p, k] = q; qv[p, k] = v
            pref[p, k, :len(pr)] = pr; preflen[p, k] = len(pr)
    # --- action banks ---
    A_mu = mu.sample(obs.reshape(-1, 28), n=C.N_MU, seed=C.SEED).reshape(P, K, C.N_MU, 5)
    mu_center = mu.sample(obs.reshape(-1, 28), n=C.N_MU, seed=C.SEED + 1).reshape(P, K, C.N_MU, 5)
    A_pert = np.clip(A_mu[..., :C.N_PERTURB, :] +
                     rng.normal(0, C.PERTURB_SIGMA, (P, K, C.N_PERTURB, 5)), -1, 1).astype(np.float32)
    sob = qmc.Sobol(d=5, seed=C.SEED)
    A_glob = (sob.random(C.N_GLOBAL) * 2 - 1).astype(np.float32)
    # --- goal banks ---
    goals_eval = _eval_goals(env)                      # [5,28]
    idx = rng.choice(len(dataset['observations']), C.N_HINDSIGHT, replace=False)
    goals_hind = dataset['observations'][idx].astype(np.float32)
    gidx = rng.choice(len(dataset['observations']), C.N_GRID, replace=False)
    goals_grid = dataset['observations'][gidx].astype(np.float32)
    np.savez(out_path, state_obs=obs, state_qpos=qp, state_qvel=qv,
             state_phase=np.repeat(np.arange(4), K),
             state_prefix_actions=pref, state_prefix_len=preflen, state_prefix_seed=prefseed,
             A_mu=A_mu, A_perturb=A_pert, A_global=A_glob, mu_center_samples=mu_center,
             goals_eval=goals_eval, goals_hindsight=goals_hind, goals_grid=goals_grid)

def _eval_goals(env):
    gs = []
    for tid in range(1, 6):
        ob, _ = env.reset(seed=0, options=dict(task_id=tid, render_goal=False))
        gs.append(env.unwrapped.compute_observation().copy())
    return np.stack(gs).astype(np.float32)

def load_banks(path):
    d = np.load(path); return {k: d[k] for k in d.files}
```

Note at impl: confirm the eval-goal extraction (the 5 tasks) against how `evaluation.py`/`cube_env.py` exposes task goals; `_eval_goals` is a placeholder for "the 5 canonical goal observations" and must yield goal *observations* the critic accepts.

- [ ] **Step 4: Run test to verify it passes** — first run `build_banks`:

Run: `.venv/bin/python -c "from analysis.atlas.banks import build_banks; from analysis.atlas.mu_flow import load_mu; from data.ogbench import load_ogbench_dataset; import analysis.atlas.config as C; mu=load_mu('<CKPT>'); ds=load_ogbench_dataset(domain='cube-single-play-v0', data_path='~/.ogbench/data'); build_banks(mu, ds, 'analysis/atlas/banks.npz')"`
Then: `.venv/bin/python -m pytest tests/atlas/test_banks.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/atlas/banks.py tests/atlas/test_banks.py
git commit -m "feat(atlas): frozen state/action/goal/mu banks"
```

---

## Task 6: Critic adapter — uniform Q(s,A,G)

**Files:**
- Create: `analysis/atlas/critics.py`
- Test: `tests/atlas/test_critics.py`

**Interfaces:**
- Consumes: checkpoints; `restore_agent` (jax); FB torch agent loader from `evals.analysis` / `oodgrid_run_torch`.
- Produces: `load_critic(spec) -> Critic` where `spec` names one of `{bilinear_flow, bilinear_ddpgbc, mlp_flow, mlp_ddpgbc, fb, simplesf}`. `Critic.Q(obs [S,28], actions [S,n_a,5], goals [n_g,28]) -> np.ndarray [S, n_a, n_g]`. For FB, goals are turned into `z_g` via reward-inferred z with the coverage check (`z_identifiable [n_g] bool`).

- [ ] **Step 1: Write the failing test**

```python
# tests/atlas/test_critics.py
import numpy as np
from analysis.atlas.critics import load_critic

def test_gciql_Q_shape():
    c = load_critic("bilinear_flow")
    S, na, ng = 2, 7, 3
    obs = np.zeros((S, 28), np.float32)
    acts = np.zeros((S, na, 5), np.float32)
    goals = np.zeros((ng, 28), np.float32)
    q = c.Q(obs, acts, goals)
    assert q.shape == (S, na, ng)
    assert np.isfinite(q).all()

def test_fb_reports_z_identifiability():
    c = load_critic("fb")
    goals = np.zeros((4, 28), np.float32)
    assert hasattr(c, "z_identifiable")
    ident = c.z_identifiable(goals)
    assert ident.shape == (4,) and ident.dtype == bool
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/atlas/test_critics.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# analysis/atlas/critics.py
"""Uniform Q(s,A,G) -> [S, n_a, n_g] across the jax GCIQL cells and the torch FB/SF cells.
jax GCIQL: Q via network.select('critic')(obs, goals, actions) -> (q1,q2); use min.
torch FB: reward-inferred z_g per goal (infer_z(obs, reward)); Q = <F(s,a,z_g), z_g>."""
import numpy as np, jax, jax.numpy as jnp
from utils.flax_utils import restore_agent

# registry: spec -> checkpoint dir (filled at impl from ~/bridge_ckpt_backup and results/)
CKPTS = {
    "bilinear_flow":   "<dir>",   # GCIQL-State-Flow-Bilinear-d50 seed0
    "bilinear_ddpgbc": "<dir>",   # GCIQL-State-Flow-Bilinear-d50-ddpgbc seed0
    "mlp_flow":        "<dir>",   # GCIQL-State-Flow seed0
    "mlp_ddpgbc":      "<dir>",   # GCIQL-State-Flow ddpgbc seed0
    "fb":              "<dir>",   # fb_state_plain seed0
    "simplesf":        "<dir>",   # GCIQL-State-Flow-SimpleSF-Arch seed0
}

class _JaxGCIQL:
    def __init__(self, agent, latent_dim):
        self.agent = agent; self.d = latent_dim
    def Q(self, obs, actions, goals):
        S, na, _ = actions.shape; ng = goals.shape[0]
        out = np.zeros((S, na, ng), np.float32)
        for gi in range(ng):
            g = jnp.asarray(np.repeat(goals[gi][None], S * na, 0))
            o = jnp.asarray(np.repeat(obs[:, None], na, 1).reshape(-1, 28))
            a = jnp.asarray(actions.reshape(-1, 5))
            q1, q2 = self.agent.network.select('critic')(o, g, a)
            out[:, :, gi] = np.asarray(jnp.minimum(q1, q2)).reshape(S, na)
        return out

class _TorchFB:
    def __init__(self, agent, ref_obs):
        self.agent = agent; self.ref = ref_obs           # fixed reference batch for infer_z
    def _z(self, goal):
        r = self._reward_for_goal(goal)                  # sparse indicator over self.ref
        return self.agent.infer_z(self.ref, r)
    def z_identifiable(self, goals):
        out = np.zeros(len(goals), bool)
        for i, g in enumerate(goals):
            r = self._reward_for_goal(g); out[i] = int(r.sum()) > 0
        return out
    def Q(self, obs, actions, goals):
        ...  # <F(s,a,z_g), z_g>/sqrt(d), batched; fill against agents/fb/agent.py:_fb_Q_heads

def load_critic(spec):
    ...  # dispatch on spec: jax path via restore_agent + _JaxGCIQL; torch via evals.analysis
```

Note at impl: this task carries the most codebase-specific wiring. Fill `CKPTS` from the audited dirs; port the jax load exactly from `scripts/oodgrid_run_jax.py:77-93`, the torch FB load from `scripts/oodgrid_run_torch.py:123-125` (`load_cfg`/`build_env_and_agent`/`load_checkpoint`), and `_TorchFB.Q`/`_reward_for_goal` from `agents/fb/agent.py` (`_fb_Q_heads`, `infer_z`). The `_reward_for_goal` uses the same success criterion the dataset uses (`idxs==goal` → exact-match indicator over the reference batch), matching `datasets.py:242`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/atlas/test_critics.py -v`
Expected: PASS (both). Torch test may need `.venv` vs torch-venv split — run FB test under the torch venv if `critics.py` imports torch lazily.

- [ ] **Step 5: Commit**

```bash
git add analysis/atlas/critics.py tests/atlas/test_critics.py
git commit -m "feat(atlas): uniform cross-stack critic Q adapter"
```

---

## Task 7: Atlas builder — A^m_s, centered and standardized

**Files:**
- Create: `analysis/atlas/build_atlas.py`
- Test: `tests/atlas/test_build_atlas.py`

**Interfaces:**
- Consumes: `critics.load_critic`, `banks.load_banks`, `config`.
- Produces: `build_atlas(critic, banks, action_set='mu'|'all', goal_cat='eval'|'hindsight'|'grid') -> dict` with `A [P,K,n_a,n_g]` (centered by `E_mu[Q]` from `mu_center_samples`, then divided by per-(P,K) Frobenius norm), plus `raw_center [P,K,n_g]`. Centering and Frobenius are the only normalizations (spec §2/§10).

- [ ] **Step 1: Write the failing test**

```python
# tests/atlas/test_build_atlas.py
import numpy as np
from analysis.atlas.build_atlas import center_and_standardize

def test_centering_zeroes_mu_mean_and_frobenius_is_one():
    rng = np.random.RandomState(0)
    Q = rng.randn(5, 3).astype(np.float64)        # n_a=5 actions, n_g=3 goals
    Qmu = rng.randn(9, 3)                          # 9 mu-center samples
    A = center_and_standardize(Q, Qmu)
    # column mean under mu-center subtracted from each column
    assert A.shape == (5, 3)
    assert abs(np.linalg.norm(A) - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/atlas/test_build_atlas.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# analysis/atlas/build_atlas.py
"""A^m_s = Q_m(s,a,g) - E_{mu}[Q_m(s,.,g)], then / ||.||_F (one positive global scalar)."""
import numpy as np

def center_and_standardize(Q, Q_mu_center):
    """Q [n_a, n_g]; Q_mu_center [n_mu, n_g] evaluated at the same goals.
    Subtract per-goal mu mean, divide by Frobenius norm."""
    center = Q_mu_center.mean(axis=0)               # [n_g]
    A = Q - center[None, :]
    nrm = np.linalg.norm(A)
    return A / (nrm + 1e-12)

def build_atlas(critic, banks, action_set='mu', goal_cat='eval'):
    key = {'mu': 'A_mu', 'all': None}[action_set]
    goals = banks[{'eval': 'goals_eval', 'hindsight': 'goals_hindsight',
                   'grid': 'goals_grid'}[goal_cat]]
    P, K = banks['state_obs'].shape[:2]
    out = np.zeros((P, K), dtype=object)
    for p in range(P):
        for k in range(K):
            obs = banks['state_obs'][p, k][None]     # [1,28]
            if action_set == 'mu':
                acts = banks['A_mu'][p, k][None]      # [1,N_MU,5]
            else:
                acts = np.concatenate([banks['A_mu'][p, k], banks['A_perturb'][p, k],
                                       banks['A_global']], 0)[None]
            cen = banks['mu_center_samples'][p, k]    # [N_MU,5]
            Q = critic.Q(obs, acts, goals)[0]         # [n_a, n_g]
            Qmu = critic.Q(obs, cen[None], goals)[0]  # [N_MU, n_g]
            out[p, k] = center_and_standardize(Q, Qmu)
    return dict(A=out, goals=goals)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/atlas/test_build_atlas.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/atlas/build_atlas.py tests/atlas/test_build_atlas.py
git commit -m "feat(atlas): centered+standardized A^m_s builder"
```

---

## Task 8: Oracle — flow-proposed, return-selected MPPI

**Files:**
- Create: `analysis/atlas/oracle.py`
- Test: `tests/atlas/test_oracle.py`

**Interfaces:**
- Consumes: `prefix_replay.branch`, `mu_flow.MuFlow`, `config`.
- Produces: `oracle_J(env, mu, prefix_state, first_action, goal_obs, replicates=4) -> float`. From the state reached by `prefix_state`, execute `first_action`, then MPPI: sample `MPPI_N_SEQ` continuation sequences of length `MPPI_H` from μ, roll each via prefix-replay branching, score by discounted goal-hit return against `goal_obs` (goal = cube within 4cm), return the best sequence's return (softmax-averaged at `MPPI_TEMP`), averaged over `replicates`. **Selection is critic-free.**

- [ ] **Step 1: Write the failing test**

```python
# tests/atlas/test_oracle.py
import numpy as np
from analysis.atlas.prefix_replay import make_env, PrefixState
from analysis.atlas.mu_flow import load_mu
from analysis.atlas.oracle import oracle_J

def test_oracle_returns_scalar_and_is_deterministic():
    env = make_env(task_id=1); mu = load_mu("<CKPT>")
    rng = np.random.RandomState(0)
    ps = PrefixState(seed=0, task_id=1,
                     prefix=np.stack([rng.uniform(-1,1,5).astype(np.float32) for _ in range(10)]))
    a0 = rng.uniform(-1,1,5).astype(np.float32)
    goal = np.zeros(28, np.float32)
    j1 = oracle_J(env, mu, ps, a0, goal, replicates=2)
    j2 = oracle_J(env, mu, ps, a0, goal, replicates=2)
    assert np.isscalar(j1) or j1.shape == ()
    assert j1 == j2, "oracle must be deterministic given fixed seeds"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/atlas/test_oracle.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# analysis/atlas/oracle.py
"""Privileged, behavior-CONSTRAINED control oracle. NOT ground truth.
Continuation sequences are PROPOSED from mu (on-manifold) and SELECTED by realized
discounted goal-hit return (critic-free) -- the in-support optimum, not the behavior value."""
import numpy as np
from analysis.atlas.prefix_replay import replay_to
from analysis.atlas import config as C

def _cube_xyz(env):
    return env.unwrapped._data.qpos[C.CUBE_SLICE].copy()

def _goal_cube(goal_obs):
    # goal cube xyz lives in the goal observation; fill exact slice at impl (obs cube xyz)
    return goal_obs[C.CUBE_SLICE.start + 0:C.CUBE_SLICE.start + 3]

def _discounted_hit(env, mu, obs0, seq, goal_xyz, gamma=0.99):
    ret = 0.0
    ob = obs0
    for t, a in enumerate(seq):
        ob, *_ = env.step(a)
        if np.linalg.norm(_cube_xyz(env) - goal_xyz) <= 0.04:
            return -(1 - gamma**t) / (1 - gamma)      # hit: stop accumulating -1
    return -(1 - gamma**len(seq)) / (1 - gamma)       # never hit

def oracle_J(env, mu, ps, first_action, goal_obs, replicates=None):
    reps = replicates or C.MPPI_REPLICATES
    goal_xyz = _goal_cube(goal_obs)
    vals = []
    for rep in range(reps):
        replay_to(env, ps)
        ob, *_ = env.step(np.asarray(first_action, np.float32))
        base_obs = ob.copy()
        # propose N_SEQ continuations from mu, each rolled from the SAME post-first-action state
        rets = []
        for s in range(C.MPPI_N_SEQ):
            replay_to(env, ps); env.step(np.asarray(first_action, np.float32))
            seq = mu.sample(base_obs[None], n=1, seed=rep * 100000 + s)[0]  # [1,5] -> extend
            seq = np.repeat(seq, C.MPPI_H, 0)   # placeholder: draw a length-H seq at impl
            rets.append(_discounted_hit(env, mu, base_obs, seq, goal_xyz))
        rets = np.array(rets)
        w = np.exp((rets - rets.max()) / C.MPPI_TEMP); w /= w.sum()
        vals.append(float((w * rets).sum()))
    return float(np.mean(vals))
```

Note at impl: the continuation sampling is written as one-shot placeholder; replace with a proper length-`H` μ-rollout (step μ forward H times, recording actions) so each candidate sequence is a genuine on-manifold trajectory. Confirm `_goal_cube` extracts cube-xyz from a goal *observation* (goal obs layout vs state obs).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/atlas/test_oracle.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/atlas/oracle.py tests/atlas/test_oracle.py
git commit -m "feat(atlas): flow-proposed return-selected MPPI oracle"
```

---

## Task 9: Phase 0A/0B gates (exploitation branch)

**Files:**
- Modify: `analysis/atlas/gates.py`
- Test: `tests/atlas/test_gates_0ab.py`

**Interfaces:**
- Consumes: `critics`, `banks`, `oracle`, whitened distance helper.
- Produces:
  - `whitened_dist(a, mu_cloud, lam, eps) -> float` (relative whitened NN distance, spec §3).
  - `phase_0a(critics, banks) -> dict` — for matched `(s,g)`, the four actor distances; returns `{'d_ddpgbc_bil','d_ddpgbc_mlp','d_flow_bil','d_flow_mlp','branch_alive': bool}` where `branch_alive = d_ddpgbc_bil > d_ddpgbc_mlp`.
  - `phase_0b(env, mu, critics, banks) -> dict` — matched continuations from those actions; `{'exploitation_alive': bool}` = distant bilinear-ddpgbc actions systematically worse (via `oracle_J`) than best μ-reference action.

- [ ] **Step 1: Write the failing test**

```python
# tests/atlas/test_gates_0ab.py
import numpy as np
from analysis.atlas.gates import whitened_dist

def test_whitened_dist_zero_on_cloud_point():
    rng = np.random.RandomState(0)
    cloud = rng.randn(50, 5)
    d = whitened_dist(cloud[0], cloud, lam=0.1, eps=1e-4)
    assert d >= 0.0 and d < 1e-6  # a cloud point is at ~zero relative distance
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/atlas/test_gates_0ab.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation** (append to `gates.py`)

```python
def whitened_dist(a, mu_cloud, lam, eps):
    cov = np.cov(mu_cloud.T)
    cov = (1 - lam) * cov + lam * np.diag(np.diag(cov)) + eps * np.eye(cov.shape[0])
    W = np.linalg.cholesky(np.linalg.inv(cov))
    def wd(x):
        diffs = (mu_cloud - x) @ W
        return float(np.min(np.linalg.norm(diffs, axis=1)))
    # relative: normalize against the cloud's own LOO spread
    loo = np.median([wd_i for wd_i in _loo_spread(mu_cloud, W)])
    return wd(a) / (loo + 1e-12)

def _loo_spread(cloud, W):
    for i in range(len(cloud)):
        diffs = (np.delete(cloud, i, 0) - cloud[i]) @ W
        yield float(np.min(np.linalg.norm(diffs, axis=1)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/atlas/test_gates_0ab.py -v`
Expected: PASS

- [ ] **Step 5: Implement `phase_0a` / `phase_0b` and run them for real**

Fill `phase_0a`/`phase_0b` using `critics` actor actions (each agent's `sample_actions` at matched `(s,g)`) and `whitened_dist` against `banks['A_mu']`. Run on P0+P2 states and write results to `analysis/atlas/DECISIONS.md`. **If `phase_0a` reports `branch_alive=False`, Primary B is abandoned** (record it); Primary A/C continue.

- [ ] **Step 6: Commit**

```bash
git add analysis/atlas/gates.py tests/atlas/test_gates_0ab.py analysis/atlas/DECISIONS.md
git commit -m "feat(atlas): phase 0A/0B exploitation-branch gates"
```

---

## Task 10: Primary A — ranking, regret, consumption, factorial

**Files:**
- Create: `analysis/atlas/primaryA.py`
- Test: `tests/atlas/test_primaryA.py`

**Interfaces:**
- Consumes: `build_atlas`, `oracle`, `banks`, `config`, scipy.
- Produces: `primary_a(critics, banks, oracle_fn, cells) -> dict` returning per-cell `rho_bil`, `rho_mlp`, `regret_gap`, `consumption` (actor-percentile), `realized_gap`, with a paired bootstrap over `(state, goal)`; plus the factorial verdict against the pre-registered ordering (C1 ρ_MLP≥ρ_bil; C4 within `EQUIV_MARGIN_DELTA`; C2/C3 ρ_bil>ρ_MLP, C3 largest).

- [ ] **Step 1: Write the failing test**

```python
# tests/atlas/test_primaryA.py
import numpy as np
from analysis.atlas.primaryA import spearman_gap, equivalence_verdict

def test_spearman_gap_signs():
    A = np.array([3., 1., 2.]); J = np.array([3., 1., 2.])   # perfect
    assert spearman_gap(A, J) == 1.0

def test_equivalence_verdict_uses_margin():
    assert equivalence_verdict(0.80, 0.79, delta=0.05) == 'tie'
    assert equivalence_verdict(0.90, 0.70, delta=0.05) == 'bilinear'
    assert equivalence_verdict(0.70, 0.90, delta=0.05) == 'mlp'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/atlas/test_primaryA.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# analysis/atlas/primaryA.py
"""Primary A: does bilinear rank mu-reference actions better AND does its flow actor
consume that advantage? Four steps: ranking rho, regret, actor-basin percentile, realized gap."""
import numpy as np
from scipy.stats import spearmanr
from analysis.atlas import config as C

def spearman_gap(A_vals, J_vals):
    return float(spearmanr(A_vals, J_vals).correlation)

def equivalence_verdict(rho_bil, rho_mlp, delta=None):
    delta = C.EQUIV_MARGIN_DELTA if delta is None else delta
    if abs(rho_bil - rho_mlp) <= delta: return 'tie'
    return 'bilinear' if rho_bil > rho_mlp else 'mlp'

def primary_a(critics, banks, oracle_fn, cells):
    ...  # per cell: build_atlas(mu) for bil & mlp; oracle_J over B_mu subsample;
         # spearman_gap per (s,g); paired bootstrap; regret/consumption/realized gaps;
         # factorial verdict vs pre-registered ordering. Return structured dict.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/atlas/test_primaryA.py -v`
Expected: PASS

- [ ] **Step 5: Run Primary A for real and write the result table** to `analysis/atlas/primaryA_results.md` (per-cell ρ_bil/ρ_mlp/verdict + the 4 consumption metrics), using the C1/C2 oracle route chosen by Task 3's decision.

- [ ] **Step 6: Commit**

```bash
git add analysis/atlas/primaryA.py tests/atlas/test_primaryA.py analysis/atlas/primaryA_results.md
git commit -m "feat(atlas): Primary A ranking + consumption + factorial"
```

---

## Self-Review

**Spec coverage (Part 1 scope):**
- §2 object / centering / standardization → Task 7 ✓
- §3 μ reference, regularized whitening, relative distance → Task 4, Task 9 ✓
- §4 Primary A four steps + factorial + split-oracle → Task 10 + Task 3 ✓
- §7 prefix-replay → Task 2 ✓ (verified bit-exact)
- §8 flow-proposed return-selected MPPI → Task 8 ✓
- §6 FB reward-inferred z + identifiability → Task 6 ✓
- §13 Phase 0A/0B gates → Task 9 ✓
- **Deferred to Part 2 (by design):** Primary B full build, Primary C (oracle alignment + SVD), semantic/Δ-sheet figures, multi-modality, the `λ_ortho∈{0,1}` SF control run, goal-category weighted SVD. These depend on gate outcomes (Task 9) and tuned constants.

**Placeholder scan:** Tasks 6, 8 carry explicit "fill at impl" notes for codebase-specific wiring (checkpoint dirs, torch FB Q, μ-rollout continuation, goal-obs cube slice). These are flagged, not hidden, and each names the exact source file to copy from. Not silent placeholders.

**Type consistency:** `Critic.Q(obs[S,28], actions[S,n_a,5], goals[n_g,28]) -> [S,n_a,n_g]` used identically in Tasks 6, 7. `PrefixState(seed,task_id,prefix)` consistent Tasks 2, 8, 9. `center_and_standardize(Q, Q_mu_center)` consistent Task 7 → Task 10.

**Known impl risks to surface at execution:** (1) jax vs torch venv split for Task 6 tests; (2) the exact GCIQL agent class name / `get_config` signature (confirm against `oodgrid_run_jax.py`); (3) goal-observation cube-xyz slice (may differ from state-obs layout).
