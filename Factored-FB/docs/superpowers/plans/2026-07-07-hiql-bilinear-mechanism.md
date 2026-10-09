# HIQL Bilinear Mechanism Knobs (E-B/E-C/E-D/E-E) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the discriminating cells of `docs/superpowers/specs/2026-07-07-hiql-bilinear-mechanism-design.md`: a Schaul-form bilinear knob (E-C), a two-head high-level-bilinear value split (E-D), their cube configs + launcher (incl. the config-only E-B `rep_dim` cell), and the antmaze generalization of the goal_rep probe (E-E).

**Architecture:** Two default-OFF config knobs on the vendored HIQL agent (`bilinear_form: rep|goal` on `_BilinearRepValue`; `value_split: shared|high_bilinear` adding a separately-TD-trained bilinear head used only by the high-level actor's advantage), mirrored in the `hiql_flow` subclass for the flow-both cell. Follows the existing `value_arch` precedent exactly (the vendored `third_party/ogbench/impls/agents/hiql.py` already carries our `_BilinearRepValue`).

**Tech Stack:** JAX/flax (OGBench impls), pytest driving the jax venv via `tests/jax_subprocess.py` (subprocess into `/dev/shm/.venv-jax`, `JAX_PLATFORMS=cpu`), hydra configs under `configs/gciql/`, bash launcher on the `launch_sota_bilinear_gaps.sh` pattern.

## Global Constraints

- All knobs **default-OFF**: `bilinear_form='rep'`, `value_split='shared'` — default param trees must be byte-identical in structure to today's (old checkpoints must keep restoring).
- Tests run from the torch venv: `.venv/bin/python -m pytest tests/<file> -v` (jax tests subprocess into `/dev/shm/.venv-jax`; they skip if it is absent).
- Git commits: NO Claude co-author trailer (repo rule).
- wandb: entity `amsks`, fresh `run_group` per new cell (never reuse a group).
- Launches always in tmux, `XLA_PYTHON_CLIENT_PREALLOCATE=false` (packing).
- Artifacts (`analysis/`, `PAPER/`, `docs/superpowers/`) are gitignored — commit only code/tests/configs/scripts.

## File Structure

- Modify `third_party/ogbench/impls/agents/hiql.py` — both knobs (module, losses, create, get_config).
- Modify `tools/wandb_mode_shim/hiql_flow.py` — mirror both knobs in the flow agent's own `create`/`high_actor_loss`/`total_loss`.
- Create `tests/test_hiql_bilinear_knobs.py` — all TDD for both files.
- Create `configs/gciql/cube_single_state_hiql_native_bilinear_d50_rep50_sg10.yaml` (E-B),
  `configs/gciql/cube_single_state_hiql_native_bilinear_d50_schaul_sg10.yaml` (E-C),
  `configs/gciql/cube_single_state_hiql_native_highbil_d50_sg10.yaml` (E-D native),
  `configs/gciql/cube_single_state_hiql_flow_both_highbil_d50_sg10.yaml` (E-D flow).
- Create `scripts/launch_hiql_mech.sh` — 4 configs × 3 seeds @500k.
- Modify `scripts/probe_hiql_goalrep.py` — dataset/goal-slice/action-dim/horizon args (E-E).

---

### Task 1: `bilinear_form` knob (E-C) in `hiql.py`

**Files:**
- Modify: `third_party/ogbench/impls/agents/hiql.py:14-35` (`_BilinearRepValue`), `:288-294` (create wiring), `:362-363` area (get_config)
- Test: `tests/test_hiql_bilinear_knobs.py`

**Interfaces:**
- Produces: `_BilinearRepValue(use_rep: bool = True, stop_rep_grad: bool = False)` — new flax fields consumed by Task 2/3. Config keys `bilinear_form: str = 'rep'` (`'rep'` = ψ(goal_rep([s;g])) option A; `'goal'` = ψ(raw g) Schaul form).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hiql_bilinear_knobs.py`:

```python
"""TDD for the HIQL bilinear mechanism knobs (spec 2026-07-07-hiql-bilinear-mechanism):

  bilinear_form: 'rep' (option A, psi(goal_rep([s;g]))) | 'goal' (Schaul, psi(raw g))
  value_split:   'shared' | 'high_bilinear' (separate TD-trained bilinear head that
                 ONLY the high-level actor advantage reads; MLP V keeps the TD/low
                 path and remains goal_rep's trainer)

Both default-OFF. Mirrors tests/test_hiql_flow_injection.py; runs in the jax venv.
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS

_BATCH = """
import jax, jax.numpy as jnp
ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
obs = jax.random.normal(jax.random.PRNGKey(1), (4, 11))
nobs = jax.random.normal(jax.random.PRNGKey(2), (4, 11))
act = jnp.tanh(jax.random.normal(jax.random.PRNGKey(3), (4, 5)))
batch = dict(observations=obs, next_observations=nobs, actions=act,
             rewards=jnp.zeros((4,)), masks=jnp.ones((4,)), value_goals=nobs,
             low_actor_goals=nobs, high_actor_goals=nobs, high_actor_targets=nobs)
def moved(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return any(bool(jnp.any(x != y)) for x, y in zip(la, lb))
"""


def _run(code):
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_bilinear_form_goal_rep_gradient_routing():
    """Schaul form must cut the value->goal_rep gradient; low_actor_rep_grad=True
    must restore a trainer; option-A 'rep' must keep training goal_rep."""
    _run(_BATCH + """
from agents.hiql import HIQLAgent, get_config
def make(form, low_rep_grad):
    cfg = dict(get_config()); cfg["encoder"] = None; cfg["frame_stack"] = None
    cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 16
    cfg["bilinear_form"] = form; cfg["low_actor_rep_grad"] = low_rep_grad
    return HIQLAgent.create(0, ex_obs, ex_act, cfg)
def rep(agent): return agent.network.params["modules_goal_rep"]
ag = make("goal", False); p0 = rep(ag); ag2, info = ag.update(batch)
assert jnp.isfinite(info["value/value_loss"])
assert not moved(p0, rep(ag2)), "form='goal'+no low rep grad: goal_rep must be FROZEN"
ag = make("goal", True); p0 = rep(ag); ag2, _ = ag.update(batch)
assert moved(p0, rep(ag2)), "low_actor_rep_grad=True must train goal_rep"
ag = make("rep", False); p0 = rep(ag); ag2, _ = ag.update(batch)
assert moved(p0, rep(ag2)), "option-A must keep training goal_rep through psi"
print("OK")
""")


def test_bilinear_form_default_rep_unchanged_tree():
    """Default config must not change the param tree (checkpoint compat)."""
    _run("""
import jax.numpy as jnp
from agents.hiql import HIQLAgent, get_config
cfg = dict(get_config()); cfg["encoder"] = None; cfg["frame_stack"] = None
assert cfg["bilinear_form"] == "rep" and cfg["value_split"] == "shared"
ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
agent = HIQLAgent.create(0, ex_obs, ex_act, cfg)
keys = set(agent.network.params.keys())
assert keys == {"modules_goal_rep", "modules_value", "modules_target_value",
                "modules_low_actor", "modules_high_actor"}, keys
print("OK")
""")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hiql_bilinear_knobs.py -v`
Expected: both FAIL — `test_..._gradient_routing` with `KeyError: 'bilinear_form'`-style stderr in the subprocess assert, `test_..._unchanged_tree` with `AssertionError` on `cfg["bilinear_form"]`.

- [ ] **Step 3: Implement the knob**

In `third_party/ogbench/impls/agents/hiql.py`, replace `_BilinearRepValue` (lines 14–35) with:

```python
class _BilinearRepValue(nn.Module):
    """HIQL bilinear value. bilinear_form='rep' (option A, use_rep=True):
    V(s,g) = <phi(s), psi(goal_rep([s;g]))> / sqrt(d) — keeps the subgoal rep, trained
    end-to-end through psi. bilinear_form='goal' (Schaul, use_rep=False):
    V(s,g) = <phi(s), psi(g)> / sqrt(d) — genuine low-rank over (s,g); goal_rep gets NO
    gradient from the value (pair with low_actor_rep_grad=True to keep it trained).
    stop_rep_grad detaches the rep input (value_split high head: never trains goal_rep).
    State-input path (no visual encoder)."""

    goal_rep: nn.Module
    hidden_dims: Any
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    use_rep: bool = True
    stop_rep_grad: bool = False

    def setup(self):
        mlp = ensemblize(MLP, 2) if self.ensemble else MLP
        self.phi = mlp((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.psi = mlp((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals):
        if self.use_rep:
            rep = self.goal_rep(jnp.concatenate([observations, goals], axis=-1))
            if self.stop_rep_grad:
                rep = jax.lax.stop_gradient(rep)
        else:
            rep = goals
        phi = self.phi(observations)
        psi = self.psi(rep)
        return (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)
```

In `create()` (lines 288–294), thread the form into both defs:

```python
        if config.get('value_arch', 'mlp') == 'bilinear':
            use_rep = config.get('bilinear_form', 'rep') == 'rep'
            value_def = _BilinearRepValue(
                goal_rep=goal_rep_def, hidden_dims=config['value_hidden_dims'],
                latent_dim=config['value_latent_dim'], layer_norm=config['layer_norm'],
                ensemble=True, use_rep=use_rep)
            target_value_def = _BilinearRepValue(
                goal_rep=goal_rep_def, hidden_dims=config['value_hidden_dims'],
                latent_dim=config['value_latent_dim'], layer_norm=config['layer_norm'],
                ensemble=True, use_rep=use_rep)
```

In `get_config()`, directly under the `value_latent_dim=512,` line add:

```python
            bilinear_form='rep',  # psi input when value_arch='bilinear': 'rep' (goal_rep([s;g]), option A) or 'goal' (raw g, Schaul).
            value_split='shared',  # 'shared' (one V both levels) or 'high_bilinear' (bilinear head for the high-actor advantage only; MLP V keeps TD/low and trains goal_rep).
```

(`value_split` is declared now so Task 1's default-tree test covers both keys; its behavior lands in Task 2.)

Also thread `use_rep` into `tools/wandb_mode_shim/hiql_flow.py:182-186` (its `create` builds its own value defs):

```python
        if config.get('value_arch', 'mlp') == 'bilinear':
            use_rep = config.get('bilinear_form', 'rep') == 'rep'
            value_def = _BilinearRepValue(goal_rep=goal_rep_def, hidden_dims=config['value_hidden_dims'],
                                          latent_dim=config['value_latent_dim'], layer_norm=config['layer_norm'],
                                          ensemble=True, use_rep=use_rep)
            target_value_def = _BilinearRepValue(goal_rep=goal_rep_def, hidden_dims=config['value_hidden_dims'],
                                                 latent_dim=config['value_latent_dim'], layer_norm=config['layer_norm'],
                                                 ensemble=True, use_rep=use_rep)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hiql_bilinear_knobs.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Regression-check the existing HIQL suites**

Run: `.venv/bin/python -m pytest tests/test_hiql_flow_injection.py -v`
Expected: all PASS (flow agent unaffected by a default-OFF knob).

- [ ] **Step 6: Commit**

```bash
git add third_party/ogbench/impls/agents/hiql.py tools/wandb_mode_shim/hiql_flow.py tests/test_hiql_bilinear_knobs.py
git commit -m "feat(hiql): bilinear_form knob - Schaul psi(g) vs option-A psi(goal_rep) (E-C)"
```

---

### Task 2: `value_split='high_bilinear'` two-head value (E-D) in `hiql.py`

**Files:**
- Modify: `third_party/ogbench/impls/agents/hiql.py` — `value_loss` (51–82, refactor), `high_actor_loss` (123–148), `total_loss` (150–168), `update` (179–190), `create` (285–349)
- Test: `tests/test_hiql_bilinear_knobs.py`

**Interfaces:**
- Consumes: `_BilinearRepValue(use_rep, stop_rep_grad)` from Task 1.
- Produces: network modules `high_value` / `target_high_value`; method `high_value_loss(batch, grad_params)`; info namespace `high_value/*`; helper `_ivl_value_loss(batch, grad_params, head)`. Task 3 mirrors these names in `hiql_flow.py` — keep them exact.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_hiql_bilinear_knobs.py`)

```python
def test_value_split_high_bilinear_builds_trains_and_keeps_rep_trainer():
    _run(_BATCH + """
from agents.hiql import HIQLAgent, get_config
cfg = dict(get_config()); cfg["encoder"] = None; cfg["frame_stack"] = None
cfg["value_split"] = "high_bilinear"; cfg["value_latent_dim"] = 16
cfg["bilinear_form"] = "goal"   # E-D cell: Schaul high head, zero rep coupling
agent = HIQLAgent.create(0, ex_obs, ex_act, cfg)
assert "modules_high_value" in agent.network.params
assert "modules_target_high_value" in agent.network.params
p0 = agent.network.params["modules_goal_rep"]
hv0 = agent.network.params["modules_high_value"]
for _ in range(2):
    agent, info = agent.update(batch)
assert jnp.isfinite(info["value/value_loss"])
assert jnp.isfinite(info["high_value/value_loss"]), sorted(info)
assert jnp.isfinite(info["high_actor/actor_loss"])
assert moved(p0, agent.network.params["modules_goal_rep"]), "MLP V must still train goal_rep"
assert moved(hv0, agent.network.params["modules_high_value"]), "high head must train by TD"
print("OK")
""")


def test_value_split_default_shared_no_extra_modules():
    _run(_BATCH + """
from agents.hiql import HIQLAgent, get_config
cfg = dict(get_config()); cfg["encoder"] = None; cfg["frame_stack"] = None
agent = HIQLAgent.create(0, ex_obs, ex_act, cfg)
assert "modules_high_value" not in agent.network.params
ag2, info = agent.update(batch)
assert not any(k.startswith("high_value/") for k in info), sorted(info)
print("OK")
""")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hiql_bilinear_knobs.py -v -k value_split`
Expected: `..._builds_trains...` FAILS (`modules_high_value` missing); `..._no_extra_modules` PASSES already (guards the default) — that is fine, it exists to catch regressions from this task's edits.

- [ ] **Step 3: Implement**

(a) Refactor `value_loss` into a head-parameterized helper (replace lines 51–82; body identical, only the two `select` names change):

```python
    def _ivl_value_loss(self, batch, grad_params, head):
        """IVL expectile TD loss for the named V head ('value' or 'high_value')."""
        (next_v1_t, next_v2_t) = self.network.select(f'target_{head}')(batch['next_observations'], batch['value_goals'])
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select(f'target_{head}')(batch['observations'], batch['value_goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select(head)(batch['observations'], batch['value_goals'], params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss (docstring of the original method moves here)."""
        return self._ivl_value_loss(batch, grad_params, 'value')

    def high_value_loss(self, batch, grad_params):
        """IVL TD loss for the high-level bilinear head (value_split='high_bilinear')."""
        return self._ivl_value_loss(batch, grad_params, 'high_value')
```

(b) `high_actor_loss` (lines 125–126) — switch the advantage head:

```python
        head = 'high_value' if self.config.get('value_split', 'shared') == 'high_bilinear' else 'value'
        v1, v2 = self.network.select(head)(batch['observations'], batch['high_actor_goals'])
        nv1, nv2 = self.network.select(head)(batch['high_actor_targets'], batch['high_actor_goals'])
```

(c) `total_loss` — after the `high_actor_loss` block (line 165), before the sum:

```python
        high_value_loss = 0.0
        if self.config.get('value_split', 'shared') == 'high_bilinear':
            high_value_loss, high_value_info = self.high_value_loss(batch, grad_params)
            for k, v in high_value_info.items():
                info[f'high_value/{k}'] = v

        loss = value_loss + low_actor_loss + high_actor_loss + high_value_loss
```

(d) `update` — after `self.target_update(new_network, 'value')` (line 188):

```python
        if self.config.get('value_split', 'shared') == 'high_bilinear':
            self.target_update(new_network, 'high_value')
```

(e) `create` — after the value/target defs (line 307), the high head (always `stop_rep_grad=True`: the MLP V must remain goal_rep's sole value-side trainer):

```python
        if config.get('value_split', 'shared') == 'high_bilinear':
            use_rep_high = config.get('bilinear_form', 'rep') == 'rep'
            high_value_def = _BilinearRepValue(
                goal_rep=goal_rep_def, hidden_dims=config['value_hidden_dims'],
                latent_dim=config['value_latent_dim'], layer_norm=config['layer_norm'],
                ensemble=True, use_rep=use_rep_high, stop_rep_grad=True)
            target_high_value_def = _BilinearRepValue(
                goal_rep=goal_rep_def, hidden_dims=config['value_hidden_dims'],
                latent_dim=config['value_latent_dim'], layer_norm=config['layer_norm'],
                ensemble=True, use_rep=use_rep_high, stop_rep_grad=True)
```

then after the `network_info = dict(...)` literal (line 338):

```python
        if config.get('value_split', 'shared') == 'high_bilinear':
            network_info['high_value'] = (high_value_def, (ex_observations, ex_goals))
            network_info['target_high_value'] = (target_high_value_def, (ex_observations, ex_goals))
```

and after `params['modules_target_value'] = params['modules_value']` (line 348):

```python
        if config.get('value_split', 'shared') == 'high_bilinear':
            params['modules_target_high_value'] = params['modules_high_value']
```

- [ ] **Step 4: Run the full knob suite + regressions**

Run: `.venv/bin/python -m pytest tests/test_hiql_bilinear_knobs.py tests/test_hiql_flow_injection.py -v`
Expected: all PASS (the refactor in (a) is behavior-preserving; the flow suite exercises the inherited `value_loss`).

- [ ] **Step 5: Commit**

```bash
git add third_party/ogbench/impls/agents/hiql.py tests/test_hiql_bilinear_knobs.py
git commit -m "feat(hiql): value_split=high_bilinear two-head value (E-D)"
```

---

### Task 3: Mirror `value_split` in `hiql_flow.py` (E-D flow-both cell)

**Files:**
- Modify: `tools/wandb_mode_shim/hiql_flow.py` — `high_actor_loss` (80–103), `total_loss` (105–119), `create` (193–236)
- Test: `tests/test_hiql_bilinear_knobs.py`

**Interfaces:**
- Consumes: `_BilinearRepValue(use_rep, stop_rep_grad)`, `self.high_value_loss(batch, grad_params)`, and the `high_value`/`target_high_value` module names from Tasks 1–2 (all inherited via `HIQLAgent`; `update` is inherited and already target-updates `high_value` conditionally).

- [ ] **Step 1: Write the failing test** (append to `tests/test_hiql_bilinear_knobs.py`)

```python
def test_hiql_flow_value_split_trains_finite_flow_both():
    _run(_BATCH + """
from hiql_flow import HIQLFlowAgent, get_config
cfg = dict(get_config()); cfg["encoder"] = None; cfg["frame_stack"] = None
cfg["flow_levels"] = "both"; cfg["value_split"] = "high_bilinear"
cfg["value_latent_dim"] = 16; cfg["bilinear_form"] = "goal"
agent = HIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
assert "modules_high_value" in agent.network.params
for _ in range(2):
    agent, info = agent.update(batch)
assert jnp.isfinite(info["high_value/value_loss"]), sorted(info)
assert jnp.isfinite(info["high_actor/actor_loss"])
a = agent.sample_actions(ex_obs, ex_obs, seed=jax.random.PRNGKey(0))
assert a.shape == (4, 5)
print("OK")
""")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hiql_bilinear_knobs.py -v -k flow_value_split`
Expected: FAIL — `modules_high_value` missing (the flow agent's own `create` never builds it).

- [ ] **Step 3: Implement**

(a) `high_actor_loss` flow branch (lines 83–84) — same head switch as stock:

```python
        head = 'high_value' if self.config.get('value_split', 'shared') == 'high_bilinear' else 'value'
        v1, v2 = self.network.select(head)(batch['observations'], batch['high_actor_goals'])
        nv1, nv2 = self.network.select(head)(batch['high_actor_targets'], batch['high_actor_goals'])
```

(b) `total_loss` — before the return (line 119):

```python
        high_value_loss = 0.0
        if self.config.get('value_split', 'shared') == 'high_bilinear':
            high_value_loss, high_value_info = self.high_value_loss(batch, grad_params)
            for k, v in high_value_info.items():
                info[f'high_value/{k}'] = v
        return value_loss + low_loss + high_loss + high_value_loss, info
```

(c) `create` — after the value/target defs (line 191), identical block to hiql.py Task 2(e):

```python
        if config.get('value_split', 'shared') == 'high_bilinear':
            use_rep_high = config.get('bilinear_form', 'rep') == 'rep'
            high_value_def = _BilinearRepValue(
                goal_rep=goal_rep_def, hidden_dims=config['value_hidden_dims'],
                latent_dim=config['value_latent_dim'], layer_norm=config['layer_norm'],
                ensemble=True, use_rep=use_rep_high, stop_rep_grad=True)
            target_high_value_def = _BilinearRepValue(
                goal_rep=goal_rep_def, hidden_dims=config['value_hidden_dims'],
                latent_dim=config['value_latent_dim'], layer_norm=config['layer_norm'],
                ensemble=True, use_rep=use_rep_high, stop_rep_grad=True)
```

then extend its `network_info` dict (after line 197):

```python
        if config.get('value_split', 'shared') == 'high_bilinear':
            network_info['high_value'] = (high_value_def, (ex_observations, ex_goals))
            network_info['target_high_value'] = (target_high_value_def, (ex_observations, ex_goals))
```

and after `params['modules_target_value'] = params['modules_value']` (line 235):

```python
        if config.get('value_split', 'shared') == 'high_bilinear':
            params['modules_target_high_value'] = params['modules_high_value']
```

- [ ] **Step 4: Run the whole suite**

Run: `.venv/bin/python -m pytest tests/test_hiql_bilinear_knobs.py tests/test_hiql_flow_injection.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/wandb_mode_shim/hiql_flow.py tests/test_hiql_bilinear_knobs.py
git commit -m "feat(hiql-flow): mirror value_split/bilinear_form for the E-D flow cell"
```

---

### Task 4: Configs (E-B/E-C/E-D) + launcher + smoke

**Files:**
- Create: `configs/gciql/cube_single_state_hiql_native_bilinear_d50_rep50_sg10.yaml`
- Create: `configs/gciql/cube_single_state_hiql_native_bilinear_d50_schaul_sg10.yaml`
- Create: `configs/gciql/cube_single_state_hiql_native_highbil_d50_sg10.yaml`
- Create: `configs/gciql/cube_single_state_hiql_flow_both_highbil_d50_sg10.yaml`
- Create: `scripts/launch_hiql_mech.sh`

**Interfaces:**
- Consumes: config keys `bilinear_form`, `value_split` (Tasks 1–2), agent `hiql_flow` + `flow_levels` (existing).
- Produces: wandb groups `HIQL-State-Native-Bilinear-d50-rep50-sg10`, `HIQL-State-Native-Bilinear-d50-schaul-sg10`, `HIQL-State-Native-HighBil-d50-sg10`, `HIQL-State-Flow-both-HighBil-d50-sg10`.

- [ ] **Step 1: Write the four configs**

`configs/gciql/cube_single_state_hiql_native_bilinear_d50_rep50_sg10.yaml` (E-B — is the global-geometry warp a rep_dim=10 bottleneck artifact?):

```yaml
# E-B (hiql-bilinear mechanism spec): option-A bilinear with rep_dim 10->50.
# If bilinear's global rep-geometry warp (probe: R2 rnd 0.14->0.10, geomSp 0.115->0.087)
# comes from squeezing value structure through 10 dims, widening the rep relieves it.
# Compare: native_bilinear_d50_sg10 (~0.21) and native_mlp_sg10 (~0.17), both @500k x3.
defaults:
  - storage: shm
  - _self_
variant: state
env_name: cube-single-play-v0
agent_file: agents/hiql.py
seed: 0
train_steps: 500000
log_interval: 5000
eval_interval: 100000
save_interval: 500000
eval_episodes: 50
eval_on_cpu: 0
run_group: HIQL-State-Native-Bilinear-d50-rep50-sg10
restore_path: null
restore_epoch: null
exp_name: null
agent_overrides:
  value_arch: bilinear
  value_latent_dim: 50
  rep_dim: 50
  subgoal_steps: 10
wandb_mode: online
```

`configs/gciql/cube_single_state_hiql_native_bilinear_d50_schaul_sg10.yaml` (E-C — H2 direct test):

```yaml
# E-C (hiql-bilinear mechanism spec): Schaul-form bilinear V=<phi(s),psi(g)>/sqrt(d).
# The value no longer trains goal_rep, so low_actor_rep_grad=True keeps the rep trained
# (via the low actor). H2: if HIQL recovers a GCIVL-sized win (+18), the BVN-form
# confound was the blocker. Compare: native_bilinear_d50_sg10 (~0.21) / mlp (~0.17).
defaults:
  - storage: shm
  - _self_
variant: state
env_name: cube-single-play-v0
agent_file: agents/hiql.py
seed: 0
train_steps: 500000
log_interval: 5000
eval_interval: 100000
save_interval: 500000
eval_episodes: 50
eval_on_cpu: 0
run_group: HIQL-State-Native-Bilinear-d50-schaul-sg10
restore_path: null
restore_epoch: null
exp_name: null
agent_overrides:
  value_arch: bilinear
  value_latent_dim: 50
  bilinear_form: goal
  low_actor_rep_grad: true
  subgoal_steps: 10
wandb_mode: online
```

`configs/gciql/cube_single_state_hiql_native_highbil_d50_sg10.yaml` (E-D native — H1' surgical test):

```yaml
# E-D (hiql-bilinear mechanism spec): two-head split. Stock MLP V keeps the TD/low path
# AND remains goal_rep's trainer; a separate Schaul bilinear head (TD-trained, d50) serves
# ONLY the high-actor advantage. Factorization where goals are distant, rep trainer intact.
# H1': if this matches/beats MLP while option-A bilinear lags, the harm is the rep trainer.
defaults:
  - storage: shm
  - _self_
variant: state
env_name: cube-single-play-v0
agent_file: agents/hiql.py
seed: 0
train_steps: 500000
log_interval: 5000
eval_interval: 100000
save_interval: 500000
eval_episodes: 50
eval_on_cpu: 0
run_group: HIQL-State-Native-HighBil-d50-sg10
restore_path: null
restore_epoch: null
exp_name: null
agent_overrides:
  value_arch: mlp
  value_split: high_bilinear
  value_latent_dim: 50
  bilinear_form: goal
  subgoal_steps: 10
wandb_mode: online
```

`configs/gciql/cube_single_state_hiql_flow_both_highbil_d50_sg10.yaml` (E-D flow — the rescue test; decision rule: ≥ flow×MLP 0.26 confirms H1'):

```yaml
# E-D flow cell (hiql-bilinear mechanism spec): value_split=high_bilinear under the
# flow-both actor. Decision rule: rescuing flow x bilinear back to >= flow x MLP (~0.26,
# vs option-A flow-bilinear ~0.14) confirms H1' (rep-trainer pathology, not flow per se).
defaults:
  - storage: shm
  - _self_
variant: state
env_name: cube-single-play-v0
agent_file: hiql_flow.py
seed: 0
train_steps: 500000
log_interval: 5000
eval_interval: 100000
save_interval: 500000
eval_episodes: 50
eval_on_cpu: 0
run_group: HIQL-State-Flow-both-HighBil-d50-sg10
restore_path: null
restore_epoch: null
exp_name: null
agent_overrides:
  flow_levels: both
  value_arch: mlp
  value_split: high_bilinear
  value_latent_dim: 50
  bilinear_form: goal
  subgoal_steps: 10
wandb_mode: online
```

**Before committing, verify the `agent_file` value for the flow config** against the in-flight one: `grep agent_file configs/gciql/cube_single_state_hiql_flow_both_mlp_sg10.yaml` and copy that exact value (the running campaign used the shim path — the sotagap process cmdline shows `--agent=<repo>/tools/wandb_mode_shim/hiql_flow.py`).

- [ ] **Step 2: Write the launcher**

`scripts/launch_hiql_mech.sh`:

```bash
#!/usr/bin/env bash
# HIQL-bilinear mechanism cells (spec 2026-07-07): E-B rep50 / E-C schaul / E-D highbil
# (native + flow-both), 3 seeds @500k = 12 jax runs. ALWAYS in tmux:
#   tmux new -s hiqlmech 'DRY_RUN=0 scripts/launch_hiql_mech.sh 2>&1 | tee /dev/shm/hiqlmech_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(pwd)/.venv/bin/python}
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-500000}
SEEDS=(${SEEDS:-0 1 2})
LOGDIR=/dev/shm/hiqlmech_logs
mkdir -p "$LOGDIR"

configs=(
  cube_single_state_hiql_native_bilinear_d50_rep50_sg10
  cube_single_state_hiql_native_bilinear_d50_schaul_sg10
  cube_single_state_hiql_native_highbil_d50_sg10
  cube_single_state_hiql_flow_both_highbil_d50_sg10
)

JOBS=()
for cfg in "${configs[@]}"; do
  for s in "${SEEDS[@]}"; do
    JOBS+=("${cfg}__s${s}|||--config-name $cfg seed=$s train_steps=$STEPS save_interval=$STEPS exp_name=${cfg}_s${s}")
  done
done

GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
echo "HIQL mechanism cells: ${#JOBS[@]} runs @${STEPS}"
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

Run: `chmod +x scripts/launch_hiql_mech.sh && scripts/launch_hiql_mech.sh`
Expected: dry-run prints 12 `GPU<n>  <cfg>__s<seed>` lines, exit 0.

- [ ] **Step 3: 2k-step live smoke of all four configs (one seed, one idle GPU)**

Check GPU load first (`nvidia-smi`), pick an idle GPU (the antmaze `amhiql` campaign holds 3/5/6/7 until it drains; e2measure/sotagap own 0/1/2/4 — pick whatever is free at run time, and set `wandb_mode: offline` via CLI to avoid polluting groups):

```bash
STEPS=2000 SEEDS="0" GPUS="5" DRY_RUN=0 bash scripts/launch_hiql_mech.sh
grep -liE "traceback|error" /dev/shm/hiqlmech_logs/*.log || echo "SMOKE CLEAN"
```

Expected: `SMOKE CLEAN` (4 runs build, train 2k steps, exit). If `run_gciql.py` supports it, append `wandb_mode=offline` to the smoke args; otherwise delete the 4 smoke runs from the wandb groups afterwards (they are `exp_name=*_s0`, step<=2k, easy to spot) — fresh-group hygiene.

- [ ] **Step 4: Commit**

```bash
git add configs/gciql/cube_single_state_hiql_native_bilinear_d50_rep50_sg10.yaml \
        configs/gciql/cube_single_state_hiql_native_bilinear_d50_schaul_sg10.yaml \
        configs/gciql/cube_single_state_hiql_native_highbil_d50_sg10.yaml \
        configs/gciql/cube_single_state_hiql_flow_both_highbil_d50_sg10.yaml \
        scripts/launch_hiql_mech.sh
git commit -m "feat(hiql): E-B/E-C/E-D mechanism configs + launcher"
```

---

### Task 5: Generalize the goal_rep probe to antmaze (E-E)

**Files:**
- Modify: `scripts/probe_hiql_goalrep.py`

**Interfaces:**
- Consumes: existing probe internals (`load_pairs`, `rep_fn`, `ridge_r2`, `spearman_dist_corr`).
- Produces: CLI args `--dataset-name` (default `cube-single-play-v0`), `--goal-slice` (default `14:17`), `--action-dim` (default `5`), `--subgoal-horizon` (default `10`). Cube invocations without new args must give byte-identical JSON.

- [ ] **Step 1: Make the dataset, goal slice, action dim, and horizon parametric**

In `scripts/probe_hiql_goalrep.py`:

Replace the module constants:

```python
DATA_DIR = Path.home() / ".ogbench" / "data"
GOAL_SLICES = {  # coordinates a subgoal rep should encode, per env family
    "cube-single-play-v0": "14:17",   # privileged cube xyz (matches eval_zcond / infer_ablation)
    "antmaze-medium-navigate-v0": "0:2",  # ant global xy (VERIFY in Step 2 before first antmaze use)
}
```

Change `load_pairs` signature and the two hardcoded uses:

```python
def load_pairs(rng: np.random.Generator, n: int, dataset_path: Path, horizon_steps: int):
```

with `data = np.load(dataset_path)` and `g = np.minimum(t + horizon_steps, rows[:, 1])` in the `subgoal` branch.

In `main()` add the args:

```python
    ap.add_argument("--dataset-name", default="cube-single-play-v0")
    ap.add_argument("--goal-slice", default=None,
                    help="obs slice 'a:b' of the goal coordinate; default from GOAL_SLICES")
    ap.add_argument("--action-dim", type=int, default=5)
    ap.add_argument("--subgoal-horizon", type=int, default=10,
                    help="steps ahead for the 'subgoal' horizon (10 cube sg10, 25 antmaze)")
```

and resolve them at the top of `main()`:

```python
    gs = args.goal_slice or GOAL_SLICES[args.dataset_name]
    lo, hi = (int(x) for x in gs.split(":"))
    goal_slice = slice(lo, hi)
    dataset_path = DATA_DIR / f"{args.dataset_name}.npz"
    pairs, obs = load_pairs(rng, args.pairs, dataset_path, args.subgoal_horizon)
```

then replace every `CUBE_QPOS` use with `goal_slice` (in the two `ridge_r2` calls and inside `spearman_dist_corr` — pass it in: `def spearman_dist_corr(rng, rep_fn, obs, goal_slice, ...)`), replace `action_dim=5` with `action_dim=args.action_dim`, and record `result["dataset"] = args.dataset_name`, `result["goal_slice"] = gs`.

- [ ] **Step 2: Verify the antmaze goal slice before trusting it**

```bash
JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python - <<'EOF'
import numpy as np
d = np.load(f"{__import__('pathlib').Path.home()}/.ogbench/data/antmaze-medium-navigate-v0.npz")
o = d["observations"]
print("obs dim:", o.shape, "dims 0:2 min/max:", o[:, :2].min(0), o[:, :2].max(0),
      "dims 2:4 min/max:", o[:, 2:4].min(0), o[:, 2:4].max(0))
EOF
```

Expected: dims `0:2` span the maze extent (tens of units, e.g. ~[-2, 30]); joint dims span ~[-1.5, 1.5]. If dims 0:2 do NOT look like maze xy, find the two dims whose range matches the maze and fix `GOAL_SLICES` before use.

- [ ] **Step 3: Regression — cube run must reproduce the existing JSON**

```bash
JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python scripts/probe_hiql_goalrep.py \
  --checkpoint-dir /dev/shm/gciql_outputs/factored-fb/HIQL-State-Native-MLP-sg10/cube_single_state_hiql_native_mlp_sg10_s0 \
  --checkpoint-step 500000 --seed 0 --value-arch mlp --pairs 20000 \
  --out /tmp/goalrep_regression.json
.venv/bin/python -c "
import json
a = json.load(open('/tmp/goalrep_regression.json'))
b = json.load(open('analysis/hiql_goalrep/cube_single_state_hiql_native_mlp_sg10_s0.json'))
assert a['horizons'] == b['horizons'], 'cube default path changed!'
print('REGRESSION OK')"
```

Expected: `REGRESSION OK` (same seed → same rng stream → identical metrics).

- [ ] **Step 4: Commit**

```bash
git add scripts/probe_hiql_goalrep.py
git commit -m "feat(analysis): goal_rep probe -> parametric dataset/goal-slice (antmaze E-E)"
```

(The actual E-E antmaze probe run happens after the `amhiql` campaign saves `params_1000000.pkl`:
`--dataset-name antmaze-medium-navigate-v0 --action-dim 8 --subgoal-horizon 25 --checkpoint-step 1000000` per checkpoint, `--value-arch` matching the cell. That is a campaign action, not part of this plan.)

---

## Self-Review

1. **Spec coverage:** E-B → Task 4 config (code-free, as specced); E-C → Tasks 1+4; E-D native + flow → Tasks 2+3+4; E-E → Task 5; E-A already running (no task, correctly). Probe script and commit hygiene match the spec's code-prereq list (the spec's `rep|goal` knob name is `bilinear_form`, values `'rep'|'goal'` — consistent).
2. **Placeholder scan:** none — every step has full code/commands/expected output.
3. **Type consistency:** `_BilinearRepValue(use_rep, stop_rep_grad)` defined Task 1, consumed Tasks 2/3 with identical kwargs; `high_value`/`target_high_value`/`high_value_loss`/`high_value/*` names identical across Tasks 2/3; config keys `bilinear_form`/`value_split` identical across Tasks 1/2/3/4; probe args defined and used consistently in Task 5.
