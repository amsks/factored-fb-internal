# GC-SimpleSF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the empty (scalar TD + explicit linear reward model) cell of the bridge grid by porting Chua et al.'s Simple SF onto the GCIQL trunk, as two cells one knob apart.

**Architecture:** One new flax value class `GCSimpleSFValue` transcribing the reference's trunk split — `h = trunk([s,a,z])`, basis `b = h/‖h‖` (zero parameters), successor features `Ψᵢ = headᵢ(h)` (twin heads), readout `Qᵢ = ⟨Ψᵢ, z⟩/√d` with `z = ψ(g)`. Selected by `value_arch='simplesf'` and used for **both** the value net and the critic. One knob, `simplesf_ground_psi`, detaches `z` inside the value/critic losses and adds a reward-prediction loss `L_rew = (⟨sg[b], ψ(g)⟩/√d − r_g)²` as ψ's sole gradient path — that cell is the paper's method.

**Tech Stack:** JAX/Flax, OGBench GCIQL trunk, pytest via jax-subprocess harness, hydra configs, tmux + wandb.

**Spec:** `docs/superpowers/specs/2026-07-10-gc-simple-sf-design.md`

## Global Constraints

- Python for tests: `.venv/bin/python -m pytest` (never `uv run`). The jax GPU venv is `/dev/shm/.venv-jax`; `tests/jax_subprocess.py` invokes it for you.
- Agent file: `tools/wandb_mode_shim/gciql_flow.py`. Tests: `tests/`. Configs: `configs/gciql/`.
- All new knobs default OFF. **Every existing config must construct and train unchanged.** `tests/test_gciql_sf_feature_td.py` (16), `tests/test_gciql_psi_base.py` (4), `tests/test_gciql_sf_joint_psi.py` (9), `tests/test_gciql_fb_forward_z.py` (2) must stay green.
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
| `tools/wandb_mode_shim/gciql_flow.py` | Modify: add `GCSimpleSFValue`; dispatch it in `create()` for `value_arch='simplesf'`; add `L_rew` + diagnostics to `total_loss`; add 4 guards; add 3 config knobs. |
| `tests/test_gciql_simplesf.py` | Create: 8 TDD tests. |
| `configs/gciql/cube_single_state_gciql_flow_simplesf_arch.yaml` | Create: rung 2 (ψ end-to-end). |
| `configs/gciql/cube_single_state_gciql_flow_simplesf_grounded.yaml` | Create: rung 3 (the paper's cell). |
| `scripts/results_map.py` | Modify: register both wandb groups. |
| `scripts/launch_simplesf.sh` | Create: 6 runs (2 cells × 3 seeds) @500k in tmux. |

---

## One deviation from the spec (decided while writing this plan)

**Spec §8 asks for a `psi_grad_norm` diagnostic. Dropped.** It cannot be computed inside a loss function without a second `jax.grad` pass over the same graph. Test 4 (`ψ receives gradient only from `L_rew``) already asserts the property the metric was meant to monitor, at higher precision (`< 1e-8`), and it asserts it *before* any GPU time is spent. The remaining diagnostics — `reward_loss`, `rhat_mean`, `rhat_std`, `b_norm_mean`, `z_norm_mean` — are all logged.

---

## Key architectural facts (read before Task 1)

- `MLP((*dims, out), activate_final=False, layer_norm=True)` applies `Dense → gelu → LayerNorm` per layer and **no activation on the final layer**, so `trunk` returns a raw pre-activation `h`. This matches the reference's final `nn.Linear(hidden, sf_dim)`.
- `ensemblize(MLP, 2)` gives twin heads, output `[2, B, d]`.
- `_sphere(x, d)` (already in `gciql_flow.py`) projects onto the radius-√d sphere.
- The value net is built with `ensemble=False` and called as `net(obs, goals)` (`actions=None`); the critic with `ensemble=True` and called as `net(obs, goals, actions)`.
- **`z` must be stop-gradded in the trunk input in BOTH rungs.** Otherwise the trunk's TD gradient flows backwards into `ψ` through the input concatenation, and `simplesf_ground_psi` would not actually isolate ψ's signal. The reference does exactly this: `task.detach()` is passed to `update_critic` (`sf_simple.py:283`). Only the **readout** `⟨Ψ, z⟩` carries ψ's gradient, and only when ungrounded.

---

### Task 1: `GCSimpleSFValue` and the `value_arch='simplesf'` dispatch

**Files:**
- Modify: `tools/wandb_mode_shim/gciql_flow.py` (new class after `GCLearnedBValue`; dispatch in `create()`; `get_config()`)
- Test: `tests/test_gciql_simplesf.py`

**Interfaces:**
- Consumes: `MLP`, `ensemblize`, `_sphere`, `GCBilinearValue` (all already in `gciql_flow.py`).
- Produces: `class GCSimpleSFValue(GCBilinearValue)` with extra fields `ground_psi: bool = False`, `project_z: bool = False`. Its `__call__(observations, goals, actions=None, info=False)` returns `v` or, with `info=True`, the 4-tuple **`(v, Psi, z, b)`** where `z` carries gradient and `b` is unit-norm. Config gains `value_arch='simplesf'` as a third option.

- [ ] **Step 1: Write the two failing tests**

Create `tests/test_gciql_simplesf.py`:

```python
"""GC-SimpleSF (spec: docs/superpowers/specs/2026-07-10-gc-simple-sf-design.md).

Chua et al.'s Simple SF ported onto the GCIQL trunk. Q = <Psi(s,a,z), z>/sqrt(d) with z = psi(g),
trained by an ordinary scalar TD; the task vector psi(g) is optionally grounded by a
reward-prediction loss and by NOTHING else. The reference's trunk split is the load-bearing part:
b = h/||h|| and Psi = head(h) are different vectors off one trunk, so the reward model <b, psi(g)>
is ASYMMETRIC and has no psi=0 saddle (unlike E3's symmetric <psi(s), psi(g)>).

Runs in the jax venv subprocess.
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def test_simplesf_readout_shapes_and_bilinear_form():
    """Q_i = <Psi_i, z>/sqrt(d) with twin heads on the critic and one head on the value net.
    info=True returns (v, Psi, z, b)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "simplesf"; cfg["value_latent_dim"] = 50
        B = 7
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
        goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        p = agent.network.params
        q, Psi, z, b = agent.network.select("critic")(ex_obs, goals, ex_act, info=True, params=p)
        assert Psi.shape == (2, B, 50), Psi.shape          # twin heads
        assert z.shape == (B, 50) and b.shape == (B, 50), (z.shape, b.shape)
        assert jnp.allclose(q, (Psi * z[None]).sum(-1) / jnp.sqrt(50.0), atol=1e-5)
        v, PsiV, zV, bV = agent.network.select("value")(ex_obs, goals, info=True, params=p)
        assert PsiV.shape == (B, 50), PsiV.shape           # value net: one head
        assert jnp.allclose(v, (PsiV * zV).sum(-1) / jnp.sqrt(50.0), atol=1e-5)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_basis_is_unit_norm_and_not_collinear_with_psi_features():
    """DECISIVE for the no-saddle property. b = h/||h|| has unit norm and is a DIFFERENT vector
    from Psi = head(h). If b were collinear with Psi, the reward loss <sg[b], psi(g)> would be a
    stop-gradded copy of Q and would demand Q(s,a,g) ~ r_g(s) -- circular and wrong."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "simplesf"; cfg["value_latent_dim"] = 50
        B = 7
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
        goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        p = agent.network.params
        _, Psi, _, b = agent.network.select("critic")(ex_obs, goals, ex_act, info=True, params=p)
        assert jnp.allclose(jnp.linalg.norm(b, axis=-1), 1.0, atol=1e-4)
        bn = b / jnp.linalg.norm(b, axis=-1, keepdims=True)
        pn = Psi[0] / (jnp.linalg.norm(Psi[0], axis=-1, keepdims=True) + 1e-8)
        cos = jnp.abs((bn * pn).sum(-1))
        assert jnp.max(cos) < 0.99, ("b must not be collinear with Psi", float(jnp.max(cos)))
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_gciql_simplesf.py -v`
Expected: both FAIL. `value_arch='simplesf'` is not a recognized option, so `create()` falls through to the `GCValue` (mlp) branch and `select("critic")(..., info=True)` raises `TypeError` (GCValue takes no `info` kwarg).

- [ ] **Step 3: Add the class**

In `gciql_flow.py`, immediately **after** `class GCLearnedBValue` and **before** `class GCIQLFlowAgent`, insert:

```python
class GCSimpleSFValue(GCBilinearValue):
    """Chua et al.'s Simple SF head, ported (reference: simple_successor_features/agent/sf_simple.py).

        h(s,a,z)  = trunk([s, a, sg[z]])          # ONE trunk, not ensembled
        b(s,a,z)  = h / ||h||                     # reward BASIS -- zero params of its own
        Psi_i     = head_i(h)                     # twin heads (CriticSF.Q1/Q2)
        z         = psi(g)                        # the task vector (sqrt(d)-sphere if project_z)
        Q_i       = <Psi_i, z_used>/sqrt(d)

    b and Psi are DIFFERENT vectors off a shared trunk, and b carries no parameters: its only
    gradient arrives through the trunk, from the TD. That is what makes the reward model
    <b, psi(g)> ASYMMETRIC (a network against a task vector) and hence free of the psi=0
    stationary point that killed E3 v1 (whose <psi(s), psi(g)> is symmetric, so its gradient
    vanishes at zero and needed sf_reward_shift to escape).

    z is stop-gradded in the TRUNK INPUT in both rungs -- the reference passes task.detach() to
    update_critic (sf_simple.py:283). Otherwise the trunk's TD gradient would reach psi through
    the input concatenation and ground_psi would not isolate psi's signal. With ground_psi the
    readout also stop-grads z, leaving the reward regression as psi's ONLY gradient path."""
    ground_psi: bool = False
    project_z: bool = False

    def setup(self):
        head_m = ensemblize(MLP, 2) if self.ensemble else MLP
        self.trunk = MLP((*self.hidden_dims, self.latent_dim), activate_final=False,
                         layer_norm=self.layer_norm)
        self.head = head_m((self.hidden_dims[-1], self.latent_dim), activate_final=False,
                           layer_norm=self.layer_norm)
        self.psi = MLP((*self.hidden_dims, self.latent_dim), activate_final=False,
                       layer_norm=self.layer_norm)

    def __call__(self, observations, goals, actions=None, info=False):
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        g = self.goal_encoder(goals) if self.goal_encoder is not None else goals
        z = self.psi(g)                                          # [B,d], grad ON
        if self.project_z:
            z = _sphere(z, self.latent_dim)
        z_in = jax.lax.stop_gradient(z)                          # trunk input NEVER trains psi
        parts = [observations] if actions is None else [observations, actions]
        h = self.trunk(jnp.concatenate([*parts, z_in], axis=-1))  # [B,d]
        b = h / (jnp.linalg.norm(h, axis=-1, keepdims=True) + 1e-6)
        Psi = self.head(h)                                       # [2,B,d] or [B,d]
        z_used = jax.lax.stop_gradient(z) if self.ground_psi else z
        v = (Psi * z_used).sum(axis=-1) / jnp.sqrt(self.latent_dim)
        if info:
            return v, Psi, z, b
        return v
```

- [ ] **Step 4: Dispatch it in `create()`**

In `create()`, find the line `if config.get('value_arch', 'mlp') == 'bilinear':` and insert this
**immediately before** it:

```python
        if config.get('value_arch', 'mlp') == 'simplesf':
            # Simple SF (Chua et al.): one trunk -> (b, Psi); Q = <Psi, psi(g)>/sqrt(d).
            # Applied to BOTH the value net and the critic: grounding only the critic's psi would
            # leak goal information back through V(s',g) into the critic target.
            d = config['value_latent_dim']
            ss_kwargs = dict(
                hidden_dims=config['value_hidden_dims'], latent_dim=d,
                layer_norm=config['layer_norm'],
                ground_psi=config.get('simplesf_ground_psi', False),
                project_z=config.get('project_z', False),
            )
            value_def = GCSimpleSFValue(
                ensemble=False, state_encoder=encoders.get('value'),
                goal_encoder=encoders.get('value'), **ss_kwargs)
            critic_def = GCSimpleSFValue(
                ensemble=True, state_encoder=encoders.get('critic'),
                goal_encoder=encoders.get('critic'), **ss_kwargs)
        elif config.get('value_arch', 'mlp') == 'bilinear':
```

Then change the existing `if config.get('value_arch', 'mlp') == 'bilinear':` line to `elif` — it
is now the second branch (the snippet above already shows the `elif`; delete the old `if` line).

- [ ] **Step 5: Add the config knobs**

In `get_config()`, replace the `config.value_arch = 'mlp'` line with:

```python
        # Value architecture: 'mlp' (canonical GCIQL) | 'bilinear' (inner-product, Direction A)
        # | 'simplesf' (Chua et al.: trunk -> b=h/||h|| and Psi=head(h); Q=<Psi,psi(g)>/sqrt(d)).
        config.value_arch = 'mlp'
        config.simplesf_ground_psi = False  # rung 3 (the paper's cell): stop-grad z in the value/critic losses so psi(g) is trained ONLY by the reward-prediction loss L_rew
        config.simplesf_reward_coef = 1.0   # weight on L_rew, summed over the value net and the critic
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_gciql_simplesf.py -v`
Expected: 2 passed.

- [ ] **Step 7: Verify backward compatibility (spec test 8)**

Run: `.venv/bin/python -m pytest tests/test_gciql_sf_feature_td.py tests/test_gciql_psi_base.py tests/test_gciql_fb_forward_z.py -q`
Expected: `22 passed`. (`value_arch` defaults to `'mlp'`, and the `bilinear` branch is untouched apart from `if` → `elif`.)

- [ ] **Step 8: Commit**

```bash
git add tools/wandb_mode_shim/gciql_flow.py tests/test_gciql_simplesf.py
git commit -m "feat(gciql): GCSimpleSFValue -- Chua et al.'s trunk split as value_arch='simplesf'

h = trunk([s,a,sg[z]]); b = h/||h|| (zero params); Psi_i = head_i(h) (twin heads);
Q_i = <Psi_i, z>/sqrt(d) with z = psi(g). Used for BOTH the value net and the critic.

b and Psi are different vectors off one trunk, so the reward model <b, psi(g)> is ASYMMETRIC and
has no psi=0 stationary point -- unlike E3's symmetric <psi(s),psi(g)>, whose gradient vanishes at
zero and needed sf_reward_shift. No shift is needed here.

z is stop-gradded in the trunk input in both rungs (the reference passes task.detach() to
update_critic, sf_simple.py:283); only the readout carries psi's gradient."
```

---

### Task 2: `L_rew`, the grounding detach, and diagnostics

**Files:**
- Modify: `tools/wandb_mode_shim/gciql_flow.py` (`total_loss`, after `critic_loss`)
- Test: `tests/test_gciql_simplesf.py`

**Interfaces:**
- Consumes: `GCSimpleSFValue` and its `info=True` 4-tuple `(v, Psi, z, b)` (Task 1).
- Produces: method `simplesf_reward_loss(self, batch, grad_params) -> (loss, info)` returning info keys `reward_loss`, `rhat_mean`, `rhat_std`, `b_norm_mean`, `z_norm_mean`; `total_loss` adds `simplesf_reward_coef * loss` and prefixes the info with `simplesf/`.

- [ ] **Step 1: Write the four failing tests**

Append to `tests/test_gciql_simplesf.py`:

```python
def test_grounded_psi_gets_gradient_only_from_the_reward_loss():
    """DECISIVE for rung 3. With simplesf_ground_psi, psi's ONLY gradient path is L_rew: the value
    and critic losses see a stop-gradded z, and the trunk input is stop-gradded in both rungs."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        def psi_grad(ground, which):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "simplesf"; cfg["value_latent_dim"] = 50
            cfg["simplesf_ground_psi"] = ground
            B = 8
            ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
            ex_nobs = jax.random.normal(jax.random.PRNGKey(4), (B, 11))
            ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
            goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
            agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
            batch = dict(observations=ex_obs, next_observations=ex_nobs, actions=ex_act,
                         rewards=-jnp.ones((B,)), masks=jnp.ones((B,)),
                         value_goals=goals, actor_goals=goals)
            def f(p):
                if which == "td":
                    return agent.value_loss(batch, p)[0] + agent.critic_loss(batch, p)[0]
                return agent.simplesf_reward_loss(batch, p)[0]
            g = jax.grad(f)(agent.network.params)
            leaves = jax.tree_util.tree_leaves(g["modules_critic"]["psi"])
            return sum(float(jnp.sum(jnp.abs(l))) for l in leaves)
        assert psi_grad(True, "td")  < 1e-8, "grounded: value+critic must NOT train psi"
        assert psi_grad(True, "rew") > 1e-3, "grounded: L_rew must train psi"
        assert psi_grad(False, "td") > 1e-3, "ungrounded: value+critic must train psi"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_trunk_receives_td_gradient_in_both_rungs():
    """b = h/||h|| has no parameters of its own; its ONLY training signal is the TD gradient
    flowing into the trunk. If the trunk froze, b would be a fixed random basis and we would have
    rebuilt the frozen-basis confound (psi_base frozen = 0.036)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        def trunk_grad(ground):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "simplesf"; cfg["value_latent_dim"] = 50
            cfg["simplesf_ground_psi"] = ground
            B = 8
            ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
            ex_nobs = jax.random.normal(jax.random.PRNGKey(4), (B, 11))
            ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
            goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
            agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
            batch = dict(observations=ex_obs, next_observations=ex_nobs, actions=ex_act,
                         rewards=-jnp.ones((B,)), masks=jnp.ones((B,)),
                         value_goals=goals, actor_goals=goals)
            g = jax.grad(lambda p: agent.critic_loss(batch, p)[0])(agent.network.params)
            leaves = jax.tree_util.tree_leaves(g["modules_critic"]["trunk"])
            return sum(float(jnp.sum(jnp.abs(l))) for l in leaves)
        assert trunk_grad(True)  > 1e-3, "grounded: trunk must still get TD gradient"
        assert trunk_grad(False) > 1e-3, "ungrounded: trunk must get TD gradient"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_reward_loss_is_exact_and_uses_the_current_index():
    """L_rew = mean((<sg[b], psi(g)>/sqrt(d) - batch['rewards'])^2), summed over the value net and
    the critic. CURRENT index: GCDataset ties rewards AND masks to idxs == value_goal_idxs
    (datasets.py:242-244; rewards = -masks exactly). The reference's (s',a') is its own convention;
    mixing the two is the class of bug fixed in E3 on 2026-07-09."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "simplesf"; cfg["value_latent_dim"] = 50
        cfg["simplesf_ground_psi"] = True
        B = 9
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
        ex_nobs = jax.random.normal(jax.random.PRNGKey(4), (B, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
        goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
        rew = jnp.where(jax.random.uniform(jax.random.PRNGKey(5), (B,)) < 0.2, 0.0, -1.0)
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_nobs, actions=ex_act,
                     rewards=rew, masks=1.0 + rew, value_goals=goals, actor_goals=goals)
        loss, info = agent.simplesf_reward_loss(batch, agent.network.params)
        p = agent.network.params
        _, _, zq, bq = agent.network.select("critic")(ex_obs, goals, ex_act, info=True, params=p)
        _, _, zv, bv = agent.network.select("value")(ex_obs, goals, info=True, params=p)
        rq = (bq * zq).sum(-1) / jnp.sqrt(50.0)
        rv = (bv * zv).sum(-1) / jnp.sqrt(50.0)
        exp = jnp.mean((rq - rew) ** 2) + jnp.mean((rv - rew) ** 2)
        assert jnp.allclose(loss, exp, rtol=1e-5), (loss, exp)
        assert jnp.allclose(info["b_norm_mean"], 1.0, atol=1e-4), info["b_norm_mean"]
        # b is evaluated at the CURRENT observation, not next_observations
        _, _, _, b_next = agent.network.select("critic")(ex_nobs, goals, ex_act, info=True, params=p)
        wrong = jnp.mean(((b_next * zq).sum(-1) / jnp.sqrt(50.0) - rew) ** 2) + jnp.mean((rv - rew) ** 2)
        assert not jnp.allclose(loss, wrong), "test vacuous: b(s) == b(s')"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_reward_model_has_no_psi_zero_saddle():
    """The asymmetric model <sg[b], psi(g)> has a NONZERO gradient at psi = 0, because b is a
    separate object. Contrast E3's symmetric <psi(s), psi(g)>, whose gradient is proportional to
    psi and therefore vanishes there -- the saddle that collapsed E3 v1 and forced
    sf_reward_shift. project_z is OFF here: the sphere projection is ill-conditioned at z = 0."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "simplesf"; cfg["value_latent_dim"] = 50
        cfg["simplesf_ground_psi"] = True; cfg["project_z"] = False
        B = 8
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
        goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=-jnp.ones((B,)), masks=jnp.ones((B,)),
                     value_goals=goals, actor_goals=goals)
        # Zero the critic's psi OUTPUT Dense layer -> z == 0 exactly. Zero ONLY the final Dense:
        # zeroing the whole subtree would also zero that layer's INPUT, making dz/dW vanish and
        # the test vacuous. And note sorted(keys)[-1] is 'LayerNorm_2', not 'Dense_3' --
        # alphabetical order puts LayerNorm after Dense. Select the highest-numbered Dense.
        p = jax.tree_util.tree_map(lambda x: x, agent.network.params)   # rebuilds dicts: safe to mutate
        psi_tree = p["modules_critic"]["psi"]
        denses = [k for k in psi_tree if k.startswith("Dense_")]
        last = max(denses, key=lambda k: int(k.split("_")[1]))
        assert last == "Dense_3", (last, sorted(psi_tree.keys()))
        psi_tree[last] = jax.tree_util.tree_map(jnp.zeros_like, psi_tree[last])
        _, _, z0, _ = agent.network.select("critic")(ex_obs, goals, ex_act, info=True, params=p)
        assert jnp.allclose(z0, 0.0), "setup failed: z is not zero"
        g = jax.grad(lambda q: agent.simplesf_reward_loss(batch, q)[0])(p)
        leaves = jax.tree_util.tree_leaves(g["modules_critic"]["psi"])
        gn = sum(float(jnp.sum(jnp.abs(l))) for l in leaves)
        assert gn > 1e-3, ("psi=0 must NOT be a stationary point of the asymmetric model", gn)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_gciql_simplesf.py -k "grounded or trunk or reward" -v`
Expected: all four FAIL with `AttributeError: 'GCIQLFlowAgent' object has no attribute 'simplesf_reward_loss'`.

- [ ] **Step 3: Add `simplesf_reward_loss`**

In `gciql_flow.py`, immediately **before** `def ortho_loss(self, batch, grad_params):`, insert:

```python
    def simplesf_reward_loss(self, batch, grad_params):
        """Simple SF's reward-prediction loss (sf_simple.py:302-325), the task vector's ONLY
        gradient path when simplesf_ground_psi:

            r_hat(s,a,g) = <sg[b(s,a,z)], psi(g)>/sqrt(d)   ->  regress on batch['rewards']

        b is stop-gradded exactly as the reference computes next_rep under torch.no_grad() and
        steps only task_opt. CURRENT index: GCDataset ties rewards AND masks to
        idxs == value_goal_idxs (datasets.py:242-244), so b is evaluated at (s,a), not (s',a').

        Applied to BOTH nets: grounding only the critic's psi would leak goal information back
        through V(s',g) into the critic target. No reward shift is needed -- the model is
        ASYMMETRIC (b is a separate object from psi), so it has no psi=0 saddle, and with
        ||b||=1, ||z||=sqrt(d) its range [-1,1] already covers the reward range {-1,0}."""
        obs, vgoals, acts = batch['observations'], batch['value_goals'], batch['actions']
        r = batch['rewards']
        d = self.config['value_latent_dim']
        _, _, z_q, b_q = self.network.select('critic')(obs, vgoals, acts, info=True, params=grad_params)
        _, _, z_v, b_v = self.network.select('value')(obs, vgoals, info=True, params=grad_params)
        rhat_q = (jax.lax.stop_gradient(b_q) * z_q).sum(axis=-1) / jnp.sqrt(d)      # [B]
        rhat_v = (jax.lax.stop_gradient(b_v) * z_v).sum(axis=-1) / jnp.sqrt(d)      # [B]
        loss = jnp.mean((rhat_q - r) ** 2) + jnp.mean((rhat_v - r) ** 2)
        return loss, {
            'reward_loss': loss,
            'rhat_mean': jnp.mean(rhat_q), 'rhat_std': jnp.std(rhat_q),
            'b_norm_mean': jnp.mean(jnp.linalg.norm(b_q, axis=-1)),
            'z_norm_mean': jnp.mean(jnp.linalg.norm(z_q, axis=-1)),
        }
```

- [ ] **Step 4: Wire it into `total_loss`**

In `total_loss`, find

```python
        loss = value_loss + critic_loss + actor_loss
```

and insert immediately **after** it:

```python
        if self.config.get('simplesf_ground_psi', False):
            # rung 3 (the paper's cell): psi(g) is detached inside value/critic (GCSimpleSFValue),
            # so this regression is its ONLY gradient. The trunk still learns from the TD, which
            # is what trains the basis b.
            ss_loss, ss_info = self.simplesf_reward_loss(batch, grad_params)
            for k, v in ss_info.items():
                info[f'simplesf/{k}'] = v
            loss = loss + self.config.get('simplesf_reward_coef', 1.0) * ss_loss
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_gciql_simplesf.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add tools/wandb_mode_shim/gciql_flow.py tests/test_gciql_simplesf.py
git commit -m "feat(gciql): simplesf_reward_loss -- the task vector's only gradient under grounding

L_rew = mean((<sg[b(s,a,z)], psi(g)>/sqrt(d) - r_g(s))^2), summed over the value net and the
critic (grounding only the critic's psi would leak goal info back through V(s',g) into the critic
target). b is stop-gradded, mirroring the reference computing next_rep under torch.no_grad() and
stepping only task_opt.

CURRENT-index convention: GCDataset ties rewards AND masks to idxs == value_goal_idxs, so b is
evaluated at (s,a). No reward shift needed -- the model is asymmetric and has no psi=0 saddle
(test asserts a nonzero gradient at psi=0), and with ||b||=1, ||z||=sqrt(d) its range [-1,1]
already covers {-1,0}.

Tests also pin: the trunk keeps its TD gradient in BOTH rungs (b's only signal), and grounded psi
receives < 1e-8 gradient from value+critic."
```

---

### Task 3: The four guards

**Files:**
- Modify: `tools/wandb_mode_shim/gciql_flow.py` (`create()`, before the architecture dispatch)
- Test: `tests/test_gciql_simplesf.py`

**Interfaces:**
- Consumes: knobs from Tasks 1–2.
- Produces: `create()` raises `AssertionError` on four invalid combinations; both valid cells construct.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gciql_simplesf.py`:

```python
def test_simplesf_guards_reject_invalid_combinations():
    """Guards encode the design so a misconfigured cell fails loudly instead of producing an
    uninterpretable number.

    1. simplesf_ground_psi => value_arch == 'simplesf'
    2. value_arch=='simplesf' excludes every knob that selects its own critic class or loss
       (sf_feature_td, measure_only, measure_coef>0, fb_forward_z, fwd_two_tower, isfb,
       sf_joint_psi) -- they would silently override the architecture.
    3. value_arch=='simplesf' => psi_base == 'learned' (psi_base dispatches GCBilinearValue).
    4. value_arch=='simplesf' => actor_cond == 'goal' (the flow actor sees the raw goal)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (6, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (6, 5))
        def build(**over):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_latent_dim"] = 50
            cfg.update(over)
            return GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        S = dict(value_arch="simplesf")
        bad = [
            dict(simplesf_ground_psi=True, value_arch="bilinear"),   # 1
            dict(S, sf_feature_td=True, sf_base="frozen"),           # 2
            dict(S, measure_only=True),                              # 2
            dict(S, measure_coef=1.0),                               # 2
            dict(S, fb_forward_z=True),                              # 2
            dict(S, fwd_two_tower=True),                             # 2
            dict(S, sf_joint_psi=True),                              # 2
            dict(S, psi_base="frozen"),                              # 3
            dict(S, actor_cond="z"),                                 # 4
        ]
        for over in bad:
            try:
                build(**over)
            except AssertionError:
                continue
            raise SystemExit("guard missing for " + repr(over))
        build(**S)                                       # rung 2: simplesf-arch
        build(**S, simplesf_ground_psi=True)             # rung 3: simplesf-grounded
        build(value_arch="bilinear")                     # legacy cells unaffected
        build()                                          # mlp default unaffected
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gciql_simplesf.py::test_simplesf_guards_reject_invalid_combinations -v`
Expected: FAIL with `SystemExit: guard missing for ...`.

- [ ] **Step 3: Add the guards**

In `create()`, immediately **after** the line `config['action_dim'] = action_dim` and **before**
the `if config.get('sf_feature_td', False):` block, insert:

```python
        if config.get('simplesf_ground_psi', False) or config.get('value_arch', 'mlp') == 'simplesf':
            # (1) the grounding knob is meaningless without the architecture.
            assert config.get('value_arch', 'mlp') == 'simplesf', \
                "simplesf_ground_psi requires value_arch='simplesf'"
            # (2) every knob below selects its own critic class or replaces the loss entirely and
            # would silently override the SimpleSF architecture.
            for knob in ('sf_feature_td', 'measure_only', 'fb_forward_z', 'fwd_two_tower',
                         'isfb', 'sf_joint_psi'):
                assert not config.get(knob, False), f"value_arch='simplesf' is incompatible with {knob}"
            assert config.get('measure_coef', 0.0) == 0.0, \
                "value_arch='simplesf' is incompatible with measure_coef>0"
            # (3) psi_base dispatches GCBilinearValue subclasses; the two must not compose silently.
            assert config.get('psi_base', 'learned') == 'learned', \
                "value_arch='simplesf' requires psi_base='learned'"
            # (4) the flow actor conditions on the raw goal in every cell.
            assert config.get('actor_cond', 'goal') == 'goal' and not config.get('actor_z_infer', False), \
                "value_arch='simplesf' requires actor_cond='goal'"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gciql_simplesf.py -v`
Expected: 7 passed.

- [ ] **Step 5: Run every affected suite**

Run: `.venv/bin/python -m pytest tests/test_gciql_simplesf.py tests/test_gciql_sf_feature_td.py tests/test_gciql_psi_base.py tests/test_gciql_sf_joint_psi.py tests/test_gciql_fb_forward_z.py -q`
Expected: `38 passed`, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add tools/wandb_mode_shim/gciql_flow.py tests/test_gciql_simplesf.py
git commit -m "feat(gciql): four guards for value_arch='simplesf'

 1. simplesf_ground_psi => value_arch='simplesf'
 2. simplesf excludes sf_feature_td / measure_only / measure_coef>0 / fb_forward_z /
    fwd_two_tower / isfb / sf_joint_psi -- each selects its own critic class or loss and would
    silently override the architecture
 3. simplesf => psi_base='learned' (psi_base dispatches GCBilinearValue subclasses)
 4. simplesf => actor_cond='goal'

38 passed across simplesf + sf_feature_td + psi_base + sf_joint_psi + fb_forward_z."
```

---

### Task 4: Configs, results_map, launcher

**Files:**
- Create: `configs/gciql/cube_single_state_gciql_flow_simplesf_arch.yaml`, `configs/gciql/cube_single_state_gciql_flow_simplesf_grounded.yaml`, `scripts/launch_simplesf.sh`
- Modify: `scripts/results_map.py`

**Interfaces:**
- Consumes: knobs and guards from Tasks 1–3.
- Produces: wandb groups `GCIQL-State-Flow-SimpleSF-Arch` (rung 2) and `GCIQL-State-Flow-SimpleSF-Grounded` (rung 3), both registered in `results_map.MAP` with `csv='bridge'`.

- [ ] **Step 1: Write rung 2's config**

Create `configs/gciql/cube_single_state_gciql_flow_simplesf_arch.yaml`:

```yaml
# Rung 2 (sf-design-space 1d): the Simple SF ARCHITECTURE with psi(g) still learned end-to-end
# through the value/critic losses. Isolates the cost of the architecture alone against bilinear
# GCIQL (0.904 +/- 0.019). Q = <Psi(s,a,z), z>/sqrt(d), z = psi(g); scalar IQL expectile TD with
# the TRUE reward. No reward-prediction loss.
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
run_group: GCIQL-State-Flow-SimpleSF-Arch
restore_path: null
restore_epoch: null
exp_name: null
agent_overrides:
  bc_coeff: 3.0
  value_arch: simplesf
  value_latent_dim: 50
  project_z: true
wandb_mode: online
```

- [ ] **Step 2: Write rung 3's config (the paper's cell)**

Create `configs/gciql/cube_single_state_gciql_flow_simplesf_grounded.yaml`:

```yaml
# Rung 3 (sf-design-space 1d): Chua et al.'s Simple SF -- the paper's cell, with our backup.
# Identical to ..._simplesf_arch.yaml EXCEPT psi(g) is stop-gradded inside the value/critic losses
# and trained ONLY by the reward-prediction loss  r_hat = <sg[b], psi(g)>/sqrt(d)  on
# batch['rewards']. Isolates the cost of REWARD-GROUNDING psi against rung 2.
#
# Deviation from the reference, deliberate (spec 4a): we keep GCIQL's expectile backup rather than
# the paper's DDPG min-twin target, so that rung 2 differs from bilinear GCIQL by architecture
# ALONE. Safe because tau0.5 -- which strips the improvement operator entirely -- scores
# 0.920 +/- 0.016, so the operator is free on this task.
#
# Read simplesf/reward_loss BEFORE the success number: with r in {-1,0} and ~79% of relabelled
# rewards equal to -1, predicting the mean scores ~0.166. A plateau near that prior means the
# reward is NOT linearly readable in b -- the reference's load-bearing assumption failing -- and
# the cell then says nothing about grounding.
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
run_group: GCIQL-State-Flow-SimpleSF-Grounded
restore_path: null
restore_epoch: null
exp_name: null
agent_overrides:
  bc_coeff: 3.0
  value_arch: simplesf
  value_latent_dim: 50
  project_z: true
  simplesf_ground_psi: true
  simplesf_reward_coef: 1.0
wandb_mode: online
```

- [ ] **Step 3: Verify both configs construct an agent**

Run:

```bash
cd /home/adityamohan/git/Factored-FB/third_party/ogbench/impls && /dev/shm/.venv-jax/bin/python -c "
import sys, yaml; sys.path.insert(0,'/home/adityamohan/git/Factored-FB/tools/wandb_mode_shim')
import jax
from gciql_flow import GCIQLFlowAgent, get_config
for n in ['simplesf_arch','simplesf_grounded']:
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
ok simplesf_arch
ok simplesf_grounded
```

- [ ] **Step 4: Register both groups in `results_map.py`**

In `scripts/results_map.py`, immediately **above** the line beginning
`# --- 1a: joint-psi SF, FB ortho + sphere`, insert:

```python
    # --- 1d: GC-SimpleSF (Chua et al.). Rung 2 isolates the architecture against bilinear GCIQL
    # (0.904); rung 3 grounds psi in the reward and isolates the grounding against rung 2.
    'GCIQL-State-Flow-SimpleSF-Arch':     dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='simplesf-arch', graft='SimpleSF trunk split (b=h/||h||, Psi=head(h)), psi learned end-to-end; scalar IQL TD'),
    'GCIQL-State-Flow-SimpleSF-Grounded': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='simplesf-grounded', graft="Chua et al.'s cell: + psi trained ONLY by reward regression <sg[b],psi(g)>"),
```

- [ ] **Step 5: Verify the map still accounts for every group**

Run: `.venv/bin/python scripts/compile_results_csvs.py 2>&1 | tail -3`
Expected: `[compile] audit -> .../compile_audit.txt  (all groups accounted for)` and exit 0. (The two new groups appear under `MAPPED BUT ABSENT FROM WANDB` until the runs start — a note, not a failure.)

- [ ] **Step 6: Write the launcher**

Create `scripts/launch_simplesf.sh` and `chmod +x` it:

```bash
#!/usr/bin/env bash
# GC-SimpleSF (spec docs/superpowers/specs/2026-07-10-gc-simple-sf-design.md).
#
#   rung 2  simplesf_arch       SimpleSF architecture, psi learned end-to-end
#                               -> isolates the ARCHITECTURE vs bilinear GCIQL 0.904 +/- 0.019
#   rung 3  simplesf_grounded   + psi trained ONLY by reward regression (Chua et al.'s cell)
#                               -> isolates the GROUNDING vs rung 2
#
# ALWAYS in tmux:
#   tmux new -d -s simplesf 'DRY_RUN=0 scripts/launch_simplesf.sh 2>&1 | tee /dev/shm/simplesf_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(pwd)/.venv/bin/python}   # run_gciql.py is a hydra launcher: it runs HERE and
                                    # subprocesses ogbench main.py into .venv-jax itself.
                                    # Do NOT point this at .venv-jax -- that venv has no hydra.
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-500000}
LOGDIR=/dev/shm/simplesf_logs
mkdir -p "$LOGDIR"

configs=(
  cube_single_state_gciql_flow_simplesf_arch
  cube_single_state_gciql_flow_simplesf_grounded
)

JOBS=()
for cfg in "${configs[@]}"; do
  for s in 0 1 2; do
    JOBS+=("${cfg}__s${s}|||--config-name $cfg seed=$s train_steps=$STEPS save_interval=$STEPS exp_name=${cfg}_s${s}")
  done
done

GPUS=(${GPUS:-0 1 2 3 5 6})
echo "GC-SimpleSF: ${#JOBS[@]} runs @${STEPS} over ${#GPUS[@]} GPUs"
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

Run: `chmod +x scripts/launch_simplesf.sh && scripts/launch_simplesf.sh`
Expected: `GC-SimpleSF: 6 runs @500000 over 6 GPUs` followed by 6 `GPU<n>  <cfg>__s<seed>` lines, one per GPU. No processes started.

- [ ] **Step 8: Commit**

```bash
git add configs/gciql/cube_single_state_gciql_flow_simplesf_arch.yaml \
        configs/gciql/cube_single_state_gciql_flow_simplesf_grounded.yaml \
        scripts/results_map.py scripts/launch_simplesf.sh
git commit -m "feat(configs): GC-SimpleSF rungs 2 and 3 + launcher

rung 2 (simplesf-arch): the SimpleSF trunk-split architecture, psi learned end-to-end. Isolates
the architecture against bilinear GCIQL (0.904 +/- 0.019).

rung 3 (simplesf-grounded): Chua et al.'s cell. Same architecture; psi(g) detached inside the
value/critic losses and trained ONLY by the reward-prediction loss. Isolates the grounding
against rung 2.

Deviation (deliberate): we keep GCIQL's expectile backup, not the paper's DDPG min-twin, so rung 2
differs from bilinear GCIQL by architecture ALONE. tau0.5 = 0.920 +/- 0.016 shows the improvement
operator is free here.

Distinct wandb groups, both registered in results_map."
```

---

### Task 5: Launch and verify the campaign

**Files:** none modified. Operational only.

**Interfaces:**
- Consumes: `scripts/launch_simplesf.sh` (Task 4).
- Produces: 6 live runs; a health check that distinguishes a real failure from the known slow first eval, and an early read on whether the reward is linearly readable in `b`.

- [ ] **Step 1: Check GPU capacity and pick free devices**

Run:

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

Use only GPUs showing `1 MiB`. Do **not** kill any surviving `psibase` straggler. Pass the free
devices via `GPUS`, e.g. `GPUS="0 1 2 3 5 6"`.

- [ ] **Step 2: Launch in tmux**

Run:

```bash
cd /home/adityamohan/git/Factored-FB
tmux ls 2>/dev/null | grep -q "^simplesf:" && echo "SESSION EXISTS - abort" || \
  tmux new -d -s simplesf 'GPUS="0 1 2 3 5 6" DRY_RUN=0 scripts/launch_simplesf.sh 2>&1 | tee /dev/shm/simplesf_dispatch.log'
```

Note: `tmux has-session -t simplesf` prefix-matches other sessions; the `grep "^simplesf:"` form
above is the safe check.

- [ ] **Step 3: Confirm all 6 dispatched and none crashed on import or a guard**

Run:

```bash
until [ "$(grep -c '\[launch\]' /dev/shm/simplesf_dispatch.log 2>/dev/null)" = "6" ]; do sleep 10; done
sleep 90
for f in /dev/shm/simplesf_logs/*.log; do
  if grep -qi "Traceback\|AssertionError" "$f"; then echo "CRASH $(basename $f .log): $(grep -m1 -iE 'error|assert' "$f" | cut -c1-100)";
  else echo "ok    $(basename $f .log)"; fi
done
```

Expected: 6 `ok` lines. `ModuleNotFoundError: hydra` means `PY` is wrong. An `AssertionError`
means a guard rejected the config — fix the config, do **not** weaken the guard.

- [ ] **Step 4: Confirm the runs reach the GPU**

Run: `until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -ge 6 ]; do sleep 15; done; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader`
Expected: 6 more compute processes than in Step 1.

**Do not mistake a slow first eval for a hang.** The step-1 eval runs a random-init flow policy
that flails, making mujoco's contact solver crawl (~20 s/episode × 5 tasks × 51 episodes ≈ 80 min)
while the GPU shows 0% util. `.venv/bin/py-spy dump --pid <child>` will show
`utils/evaluation.py:evaluate`. That is normal.

- [ ] **Step 5: THE PRE-REGISTERED CHECK — is the reward linearly readable in `b`?**

Once any grounded run logs `_step >= 20000`:

```bash
cd /home/adityamohan/git/Factored-FB
.venv/bin/python -c "
import wandb; api=wandb.Api(timeout=120)
K=['training/simplesf/reward_loss','training/simplesf/rhat_mean','training/simplesf/rhat_std',
   'training/simplesf/b_norm_mean','training/simplesf/z_norm_mean']
for r in api.runs('amsks/factored-fb',{'group':'GCIQL-State-Flow-SimpleSF-Grounded'},per_page=10):
    pts=[p for p in r.history(keys=K,samples=500,pandas=False) if p.get(K[0]) is not None]
    if pts:
        p=pts[-1]
        print('seed',r.config.get('seed'),'step',p['_step'],{k.split('/')[-1]: round(p[k],4) for k in K})
"
```

Expected sanity: `b_norm_mean ≈ 1.0` and `z_norm_mean ≈ 7.071` (= √50, pinned by `project_z`).

**The decisive number is `reward_loss`.** With `r ∈ {−1,0}` and ~79% of relabelled rewards equal
to −1, a model that predicts the mean scores `0.79 × 0.21 ≈ 0.166`.
- `reward_loss` falling well below 0.166 ⇒ the reward **is** linearly readable in `b`, the
  reference's load-bearing assumption holds, and the success number is a clean test of grounding.
- `reward_loss` plateauing near 0.166 ⇒ the assumption **fails**, and rung 3's success number says
  nothing about grounding. Report this before reporting success. It is the same failure mode that
  made the identity/frozen SF cells uninterpretable.

- [ ] **Step 6: Record the campaign in memory and stop**

Write a memory note for the bridge track with: tmux session `simplesf`, log dir
`/dev/shm/simplesf_logs/`, the two groups, the anchors (bilinear GCIQL 0.904 ± 0.019, `B1` 0.844 ±
0.008, frozen-ψ 0.036, E3 0.003), the pre-registered `reward_loss` check above, and the prediction
(rung 2 ≈ 0.84–0.90; rung 3 far below, plausibly 0.0–0.2; a rung 3 near 0.90 falsifies the
"grounding is a repair, not a virtue" claim and rewrites the objective axis).

Collection is a separate session: both groups are already in `results_map` (Task 4), so run
`.venv/bin/python scripts/compile_results_csvs.py --refresh` and fold the numbers into
`PAPER/fb-gciql-bridge/sf-design-space.md` §1d.

---

## Self-Review

**Spec coverage.** §2 architecture → Task 1 (`GCSimpleSFValue`, dispatch). §3 the two rungs →
Task 4 configs. §4a expectile backup → encoded in rung 3's config header and the commit message
(no code needed: `total_loss` already uses `value_loss`/`critic_loss`). §4b current-index `b` →
Task 2 `simplesf_reward_loss` + its test. §4c two task vectors → Task 2 (loss sums over both nets)
+ the dispatch building `GCSimpleSFValue` for value and critic. §5 knobs → Task 1 Step 5; guards →
Task 3. §6 tests 1–8 → Task 1 (tests 1, 2, 8-as-Step-7), Task 2 (tests 3, 4, 5, 6), Task 3
(test 7). §7 configs → Task 4. §8 diagnostics → Task 2 info keys + Task 5 Step 5 (minus
`psi_grad_norm`, dropped with reason above). §9 anchors → config headers and Task 5 Step 6.
§10 sequencing → Task 5.

**Deviation, flagged above and justified in-plan:** `psi_grad_norm` dropped; Test 4 asserts the
same property at `< 1e-8` before any GPU time is spent.

**Type/name consistency.** Knobs spelled `value_arch='simplesf'`, `simplesf_ground_psi`,
`simplesf_reward_coef` identically in `get_config()`, the dispatch, guards, tests, both YAMLs, and
`results_map`. The class is `GCSimpleSFValue` with fields `ground_psi` / `project_z` everywhere.
`info=True` returns the 4-tuple `(v, Psi, z, b)` — produced in Task 1, consumed by
`simplesf_reward_loss` in Task 2 and by the tests in both. Info keys `reward_loss`, `rhat_mean`,
`rhat_std`, `b_norm_mean`, `z_norm_mean` are produced in Task 2 and consumed in Task 5 Step 5 as
`training/simplesf/<key>` (the agent prefixes `simplesf/`, wandb prefixes `training/`).
`_sphere(x, d)`, `MLP`, `ensemblize`, `GCBilinearValue` match the existing source.

**No placeholders.** Every code step carries the code; every run step carries the command and its
expected output.
