# Annealed SM Finetuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Warm-start from converged bilinear GCIQL (0.904) and grow the successor-measure loss weight `λ(t)` slowly, to find the dose at which the SM objective destroys an already-good value geometry — separating *fixed-point incompatibility* from an *optimization-path* artifact.

**Architecture:** Two schedules keyed on `network.step`, which the prep script resets to 0: a linear λ ramp (300k) that then **holds** at `lam_max` for 100k, and a short linear LR warmup (10k) via `optax.adam(learning_rate=<callable>)`. A prep script writes a finetune-ready checkpoint — anchor params, a **fresh** `opt_state` matching the scheduled `tx`, and `step = 0` — which is the only way the two schedules can start at zero and the only way an LR schedule can be restored at all.

**Tech Stack:** JAX/Flax/optax, OGBench GCIQL trunk, pytest via jax-subprocess harness, hydra configs, tmux + wandb.

**Spec:** `docs/superpowers/specs/2026-07-10-annealed-sm-finetune-design.md`

## Global Constraints

- **DO NOT LAUNCH ANY RUNS.** GPUs are busy with `tasknorm` + `simplesf`. This plan ends at a dry-run; a code audit and a GPU-free window come first.
- Python for tests: `.venv/bin/python -m pytest` (never `uv run`). The jax GPU venv is `/dev/shm/.venv-jax`; `tests/jax_subprocess.py` invokes it. Scripts that import jax must run under `/dev/shm/.venv-jax/bin/python` **from** `third_party/ogbench/impls` (so `agents.gciql` and `utils.*` resolve).
- Agent file: `tools/wandb_mode_shim/gciql_flow.py`. Tests: `tests/`. Configs: `configs/gciql/`. Scripts: `scripts/`.
- All new knobs default OFF. **`anneal_measure=False` and `anneal_lr_warmup=0` must reproduce today's `total_loss` and today's `tx` exactly.** Existing suites must stay green: `test_gciql_sf_feature_td.py` (16), `test_gciql_psi_base.py` (4), `test_gciql_sf_joint_psi.py` (9), `test_gciql_fb_forward_z.py` (2), `test_gciql_measure_scale.py` (5), `test_gciql_simplesf.py` (7), `test_gciql_task_norm.py` (6).
- Commit messages: **omit** any `Co-Authored-By: Claude` trailer (repo convention).
- Commit only code/tests/configs/scripts. `PAPER/`, `results/`, `analysis/`, `docs/superpowers/` are gitignored.
- wandb entity is `amsks` (defaulted by `run_gciql.py:128`). Every training job runs in **tmux**.
- Launch scripts set `PY` to the **repo** `.venv/bin/python` — `run_gciql.py` is a hydra launcher that subprocesses ogbench `main.py` into `.venv-jax` itself. Pointing `PY` at `.venv-jax` dies on missing hydra, silently, per-run.
- **Never reuse a wandb `run_group`.** `results_map.py` keys on the group.
- **Never edit `third_party/`.** The prep script exists precisely so `main.py` and `restore_agent` stay untouched.
- Warm-start source: `~/bridge_ckpt_backup/gciql_outputs/factored-fb/GCIQL-State-Flow-Bilinear-d50/sd00{0,1,2}_*/params_1000000.pkl` (verified present, 85 MB each).

---

## File Structure

| File | Responsibility |
|---|---|
| `tools/wandb_mode_shim/gciql_flow.py` | Modify: `total_loss` (λ ramp on the measure term + `anneal/*` logging); `create()` (LR warmup `tx`, 5 guards); `get_config()` (3 knobs). |
| `scripts/prepare_anneal_ckpt.py` | Create: writes a finetune-ready checkpoint (anchor params, fresh `opt_state`, `step=0`). |
| `tests/test_gciql_anneal.py` | Create: 8 TDD tests. |
| `configs/gciql/cube_single_state_gciql_flow_anneal_zero.yaml` | Create: λ≡0 control. |
| `configs/gciql/cube_single_state_gciql_flow_anneal_sgphi.yaml` | Create: stopgrad-φ, λ 0→10. |
| `configs/gciql/cube_single_state_gciql_flow_anneal_shared.yaml` | Create: shared legs, λ 0→100. |
| `scripts/results_map.py` | Modify: register the three wandb groups. |
| `scripts/launch_anneal.sh` | Create: prep + 9 runs @400k in tmux. **Dry-run only in this plan.** |

---

## Key facts (read before Task 1)

- `total_loss` is `@jax.jit`-decorated and closes over `self`, so **`self.network.step` is readable inside it** (it is a pytree leaf of `TrainState`, traced under jit). `GCIQLAgent.update` builds `loss_fn(grad_params) = self.total_loss(batch, grad_params, rng=rng)` and calls `network.apply_loss_fn`, which increments `step` *after* the gradient. So during update `i`, the loss sees `step == i`.
- `TrainState.create` sets `step=1`. The prep script overwrites it to `0`, so the first update sees `step=0` and `λ(0)=0` exactly.
- Today's optimizer (`gciql_flow.py:1244`) is `network_tx = optax.adam(learning_rate=config['lr'])`.
- **`optax.adam(learning_rate=3e-4)` ends in `EmptyState`; `optax.adam(learning_rate=<callable>)` ends in `ScaleByScheduleState(count)`.** Verified: the two `opt_state` trees differ. `restore_agent` does `flax.serialization.from_state_dict(agent, ...)` over the whole `TrainState`, so a constant-LR checkpoint **cannot** restore into a scheduled optimizer. This is why the prep script must build the fresh `opt_state` from the *finetune* config's `tx`.
- `restore_agent` restores `network.step` too (the anchor's is `1_000_001`). Resetting it in the prep script removes the offset trap entirely — there is **no `anneal_t0` knob**.
- `config` is a `nonpytree_field()` (`agents/gciql.py:22`), so it is **not** in the state dict — the new knobs survive restore, and the param tree matches (the measure loss adds no parameters).
- `total_loss`'s measure block is gated on `measure_coef > 0` (`gciql_flow.py:925`). The λ ramp multiplies **inside** that block, so `lam_max = 0` (the control) skips the measure loss entirely, and the anneal cells (`lam_max > 0`) always compute it — static shapes under jit — scaled by `λ(t)`.
- `optax.linear_schedule(init_value, end_value, transition_steps)` holds at `end_value` after `transition_steps`. That is exactly ramp-then-hold; no custom schedule needed.

---

### Task 1: The λ ramp and `anneal/*` logging

**Files:**
- Modify: `tools/wandb_mode_shim/gciql_flow.py` (`total_loss` measure block at `:925-930`; `get_config()`)
- Test: `tests/test_gciql_anneal.py`

**Interfaces:**
- Consumes: existing `measure_loss`, `measure_coef`.
- Produces: config knobs `anneal_measure: bool = False`, `anneal_ramp: int = 300_000`. When `anneal_measure`, `total_loss` scales the measure term by `lam(t) = measure_coef * clip(step/anneal_ramp, 0, 1)` and adds info keys `anneal/lambda`, `anneal/t`.

- [ ] **Step 1: Write the three failing tests**

Create `tests/test_gciql_anneal.py`:

```python
"""Annealed SM finetuning (spec: docs/superpowers/specs/2026-07-10-annealed-sm-finetune-design.md).

Warm-start from converged bilinear GCIQL and grow the successor-measure weight lambda(t) slowly:

    lam(t) = measure_coef * clip(t / anneal_ramp, 0, 1)      # ramp, then HOLD
    L      = L_IQL + lam(t) * L_SM                            # L_IQL untouched

t is network.step, which the prep script resets to 0. Two facts make that necessary:
  * restore_agent restores network.step (the anchor's is 1_000_001), so a schedule keyed on it
    would start fully ramped and the run would silently be a constant-lambda cold start;
  * optax.adam(learning_rate=<callable>) changes the opt_state tree, so a constant-LR checkpoint
    cannot restore into a scheduled optimizer at all.

Runs in the jax venv subprocess.
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS

# Builds a measure-graft agent on the bilinear-d50 architecture (the E2 mix shape).
PREAMBLE = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config

        def build(step=0, lam_max=10.0, ramp=100, anneal=True, warm=0, sgphi=True):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
            cfg["measure_only"] = False; cfg["measure_coef"] = lam_max
            cfg["measure_legs"] = "critic"; cfg["measure_ortho_coef"] = 0.0
            cfg["measure_scale_norm"] = True; cfg["measure_stopgrad_phi"] = sgphi
            cfg["anneal_measure"] = anneal; cfg["anneal_ramp"] = ramp
            cfg["anneal_lr_warmup"] = warm
            B = 12
            obs  = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
            nobs = jax.random.normal(jax.random.PRNGKey(4), (B, 11))
            act  = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
            goals= jax.random.normal(jax.random.PRNGKey(3), (B, 11))
            rew  = jnp.where(jax.random.uniform(jax.random.PRNGKey(5), (B,)) < 0.2, 0.0, -1.0)
            ag = GCIQLFlowAgent.create(0, obs, act, cfg)
            ag = ag.replace(network=ag.network.replace(step=step))   # TrainState is a flax struct
            batch = dict(observations=obs, next_observations=nobs, actions=act,
                         rewards=rew, masks=1.0 + rew, value_goals=goals, actor_goals=goals)
            return ag, batch
"""


def test_lambda_ramps_linearly_then_holds():
    """DECISIVE for the schedule. lam(t) = lam_max * clip(t/ramp, 0, 1): zero at t=0, half at the
    midpoint, exactly lam_max at t=ramp, and HELD at lam_max beyond it. Without the hold we would
    only ever see the transient, never the steady state at full dose."""
    code = PREAMBLE + """
        for t, expect in [(0, 0.0), (50, 5.0), (100, 10.0), (200, 10.0), (10_000, 10.0)]:
            ag, batch = build(step=t, lam_max=10.0, ramp=100)
            _, info = ag.total_loss(batch, ag.network.params, jax.random.PRNGKey(7))
            got = float(info["anneal/lambda"])
            assert abs(got - expect) < 1e-5, (t, got, expect)
            assert abs(float(info["anneal/t"]) - t) < 1e-5
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_lambda_zero_contributes_no_gradient():
    """At t=0 the SM term is multiplied by exactly 0, so the gradient of total_loss must equal the
    pure-IQL gradient. This is what makes the very first finetune step a no-op on the restored
    geometry -- and what makes the first eval a free restore check (it must read ~0.90)."""
    code = PREAMBLE + """
        ag0, batch = build(step=0, lam_max=10.0, ramp=100, anneal=True)
        agN, _     = build(step=0, lam_max=0.0,  ramp=100, anneal=False)   # pure IQL
        g0 = jax.grad(lambda p: ag0.total_loss(batch, p, jax.random.PRNGKey(7))[0])(ag0.network.params)
        gN = jax.grad(lambda p: agN.total_loss(batch, p, jax.random.PRNGKey(7))[0])(agN.network.params)
        d = jax.tree_util.tree_map(lambda a, b: jnp.max(jnp.abs(a - b)), g0["modules_critic"], gN["modules_critic"])
        worst = max(float(x) for x in jax.tree_util.tree_leaves(d))
        assert worst < 1e-8, ("lambda(0) must contribute no gradient", worst)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_measure_loss_is_still_computed_at_lambda_zero():
    """Static shapes under jit: the measure loss must still be COMPUTED at t=0 (and logged), just
    scaled to zero. If it were skipped, the jitted graph would change shape mid-run the moment
    lambda became positive."""
    code = PREAMBLE + """
        ag, batch = build(step=0, lam_max=10.0, ramp=100)
        _, info = ag.total_loss(batch, ag.network.params, jax.random.PRNGKey(7))
        assert "measure/measure_loss" in info, sorted(info)
        assert bool(jnp.isfinite(info["measure/measure_loss"]))
        assert float(info["anneal/lambda"]) == 0.0
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_gciql_anneal.py -v`
Expected: all three FAIL with `KeyError: 'anneal/lambda'` (the info key does not exist; `anneal_measure` is ignored).

- [ ] **Step 3: Add the three config knobs**

In `get_config()`, immediately after `config.measure_z_random_frac = 0.0`, insert:

```python
        config.anneal_measure = False    # ramp the SM loss weight instead of holding it constant: lam(t) = measure_coef * clip(step/anneal_ramp, 0, 1), then HOLD. measure_coef is reinterpreted as lam_max. Warm-start finetuning only (see scripts/prepare_anneal_ckpt.py, which resets network.step to 0)
        config.anneal_ramp = 300_000     # linear ramp length in updates; lam holds at lam_max afterwards
        config.anneal_lr_warmup = 0      # linear LR warmup length in updates (0 = constant LR, today's behaviour). NOTE: a nonzero value changes the optax opt_state tree, so such a run can only restore from a checkpoint written by prepare_anneal_ckpt.py
```

- [ ] **Step 4: Scale the measure term by λ(t) in `total_loss`**

Replace the measure block (`gciql_flow.py:925-930`):

```python
        if self.config.get('measure_coef', 0.0) > 0:   # graft FB measure onto GCIQL's rep
            rng, m_rng = jax.random.split(rng)
            m_loss, m_info = self.measure_loss(batch, grad_params, m_rng)
            loss = loss + self.config['measure_coef'] * m_loss
            for k, v in m_info.items():
                info[f'measure/{k}'] = v
```

with:

```python
        if self.config.get('measure_coef', 0.0) > 0:   # graft FB measure onto GCIQL's rep
            rng, m_rng = jax.random.split(rng)
            m_loss, m_info = self.measure_loss(batch, grad_params, m_rng)
            lam = self.config['measure_coef']
            if self.config.get('anneal_measure', False):
                # Warm-start dose-response: grow the SM weight from 0, then HOLD at lam_max.
                # t is network.step, which prepare_anneal_ckpt.py resets to 0 -- restore_agent
                # otherwise brings back the anchor's step (1_000_001) and the ramp would start
                # fully saturated. The measure loss is still COMPUTED at lam=0 (static jit shapes),
                # just scaled to zero, so the first finetune step is a no-op on the restored geometry.
                # jnp.asarray, not .astype: TrainState.step is a plain Python int leaf until jit
                # traces it, and `int.astype` does not exist.
                t = jnp.asarray(self.network.step, jnp.float32)
                ramp = float(self.config.get('anneal_ramp', 300_000))
                lam = lam * jnp.clip(t / ramp, 0.0, 1.0)
                info['anneal/lambda'] = lam
                info['anneal/t'] = t
            loss = loss + lam * m_loss
            for k, v in m_info.items():
                info[f'measure/{k}'] = v
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_gciql_anneal.py -v`
Expected: 3 passed.

- [ ] **Step 6: Verify nothing existing moved**

Run: `.venv/bin/python -m pytest tests/test_gciql_measure_scale.py tests/test_gciql_fb_forward_z.py -q`
Expected: `7 passed` (these exercise `measure_loss` and `total_loss` with `anneal_measure=False`).

- [ ] **Step 7: Commit**

```bash
git add tools/wandb_mode_shim/gciql_flow.py tests/test_gciql_anneal.py
git commit -m "feat(gciql): anneal_measure -- ramp-then-hold lambda on the SM loss

lam(t) = measure_coef * clip(step/anneal_ramp, 0, 1), holding at lam_max afterwards. Warm-start
dose-response: L = L_IQL + lam(t)*L_SM, with L_IQL untouched.

t is network.step. prepare_anneal_ckpt.py resets it to 0, because restore_agent deserializes the
whole TrainState and would otherwise bring back the anchor's step (1_000_001) -- a schedule keyed
on it would start fully ramped and the run would silently be a constant-lambda cold start.

The measure loss is still computed at lam=0 (static jit shapes), just scaled to zero, so the first
finetune step is a no-op on the restored geometry and the first eval is a free restore check."
```

---

### Task 2: The LR warmup

**Files:**
- Modify: `tools/wandb_mode_shim/gciql_flow.py:1244` (`network_tx`), `get_config()` (already done in Task 1)
- Test: `tests/test_gciql_anneal.py`

**Interfaces:**
- Consumes: `anneal_lr_warmup` (Task 1 Step 3).
- Produces: when `anneal_lr_warmup > 0`, `create()` builds `network_tx = optax.adam(learning_rate=optax.linear_schedule(0.0, config['lr'], anneal_lr_warmup))`. This **changes the `opt_state` tree**, which Task 3's prep script exists to accommodate.

- [ ] **Step 1: Write the two failing tests**

Append to `tests/test_gciql_anneal.py`:

```python
def test_lr_warmup_schedule_is_linear_then_constant():
    """optax.linear_schedule(0, lr, N) rises linearly and HOLDS at lr past N. The warmup exists to
    protect the restored weights from a large first step. It cannot be faked by scaling the loss:
    Adam is approximately scale-invariant per parameter (m/sqrt(v)), so a global loss factor is a
    no-op. lambda works only because it changes the RELATIVE weight of L_SM against L_IQL."""
    code = """
        import optax, jax.numpy as jnp
        lr, N = 3e-4, 10
        s = optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=N)
        for c, expect in [(0, 0.0), (5, lr/2), (10, lr), (20, lr), (10_000, lr)]:
            got = float(s(c))
            assert abs(got - expect) < 1e-12, (c, got, expect)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_scheduled_optimizer_changes_the_opt_state_tree():
    """THE FACT THAT FORCES THE PREP SCRIPT. optax.adam(constant lr) ends in EmptyState;
    optax.adam(callable lr) ends in ScaleByScheduleState(count). restore_agent deserializes the
    whole TrainState, so a checkpoint saved with a constant LR CANNOT restore into a scheduled
    optimizer. Asserted explicitly so a future optax that unifies them is caught here, not by a
    five-hour run that dies on restore."""
    code = """
        import jax, jax.numpy as jnp, optax
        p = {'w': jnp.ones((3,))}
        const = optax.adam(learning_rate=3e-4)
        sched = optax.adam(learning_rate=optax.linear_schedule(0.0, 3e-4, 10))
        ts_c = jax.tree_util.tree_structure(const.init(p))
        ts_s = jax.tree_util.tree_structure(sched.init(p))
        assert ts_c != ts_s, ("optax unified the trees; prepare_anneal_ckpt.py may be simplifiable", ts_c)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_agent_builds_a_scheduled_tx_only_when_warmup_is_set():
    """anneal_lr_warmup=0 must reproduce today's optimizer exactly (EmptyState); >0 must produce the
    scheduled one. This is the switch prepare_anneal_ckpt.py has to mirror."""
    code = PREAMBLE + """
        import optax
        a0, _ = build(warm=0)
        a1, _ = build(warm=10)
        t0 = jax.tree_util.tree_structure(a0.network.opt_state)
        t1 = jax.tree_util.tree_structure(a1.network.opt_state)
        assert t0 != t1, "anneal_lr_warmup did not change the optimizer"
        ref = jax.tree_util.tree_structure(optax.adam(learning_rate=3e-4).init(a0.network.params))
        assert t0 == ref, "warmup=0 must reproduce today's constant-LR adam exactly"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
```

- [ ] **Step 2: Run them to verify the third fails**

Run: `.venv/bin/python -m pytest tests/test_gciql_anneal.py -k "lr_warmup or opt_state or scheduled_tx" -v`
Expected: the first two PASS (they test optax directly, no agent code needed); `test_agent_builds_a_scheduled_tx_only_when_warmup_is_set` FAILS with `AssertionError: anneal_lr_warmup did not change the optimizer`.

- [ ] **Step 3: Build the scheduled optimizer**

Replace `gciql_flow.py:1244`:

```python
        network_tx = optax.adam(learning_rate=config['lr'])
```

with:

```python
        # anneal_lr_warmup: linear LR warmup, then constant. Protects the restored weights from a
        # large first step, and lets Adam's fresh moments adapt before the SM gradient grows.
        # It CANNOT be faked by scaling the loss (Adam is ~scale-invariant per parameter), so it
        # must be a real LR schedule -- which changes the opt_state tree and is why such a run can
        # only restore from a checkpoint written by scripts/prepare_anneal_ckpt.py.
        warmup = int(config.get('anneal_lr_warmup', 0))
        if warmup > 0:
            lr_sched = optax.linear_schedule(init_value=0.0, end_value=config['lr'],
                                             transition_steps=warmup)
            network_tx = optax.adam(learning_rate=lr_sched)
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_gciql_anneal.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/wandb_mode_shim/gciql_flow.py tests/test_gciql_anneal.py
git commit -m "feat(gciql): anneal_lr_warmup -- linear LR warmup via optax.linear_schedule

The warmup protects the restored weights from a large first step and lets Adam's fresh moments
adapt before the SM gradient grows. It cannot be faked by scaling the loss: Adam is approximately
scale-invariant per parameter, so a global loss factor is a no-op. lambda works only because it
changes the RELATIVE weight of L_SM against L_IQL.

A real LR schedule changes the optax opt_state tree (EmptyState -> ScaleByScheduleState), so a
scheduled run can only restore from a checkpoint written by prepare_anneal_ckpt.py. Asserted by a
test, so a future optax that unifies the trees is caught here rather than by a dead five-hour run.

anneal_lr_warmup=0 reproduces today's optimizer exactly."
```

---

### Task 3: `scripts/prepare_anneal_ckpt.py`

**Files:**
- Create: `scripts/prepare_anneal_ckpt.py`
- Test: `tests/test_gciql_anneal.py`

**Interfaces:**
- Consumes: `anneal_lr_warmup` (Task 2) — the fresh `opt_state` must come from the *finetune* config's `tx`.
- Produces: a directory containing `params_0.pkl`, restorable by `restore_agent(agent, <dir>, 0)` for an agent built with the finetune config. Callable as
  `prepare_anneal_ckpt.py --src <anchor_run_dir> --epoch 1000000 --out <dir> --lr-warmup 10000`.

- [ ] **Step 1: Write the failing round-trip test**

Append to `tests/test_gciql_anneal.py`:

```python
def test_prepare_anneal_ckpt_round_trip():
    """DECISIVE for the whole experiment. The prep script must produce a checkpoint that:
      * restores into an agent built with the FINETUNE config (scheduled tx),
      * carries the anchor's params byte-for-byte,
      * has step == 0 (so lam(0) = 0 and the LR warmup starts at 0), and
      * has a FRESH opt_state matching the scheduled tx (not the anchor's constant-LR one).
    Then the restored agent must take an update without error."""
    code = PREAMBLE + """
        import pickle, os, sys, flax, tempfile
        sys.path.insert(0, os.path.join(os.environ['REPO'], 'scripts'))
        from prepare_anneal_ckpt import splice_checkpoint

        # 1. build a "source" agent with today's constant-LR tx, and save it like main.py would
        src_agent, batch = build(step=1234, lam_max=0.0, anneal=False, warm=0)
        tmp = tempfile.mkdtemp()
        src_dir = os.path.join(tmp, 'src'); os.makedirs(src_dir)
        with open(os.path.join(src_dir, 'params_1234.pkl'), 'wb') as f:
            pickle.dump(dict(agent=flax.serialization.to_state_dict(src_agent)), f)

        # 2. build the FINETUNE agent (scheduled tx) and splice
        ft_agent, _ = build(step=99, lam_max=10.0, anneal=True, warm=10)
        out_dir = os.path.join(tmp, 'out')
        splice_checkpoint(ft_agent, os.path.join(src_dir, 'params_1234.pkl'), out_dir)

        # 3. restore, exactly as main.py does
        from utils.flax_utils import restore_agent
        restored = restore_agent(ft_agent, out_dir, 0)

        # params equal the source's
        d = jax.tree_util.tree_map(lambda a, b: jnp.max(jnp.abs(a - b)),
                                   restored.network.params, src_agent.network.params)
        assert max(float(x) for x in jax.tree_util.tree_leaves(d)) == 0.0, "params not preserved"
        # step reset
        assert int(restored.network.step) == 0, int(restored.network.step)
        # opt_state matches a FRESH init of the scheduled tx (not the source's)
        assert (jax.tree_util.tree_structure(restored.network.opt_state)
                == jax.tree_util.tree_structure(ft_agent.network.opt_state))
        # and it trains
        restored, info = restored.update(batch)
        assert bool(jnp.isfinite(info['actor/actor_loss']))
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS), env_extra={'REPO': str(OGBENCH_IMPLS.parents[2])})
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gciql_anneal.py::test_prepare_anneal_ckpt_round_trip -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prepare_anneal_ckpt'`.

- [ ] **Step 3: Write the prep script**

Create `scripts/prepare_anneal_ckpt.py`:

```python
#!/usr/bin/env python
"""prepare_anneal_ckpt.py — write a finetune-ready checkpoint for annealed SM finetuning.

Two facts about `restore_agent` make this necessary (spec section 3):

1. It deserializes the WHOLE TrainState, so `network.step` comes back as the anchor's value
   (1_000_001). A lambda schedule keyed on `network.step` would therefore start fully ramped and
   the run would silently be a constant-lambda cold start.
2. `optax.adam(learning_rate=<callable>)` ends in `ScaleByScheduleState`, while a constant LR ends
   in `EmptyState`. So an anchor saved with a constant LR CANNOT restore into a scheduled optimizer.

Rather than fight either, we write a new checkpoint: the anchor's params, a FRESH `opt_state` taken
from the finetune agent's own (scheduled) `tx`, and `step = 0`.

Fresh Adam moments are deliberate. Restored moments are tiny at convergence, so the first SM
gradient would be amplified by 1/sqrt(v) -- a concrete mechanism by which one large early step
destroys the value geometry and manufactures a spurious "the measure objective is incompatible"
result. Fresh moments + the LR warmup remove that failure mode.

Run under the jax venv, from third_party/ogbench/impls:

    cd third_party/ogbench/impls && /dev/shm/.venv-jax/bin/python \\
        ../../../scripts/prepare_anneal_ckpt.py \\
        --src ~/bridge_ckpt_backup/.../GCIQL-State-Flow-Bilinear-d50/sd000_20260615_011444 \\
        --epoch 1000000 --out /dev/shm/anneal_ckpt/sd000 --lr-warmup 10000
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import flax
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'tools', 'wandb_mode_shim'))


def splice_checkpoint(finetune_agent, src_pkl: str, out_dir: str) -> str:
    """Write `<out_dir>/params_0.pkl`: the source checkpoint's params inside `finetune_agent`'s
    state (fresh opt_state, step=0). Returns the written path.

    `finetune_agent` must already be constructed with the finetune config, so its `opt_state`
    matches the scheduled `tx`. We copy ONLY `network.params` across; everything else stays as the
    finetune agent built it."""
    with open(src_pkl, 'rb') as f:
        src = pickle.load(f)['agent']

    state = flax.serialization.to_state_dict(finetune_agent)
    src_params = src['network']['params']

    # The param trees must match exactly, or the architectures differ (e.g. fb_forward_z changes
    # phi's input dim). Fail loudly here rather than mid-run.
    tgt_params = state['network']['params']
    s_struct = jax.tree_util.tree_structure(src_params)
    t_struct = jax.tree_util.tree_structure(tgt_params)
    if s_struct != t_struct:
        raise SystemExit(
            f'param tree mismatch: the finetune config does not match the checkpoint architecture\n'
            f'  source: {s_struct}\n  target: {t_struct}')
    for a, b in zip(jax.tree_util.tree_leaves(src_params), jax.tree_util.tree_leaves(tgt_params)):
        if a.shape != b.shape:
            raise SystemExit(f'param shape mismatch: {a.shape} vs {b.shape}')

    state['network']['params'] = src_params
    state['network']['step'] = 0          # <- lam(0)=0 and the LR warmup both start here
    # state['network']['opt_state'] is left as the finetune agent's FRESH init (deliberate)

    spliced = flax.serialization.from_state_dict(finetune_agent, state)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'params_0.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(dict(agent=flax.serialization.to_state_dict(spliced)), f)
    return out_path


def _dims_from_params(params):
    """Infer (obs_dim, action_dim) from the bilinear param tree, so the script needs no env.

    value.phi  Dense_0 kernel: (obs_dim, hidden)              [ensemble=False]
    critic.phi Dense_0 kernel: (2, obs_dim + action_dim, hidden)  [ensemble=True -> leading head axis]
    """
    obs_dim = params['modules_value']['phi']['Dense_0']['kernel'].shape[0]
    critic_in = params['modules_critic']['phi']['Dense_0']['kernel'].shape[1]
    return obs_dim, critic_in - obs_dim


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='anchor run directory containing params_<epoch>.pkl')
    ap.add_argument('--epoch', type=int, default=1_000_000)
    ap.add_argument('--out', required=True, help='output directory; receives params_0.pkl')
    ap.add_argument('--lr-warmup', type=int, default=10_000)
    ap.add_argument('--latent-dim', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    from gciql_flow import GCIQLFlowAgent, get_config

    src_pkl = os.path.join(args.src, f'params_{args.epoch}.pkl')
    if not os.path.exists(src_pkl):
        raise SystemExit(f'no such checkpoint: {src_pkl}')
    with open(src_pkl, 'rb') as f:
        src_params = pickle.load(f)['agent']['network']['params']
    obs_dim, act_dim = _dims_from_params(src_params)
    print(f'  inferred obs_dim={obs_dim} action_dim={act_dim} from the checkpoint')

    cfg = dict(get_config())
    cfg['encoder'] = None
    cfg['frame_stack'] = None
    cfg['value_arch'] = 'bilinear'
    cfg['value_latent_dim'] = args.latent_dim
    cfg['anneal_lr_warmup'] = args.lr_warmup      # <- makes opt_state match the finetune tx
    ex_obs = jnp.zeros((2, obs_dim))
    ex_act = jnp.zeros((2, act_dim))
    agent = GCIQLFlowAgent.create(args.seed, ex_obs, ex_act, cfg)

    out = splice_checkpoint(agent, src_pkl, args.out)
    print(f'  wrote {out}  (step=0, fresh opt_state, lr_warmup={args.lr_warmup})')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gciql_anneal.py::test_prepare_anneal_ckpt_round_trip -v`
Expected: PASS.

- [ ] **Step 5: Prepare the three real checkpoints and verify they restore**

Run:

```bash
cd /home/adityamohan/git/Factored-FB/third_party/ogbench/impls
for s in 000 001 002; do
  SRC=$(ls -d ~/bridge_ckpt_backup/gciql_outputs/factored-fb/GCIQL-State-Flow-Bilinear-d50/sd${s}_*)
  /dev/shm/.venv-jax/bin/python ../../../scripts/prepare_anneal_ckpt.py \
    --src "$SRC" --epoch 1000000 --out /dev/shm/anneal_ckpt/sd${s} --lr-warmup 10000 --seed ${s#00}
done
ls -la /dev/shm/anneal_ckpt/*/params_0.pkl
```

Expected: three `params_0.pkl` files, each printing `inferred obs_dim=29 action_dim=5`.

- [ ] **Step 6: Commit**

```bash
git add scripts/prepare_anneal_ckpt.py tests/test_gciql_anneal.py
git commit -m "feat(scripts): prepare_anneal_ckpt -- finetune-ready checkpoint for the SM anneal

restore_agent deserializes the whole TrainState, so (a) network.step comes back as the anchor's
1_000_001 and a schedule keyed on it would start fully ramped, and (b) an LR schedule changes the
opt_state tree so a constant-LR anchor cannot restore into a scheduled optimizer at all.

This script writes a new checkpoint instead: the anchor's params, a FRESH opt_state taken from the
finetune agent's own scheduled tx, and step=0. No third_party file is touched.

Fresh Adam moments are deliberate: restored moments are tiny at convergence, so the first SM
gradient would be amplified by 1/sqrt(v) -- a mechanism that would destroy the value geometry and
manufacture a spurious 'the measure objective is incompatible' result.

Param trees are compared before splicing, so an architecture mismatch fails loudly here rather
than mid-run. Round-trip test asserts params preserved, step==0, fresh opt_state, and that the
restored agent trains."
```

---

### Task 4: The five guards

**Files:**
- Modify: `tools/wandb_mode_shim/gciql_flow.py` (`create()`, after `config['action_dim'] = action_dim`)
- Test: `tests/test_gciql_anneal.py`

**Interfaces:**
- Consumes: knobs from Tasks 1–2.
- Produces: `create()` raises `AssertionError` on invalid `anneal_measure` combinations; the three cells construct.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gciql_anneal.py`:

```python
def test_anneal_guards_reject_invalid_combinations():
    """Guards encode the design so a misconfigured cell fails loudly rather than producing an
    uninterpretable number.

    1. anneal_measure => not measure_only (we graft onto the IQL losses, not replace them)
    2. anneal_measure => measure_scale_norm (raw measure grafts blew the Q scale, 18/18 crashed)
    3. anneal_measure => measure_legs == 'critic'
    4. anneal_measure => anneal_ramp > 0 and measure_coef >= 0
    5. anneal_measure => PARAM-TREE IDENTITY with the checkpoint: value_arch == 'bilinear',
       psi_base == 'learned', and none of fb_forward_z / fwd_two_tower / isfb / sf_feature_td /
       simplesf / the task-norm knobs (each changes the param tree and breaks restore)."""
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
        A = dict(anneal_measure=True, measure_coef=10.0, measure_legs="critic",
                 measure_scale_norm=True, value_arch="bilinear", anneal_ramp=1000)
        bad = [
            dict(A, measure_only=True),                       # 1
            dict(A, measure_scale_norm=False),                # 2
            dict(A, measure_legs="value"),                    # 3
            dict(A, anneal_ramp=0),                           # 4
            dict(A, value_arch="mlp"),                        # 5
            dict(A, fb_forward_z=True),                       # 5
            dict(A, fwd_two_tower=True),                      # 5
            dict(A, psi_base="frozen"),                       # 5
            dict(A, measure_test_states="goals"),             # 5
        ]
        for over in bad:
            try:
                build(**over)
            except AssertionError:
                continue
            raise SystemExit("guard missing for " + repr(over))
        build(**A, measure_stopgrad_phi=True,  anneal_lr_warmup=10)   # Anneal-SgPhi
        build(**A, measure_stopgrad_phi=False, anneal_lr_warmup=10)   # Anneal-Shared
        build(value_arch="bilinear", anneal_lr_warmup=10)             # Anneal-Zero (no SM loss)
        build(value_arch="bilinear")                                  # stock cells unaffected
        build()                                                       # mlp default unaffected
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gciql_anneal.py::test_anneal_guards_reject_invalid_combinations -v`
Expected: FAIL with `SystemExit: guard missing for ...`.

- [ ] **Step 3: Add the guards**

In `create()`, immediately after `config['action_dim'] = action_dim`, insert:

```python
        if config.get('anneal_measure', False):
            # (1) we graft the SM loss onto the IQL losses; measure_only would replace them.
            assert not config.get('measure_only', False), \
                "anneal_measure grafts onto the IQL losses; measure_only=True replaces them"
            # (2) raw (non-scale-normed) measure grafts blew the Q scale and crashed 18/18 runs
            # (audit-20260706 section 1). Every surviving E2 cell used scale_norm.
            assert config.get('measure_scale_norm', False), \
                "anneal_measure requires measure_scale_norm=True (raw measure grafts blow the Q scale)"
            # (3) the value-legs measure is a different object; the dose-response is on the critic.
            assert config.get('measure_legs', 'critic') == 'critic', \
                "anneal_measure requires measure_legs='critic'"
            # (4) a zero-length ramp would divide by zero; lam_max may be 0 (the control cell).
            assert int(config.get('anneal_ramp', 300_000)) > 0, "anneal_ramp must be > 0"
            assert config.get('measure_coef', 0.0) >= 0.0, "measure_coef (lam_max) must be >= 0"
            # (5) PARAM-TREE IDENTITY with the warm-start checkpoint. Any of these changes the
            # parameter tree, so restore_agent would fail (or, worse, the prep script's splice
            # would). Refuse at construction.
            assert config.get('value_arch', 'mlp') == 'bilinear', \
                "anneal_measure requires value_arch='bilinear' (the checkpoint's architecture)"
            assert config.get('psi_base', 'learned') == 'learned', \
                "anneal_measure requires psi_base='learned' (psi_base changes the critic class)"
            for knob in ('fb_forward_z', 'fwd_two_tower', 'isfb', 'sf_feature_td'):
                assert not config.get(knob, False), \
                    f"anneal_measure is incompatible with {knob} (it changes the param tree)"
            assert config.get('measure_test_states', 'rho') == 'rho' \
                and config.get('measure_test_rank', 'full') == 'full', \
                "anneal_measure is incompatible with the task-norm knobs"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gciql_anneal.py -v`
Expected: 8 passed.

- [ ] **Step 5: Run every affected suite**

Run: `.venv/bin/python -m pytest tests/test_gciql_anneal.py tests/test_gciql_measure_scale.py tests/test_gciql_fb_forward_z.py tests/test_gciql_task_norm.py tests/test_gciql_simplesf.py tests/test_gciql_sf_feature_td.py tests/test_gciql_psi_base.py tests/test_gciql_sf_joint_psi.py -q`
Expected: `57 passed`, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add tools/wandb_mode_shim/gciql_flow.py tests/test_gciql_anneal.py
git commit -m "feat(gciql): five guards for anneal_measure

 1. not measure_only (we graft onto the IQL losses, not replace them)
 2. measure_scale_norm required (raw measure grafts blew the Q scale, 18/18 crashed)
 3. measure_legs='critic'
 4. anneal_ramp > 0 and measure_coef >= 0
 5. param-tree identity with the warm-start checkpoint: value_arch='bilinear', psi_base='learned',
    and none of fb_forward_z / fwd_two_tower / isfb / sf_feature_td / the task-norm knobs

Guard 5 matters most: any of those changes the parameter tree, so restore would fail -- and the
prep script's splice would fail first. Better to refuse at construction.

57 passed across all eight gciql suites."
```

---

### Task 5: Configs, results_map, launcher (DRY-RUN ONLY)

**Files:**
- Create: three configs under `configs/gciql/`, `scripts/launch_anneal.sh`
- Modify: `scripts/results_map.py`

**Interfaces:**
- Consumes: knobs, guards, and the prep script.
- Produces: wandb groups `GCIQL-State-Flow-Anneal-Zero`, `GCIQL-State-Flow-Anneal-SgPhi`, `GCIQL-State-Flow-Anneal-Shared`.

**DO NOT LAUNCH.** This task ends at a dry-run.

- [ ] **Step 1: Write the λ=0 control config**

Create `configs/gciql/cube_single_state_gciql_flow_anneal_zero.yaml`:

```yaml
# Anneal-Zero (sf-design-space 1f): the CONTROL. Restore the converged bilinear GCIQL anchor and
# continue training with NO successor-measure loss at all (measure_coef=0), same LR warmup.
#
# Two jobs:
#   1. validates the restore pipeline end-to-end -- the first eval must read ~0.90;
#   2. isolates "400k more IQL steps" from "the SM loss". If this drifts, every anneal number must
#      be re-baselined against it rather than against the 0.904 anchor.
#
# restore_path is supplied per-seed by scripts/launch_anneal.sh (it points at the finetune-ready
# checkpoint written by scripts/prepare_anneal_ckpt.py, which sets step=0 and a fresh opt_state).
defaults:
  - storage: shm
  - _self_
variant: state
env_name: cube-single-play-v0
agent_file: tools/wandb_mode_shim/gciql_flow.py
seed: 0
train_steps: 400000
log_interval: 5000
eval_interval: 25000
save_interval: 400000
eval_episodes: 50
eval_on_cpu: 1
run_group: GCIQL-State-Flow-Anneal-Zero
restore_path: null
restore_epoch: 0
exp_name: null
agent_overrides:
  bc_coeff: 3.0
  value_arch: bilinear
  value_latent_dim: 50
  measure_coef: 0.0
  anneal_measure: false
  anneal_lr_warmup: 10000
wandb_mode: online
```

- [ ] **Step 2: Write the stopgrad-φ config**

Create `configs/gciql/cube_single_state_gciql_flow_anneal_sgphi.yaml`:

```yaml
# Anneal-SgPhi (sf-design-space 1f): warm-start from converged bilinear GCIQL (0.904 +/- 0.019) and
# grow the SM loss weight from 0 to 10 over 300k, then HOLD for 100k.
#
# This ramp crosses a KNOWN cold-start cliff. E2, constant lambda, cold:
#   sgphi l1 = 0.857   l3 = 0.176   l10 = 0.024
# So:
#   survives past lambda=3  => the cold collapse was an OPTIMIZATION PATH artifact
#   collapses near lambda~3 => the FIXED POINT itself is incompatible with a good value geometry
#
# PRE-REGISTERED: critic/q_mean must stay in [-21, -8]. Raw measure grafts blew the Q scale and
# crashed 18/18; every surviving E2 cell sat in that band. Outside it the cell is a scale blowup and
# its success number is uninterpretable -- report that BEFORE reporting success.
#
# The two schedules do not confound: at t=10k (LR at full value) lambda = 10/30 = 0.33, an order of
# magnitude below the cliff. The whole dose-response happens at constant LR.
defaults:
  - storage: shm
  - _self_
variant: state
env_name: cube-single-play-v0
agent_file: tools/wandb_mode_shim/gciql_flow.py
seed: 0
train_steps: 400000
log_interval: 5000
eval_interval: 25000
save_interval: 400000
eval_episodes: 50
eval_on_cpu: 1
run_group: GCIQL-State-Flow-Anneal-SgPhi
restore_path: null
restore_epoch: 0
exp_name: null
agent_overrides:
  bc_coeff: 3.0
  value_arch: bilinear
  value_latent_dim: 50
  measure_only: false
  measure_coef: 10.0          # lam_max
  measure_legs: critic
  measure_ortho_coef: 0.0
  measure_scale_norm: true
  measure_stopgrad_phi: true
  anneal_measure: true
  anneal_ramp: 300000
  anneal_lr_warmup: 10000
wandb_mode: online
```

- [ ] **Step 3: Write the shared-legs config**

Create `configs/gciql/cube_single_state_gciql_flow_anneal_shared.yaml`:

```yaml
# Anneal-Shared (sf-design-space 1f): same warm start, shared legs (phi may co-adapt), lambda grows
# 0 -> 100 over 300k then HOLDs for 100k.
#
# Shared legs absorb even lambda=10 cold (E2-mix-l10 = 0.917 +/- 0.041), so their cliff is ABOVE 10
# and has never been located. Ramping to 100 asks how much measure pressure a good value geometry
# can absorb when phi is free to co-adapt. Combined with mdose (phi co-adaptation >= 10x margin),
# a large margin here localizes the sgphi conflict to the psi leg.
#
# PRE-REGISTERED: critic/q_mean must stay in [-21, -8], else the cell is a scale blowup and its
# success number is uninterpretable.
defaults:
  - storage: shm
  - _self_
variant: state
env_name: cube-single-play-v0
agent_file: tools/wandb_mode_shim/gciql_flow.py
seed: 0
train_steps: 400000
log_interval: 5000
eval_interval: 25000
save_interval: 400000
eval_episodes: 50
eval_on_cpu: 1
run_group: GCIQL-State-Flow-Anneal-Shared
restore_path: null
restore_epoch: 0
exp_name: null
agent_overrides:
  bc_coeff: 3.0
  value_arch: bilinear
  value_latent_dim: 50
  measure_only: false
  measure_coef: 100.0         # lam_max
  measure_legs: critic
  measure_ortho_coef: 0.0
  measure_scale_norm: true
  measure_stopgrad_phi: false
  anneal_measure: true
  anneal_ramp: 300000
  anneal_lr_warmup: 10000
wandb_mode: online
```

- [ ] **Step 4: Verify all three configs construct an agent**

Run:

```bash
cd /home/adityamohan/git/Factored-FB/third_party/ogbench/impls && /dev/shm/.venv-jax/bin/python -c "
import sys, yaml; sys.path.insert(0,'/home/adityamohan/git/Factored-FB/tools/wandb_mode_shim')
import jax
from gciql_flow import GCIQLFlowAgent, get_config
for n in ['anneal_zero','anneal_sgphi','anneal_shared']:
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
ok anneal_zero
ok anneal_sgphi
ok anneal_shared
```

- [ ] **Step 5: Register the three groups in `results_map.py`**

In `scripts/results_map.py`, immediately **above** the line beginning
`# --- 1e: task-weighted successor moments`, insert:

```python
    # --- 1f: annealed SM finetuning from the converged bilinear-GCIQL anchor (0.904). lambda(t)
    # ramps 0 -> lam_max over 300k then holds; dense evals make each run a success-vs-lambda dose
    # curve. Separates fixed-point incompatibility from an optimization-path artifact.
    'GCIQL-State-Flow-Anneal-Zero':   dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='anneal-zero', graft='CONTROL: finetune the 0.904 anchor 400k more steps with NO measure loss (validates restore; isolates extra IQL steps)'),
    'GCIQL-State-Flow-Anneal-SgPhi':  dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='anneal-sgphi', graft='warm-start + lambda 0->10 ramp on the stopgrad-phi measure (crosses the cold cliff at lambda=3)'),
    'GCIQL-State-Flow-Anneal-Shared': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='anneal-shared', graft='warm-start + lambda 0->100 ramp on the shared-legs measure (cold cliff is above 10, never located)'),
```

- [ ] **Step 6: Verify the map still accounts for every group**

Run: `.venv/bin/python scripts/compile_results_csvs.py 2>&1 | tail -3`
Expected: `[compile] audit -> .../compile_audit.txt  (all groups accounted for)` and exit 0. (The three new groups appear under `MAPPED BUT ABSENT FROM WANDB` until the runs start — a note, not a failure.)

- [ ] **Step 7: Write the launcher**

Create `scripts/launch_anneal.sh` and `chmod +x` it:

```bash
#!/usr/bin/env bash
# Annealed SM finetuning (spec docs/superpowers/specs/2026-07-10-annealed-sm-finetune-design.md).
#
#   Anneal-Zero    control: no measure loss. First eval MUST read ~0.90 or the restore is broken.
#   Anneal-SgPhi   lambda 0 -> 10  over 300k, hold 100k. Crosses the cold cliff (l3 = 0.176).
#   Anneal-Shared  lambda 0 -> 100 over 300k, hold 100k. Cold cliff above 10, never located.
#
# Step 0 writes the finetune-ready checkpoints (anchor params, fresh opt_state, step=0). Without
# them the LR schedule cannot restore at all, and lambda would start fully ramped.
#
# ALWAYS in tmux:
#   tmux new -d -s anneal 'DRY_RUN=0 scripts/launch_anneal.sh 2>&1 | tee /dev/shm/anneal_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
PY=${PY:-$REPO/.venv/bin/python}    # run_gciql.py is a hydra launcher: it runs HERE and
                                    # subprocesses ogbench main.py into .venv-jax itself.
                                    # Do NOT point this at .venv-jax -- that venv has no hydra.
JAXPY=${JAXPY:-/dev/shm/.venv-jax/bin/python}
DRY_RUN=${DRY_RUN:-1}
STEPS=${STEPS:-400000}
CKPT_ROOT=${CKPT_ROOT:-/dev/shm/anneal_ckpt}
ANCHOR=~/bridge_ckpt_backup/gciql_outputs/factored-fb/GCIQL-State-Flow-Bilinear-d50
LOGDIR=/dev/shm/anneal_logs
mkdir -p "$LOGDIR"

# ---- step 0: finetune-ready checkpoints (idempotent) -------------------------------------------
prepare() {
  for s in 000 001 002; do
    if [ -f "$CKPT_ROOT/sd${s}/params_0.pkl" ]; then
      echo "[prep] sd${s} already present"
      continue
    fi
    SRC=$(ls -d ${ANCHOR}/sd${s}_* 2>/dev/null | head -1)
    [ -z "$SRC" ] && { echo "[prep] MISSING anchor for sd${s}"; exit 1; }
    echo "[prep] sd${s} <- $SRC"
    ( cd third_party/ogbench/impls && "$JAXPY" "$REPO/scripts/prepare_anneal_ckpt.py" \
        --src "$SRC" --epoch 1000000 --out "$CKPT_ROOT/sd${s}" --lr-warmup 10000 --seed "$((10#$s))" )
  done
}

configs=(
  cube_single_state_gciql_flow_anneal_zero
  cube_single_state_gciql_flow_anneal_sgphi
  cube_single_state_gciql_flow_anneal_shared
)

JOBS=()
for cfg in "${configs[@]}"; do
  for s in 0 1 2; do
    sd=$(printf "sd%03d" "$s")
    JOBS+=("${cfg}__s${s}|||--config-name $cfg seed=$s train_steps=$STEPS save_interval=$STEPS exp_name=${cfg}_s${s} restore_path=$CKPT_ROOT/$sd restore_epoch=0")
  done
done

GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
echo "annealed SM finetune: ${#JOBS[@]} runs @${STEPS} over ${#GPUS[@]} GPUs"
if [ "$DRY_RUN" = "1" ]; then
  echo "(dry run: would prepare checkpoints under $CKPT_ROOT, then:)"
  i=0; for j in "${JOBS[@]}"; do echo "GPU${GPUS[$((i % ${#GPUS[@]}))]}  ${j%%|||*}"; i=$((i+1)); done
  exit 0
fi

prepare

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

- [ ] **Step 8: Dry-run the launcher (and stop)**

Run: `chmod +x scripts/launch_anneal.sh && scripts/launch_anneal.sh`
Expected: `annealed SM finetune: 9 runs @400000 over 8 GPUs`, a note about preparing checkpoints, then 9 `GPU<n>  <cfg>__s<seed>` lines. **No processes started, no checkpoints written.**

- [ ] **Step 9: Commit**

```bash
git add configs/gciql/cube_single_state_gciql_flow_anneal_zero.yaml \
        configs/gciql/cube_single_state_gciql_flow_anneal_sgphi.yaml \
        configs/gciql/cube_single_state_gciql_flow_anneal_shared.yaml \
        scripts/results_map.py scripts/launch_anneal.sh
git commit -m "feat(configs): annealed SM finetune cells + launcher

Anneal-Zero (control, no SM loss; first eval must read ~0.90 or restore is broken),
Anneal-SgPhi (lambda 0->10, crosses the known cold cliff at lambda=3: 0.857 -> 0.176),
Anneal-Shared (lambda 0->100; shared legs absorb lambda=10 cold at 0.917, cliff never located).

All three restore from the finetune-ready checkpoints written by prepare_anneal_ckpt.py (step=0,
fresh opt_state matching the scheduled tx). Dense evals every 25k make each run a success-vs-lambda
dose curve.

Distinct wandb groups, all registered in results_map. NOT LAUNCHED: GPUs are busy with
tasknorm+simplesf, and a code audit comes first."
```

---

## Self-Review

**Spec coverage.** §1 motivation → config headers. §1.1 cold priors → the SgPhi/Shared headers.
§2 schedules → Task 1 (λ) and Task 2 (LR); the non-confound argument → SgPhi's header. §3 the two
`restore_agent` facts → Task 2 test `..._changes_the_opt_state_tree` and Task 3's docstring.
§4 prep script → Task 3 (including the fresh-moments justification, in the module docstring and the
commit message). §5 three cells → Task 5 configs. §6 knobs → Task 1 Step 3; guards → Task 4.
§7 tests 1–8 → Task 1 (1,2,3), Task 2 (4,5, plus the agent-level tx switch), Task 3 (6), Task 4 (7),
and test 8 (backward compat) → Task 1 Step 6 + Task 2's `warmup=0 must reproduce today's adam`
assertion + Task 4 Step 5. §8 diagnostics → `anneal/lambda`, `anneal/t` in Task 1; `critic/q_mean`
is already logged by `critic_loss`. §9 reading → config headers. §10 anchors → SgPhi/Shared headers.
§11 sequencing → Task 5 ends at a dry-run.

**Deviations from the spec, flagged here.**
1. **`anneal/lr` is not logged.** The LR lives inside optax's state and is not reachable from
   `total_loss` without duplicating the schedule. `anneal/t` is logged, and the warmup is a pure
   function of it, so the LR is recoverable offline. Task 2's test pins the schedule exactly.
2. **The λ=0 control uses `anneal_measure=false, measure_coef=0.0`** rather than
   `anneal_measure=true, lam_max=0`. Identical mathematics (the measure block is gated on
   `measure_coef > 0`), and it skips the SM forward pass entirely, so the control is a pure
   IQL continuation — which is exactly what it is meant to isolate.
3. **No `anneal_t0` knob**, per spec §4: the prep script resets `step` to 0, so the offset cannot
   arise. The spec's §6 already says this; the plan does not add one.

**Type/name consistency.** Knobs spelled `anneal_measure`, `anneal_ramp`, `anneal_lr_warmup`
identically in `get_config()`, `total_loss`, `create()`'s guards, all eight tests, all three YAMLs,
and the prep script's `--lr-warmup` flag (which sets `anneal_lr_warmup`). Info keys `anneal/lambda`
and `anneal/t` are produced in Task 1 and consumed by Task 1's tests. `splice_checkpoint(finetune_agent,
src_pkl, out_dir) -> str` is defined in Task 3 and imported by Task 3's test. `_dims_from_params(params)
-> (obs_dim, action_dim)` is internal to the prep script. `optax.linear_schedule(init_value,
end_value, transition_steps)` matches optax's signature.

**No placeholders.** Every code step carries the code; every run step carries the command and its
expected output.
