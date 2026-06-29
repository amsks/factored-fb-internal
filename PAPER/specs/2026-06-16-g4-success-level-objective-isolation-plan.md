# G4 — Success-Level Objective Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a zero-shot z-conditioned actor variant to the `gciql_flow` agent and an eval script that measures success under exact-z and inferred-z, putting the IQL-trained bilinear representation into FB's full zero-shot pipeline.

**Architecture:** One default-off config flag (`actor_cond: 'goal'|'z'`) on the existing `GCIQLFlowAgent` (`tools/wandb_mode_shim/gciql_flow.py`). When `'z'`, the flow actor conditions on `z = stop_grad(value.psi(goal))` (B := the value net's psi, FB-faithful single backward map) instead of the raw goal observation; value/critic/vf are unchanged. A standalone JAX eval script (`scripts/eval_zcond_gciql.py`) rolls each trained checkpoint twice per task — exact z=psi(goal) and inferred z=E[r·psi(s')] — and reports success with the faithful cube-qpos reward.

**Tech Stack:** JAX/Flax (vendored OGBench `impls/`), the repo's jax venv `/dev/shm/.venv-jax`, hydra launch via `run_gciql.py`, tests via `tests/jax_subprocess.run_jax`.

**Spec:** `PAPER/specs/2026-06-16-g4-success-level-objective-isolation.md`

**Conventions reminder:** code/tests are committed; `PAPER/`, `analysis/`, `scripts/` outputs stay untracked per CLAUDE.md (but the new `scripts/*.py` and `tests/*.py` ARE code — commit them). Omit the Claude co-author trailer (memory `commit-no-claude-coauthor`). Jax tests run in a subprocess; `gciql_flow.py` and the eval script live under `tools/wandb_mode_shim/` and `scripts/` respectively. Work on the current branch `feat/psm-integration`.

---

## Task 1: Add the `actor_cond` config flag and actor example-goal wiring

Introduce the flag (default `'goal'` → byte-identical to current agent) and make `create()` build the actor module with the correct goal-input dimension (d-dim z when `'z'`, raw goal obs otherwise), plus the `value_arch='bilinear'` guard.

**Files:**
- Modify: `tools/wandb_mode_shim/gciql_flow.py` (`get_config` ~line 198-213; `create` ~line 116-195)
- Test: `tests/test_gciql_zcond.py` (create)

- [ ] **Step 1: Write the failing test — flag exists, default builds stock, z-mode builds with bilinear**

Create `tests/test_gciql_zcond.py`:

```python
"""Tests for gciql_flow actor_cond='z' (G4: zero-shot z-conditioned actor on the
IQL-value rep). Default actor_cond='goal' must reproduce the stock gciql_flow.
Runs in the jax venv subprocess. See PAPER/specs/2026-06-16-g4-*.md."""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def _full_batch(obs="ex_obs", act="ex_act"):
    return (
        f"dict(observations={obs}, next_observations={obs}, actions={act}, "
        f"rewards=jnp.zeros((2,)), masks=jnp.ones((2,)), "
        f"value_goals={obs}, actor_goals={obs})"
    )


def test_actor_cond_defaults_to_goal():
    code = """
        from gciql_flow import get_config
        cfg = get_config()
        assert cfg.actor_cond == 'goal', cfg.actor_cond
        print('OK')
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_zcond_requires_bilinear():
    code = """
        import jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config()); cfg['encoder'] = None; cfg['frame_stack'] = None
        cfg['actor_cond'] = 'z'; cfg['value_arch'] = 'mlp'
        ex_obs = jnp.zeros((2, 11)); ex_act = jnp.zeros((2, 5))
        try:
            GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        except AssertionError:
            print('OK')
        else:
            print('NO-RAISE')
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_zcond_builds_and_actor_takes_d_dim_z():
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config()); cfg['encoder'] = None; cfg['frame_stack'] = None
        cfg['actor_cond'] = 'z'; cfg['value_arch'] = 'bilinear'; cfg['value_latent_dim'] = 16
        ex_obs = jnp.zeros((2, 11)); ex_act = jnp.zeros((2, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        # actor now consumes a d-dim z as its 'goals' arg (not the 11-dim obs).
        z = jnp.zeros((2, 16))
        a = agent.sample_actions(ex_obs, z, seed=jax.random.PRNGKey(0))
        assert a.shape == (2, 5), a.shape
        assert jnp.all(jnp.abs(a) <= 1.0 + 1e-5)
        print('OK')
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_gciql_zcond.py -v`
Expected: FAIL — `test_actor_cond_defaults_to_goal` errors on `cfg.actor_cond` (attribute missing); the z-mode tests fail because there is no guard / actor still expects 11-dim goals.

- [ ] **Step 3: Add the `actor_cond` config default**

In `tools/wandb_mode_shim/gciql_flow.py`, `get_config`, inside the `with config.unlocked():` block (after the `value_latent_dim` line ~206), add:

```python
        # Actor conditioning identity (G4 objective-isolation arm):
        # 'goal' = raw goal obs (stock); 'z' = z=stop_grad(value.psi(goal)),
        # FB-style single backward map. 'z' requires value_arch='bilinear'.
        config.actor_cond = 'goal'
```

- [ ] **Step 4: Wire the actor example-goal dim + guard in `create`**

In `create`, just after `config['action_dim'] = action_dim` (~line 123), add:

```python
        # Actor goal-input example. In z-conditioning (G4) the actor consumes a
        # d-dim task vector z=psi(goal) (B := value net's psi), not the raw goal
        # observation, so its goal-leg params must be sized to value_latent_dim.
        if config.get('actor_cond', 'goal') == 'z':
            assert config.get('value_arch', 'mlp') == 'bilinear', \
                "actor_cond='z' requires value_arch='bilinear' (B := value net's psi)"
            ex_actor_goals = jnp.zeros((ex_observations.shape[0], config['value_latent_dim']))
        else:
            ex_actor_goals = ex_goals
```

Then change the `actor` entry of `network_info` (~line 183) from:

```python
            actor=(actor_def, (ex_observations, ex_goals, ex_noise)),
```

to:

```python
            actor=(actor_def, (ex_observations, ex_actor_goals, ex_noise)),
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_gciql_zcond.py -v`
Expected: PASS (all 3). NB `test_zcond_builds_and_actor_takes_d_dim_z` passes only once Task 2 is NOT yet required — `sample_actions` already forwards `goals` straight to the actor, so the d-dim z path works at build time. (If `sample_actions` errors on z, that is a real bug to fix here, not in Task 2.)

- [ ] **Step 6: Commit**

```bash
git add tools/wandb_mode_shim/gciql_flow.py tests/test_gciql_zcond.py
git commit -m "feat(gciql): actor_cond flag (goal|z) + d-dim z actor wiring for G4"
```

---

## Task 2: z-conditioned actor loss (B := value psi, stop-grad)

Make `actor_loss` feed `z = stop_grad(value.psi(goal))` to the actor when `actor_cond='z'`, while the critic Q-guidance term keeps taking the raw goal obs (it computes psi internally). Verify finiteness and that **no gradient leaks into the value net** from the actor loss.

**Files:**
- Modify: `tools/wandb_mode_shim/gciql_flow.py` (`actor_loss` ~line 39-80; add `_value_psi` helper)
- Test: `tests/test_gciql_zcond.py` (append)

- [ ] **Step 1: Write the failing tests — z-mode trains finite + stop-grad on value**

Append to `tests/test_gciql_zcond.py`:

```python
def test_zcond_trains_finite():
    code = f"""
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config()); cfg['encoder'] = None; cfg['frame_stack'] = None
        cfg['actor_cond'] = 'z'; cfg['value_arch'] = 'bilinear'; cfg['value_latent_dim'] = 16
        ex_obs = jnp.zeros((2, 11)); ex_act = jnp.zeros((2, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = {_full_batch()}
        for _ in range(5):
            agent, info = agent.update(batch)
        for k in ('actor/actor_loss', 'actor/q_mean', 'value/value_loss', 'critic/critic_loss'):
            assert jnp.isfinite(info[k]), (k, info[k])
        print('OK')
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_zcond_actor_loss_no_grad_into_value():
    code = f"""
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config()); cfg['encoder'] = None; cfg['frame_stack'] = None
        cfg['actor_cond'] = 'z'; cfg['value_arch'] = 'bilinear'; cfg['value_latent_dim'] = 16
        ex_obs = jnp.ones((2, 11)); ex_act = jnp.ones((2, 5)) * 0.3
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = {_full_batch()}
        def aloss(p):
            l, _ = agent.actor_loss(batch, p, rng=jax.random.PRNGKey(0))
            return l
        g = jax.grad(aloss)(agent.network.params)
        vnorm = sum(float(jnp.sum(x ** 2)) for x in jax.tree_util.tree_leaves(g['modules_value']))
        assert vnorm == 0.0, vnorm  # stop_grad keeps the rep purely IQL-trained
        print('OK')
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_gciql_zcond.py::test_zcond_actor_loss_no_grad_into_value tests/test_gciql_zcond.py::test_zcond_trains_finite -v`
Expected: FAIL — the actor still receives the raw 11-dim goal (`actor_goals`), which mismatches the d-dim actor goal-leg built in Task 1, so the forward pass shape-errors (and no value-psi path exists yet).

- [ ] **Step 3: Add the `_value_psi` helper**

In `tools/wandb_mode_shim/gciql_flow.py`, add this method to `GCIQLFlowAgent` (e.g. just above `actor_loss`):

```python
    def _value_psi(self, goals, params):
        """B := the value net's psi (action-free GCBilinearValue). psi depends
        only on the goal arg; the obs arg is a placeholder for phi(s)."""
        _v, _phi, psi = self.network.select('value')(goals, goals, info=True, params=params)
        return psi
```

- [ ] **Step 4: Branch the actor conditioning in `actor_loss`**

In `actor_loss`, the Q-guidance block currently reads (~line 57-59):

```python
        noise = jax.random.normal(n_rng, actions.shape)
        actor_actions = self.network.select('actor')(obs, goals, noise, params=grad_params)
        q1, q2 = self.network.select('critic')(obs, goals, actor_actions)
```

Replace with:

```python
        noise = jax.random.normal(n_rng, actions.shape)
        # z-conditioning (G4): the actor conditions on z=stop_grad(psi(goal))
        # (B := value net's psi). stop_grad keeps the representation purely
        # IQL-trained. The critic Q-term keeps the raw goal obs -- it computes
        # its own psi internally.
        if self.config['actor_cond'] == 'z':
            actor_cond_input = jax.lax.stop_gradient(self._value_psi(goals, grad_params))
        else:
            actor_cond_input = goals
        actor_actions = self.network.select('actor')(obs, actor_cond_input, noise, params=grad_params)
        q1, q2 = self.network.select('critic')(obs, goals, actor_actions)
```

(The `bc_flow_loss` and one-shot↔flow distillation are state-only and unchanged.)

- [ ] **Step 5: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_gciql_zcond.py -v`
Expected: PASS (all 5 tests in the file).

- [ ] **Step 6: Run the full gciql_flow suite to confirm no regression (default-off contract)**

Run: `.venv/bin/python -m pytest tests/test_gciql_flow_injection.py -v`
Expected: PASS (all 6) — `actor_cond` defaults to `'goal'`, so stock behavior is unchanged.

- [ ] **Step 7: Commit**

```bash
git add tools/wandb_mode_shim/gciql_flow.py tests/test_gciql_zcond.py
git commit -m "feat(gciql): z-conditioned actor loss (z=stop_grad(value.psi(goal)))"
```

---

## Task 3: Teach `load_agent` the `actor_cond` flag

The eval script restores a trained z-cond checkpoint; `phase_probe_gciql.load_agent` must build the template agent with `actor_cond='z'` so the param tree aligns. Default keeps existing callers (`phase_probe`, `infer_ablation`) unchanged.

**Files:**
- Modify: `scripts/phase_probe_gciql.py` (`load_agent` ~line 89-125)
- Test: `tests/test_gciql_zcond.py` (append)

- [ ] **Step 1: Write the failing test — load_agent accepts actor_cond**

Append to `tests/test_gciql_zcond.py`:

```python
def test_load_agent_accepts_actor_cond():
    # Signature-level guard: load_agent must expose an actor_cond param so the
    # eval script can rebuild a z-cond template before restore.
    code = """
        import inspect, sys
        from pathlib import Path
        REPO = Path.cwd().parent.parent  # OGBENCH_IMPLS -> repo root
        sys.path.insert(0, str(REPO))
        from scripts.phase_probe_gciql import load_agent
        params = inspect.signature(load_agent).parameters
        assert 'actor_cond' in params, list(params)
        assert params['actor_cond'].default == 'goal', params['actor_cond'].default
        print('OK')
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gciql_zcond.py::test_load_agent_accepts_actor_cond -v`
Expected: FAIL — `'actor_cond' not in params`.

- [ ] **Step 3: Add the parameter to `load_agent`**

In `scripts/phase_probe_gciql.py`, change the signature (~line 89):

```python
def load_agent(checkpoint_dir: Path, checkpoint_step: int, seed: int,
               value_arch: str = "bilinear", value_latent_dim: int = 512,
               actor_cond: str = "goal"):
```

and in its body, where the cfg keys are set (after `cfg["value_latent_dim"] = value_latent_dim`), add:

```python
    cfg["actor_cond"] = actor_cond
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gciql_zcond.py::test_load_agent_accepts_actor_cond -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/phase_probe_gciql.py tests/test_gciql_zcond.py
git commit -m "feat(gciql): load_agent actor_cond passthrough for z-cond eval"
```

---

## Task 4: Eval script `scripts/eval_zcond_gciql.py` (exact-z + inferred-z success)

Per task, compute z two ways and roll the z-cond actor, reporting success with the faithful cube-qpos reward. Reuses `phase_probe_gciql.load_agent`, the `cube_reward` (qpos `[14:17]`, thresh 0.04) and the rollout convention from `infer_ablation_gciql`.

**Files:**
- Create: `scripts/eval_zcond_gciql.py`
- Test: `tests/test_gciql_zcond.py` (append — import/smoke only; full eval is a manual run)

- [ ] **Step 1: Write the eval script**

Create `scripts/eval_zcond_gciql.py`:

```python
#!/usr/bin/env python
"""scripts/eval_zcond_gciql.py — G4 success-level objective isolation.

Rolls a trained z-conditioned GCIQL-Flow checkpoint (actor_cond='z',
value_arch='bilinear') two ways per task and reports SUCCESS:

  exact-z     : z = value.psi(goal_obs)          (G3-analog with z-conditioning)
  inferred-z  : z = sum_i r_i * value.psi(s'_i)  (FB's L0 rule, full FB pipeline)

The inference relabel batch is gathered by rolling the exact-z policy and scoring
states with the FAITHFUL goal-reaching reward from privileged cube qpos[14:17]
(dist<0.04 -- the same reward the FB evaluator and infer_ablation_gciql use; the
obs cube xy has unrecoverable scale/offset, so privileged qpos is required and
the offline dataset, which lacks qpos, cannot supply it). Same-policy state dist
caveat noted in the spec.

Run: JAX_PLATFORMS=cpu MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python \
       scripts/eval_zcond_gciql.py --checkpoint-dir <dir> --checkpoint-step 500000 \
       --seed 0 --value-latent-dim 512 --episodes 50 \
       --out analysis/eval_zcond_gciql/s0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "ogbench" / "impls"))
sys.path.insert(0, str(REPO / "tools" / "wandb_mode_shim"))

EVAL_TASKS = [f"cube-single-play-singletask-task{i}-v0" for i in range(1, 6)]
CUBE_QPOS = slice(14, 17)   # privileged cube xyz (matches infer_ablation_gciql)
THRESH = 0.04


def rollout(agent, env, z, goal_xyz, episodes, max_steps=200, seed0=0):
    """Roll the z-cond actor (z passed as the actor's goals arg). Return
    per-step next_obs, cube_xyz, and per-EPISODE success (cube within THRESH of
    goal at any step). z is a fixed d-vector for the task."""
    import jax
    z = np.asarray(z, np.float32)[None]              # (1, d) -> broadcast over the single obs
    rng = jax.random.PRNGKey(seed0)
    obss, cubes, succ = [], [], []
    for ep in range(episodes):
        obs, _info = env.reset(seed=seed0 + ep)
        ep_hit = False
        for _t in range(max_steps):
            rng, sub = jax.random.split(rng)
            a = np.asarray(agent.sample_actions(
                observations=np.asarray(obs, np.float32), goals=z[0],
                seed=sub, temperature=0.0))
            obs, _r, term, trunc, _info = env.step(np.clip(a, -1, 1))
            cube = env.unwrapped.data.qpos[CUBE_QPOS].copy()
            obss.append(np.asarray(obs, np.float32)); cubes.append(cube)
            if np.linalg.norm(cube - goal_xyz) < THRESH:
                ep_hit = True
            if term or trunc:
                break
        succ.append(float(ep_hit))
    return np.asarray(obss), np.asarray(cubes), float(np.mean(succ))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--value-latent-dim", type=int, default=512)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--infer-episodes", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import jax.numpy as jnp
    import ogbench
    from phase_probe_gciql import load_agent

    agent = load_agent(Path(args.checkpoint_dir).resolve(), args.checkpoint_step,
                       args.seed, value_arch="bilinear",
                       value_latent_dim=args.value_latent_dim, actor_cond="z")
    print(f"[eval_zcond] agent restored (d={args.value_latent_dim}, actor_cond=z)")

    def psi(goal):
        """B := value net's psi(goal). Returns (d,) for a single goal / (N,d)."""
        g = jnp.asarray(goal, jnp.float32)
        _v, _phi, ps = agent.network.select("value")(g, g, info=True)
        return np.asarray(ps)

    rows = {}
    for task in EVAL_TASKS:
        env = ogbench.make_env_and_datasets(task, env_only=True)
        _obs, info = env.reset(seed=0)
        goal_obs = np.asarray(info["goal"], np.float32)
        goal_xyz = np.asarray(env.unwrapped.cur_task_info["goal_xyzs"]).reshape(-1)[:3]

        # exact-z: z = psi(goal_obs)
        z_exact = psi(goal_obs[None])[0]
        nobs, _cube, succ_exact = rollout(agent, env, z_exact, goal_xyz, args.episodes)

        # inferred-z: relabel the exact-z rollout, z = sum r*psi(s')
        r = (np.linalg.norm(
            np.asarray([env.unwrapped.data.qpos[CUBE_QPOS] for _ in range(0)])  # placeholder
            , axis=-1) if False else None)
        # Gather a fresh relabel batch (states + privileged cube xyz) under exact-z.
        rb_obs, rb_cube, _ = rollout(agent, env, z_exact, goal_xyz, args.infer_episodes)
        rb_r = (np.linalg.norm(rb_cube - goal_xyz[None, :], axis=-1) < THRESH).astype(np.float32)
        z_inf = (rb_r[:, None] * psi(rb_obs)).sum(0)        # L0: z = E[r.B]
        _nobs2, _cube2, succ_inf = rollout(agent, env, z_inf, goal_xyz, args.episodes)

        if hasattr(env, "close"):
            env.close()
        rows[task] = {"success_exact": succ_exact, "success_inferred": succ_inf,
                      "r_pos_relabel": float(rb_r.mean())}
        print(f"  {task}: success exact={succ_exact:.3f}  inferred={succ_inf:.3f}  "
              f"(relabel r_pos={rb_r.mean():.3f})")

    summ = {
        "seed": args.seed, "value_latent_dim": args.value_latent_dim,
        "mean_success_exact": float(np.mean([v["success_exact"] for v in rows.values()])),
        "mean_success_inferred": float(np.mean([v["success_inferred"] for v in rows.values()])),
        "per_task": rows,
    }
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summ, indent=2))
    print(f"[eval_zcond] mean success  exact={summ['mean_success_exact']:.3f}  "
          f"inferred={summ['mean_success_inferred']:.3f}  -> {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

(Delete the inert `r = (... if False else None)` placeholder block — it is left out below; do not paste it. Use the version in Step 1b.)

- [ ] **Step 1b: Remove the placeholder line**

Delete these three lines from `eval_zcond_gciql.py` (they are a leftover and do nothing):

```python
        r = (np.linalg.norm(
            np.asarray([env.unwrapped.data.qpos[CUBE_QPOS] for _ in range(0)])  # placeholder
            , axis=-1) if False else None)
```

The inferred-z block then starts directly at `# Gather a fresh relabel batch ...`.

- [ ] **Step 2: Write the import/smoke test**

Append to `tests/test_gciql_zcond.py`:

```python
def test_eval_script_imports():
    # The eval script must import cleanly (path wiring + helpers resolve) without
    # a checkpoint. Full eval is a manual GPU/CPU run, not a unit test.
    code = """
        import importlib.util, sys
        from pathlib import Path
        REPO = Path.cwd().parent.parent
        path = REPO / 'scripts' / 'eval_zcond_gciql.py'
        spec = importlib.util.spec_from_file_location('eval_zcond_gciql', path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        assert hasattr(m, 'rollout') and hasattr(m, 'main')
        assert m.THRESH == 0.04 and m.CUBE_QPOS == slice(14, 17)
        print('OK')
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
```

- [ ] **Step 3: Run the import test**

Run: `.venv/bin/python -m pytest tests/test_gciql_zcond.py::test_eval_script_imports -v`
Expected: PASS.

- [ ] **Step 4: Run the whole new test file**

Run: `.venv/bin/python -m pytest tests/test_gciql_zcond.py -v`
Expected: PASS (all 7).

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_zcond_gciql.py tests/test_gciql_zcond.py
git commit -m "feat(gciql): eval_zcond_gciql success probe (exact-z + inferred-z)"
```

---

## Task 5: Hydra launch config + smoke + 3-seed run

Add a launch config mirroring the G3 bilinear bridge run with `actor_cond=z`, smoke-test the training path, then launch 3 seeds × 500k in tmux.

**Files:**
- Create: `configs/gciql/cube_single_state_gciql_flow_g4.yaml`

- [ ] **Step 1: Create the launch config**

Create `configs/gciql/cube_single_state_gciql_flow_g4.yaml`:

```yaml
# G4 (success-level objective isolation): GCIQL bilinear IQL value/critic + FB
# FlowBC actor conditioned on z=stop_grad(value.psi(goal)) (actor_cond=z), eval
# zero-shot. Mirrors the G3 bilinear bridge run (GCIQL-State-Flow-Bilinear) but
# swaps exact-goal actor conditioning for FB's z-conditioned actor. State => no
# encoder. See PAPER/specs/2026-06-16-g4-success-level-objective-isolation.md.
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
run_group: gciql_g4_zcond
restore_path: null
restore_epoch: null
exp_name: null
agent_overrides:
  bc_coeff: 3.0
  value_arch: bilinear
  value_latent_dim: 512
  actor_cond: z
wandb_mode: online
```

- [ ] **Step 2: Smoke the training path (short, 1 seed, tiny steps)**

Run (TORCH venv launches; it dispatches to the jax venv via storage=shm):

```bash
cd /home/adityamohan/git/Factored-FB && XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python run_gciql.py --config-name cube_single_state_gciql_flow_g4 \
  train_steps=2000 save_interval=2000 eval_interval=2000 eval_episodes=2 \
  log_interval=500 wandb_mode=disabled seed=0
```

Expected: completes without error; logs show finite `actor/*`, `value/*`, `critic/*`; a `params_2000.pkl` is written under the run dir. (If `run_gciql.py` rejects bare hydra dotted overrides, instead copy the config to a `*_smoke.yaml` with those values and `--config-name` it.)

- [ ] **Step 3: Verify the actor really is z-conditioned in the smoke checkpoint**

Run:

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python -c "
from pathlib import Path
import sys; sys.path.insert(0,'scripts')
from phase_probe_gciql import load_agent
# point at the smoke run dir written in Step 2:
import glob
d = sorted(glob.glob('/dev/shm/**/gciql_g4_zcond*/', recursive=True))[-1]
ag = load_agent(Path(d), 2000, 0, value_arch='bilinear', value_latent_dim=512, actor_cond='z')
ap = ag.network.params['modules_actor']
print('actor goal leg restored OK; param keys:', list(ap.keys())[:4])
"
```

Expected: restores without a param-tree mismatch (confirms d-dim z actor leg aligns with the saved checkpoint).

- [ ] **Step 4: Launch the 3-seed × 500k run in tmux**

Pick 3 free GPUs (check `nvidia-smi`); pack with `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Launch each seed in a detached tmux session (memory `always-run-in-tmux`), wandb `entity=amsks` (memory `wandb-entity-amsks` — verify the run logs to amsks, not a team default):

```bash
for s in 0 1 2; do
  tmux new-session -d -s g4_s$s "cd /home/adityamohan/git/Factored-FB && \
    CUDA_VISIBLE_DEVICES=$s XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python run_gciql.py --config-name cube_single_state_gciql_flow_g4 \
    seed=$s 2>&1 | tee /dev/shm/g4_zcond_s$s.log"
done
tmux ls | grep g4_
```

Expected: three `g4_s{0,1,2}` sessions running; logs accrue under `/dev/shm/g4_zcond_s*.log`; checkpoints land in the gciql `storage=shm` output tree.

- [ ] **Step 5: Commit the config; record the run in memory**

```bash
git add configs/gciql/cube_single_state_gciql_flow_g4.yaml
git commit -m "feat(gciql): G4 z-conditioned launch config (3x500k state)"
```

Then update the master handoff memory (`handoff-iwr-program.md`): add G4 to RUNNING with the tmux sessions, log paths, and the eval command for when checkpoints land. Add a `tmux-sessions-registry` entry for `g4_s{0,1,2}`.

- [ ] **Step 6 (deferred — after training): run the eval and fill the table**

When `params_500000.pkl` exists for all seeds, for each seed:

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python \
  scripts/eval_zcond_gciql.py --checkpoint-dir <run_dir_for_seed> \
  --checkpoint-step 500000 --seed <s> --value-latent-dim 512 --episodes 50 \
  --out analysis/eval_zcond_gciql/s<s>.json
```

Then average `mean_success_exact` / `mean_success_inferred` across seeds and fill the G4 rows of the spec's comparison table (and `PAPER/fb-iwr/fb-iwr.tex`).

---

## Self-Review

**Spec coverage:**
- `actor_cond` flag, default-off → Task 1. ✓
- B := value psi, stop-grad, z in actor loss → Task 2. ✓
- Critic keeps raw goal → Task 2 Step 4. ✓
- Flow vf unchanged → untouched (noted in Task 2 Step 4). ✓
- value_arch='bilinear' guard → Task 1 Step 4. ✓
- Eval: exact-z + inferred-z success, faithful cube reward, FB L0 rule → Task 4. ✓
- Tests: parity, z-build, finite, stop-grad, eval import → Tasks 1-4. ✓
- Runs: 3×500k, d=512, tmux, wandb amsks, group gciql_g4_zcond, save_interval≤train_steps → Task 5. ✓
- Table fill → Task 5 Step 6. ✓

**Placeholder scan:** the only literal placeholder is the inert `if False` block in the first paste of `eval_zcond_gciql.py`; Step 1b explicitly deletes it. `<run_dir_for_seed>`/`<s>` in Step 6 are genuine runtime values (the run dir is created at launch), not plan gaps.

**Type/name consistency:** `actor_cond` ('goal'|'z'), `value_arch='bilinear'`, `value_latent_dim`, `_value_psi(goals, params)`, `load_agent(..., actor_cond=)`, `rollout(...)`, `psi(goal)`, `THRESH`/`CUBE_QPOS` are used identically across tasks. `sample_actions(observations, goals, seed, temperature)` matches the existing agent signature; the eval passes z as `goals`. ✓
