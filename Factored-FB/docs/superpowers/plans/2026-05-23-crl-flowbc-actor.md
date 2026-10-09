# CRL + FlowBC actor (JAX) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement FB's FlowBC actor (velocity field + one-shot noise-conditioned actor) in JAX/Flax and wire it into OGBench's CRL agent (contrastive critic kept), runnable with the DrQ encoder, gated by two cross-framework parity checks before any deployment.

**Architecture:** Two pure Flax nets in `tools/wandb_mode_shim/ogbench_flow.py`, parity-tested against FB torch; a `CRLFlowBCAgent(CRLAgent)` in `tools/wandb_mode_shim/crl_flowbc.py` that reuses CRL's `contrastive_loss` and replaces the actor with FB's flow-matching + Q-max (CRL's `min(q1,q2)`) + BC-distillation; injection via `sitecustomize` registering `agents['crl_flowbc']` plus a repo-relative→absolute `agent_file` resolve in `run_gciql.py`; four Hydra configs. Never edits vendored OGBench.

**Tech Stack:** JAX/Flax (`/dev/shm/.venv-jax`), PyTorch (`.venv`), Hydra, pytest, optax.

---

## Spec reference

`docs/superpowers/specs/2026-05-22-crl-flowbc-actor-design.md` (committed).

## Critical context (verified against the codebase)

- **Two isolated venvs.** pytest runs under `.venv` (torch, no jax). JAX tests shell out to `/dev/shm/.venv-jax` via `tests/jax_subprocess.py` (`run_jax`, `jax_python`, `SHIM_DIR`, `OGBENCH_IMPLS`) and SKIP when it is absent. Cross-framework parity bridges the two via an `.npz`.
- **FB nets to mirror** (`nn_models.py`): `VectorField` (`:416-433`), `NoiseConditionedActor` (`:280-298`), `simple_embedding` (`:199-206` = `[Linear→LayerNorm→Tanh] + [Linear→ReLU]×(L-2) + [Linear(h//2)→ReLU]`), `linear`/`layernorm` (`:144`, plain `nn.Linear`/`nn.LayerNorm`). FB actor update + `compute_flow_actions`: `agents/fb/flow_bc/agent.py:62-125`.
- **Exactness traps:** torch `nn.GELU()` is the **erf** form → Flax must use `nn.gelu(x, approximate=False)` (Flax default is the tanh approx). torch `nn.LayerNorm` eps=1e-5 → Flax `nn.LayerNorm(epsilon=1e-5)` (Flax default 1e-6).
- **CRL** (`third_party/ogbench/impls/agents/crl.py`): `contrastive_loss` (reuse), `GCBilinearValue` critic returns ensemble `v` of shape `(2, B)` → `q1, q2 = critic(obs, goals, actions)`. `create`/`actor_loss`/`total_loss`/`sample_actions` are the surface we override. `config` is a `nonpytree_field` (static → Python `if` on config values works inside jit).
- **Agent registry:** `main.py:77` does `agents[config['agent_name']]`; `agents` dict (`agents/__init__.py`) is editable at runtime. `--agent` is a `config_flags` file providing `get_config()` (`main.py:42`). Same injection shape as DrQ's `encoder_modules`.
- **run_gciql** (`run_gciql.py:25-43`): `build_ogbench_argv` emits `--agent={cfg['agent_file']}`; `REPO_ROOT` defined at `:21`. Child runs with `cwd=impls`, shim dir first on PYTHONPATH (`:101-103`).
- **Repo-root `agents/` is FB's** (`agents/fb/...`) — `agents/crl.py` does **not** exist there, so the absolute-resolve rule leaves vendored `agents/*.py` untouched.
- **Hard constraint:** never edit `third_party/ogbench`. Run tests with `.venv/bin/python -m pytest`.

## File structure

| File | Create/Modify | Responsibility |
|------|---------------|----------------|
| `tools/wandb_mode_shim/ogbench_flow.py` | Create | Flax `VectorField`, `NoiseConditionedActor`, GC wrappers, `compute_flow_actions`. No OGBench imports. |
| `tools/wandb_mode_shim/crl_flowbc.py` | Create | `CRLFlowBCAgent(CRLAgent)` + `get_config()`. Imports OGBench utils + the flow nets. |
| `tools/wandb_mode_shim/sitecustomize.py` | Modify | Register `agents['crl_flowbc']` (guarded block). |
| `run_gciql.py` | Modify | Resolve repo-relative `agent_file` to absolute in `build_ogbench_argv`. |
| `configs/gciql/cube_single_state_crl_flowbc.yaml` | Create | CRL+FlowBC state. |
| `configs/gciql/cube_single_visual_crl_flowbc_drq.yaml` | Create | CRL+FlowBC pixel/DrQ. |
| `tests/test_ogbench_flow_nets.py` | Create | Shape/determinism (jax subprocess). |
| `tests/test_ogbench_flow_parity.py` | Create | **Gate 1** forward parity (<1e-5). |
| `tests/test_ogbench_flow_train_parity.py` | Create | **Gate 2** training-dynamics parity (<1e-6, float64). |
| `tests/test_crl_flowbc_injection.py` | Create | Agent registered + builds (jax subprocess). |
| `tests/test_run_gciql_agent_resolve.py` | Create | `agent_file` resolve rule. |
| `tests/test_crl_flowbc_configs.py` | Create | Config argv. |

---

## Task 1: Flow networks (Flax)

**Files:**
- Create: `tools/wandb_mode_shim/ogbench_flow.py`
- Test: `tests/test_ogbench_flow_nets.py`

- [ ] **Step 1: Write the failing test**

```python
from tests.jax_subprocess import run_jax


def test_flow_nets_shapes_and_determinism():
    code = """
        import jax, jax.numpy as jnp
        from ogbench_flow import VectorField, NoiseConditionedActor, compute_flow_actions
        B, obs_dim, z_dim, act = 4, 11, 7, 5
        obs = jnp.zeros((B, obs_dim)); a = jnp.zeros((B, act)); t = jnp.zeros((B, 1)); z = jnp.zeros((B, z_dim)); n = jnp.zeros((B, act))
        vf = VectorField(hidden_dim=512, hidden_layers=4)
        pv = vf.init(jax.random.PRNGKey(0), obs, a, t)["params"]
        ov = vf.apply({"params": pv}, obs, a, t)
        assert ov.shape == (B, act), ov.shape
        ac = NoiseConditionedActor(hidden_dim=512, hidden_layers=2, embedding_layers=2)
        pa = ac.init(jax.random.PRNGKey(1), obs, z, n)["params"]
        oa = ac.apply({"params": pa}, obs, z, n)
        assert oa.shape == (B, act), oa.shape
        assert float(jnp.max(jnp.abs(oa))) <= 1.0  # tanh output
        roll = compute_flow_actions(vf, pv, obs, n, flow_steps=10)
        assert roll.shape == (B, act)
        # determinism: same params+input -> same output
        oa2 = ac.apply({"params": pa}, obs, z, n)
        assert float(jnp.max(jnp.abs(oa - oa2))) == 0.0
        print("OK")
    """
    proc = run_jax(code)
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
```

- [ ] **Step 2: Run it — expect SKIP or FAIL (module missing)**

Run: `.venv/bin/python -m pytest tests/test_ogbench_flow_nets.py -v`
Expected: SKIP without `.venv-jax`; else FAIL `ModuleNotFoundError: ogbench_flow`.

- [ ] **Step 3: Implement `tools/wandb_mode_shim/ogbench_flow.py`**

```python
"""JAX/Flax FlowBC actor nets for vendored OGBench CRL, matching FB's PyTorch
VectorField (nn_models.py:416) and NoiseConditionedActor (nn_models.py:280).

Pure nets (no OGBench imports); the GC* wrappers hold optional encoders so the
agent can track encoder params via the OGBench ModuleDict. GELU is the exact
(erf) form and LayerNorm eps=1e-5 to match torch defaults. Imported by
crl_flowbc.py inside the OGBench child (jax env) and by the parity tests.
"""
import flax.linen as nn
import jax
import jax.numpy as jnp


class VectorField(nn.Module):
    """Unconditional (obs-only) velocity field: concat[obs, action, t] -> velocity.
    Mirrors FB VectorField: [Dense->GELU] + [Dense->GELU]*(L-1) + Dense(act)."""

    hidden_dim: int = 512
    hidden_layers: int = 4

    @nn.compact
    def __call__(self, obs, action, t):
        out_dim = action.shape[-1]
        x = jnp.concatenate([obs, action, t], axis=-1)
        x = nn.gelu(nn.Dense(self.hidden_dim, name="l0")(x), approximate=False)
        for i in range(self.hidden_layers - 1):
            x = nn.gelu(nn.Dense(self.hidden_dim, name=f"l{i + 1}")(x), approximate=False)
        return nn.Dense(out_dim, name="out")(x)


class NoiseConditionedActor(nn.Module):
    """One-shot actor: tanh(policy(concat[embed_s(obs,noise), embed_z(obs,z,noise)])).
    Mirrors FB NoiseConditionedActor + simple_embedding (embedding_layers>=2)."""

    hidden_dim: int = 512
    hidden_layers: int = 2
    embedding_layers: int = 2

    def _embed(self, x, prefix):
        h = self.hidden_dim
        x = nn.Dense(h, name=f"{prefix}_l0")(x)
        x = nn.LayerNorm(epsilon=1e-5, name=f"{prefix}_ln")(x)
        x = jnp.tanh(x)
        for i in range(self.embedding_layers - 2):
            x = nn.relu(nn.Dense(h, name=f"{prefix}_l{i + 1}")(x))
        return nn.relu(nn.Dense(h // 2, name=f"{prefix}_lout")(x))

    @nn.compact
    def __call__(self, obs, z, noise):
        out_dim = noise.shape[-1]
        z_emb = self._embed(jnp.concatenate([obs, z, noise], axis=-1), "embed_z")
        s_emb = self._embed(jnp.concatenate([obs, noise], axis=-1), "embed_s")
        x = jnp.concatenate([s_emb, z_emb], axis=-1)  # FB order: s then z
        for i in range(self.hidden_layers):
            x = nn.relu(nn.Dense(self.hidden_dim, name=f"policy_l{i}")(x))
        return jnp.tanh(nn.Dense(out_dim, name="policy_out")(x))


class GCVectorField(nn.Module):
    """VectorField with an optional obs encoder (for pixel obs)."""

    hidden_dim: int = 512
    hidden_layers: int = 4
    obs_encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations, actions, t):
        if self.obs_encoder is not None:
            observations = self.obs_encoder(observations)
        return VectorField(self.hidden_dim, self.hidden_layers)(observations, actions, t)


class GCNoiseActor(nn.Module):
    """NoiseConditionedActor with optional obs and goal encoders."""

    hidden_dim: int = 512
    hidden_layers: int = 2
    embedding_layers: int = 2
    obs_encoder: nn.Module = None
    goal_encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations, goals, noise):
        if self.obs_encoder is not None:
            observations = self.obs_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)
        return NoiseConditionedActor(
            self.hidden_dim, self.hidden_layers, self.embedding_layers
        )(observations, goals, noise)


def compute_flow_actions(vf, params, obs, noise, flow_steps):
    """Euler ODE rollout of `vf` (matches FB agent.py:118-125). `vf` is a
    VectorField module; `params` its param tree (already obs-encoded inputs)."""
    actions = noise
    for i in range(flow_steps):
        t = jnp.ones((noise.shape[0], 1)) * (i / flow_steps)
        vels = vf.apply({"params": params}, obs, actions, t)
        actions = actions + vels / flow_steps
    return jnp.clip(actions, -1.0, 1.0)
```

- [ ] **Step 4: Run the test — expect PASS (or SKIP)**

Run: `.venv/bin/python -m pytest tests/test_ogbench_flow_nets.py -v`
Expected: PASS, or SKIP without `.venv-jax`.

- [ ] **Step 5: Commit**

```bash
git add tools/wandb_mode_shim/ogbench_flow.py tests/test_ogbench_flow_nets.py
git commit -m "feat(crl-flowbc): Flax VectorField + NoiseConditionedActor + rollout"
```

---

## Task 2: Gate 1 — forward parity (weight transplant, <1e-5)

Proves the Flax nets equal FB's torch nets for the same weights/input, and that the JAX flow rollout equals an inline torch rollout. torch in-proc (`.venv`), jax subprocess (`.venv-jax`), npz bridge.

**Files:**
- Test: `tests/test_ogbench_flow_parity.py`

- [ ] **Step 1: Write the test**

```python
import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

import numpy as np

from tests.jax_subprocess import jax_python, SHIM_DIR, OGBENCH_IMPLS

B, OBS, Z, ACT, H = 4, 11, 7, 5, 512
FLOW_STEPS = 10


def _np(x):
    return x.detach().numpy()


def test_flow_nets_forward_parity_under_transplant():
    import gymnasium
    import torch
    from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig

    torch.manual_seed(0)
    obs_space = gymnasium.spaces.Box(-np.inf, np.inf, (OBS,), np.float32)
    vf = SimpleVectorFieldArchiConfig(hidden_dim=H, hidden_layers=4).build(obs_space, ACT).eval()
    ac = NoiseConditionedActorArchiConfig(hidden_dim=H, hidden_layers=2, embedding_layers=2).build(
        obs_space, Z, ACT
    ).eval()

    obs = torch.randn(B, OBS)
    action = torch.randn(B, ACT)
    t = torch.rand(B, 1)
    z = torch.randn(B, Z)
    noise = torch.randn(B, ACT)
    with torch.no_grad():
        out_vf = _np(vf(obs, action, t))
        out_ac = _np(ac(obs, z, noise))
        # inline torch rollout == FB compute_flow_actions
        roll = noise.clone()
        for i in range(FLOW_STEPS):
            tt = torch.ones(B, 1) * i / FLOW_STEPS
            roll = roll + vf(obs, roll, tt) / FLOW_STEPS
        out_roll = _np(torch.clamp(roll, -1, 1))

    vfsd = vf.net  # Sequential; Linears at 0,2,4,6,8
    acsd = ac
    arrs = dict(
        obs=_np(obs), action=_np(action), t=_np(t), z=_np(z), noise=_np(noise),
        out_vf=out_vf, out_ac=out_ac, out_roll=out_roll,
        # VectorField linears
        vf_l0_w=_np(vfsd[0].weight), vf_l0_b=_np(vfsd[0].bias),
        vf_l1_w=_np(vfsd[2].weight), vf_l1_b=_np(vfsd[2].bias),
        vf_l2_w=_np(vfsd[4].weight), vf_l2_b=_np(vfsd[4].bias),
        vf_l3_w=_np(vfsd[6].weight), vf_l3_b=_np(vfsd[6].bias),
        vf_out_w=_np(vfsd[8].weight), vf_out_b=_np(vfsd[8].bias),
        # embed_z (Linear@0, LN@1, Linear@3)
        ez_l0_w=_np(acsd.embed_z[0].weight), ez_l0_b=_np(acsd.embed_z[0].bias),
        ez_ln_w=_np(acsd.embed_z[1].weight), ez_ln_b=_np(acsd.embed_z[1].bias),
        ez_lout_w=_np(acsd.embed_z[3].weight), ez_lout_b=_np(acsd.embed_z[3].bias),
        # embed_s
        es_l0_w=_np(acsd.embed_s[0].weight), es_l0_b=_np(acsd.embed_s[0].bias),
        es_ln_w=_np(acsd.embed_s[1].weight), es_ln_b=_np(acsd.embed_s[1].bias),
        es_lout_w=_np(acsd.embed_s[3].weight), es_lout_b=_np(acsd.embed_s[3].bias),
        # policy (Linear@0,2,4)
        p_l0_w=_np(acsd.policy[0].weight), p_l0_b=_np(acsd.policy[0].bias),
        p_l1_w=_np(acsd.policy[2].weight), p_l1_b=_np(acsd.policy[2].bias),
        p_out_w=_np(acsd.policy[4].weight), p_out_b=_np(acsd.policy[4].bias),
    )

    with tempfile.TemporaryDirectory() as d:
        npz = Path(d) / "flow_parity.npz"
        np.savez(npz, **arrs)
        jax_code = textwrap.dedent(
            """
            import sys, numpy as np, jax, jax.numpy as jnp
            from ogbench_flow import VectorField, NoiseConditionedActor, compute_flow_actions
            d = np.load(sys.argv[1])
            def T(w):  # torch Linear [out,in] -> flax Dense [in,out]
                return jnp.asarray(w).T
            obs, action, t = jnp.asarray(d["obs"]), jnp.asarray(d["action"]), jnp.asarray(d["t"])
            z, noise = jnp.asarray(d["z"]), jnp.asarray(d["noise"])
            vf = VectorField(hidden_dim=512, hidden_layers=4)
            vfp = {
                "l0": {"kernel": T(d["vf_l0_w"]), "bias": jnp.asarray(d["vf_l0_b"])},
                "l1": {"kernel": T(d["vf_l1_w"]), "bias": jnp.asarray(d["vf_l1_b"])},
                "l2": {"kernel": T(d["vf_l2_w"]), "bias": jnp.asarray(d["vf_l2_b"])},
                "l3": {"kernel": T(d["vf_l3_w"]), "bias": jnp.asarray(d["vf_l3_b"])},
                "out": {"kernel": T(d["vf_out_w"]), "bias": jnp.asarray(d["vf_out_b"])},
            }
            ac = NoiseConditionedActor(hidden_dim=512, hidden_layers=2, embedding_layers=2)
            acp = {
                "embed_z_l0": {"kernel": T(d["ez_l0_w"]), "bias": jnp.asarray(d["ez_l0_b"])},
                "embed_z_ln": {"scale": jnp.asarray(d["ez_ln_w"]), "bias": jnp.asarray(d["ez_ln_b"])},
                "embed_z_lout": {"kernel": T(d["ez_lout_w"]), "bias": jnp.asarray(d["ez_lout_b"])},
                "embed_s_l0": {"kernel": T(d["es_l0_w"]), "bias": jnp.asarray(d["es_l0_b"])},
                "embed_s_ln": {"scale": jnp.asarray(d["es_ln_w"]), "bias": jnp.asarray(d["es_ln_b"])},
                "embed_s_lout": {"kernel": T(d["es_lout_w"]), "bias": jnp.asarray(d["es_lout_b"])},
                "policy_l0": {"kernel": T(d["p_l0_w"]), "bias": jnp.asarray(d["p_l0_b"])},
                "policy_l1": {"kernel": T(d["p_l1_w"]), "bias": jnp.asarray(d["p_l1_b"])},
                "policy_out": {"kernel": T(d["p_out_w"]), "bias": jnp.asarray(d["p_out_b"])},
            }
            o_vf = np.asarray(vf.apply({"params": vfp}, obs, action, t))
            o_ac = np.asarray(ac.apply({"params": acp}, obs, z, noise))
            o_roll = np.asarray(compute_flow_actions(vf, vfp, obs, noise, 10))
            diffs = {
                "vf": float(np.max(np.abs(o_vf - d["out_vf"]))),
                "ac": float(np.max(np.abs(o_ac - d["out_ac"]))),
                "roll": float(np.max(np.abs(o_roll - d["out_roll"]))),
            }
            print("DIFFS", diffs)
            sys.exit(0 if max(diffs.values()) < 1e-5 else 1)
            """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(SHIM_DIR), str(OGBENCH_IMPLS), env.get("PYTHONPATH", "")])
        env["JAX_PLATFORMS"] = "cpu"
        proc = subprocess.run([jax_python(), "-c", jax_code, str(npz)], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, f"parity failed\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "DIFFS" in proc.stdout
```

- [ ] **Step 2: Run it — expect PASS (or SKIP)**

Run: `.venv/bin/python -m pytest tests/test_ogbench_flow_parity.py -v -s`
Expected: PASS; `DIFFS {...}` printed, all `< 1e-5`. If it FAILS, the printed diffs localize which net/path drifts — check, in order: `nn.gelu(approximate=False)`, LayerNorm `epsilon=1e-5`, Linear transpose `.T`, the `concat[s_emb, z_emb]` order, and the embed/policy index→name mapping.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ogbench_flow_parity.py
git commit -m "test(crl-flowbc): Gate 1 forward parity (jax == fb torch < 1e-5)"
```

---

## Task 3: Gate 2 — training-dynamics parity (<1e-6, float64)

Trains each net on identical pre-generated data, identical transplanted init weights, identical Adam, in float64, in both frameworks, and asserts per-step loss curves + final outputs + final weights match `< 1e-6`. Velocity field trains on flow-matching L2; the one-shot actor trains on supervised regression toward a fixed target (the supervised core of BC-distillation; the rollout itself is already parity-checked in Gate 1).

**Files:**
- Test: `tests/test_ogbench_flow_train_parity.py`

- [ ] **Step 1: Write the test**

```python
import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

import numpy as np

from tests.jax_subprocess import jax_python, SHIM_DIR, OGBENCH_IMPLS

B, OBS, Z, ACT, H, N, LR = 4, 11, 7, 5, 512, 100, 3e-4


def test_flow_training_dynamics_parity_float64():
    import gymnasium
    import torch
    from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig

    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(0)
    obs_space = gymnasium.spaces.Box(-np.inf, np.inf, (OBS,), np.float32)
    vf = SimpleVectorFieldArchiConfig(hidden_dim=H, hidden_layers=4).build(obs_space, ACT).double()
    ac = NoiseConditionedActorArchiConfig(hidden_dim=H, hidden_layers=2, embedding_layers=2).build(
        obs_space, Z, ACT
    ).double()

    # Pre-generated, framework-shared data for N steps.
    obs = rng.standard_normal((N, B, OBS))
    action = rng.standard_normal((N, B, ACT))     # x_1 for vf; target for actor uses its own
    x0 = rng.standard_normal((N, B, ACT))
    tt = rng.uniform(size=(N, B, 1))
    z = rng.standard_normal((N, B, Z))
    noise = rng.standard_normal((N, B, ACT))
    actor_target = np.tanh(rng.standard_normal((N, B, ACT)))  # fixed supervised targets in (-1,1)
    eval_obs = rng.standard_normal((B, OBS)); eval_a = rng.standard_normal((B, ACT))
    eval_t = rng.uniform(size=(B, 1)); eval_z = rng.standard_normal((B, Z)); eval_n = rng.standard_normal((B, ACT))

    def torch_init(mod):
        return {k: v.detach().numpy().copy() for k, v in mod.state_dict().items()}

    init_vf, init_ac = torch_init(vf), torch_init(ac)

    opt_vf = torch.optim.Adam(vf.parameters(), lr=LR)
    losses_vf = []
    for i in range(N):
        o = torch.tensor(obs[i]); x1 = torch.tensor(action[i]); xt = (1 - torch.tensor(tt[i])) * torch.tensor(x0[i]) + torch.tensor(tt[i]) * x1
        vel = x1 - torch.tensor(x0[i])
        loss = ((vf(o, xt, torch.tensor(tt[i])) - vel) ** 2).mean()
        opt_vf.zero_grad(); loss.backward(); opt_vf.step()
        losses_vf.append(float(loss.detach()))

    opt_ac = torch.optim.Adam(ac.parameters(), lr=LR)
    losses_ac = []
    for i in range(N):
        o = torch.tensor(obs[i]); zz = torch.tensor(z[i]); nn_ = torch.tensor(noise[i]); tgt = torch.tensor(actor_target[i])
        loss = ((ac(o, zz, nn_) - tgt) ** 2).mean()
        opt_ac.zero_grad(); loss.backward(); opt_ac.step()
        losses_ac.append(float(loss.detach()))

    with torch.no_grad():
        fout_vf = vf(torch.tensor(eval_obs), torch.tensor(eval_a), torch.tensor(eval_t)).numpy()
        fout_ac = ac(torch.tensor(eval_obs), torch.tensor(eval_z), torch.tensor(eval_n)).numpy()
    torch.set_default_dtype(torch.float32)

    with tempfile.TemporaryDirectory() as d:
        npz = Path(d) / "train_parity.npz"
        np.savez(
            npz, obs=obs, action=action, x0=x0, tt=tt, z=z, noise=noise, actor_target=actor_target,
            eval_obs=eval_obs, eval_a=eval_a, eval_t=eval_t, eval_z=eval_z, eval_n=eval_n,
            losses_vf=np.array(losses_vf), losses_ac=np.array(losses_ac),
            fout_vf=fout_vf, fout_ac=fout_ac,
            **{f"ivf__{k}": v for k, v in init_vf.items()},
            **{f"iac__{k}": v for k, v in init_ac.items()},
        )
        jax_code = textwrap.dedent(
            """
            import sys, numpy as np, jax, optax
            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp
            from ogbench_flow import VectorField, NoiseConditionedActor
            d = np.load(sys.argv[1])
            def T(w): return jnp.asarray(w, dtype=jnp.float64).T
            def V(w): return jnp.asarray(w, dtype=jnp.float64)
            # init params transplanted from torch (keys: <module>.<idx>.weight/bias)
            g = lambda p: d[p]
            vfp = {
                "l0": {"kernel": T(g("ivf__net.0.weight")), "bias": V(g("ivf__net.0.bias"))},
                "l1": {"kernel": T(g("ivf__net.2.weight")), "bias": V(g("ivf__net.2.bias"))},
                "l2": {"kernel": T(g("ivf__net.4.weight")), "bias": V(g("ivf__net.4.bias"))},
                "l3": {"kernel": T(g("ivf__net.6.weight")), "bias": V(g("ivf__net.6.bias"))},
                "out": {"kernel": T(g("ivf__net.8.weight")), "bias": V(g("ivf__net.8.bias"))},
            }
            acp = {
                "embed_z_l0": {"kernel": T(g("iac__embed_z.0.weight")), "bias": V(g("iac__embed_z.0.bias"))},
                "embed_z_ln": {"scale": V(g("iac__embed_z.1.weight")), "bias": V(g("iac__embed_z.1.bias"))},
                "embed_z_lout": {"kernel": T(g("iac__embed_z.3.weight")), "bias": V(g("iac__embed_z.3.bias"))},
                "embed_s_l0": {"kernel": T(g("iac__embed_s.0.weight")), "bias": V(g("iac__embed_s.0.bias"))},
                "embed_s_ln": {"scale": V(g("iac__embed_s.1.weight")), "bias": V(g("iac__embed_s.1.bias"))},
                "embed_s_lout": {"kernel": T(g("iac__embed_s.3.weight")), "bias": V(g("iac__embed_s.3.bias"))},
                "policy_l0": {"kernel": T(g("iac__policy.0.weight")), "bias": V(g("iac__policy.0.bias"))},
                "policy_l1": {"kernel": T(g("iac__policy.2.weight")), "bias": V(g("iac__policy.2.bias"))},
                "policy_out": {"kernel": T(g("iac__policy.4.weight")), "bias": V(g("iac__policy.4.bias"))},
            }
            vf = VectorField(hidden_dim=512, hidden_layers=4)
            ac = NoiseConditionedActor(hidden_dim=512, hidden_layers=2, embedding_layers=2)
            N = d["losses_vf"].shape[0]
            obs, action, x0, tt = V(d["obs"]), V(d["action"]), V(d["x0"]), V(d["tt"])
            z, noise, tgt = V(d["z"]), V(d["noise"]), V(d["actor_target"])

            def run(model, params, loss_at):
                opt = optax.adam(3e-4); state = opt.init(params); curve = []
                for i in range(N):
                    l, grads = jax.value_and_grad(loss_at)(params, i)
                    upd, state = opt.update(grads, state, params)
                    params = optax.apply_updates(params, upd); curve.append(float(l))
                return params, np.array(curve)

            def vf_loss(p, i):
                xt = (1 - tt[i]) * x0[i] + tt[i] * action[i]; vel = action[i] - x0[i]
                return jnp.mean((vf.apply({"params": p}, obs[i], xt, tt[i]) - vel) ** 2)
            def ac_loss(p, i):
                return jnp.mean((ac.apply({"params": p}, obs[i], z[i], noise[i]) - tgt[i]) ** 2)

            vfp2, cv = run(vf, vfp, vf_loss)
            acp2, ca = run(ac, acp, ac_loss)
            fvf = np.asarray(vf.apply({"params": vfp2}, V(d["eval_obs"]), V(d["eval_a"]), V(d["eval_t"])))
            fac = np.asarray(ac.apply({"params": acp2}, V(d["eval_obs"]), V(d["eval_z"]), V(d["eval_n"])))
            diffs = {
                "curve_vf": float(np.max(np.abs(cv - d["losses_vf"]))),
                "curve_ac": float(np.max(np.abs(ca - d["losses_ac"]))),
                "fout_vf": float(np.max(np.abs(fvf - d["fout_vf"]))),
                "fout_ac": float(np.max(np.abs(fac - d["fout_ac"]))),
            }
            print("DIFFS", diffs)
            sys.exit(0 if max(diffs.values()) < 1e-6 else 1)
            """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(SHIM_DIR), str(OGBENCH_IMPLS), env.get("PYTHONPATH", "")])
        env["JAX_PLATFORMS"] = "cpu"
        proc = subprocess.run([jax_python(), "-c", jax_code, str(npz)], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, f"train parity failed\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "DIFFS" in proc.stdout
```

- [ ] **Step 2: Run it — expect PASS (or SKIP)**

Run: `.venv/bin/python -m pytest tests/test_ogbench_flow_train_parity.py -v -s`
Expected: PASS; `DIFFS {...}` all `< 1e-6`. If only the curve diffs exceed `1e-6` while step-0 matches Gate 1, the divergence is optimizer-impl (optax vs torch Adam): re-run mentally with `optax.sgd(3e-4)` / `torch.optim.SGD` to confirm the nets+grads match, then investigate the Adam delta (eps/bias-correction) separately. If step-0 already differs, it is a forward-parity regression — fix Task 2 first.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ogbench_flow_train_parity.py
git commit -m "test(crl-flowbc): Gate 2 training-dynamics parity (<1e-6, float64)"
```

---

## Task 4: CRLFlowBCAgent

**Files:**
- Create: `tools/wandb_mode_shim/crl_flowbc.py`
- Test: `tests/test_crl_flowbc_injection.py` (Step 1 here; injection wiring in Task 6)

- [ ] **Step 1: Write a failing build test**

Create `tests/test_crl_flowbc_injection.py`:

```python
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def test_crl_flowbc_agent_builds_state():
    code = """
        import jax, jax.numpy as jnp
        from crl_flowbc import CRLFlowBCAgent, get_config
        cfg = get_config()
        cfg = dict(cfg); cfg["encoder"] = None; cfg["frame_stack"] = None
        ex_obs = jnp.zeros((2, 11)); ex_act = jnp.zeros((2, 5))
        agent = CRLFlowBCAgent.create(0, ex_obs, ex_act, cfg)
        a = agent.sample_actions(ex_obs, ex_obs, seed=jax.random.PRNGKey(0))
        assert a.shape == (2, 5), a.shape
        batch = dict(observations=ex_obs, actions=ex_act, value_goals=ex_obs, actor_goals=ex_obs)
        agent2, info = agent.update(batch)
        assert "actor/bc_flow_loss" in info, list(info)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
```

- [ ] **Step 2: Run it — expect SKIP or FAIL (module missing)**

Run: `.venv/bin/python -m pytest tests/test_crl_flowbc_injection.py::test_crl_flowbc_agent_builds_state -v`
Expected: SKIP without `.venv-jax`; else FAIL `ModuleNotFoundError: crl_flowbc`.

- [ ] **Step 3: Implement `tools/wandb_mode_shim/crl_flowbc.py`**

```python
"""CRL + FB FlowBC actor (JAX). Subclasses vendored CRLAgent: keeps the
contrastive critic, replaces the actor with FB's flow-matching velocity field
+ one-shot noise-conditioned actor (Q-guidance from CRL's min(q1,q2)).
Registered into agents.agents['crl_flowbc'] by sitecustomize. Never edits OGBench.
"""
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from agents.crl import CRLAgent, get_config as _crl_get_config
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import GCBilinearValue

from ogbench_flow import GCNoiseActor, GCVectorField, NoiseConditionedActor


class CRLFlowBCAgent(CRLAgent):
    """CRL with FB's FlowBC actor. Inherits contrastive_loss + update."""

    def _flow_rollout(self, obs, noise, params):
        fs = self.config['flow_steps']
        actions = noise
        for i in range(fs):
            t = jnp.ones((noise.shape[0], 1)) * (i / fs)
            vels = self.network.select('actor_vf')(obs, actions, t, params=params)
            actions = actions + vels / fs
        return jnp.clip(actions, -1.0, 1.0)

    def actor_loss(self, batch, grad_params, rng=None):
        obs, goals, actions = batch['observations'], batch['actor_goals'], batch['actions']
        rng = rng if rng is not None else self.rng
        rng, x0_rng, t_rng, n_rng = jax.random.split(rng, 4)

        # 1) flow-matching loss trains actor_vf
        x_1 = actions
        x_0 = jax.random.normal(x0_rng, actions.shape)
        t = jax.random.uniform(t_rng, (actions.shape[0], 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.network.select('actor_vf')(obs, x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # 2) Q-max on one-shot actor's actions (CRL contrastive Q)
        noise = jax.random.normal(n_rng, actions.shape)
        actor_actions = self.network.select('actor')(obs, goals, noise, params=grad_params)
        q1, q2 = self.network.select('critic')(obs, goals, actor_actions, params=grad_params)
        q = jnp.minimum(q1, q2)
        actor_loss = -q.mean()

        # 3) BC-distillation toward frozen flow rollout (FB normalization)
        bc_loss = 0.0
        bc_error = 0.0
        if self.config['bc_coeff'] > 0:
            target = self._flow_rollout(obs, noise, jax.lax.stop_gradient(grad_params))
            bc_error = jnp.mean((actor_actions - target) ** 2)
            bc_loss = self.config['bc_coeff'] * bc_error
            actor_loss = actor_loss / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6) + bc_loss
        actor_loss = actor_loss + bc_flow_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'bc_error': bc_error,
            'q_mean': q.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        critic_loss, critic_info = self.contrastive_loss(batch, grad_params, 'critic')
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v
        return critic_loss + actor_loss, info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        del temperature  # flow actor is noise-driven
        noise = jax.random.normal(seed, (observations.shape[0], self.config['action_dim']))
        actions = self.network.select('actor')(observations, goals, noise)
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]
        config = dict(config)
        config['action_dim'] = action_dim

        encoders = {}
        if config['encoder'] is not None:
            em = encoder_modules[config['encoder']]
            encoders['critic_state'] = em()
            encoders['critic_goal'] = em()
            encoders['actor_obs'] = em()
            encoders['actor_goal'] = em()

        critic_def = GCBilinearValue(
            hidden_dims=config['value_hidden_dims'],
            latent_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            value_exp=False,
            state_encoder=encoders.get('critic_state'),
            goal_encoder=encoders.get('critic_goal'),
        )
        actor_vf_def = GCVectorField(
            hidden_dim=config['actor_vf_hidden_dim'],
            hidden_layers=config['actor_vf_hidden_layers'],
            obs_encoder=encoders.get('actor_obs'),
        )
        actor_def = GCNoiseActor(
            hidden_dim=config['actor_hidden_dim'],
            hidden_layers=config['actor_hidden_layers'],
            embedding_layers=config['actor_embedding_layers'],
            obs_encoder=encoders.get('actor_obs'),
            goal_encoder=encoders.get('actor_goal'),
        )

        ex_noise = jnp.zeros_like(ex_actions)
        ex_t = jnp.zeros((ex_actions.shape[0], 1))
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals, ex_noise)),
            actor_vf=(actor_vf_def, (ex_observations, ex_actions, ex_t)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr_actor_vf'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = _crl_get_config()
    with config.unlocked():
        config.agent_name = 'crl_flowbc'
        config.flow_steps = 10
        config.bc_coeff = 3.0
        config.lr_actor_vf = 3e-4
        config.actor_vf_hidden_dim = 512
        config.actor_vf_hidden_layers = 4
        config.actor_hidden_dim = 512
        config.actor_hidden_layers = 2
        config.actor_embedding_layers = 2
    return config
```

- [ ] **Step 4: Run the build test — expect PASS (or SKIP)**

Run: `.venv/bin/python -m pytest tests/test_crl_flowbc_injection.py::test_crl_flowbc_agent_builds_state -v`
Expected: PASS (or SKIP). If it fails on `lr` vs `lr_actor_vf` or a missing config key, confirm `get_config()` set every key `create` reads.

- [ ] **Step 5: Commit**

```bash
git add tools/wandb_mode_shim/crl_flowbc.py tests/test_crl_flowbc_injection.py
git commit -m "feat(crl-flowbc): CRLFlowBCAgent (contrastive critic + FB flow actor)"
```

---

## Task 5: run_gciql resolves repo-relative agent_file

**Files:**
- Modify: `run_gciql.py` (`build_ogbench_argv`, `:25-43`)
- Test: `tests/test_run_gciql_agent_resolve.py`

- [ ] **Step 1: Write the failing test**

```python
import os

from run_gciql import build_ogbench_argv

_BASE = dict(
    env_name="cube-single-play-v0", seed=0, train_steps=10, log_interval=5,
    eval_interval=5, save_interval=5, eval_episodes=2, eval_on_cpu=1,
    run_group="g", output_root="/tmp/out", agent_overrides={},
)


def test_repo_relative_agent_file_made_absolute():
    cfg = {**_BASE, "agent_file": "tools/wandb_mode_shim/crl_flowbc.py"}
    argv = build_ogbench_argv(cfg)
    flag = [a for a in argv if a.startswith("--agent=")][0]
    path = flag.split("=", 1)[1]
    assert os.path.isabs(path)
    assert path.endswith("tools/wandb_mode_shim/crl_flowbc.py")
    assert os.path.isfile(path)


def test_vendored_agent_file_passthrough():
    cfg = {**_BASE, "agent_file": "agents/crl.py"}
    argv = build_ogbench_argv(cfg)
    assert "--agent=agents/crl.py" in argv
```

- [ ] **Step 2: Run it — expect FAIL (still relative)**

Run: `.venv/bin/python -m pytest tests/test_run_gciql_agent_resolve.py -v`
Expected: `test_repo_relative_agent_file_made_absolute` FAILS (path not absolute).

- [ ] **Step 3: Edit `build_ogbench_argv`** — change the `--agent` line in `run_gciql.py:25-43` from:

```python
        f"--agent={cfg['agent_file']}",
```

to:

```python
        f"--agent={_resolve_agent_file(cfg['agent_file'])}",
```

and add this helper directly above `build_ogbench_argv` (uses the existing module-level `REPO_ROOT`):

```python
def _resolve_agent_file(agent_file: str) -> str:
    """Make a repo-relative agent file absolute (so custom agents outside the
    vendored tree work as --agent from cwd=impls). Vendored 'agents/*.py' do not
    exist under REPO_ROOT, so they pass through unchanged."""
    candidate = REPO_ROOT / str(agent_file)
    return str(candidate.resolve()) if candidate.is_file() else str(agent_file)
```

- [ ] **Step 4: Run the test — expect PASS**

Run: `.venv/bin/python -m pytest tests/test_run_gciql_agent_resolve.py -v`
Expected: both PASS.

- [ ] **Step 5: Regression — existing runner tests**

Run: `.venv/bin/python -m pytest tests/test_run_gciql_drq_env.py tests/test_run_gciql_visual.py -v`
Expected: all PASS (resolution does not affect env building or vendored paths).

- [ ] **Step 6: Commit**

```bash
git add run_gciql.py tests/test_run_gciql_agent_resolve.py
git commit -m "feat(crl-flowbc): resolve repo-relative agent_file to absolute"
```

---

## Task 6: Register the agent in sitecustomize

**Files:**
- Modify: `tools/wandb_mode_shim/sitecustomize.py` (append a new guarded block, after the DrQ block)
- Test: `tests/test_crl_flowbc_injection.py` (add registration test)

- [ ] **Step 1: Add the failing registration test** — append to `tests/test_crl_flowbc_injection.py`:

```python
def test_crl_flowbc_registered_in_agents():
    code = """
        import agents
        assert 'crl_flowbc' in agents.agents, list(agents.agents)
        from crl_flowbc import CRLFlowBCAgent
        assert agents.agents['crl_flowbc'] is CRLFlowBCAgent
        print('OK')
    """
    from tests.jax_subprocess import run_jax, OGBENCH_IMPLS
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
```

- [ ] **Step 2: Run it — expect SKIP or FAIL (not registered)**

Run: `.venv/bin/python -m pytest tests/test_crl_flowbc_injection.py::test_crl_flowbc_registered_in_agents -v`
Expected: SKIP without `.venv-jax`; else FAIL (`crl_flowbc` not in `agents.agents`).

- [ ] **Step 3: Append the registration block to `tools/wandb_mode_shim/sitecustomize.py`** (at end of file, after the DrQ block):

```python
# --- CRL+FlowBC agent injection (independent of the blocks above) ---
# Register CRLFlowBCAgent so configs may select agent_name='crl_flowbc' via
# --agent=tools/wandb_mode_shim/crl_flowbc.py. Guarded: inert in non-jax
# processes (e.g. .venv tests) where these imports fail.
try:
    import agents as _ogb_agents
    from crl_flowbc import CRLFlowBCAgent as _CRLFlowBCAgent

    _ogb_agents.agents.setdefault("crl_flowbc", _CRLFlowBCAgent)
except Exception:
    pass
```

- [ ] **Step 4: Run injection tests + wandb-shim regression**

Run: `.venv/bin/python -m pytest tests/test_crl_flowbc_injection.py tests/test_wandb_mode_shim.py -v`
Expected: injection tests PASS (or SKIP without `.venv-jax`); both `test_wandb_mode_shim` tests still PASS (new block is guarded/inert under `.venv`).

- [ ] **Step 5: Commit**

```bash
git add tools/wandb_mode_shim/sitecustomize.py tests/test_crl_flowbc_injection.py
git commit -m "feat(crl-flowbc): register crl_flowbc agent in shim"
```

---

## Task 7: Hydra configs for CRL+FlowBC

**Files:**
- Create: `configs/gciql/cube_single_state_crl_flowbc.yaml`
- Create: `configs/gciql/cube_single_visual_crl_flowbc_drq.yaml`
- Test: `tests/test_crl_flowbc_configs.py`

- [ ] **Step 1: Write the failing test**

```python
from hydra import compose, initialize

from run_gciql import build_ogbench_argv


def _argv(name):
    with initialize(version_base="1.3", config_path="../configs/gciql"):
        cfg = compose(config_name=name)
    cfg_d = dict(cfg)
    cfg_d["output_root"] = "/tmp/out"
    return build_ogbench_argv(cfg_d)


def test_crl_flowbc_state_config_argv():
    argv = _argv("cube_single_state_crl_flowbc")
    assert any(a.startswith("--agent=") and a.endswith("tools/wandb_mode_shim/crl_flowbc.py") for a in argv)
    assert "--agent.bc_coeff=3.0" in argv
    assert "--run_group=factored-fb-crl-flowbc" in argv
    assert "--env_name=cube-single-play-v0" in argv
    assert not any(a.startswith("--agent.encoder=") for a in argv)


def test_crl_flowbc_pixel_config_argv():
    argv = _argv("cube_single_visual_crl_flowbc_drq")
    assert any(a.startswith("--agent=") and a.endswith("tools/wandb_mode_shim/crl_flowbc.py") for a in argv)
    assert "--agent.encoder=drq" in argv
    assert "--agent.frame_stack=3" in argv
    assert "--agent.p_aug=1.0" in argv
    assert "--agent.bc_coeff=3.0" in argv
    assert "--run_group=factored-fb-crl-flowbc-pixel-drq" in argv
    assert "--env_name=visual-cube-single-play-v0" in argv
```

- [ ] **Step 2: Run it — expect FAIL (configs missing)**

Run: `.venv/bin/python -m pytest tests/test_crl_flowbc_configs.py -v`
Expected: FAIL with Hydra `MissingConfigException`.

- [ ] **Step 3: Create `configs/gciql/cube_single_state_crl_flowbc.yaml`**

```yaml
# CRL + FB FlowBC actor on state cube-single. agent_name=crl_flowbc is registered,
# and the agent file is resolved to an absolute path, by tools/wandb_mode_shim
# (see run_gciql._resolve_agent_file). Deltas vs cube_single_state_crl.yaml: the
# agent file, bc_coeff (3.0, FB manipulation default), run_group. State => no encoder.
defaults:
  - storage: shm
  - _self_
variant: state
env_name: cube-single-play-v0
agent_file: tools/wandb_mode_shim/crl_flowbc.py
seed: 0
train_steps: 1000000
log_interval: 5000
eval_interval: 100000
save_interval: 1000000
eval_episodes: 50
eval_on_cpu: 1
run_group: factored-fb-crl-flowbc
agent_overrides:
  bc_coeff: 3.0
wandb_mode: online
```

- [ ] **Step 4: Create `configs/gciql/cube_single_visual_crl_flowbc_drq.yaml`**

```yaml
# CRL + FB FlowBC actor on visual cube-single with the DrQ front-end. Deltas vs
# cube_single_visual_crl_flowbc state: the env, the DrQ encoder overrides
# (encoder/frame_stack/p_aug), 500k steps, eval_on_cpu=0, run_group. OGBENCH_DRQ_FRONTEND
# is set by run_gciql because encoder == 'drq'.
defaults:
  - storage: shm
  - _self_
variant: visual
env_name: visual-cube-single-play-v0
agent_file: tools/wandb_mode_shim/crl_flowbc.py
seed: 0
train_steps: 500000
log_interval: 5000
eval_interval: 100000
save_interval: 50000
eval_episodes: 50
eval_on_cpu: 0
run_group: factored-fb-crl-flowbc-pixel-drq
agent_overrides:
  bc_coeff: 3.0
  batch_size: 256
  encoder: drq
  frame_stack: 3
  p_aug: 1.0
wandb_mode: online
```

- [ ] **Step 5: Run the test — expect PASS**

Run: `.venv/bin/python -m pytest tests/test_crl_flowbc_configs.py -v`
Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add configs/gciql/cube_single_state_crl_flowbc.yaml configs/gciql/cube_single_visual_crl_flowbc_drq.yaml tests/test_crl_flowbc_configs.py
git commit -m "feat(crl-flowbc): GCIQL-track CRL+FlowBC configs (state + pixel/DrQ)"
```

---

## Task 8: Full suite + behavioral smoke (manual gate)

No new code. Verifies the gates pass and the agent trains end-to-end. **Deployment is blocked until Gate 1 (Task 2) and Gate 2 (Task 3) pass** (or SKIP only when `.venv-jax` is absent).

- [ ] **Step 1: Run the whole flow test suite**

Run: `.venv/bin/python -m pytest tests/test_ogbench_flow_nets.py tests/test_ogbench_flow_parity.py tests/test_ogbench_flow_train_parity.py tests/test_crl_flowbc_injection.py tests/test_run_gciql_agent_resolve.py tests/test_crl_flowbc_configs.py -v`
Expected: all PASS, or the jax-dependent ones SKIP only if `.venv-jax` is absent. Confirm Gate 1 (`<1e-5`) and Gate 2 (`<1e-6`) both PASS — if either is SKIP, install `.venv-jax` (`make install-jax`) and re-run; do not proceed on a SKIP.

- [ ] **Step 2: State CRL+FlowBC smoke on GPU 6**

```bash
CUDA_VISIBLE_DEVICES=6 .venv/bin/python run_gciql.py \
    --config-name cube_single_state_crl_flowbc \
    storage=shm seed=0 wandb_mode=disabled \
    train_steps=2000 eval_interval=1000 eval_episodes=2 save_interval=2000 \
    run_group=crl-flowbc-smoke
```
Check: starts (agent `crl_flowbc` resolves; `--agent` is absolute), `actor/bc_flow_loss`, `actor/bc_error`, `actor/q_mean`, and `critic/contrastive_loss` are finite and trending; eval at 1000/2000 runs; checkpoint `params_2000.pkl` under `/dev/shm/gciql_outputs/factored-fb/crl-flowbc-smoke/sd000_*/`.

- [ ] **Step 3: Pixel CRL+FlowBC+DrQ smoke on GPU 7**

```bash
CUDA_VISIBLE_DEVICES=7 .venv/bin/python run_gciql.py \
    --config-name cube_single_visual_crl_flowbc_drq \
    storage=shm seed=0 wandb_mode=disabled \
    train_steps=2000 eval_interval=1000 eval_episodes=2 save_interval=2000 \
    run_group=crl-flowbc-smoke
```
Check: `encoder=drq` resolves for the flow actor's `actor_obs`/`actor_goal` + critic encoders (no `KeyError: 'drq'`, no shape error); fits one 23 GB GPU (no OOM); losses finite; eval runs; throughput healthy (~20+ it/s, not ~20 s/it); checkpoint written. If OOM, STOP and report (CRL+FlowBC builds critic ×2 + actor obs/goal encoders).

- [ ] **Step 4: Record outcome**

Note both smoke results (it/s, loss trends, checkpoint paths) in the run notes. Then the CRL+FlowBC configs are ready; updating the CRL sweep launch to point at them is a separate launch decision (see the CRL plan, `2026-05-22-crl-cube-runs.md`).

---

## Self-review (completed during planning)

- **Spec coverage:** Component 1 flow nets → Task 1; Component 2 agent → Task 4; Component 3 injection → Tasks 5 (run_gciql) + 6 (sitecustomize); Component 4 configs → Task 7; Component 5 tests → Tasks 1–7; **Gate 1** → Task 2; **Gate 2** → Task 3; deployment-blocked-on-gates + smoke → Task 8. Deviations (CRL Q, no z-norm, separate encoders) realized in Task 4's `create`/`actor_loss`.
- **Placeholder scan:** none — every code/test step is complete; the only manual steps (smoke) have concrete commands and pass criteria.
- **Type/name consistency:** module/param names (`VectorField` l0..l3/out; `NoiseConditionedActor` embed_z/embed_s `_l0`/`_ln`/`_lout`, policy `_l0`/`_l1`/`_out`), `compute_flow_actions(vf, params, obs, noise, flow_steps)`, ModuleDict keys `critic`/`actor`/`actor_vf`, config keys (`flow_steps`, `bc_coeff`, `lr_actor_vf`, `actor_vf_hidden_dim/layers`, `actor_hidden_dim/layers`, `actor_embedding_layers`, `action_dim`), `_resolve_agent_file`, `agents['crl_flowbc']`, run_groups `factored-fb-crl-flowbc{,-pixel-drq}` — used identically across Tasks 1–8. The transplant index→name mapping is identical in Gate 1 and Gate 2.
```
