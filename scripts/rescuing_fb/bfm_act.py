#!/usr/bin/env python
"""scripts/rescuing_fb/bfm_act.py -- spec 26 R0b(i): the BFM interface.

One entry point `act(obs, prompt, ckpt)` with prompt in {goal observation,
reward function, demo trajectory} (PAPER/Multistep-FB/specs/26-bfm-roadmap.md
sec.2). This file is a THIN WRAPPER over existing audited code -- the repo's
hard rule after the spec-24 v1 harness reimplemented protocol and produced a
broken instrument (see eval_density_z_rewards.py's module docstring for that
post-mortem). Every branch routes into machinery that already exists:

  goal    -> direct conditioning (native): `agent.sample_actions(obs, goals=g,
             temperature=0)` -- the exact call every audited rollout in
             eval_density_zeroshot.py / eval_measure_g4_spearman.py makes.
  reward  -> the v2 zero-shot machinery's search-based index selection,
             IMPORTED not reimplemented: `build_pool` +
             `decode_goal_retrieval_density` from eval_density_zeroshot.py,
             with the SAME rng stream ORDER the audited harnesses pin
             (relabel draw of `relabel_size` FIRST, then the pool draw -- see
             eval_density_z_rewards.py "rng stream: IDENTICAL ORDER") and the
             archived RET_k1 gate protocol's retrieval_k=1 default. For a
             measure_density_z checkpoint the same decode runs in its e2b
             form (z candidates in the cand_obs slot through `_ZIndexView` /
             `z_candidates`, both imported from eval_density_z_rewards.py).
             The decoded index (goal obs, or z) is cached per prompt object
             and the per-step call is then the goal branch.
  demo    -> sliding-window waypoint conditioning (the one genuinely NEW
             piece of protocol in R0b, kept pure-numpy and unit-tested):
             extract the demo's object-coordinate path via the checkpoint's
             OWN hit-spec (`env_geometry.geometry_for(...).hit_spec`, the
             single geometry definition every eval script shares; the
             checkpoint's recorded `measure_hit_spec` wins), then condition
             the actor on the demo observation `delta` steps ahead of the
             current nearest-progress point. Progress = argmin object
             distance (eps-scaled hit-spec coords) along the demo, MONOTONE
             NON-DECREASING by construction (the argmin is restricted to
             indices >= the previous progress). delta default 25 (contract).

The pure waypoint logic (`update_progress` / `waypoint_index` /
`waypoint_schedule` / `conditioning_permutation` / `WaypointTracker`) is
jax-free and unit-tested in tests/test_spec26_bfm_tracking.py; the shuffled-
waypoint CONTROL used by eval_tracking.py lives here too (a seeded
non-identity permutation applied ONLY to the conditioning row -- progress and
metrics always run against the true demo), so the control differs from the
treatment in exactly one place.

Per-prompt state (the demo cursor, the decoded reward index) is keyed on the
prompt OBJECT's identity and the prompt object is retained in the cache (so
`id()` cannot be recycled). A fresh prompt object = a fresh episode; call
`BFMPolicy.reset()` to drop all per-prompt state instead.

Run environment (CPU jax, same as every eval in this family):
  env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    /dev/shm/.venv-jax/bin/python -c "
  import sys; sys.path.insert(0, 'scripts/rescuing_fb')
  from bfm_act import act; import numpy as np
  a = act(obs, goal_obs, '/home/adityamohan/exp_spec18/factored-fb/GCIQL-CubeSingle-Flow-Measure-Density-d50/d18_cs_dens_s0')"
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "ogbench" / "impls"))
sys.path.insert(0, str(REPO / "tools" / "wandb_mode_shim"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "rescuing_fb"))

# Audited machinery this interface wraps. Imported, never reimplemented.
from env_geometry import geometry_for  # noqa: E402
from eval_density_zeroshot import (  # noqa: E402
    build_pool,
    check_density_consistency,
    decode_goal_retrieval_density,
)
from eval_zeroshot_rewards import eval_tasks_for  # noqa: E402

PROMPT_KINDS = ("goal", "reward", "demo")


# ======================= prompt classification (pure, unit-tested) =================

def classify_prompt(prompt) -> str:
    """goal = 1-D observation array; reward = callable r(coords [n,d_c]) ->
    [n]; demo = 2-D [T, obs_dim] trajectory of observations. Anything else is
    refused loudly -- there is no fallback guessing."""
    if callable(prompt):
        return "reward"
    arr = np.asarray(prompt)
    if arr.ndim == 1:
        return "goal"
    if arr.ndim == 2:
        return "demo"
    raise TypeError(
        f"prompt must be a callable reward, a 1-D goal observation, or a 2-D "
        f"[T, obs_dim] demo trajectory; got ndim={arr.ndim} shape={arr.shape}")


# ======================= sliding-waypoint logic (pure numpy, unit-tested) ==========

def update_progress(demo_scaled: np.ndarray, cur_scaled: np.ndarray, prev: int) -> int:
    """Nearest-progress point along the demo, MONOTONE NON-DECREASING: the
    argmin of the eps-scaled object-coordinate distance is restricted to
    indices >= `prev`, so progress can never move backwards even if the object
    physically retreats past an earlier waypoint (contract: "progress = argmin
    object distance along the demo, monotone non-decreasing").
    demo_scaled: [T, d_c] eps-scaled demo coords; cur_scaled: [d_c]."""
    demo_scaled = np.asarray(demo_scaled, np.float64)
    cur = np.asarray(cur_scaled, np.float64).reshape(-1)
    prev = int(prev)
    assert 0 <= prev < len(demo_scaled), \
        f"progress {prev} out of range for demo of length {len(demo_scaled)}"
    d = np.linalg.norm(demo_scaled[prev:] - cur[None, :], axis=1)
    return prev + int(np.argmin(d))


def waypoint_index(progress: int, delta: int, T: int) -> int:
    """The conditioning row: `delta` steps ahead of the progress point,
    clamped to the demo's final row."""
    return int(min(int(progress) + int(delta), int(T) - 1))


def waypoint_schedule(T: int, delta: int) -> np.ndarray:
    """The demo's waypoint set for the hit-rate metric: rows delta, 2*delta,
    ... plus the final row, deduplicated. (The metric's waypoints are the
    conditioning targets the delta-window sweeps through from progress=0.)"""
    assert T >= 2 and delta >= 1, f"degenerate schedule T={T} delta={delta}"
    return np.unique(np.append(np.arange(delta, T, delta, dtype=np.int64), T - 1))


def conditioning_permutation(T: int, seed: int) -> np.ndarray:
    """Seeded NON-IDENTITY permutation of the demo rows for the shuffled-
    waypoint control (eval_tracking.py): the control conditions the actor on
    demo row perm[k] wherever the treatment would use row k; progress
    computation and every metric still run against the TRUE demo, so the
    control isolates exactly "does the conditioning CONTENT matter, or is the
    hit-rate just drift". Deterministic: re-draws with seed+1, seed+2, ...
    until the permutation is non-identity (T >= 2 guaranteed to terminate)."""
    assert T >= 2, "a length-<2 demo has no non-identity permutation"
    for s in range(int(seed), int(seed) + 10_000):
        perm = np.random.default_rng(s).permutation(T)
        if not np.array_equal(perm, np.arange(T)):
            return perm
    raise RuntimeError(f"no non-identity permutation found from seed {seed}")


class WaypointTracker:
    """Per-demo sliding-waypoint state machine (pure numpy; the jax-side actor
    call stays in the caller). Holds the demo, its hit-spec coords, and the
    monotone progress cursor; `goal(cur_obs)` returns the conditioning
    observation for the current state plus a diagnostics dict.

    perm (shuffled control only): applied to the CONDITIONING row alone --
    `info['waypoint']` always reports the TRUE waypoint index k, and progress
    is always computed against the true demo path."""

    def __init__(self, demo_obs: np.ndarray, spec, delta: int = 25, perm=None):
        demo_obs = np.asarray(demo_obs, np.float32)
        assert demo_obs.ndim == 2 and len(demo_obs) >= 2, \
            f"demo must be [T>=2, obs_dim], got {demo_obs.shape}"
        self.demo_obs = demo_obs
        self.spec = spec
        self.delta = int(delta)
        assert self.delta >= 1, f"delta must be >= 1, got {delta}"
        self.demo_coords = spec.coords(demo_obs)                       # [T, d_c]
        self.demo_scaled = spec.scaled(self.demo_coords)               # [T, d_c]
        self.progress = 0
        if perm is not None:
            perm = np.asarray(perm, np.int64)
            assert perm.shape == (len(demo_obs),) and \
                np.array_equal(np.sort(perm), np.arange(len(demo_obs))), \
                f"perm must be a permutation of range({len(demo_obs)})"
        self.perm = perm

    def reset(self):
        self.progress = 0

    def goal(self, cur_obs: np.ndarray):
        """Update progress from the current observation, return
        (conditioning goal obs [obs_dim], info dict)."""
        cur_scaled = self.spec.scaled(
            self.spec.coords(np.asarray(cur_obs)[None]))[0]
        self.progress = update_progress(self.demo_scaled, cur_scaled, self.progress)
        k = waypoint_index(self.progress, self.delta, len(self.demo_obs))
        row = k if self.perm is None else int(self.perm[k])
        return self.demo_obs[row], {"progress": int(self.progress),
                                    "waypoint": int(k),
                                    "conditioned_row": int(row)}


# ======================= the policy object (jax lazy) ==============================

class BFMPolicy:
    """One checkpoint, three prompt modalities. Loads the agent lazily (first
    `act`); the reward branch additionally loads the dataset/pool machinery
    lazily (first reward prompt). All jax keys derive from `seed` and a
    per-policy action counter, so a fresh policy with the same seed replays
    the same action stream (determinism)."""

    def __init__(self, run_dir: str, checkpoint_step: int = 500_000,
                 delta: int = 25, seed: int = 0, pool_size: int = 16_384,
                 relabel_size: int = 50_000, retrieval_k: int = 1):
        cands = sorted(glob.glob(str(Path(run_dir).expanduser())))
        assert cands, f"no run dir matched {run_dir}"
        self.run_dir = Path(cands[0]).resolve()
        assert (self.run_dir / "flags.json").exists(), \
            f"{self.run_dir} has no flags.json (the arch would silently mis-build)"
        self.flags = json.loads((self.run_dir / "flags.json").read_text())
        self.cfg = dict(self.flags["agent"])
        self.checkpoint_step = int(checkpoint_step)
        self.env_name = self.flags["env_name"]
        self.delta = int(delta)
        self.seed = int(seed)
        self.pool_size = int(pool_size)
        self.relabel_size = int(relabel_size)
        self.retrieval_k = int(retrieval_k)
        # geometry travels with the weights (spec 14 sec.3.4)
        self.geo = geometry_for(self.env_name, str(self.cfg.get("measure_hit_spec") or ""))
        self.spec = self.geo.hit_spec
        self.blocks = self.spec.blocks
        self.gamma = float(self.cfg["discount"])
        self.double_q = bool(self.cfg.get("double_q", True))
        self.temp = float(self.cfg.get("measure_density_temp", 1.0))
        self.is_density = bool(self.cfg.get("measure_density", False))
        self.is_density_z = bool(self.cfg.get("measure_density_z", False))
        self._agent = None
        self._mach = None            # reward-branch dataset/pool machinery
        self._prompt_state = {}      # id(prompt) -> (prompt ref, state)
        self._step = 0               # jax key counter

    # ---- lazy agent restore (the audited load_agent_dims + sys.path dance) ----
    def _ensure_agent(self):
        if self._agent is not None:
            return self._agent
        import ogbench
        env = ogbench.make_env_and_datasets(
            eval_tasks_for(self.env_name)[1], env_only=True)
        obs_dim = int(env.observation_space.shape[0])
        act_dim = int(env.action_space.shape[0])
        if hasattr(env, "close"):
            env.close()
        from t3_offpeak_multienv import load_agent_dims
        _impls = str(REPO / "third_party" / "ogbench" / "impls")
        while _impls in sys.path:
            sys.path.remove(_impls)
        sys.path.insert(0, _impls)
        _u = sys.modules.get("utils")
        if _u is not None and not hasattr(_u, "__path__"):
            del sys.modules["utils"]
        self._agent = load_agent_dims(str(self.run_dir), self.checkpoint_step,
                                      obs_dim, act_dim)
        return self._agent

    def reset(self):
        """Drop all per-prompt state (demo cursors, decoded reward indices)."""
        self._prompt_state = {}

    # ---- the single actor call every branch funnels into ----
    def _sample(self, obs, goal, view=None) -> np.ndarray:
        import jax
        import jax.numpy as jnp
        agent = view if view is not None else self._ensure_agent()
        key = jax.random.PRNGKey(self.seed * 1_000_003 + self._step)
        self._step += 1
        a = np.asarray(agent.sample_actions(
            observations=jnp.asarray(obs, jnp.float32)[None],
            goals=jnp.asarray(goal, jnp.float32)[None],
            seed=key, temperature=0.0))[0]
        return np.clip(a, -1, 1)

    # ---- reward branch: v2 zero-shot index selection, imported end-to-end ----
    def _reward_machinery(self):
        """Dataset sample + query pool, drawn ONCE per policy with the audited
        rng stream ORDER (relabel first, then pool -- eval_density_z_rewards
        .py mirrors eval_density_zeroshot.main() and so does this)."""
        if self._mach is not None:
            return self._mach
        for w in check_density_consistency(self.is_density, True, force=False,
                                           run_dir=str(self.run_dir)):
            print(f"[bfm-act] {w}", flush=True)
        agent = self._ensure_agent()
        import ogbench
        _e, ds1, _ = ogbench.make_env_and_datasets(eval_tasks_for(self.env_name)[1])
        obs_all = np.asarray(ds1["observations"], np.float32)
        coords_all = self.spec.coords(obs_all)
        rng_np = np.random.default_rng(self.seed)
        idx = rng_np.choice(len(coords_all),
                            size=min(self.relabel_size, len(coords_all)),
                            replace=False)
        S_obs, S_coords = obs_all[idx], coords_all[idx]
        n_ds = len(obs_all)
        pool_idx = rng_np.choice(n_ds, size=min(self.pool_size, n_ds),
                                 replace=(self.pool_size > n_ds))
        pool_obs = obs_all[pool_idx]
        u_raw, b_pool = build_pool(agent, self.blocks, pool_obs, self.spec)
        mach = {"S_obs": S_obs, "S_coords": S_coords, "u_raw": u_raw,
                "b_pool": b_pool, "pool_coords": coords_all[pool_idx]}
        if self.is_density_z:
            # spec 24 e2b form: z candidates ride in the decode's cand_obs slot.
            from eval_density_z_rewards import _ZIndexView, z_candidates
            mach["zview"] = _ZIndexView(agent)
            mach["cand_obs_z"] = z_candidates(agent, pool_obs)
            mach["cand_coords_z"] = coords_all[pool_idx]
        self._mach = mach
        return mach

    def _decode_reward(self, reward_fn, x0_obs):
        """One decode per prompt: the audited `decode_goal_retrieval_density`
        (g-indexed control form; e2b z form for measure_density_z), scored
        from the CURRENT observation as the deployment start state. Returns
        (index array to ride the goals slot, view-or-None, decode info)."""
        m = self._reward_machinery()
        agent = self._ensure_agent()
        r_pool = np.asarray(reward_fn(m["pool_coords"]), np.float64)
        assert r_pool.shape == (len(m["pool_coords"]),), \
            f"reward fn must map [P,d_c] coords -> [P], got {r_pool.shape}"
        if self.is_density_z:
            z_star, info = decode_goal_retrieval_density(
                m["zview"], np.asarray(x0_obs, np.float32), m["u_raw"], m["b_pool"],
                r_pool, self.blocks, self.temp, self.double_q,
                m["cand_coords_z"], m["cand_obs_z"], retrieval_k=1,
                rng_seed=self.seed)
            return np.asarray(z_star, np.float32), m["zview"], info
        g_hat, info = decode_goal_retrieval_density(
            agent, np.asarray(x0_obs, np.float32), m["u_raw"], m["b_pool"],
            r_pool, self.blocks, self.temp, self.double_q,
            m["S_coords"], m["S_obs"], retrieval_k=self.retrieval_k,
            rng_seed=self.seed)
        return np.asarray(g_hat, np.float32), None, info

    # ---- the entry point ----
    def act(self, obs, prompt) -> np.ndarray:
        kind = classify_prompt(prompt)
        if kind == "goal":
            return self._sample(obs, np.asarray(prompt, np.float32))
        if kind == "demo":
            st = self._prompt_state.get(id(prompt))
            if st is None:
                st = (prompt, WaypointTracker(np.asarray(prompt, np.float32),
                                              self.spec, delta=self.delta))
                self._prompt_state[id(prompt)] = st
            goal, _info = st[1].goal(obs)
            return self._sample(obs, goal)
        # reward
        st = self._prompt_state.get(id(prompt))
        if st is None:
            index, view, _info = self._decode_reward(prompt, obs)
            st = (prompt, (index, view))
            self._prompt_state[id(prompt)] = st
        index, view = st[1]
        return self._sample(obs, index, view=view)


# ======================= module-level convenience ==================================

_POLICIES: dict = {}


def act(obs, prompt, ckpt, checkpoint_step: int = 500_000, delta: int = 25,
        seed: int = 0) -> np.ndarray:
    """The contract's one entry point. `ckpt` is a run dir (or glob) holding
    flags.json + params_<step>.pkl; the restored policy is cached per
    (ckpt, checkpoint_step, delta, seed)."""
    key = (str(ckpt), int(checkpoint_step), int(delta), int(seed))
    pol = _POLICIES.get(key)
    if pol is None:
        pol = BFMPolicy(str(ckpt), checkpoint_step=checkpoint_step,
                        delta=delta, seed=seed)
        _POLICIES[key] = pol
    return pol.act(obs, prompt)
