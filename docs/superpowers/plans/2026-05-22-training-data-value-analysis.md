# Training-Data Value Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure FB's and GCIQL's value↔(cube-to-goal-distance) relationship on the offline **training** states, per manipulation phase, to test whether the rollout-diagnosed value pathology is intrinsic to the learned value.

**Architecture:** A new profile+aggregate pair (two-venv, like `outcome_clips`). The FB pass (`.venv`) samples aligned `(obs, physics, action)` from the raw buffer npz, classifies each state's phase vectorized from `physics`, scores FB policy/data values per task, and writes a shared `training_states.npz`. The GCIQL pass (`.venv-jax-cpu`) scores `V(s,g)` on the *same* states. The aggregate builds the Spearman table, value-vs-distance curves, dataset phase-composition bar, and value-over-cube-xy scenes.

**Tech Stack:** Python 3 (`.venv` torch / `.venv-jax-cpu` jax), numpy, pandas, matplotlib, MuJoCo/ogbench.

**Spec:** `docs/superpowers/specs/2026-05-22-training-data-value-analysis-design.md`

---

## Conventions & ground rules

- FB/aggregate run under `.venv/bin/python`; GCIQL under `.venv-jax-cpu/bin/python`. Tests under `.venv`.
- Commit only `scripts/`, `evals/`, `tests/`, `docs/`. `analysis/`, `RESULTS/`, `datasets/` stay untracked.
- Phase taxonomy = reach / grasp / transport. Confirmed buffer facts: `physics` is 21-d qpos; cube xyz = `physics[:,14:17]`; gripper opening = `clip(physics[:,6]/0.8, 0, 1)` (joint `ur5e/robotiq/right_driver_joint`, id 6); `table_z`≈0.02 (`cur_task_info['init_xyzs'][tb][2]`).
- Thresholds from `evals.phase_probe.Thresholds` (`delta_lift`, `tau_grip`); region rule mirrors `evals._profile_core._region`: transport = lift>delta & grip>tau; grasp = grip>tau; reach = else.
- State checkpoints: FB `RESULTS/Factored-FB-cube-run/*__s3/checkpoints/final.pt`; GCIQL `RESULTS/gciql_20260518_201030/factored-fb/factored-fb-gciql/sd003_*` (`--step 1000000`).

## File structure

- `evals/training_value.py` (create) — pure helpers: `region_labels`, `cube_to_goal_dist`, `phase_spearman_table`.
- `scripts/training_value_profile.py` (create) — `--method fb` (sampler+classifier+FB values, writes shared npz) and `--method gciql` (scores shared states).
- `scripts/training_value_aggregate.py` (create) — tables/curves/composition/scenes.
- `tests/test_training_value.py` (create) — unit tests for the pure helpers.

---

## Task 1: Pure helpers (`evals/training_value.py`)

**Files:**
- Create: `evals/training_value.py`
- Test: `tests/test_training_value.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_training_value.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.training_value import (region_labels, cube_to_goal_dist,
                                   phase_spearman_table)
from evals.phase_probe import Thresholds


def test_region_labels_reach_grasp_transport():
    thr = Thresholds()
    grip = np.array([0.0, 0.9, 0.9])          # open, closed, closed
    lift = np.array([0.0, 0.0, 0.20])         # on table, on table, lifted
    out = region_labels(grip, lift, thr)
    assert list(out) == ["reach", "grasp", "transport"]


def test_cube_to_goal_dist():
    cube = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    d = cube_to_goal_dist(cube, np.array([0.0, 0.0, 0.0]))
    assert np.allclose(d, [0.0, 5.0])


def test_phase_spearman_table_monotone():
    # value rises as d falls within each phase -> rho ~ +1
    rows = []
    for ph in ("reach", "transport"):
        for i in range(8):
            d = 0.3 - 0.03 * i
            rows.append({"task": "task1", "region": ph, "d": d,
                         "V": 1.0 - d})
    df = pd.DataFrame(rows)
    tab = phase_spearman_table(df, value_col="V")
    r = tab.set_index("region")["rho_mean"]
    assert r.loc["transport"] > 0.99 and r.loc["reach"] > 0.99
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_training_value.py -v`
Expected: FAIL — `No module named 'evals.training_value'`.

- [ ] **Step 3: Implement the helpers**

Create `evals/training_value.py`:

```python
"""evals/training_value.py — pure helpers for the training-data value
analysis (numpy/pandas only; no torch/jax)."""

from __future__ import annotations

import numpy as np

from evals._profile_core import _spearman


def region_labels(grip: np.ndarray, lift: np.ndarray, thr) -> np.ndarray:
    """Per-state phase: 'transport' if lifted+gripped, else 'grasp' if
    gripped, else 'reach'. Mirrors evals._profile_core._region (lift->grasp)."""
    grip = np.asarray(grip, np.float64).reshape(-1)
    lift = np.asarray(lift, np.float64).reshape(-1)
    transport = (lift > thr.delta_lift) & (grip > thr.tau_grip)
    return np.where(transport, "transport",
                    np.where(grip > thr.tau_grip, "grasp", "reach"))


def cube_to_goal_dist(cube_xyz: np.ndarray, goal_xyz: np.ndarray) -> np.ndarray:
    """Euclidean cube-to-goal distance per row."""
    cube = np.asarray(cube_xyz, np.float64).reshape(-1, 3)
    goal = np.asarray(goal_xyz, np.float64).reshape(3)
    return np.linalg.norm(cube - goal, axis=1)


def phase_spearman_table(df, value_col: str = "V"):
    """Per (region) Spearman rho(value, -d): mean/std over tasks. `df` has
    columns task, region, d, <value_col>. Returns a tidy DataFrame."""
    import pandas as pd

    rows = []
    for region, g in df.groupby("region"):
        per_task = []
        for task, gt in g.groupby("task"):
            per_task.append(_spearman(gt[value_col].to_numpy(),
                                      -gt["d"].to_numpy()))
        per_task = np.array(per_task, dtype=float)
        rows.append({"region": region,
                     "rho_mean": float(np.nanmean(per_task)),
                     "rho_std": float(np.nanstd(per_task)),
                     "n_tasks": int(np.isfinite(per_task).sum())})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_training_value.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add evals/training_value.py tests/test_training_value.py
git commit -m "feat(training_value): pure phase/distance/spearman helpers"
```

---

## Task 2: FB profile pass + shared sampler (`scripts/training_value_profile.py`)

**Files:**
- Create: `scripts/training_value_profile.py`

- [ ] **Step 1: Create the script with the FB backend**

```python
"""scripts/training_value_profile.py — value vs cube-to-goal-distance on the
offline TRAINING states, per phase, for FB (torch) and GCIQL (jax).

--method fb    : samples aligned (obs, physics, action) from the buffer npz,
                 classifies phases, scores FB policy/data values per task, and
                 writes the shared training_states.npz (run under .venv).
--method gciql : scores GCIQL V(s,g) on the SAME shared states
                 (run under .venv-jax-cpu).

Usage (FB):
    .venv/bin/python scripts/training_value_profile.py --method fb \
        --config RESULTS/Factored-FB-cube-run/<run>/.hydra/config.yaml \
        --checkpoint RESULTS/Factored-FB-cube-run/<run>/checkpoints/final.pt \
        --data-path datasets --mujoco-gl glfw --n-states 20000 \
        --out analysis/training_value

Usage (GCIQL):
    .venv-jax-cpu/bin/python scripts/training_value_profile.py --method gciql \
        --run-dir RESULTS/gciql_.../sd003_... --step 1000000 \
        --out analysis/training_value --mujoco-gl glfw
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.training_value import region_labels, cube_to_goal_dist

CUBE_SLICE = slice(14, 17)
GRIP_QPOS_IDX = 6      # ur5e/robotiq/right_driver_joint id (verified)
TASK_TMPL = "cube-single-play-singletask-task{n}-v0"


def _episode_files(data_path: str, domain: str) -> List[str]:
    return sorted(glob.glob(str(Path(data_path) / domain / "buffer"
                                / "episode_*.npz")))


def _sample_states(files, n_states, seed=0):
    """Aligned (obs, physics, action) sampled across episodes."""
    rng = np.random.default_rng(seed)
    obs_l, phys_l, act_l = [], [], []
    for f in files:
        z = np.load(f)
        obs_l.append(np.asarray(z["observation"], np.float32))
        phys_l.append(np.asarray(z["physics"], np.float32))
        act_l.append(np.asarray(z["action"], np.float32))
    obs = np.concatenate(obs_l); phys = np.concatenate(phys_l)
    act = np.concatenate(act_l)
    if len(obs) > n_states:
        idx = rng.choice(len(obs), n_states, replace=False)
        obs, phys, act = obs[idx], phys[idx], act[idx]
    return obs, phys, act


def _phase_composition(files, thr, table_z):
    """Fraction reach/grasp/transport over ALL transitions (trajectory pass)."""
    import pandas as pd
    grip_l, lift_l = [], []
    for f in files:
        z = np.load(f)
        phys = np.asarray(z["physics"], np.float32)
        grip_l.append(np.clip(phys[:, GRIP_QPOS_IDX] / 0.8, 0, 1))
        lift_l.append(phys[:, 16] - table_z)
    reg = region_labels(np.concatenate(grip_l), np.concatenate(lift_l), thr)
    vc = pd.Series(reg).value_counts(normalize=True)
    return pd.DataFrame({"region": vc.index, "fraction": vc.values})


def _fb(args) -> None:
    import torch
    import pandas as pd
    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from evals.representation_profile import q_values, v_values
    from evals.ogbench import OGBenchEvaluator
    from envs.ogbench import ALL_TASKS, create_ogbench_env
    from evals.phase_probe import Thresholds
    from data.ogbench import load_ogbench_dataset

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(args.config, device=args.device)
    if args.data_path:
        cfg.data_path = args.data_path
    env, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint, map_location=args.device)
    # table_z + thresholds from a task env
    e0, _ = create_ogbench_env(TASK_TMPL.format(n=1), seed=cfg.seed,
                               obs_type=cfg.obs_type)
    tb = int(getattr(e0.unwrapped, "_target_block", 0) or 0)
    table_z = float(e0.unwrapped.cur_task_info["init_xyzs"][tb][2])
    e0.close()
    thr = Thresholds()

    files = _episode_files(cfg.data_path, cfg.domain)
    obs, phys, act = _sample_states(files, args.n_states, seed=cfg.seed)
    cube = phys[:, CUBE_SLICE]
    grip = np.clip(phys[:, GRIP_QPOS_IDX] / 0.8, 0, 1)
    region = region_labels(grip, phys[:, 16] - table_z, thr)

    # z per task via the evaluator (needs the buffer)
    buffer = load_ogbench_dataset(domain=cfg.domain, data_path=cfg.data_path,
                                  load_n_episodes=cfg.load_n_episodes,
                                  device=args.device,
                                  n_transitions=cfg.n_transitions,
                                  obs_type=cfg.obs_type)
    evaluator = OGBenchEvaluator(domain=cfg.domain, agent=agent,
                                 offline_buffer=buffer,
                                 relabel_size=cfg.eval_relabel_size,
                                 n_episodes=1, shift_reward=cfg.eval_shift_reward,
                                 obs_type=cfg.obs_type, seed=cfg.seed,
                                 device=args.device, use_wandb=False)

    tasks = list(ALL_TASKS.get(cfg.domain, []))
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    act_t = torch.as_tensor(act, dtype=torch.float32)
    rows, goals = [], {}
    for ti, task in enumerate(tasks, start=1):
        z, _ = evaluator._infer_z(task)
        e, _ = create_ogbench_env(task, seed=cfg.seed, obs_type=cfg.obs_type)
        tbk = int(getattr(e.unwrapped, "_target_block", 0) or 0)
        g = np.asarray(e.unwrapped.cur_task_info["goal_xyzs"][tbk], np.float64)
        e.close()
        goals[f"task{ti}"] = g
        d = cube_to_goal_dist(cube, g)
        v_pol = v_values(agent.model, agent, obs_t, z)
        v_dat = q_values(agent.model, obs_t, act_t, z)
        for i in range(len(obs)):
            rows.append({"task": f"task{ti}", "region": str(region[i]),
                         "d": float(d[i]), "cube_x": float(cube[i, 0]),
                         "cube_y": float(cube[i, 1]),
                         "V_policy": float(v_pol[i]),
                         "V_data": float(v_dat[i])})
    pd.DataFrame(rows).to_parquet(out / "fb_values.parquet")
    _phase_composition(files, thr, table_z).to_parquet(
        out / "phase_composition.parquet")
    np.savez(out / "training_states.npz", obs=obs, cube=cube,
             region=region.astype("U12"),
             goals=np.stack([goals[f"task{i}"] for i in range(1, 6)]))
    print(f"[training_value:fb] {len(obs)} states x {len(tasks)} tasks -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=["fb", "gciql"])
    ap.add_argument("--out", default="analysis/training_value")
    ap.add_argument("--n-states", type=int, default=20000)
    ap.add_argument("--mujoco-gl", default=None)
    # fb
    ap.add_argument("--config")
    ap.add_argument("--checkpoint")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-path", default="datasets")
    # gciql
    ap.add_argument("--run-dir")
    ap.add_argument("--step", type=int, default=1000000)
    args = ap.parse_args()
    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    if args.method == "fb":
        assert args.config and args.checkpoint, "fb needs --config/--checkpoint"
        _fb(args)
    else:
        assert args.run_dir, "gciql needs --run-dir"
        _gciql(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Parse-check**

Run: `.venv/bin/python -c "import ast; ast.parse(open('scripts/training_value_profile.py').read()); print('parse OK')"` (will reference `_gciql` — add a stub in Task 3 before running).
Expected: `parse OK` only after Task 3 adds `_gciql`; for now verify the FB path by the smoke in Step 3.

- [ ] **Step 3: Smoke the FB pass (reduced)**

Add a temporary `def _gciql(args): raise NotImplementedError` if needed to import, then:
```bash
D=$(ls -d RESULTS/Factored-FB-cube-run/*__s3 | head -1)
MUJOCO_GL=glfw .venv/bin/python scripts/training_value_profile.py --method fb \
  --config "$D/.hydra/config.yaml" --checkpoint "$D/checkpoints/final.pt" \
  --data-path datasets --mujoco-gl glfw --n-states 800 --out /tmp/tv_smoke
```
Expected: prints `[training_value:fb] 800 states x 5 tasks -> /tmp/tv_smoke`; `/tmp/tv_smoke/{fb_values.parquet,phase_composition.parquet,training_states.npz}` exist; `fb_values.parquet` has columns `task,region,d,cube_x,cube_y,V_policy,V_data` and all three regions present in `phase_composition.parquet`.

- [ ] **Step 4: Commit**

```bash
git add scripts/training_value_profile.py
git commit -m "feat(training_value): FB pass — sample+classify training states, score policy/data values"
```

---

## Task 3: GCIQL profile pass (same script)

**Files:**
- Modify: `scripts/training_value_profile.py` (add `_gciql`)

- [ ] **Step 1: Add the GCIQL backend**

Insert into `scripts/training_value_profile.py` (replacing any temporary stub):

```python
def _gciql(args) -> None:
    ogb_impls = REPO_ROOT / "third_party" / "ogbench" / "impls"
    if str(ogb_impls) not in sys.path:
        sys.path.insert(0, str(ogb_impls))
    import jax
    import jax.numpy as jnp
    import pandas as pd
    from utils.flax_utils import restore_agent
    from scripts.gciql_profile import parse_flags

    out = Path(args.out)
    st = np.load(out / "training_states.npz", allow_pickle=True)
    obs = np.asarray(st["obs"], np.float32)
    cube = np.asarray(st["cube"], np.float64)
    region = [str(r) for r in st["region"]]

    ogb_impls = REPO_ROOT / "third_party" / "ogbench" / "impls"
    import ogbench  # noqa: F401  (vendored, registers envs)
    flags = parse_flags(args.run_dir)
    saved = flags.get("agent")
    agent_name = (saved.get("agent_name") if isinstance(saved, dict)
                  else None) or "gciql"
    if agent_name == "gcivl":
        from agents.gcivl import GCIVLAgent as AgentCls, get_config
    else:
        from agents.gciql import GCIQLAgent as AgentCls, get_config
    config = get_config()
    if isinstance(saved, dict):
        for k, v in saved.items():
            if k in config:
                config[k] = v

    env = ogbench.make_env_and_datasets(flags["env_name"], env_only=True)
    ex_obs, _ = env.reset(options=dict(task_id=1))
    agent = AgentCls.create(flags["seed"],
                            np.asarray(ex_obs, np.float32)[None],
                            np.asarray(env.action_space.sample(),
                                       np.float32)[None], config)
    agent = restore_agent(agent, str(args.run_dir), args.step)

    @jax.jit
    def _value(o, g):
        v = agent.network.select("value")(o, g)
        return jnp.asarray(v).mean(0) if isinstance(v, (tuple, list)) else v

    rows = []
    obs_j = jnp.asarray(obs)
    for ti in range(1, 6):
        _, info = env.reset(options=dict(task_id=ti))
        goal = np.asarray(info["goal"], np.float32)
        tbk = int(getattr(env.unwrapped, "_target_block", 0) or 0)
        g_xyz = np.asarray(env.unwrapped.cur_task_info["goal_xyzs"][tbk],
                           np.float64)
        d = cube_to_goal_dist(cube, g_xyz)
        gj = jnp.broadcast_to(jnp.asarray(goal), obs_j.shape) \
            if goal.shape == obs.shape[1:] else jnp.tile(
                jnp.asarray(goal)[None], (len(obs), 1))
        V = np.asarray(_value(obs_j, gj)).reshape(-1)
        for i in range(len(obs)):
            rows.append({"task": f"task{ti}", "region": region[i],
                         "d": float(d[i]), "cube_x": float(cube[i, 0]),
                         "cube_y": float(cube[i, 1]), "V": float(V[i])})
    env.close()
    pd.DataFrame(rows).to_parquet(out / "gciql_values.parquet")
    print(f"[training_value:gciql] {len(obs)} states x 5 tasks -> {out}")
```

- [ ] **Step 2: Parse-check + smoke (after FB smoke produced training_states.npz)**

```bash
.venv/bin/python -c "import ast; ast.parse(open('scripts/training_value_profile.py').read()); print('parse OK')"
D=$(ls -d RESULTS/gciql_20260518_201030/factored-fb/factored-fb-gciql/sd003_* | head -1)
MUJOCO_GL=glfw .venv-jax-cpu/bin/python scripts/training_value_profile.py --method gciql \
  --run-dir "$D" --step 1000000 --out /tmp/tv_smoke --mujoco-gl glfw
```
Expected: `parse OK`; `[training_value:gciql] 800 states x 5 tasks`; `/tmp/tv_smoke/gciql_values.parquet` has columns `task,region,d,cube_x,cube_y,V`. If `info["goal"]` is a full goal observation (shape == obs row), it is used as g; otherwise it is tiled — verify V varies across states (not constant).

- [ ] **Step 3: Commit**

```bash
git add scripts/training_value_profile.py
git commit -m "feat(training_value): GCIQL pass — score V(s,g) on shared training states"
```

---

## Task 4: Aggregate (`scripts/training_value_aggregate.py`)

**Files:**
- Create: `scripts/training_value_aggregate.py`

- [ ] **Step 1: Create the aggregate**

```python
"""scripts/training_value_aggregate.py — Spearman table + value-vs-distance
curves + phase-composition bar + value-over-cube-xy scenes for the
training-data value analysis. Reads fb_values/gciql_values/phase_composition
from --root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.training_value import phase_spearman_table

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REGIONS = ["reach", "grasp", "transport"]
# (df, value_col, label)
def _methods(root: Path):
    fb = pd.read_parquet(root / "fb_values.parquet")
    out = [(fb, "V_policy", "FB policy"), (fb, "V_data", "FB data")]
    gq = root / "gciql_values.parquet"
    if gq.exists():
        out.append((pd.read_parquet(gq), "V", "GCIQL"))
    return out


def aggregate(root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    methods = _methods(root)

    # 1. Spearman table (region x method, mean over tasks).
    rows = []
    for df, vc, label in methods:
        tab = phase_spearman_table(df, value_col=vc)
        for _, r in tab.iterrows():
            rows.append({"method": label, "region": r["region"],
                         "rho_mean": r["rho_mean"], "rho_std": r["rho_std"]})
    sp = pd.DataFrame(rows)
    sp.to_parquet(out / "spearman_table.parquet")
    md = ["# Training-state value gradient: Spearman rho(V, -d) by phase", "",
          "| method | " + " | ".join(REGIONS) + " |",
          "| :--- | " + " | ".join([":---"] * len(REGIONS)) + " |"]
    for label in [m[2] for m in methods]:
        cells = [label]
        for rg in REGIONS:
            s = sp[(sp.method == label) & (sp.region == rg)]
            cells.append(f"{s['rho_mean'].iloc[0]:.3f}" if len(s) else "—")
        md.append("| " + " | ".join(cells) + " |")
    (out / "spearman_table.md").write_text("\n".join(md) + "\n")

    # 2. value-vs-distance curves (faceted by phase, line per method, z-scored).
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(5 * len(REGIONS), 4),
                             squeeze=False)
    for ax, rg in zip(axes[0], REGIONS):
        for df, vc, label in methods:
            sub = df[df.region == rg]
            if sub.empty:
                continue
            v = sub[vc].to_numpy(float)
            vz = (v - v.mean()) / (v.std() + 1e-9)
            edges = np.unique(np.quantile(sub["d"], np.linspace(0, 1, 10)))
            ctr = 0.5 * (edges[:-1] + edges[1:])
            idx = np.clip(np.digitize(sub["d"], edges) - 1, 0, len(ctr) - 1)
            mv = pd.DataFrame({"b": idx, "vz": vz}).groupby("b")["vz"].mean()
            ax.plot(ctr[mv.index], mv.values, marker="o", label=label)
        ax.set_title(rg); ax.set_xlabel("cube -> goal distance")
        ax.set_ylabel("z-scored value"); ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Training states: value vs cube-to-goal distance, by phase")
    fig.tight_layout(); fig.savefig(out / "value_vs_dist.png", dpi=130)
    plt.close(fig)

    # 3. dataset phase composition bar.
    comp = pd.read_parquet(root / "phase_composition.parquet")
    fig, ax = plt.subplots(figsize=(5, 4))
    comp = comp.set_index("region").reindex(REGIONS).fillna(0.0)
    ax.bar(REGIONS, comp["fraction"].values)
    ax.set_ylabel("fraction of training transitions")
    ax.set_title("Dataset phase composition")
    fig.tight_layout(); fig.savefig(out / "phase_composition.png", dpi=130)
    plt.close(fig)

    # 4. value-over-cube-xy scenes (phase x method): binned mean value.
    for df, vc, label in methods:
        fig, axes = plt.subplots(1, len(REGIONS),
                                 figsize=(5 * len(REGIONS), 4), squeeze=False)
        for ax, rg in zip(axes[0], REGIONS):
            sub = df[df.region == rg]
            ax.set_title(f"{label} — {rg} (n={len(sub)})")
            if len(sub) >= 20:
                h = ax.hexbin(sub["cube_x"], sub["cube_y"], C=sub[vc],
                              gridsize=20, reduce_C_function=np.mean,
                              cmap="Reds")
                fig.colorbar(h, ax=ax, shrink=0.8)
            ax.set_xlabel("cube x"); ax.set_ylabel("cube y")
        fig.tight_layout()
        fname = f"scene_{label.replace(' ', '_').lower()}.png"
        fig.savefig(out / fname, dpi=130)
        plt.close(fig)

    print(f"[training_value_aggregate] -> {out}")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="analysis/training_value")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    aggregate(root, Path(args.out) if args.out else root / "aggregate")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke the aggregate on the /tmp smoke outputs**

```bash
.venv/bin/python scripts/training_value_aggregate.py --root /tmp/tv_smoke --out /tmp/tv_smoke/aggregate 2>&1 | tail -12
ls /tmp/tv_smoke/aggregate
```
Expected: prints the Spearman table; `aggregate/` has `spearman_table.{md,parquet}`, `value_vs_dist.png`, `phase_composition.png`, and `scene_fb_policy.png`, `scene_fb_data.png`, `scene_gciql.png`.

- [ ] **Step 3: Commit**

```bash
git add scripts/training_value_aggregate.py
git commit -m "feat(training_value): aggregate — spearman table, value-vs-dist, composition, scenes"
```

---

## Task 5: Real run + verification

**Files:** none modified (run + verify).

- [ ] **Step 1: Full FB pass** (seed s3, 20k states)

```bash
D=$(ls -d RESULTS/Factored-FB-cube-run/*__s3 | head -1)
MUJOCO_GL=glfw .venv/bin/python scripts/training_value_profile.py --method fb \
  --config "$D/.hydra/config.yaml" --checkpoint "$D/checkpoints/final.pt" \
  --data-path datasets --mujoco-gl glfw --n-states 20000 --out analysis/training_value
```
Expected: `[training_value:fb] 20000 states x 5 tasks`.

- [ ] **Step 2: Full GCIQL pass** (same states)

```bash
D=$(ls -d RESULTS/gciql_20260518_201030/factored-fb/factored-fb-gciql/sd003_* | head -1)
MUJOCO_GL=glfw .venv-jax-cpu/bin/python scripts/training_value_profile.py --method gciql \
  --run-dir "$D" --step 1000000 --out analysis/training_value --mujoco-gl glfw
```
Expected: `[training_value:gciql] 20000 states x 5 tasks`.

- [ ] **Step 3: Aggregate + read the table**

```bash
.venv/bin/python scripts/training_value_aggregate.py --root analysis/training_value
cat analysis/training_value/aggregate/spearman_table.md
```
Expected: the per-phase Spearman table for FB-policy / FB-data / GCIQL, plus the scenes/curves/composition PNGs.

- [ ] **Step 4: Full test suite + commit boundary**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (incl. the new `test_training_value.py`).
Run: `git status --short` — only `scripts/`,`evals/`,`tests/`,`docs/` tracked changes; `analysis/` untracked.

- [ ] **Step 5: Commit the plan doc**

```bash
git add docs/superpowers/plans/2026-05-22-training-data-value-analysis.md
git commit -m "docs(training-value): implementation plan"
```

---

## Self-review notes

- **Spec coverage:** FB policy + data values (Task 2), GCIQL V(s,g) (Task 3), per-phase reach/grasp/transport (`region_labels`, Task 1), dataset phase composition (Task 2 `_phase_composition`), per-task→aggregate (loops over 5 tasks; Spearman mean over tasks in `phase_spearman_table`), Spearman table + value-vs-distance curves + composition bar + value-over-cube-xy scenes (Task 4), shared states (FB writes `training_states.npz`, GCIQL reads it). All covered.
- **Placeholder scan:** none — every step has runnable code/commands. The Task 2 Step 2 note about the temporary `_gciql` stub is a real ordering instruction (the stub is removed in Task 3).
- **Type/name consistency:** `region_labels(grip, lift, thr)`, `cube_to_goal_dist(cube_xyz, goal_xyz)`, `phase_spearman_table(df, value_col)` used identically across tasks; parquet columns (`task, region, d, cube_x, cube_y, V_policy, V_data` / `V`) consistent between profile writers and the aggregate reader; `training_states.npz` keys (`obs, cube, region, goals`) written in Task 2 and read in Task 3.
- **Open risk (flagged in spec):** GCIQL `info["goal"]` modality — if the state GCIQL goal is not a full observation vector, Task 3 Step 2 verification (V varies) catches it and the goal construction is adjusted (use the goal observation the agent expects).
