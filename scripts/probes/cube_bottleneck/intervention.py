"""Intervention: does correcting FB's action values prevent its manipulation failures?
(cube-single, FB and Density-FB; docs/notes/2026-09-20-cube-intervention.md)

The decision-by-phase figure shows an ASSOCIATION: FB's critic orders the decisive action
pair at chance once the cube is carried, and FB fails there. This probe intervenes on the
ONE thing that differs, the values used to pick the first action, and keeps everything else
fixed: the same restored state, the same 12 candidate actions, the same frozen policy for
every later step.

At every restored state s of the failure-figure banks (approach / in-hand / carrying, 64 per
bank per seed, seeds 0/1/2), the same 12 candidates as the action probe (each policy's own
proposal, secure, release, 4 Gaussian perturbations of each proposal; the action probe's rng
keys), and the critic's own reward-inferred task vector z:

  (a) actor    the actor's own proposal (candidate `fb_policy` for FB, `density_policy` for
               Density-FB)
  (b) critic   the candidate with the highest critic value q(s, a, z) -- the critic's own
               actor readout (FB: F(s,a,z).z; Density-FB: the measure-weighted task reward)
  (c) rollout  the candidate with the highest Monte-Carlo return under the SAME frozen
               policy: the action probe's K = 10 continuations x 100 steps,
               return = sum_t gamma^t 1[cube within 0.04 m of its goal], gamma = 0.99
               (the run's discount) -- the same reward and discount the critic is trained on.

The selection estimates (q and the MC return of every candidate) are READ from the cached
action probe (analysis/failure_figs/<env>/actions/actions_long_seed*_1000000_{app,car}.csv,
written by failure_figs.py stage `actionprobe` 2026-09-12); nothing is re-estimated for the
selection. Ties at the maximum keep the actor's own proposal when it is among the tied
(no information => no intervention).

Evaluation: FRESH rollouts with keys disjoint from the selection rollouts
(fold_in(PRNGKey(31337 + 13 i + seed), candidate index)), K_eval = 10 per state and chosen
candidate, horizon H = 100 env steps (the probes' horizon; the forced first action is step
0), the same full simulator restore (restore_full: qpos/qvel, cube goal mocap, settle) for
every condition. Conditions that pick the same candidate share the same rollouts (same
action, same policy => the same distribution; the rollouts are keyed by candidate).
Per rollout: success at any step (cube within 0.04 m of its goal), success at the final
step, discounted return, held over the first 20 steps and over the whole horizon (held =
phases.transport_mask: lifted > 3 cm above the table and gripper closed > 0.5; reported for
states that start in hand or carrying).

Stages:
    select      build the per-state selection table from the cached action probe
    roll        fresh evaluation rollouts for --seeds x --banks (one CSV per seed x bank)
    aggregate   per (critic, phase, condition) tables, pooled and per seed, the figures
    --smoke     2 carrying states x 2 rollouts (both critics), then aggregate that

    python scripts/probes/cube_bottleneck/intervention.py --smoke
    python scripts/probes/cube_bottleneck/intervention.py select
    python scripts/probes/cube_bottleneck/intervention.py roll --seeds 0 --banks carrying
    python scripts/probes/cube_bottleneck/intervention.py aggregate
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO = str(Path(__file__).resolve().parents[3])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

ENV = "cube-single-play-v0"
EPOCH = 1000000
ROOT = Path("analysis/failure_figs")
FIG = Path("PAPER/ICLR/figures/gap_preservation")
OUT = ROOT / ENV / "intervention"
METHODS = ["fb_tdj", "density_fb_z"]
METHOD_LABEL = {"fb_tdj": "FB", "density_fb_z": "Density-FB"}
OWN_CANDIDATE = {"fb_tdj": "fb_policy", "density_fb_z": "density_policy"}
BANKS = ["approach", "inhand", "carrying"]
BANK_IDX = {"approach": 0, "inhand": 1, "carrying": 2}
BANK_LABEL = {"approach": "approach", "inhand": "maintain", "carrying": "transport"}
BANK_COLOR = {"approach": "#CC78BC", "inhand": "#0173B2", "carrying": "#D55E00"}
HELD_BANKS = ["inhand", "carrying"]
CONDS = ["actor", "critic", "rollout"]
COND_LABEL = {"actor": "(a) actor's own action", "critic": "(b) critic-selected",
              "rollout": "(c) rollout-selected"}
K_EVAL = 10
HORIZON = 100
HELD_STEPS = 20
EVAL_KEY_BASE = 31337
SUCCESS_RADIUS = 0.04
TABLE_Z = 0.02             # banks.TABLE_Z, measured
GRIPPER_SCALER = 3.0       # restore.GRIPPER_SCALER: obs[17] / 3 = gripper opening in [0, 1]
STATE_KEYS = ["critic", "seed", "bank", "state"]
METRICS = ["success_any", "success_final", "ret", "held20", "held_all"]


# ── pure functions (unit-tested) ────────────────────────────────────────────────────────

def select_candidate(values: np.ndarray, own_idx: int, tol: float = 0.0) -> int:
    """Index of the largest value; if the actor's own candidate is within `tol` of the
    maximum (a tie), the actor's own candidate is kept (no information => no change)."""
    v = np.asarray(values, float)
    m = float(np.max(v))
    if v[own_idx] >= m - tol:
        return int(own_idx)
    return int(np.argmax(v))


def selection_table(long: pd.DataFrame, own: Dict[str, str] = OWN_CANDIDATE) -> pd.DataFrame:
    """One row per (critic, seed, bank, state): the candidate each rule picks (index and
    name), its selection-time MC return and critic value, the agreement flags, and whether
    the MC returns tie (all candidates equal -- no rollout-based decision exists)."""
    rows = []
    for (c, s, b, i), g in long.groupby(STATE_KEYS, sort=False):
        g = g.reset_index(drop=True)
        names = g["candidate"].tolist()
        q = g["q_pred"].to_numpy(float)
        r = g["return_true"].to_numpy(float)
        own_i = names.index(own[c])
        ia = own_i
        ib = select_candidate(q, own_i)
        ic = select_candidate(r, own_i)
        row = dict(critic=c, seed=int(s), bank=b, state=int(i),
                   task=int(g["task"].iloc[0]) if "task" in g else -1, n_candidates=len(g),
                   idx_actor=ia, idx_critic=ib, idx_rollout=ic,
                   cand_actor=names[ia], cand_critic=names[ib], cand_rollout=names[ic],
                   mc_actor=float(r[ia]), mc_critic=float(r[ib]), mc_rollout=float(r[ic]),
                   q_actor=float(q[ia]), q_critic=float(q[ib]), q_rollout=float(q[ic]),
                   mc_best=float(r.max()), mc_worst=float(r.min()),
                   agree_critic_actor=int(ib == ia), agree_rollout_actor=int(ic == ia),
                   agree_critic_rollout=int(ib == ic), tie_mc=int(np.ptp(r) <= 0))
        rows.append(row)
    return pd.DataFrame(rows)


def rollout_metrics(hit: np.ndarray, held: np.ndarray, gamma: float, held_steps: int = HELD_STEPS
                    ) -> Dict[str, np.ndarray]:
    """hit, held: [H, K] booleans per step and rollout (step 0 = after the forced action).
    Per rollout: success at any step, success at the last step, discounted return
    sum_t gamma^t hit_t, held at every one of the first `held_steps` steps, held at every
    step of the horizon."""
    hit = np.asarray(hit, float)
    held = np.asarray(held, bool)
    H = hit.shape[0]
    w = gamma ** np.arange(H, dtype=np.float64)
    return dict(success_any=(hit.max(0) > 0).astype(float),
                success_final=hit[-1].astype(float),
                ret=(w[:, None] * hit).sum(0),
                held20=held[:min(held_steps, H)].all(0).astype(float),
                held_all=held.all(0).astype(float))


def _se(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def aggregate_rollouts(rolls: pd.DataFrame, sel: Optional[pd.DataFrame] = None,
                       by_seed: bool = False) -> pd.DataFrame:
    """Per (critic, bank, cond[, seed]): every metric averaged first within a state (over
    its rollouts), then over states; the standard error is over states. n_states,
    n_rollouts; when `sel` is given, the fraction of states where the condition's candidate
    is the actor's own, the fraction where (b) and (c) agree, and the mean selection-time
    MC return of the chosen candidate."""
    keys = ["critic", "bank", "cond"] + (["seed"] if by_seed else [])
    per_state = rolls.groupby(STATE_KEYS + ["cond"], sort=False)[METRICS].mean().reset_index()
    counts = rolls.groupby(STATE_KEYS + ["cond"], sort=False).size().rename("k").reset_index()
    per_state = per_state.merge(counts, on=STATE_KEYS + ["cond"])
    if sel is not None:
        sel_cols = ["agree_critic_actor", "agree_rollout_actor", "agree_critic_rollout", "tie_mc",
                    "mc_actor", "mc_critic", "mc_rollout"]
        per_state = per_state.merge(sel[STATE_KEYS + sel_cols], on=STATE_KEYS, how="left")
    out = []
    for k, g in per_state.groupby(keys, sort=False):
        k = (k,) if not isinstance(k, tuple) else k
        row = dict(zip(keys, k))
        row.update(method=METHOD_LABEL.get(row["critic"], row["critic"]),
                   n_states=len(g), n_rollouts=int(g["k"].sum()))
        for m in METRICS:
            row[f"{m}_mean"] = float(g[m].mean())
            row[f"{m}_se"] = _se(g[m].to_numpy())
        if sel is not None:
            cond = row["cond"]
            row["frac_same_as_actor"] = float(g[f"agree_{cond}_actor"].mean()) if cond != "actor" else 1.0
            row["frac_critic_eq_rollout"] = float(g["agree_critic_rollout"].mean())
            row["mc_sel_mean"] = float(g[f"mc_{cond}"].mean())
            row["mc_sel_se"] = _se(g[f"mc_{cond}"].to_numpy())
            row["frac_tie_mc"] = float(g["tie_mc"].mean())
        out.append(row)
    t = pd.DataFrame(out)
    t["bank"] = pd.Categorical(t["bank"], BANKS, ordered=True)
    t["cond"] = pd.Categorical(t["cond"], CONDS, ordered=True)
    return t.sort_values(keys).reset_index(drop=True)


PAIRS = [("rollout", "critic"), ("rollout", "actor"), ("critic", "actor")]


def paired_table(rolls: pd.DataFrame, metrics: Sequence[str] = ("success_any", "held20", "ret"),
                 by_seed: bool = False) -> pd.DataFrame:
    """Per (critic, bank[, seed]) and pair (x, y) of conditions: the mean over states of the
    within-state difference metric_x - metric_y (state means over rollouts) with its s.e.
    over states, over all states and over the states where the two rules picked DIFFERENT
    candidates (`n_changed`; on the others the difference is identically 0 since the
    rollouts are shared)."""
    keys = ["critic", "bank"] + (["seed"] if by_seed else [])
    ps = rolls.groupby(STATE_KEYS + ["cond"], sort=False)[list(metrics)].mean().reset_index()
    ci = rolls.groupby(STATE_KEYS + ["cond"], sort=False)["cand_idx"].first().reset_index()
    ps = ps.merge(ci, on=STATE_KEYS + ["cond"])
    wide = ps.pivot_table(index=STATE_KEYS, columns="cond", values=list(metrics) + ["cand_idx"])
    out = []
    for k, g in wide.groupby(level=[STATE_KEYS.index(c) for c in keys], sort=False):
        k = (k,) if not isinstance(k, tuple) else k
        for x, y in PAIRS:
            row = dict(zip(keys, k))
            row.update(method=METHOD_LABEL.get(row["critic"], row["critic"]), pair=f"{x}-{y}", n=len(g))
            changed = (g[("cand_idx", x)] != g[("cand_idx", y)]).to_numpy()
            row["n_changed"] = int(changed.sum())
            for m in metrics:
                d = (g[(m, x)] - g[(m, y)]).to_numpy(float)
                row[f"{m}_diff"] = float(d.mean()); row[f"{m}_diff_se"] = _se(d)
                dc = d[changed]
                row[f"{m}_diff_changed"] = float(dc.mean()) if len(dc) else float("nan")
                row[f"{m}_diff_changed_se"] = _se(dc)
            out.append(row)
    t = pd.DataFrame(out)
    t["bank"] = pd.Categorical(t["bank"], BANKS, ordered=True)
    return t.sort_values(keys + ["pair"]).reset_index(drop=True)


# ── cached selection estimates ──────────────────────────────────────────────────────────

def load_long(root: Path = ROOT, env: str = ENV, epoch: int = EPOCH, tags: Optional[Sequence[str]] = None,
              seeds: Optional[Sequence[int]] = None, critics: Sequence[str] = METHODS) -> pd.DataFrame:
    """The action probe's per-candidate rows (q_pred, return_true); the campaign tags by
    default (`_app` = approach + inhand, `_car` = carrying), never the smoke rows."""
    fs = sorted(glob.glob(str(root / env / "actions" / f"actions_long_seed*_{epoch}*.csv")))
    if tags is None:
        fs = [f for f in fs if "smoke" not in f]
    else:
        fs = [f for f in fs if any(f.endswith(f"_{epoch}{t}.csv") for t in tags)]
    assert fs, (f"no action-probe rows under {root / env / 'actions'}; regenerate with "
                "`gap_preservation.py --probe` (failure_figs.stage_actionprobe)")
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df = df[df["bank"].isin(BANKS) & df["critic"].isin(list(critics))]
    if seeds is not None:
        df = df[df["seed"].isin(list(seeds))]
    return df.reset_index(drop=True)


def stage_select(out_dir: Path, seeds: Optional[Sequence[int]] = None, critics: Sequence[str] = METHODS) -> pd.DataFrame:
    long = load_long(seeds=seeds, critics=critics)
    n_per = long.groupby(STATE_KEYS).size()
    assert (n_per == n_per.iloc[0]).all(), "uneven candidate counts in the cached probe"
    sel = selection_table(long)
    out_dir.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out_dir / "intervention_selection.csv", index=False)
    print(f"[select] {len(sel)} (critic, seed, bank, state) rows, {int(n_per.iloc[0])} candidates each "
          f"-> {out_dir / 'intervention_selection.csv'}")
    return sel


# ── stage roll ──────────────────────────────────────────────────────────────────────────

def roll_candidates(cond_agent, envs, rec, acts, K: int, H: int, cidx: int, goal_xyz: np.ndarray,
                    keys: Sequence, table_z: float = TABLE_Z):
    """Each of the len(acts) actions once (from the same restored state), then K
    continuations of the frozen policy for H - 1 more steps, in lockstep; the actor's rng
    stream is keys[i] for action i. Returns (hit [n_acts, H, K], held [n_acts, H, K])."""
    import jax
    from scripts.probes.cube_bottleneck.failure_figs import restore_full
    from scripts.probes.cube_bottleneck.phases import PhaseThresholds, transport_mask

    thr = PhaseThresholds()
    n_acts, n = len(acts), len(acts) * K
    assert len(envs) >= n
    obs = np.zeros((n, envs[0].observation_space.shape[0]), np.float32)
    alive = np.ones(n, bool)
    cube = np.zeros((H, n, 3), np.float64)
    grip = np.zeros((H, n), np.float64)
    rngs = list(keys)

    def _record(t):
        for j in range(n):
            u = envs[j].unwrapped
            cube[t, j] = np.asarray(u._data.joint(f"object_joint_{cidx}").qpos[:3])
            grip[t, j] = obs[j, 17] / GRIPPER_SCALER
    for i in range(n_acts):
        for k in range(K):
            j = i * K + k
            restore_full(envs[j], rec)
            o, _r, term, trunc, _info = envs[j].step(np.asarray(acts[i], np.float32))
            obs[j] = np.asarray(o, np.float32)
            if term or trunc:
                alive[j] = False
    _record(0)
    for t in range(1, H):
        a = np.zeros((n, acts.shape[-1]), np.float32)
        for i in range(n_acts):
            rngs[i], key = jax.random.split(rngs[i])
            sl = slice(i * K, (i + 1) * K)
            a[sl] = np.clip(np.asarray(cond_agent.sample_actions(observations=obs[sl], goals=None,
                                                                 seed=key, temperature=0.0)), -1, 1)
        for j in range(n):
            if alive[j]:
                o, _r, term, trunc, _info = envs[j].step(a[j].astype(np.float32))
                obs[j] = np.asarray(o, np.float32)
                if term or trunc:
                    alive[j] = False
        _record(t)
    hit = np.linalg.norm(cube - goal_xyz[None, None], axis=-1) <= SUCCESS_RADIUS          # [H, n]
    held = np.stack([transport_mask(dict(cube=cube[:, j], grip=grip[:, j], table_z=table_z), thr)
                     for j in range(n)], axis=1)                                          # [H, n]
    return (hit.reshape(H, n_acts, K).transpose(1, 0, 2), held.reshape(H, n_acts, K).transpose(1, 0, 2))


def stage_roll(seed: int, banks: Sequence[str], out_dir: Path, sel: pd.DataFrame, critics: Sequence[str] = METHODS,
               max_states: int = 0, K: int = K_EVAL, H: int = HORIZON, tag: str = "") -> List[Path]:
    import jax
    from scripts.probes.cube_bottleneck import agents as A
    from scripts.probes.cube_bottleneck import failure_figs as FF
    from scripts.probes.cube_bottleneck import restore as R

    runs = {r["critic"]: r for r in A.discover_roots(ENV, METHODS, [seed], EPOCH)}
    assert set(runs) == set(METHODS), f"seed {seed}: found {sorted(runs)}"
    arms = {m: A.load_arm_eval(runs[m]["run_dir"], EPOCH) for m in METHODS}
    tasks = A.eval_tasks(arms["fb_tdj"])
    conds = {m: A.relabel_cond_agents(arms[m], tasks) for m in METHODS}
    gamma = {m: float(arms[m].config.get("discount", 0.99)) for m in METHODS}
    pool_obs = FF.build_pool(ENV, ROOT / ENV / "actions")["obs"]
    bk = np.load(ROOT / ENV / "banks" / "banks.npz")
    envs = FF.make_env_pool(ENV, len(CONDS) * K)
    R.ensure_cube_env(envs[0])
    adim = int(envs[0].action_space.shape[0])
    sel = sel[sel["seed"] == seed].set_index(["critic", "bank", "state"])
    written = []
    for bank in banks:
        t0 = time.time()
        n = len(bk[f"{bank}_qpos"]) if not max_states else min(max_states, len(bk[f"{bank}_qpos"]))
        rows, checks = [], []
        for i in range(n):
            rec = dict(idx=i, task=int(bk[f"{bank}_task"][i]), goal=bk[f"{bank}_goal"][i],
                       qpos=bk[f"{bank}_qpos"][i], qvel=bk[f"{bank}_qvel"][i],
                       grip=float(bk[f"{bank}_grip"][i]), cube=int(bk[f"{bank}_cube"][i]))
            c, tid = rec["cube"], rec["task"]
            goal_xyz = np.asarray(rec["goal"])[c]
            # the SAME candidate set as failure_figs.stage_actionprobe (same rng keys)
            ob = FF.restore_full(envs[0], rec)
            key = jax.random.PRNGKey(4242 + i * 7 + seed)
            a_fb = np.asarray(conds["fb_tdj"][tid].sample_actions(ob, goals=None, seed=key))
            a_d = np.asarray(conds["density_fb_z"][tid].sample_actions(ob, goals=None, seed=key))
            rng = np.random.default_rng([FF.BANK_SEED, seed, i, BANK_IDX[bank]])
            cands = FF.candidate_actions(a_fb.astype(np.float32), a_d.astype(np.float32), adim, rng)
            obs_rep = np.repeat(ob[None], len(cands), axis=0)
            for m in critics:
                s = sel.loc[(m, bank, i)]
                arm, ag = arms[m], conds[m][tid]
                z = np.asarray(ag.cond_eval["z"])
                # identity check of the regenerated candidates against the cached readout
                q = FF.q_values(arm, obs_rep, cands, z, pool_obs)
                q_cached = np.array([s["q_actor"], s["q_critic"], s["q_rollout"]])
                q_regen = q[[int(s["idx_actor"]), int(s["idx_critic"]), int(s["idx_rollout"])]]
                qdev = float(np.max(np.abs(q_regen - q_cached)) / max(float(np.ptp(q)), 1e-9))
                checks.append(dict(critic=m, seed=seed, bank=bank, state=i, q_rel_dev=qdev))
                chosen = {cd: int(s[f"idx_{cd}"]) for cd in CONDS}
                uniq = sorted(set(chosen.values()))
                keys = [jax.random.fold_in(jax.random.PRNGKey(EVAL_KEY_BASE + i * 13 + seed), ci) for ci in uniq]
                hit, held = roll_candidates(ag, envs, rec, cands[uniq], K, H, c, goal_xyz, keys)
                per_cand = {ci: rollout_metrics(hit[u], held[u], gamma[m]) for u, ci in enumerate(uniq)}
                for cd in CONDS:
                    ci = chosen[cd]
                    met = per_cand[ci]
                    for k in range(K):
                        rows.append(dict(critic=m, seed=seed, bank=bank, state=i, task=tid, cond=cd,
                                         cand_idx=ci, cand=FF.CANDIDATE_NAMES[ci], k=k, K=K, H=H,
                                         gamma=gamma[m], **{mm: float(met[mm][k]) for mm in METRICS}))
            if (i + 1) % 8 == 0 or i + 1 == n:
                df = pd.DataFrame(rows)
                msg = " | ".join(f"{METHOD_LABEL[m]} " + " ".join(
                    f"{cd[:4]} {df[(df.critic == m) & (df.cond == cd)]['success_any'].mean():.2f}" for cd in CONDS)
                    for m in critics)
                print(f"  [int seed{seed}] {bank} {i + 1}/{n} ({time.time() - t0:.0f}s) {msg}", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        f = out_dir / f"rollouts_seed{seed}_{bank}{tag}.csv"
        pd.DataFrame(rows).to_csv(f, index=False)
        pd.DataFrame(checks).to_csv(out_dir / f"qcheck_seed{seed}_{bank}{tag}.csv", index=False)
        written.append(f)
        print(f"[int] seed {seed} {bank}: {len(rows)} rollout rows in {time.time() - t0:.0f}s -> {f}", flush=True)
    return written


# ── stage aggregate ─────────────────────────────────────────────────────────────────────

def load_rollouts(out_dir: Path, tag: Optional[str]) -> pd.DataFrame:
    fs = sorted(glob.glob(str(out_dir / "rollouts_seed*.csv")))
    fs = [f for f in fs if (("smoke" not in f) if tag is None else f.endswith(f"{tag}.csv"))]
    assert fs, f"no rollouts under {out_dir}"
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def plot_by_phase(t: pd.DataFrame, ts: pd.DataFrame, critic: str, metric: str, ylabel: str, out: Path,
                  banks: Sequence[str] = BANKS, ylim=(0, 1)) -> None:
    """Bars per phase (phase colour), three conditions per phase (increasing opacity + hatch),
    s.e. over states, faint per-seed markers, legend inside. 3.2 x 2.3 in, saved at the exact
    figure size (no bbox_inches)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    INK = "#2a2a2a"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "text.color": INK, "axes.edgecolor": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "hatch.linewidth": 0.5})
    STYLE = {"actor": dict(alpha=0.35, hatch=None), "critic": dict(alpha=0.65, hatch="////"),
             "rollout": dict(alpha=1.0, hatch=None)}
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    w = 0.26
    tt = t[t["critic"] == critic]
    tss = ts[ts["critic"] == critic]
    for i, bank in enumerate(banks):
        ax.axvspan(i - 0.5, i + 0.5, color=BANK_COLOR[bank], alpha=0.07, zorder=0)
        for j, cd in enumerate(CONDS):
            r = tt[(tt["bank"] == bank) & (tt["cond"] == cd)]
            if r.empty:
                continue
            r = r.iloc[0]
            x = i + (j - 1) * w
            ax.bar(x, r[f"{metric}_mean"], w * 0.92, color=BANK_COLOR[bank], alpha=STYLE[cd]["alpha"],
                   hatch=STYLE[cd]["hatch"], edgecolor="white", linewidth=0.5, zorder=2)
            ax.errorbar(x, r[f"{metric}_mean"], yerr=r[f"{metric}_se"], fmt="none", ecolor=INK,
                        elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=4)
            ps = tss[(tss["bank"] == bank) & (tss["cond"] == cd)]
            ax.scatter(np.full(len(ps), x), ps[f"{metric}_mean"], s=7, color=INK, alpha=0.45, linewidths=0, zorder=3)
    ax.set_xticks(range(len(banks)))
    ax.set_xticklabels([BANK_LABEL[b] for b in banks])
    for tick, b in zip(ax.get_xticklabels(), banks):
        tick.set_color(BANK_COLOR[b])
    ax.set_xlim(-0.5, len(banks) - 0.5)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_xlabel("phase of the restored state", fontsize=8.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    handles = [Patch(facecolor="#7a7a7a", alpha=STYLE[cd]["alpha"], hatch=STYLE[cd]["hatch"], edgecolor="white",
                     label=COND_LABEL[cd]) for cd in CONDS]
    ax.legend(handles=handles, fontsize=6.5, frameon=False, loc="upper left", handlelength=1.2, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    print(f"[fig] {out}.pdf/.png")


def stage_aggregate(out_dir: Path, fig_dir: Path, tag: Optional[str] = None, sel: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    rolls = load_rollouts(out_dir, tag)
    if sel is None:
        sel = pd.read_csv(fig_dir / "intervention_selection.csv") if (fig_dir / "intervention_selection.csv").exists() \
            else stage_select(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    t = aggregate_rollouts(rolls, sel)
    ts = aggregate_rollouts(rolls, sel, by_seed=True)
    t.to_csv(fig_dir / "intervention_by_phase.csv", index=False)
    ts.to_csv(fig_dir / "intervention_by_phase_per_seed.csv", index=False)
    per_state = rolls.groupby(STATE_KEYS + ["cond", "cand_idx", "cand"], sort=False)[METRICS].mean().reset_index()
    per_state.to_csv(fig_dir / "intervention_per_state.csv", index=False)
    # candidate-identity check of the regenerated candidate set against the cached readout
    qs = sorted(glob.glob(str(out_dir / "qcheck_seed*.csv")))
    qs = [f for f in qs if (("smoke" not in f) if tag is None else f.endswith(f"{tag}.csv"))]
    if qs:
        qc = pd.concat([pd.read_csv(f) for f in qs], ignore_index=True)
        chk = dict(n=int(len(qc)), q_rel_dev_max=float(qc["q_rel_dev"].max()), q_rel_dev_mean=float(qc["q_rel_dev"].mean()))
        (fig_dir / "intervention_qcheck.json").write_text(json.dumps(chk, indent=2))
        print(f"=== candidate identity check (|q_regen - q_cached| / range(q), per state): {chk}")
    cols = ["method", "bank", "cond", "n_states", "n_rollouts", "success_any_mean", "success_any_se",
            "success_final_mean", "ret_mean", "ret_se", "held20_mean", "held20_se", "held_all_mean",
            "frac_same_as_actor", "frac_critic_eq_rollout", "mc_sel_mean", "frac_tie_mc"]
    print("=== intervention by phase (pooled over seeds; s.e. over states) ===")
    print(t[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("=== per seed ===")
    print(ts[["method", "seed", "bank", "cond", "n_states", "success_any_mean", "held20_mean", "ret_mean"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    pt = paired_table(rolls)
    pts = paired_table(rolls, by_seed=True)
    pt.to_csv(fig_dir / "intervention_paired.csv", index=False)
    pts.to_csv(fig_dir / "intervention_paired_per_seed.csv", index=False)
    print("=== paired within-state differences (x - y; s.e. over states; 'changed' = the two rules picked different candidates) ===")
    print(pt[["method", "bank", "pair", "n", "n_changed", "success_any_diff", "success_any_diff_se",
              "success_any_diff_changed", "success_any_diff_changed_se", "held20_diff", "held20_diff_se",
              "ret_diff", "ret_diff_se"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    picks = sel.groupby(["critic", "bank"])[["cand_critic", "cand_rollout"]].agg(lambda s: s.value_counts().to_dict())
    picks.to_json(fig_dir / "intervention_picks.json", indent=2)
    print("=== which candidate each rule picks (counts per critic, bank) ===")
    for (c, b), r in picks.iterrows():
        print(f"  {METHOD_LABEL.get(c, c)} {b}: critic {r['cand_critic']}\n      rollout {r['cand_rollout']}")
    for m in t["critic"].unique():
        suffix = "" if m == "fb_tdj" else "_density"
        plot_by_phase(t, ts, m, "success_any", "success after intervention", fig_dir / f"fig_intervention_by_phase{suffix}")
        hb = [b for b in HELD_BANKS if b in set(t[t["critic"] == m]["bank"].astype(str))]
        if hb:
            plot_by_phase(t, ts, m, "held20", f"cube still held after {HELD_STEPS} steps",
                          fig_dir / f"fig_intervention_held_by_phase{suffix}", banks=hb)
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=["select", "roll", "aggregate"], default="aggregate")
    ap.add_argument("--smoke", action="store_true", help="2 carrying states x 2 rollouts, then aggregate")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--banks", nargs="+", default=BANKS)
    ap.add_argument("--critics", nargs="+", default=METHODS)
    ap.add_argument("--max-states", type=int, default=0)
    ap.add_argument("--K", type=int, default=K_EVAL)
    ap.add_argument("--H", type=int, default=HORIZON)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--fig", default=str(FIG))
    args = ap.parse_args()
    out, fig = Path(args.out), Path(args.fig)
    if args.smoke:
        sel = stage_select(fig / "smoke_int", seeds=[args.seeds[0]])
        stage_roll(args.seeds[0], ["carrying"], out, sel, critics=args.critics, max_states=2, K=2, H=args.H, tag="_smoke")
        stage_aggregate(out, fig / "smoke_int", tag="_smoke", sel=sel)
        return 0
    if args.stage == "select":
        stage_select(fig, critics=args.critics)
    elif args.stage == "roll":
        sel = stage_select(fig, critics=args.critics)
        for s in args.seeds:
            stage_roll(s, args.banks, out, sel, critics=args.critics, max_states=args.max_states,
                       K=args.K, H=args.H, tag=args.tag)
    else:
        stage_aggregate(out, fig, tag=args.tag or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
