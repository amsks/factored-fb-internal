"""Puzzle probe: at each evaluation task's start state, how much mass does the trained
Density-FB critic put on batch candidates whose nine buttons match the goal, and does that
mass depend on the action? The actor scores an action by  J(a) = sum_u w(u|s,a,g) r_g(u)
with r_g = 1 on exact-match candidates, so a flat J across actions means the actor has no
signal and falls back to behaviour cloning. Also scored: goals 1, 2 and 3 presses away from
the start (built from dataset states with the required button configuration).
Usage: python puzzle_probe.py --run_dir DIR --epoch 500000 --out out.json"""
import argparse, json, os, sys, itertools
import numpy as np
sys.path.insert(0, os.getcwd())
ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True); ap.add_argument("--epoch", type=int, required=True)
ap.add_argument("--n_cand", type=int, default=1024); ap.add_argument("--n_act", type=int, default=32)
ap.add_argument("--n_draws", type=int, default=4); ap.add_argument("--kernel", action="store_true", help="match = the critic's own goal kernel (any env) instead of exact button config"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
a = ap.parse_args()
import jax, jax.numpy as jnp
from env import get_adapter
from impls.agents import Agent
from impls.utils.datasets import make_dataset
from impls.utils.flax_utils import restore_agent
from impls.critics.density_fb import _delta_phi, _raw_block_cols

rng = np.random.default_rng(a.seed)
meta = json.load(open(os.path.join(a.run_dir, "config.json"))); cfg = dict(meta["config"])
adapter = get_adapter(cfg); env, train_ds, _ = adapter.make_env_and_datasets(cfg)
ds = make_dataset(train_ds, cfg); ex = ds.sample(1)
agent = Agent.create(meta["critic"], meta["actor"], meta["seed"], jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]), cfg)
agent = restore_agent(agent, a.run_dir, a.epoch)
critic = agent.critic; net = {g: agent.net[g].params for g in agent.net}
obs_all = np.asarray(train_ds["observations"], np.float32); term = np.asarray(train_ds["terminals"]).astype(bool)
nxt_pool = np.flatnonzero(~term) + 1
def config_of(o): return (o[..., [20 + 4 * i for i in range(9)]] > 0.5).astype(int)   # one-hot "state 1" column per button
if not a.kernel: cfg_all = config_of(obs_all); key_all = cfg_all @ (2 ** np.arange(9))

def toggle(s, i):
    s = s.copy(); r, c = divmod(i, 3)
    for dr, dc in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < 3 and 0 <= cc < 3: s[rr * 3 + cc] ^= 1
    return s
def presses_needed(s, g):
    for n in range(10):
        for comb in itertools.combinations(range(9), n):
            t = s.copy()
            for i in comb: t = toggle(t, i)
            if (t == g).all(): return n
    return -1
def state_with_config(c):
    idx = np.flatnonzero(key_all == int(c @ (2 ** np.arange(9))))
    return obs_all[rng.choice(idx)]

adim = ex["actions"].shape[-1]
def readout(s0, g, U, acts):
    """w[a, m] = critic's predicted mass on candidate m from (s0, a, g)."""
    sb = jnp.asarray(np.repeat(s0[None], len(acts), 0)); gb = jnp.asarray(np.repeat(g[None], len(acts), 0)); ab = jnp.asarray(acts); Uj = jnp.asarray(U)
    f = critic._F(net["density_f"], critic._sobs(net, sb), ab, gb)
    b = critic._B(net["density_b"], critic._b_query(net, Uj, Uj))
    return np.asarray(jax.nn.softmax(jnp.einsum("hbd,md->hbm", f, b) / critic.temp, -1).mean(0))

def score(s0, g, label):
    gcfg = config_of(g) if not a.kernel else None; res = dict(label=label)
    if not a.kernel: res.update(presses=int(presses_needed(config_of(s0), gcfg)), hamming=int((config_of(s0) != gcfg).sum()))
    J_act, J_rnd, J_ds, nmatch, self_mass = [], [], [], [], []
    for d in range(a.n_draws):
        U = obs_all[rng.choice(nxt_pool, a.n_cand, replace=False)]
        if a.kernel: r = np.asarray(_delta_phi(jnp.asarray(g[None]), _raw_block_cols(jnp.asarray(U), critic.blocks), critic.blocks))[0].astype(np.float32)
        else: r = (config_of(U) == gcfg).all(1).astype(np.float32)
        nmatch.append(float(r.sum()))
        seeds = jax.random.split(jax.random.PRNGKey(1000 * d + a.seed), a.n_act)
        act_a = np.stack([np.asarray(agent.sample_actions(s0, goals=g, seed=k, temperature=1.0)) for k in seeds])
        act_r = rng.uniform(-1, 1, (a.n_act, adim)).astype(np.float32)
        act_d = np.asarray(train_ds["actions"][rng.choice(len(obs_all), a.n_act)], np.float32)
        for acts, store in ((act_a, J_act), (act_r, J_rnd), (act_d, J_ds)):
            w = readout(s0, g, U, acts); store.append(w @ r)
            if store is J_act: self_mass.append((w @ (config_of(U) == config_of(s0)).all(1)).mean() if not a.kernel else float((w @ np.asarray(_delta_phi(jnp.asarray(s0[None]), _raw_block_cols(jnp.asarray(U), critic.blocks), critic.blocks))[0]).mean()))
    for k, v in (("J_actor", J_act), ("J_random", J_rnd), ("J_dataset_actions", J_ds)):
        v = np.concatenate(v); res[k] = dict(mean=float(v.mean()), max=float(v.max()), min=float(v.min()), std=float(v.std()))
    res["n_match_per_batch"] = float(np.mean(nmatch)); res["uniform_level"] = float(np.mean(nmatch)) / a.n_cand
    res["mass_on_start_config"] = float(np.mean(self_mass))
    return res

out = dict(run_dir=a.run_dir, epoch=a.epoch, tasks=[], near=[])
for tid in range(1, 6):
    o0, info = env.reset(options=dict(task_id=tid)); g = np.asarray(info["goal"], np.float32); o0 = np.asarray(o0, np.float32)
    out["tasks"].append(score(o0, g, f"task{tid}")); print(json.dumps(out["tasks"][-1]), flush=True)
# near goals from task 1's start (all buttons off): 1, 2, 3 presses away (puzzle only)
if a.kernel: json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out); sys.exit(0)
o0, _ = env.reset(options=dict(task_id=1)); o0 = np.asarray(o0, np.float32); c0 = config_of(o0)
for n in (1, 2, 3):
    for trial in range(2):
        comb = rng.choice(9, n, replace=False); c = c0.copy()
        for i in comb: c = toggle(c, i)
        g = state_with_config(c); out["near"].append(score(o0, g, f"{n}-press goal {tuple(int(i) for i in comb)}")); print(json.dumps(out["near"][-1]), flush=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out)
