"""OOD-generalization grid, jax side. One row of the manifest -> one result JSON.

Three cells, all sharing byte-identical start states and goals (cells.npz):
  C1  dataset start (on-manifold)  + hindsight goal      "can it do the task at all?"
  C2  dataset start (on-manifold)  + EVAL task goal      C1->C2 isolates the GOAL shift
  C3  EVAL reset start (OOD)       + EVAL task goal      C2->C3 isolates the START shift

PROTOCOL (matched to the env's own eval, so C3 reproduces published success):
  * horizon H=200            = the OGBench cube episode length
  * success = ||cube_xyz - goal_cube_xyz|| <= 0.04 m   = cube_env.py:715, verbatim
  * batched: N envs stepped in lockstep, ONE model call per timestep (an unbatched
    version of this was latency-bound: 67 min wall for 4 min of CPU).

Fail-loud: a row that cannot be loaded writes an ERROR record. Nothing is silently skipped.
"""
import argparse, json, os, sys, time
import numpy as np

R = '/home/adityamohan/git/Factored-FB'
for p_ in (R, f'{R}/scripts', f'{R}/third_party/ogbench/impls', f'{R}/tools/wandb_mode_shim'):
    sys.path.insert(0, p_)

SP = f'{R}/analysis/oodgrid'
TASKS = [f'cube-single-play-singletask-task{i}-v0' for i in range(1, 6)]
H = 200                     # OGBench cube episode length
THR = 0.4                   # obs units; 1 u = 10 cm  ->  0.04 m = the env's own threshold

ap = argparse.ArgumentParser()
ap.add_argument('--manifest', default=f'{SP}/manifest.json')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--episodes', type=int, default=100)
a = ap.parse_args()

import jax, jax.numpy as jnp, mujoco, ogbench
from utils.flax_utils import restore_agent

M = json.load(open(a.manifest))
rows = [r for r in M['rows'] if r['framework'] == 'jax']
rows = [r for i, r in enumerate(rows) if i % a.nshards == a.shard]
os.makedirs(f'{SP}/results', exist_ok=True)

C = np.load(f'{SP}/cells.npz')
N = min(a.episodes, len(C['ds_obs']))
OD = C['ds_obs'].shape[1]
# The full 2x2: {in-distribution start, OOD start} x {in-distribution goal, EVAL-task goal}.
# C4 is what separates a main effect from an interaction, so it is not optional.
CELLS = [('C1', 'ds', 'goal_hind'),    # ID start, ID goal      -- can it do the task at all?
         ('C2', 'ds', 'goal_eval'),    # ID start, EVAL goal    -- the goal shift alone
         ('C4', 'er', 'goal_hind'),    # OOD start, ID goal     -- the start shift alone
         ('C3', 'er', 'goal_eval')]    # OOD start, EVAL goal   -- BOTH = the real eval

pool = [ogbench.make_env_and_datasets(TASKS[j % 5], env_only=True) for j in range(N)]
for e in pool:
    e.reset(seed=0)


# Registry, not a hand-maintained allowlist: an unknown agent_name must FAIL LOUDLY,
# never be quietly skipped. Class names verified from source (isfb_flow exports ISFBAgent,
# NOT ISFBFlowAgent). Every one of these exposes
# sample_actions(observations, goals, seed, temperature).
JAX_AGENTS = {
    'gciql_flow':  ('gciql_flow',      'GCIQLFlowAgent'),
    'gciql':       ('agents.gciql',    'GCIQLAgent'),
    'isfb_flow':   ('isfb_flow',       'ISFBAgent'),
    'gcivl':       ('agents.gcivl',    'GCIVLAgent'),
    'gcivl_flow':  ('gcivl_flow',      'GCIVLFlowAgent'),
    'hiql':        ('agents.hiql',     'HIQLAgent'),
    'hiql_flow':   ('hiql_flow',       'HIQLFlowAgent'),
    'crl':         ('agents.crl',      'CRLAgent'),
    'crl_flowbc':  ('crl_flowbc',      'CRLFlowBCAgent'),
    'gcbc':        ('agents.gcbc',     'GCBCAgent'),
    'qrl':         ('agents.qrl',      'QRLAgent'),
}


def load(r):
    import importlib
    ag = r['cfg']
    name = ag.get('agent_name', 'gciql_flow')
    if name not in JAX_AGENTS:
        raise ValueError(f'unhandled agent_name={name!r} -- add it to JAX_AGENTS')
    mod, cls = JAX_AGENTS[name]
    m = importlib.import_module(mod)
    A = getattr(m, cls)
    get_config = m.get_config
    cfg = dict(get_config())
    cfg.update({k: v for k, v in ag.items() if k in cfg})
    cfg['encoder'] = None
    if 'frame_stack' in cfg:
        cfg['frame_stack'] = None
    agent = A.create(0, jnp.zeros((1, OD), jnp.float32), jnp.zeros((1, 5), jnp.float32), cfg)
    return restore_agent(agent, r['dir'], r['step'])


for r in rows:
    tag = f"{r['group']}__{r['seed']}".replace('/', '_')
    out = f'{SP}/results/{tag}.json'
    rec = {**{k: r[k] for k in ('framework', 'group', 'seed', 'dir', 'step',
                                'agent_name', 'value_arch', 'value_latent_dim',
                                'actor_kind', 'actor_cond')},
           'episodes': N, 'H': H}
    if os.path.exists(out):
        prev = json.load(open(out))
        if 'error' in prev and 'ISFB' not in r['group']:
            continue                                   # a real out-of-scope error; leave it
        rec.update({k: v for k, v in prev.items() if k.startswith('C')})
    todo = [c for c in CELLS if c[0] not in rec]       # only compute MISSING cells
    if not todo:
        continue
    t0 = time.time()
    try:
        agent = load(r)
    except Exception as e:
        rec['error'] = f'{type(e).__name__}: {e}'
        json.dump(rec, open(out, 'w'), indent=1)
        print(f'[ERR ] {tag}: {rec["error"]}', flush=True)
        continue

    for cname, sk, gk in todo:
        qpos, qvel = C[f'{sk}_qpos'][:N], C[f'{sk}_qvel'][:N]
        ob = C[f'{sk}_obs'][:N].copy()
        G = C[gk][:N]
        for j in range(N):
            u = pool[j].unwrapped
            u._data.qpos[:] = qpos[j]
            u._data.qvel[:] = qvel[j]
            mujoco.mj_forward(u._model, u._data)
        gj = jnp.asarray(G)
        hit = np.zeros(N, bool); got = np.zeros(N, bool); done = np.zeros(N, bool)
        for t in range(H):
            A_ = np.clip(np.asarray(agent.sample_actions(
                observations=jnp.asarray(ob), goals=gj,
                seed=jax.random.PRNGKey(t), temperature=0.0)), -1, 1)
            for j in range(N):
                if done[j]:
                    continue
                o2, *_ = pool[j].step(A_[j])
                ob[j] = o2
                got[j] |= o2[18] > 0.5
                if np.linalg.norm(o2[19:22] - G[j][19:22]) < THR:
                    hit[j] = True; done[j] = True
            if done.all():
                break
        rec[cname] = {'success': float(hit.mean()), 'grasped': float(got.mean())}
    rec['secs'] = round(time.time() - t0, 1)
    json.dump(rec, open(out, 'w'), indent=1)
    print(f"[ok  ] {tag}  C1={rec['C1']['success']:.3f} C2={rec['C2']['success']:.3f} "
          f"C3={rec['C3']['success']:.3f}  ({rec['secs']}s)", flush=True)
print('SHARD DONE', a.shard, flush=True)
