"""OOD-generalization grid, torch/FB side. Same 3 cells, same protocol as the jax runner.

FB has TWO ways to get its task vector, and they are NOT interchangeable (measured:
0.185 vs 0.390 on the same cell). Conflating them would confound the start-shift effect
with the z-route, so BOTH are run and reported separately:

  zB  z = project_z(B(g))         the direct analogue of GCIQL's psi(g). Defined for every
                                  cell (incl. C1's hindsight goals). This is the cell that
                                  is matched to GCIQL.
  zR  z = reward-inference        FB's DEPLOYED eval path (relabel the offline buffer with
                                  the privileged task reward, z = E[r * B(s')]). Only defined
                                  for the EVAL-task goals, so C2/C3 only.

Protocol identical to oodgrid_run_jax.py: H=200, success = ||cube - goal_cube|| <= 0.04 m
(cube_env.py:715), batched (one act() call per timestep).
"""
import argparse, json, os, sys, time
import numpy as np
import torch

R = '/home/adityamohan/git/Factored-FB'
sys.path.insert(0, R)
import mujoco, ogbench
from evals.analysis import load_cfg, build_env_and_agent, load_checkpoint

SP = f'{R}/analysis/oodgrid'
TASKS = [f'cube-single-play-singletask-task{i}-v0' for i in range(1, 6)]
H, THR = 200, 0.4

ap = argparse.ArgumentParser()
ap.add_argument('--manifest', default=f'{SP}/manifest.json')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--episodes', type=int, default=100)
ap.add_argument('--device', default='cuda')
a = ap.parse_args()

M = json.load(open(a.manifest))
# Verified from source:
#   fb / fb_flowbc / isfb_flowbc  -> agents/fb/*  : act(obs, z) AND model.backward_map  => BOTH z-routes
#   psm / psm_flowbc              -> agents/psm/* : act(obs, z) but NO backward_map     => zR (eval-goal) cells ONLY
#   rldp / rldp_flowbc            -> no agent-level act()                               => cannot be driven
#   tdmpc2                        -> planner, different API                             => cannot be driven
# Anything not listed FAILS LOUDLY with a recorded reason; nothing is silently dropped.
BOTH_Z = {'fb', 'fb_flowbc', 'isfb_flowbc'}
ZR_ONLY = {'psm', 'psm_flowbc'}
UNSUPPORTED = {'rldp', 'rldp_flowbc', 'tdmpc2', 'crl', 'crl_flowbc'}
SMOKE = ('smoke',)          # never-trained stubs

rows = [r for r in M['rows'] if r['framework'] == 'torch'
        and not any(t in r['group'] for t in SMOKE)]
rows = [r for i, r in enumerate(rows) if i % a.nshards == a.shard]
os.makedirs(f'{SP}/results', exist_ok=True)

C = np.load(f'{SP}/cells.npz')
N = min(a.episodes, len(C['ds_obs']))
# Full 2x2: {ID start, OOD start} x {ID goal, EVAL goal}. C4 separates main effect
# from interaction, so it is not optional.
CELLS = [('C1', 'ds', 'goal_hind'), ('C2', 'ds', 'goal_eval'),
         ('C4', 'er', 'goal_hind'), ('C3', 'er', 'goal_eval')]
pool = [ogbench.make_env_and_datasets(TASKS[j % 5], env_only=True) for j in range(N)]
for e in pool:
    e.reset(seed=0)
slot_task = np.array([j % 5 for j in range(N)])

# The offline buffer is needed for FB's reward-inference z. Load it ONCE per shard
# (it is ~1M transitions); every row in the shard reuses it.
from data.ogbench import load_ogbench_dataset
_c0 = load_cfg(f"{rows[0]['dir']}/.hydra/config.yaml", device=a.device) if rows else None
BUFFER = None
if _c0 is not None:
    BUFFER = load_ogbench_dataset(
        domain=_c0.domain, data_path=_c0.data_path,
        load_n_episodes=_c0.load_n_episodes, device=a.device,
        n_transitions=_c0.n_transitions, obs_type=_c0.obs_type,
        frame_stack=int(getattr(_c0, 'frame_stack', 1)),
    )
    print(f'[buffer] loaded for reward-inference z ({_c0.data_path})', flush=True)


def rollout(agent, z_per_slot, sk, gk):
    qpos, qvel = C[f'{sk}_qpos'][:N], C[f'{sk}_qvel'][:N]
    ob = C[f'{sk}_obs'][:N].copy()
    G = C[gk][:N]
    for j in range(N):
        u = pool[j].unwrapped
        u._data.qpos[:] = qpos[j]
        u._data.qvel[:] = qvel[j]
        mujoco.mj_forward(u._model, u._data)
    hit = np.zeros(N, bool); got = np.zeros(N, bool); done = np.zeros(N, bool)
    for t in range(H):
        with torch.no_grad():
            A_ = agent.act(torch.as_tensor(ob, device=a.device),
                           z_per_slot, eval_mode=True).cpu().numpy()
        A_ = np.clip(A_, -1, 1)
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
    return {'success': float(hit.mean()), 'grasped': float(got.mean())}


for r in rows:
    tag = f"{r['group']}__{r['seed']}".replace('/', '_')
    out = f'{SP}/results/{tag}.json'
    rec = {k: r[k] for k in ('framework', 'group', 'seed', 'dir', 'agent_name')}
    rec.update(episodes=N, H=H, ckpt=r['ckpt'])
    if os.path.exists(out):
        prev = json.load(open(out))
        rec.update({k: v for k, v in prev.items() if k.startswith('C') or k == 'zR_error'})
    _need_zb = r.get('agent_name') not in ZR_ONLY
    if ((not _need_zb or all(f'{c}_zB' in rec for c, _, _ in CELLS))
            and 'C2_zR' in rec and 'C3_zR' in rec):
        continue
    t0 = time.time()
    try:
        cfg = load_cfg(f"{r['dir']}/.hydra/config.yaml", device=a.device)
        env0, agent, _, _ = build_env_and_agent(cfg)
        load_checkpoint(agent, r['ckpt'], map_location=a.device)
        agent.model.eval()
    except Exception as e:
        rec['error'] = f'{type(e).__name__}: {e}'
        json.dump(rec, open(out, 'w'), indent=1)
        print(f'[ERR ] {tag}: {rec["error"]}', flush=True)
        continue

    kind = r.get('agent_name')
    if kind in UNSUPPORTED:
        rec['error'] = f"unsupported torch agent for this harness: {kind} (no act(obs,z))"
        json.dump(rec, open(out, 'w'), indent=1)
        print(f'[SKIP] {tag}: {rec["error"]}', flush=True)
        continue

    # --- zB: z = project_z(B(g)), per goal. Matched to GCIQL's psi(g). ---
    zb_cells = [] if kind in ZR_ONLY else [c for c in CELLS if f'{c[0]}_zB' not in rec]
    if kind in ZR_ONLY:
        rec['zB_note'] = f'{kind} has no backward_map; zB cells (C1/C4) not defined'
    for cname, sk, gk in zb_cells:
        with torch.no_grad():
            z = agent.model.project_z(agent.model.backward_map(
                torch.as_tensor(C[gk][:N], device=a.device)))
        rec[f'{cname}_zB'] = rollout(agent, z, sk, gk)

    # --- zR: FB's DEPLOYED reward-inference z, per EVAL TASK (C2/C3 only) ---
    # z = E_rho[r_task(s') * B(s')] over a relabelled subsample of the offline buffer.
    # Defined only for the 5 eval tasks, so it cannot score C1's hindsight goals.
    try:
        from evals.ogbench import OGBenchEvaluator
        ev = OGBenchEvaluator(
            domain=cfg.domain, agent=agent, offline_buffer=BUFFER,
            relabel_size=cfg.eval_relabel_size, n_episodes=1,
            shift_reward=cfg.eval_shift_reward, obs_type=cfg.obs_type,
            frame_stack=int(getattr(cfg, 'frame_stack', 1)),
            seed=cfg.seed, device=a.device, save_videos=False,
            video_dir=None, use_wandb=False,
        )
        z_task = torch.stack([ev._infer_z(t)[0].reshape(-1) for t in TASKS])  # [5, z_dim]
        z_slots = z_task[torch.as_tensor(slot_task, device=z_task.device)]    # [N, z_dim]
        for cname, sk, gk in (('C2', 'ds', 'goal_eval'), ('C4', 'er', 'goal_hind'),
                              ('C3', 'er', 'goal_eval')):
            if f'{cname}_zR' in rec or gk == 'goal_hind':
                continue          # zR is only defined for EVAL-task goals
            rec[f'{cname}_zR'] = rollout(agent, z_slots.to(a.device), sk, gk)
    except Exception as e:
        rec['zR_error'] = f'{type(e).__name__}: {e}'
        print(f'[zR!! ] {tag}: {rec["zR_error"]}', flush=True)

    rec['secs'] = round(time.time() - t0, 1)
    json.dump(rec, open(out, 'w'), indent=1)
    c1 = rec.get('C1_zB', {}).get('success', float('nan'))
    print(f"[ok  ] {tag}  zB:C1={c1:.3f} "
          f"zR:C3={rec.get('C3_zR', {}).get('success', float('nan')):.3f} "
          f"({rec['secs']}s)", flush=True)
print('SHARD DONE', a.shard, flush=True)
