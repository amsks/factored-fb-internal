"""Behavior-sample reranking audit for the torch FB successor-measure critic on OGBench cube.

Torch/FB analogue of scripts/atlas/rerank_audit_multienv.py (which is jax/GCIQL). Tests whether
deploying the FB critic's action ORDERING as the policy produces a strong policy:

  per env step, sample N candidate actions a_i ~ mu(.|s, z_g) from FB's distilled flow-BC actor
  (agent.model.act, fresh noise per draw), execute argmax_i Q_FB(s, a_i, z_g), where
      Q_FB(s,a,z) = <F(s,a,z), z>  (agent._fb_Q_heads, pessimistic min over the 2 forward heads)
  and z_g is the FB task vector inferred ONCE per task.

Interpretation:
  fb_rerank rises with N toward trained_fb  -> policy EXTRACTION was the bottleneck
                                               (FB's representation ordered actions well).
  fb_rerank stays low / flat                -> FB's REPRESENTATION failed to learn the
                                               policy-relevant action differences.

z-inference: the *exact* FB-eval reward-relabel inference — OGBenchEvaluator._infer_z(task):
  relabel a large offline sub-batch with the task's cube_reward_fn, shift by +1, then
  z = project(sum_i r_i * B(next_obs_i)) via model.reward_inference. This is the same z the
  published FB eval uses (NOT the sparse _TorchFB hit-indicator z in critics.py).

obs normalization: FB's obs_normalizer is Identity for this state run (no obs_normalizer in the
  hydra config). The actor (model.act) and the critic left-encoder both route obs through
  model._normalize -> model._fw_encoder (both Identity here) exactly as training/eval does, so
  raw env obs are fed and normalization is consistent by construction.

Anchors (N-independent, computed once):
  floor      = N=1 rerank = a single flow-BC sample (== the FB policy itself).
  trained_fb = the FB policy via agent.act(obs, z_g)  (model.act mean flag is ignored; the
               flow-BC actor is inherently stochastic). MUST reproduce FB cube ~0.40-0.45.

Run under the repo torch venv: /home/adityamohan/git/Factored-FB/.venv/bin/python
(FB is torch, not jax). CUDA is optional (set CUDA_VISIBLE_DEVICES and --device cuda).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

# CHECKPOINT NOTE (important):
#   The run dir named in the task prompt (2026-05-14_21-34-52...) and hard-coded in
#   scripts/atlas/critics.py::_load_fb is a 100-TRAIN-STEP SMOKE stub (num_train_steps=100,
#   load_n_episodes=4); its final.pt is untrained and scores 0.0 (verified).
#   The published FB cube ~0.42 came from a 1M-step STATE fb_flowbc run whose checkpoints
#   lived on volatile /dev/shm and no longer exist on disk (the 06-04 runs have empty
#   checkpoints/ dirs). A full-disk search shows the ONLY surviving genuine plain-FB
#   (agent=fb_flowbc) STATE checkpoint is the 60k-step run below (every 500k state run on
#   disk is psm_flowbc / rldp_flowbc / tdmpc2 -- different agents, not plain FB).
#   This 60k checkpoint's TRUE success is ~0.10, verified by running the canonical
#   OGBenchEvaluator.run() on it (0.10) vs this harness's trained_fb anchor (0.12) -- a
#   match within seed noise, so the harness is faithful; the sub-0.42 value is purely the
#   under-trained checkpoint, not a wiring bug. Point --run/--ckpt at a 1M state fb_flowbc
#   checkpoint if one is ever restored.
FB_RUN = ("/home/adityamohan/bridge_ckpt_backup/factored-fb_outputs/"
          "cube-single-play-v0__fb_flowbc__s0")
FB_CKPT = "checkpoints/final.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=15)
    ap.add_argument('--max-steps', type=int, default=200)
    ap.add_argument('--n-list', default='1,2,4,8,16,32,64,128')
    ap.add_argument('--relabel-size', type=int, default=10000)
    ap.add_argument('--load-episodes', type=int, default=400)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--run', default=FB_RUN)
    ap.add_argument('--ckpt', default=FB_CKPT)
    ap.add_argument('--out', default=str(REPO / 'analysis/atlas/rerank_audit_fb_cube.json'))
    args = ap.parse_args()

    import torch
    from data.ogbench import load_ogbench_dataset
    from envs.ogbench import ALL_TASKS, create_ogbench_env
    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from evals.ogbench import OGBenchEvaluator

    device = args.device
    N_LIST = [int(x) for x in args.n_list.split(',')]
    domain = 'cube-single-play-v0'

    # ---- load FB agent ----
    cfg = load_cfg(f'{args.run}/.hydra/config.yaml', device=device)
    _, agent, obs_space, action_dim = build_env_and_agent(cfg)
    load_checkpoint(agent, f'{args.run}/{args.ckpt}', map_location=device)
    agent.model.eval()
    m = agent.model
    obs_dim = int(obs_space.shape[-1])

    # ---- offline buffer for the FB-eval z inference ----
    ds = load_ogbench_dataset(domain=domain,
                              data_path='/home/adityamohan/bridge_ckpt_backup/datasets',
                              load_n_episodes=args.load_episodes, device=device)
    evaluator = OGBenchEvaluator(domain=domain, agent=agent, offline_buffer=ds,
                                 relabel_size=args.relabel_size, shift_reward=1.0, device=device)
    tasks = evaluator.tasks
    assert len(tasks) == 5, tasks

    # ---- infer z_g once per task (exact FB-eval reward-relabel inference) ----
    zs = {}
    for t in tasks:
        z, rmet = evaluator._infer_z(t)
        zs[t] = z.detach().to(device).float()
        print(f"[z] {t}: |z|={zs[t].norm().item():.3f}  relabel_nonzero={rmet.get('relabel_reward#nonzero')}",
              flush=True)

    # ---- critic left-encoding (mirrors the training/eval forward path) ----
    def left_enc(obs_t):
        x = m._normalize(obs_t)          # Identity here
        x = m._fw_encoder(x)             # Identity here (state)
        return m._left_encoder(x)        # real MLP -> L_dim

    @torch.no_grad()
    def q_fb(obs_np, acts_t, z_g):
        """obs_np [obs_dim], acts_t [N, action_dim], z_g [z_dim] -> Q[N] (min over heads)."""
        N = acts_t.shape[0]
        o = torch.as_tensor(np.repeat(obs_np[None], N, axis=0), dtype=torch.float32, device=device)
        le = left_enc(o)
        z_b = z_g[None].expand(N, -1)
        q = agent._fb_Q_heads(le, z_b, acts_t, m._forward_map)   # [num_parallel, N]
        q = q.min(0).values if q.dim() == 2 else q               # pessimistic over forward heads
        return q

    @torch.no_grad()
    def sample_candidates(obs_np, z_g, N):
        """N flow-BC candidates a_i ~ mu(.|s, z_g), each with fresh noise."""
        o = torch.as_tensor(np.repeat(obs_np[None], N, axis=0), dtype=torch.float32, device=device)
        z_b = z_g[None].expand(N, -1)
        a = m.act(o, z_b, mean=False)     # [N, action_dim]; mean flag ignored (stochastic flow)
        return a

    # ---- policies ----
    @torch.no_grad()
    def rerank_policy(obs_np, z_g, N):
        cand = sample_candidates(obs_np, z_g, N)          # [N, ad]
        if N == 1:
            a = cand[0]
        else:
            Q = q_fb(obs_np, cand, z_g)                   # [N]
            a = cand[int(Q.argmax().item())]
        return np.clip(a.cpu().numpy(), -1, 1)

    @torch.no_grad()
    def trained_policy(obs_np, z_g, N=None):
        a = agent.act(torch.as_tensor(obs_np, dtype=torch.float32, device=device)[None], z_g)
        return np.clip(a.cpu().numpy()[0], -1, 1)

    # ---- rollout over the 5 tasks, `episodes` per task ----
    def rollout(policy_fn, N):
        srs = []
        for ti, task in enumerate(tasks):
            env, _ = create_ogbench_env(task, seed=1000 * (ti + 1))
            z_g = zs[task]
            hits = []
            for ep in range(args.episodes):
                obs, _ = env.reset(seed=1000 * (ti + 1) + ep)
                obs = np.asarray(obs, np.float32)
                hit = False
                for _t in range(args.max_steps):
                    a = policy_fn(obs, z_g, N)
                    obs, _, term, trunc, info = env.step(a)
                    obs = np.asarray(obs, np.float32)
                    if info.get('success', False):
                        hit = True
                    if term or trunc:
                        break
                hits.append(float(hit))
            env.close()
            srs.append(float(np.mean(hits)))
        return float(np.mean(srs)), srs

    res = {'env': domain, 'episodes': args.episodes, 'max_steps': args.max_steps,
           'n_list': N_LIST, 'z_inference': 'fb_eval_reward_relabel',
           'obs_norm': 'identity', 'sweep': {}, 'anchors': {}, 'per_task': {}}

    t0 = time.time()
    # trained_fb anchor (correctness gate)
    v, per = rollout(trained_policy, None)
    res['anchors']['trained_fb'] = v
    res['per_task']['trained_fb'] = per
    print(f"[anchor] trained_fb {v:.3f}  per_task={[round(x,3) for x in per]}  ({time.time()-t0:.0f}s)", flush=True)

    # floor (N=1 rerank == single flow sample)
    v, per = rollout(rerank_policy, 1)
    res['anchors']['floor'] = v
    res['per_task']['floor'] = per
    print(f"[anchor] floor(N=1) {v:.3f}  per_task={[round(x,3) for x in per]}  ({time.time()-t0:.0f}s)", flush=True)

    for N in N_LIST:
        if N == 1:
            res['sweep'][N] = res['anchors']['floor']
            print(f"[sweep] N={N:<4} fb_rerank {res['sweep'][N]:.3f} (=floor)", flush=True)
            continue
        v, per = rollout(rerank_policy, N)
        res['sweep'][N] = v
        res['per_task'][f'N{N}'] = per
        print(f"[sweep] N={N:<4} fb_rerank {v:.3f}  per_task={[round(x,3) for x in per]}  ({time.time()-t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[rerank-audit:fb] -> {out}", flush=True)


if __name__ == '__main__':
    main()
