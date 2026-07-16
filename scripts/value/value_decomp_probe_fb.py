"""FB row of the value-geometry vs advantage-geometry decomposition (Finding 5).

Idea (user, 2026-07-16): subtract a STATE baseline from FB's measure so the readout focuses on
the ACTION-dependent part. Decompose the in-hand FB value

    Q_FB(s,a,z) = <F(s,a,z), z>
    U_FB(s,z)   = E_{a~mu}[Q_FB]            # state baseline (the shared future M-bar_mu)
    AdvPi(s,z)  = Q_FB(s,pi,z) - U_FB(s,z)  # action-dependent part (the successor ADVANTAGE)

and AUC each readout against realized in-hand grasp->goal success, exactly as
scripts/value/value_decomp_probe.py does for jax bilinear/MLP. This is the FB analogue that was
missing from the Finding-5 table (bilinear/MLP only).

WHY this is the meaningful inference-time test of the baseline-subtraction idea and reranking is
NOT: within a fixed state the baseline U_FB(s,z) is action-independent, so
  argmax_a AdvPi = argmax_a Q_FB  and  Spearman_a(AdvPi, .) = Spearman_a(Q_FB, .)
-- subtracting a state-only baseline is a no-op for any WITHIN-state action ranking (rerank /
per-state transport alignment). It only changes CROSS-state pooled metrics like this AUC over the
200 grasps, where removing the state offset isolates whether the action part carries reachability.

Labels are REUSED from the existing FB in-hand rollout records
(analysis/value/repsep/cf_records_fb_k*.parquet) so no ~1hr rollout is needed -- only forward
passes to add U_FB. The stored `value` column IS Q_FB(s,pi,z) from value_q(); we recompute it as a
provenance check and compute U_FB with the byte-identical readout (mean over forward heads, no
value_scale), so AdvPi = value - U is on one scale.

Run under the repo torch venv: /home/adityamohan/git/Factored-FB/.venv/bin/python
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

# Plain FB, state, cube: agent=fb_flowbc, z_dim=50, ortho_coef=1000, 1.5M steps (inventory row
# fb_state_plain, load-verified) -- the SAME checkpoint critics.py::_load_fb and rerank_audit_fb use.
FB_RUN = ('/home/adityamohan/bridge_ckpt_backup/factored-fb/runs/'
          '20260607_070225__fb_state_plain__s0')
FB_CKPT = 'checkpoints/step_1000000.pt'
DATA_PATH = '/home/adityamohan/bridge_ckpt_backup/datasets'


def _auc(y, s):
    """Rank AUC P(s_pos > s_neg); nan unless both classes have >=5 (matches the probe's guard)."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, int)
    if not (5 <= int(y.sum()) <= len(y) - 5):
        return float('nan')
    return float(roc_auc_score(y, np.asarray(s, float)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', default=str(REPO / 'analysis/value/repsep/cf_records_fb_k0.parquet'),
                    help='FB in-hand rollout records (obs + success labels + stored Qpi value)')
    ap.add_argument('--run', default=FB_RUN)
    ap.add_argument('--ckpt', default=FB_CKPT)
    ap.add_argument('--mu-k', type=int, default=32, help='mu action samples per grasp for U_FB')
    ap.add_argument('--relabel-size', type=int, default=10000)
    ap.add_argument('--load-episodes', type=int, default=400)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default=str(REPO / 'analysis/value/repsep/value_decomp_fb.json'))
    args = ap.parse_args()

    import pandas as pd
    import torch
    from data.ogbench import load_ogbench_dataset
    from envs.ogbench import ALL_TASKS
    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from evals.ogbench import OGBenchEvaluator

    device = args.device

    # ---- load FB agent ----
    cfg = load_cfg(f'{args.run}/.hydra/config.yaml', device=device)
    _, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, f'{args.run}/{args.ckpt}', map_location=device)
    agent.model.eval()
    m = agent.model

    # ---- relabel-z per eval task (task_id 1..5), the exact FB-eval inference ----
    ds = load_ogbench_dataset(domain='cube-single-play-v0', data_path=DATA_PATH,
                              load_n_episodes=args.load_episodes, device=device)
    evaluator = OGBenchEvaluator(domain='cube-single-play-v0', agent=agent, offline_buffer=ds,
                                 relabel_size=args.relabel_size, shift_reward=1.0, device=device)
    tasks = ALL_TASKS.get('cube-single-play-v0', [])
    zs = {ti: evaluator._infer_z(t)[0].detach().float().reshape(-1)
          for ti, t in enumerate(tasks, start=1)}

    @torch.no_grad()
    def q_fb(obs_t, z1, acts_t):
        """obs_t [B,28], z1 [z_dim], acts_t [B,ad] -> Q[B]. Byte-identical readout to value_q():
        model.forward_map(obs, z, a) [mean over heads] . z, no value_scale."""
        zb = z1[None].expand(obs_t.shape[0], -1).float()
        F = m.forward_map(obs_t, zb, acts_t.float())
        if F.dim() == 3:
            F = F.mean(0)                       # mean over forward heads (matches value_q)
        return (F * zb).sum(-1)

    @torch.no_grad()
    def readouts(obs_np, z1):
        """U = E_mu Q, Qpi = Q at pi, AdvPi = Qpi - U for a grasp batch [S,28]."""
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        S = obs_t.shape[0]
        # Qpi: policy action (eval/mean) -- reproduces the stored `value` column
        a_pi = agent.act(obs_t, z1[None].expand(S, -1).float())
        a_pi = torch.as_tensor(np.asarray(a_pi), dtype=torch.float32, device=device).reshape(S, -1)
        Qpi = q_fb(obs_t, z1, a_pi)
        # U: mean over K mu-samples (stochastic flow BC), vectorized over S*K
        K = args.mu_k
        o_rep = obs_t.repeat_interleave(K, 0)                       # [S*K, 28]
        zb = z1[None].expand(S * K, -1).float()
        a_mu = m.act(o_rep, zb, mean=False)                        # [S*K, ad] fresh noise
        a_mu = torch.as_tensor(np.asarray(a_mu), dtype=torch.float32, device=device).reshape(S * K, -1)
        Qmu = q_fb(o_rep, z1, a_mu).reshape(S, K)
        U = Qmu.mean(1)
        return (U.cpu().numpy(), Qpi.cpu().numpy(), (Qpi - U).cpu().numpy())

    # ---- records: obs + labels + stored Qpi ----
    df = pd.read_parquet(args.records)
    obs_cols = [c for c in df.columns if c.startswith('obs_')]
    obs_cols.sort(key=lambda c: int(c.split('_')[1]))
    assert len(obs_cols) == 28, obs_cols

    aucs = {k: [] for k in ('U', 'Qpi', 'AdvPi')}
    prov = []       # per-task |Qpi_recomputed - stored value| provenance
    per_task = {k: [] for k in ('U', 'Qpi', 'AdvPi')}
    for ti in range(1, len(tasks) + 1):
        sub = df[df.task == ti]
        obs = sub[obs_cols].to_numpy(np.float32)
        y = sub['success'].to_numpy().astype(int)
        U, Qpi, Adv = readouts(obs, zs[ti])
        prov.append(float(np.mean(np.abs(Qpi - sub['value'].to_numpy(np.float64)))))
        for k, s in (('U', U), ('Qpi', Qpi), ('AdvPi', Adv)):
            a = _auc(y, s)
            aucs[k].append(a)
            per_task[k].append(None if a != a else round(a, 3))
        print(f"  task{ti}: sr={y.mean():.2f}  AUC  U={aucs['U'][-1]:.3f}  "
              f"Qpi={aucs['Qpi'][-1]:.3f}  AdvPi={aucs['AdvPi'][-1]:.3f}  "
              f"(|Qpi-stored|={prov[-1]:.3g})", flush=True)

    summ = {k: float(np.nanmean(np.array(v, float))) for k, v in aucs.items()}
    out = dict(method='fb', n_states=int(len(df) // len(tasks)), mu_k=args.mu_k,
               records=str(args.records), auc=summ, per_task=per_task,
               qpi_provenance_maxabs=float(np.max(prov)))
    print(f"\n  [fb decomposition]  U={summ['U']:.3f}  Qpi={summ['Qpi']:.3f}  "
          f"AdvPi={summ['AdvPi']:.3f}   (Qpi-provenance maxabs {np.max(prov):.3g})", flush=True)

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"[value-decomp:fb] -> {p}", flush=True)


if __name__ == '__main__':
    main()
