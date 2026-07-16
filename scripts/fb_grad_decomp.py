"""Which loss actually moves FB's representation: the measure loss, or the orthonormality
regulariser? (And does the flow actor's gradient reach the critic at all?)

Plain FB's critic loss is

    fb_loss = fb_offdiag + fb_diag  +  ortho_coef * (orth_offdiag + orth_diag)
              \_____ the measure loss _____/     \____ the regulariser ____/

with ortho_coef = 1000 in the cube runs (read from the run config, NOT the yaml default of 1).
Loss *values* say nothing about who steers: we take a separate backward per term and report
the gradient NORM each one puts on the forward map F, the backward map B, and the encoder.

Method: we wrap agent.update_fb so the agent itself builds the batch, goals and z exactly as
in training (no reimplementation of the loss), capture the individual loss tensors, and
autograd each one separately with retain_graph.

Usage:
  .venv/bin/python scripts/fb_grad_decomp.py <run_dir> [--ckpt step_1000000.pt] [--batches 8]
"""
import argparse
import sys

import torch

sys.path.insert(0, '/home/adityamohan/git/Factored-FB')

from data.ogbench import load_ogbench_dataset  # noqa: E402
from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('run_dir')
p.add_argument('--ckpt', default=None, help='default: checkpoints/final.pt')
p.add_argument('--batches', type=int, default=8)
p.add_argument('--device', default='cuda:0')
p.add_argument('--data-path', default=None,
               help="override cfg.data_path (the runs' /dev/shm copy is usually gone)")
a = p.parse_args()

cfg = load_cfg(f'{a.run_dir}/.hydra/config.yaml', device=a.device)
env, agent, _, _ = build_env_and_agent(cfg)
ckpt = a.ckpt or f'{a.run_dir}/checkpoints/final.pt'
load_checkpoint(agent, ckpt, map_location=a.device)

import os  # noqa: E402

dp = a.data_path or cfg.data_path
if not os.path.isdir(dp):
    alt = os.path.expanduser('~/.ogbench/data')
    print(f'[warn] cfg.data_path {dp} is gone; falling back to {alt}')
    dp = alt
buffer = load_ogbench_dataset(
    domain=cfg.domain, data_path=dp, load_n_episodes=cfg.load_n_episodes,
    device=a.device, n_transitions=cfg.n_transitions, obs_type=cfg.obs_type,
    frame_stack=int(getattr(cfg, 'frame_stack', 1)),
)

m = agent.model
GROUPS = {
    'F  (forward map)': list(m._forward_map.parameters()),
    'B  (backward map)': list(m._backward_map.parameters()),
    'enc (left encoder)': list(m._left_encoder.parameters()),
}
GROUPS = {k: v for k, v in GROUPS.items() if v}


def gnorm(params):
    tot = 0.0
    for q in params:
        if q.grad is not None:
            tot += float(q.grad.detach().pow(2).sum())
    return tot ** 0.5


def zero():
    for ps in GROUPS.values():
        for q in ps:
            q.grad = None


captured = {}
real_update_fb = agent.update_fb


def spy(*args, **kw):
    """Re-run update_fb's forward, but hand us the individual loss tensors."""
    import agents.fb.agent as A
    real_terms = A.fb_successor_terms
    real_cov = A.ortho_cov

    def grab_terms(*aa, **kk):
        od, dg = real_terms(*aa, **kk)
        captured['fb_offdiag'], captured['fb_diag'] = od, dg
        return od, dg

    def grab_cov(*aa, **kk):
        C = real_cov(*aa, **kk)
        captured['Cov'] = C
        return C

    A.fb_successor_terms, A.ortho_cov = grab_terms, grab_cov
    try:
        return real_update_fb(*args, **kw)
    finally:
        A.fb_successor_terms, A.ortho_cov = real_terms, real_cov


agent.update_fb = spy

# Does the flow actor's gradient reach the critic at all? Decisive test: is ANY critic
# parameter inside actor_optimizer's param groups? The actor's backward does deposit grads on
# F (its Q term reads the forward map), but update_fb zero_grads F/B before its own backward,
# so those are discarded -- and the actor optimizer can only step params it owns.
_actor_owned = {id(q) for grp in agent.actor_optimizer.param_groups for q in grp['params']}
_critic_owned = {id(q) for grp in (list(agent.forward_optimizer.param_groups)
                                   + (list(agent.backward_optimizer.param_groups)
                                      if agent.backward_optimizer else []))
                 for q in grp['params']}
_overlap = _actor_owned & _critic_owned
print(f'actor_optimizer owns {len(_actor_owned)} tensors; critic optimizers own '
      f'{len(_critic_owned)}; OVERLAP = {len(_overlap)}')
print('=> the flow actor ' + ('CAN' if _overlap else 'CANNOT') +
      ' update the FB representation (F/B). Its gradient contribution to the critic is '
      + ('nonzero.' if _overlap else 'structurally ZERO.') + '\n')

# The agent's own fb_loss.backward() frees the graph before we can re-use it, and its
# optimizer.step() would drift the model across probe batches. Force retain_graph, and
# neuter the steps so every batch is measured at the SAME checkpoint.
_real_backward = torch.Tensor.backward


def _retaining_backward(self, *aa, **kk):
    kk['retain_graph'] = True
    return _real_backward(self, *aa, **kk)


torch.Tensor.backward = _retaining_backward
for _opt in ('forward_optimizer', 'backward_optimizer', 'actor_optimizer', 'value_optimizer'):
    _o = getattr(agent, _opt, None)
    if _o is not None:
        _o.step = (lambda *aa, **kk: None)

oc = float(agent.ortho_coef)
print(f'run       : {a.run_dir.rstrip("/").split("/")[-1]}')
print(f'ckpt      : {ckpt.split("/")[-1]}')
print(f'ortho_coef: {oc}   z_dim: {agent.z_dim}   fixed_b: {agent.fixed_b}')
print()

acc = {}
for it in range(a.batches):
    captured.clear()
    batch = buffer.sample(cfg.batch_size,
                          horizon=int(getattr(cfg, 'horizon', 1)),
                          future_gamma=float(getattr(cfg, 'future_gamma', 0.0)))
    agent.update(batch, 10_000)          # agent builds goals/z/normalisation itself
    if 'fb_offdiag' not in captured:
        raise SystemExit('update_fb was not called — is this an FB-measure run?')

    od, dg, Cov = captured['fb_offdiag'], captured['fb_diag'], captured['Cov']
    orth_diag = -Cov.diag().mean()
    orth_offdiag = 0.5 * (Cov * agent.off_diag).pow(2).sum() / agent.off_diag_sum

    terms = {
        'measure: off-diag': od,
        'measure: diag': dg,
        f'ortho: off-diag x{oc:g}': oc * orth_offdiag,
        f'ortho: diag x{oc:g}': oc * orth_diag,
    }
    for name, t in terms.items():
        zero()
        if not t.requires_grad:
            for g in GROUPS:
                acc.setdefault((name, g), []).append(0.0)
            acc.setdefault((name, '_val'), []).append(float(t))
            continue
        t.backward(retain_graph=True)
        for g, ps in GROUPS.items():
            acc.setdefault((name, g), []).append(gnorm(ps))
        acc.setdefault((name, '_val'), []).append(float(t))
    zero()

mean = lambda k: sum(acc[k]) / len(acc[k]) if k in acc else float('nan')
names = list(terms)
w = max(len(n) for n in names) + 2
print(f'{"loss term":<{w}}{"value":>11}' + ''.join(f'{g:>21}' for g in GROUPS))
print('-' * (w + 11 + 21 * len(GROUPS)))
tot = {g: sum(mean((n, g)) for n in names) for g in GROUPS}
for n in names:
    row = f'{n:<{w}}{mean((n, "_val")):>11.2f}'
    for g in GROUPS:
        v = mean((n, g))
        sh = 100 * v / tot[g] if tot[g] > 0 else 0
        row += f'{v:>13.3f} ({sh:4.1f}%)'
    print(row)

print()
meas = {g: mean(('measure: off-diag', g)) + mean(('measure: diag', g)) for g in GROUPS}
orth = {g: tot[g] - meas[g] for g in GROUPS}
for g in GROUPS:
    if meas[g] > 0:
        print(f'{g:<20} ortho / measure gradient-norm ratio = {orth[g] / meas[g]:6.2f}x')
print('\nNOTE: norms are summed per-term, so shares are indicative of magnitude, not an '
      'exact decomposition of the summed gradient (terms can cancel).')
