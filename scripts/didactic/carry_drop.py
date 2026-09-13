"""Full-capacity carry/drop successor-learning diagnostic (single fixed policy).

This isolates the critic objectives; it is not the production task-conditioned
actor/critic pipeline. See the companion README for conventions and differences.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

jax.config.update('jax_enable_x64', True)
METHODS = ('fb', 'fb_no_ortho', 'density', 'normalized_squared', 'direct')


def build_mdp(depth, persistence):
    if depth < 1 or persistence < 1:
        raise ValueError('depth and persistence must be at least one')
    n = depth + 2
    p = np.zeros((n, 2, n))
    for s in range(depth):
        p[s, 0, s+1] = .9
        p[s, 0, -1] = .1
        p[s, 1, -1] = 1
    p[depth, :, 0] = 1
    p[-1, :, 0] = 1 / persistence
    p[-1, :, -1] = 1 - 1 / persistence
    return p


def exact_successor(p, gamma, policy=None):
    n = p.shape[0]
    pi = np.full((n, 2), .5) if policy is None else np.asarray(policy)
    transition = np.einsum('sa,san->sn', pi, p)
    return np.linalg.solve((np.eye(n) - gamma*transition).T,
                           p.reshape(-1, n).T).T.reshape(p.shape)


def predict(params, method):
    if method == 'direct':
        return params['table']
    b = params['b']
    if method != 'density':
        b = b * jnp.sqrt(b.shape[0]) / jnp.maximum(jnp.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    logits = params['f'] @ b.T
    return jax.nn.softmax(logits, axis=-1) if method == 'density' else logits


def as_probability(pred, method, gamma):
    return pred if method in ('density', 'normalized_squared') else (1-gamma)*pred/pred.shape[-1]


def exact_parameters(m, method, gamma):
    n = m.shape[0]
    m = m.reshape(-1, n)
    if method == 'direct':
        return {'table': jnp.array(n*m)}
    if method == 'density':
        return {'f': jnp.array(np.log((1-gamma)*m)), 'b': jnp.eye(n)}
    pred = (1-gamma)*m if method == 'normalized_squared' else n*m
    return {'f': jnp.array(pred/np.sqrt(n)), 'b': jnp.eye(n)*np.sqrt(n)}


def loss_fn(params, target, p, method, gamma):
    n = p.shape[0]
    pred = predict(params, method)
    tp = jax.lax.stop_gradient(predict(target, method)).reshape(n, 2, n).mean(1)
    kernel = p.reshape(-1, n)
    boot = gamma * (kernel @ tp)
    if method == 'density':
        labels = (1-gamma)*kernel + boot
        # Use logits directly for numerical stability, rather than log(softmax).
        logp = jax.nn.log_softmax(params['f'] @ params['b'].T, axis=-1)
        return -jnp.mean(jnp.sum(labels * logp, axis=-1))
    immediate_coef = (1-gamma)/n if method == 'normalized_squared' else 1.
    loss = .5*jnp.mean((pred - boot)**2) - immediate_coef*jnp.mean(jnp.sum(kernel*pred, axis=-1))
    if method in ('fb', 'normalized_squared'):
        b = params['b']
        b = b*jnp.sqrt(n)/jnp.maximum(jnp.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
        loss += .5*jnp.mean((b @ b.T)**2) - jnp.mean(jnp.sum(b*b, axis=-1))
    return loss


def metrics(prob, p, exact, gamma):
    n = p.shape[0]
    depth = n-2
    prob = np.asarray(prob).reshape(n, 2, n)
    true = (1-gamma)*exact
    q = prob[:, :, depth]/(1-gamma)
    tq = exact[:, :, depth]
    gap = q[:depth, 0]-q[:depth, 1]
    true_gap = tq[:depth, 0]-tq[:depth, 1]
    greedy = np.eye(2)[np.argmax(q, axis=1)]
    policy_m = exact_successor(p, gamma, greedy)
    policy_q = policy_m[:, :, depth]
    ret = np.sum(greedy[0]*policy_q[0])
    optimal_pi = np.tile([1., 0.], (n, 1))
    optimal_ret = exact_successor(p, gamma, optimal_pi)[0, 0, depth]
    err = prob-true
    return {
        'start_gap_ratio': float(gap[0]/true_gap[0]),
        'min_gap_ratio': float(np.min(gap/true_gap)),
        'mean_gap_ratio': float(np.mean(gap/true_gap)),
        'wrong_actions': int(np.sum(gap <= 0)),
        'greedy_return_ratio': float(ret/optimal_ret),
        'start_gap': float(gap[0]), 'true_start_gap': float(true_gap[0]),
        'greedy_return': float(ret), 'optimal_return': float(optimal_ret),
        'rmse': float(np.sqrt(np.mean(err**2))),
        'rest_column_rmse': float(np.sqrt(np.mean(err[..., -1]**2))),
        'held_column_rmse': float(np.sqrt(np.mean(err[..., :depth]**2))),
        'goal_column_rmse': float(np.sqrt(np.mean(err[..., depth]**2))),
        'max_mass_error': float(np.max(np.abs(prob.sum(-1)-1))),
        'min_entry': float(prob.min()),
    }


def init_params(seed, n, method):
    rng = np.random.default_rng(seed)
    if method == 'direct':
        return {'table': jnp.array(rng.normal(0, .1, (n*2, n)))}
    return {'f': jnp.array(rng.normal(0, .1, (n*2, n))),
            'b': jnp.array(rng.normal(0, .1, (n, n)))}


def train_group(p_all, configs, method, steps, interval, gamma, tau, samples, out):
    """Batch independent runs; sampled kernels share the RNG schedule across methods.

    Every sampled update has `samples` transitions from each state/action, with
    the continuation action analytically averaged. Candidates are all N states.
    """
    n = p_all.shape[1]
    params = jax.tree.map(lambda *xs: jnp.stack(xs),
                          *[init_params(c['seed'], n, method) for c in configs])
    optimizer = optax.scale_by_adam()
    state = jax.vmap(optimizer.init)(params)
    keys = jnp.stack([jax.random.PRNGKey(c['seed']+10000) for c in configs])
    lr = jnp.array([c['lr'] for c in configs])
    carry = (params, params, state, keys)
    kernels = jnp.array(p_all)

    def one_update(params, target, state, key, p, rate):
        if samples:
            key, draw_key = jax.random.split(key)
            draws = jax.random.uniform(draw_key, (n, 2, samples, 1))
            nxt = jnp.sum(draws > jnp.cumsum(p, axis=-1)[:, :, None, :], axis=-1)
            p_used = jax.nn.one_hot(nxt, n).mean(2)
        else:
            p_used = p
        grad = jax.grad(loss_fn)(params, target, p_used, method, gamma)
        updates, state = optimizer.update(grad, state, params)
        params = jax.tree.map(lambda x, u: x-rate*u, params, updates)
        target = jax.tree.map(lambda t, x: (1-tau)*t + tau*x, target, params)
        return params, target, state, key

    def scan_update(carry, _):
        return jax.vmap(one_update)(*carry, kernels, lr), None

    @jax.jit
    def chunk(carry):
        return jax.lax.scan(scan_update, carry, None, length=interval)[0]

    exacts = [exact_successor(p, gamma) for p in p_all]
    rows, snapshots = [], []
    t0 = time.monotonic()
    for step in range(0, steps+1, interval):
        if step:
            carry = chunk(carry)
        pred = np.asarray(jax.vmap(lambda x: as_probability(predict(x, method), method, gamma))(carry[0]))
        if not np.isfinite(pred).all():
            raise FloatingPointError(f'non-finite prediction at {method}, step {step}')
        snapshots.append(pred)
        for i, c in enumerate(configs):
            rows.append(dict(c, method=method, step=step, **metrics(pred[i], p_all[i], exacts[i], gamma)))
        if step % 10000 == 0 or step == steps:
            last = rows[-len(configs):]
            print(json.dumps({'method': method, 'depth': n-2, 'step': step,
                'elapsed_s': round(time.monotonic()-t0, 1),
                'max_wrong': max(r['wrong_actions'] for r in last),
                'gap_range': [min(r['start_gap_ratio'] for r in last), max(r['start_gap_ratio'] for r in last)]}), flush=True)
    stem = out / f'depth{n-2}_{method}'
    np.savez_compressed(str(stem)+'.npz', probabilities=np.stack(snapshots),
                        exact=np.stack(exacts), kernels=p_all,
                        **{f'final_{k}': np.asarray(v) for k, v in carry[0].items()})
    with Path(str(stem)+'.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('analysis/carry_drop/expected'))
    parser.add_argument('--depths', nargs='+', type=int, default=[1, 3, 6])
    parser.add_argument('--persistence', nargs='+', type=int, default=[1, 10, 50])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--rates', nargs='+', type=float, default=[.001, .01])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--interval', type=int, default=1000)
    parser.add_argument('--samples', type=int, default=0, help='draws per state/action/update; 0 uses exact expectations')
    parser.add_argument('--gamma', type=float, default=.98)
    parser.add_argument('--tau', type=float, default=.005)
    args = parser.parse_args()
    if args.steps % args.interval:
        parser.error('steps must be divisible by interval')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'config.json').exists():
        raise FileExistsError(f'{args.out} already has a run; choose a new output directory')
    config = vars(args).copy()
    config.update(out=str(args.out), jax=jax.__version__, optax=optax.__version__,
                  numpy=np.__version__, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  protocol='one fixed uniform policy, full candidate support, uniform state-action weights, float64')
    (args.out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    all_rows, capacity = [], []
    for depth in args.depths:
        configs = [dict(depth=depth, persistence=k, seed=seed, lr=rate)
                   for k in args.persistence for seed in args.seeds for rate in args.rates]
        kernels = np.stack([build_mdp(depth, c['persistence']) for c in configs])
        for k in args.persistence:
            p = build_mdp(depth, k)
            exact = exact_successor(p, args.gamma)
            for method in args.methods:
                prob = np.asarray(as_probability(predict(exact_parameters(exact, method, args.gamma), method), method, args.gamma))
                capacity.append(dict(depth=depth, persistence=k, method=method,
                    max_exact_error=float(np.max(np.abs(prob.reshape(p.shape)-(1-args.gamma)*exact))),
                    matrix_rank=int(np.linalg.matrix_rank(exact.reshape(-1, depth+2)))))
        for method in args.methods:
            all_rows.extend(train_group(kernels, configs, method, args.steps, args.interval,
                                         args.gamma, args.tau, args.samples, args.out))
    (args.out/'capacity.json').write_text(json.dumps(capacity, indent=2)+'\n')
    with (args.out/'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0])
        writer.writeheader()
        writer.writerows(r for r in all_rows if r['step'] == args.steps)


if __name__ == '__main__':
    main()
