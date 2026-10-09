"""Actual density stage methods on a tabular deterministic Bellman fixed point.

Small tabular adapters replace neural towers, not the stage methods under test.
This isolates target orientation, mass, stopping of gradients and CE direction.
"""
from pathlib import Path
import json
import jax
import jax.numpy as jnp
import numpy as np
from impls.critics.rldp_density import RLDPDensityCritic
from impls.critics.hilp_density import HILPDensityCritic

jax.config.update('jax_enable_x64', True)
N, GAMMA = 4, .8
obs = jnp.arange(N)[:, None]
nobs = (obs + 1) % N
transition = np.roll(np.eye(N), 1, axis=1)
q_exact = (1 - GAMMA) * np.linalg.inv(np.eye(N) - GAMMA * transition)
logits = jnp.asarray(np.log(q_exact))[None]


class Tabular:
    discount = GAMMA
    temp = 1.0
    ortho_coef = 0.0
    sf_loss_scale = 1.0
    num_parallel = 1
    fb_target_online_b = False
    ortho_inert = False
    _ortho = RLDPDensityCritic._ortho

    def _le(self, _params, x):
        return x

    def _F(self, params, x, _z, _action):
        return params[:, x[:, 0].astype(int), :]

    _sf = _F

    def _B(self, _params, x):
        # Candidate order is [s1,s2,s3,s0], matching the sampled next states.
        return jax.nn.one_hot((x[:, 0].astype(int) - 1) % N, N, dtype=jnp.float64)

    def _feature_net(self, params, _stats, x):
        return self._B(params, x)

    def _sp_loss(self, _params, _target, _batch):
        return 0.0, jnp.eye(N)


batch = dict(observations=obs, next_observations=nobs,
             actions=jnp.zeros((N, 1)), terminals=jnp.zeros(N))
cond = {'z': jnp.zeros((N, N))}
boot = {'next_action': jnp.zeros((N, 1))}
results = {}
for name, cls, stage, key in [('rldp', RLDPDensityCritic, 'fb', 'forward'),
                             ('hilp', HILPDensityCritic, 'sf', 'hilp_sf')]:
    adapter = Tabular()
    adapter._row_discount = cls._row_discount.__get__(adapter, Tabular)
    params = {'forward': logits, 'left_encoder': jnp.zeros(()),
              'backward': jnp.zeros(()), 'hilp_sf': logits,
              'hilp_phi1': jnp.zeros(()), 'hilp_stats': jnp.zeros(())}
    target = dict(params)

    def loss(p, t=target, b=batch):
        return cls.stage_loss(adapter, stage, p, t, b, cond, boot)[0]

    value, info = cls.stage_loss(adapter, stage, params, target, batch, cond, boot)
    grad = jax.grad(loss)(params)[key]
    target_grad = jax.grad(lambda t: loss(params, t))(target)[key]
    bellman = (1 - GAMMA) * np.eye(N) + GAMMA * q_exact[np.asarray(nobs[:, 0])]
    np.testing.assert_allclose(q_exact, bellman, atol=1e-14)
    assert float(jnp.abs(grad).max()) < 1e-12
    assert float(jnp.abs(target_grad).max()) == 0.0

    perturbed = {**params, key: params[key].at[0, 0, 0].add(.2)}
    grad_perturbed = jax.grad(loss)(perturbed)[key]
    assert float(grad_perturbed[0, 0, 0]) > 0.0
    assert np.all(np.asarray(grad_perturbed[0, 0, 1:]) < 0.0)

    # Gradient wrt logits exposes the stage's detached target directly:
    # dL/dlogits = (p-t)/N. This checks source broadcasting and orientation.
    terminals = jnp.asarray([1., 0., 1., 0.])
    terminal_batch = {**batch, 'terminals': terminals}
    tg = jax.grad(lambda p: loss(p, b=terminal_batch))(params)[key]
    inferred_target = q_exact - N * np.asarray(tg[0])
    assert np.all(inferred_target >= -1e-14)
    np.testing.assert_allclose(inferred_target.sum(-1), 1.0, atol=1e-14)
    np.testing.assert_allclose(inferred_target[[0, 2]], np.eye(N)[[0, 2]], atol=1e-14)

    discounts = {}
    for label, b in [('terminal_priority', {'terminals': jnp.zeros(N), 'masks': jnp.zeros(N)}),
                     ('mask_fallback', {'masks': jnp.zeros(N)}), ('scalar_fallback', {})]:
        discounts[label] = np.asarray(adapter._row_discount(b, N)).tolist()
    results[name] = {
        'fixedpoint_ce': float(info['density_ce']),
        'fixedpoint_max_logit_gradient': float(jnp.abs(grad).max()),
        'target_max_gradient': float(jnp.abs(target_grad).max()),
        'overpredicted_candidate_logit_gradient': float(grad_perturbed[0, 0, 0]),
        'terminal_target_min': float(inferred_target.min()),
        'terminal_target_row_mass': inferred_target.sum(-1).tolist(),
        'discount_precedence': discounts,
    }
report = {'scope': 'Actual current CE stage methods, tabular towers, fixed continuing deterministic '
                  'four-cycle, gamma=.8, full candidate support; no neural optimization or control claim.',
          'exact_distribution': q_exact.tolist(), 'results': results}
output = Path(__file__).with_suffix('.json')
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(results, indent=2))
print(output)
