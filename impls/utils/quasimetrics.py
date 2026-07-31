"""JAX Interval Quasimetric Embedding (IQE), after Wang & Isola (arXiv:2211.15120).

Latents u, v of width k*l are reshaped to (k, l). Component i is the Lebesgue measure of
the union of intervals [u_ij, max(u_ij, v_ij)] (empty when v_ij <= u_ij — the asymmetry).
Reduction is IQE-maxmean with a learnable alpha in [0,1]. d(u,u)=0 and the triangle
inequality hold architecturally for any encoder.
"""
import jax
import jax.numpy as jnp


def interval_union_measure(a, b):
    """Measure of the union of intervals [a_j, b_j] along the last axis (b >= a).

    Sort by left endpoint, sweep a coverage frontier (running max of right endpoints);
    each interval contributes only its part beyond the frontier.
    """
    a, b = jnp.broadcast_arrays(a, b)   # pairwise callers pass u[:, None] vs v[None, :]
    order = jnp.argsort(a, axis=-1)
    a_s = jnp.take_along_axis(a, order, axis=-1)
    b_s = jnp.take_along_axis(b, order, axis=-1)
    frontier = jnp.concatenate(
        [a_s[..., :1], jax.lax.cummax(b_s, axis=b_s.ndim - 1)[..., :-1]], axis=-1)
    return jnp.maximum(b_s - jnp.maximum(a_s, frontier), 0.0).sum(axis=-1)


def iqe(u, v, components, alpha):
    """IQE-maxmean distance. u, v: (..., k*l); alpha scalar in [0,1]. Returns (...,)."""
    k = components
    u = u.reshape(*u.shape[:-1], k, -1)
    v = v.reshape(*v.shape[:-1], k, -1)
    comp = interval_union_measure(u, jnp.maximum(u, v))       # (..., k)
    return alpha * comp.max(axis=-1) + (1.0 - alpha) * comp.mean(axis=-1)
