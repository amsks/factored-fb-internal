import jax.numpy as jnp, numpy as np
from impls.utils.fb_math import project_z, targets_uncertainty, contrastive_loss, ortho_loss


def test_project_z_norm():
    z = jnp.array([[3.0, 4.0]])  # ||z||=5, d=2
    out = project_z(z, True)
    np.testing.assert_allclose(np.asarray(out), np.sqrt(2) * np.array([[0.6, 0.8]]), atol=1e-12)
    # norm_z=False is identity
    np.testing.assert_allclose(np.asarray(project_z(z, False)), np.asarray(z))


def test_targets_uncertainty_shapes_and_mean():
    preds = jnp.stack([jnp.ones((3, 3)), 3 * jnp.ones((3, 3))])  # [P=2, B, B]
    mean, unc = targets_uncertainty(preds, 2)
    np.testing.assert_allclose(np.asarray(mean), 2 * np.ones((3, 3)))
    # sum|d1-d2| over the 2x2 head grid / (P^2 - P) = (0+2+2+0)/2 = 2
    np.testing.assert_allclose(np.asarray(unc), 2 * np.ones((3, 3)))


def test_ortho_loss_matches_manual():
    B = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    off = 1.0 - jnp.eye(3)
    loss, diag, offdiag = ortho_loss(B, off, off.sum())
    cov = np.asarray(B @ B.T)
    exp_off = 0.5 * np.sum((cov * np.asarray(off)) ** 2) / float(off.sum())
    exp_diag = -np.mean(np.diagonal(cov))
    np.testing.assert_allclose(float(loss), exp_off + exp_diag, atol=1e-12)


def test_contrastive_loss_matches_manual():
    M = jnp.arange(2 * 3 * 3, dtype=jnp.float64).reshape(2, 3, 3)
    tM = jnp.ones((2, 3, 3))
    off = 1.0 - jnp.eye(3)
    loss, diag, offdiag = contrastive_loss(M, tM, 0.99, off, off.sum())
    diff = np.asarray(M) - 0.99 * np.asarray(tM)
    exp_off = 0.5 * np.sum((diff * np.asarray(off)) ** 2) / float(off.sum())
    exp_diag = -np.mean(np.diagonal(diff, axis1=1, axis2=2)) * 2
    np.testing.assert_allclose(float(loss), exp_off + exp_diag, atol=1e-12)
