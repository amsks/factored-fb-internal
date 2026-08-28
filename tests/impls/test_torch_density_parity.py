"""Parity of the torch density loss (impls/torch_density/fbcpr_density.py) with the jax
stage_loss math (impls/critics/density_fb.py, hard mode, scalar gamma).

Runs in .venv-humenv (torch, no jax): the reference is a numpy transcription of the jax
code, kept line-for-line comparable. Row-wise discount and the terminated path are also
checked, which the jax arm (scalar gamma) never exercises.

    $SCRATCH/.venv-humenv/bin/python -m pytest tests/impls/test_torch_density_parity.py -q
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
# fbcpr_density imports metamotivo at module scope; the jax venv has torch but not
# metamotivo, and a bare import there aborts COLLECTION of the whole suite (so
# `make test` fails on a torch-only file). Skip on the real dependency, not just torch.
pytest.importorskip("metamotivo")

from impls.torch_density.fbcpr_density import density_ce_loss  # noqa: E402


def _np_softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _np_reference(Fs, B, tFs, tB, gamma_rows, tau):
    """jax stage_loss, hard mode, transcribed: logits [P,B,M], w_bar from targets,
    t = (1-g) I + g w_bar, ce = -mean over (P, B) of sum_M t * logp."""
    logits = np.einsum("hbd,md->hbm", Fs, B) / tau
    logp = logits - logits.max(-1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(-1, keepdims=True))
    w_bar = _np_softmax(np.einsum("hbd,md->hbm", tFs, tB) / tau)
    k = np.eye(Fs.shape[1])
    g = gamma_rows.reshape(1, -1, 1)
    t = (1.0 - g) * k[None] + g * w_bar
    return -np.mean(np.sum(t * logp, axis=-1))


@pytest.mark.parametrize("tau", [1.0, 0.5])
@pytest.mark.parametrize("terminated", [False, True])
def test_density_ce_matches_jax_math(tau, terminated):
    rng = np.random.default_rng(0)
    P, B, d = 2, 16, 8
    Fs, tFs = rng.standard_normal((P, B, d)), rng.standard_normal((P, B, d))
    Bm, tBm = rng.standard_normal((B, d)), rng.standard_normal((B, d))
    gamma = np.full((B,), 0.98)
    if terminated:
        gamma[rng.integers(0, B, 4)] = 0.0
    ref = _np_reference(Fs, Bm, tFs, tBm, gamma, tau)
    ce, info = density_ce_loss(
        torch.tensor(Fs), torch.tensor(Bm), torch.tensor(tFs), torch.tensor(tBm),
        torch.tensor(gamma).reshape(-1, 1), tau)
    assert abs(float(ce) - ref) < 1e-10, (float(ce), ref)
    assert np.isfinite(float(info["density_ess_median"]))


def test_gradients_flow_only_through_online_towers():
    torch.manual_seed(0)
    P, B, d = 2, 8, 4
    Fs = torch.randn(P, B, d, requires_grad=True)
    Bm = torch.randn(B, d, requires_grad=True)
    tFs, tBm = torch.randn(P, B, d, requires_grad=True), torch.randn(B, d, requires_grad=True)
    disc = torch.full((B, 1), 0.98)
    ce, _ = density_ce_loss(Fs, Bm, tFs, tBm, disc, 1.0)
    ce.backward()
    assert Fs.grad is not None and Fs.grad.abs().sum() > 0
    assert Bm.grad is not None and Bm.grad.abs().sum() > 0
    assert tFs.grad is None and tBm.grad is None  # targets under no_grad
