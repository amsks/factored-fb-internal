"""Non-negativity penalty on the bounded successor measure
(`FBMeasureFlowBCAgent.nonneg_coef`, added 2026-07-28).

Spec 10 L1 measured that ~45% of the learned M is negative, which is a type
error rather than a calibration error and is the leading suspect for the
65-215x set-mass miscalibration. The fix is a one-sided squared hinge on the
negative part.

The load-bearing properties, in order:
  1. coef = 0 is BIT-IDENTICAL to the previous objective (every launched
     checkpoint so far was trained at coef 0; if this drifts, none of them are
     reproducible).
  2. the penalty is exactly zero on a non-negative M -- it must not tax a model
     that is already valid.
  3. it is strictly positive, and grows, as M goes more negative.
  4. it leaves M BILINEAR (the penalty is a loss term, not a reparameterization)
     -- the zero-shot readout Q_r = <F,z_r> depends on linearity in psi.
"""
import torch


def _terms(Ms, diff, scale_norm=True, nonneg_coef=0.0):
    """The loss assembly lifted verbatim from FBMeasureFlowBCAgent.update_fb."""
    measure_loss = diff.pow(2).mean()
    denom = (Ms.detach().pow(2).mean() + 1e-6) if scale_norm else 1.0
    if scale_norm:
        measure_loss = measure_loss / denom
    fb_loss = measure_loss
    nonneg_loss = torch.relu(-Ms).pow(2).mean()
    if scale_norm:
        nonneg_loss = nonneg_loss / denom
    if nonneg_coef > 0.0:
        fb_loss = fb_loss + nonneg_coef * nonneg_loss
    return fb_loss, measure_loss, nonneg_loss


def test_coef_zero_is_bit_identical():
    """coef=0 must reproduce the pre-existing loss exactly, both scale_norm modes."""
    torch.manual_seed(0)
    for scale_norm in (True, False):
        Ms = torch.randn(2, 32, 32, dtype=torch.float64)
        diff = torch.randn(2, 32, 32, dtype=torch.float64)
        # the objective as it stood before the knob
        ref = diff.pow(2).mean()
        if scale_norm:
            ref = ref / (Ms.detach().pow(2).mean() + 1e-6)
        got, _, _ = _terms(Ms, diff, scale_norm=scale_norm, nonneg_coef=0.0)
        assert torch.equal(got, ref), f"scale_norm={scale_norm}: {got} vs {ref}"


def test_penalty_is_zero_on_a_valid_measure():
    """A non-negative M must incur no penalty at any coefficient."""
    torch.manual_seed(0)
    Ms = torch.rand(2, 16, 16, dtype=torch.float64)          # strictly >= 0
    diff = torch.randn(2, 16, 16, dtype=torch.float64)
    base, _, _ = _terms(Ms, diff, nonneg_coef=0.0)
    for coef in (1.0, 10.0, 1e3):
        got, _, pen = _terms(Ms, diff, nonneg_coef=coef)
        assert pen.item() == 0.0, f"penalty {pen} on a non-negative M"
        assert torch.equal(got, base), "a valid measure must not be taxed"


def test_penalty_grows_with_negativity():
    """Strictly positive on negative mass, and monotone in how negative it is."""
    diff = torch.zeros(1, 8, 8, dtype=torch.float64)
    mild = -0.1 * torch.ones(1, 8, 8, dtype=torch.float64)
    severe = -1.0 * torch.ones(1, 8, 8, dtype=torch.float64)
    _, _, p_mild = _terms(mild, diff, scale_norm=False, nonneg_coef=1.0)
    _, _, p_sev = _terms(severe, diff, scale_norm=False, nonneg_coef=1.0)
    assert p_mild > 0 and p_sev > 0
    assert p_sev > p_mild, f"{p_sev} !> {p_mild}"


def test_only_the_negative_part_is_penalized():
    """Positive entries contribute nothing; the hinge is one-sided."""
    a = torch.tensor([[[-2.0, 5.0], [0.0, 7.0]]], dtype=torch.float64)
    b = torch.tensor([[[-2.0, 1.0], [0.0, 1.0]]], dtype=torch.float64)
    diff = torch.zeros_like(a)
    _, _, pa = _terms(a, diff, scale_norm=False, nonneg_coef=1.0)
    _, _, pb = _terms(b, diff, scale_norm=False, nonneg_coef=1.0)
    assert torch.equal(pa, pb), "changing positive entries must not move the penalty"
    assert pa.item() == 4.0 / 4, f"expected relu(-M)^2 mean = 1.0, got {pa}"


def test_penalty_gradient_pushes_negative_entries_up():
    """The gradient must raise negative entries and leave positive ones alone."""
    Ms = torch.tensor([[[-1.0, 2.0]]], dtype=torch.float64, requires_grad=True)
    diff = torch.zeros(1, 1, 2, dtype=torch.float64)
    _, _, pen = _terms(Ms, diff, scale_norm=False, nonneg_coef=1.0)
    pen.backward()
    assert Ms.grad[0, 0, 0] < 0, "descending must INCREASE the negative entry"
    assert Ms.grad[0, 0, 1] == 0.0, "positive entry must get zero gradient"
