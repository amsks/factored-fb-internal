"""FBHitDiracFlowBCAgent — the deconfounding control for the 2026-07-28 torch-FB
measure result (agents/fb/hitdirac/agent.py).

The control's whole scientific value rests on it being *FB's loss with one axis
changed* (Dirac support I -> cube ball), not a reimplementation. So the load-
bearing test is the EXACT REDUCTION: with hit = I the control's (offdiag, diag)
must equal `fb_successor_terms`, the stock FB split, to floating-point equality.
If that ever breaks, a difference in the 3-arm comparison stops being
attributable to the reward support.
"""
import pytest
import torch

from agents.fb.agent import fb_successor_terms


def _hitdirac_terms(diff: torch.Tensor, hit: torch.Tensor):
    """The control's split, lifted verbatim from FBHitDiracFlowBCAgent.update_fb."""
    nonhit = 1.0 - hit
    fb_offdiag = 0.5 * (diff * nonhit).pow(2).sum() / nonhit.sum().clamp_min(1.0)
    fb_diag = -(diff * hit).sum() / hit.sum().clamp_min(1.0)
    return fb_offdiag, fb_diag


def test_reduces_to_stock_fb_when_hit_is_identity():
    """hit = I  =>  algebraically identical to FB's quadratic-offdiag/linear-diag."""
    torch.manual_seed(0)
    for n_parallel, batch in [(1, 8), (2, 16), (3, 7)]:
        diff = torch.randn(n_parallel, batch, batch, dtype=torch.float64)
        eye = torch.eye(batch, dtype=torch.float64)
        off_diag = 1.0 - eye
        ref_off, ref_diag = fb_successor_terms(diff, off_diag, off_diag.sum())
        got_off, got_diag = _hitdirac_terms(diff, eye)
        assert torch.allclose(got_off, ref_off, atol=0, rtol=1e-12), (
            f"offdiag mismatch P={n_parallel} B={batch}: {got_off} vs {ref_off}")
        assert torch.allclose(got_diag, ref_diag, atol=0, rtol=1e-12), (
            f"diag mismatch P={n_parallel} B={batch}: {got_diag} vs {ref_diag}")


def test_ball_differs_from_identity():
    """A non-trivial ball must actually change the loss (guards a no-op control)."""
    torch.manual_seed(0)
    diff = torch.randn(2, 16, 16, dtype=torch.float64)
    eye = torch.eye(16, dtype=torch.float64)
    hit = eye.clone()
    hit[0, 5] = hit[5, 0] = 1.0          # one off-diagonal pair inside the ball
    off_e, diag_e = _hitdirac_terms(diff, eye)
    off_b, diag_b = _hitdirac_terms(diff, hit)
    assert not torch.allclose(off_b, off_e), "ball must move mass out of the offdiag term"
    assert not torch.allclose(diag_b, diag_e), "ball must extend the linear Dirac term"


def test_hit_matrix_is_symmetric_unit_diagonal_and_eps_monotone():
    """The ball is a metric ball on the cube slice: diag=1, symmetric, monotone in eps."""
    from agents.fb.hitdirac.agent import FBHitDiracFlowBCAgent
    obs = torch.randn(12, 28)
    small = FBHitDiracFlowBCAgent.hit_matrix.__wrapped__ if hasattr(
        FBHitDiracFlowBCAgent.hit_matrix, "__wrapped__") else FBHitDiracFlowBCAgent.hit_matrix

    class _Stub:
        hit_slice = (19, 22)
        hit_blocks = None            # single-ball path
        def __init__(self, eps): self.hit_eps = eps

    h_small = small(_Stub(0.1), obs)
    h_big = small(_Stub(5.0), obs)
    assert torch.equal(h_small, h_small.T), "hit must be symmetric"
    assert torch.all(h_small.diagonal() == 1.0), "diagonal must always hit"
    assert (h_big >= h_small).all(), "larger eps must be a superset"
    assert h_big.sum() > h_small.sum(), "eps=5.0 must capture strictly more pairs"


def test_hit_matrix_matches_measure_agent_definition():
    """Both arms must share ONE reward-support definition, or the comparison is void."""
    from agents.fb.hitdirac.agent import FBHitDiracFlowBCAgent
    from agents.fb.measure.agent import FBMeasureFlowBCAgent
    obs = torch.randn(10, 28)

    class _Stub:
        hit_slice = (19, 22)
        hit_eps = 0.4
        hit_identity = False
        hit_kernel = "ball"
        hit_blocks = None

    a = FBHitDiracFlowBCAgent.hit_matrix(_Stub(), obs)
    b = FBMeasureFlowBCAgent.hit_matrix(_Stub(), obs)
    assert torch.equal(a, b), "hitdirac and measure must compute the same hit ball"


# --- conjunctive regions (2026-08-05) ---------------------------------------
# The multi-object cells of the paper's loss-form x support 2x2 need hitdirac to
# query the SAME conjunctive region the measure arm uses. Without this, the
# control would compare a one-object region against a two/five-object region --
# the support confound the 2x2 exists to remove.

CDBL = "19:22:0.4,28:31:0.4"          # cube-double success criterion
SCENE = "19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6"


def test_hit_spec_parses_into_blocks():
    from agents.fb.hitdirac.agent import FBHitDiracFlowBCAgent
    a = FBHitDiracFlowBCAgent.__new__(FBHitDiracFlowBCAgent)
    a._init_hit_spec(CDBL, hit_slice=(19, 22), hit_eps=0.4)
    assert a.hit_blocks == ((19, 22, 0.4), (28, 31, 0.4))
    b = FBHitDiracFlowBCAgent.__new__(FBHitDiracFlowBCAgent)
    b._init_hit_spec("", hit_slice=(19, 22), hit_eps=0.4)
    assert b.hit_blocks is None, "empty spec must keep the historical single ball"


def test_hit_spec_is_a_conjunction_not_a_union():
    """Two cubes: a pair hits only when BOTH are within their radii."""
    from agents.fb.hitdirac.agent import FBHitDiracFlowBCAgent
    obs = torch.zeros(3, 40)
    obs[1, 19:22] = 10.0                       # row 1: cube-1 far, cube-2 co-located
    obs[2, 28:31] = 10.0                       # row 2: cube-1 co-located, cube-2 far
    a = FBHitDiracFlowBCAgent.__new__(FBHitDiracFlowBCAgent)
    a._init_hit_spec(CDBL, hit_slice=(19, 22), hit_eps=0.4)
    hit = FBHitDiracFlowBCAgent.hit_matrix(a, obs)
    assert torch.all(hit.diagonal() == 1.0), "a state always hits itself"
    assert hit[0, 1] == 0.0 and hit[0, 2] == 0.0, "one matching block must not hit"
    assert torch.equal(hit, hit.T), "hit must stay symmetric"


def test_conjunctive_hit_matches_measure_agent():
    """The 2x2 is only valid if both arms share ONE support definition."""
    from agents.fb.hitdirac.agent import FBHitDiracFlowBCAgent
    from agents.fb.measure.agent import FBMeasureFlowBCAgent
    torch.manual_seed(0)
    obs = torch.randn(16, 40) * 0.3
    for spec in (CDBL, SCENE):
        hd = FBHitDiracFlowBCAgent.__new__(FBHitDiracFlowBCAgent)
        hd._init_hit_spec(spec, hit_slice=(19, 22), hit_eps=0.4)

        class _M:
            hit_identity = False
            hit_kernel = "ball"
            hit_slice = (19, 22)
            hit_eps = 0.4
            hit_blocks = hd.hit_blocks

        a = FBHitDiracFlowBCAgent.hit_matrix(hd, obs)
        b = FBMeasureFlowBCAgent.hit_matrix(_M(), obs)
        assert torch.equal(a, b), f"arms disagree on the support for {spec!r}"


def test_hit_spec_rejects_malformed_blocks():
    from agents.fb.hitdirac.agent import FBHitDiracFlowBCAgent
    for bad in ("19:22", "22:19:0.4", "19:22:0"):
        a = FBHitDiracFlowBCAgent.__new__(FBHitDiracFlowBCAgent)
        with pytest.raises((AssertionError, ValueError)):
            a._init_hit_spec(bad, hit_slice=(19, 22), hit_eps=0.4)
