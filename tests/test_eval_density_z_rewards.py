"""Tests for scripts/rescuing_fb/eval_density_z_rewards.py (v2, the thin
wrapper over eval_density_zeroshot.py -- spec 24 sec.3-4 E2 arms + control
gate). Pure numpy, no jax/env/checkpoint at module level (same discipline as
tests/test_eval_density_zeroshot.py; the script's jax pieces are lazy).

Pins exactly the three things the v2 task brief asked for:
  (a) the control-match gate logic (gate_control_match / extract_family_means)
  (b) the E2a z_r numpy reference (analytic_zr, _sphere convention)
  (c) the consistency refusals: e2a/e2b refuse a non-z checkpoint, control
      refuses a z checkpoint, --force downgrades to a loud warning.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "rescuing_fb" / "eval_density_z_rewards.py"


@pytest.fixture(scope="module")
def dzr():
    spec = importlib.util.spec_from_file_location("eval_density_z_rewards_v2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # numpy-only at module level (jax is lazy)
    return mod


# ============ (a) control-match gate logic =========================================

def test_gate_control_match_pass_and_fail(dzr):
    """The HARD ACCEPTANCE GATE's comparison: dirac/dense compare on 'mean',
    region on 'mean_final'/'mean_any'; every family within tol => PASS, any
    family out => FAIL with that row flagged."""
    archived = {"dirac": {"t1": {}, "mean": 0.624},
                "dense": {"mean": 0.560},
                "region": {"mean_final": 0.25, "mean_any": 0.51}}
    mine_good = {"dirac": {"mean": 0.632},          # +0.008
                 "dense": {"mean": 0.552},          # -0.008
                 "region": {"mean_final": 0.26, "mean_any": 0.50}}
    ok, rows = dzr.gate_control_match(mine_good, archived, tol=0.02)
    assert ok and all(r["ok"] for r in rows)
    assert {r["family"] for r in rows} == {"dirac", "dense", "region_final",
                                           "region_any"}

    mine_bad = {"dirac": {"mean": 0.464},           # v1's actual broken read
                "dense": {"mean": 0.552},
                "region": {"mean_final": 0.30, "mean_any": 0.50}}
    ok, rows = dzr.gate_control_match(mine_bad, archived, tol=0.02)
    assert not ok
    bad = {r["family"] for r in rows if not r["ok"]}
    assert "dirac" in bad and "region_final" in bad
    # diffs are recorded, not just booleans
    d = {r["family"]: r["diff"] for r in rows}
    assert abs(d["dirac"] - 0.16) < 1e-12

    # families missing on either side are skipped, not crashed on
    ok, rows = dzr.gate_control_match({"dirac": {"mean": 0.62}},
                                      archived, tol=0.02)
    assert ok and len(rows) == 1

    with pytest.raises(AssertionError):
        dzr.gate_control_match({"dense": {"mean": None}}, {"dirac": {"mean": 0.1}})


# ============ (b) E2a z_r numpy reference ==========================================

def test_analytic_zr_matches_independent_rederivation(dzr):
    """z_r = sqrt(d) * v / (||v|| + 1e-6), v = mean_j r_j B_j -- gciql_flow
    _sphere's exact convention (eps in the DENOMINATOR, not a clamp), checked
    against an element-by-element rederivation on hand-buildable inputs."""
    b = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    r = np.array([2.0, 3.0, 4.0])
    d = 2
    v = np.array([(2 * 1 + 3 * 0 + 4 * 1) / 3.0, (2 * 0 + 3 * 1 + 4 * 1) / 3.0])
    expected = np.sqrt(2.0) * v / (np.linalg.norm(v) + 1e-6)
    got = dzr.analytic_zr(r, b, d)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-15)
    # norm convention: ||z|| = sqrt(d) up to the 1e-6 eps
    assert abs(np.linalg.norm(got) - np.sqrt(d)) < 1e-5
    # linear-in-r direction: scaling r leaves z_r invariant up to the 1e-6
    # denominator eps (exact only in the eps->0 limit; _sphere convention)
    np.testing.assert_allclose(dzr.analytic_zr(10.0 * r, b, d), got, atol=1e-5)
    # determinism: bit-identical repeat
    assert np.array_equal(dzr.analytic_zr(r, b, d), got)
    # shape guards
    with pytest.raises(AssertionError):
        dzr.analytic_zr(r[:2], b, d)


# ============ (c) consistency refusals =============================================

def test_e2_arms_refuse_non_z_checkpoint(dzr):
    """e2a/e2b (check_density_z_consistency) must refuse a checkpoint whose
    recorded config lacks measure_density_z=True; --force downgrades to a
    loud warning string, never a silent pass."""
    # the happy path: density + z => no warnings
    assert dzr.check_density_z_consistency(True, True) == []
    # g-indexed checkpoint into an E2 arm => hard refusal
    with pytest.raises(dzr.DensityZMismatchError):
        dzr.check_density_z_consistency(True, False)
    # non-density checkpoint refused too (base check, reused not bypassed)
    with pytest.raises(Exception):
        dzr.check_density_z_consistency(False, True)
    # force => loud warning, not silence
    w = dzr.check_density_z_consistency(True, False, force=True)
    assert len(w) == 1 and "GARBAGE" in w[0]


def test_control_arm_refuses_z_checkpoint(dzr):
    """The inverse refusal: --arm control on a measure_density_z=True
    checkpoint is not a control at all."""
    assert dzr.check_control_consistency(True, False) == []
    with pytest.raises(dzr.DensityZMismatchError):
        dzr.check_control_consistency(True, True)
    w = dzr.check_control_consistency(True, True, force=True)
    assert len(w) == 1 and "GARBAGE" in w[0]


# ============ smoke: parser defaults match the archived gate protocol ==============

def test_parser_defaults_pin_the_archived_protocol(dzr):
    ap = dzr.build_parser()
    ns = ap.parse_args(["--run-dir", "x", "--arm", "control"])
    assert ns.pool_size == 16384 and ns.episodes == 25 and ns.seed == 0
    assert ns.retrieval_k == 1          # archived RET_k1 protocol
    assert ns.relabel_size == 50000     # audited harness candidate pool
    assert ns.gate_tol == 0.02
