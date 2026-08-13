"""Tests for the full-measure wiring into the SMB harness (spec 10 L1+L2 on top
of spec 11): the battery-query adapter + density set-mass path in
eval_smb_measure.py and the cmoment / measure-ranked-selection wiring in
eval_smb.py. Pure logic with mock readout closures -- no GPU, no checkpoint,
no env (full_measure itself is covered by tests/test_full_measure.py)."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts" / "rescuing_fb"
sys.path.insert(0, str(SCRIPTS))

from full_measure import FullMeasure  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def esm():
    return _load("eval_smb_measure")


@pytest.fixture(scope="module")
def esa():
    return _load("eval_smb")


@pytest.fixture(scope="module")
def smc():
    return _load("smb_common")


@pytest.fixture(scope="module")
def cloud():
    rng = np.random.default_rng(0)
    cube = np.concatenate([
        rng.normal([0, 0, 0], 0.3, (2000, 3)),
        rng.normal([2, 0, 0], 0.3, (1000, 3)),
        rng.uniform(-1, 3, (1000, 3)),
    ])
    return cube, FullMeasure(cube, eps=0.4, seed=0)


# ============================ battery-query adapter ==============================

def test_battery_to_fm_query_mapping(esm):
    ball = {"type": "ball", "center": [1, 2, 3], "eps": 0.4}
    box = {"type": "box", "center": [0, 0, 0], "half": [0.2, 0.3, 0.4]}
    uni = {"type": "union2", "centers": [[0, 0, 0], [2, 0, 0]], "eps": 0.4}
    fb = esm.battery_to_fm_query(ball)
    assert fb == {"kind": "ball", "center": [1, 2, 3], "eps": 0.4}
    fx = esm.battery_to_fm_query(box)
    assert fx == {"kind": "box", "center": [0, 0, 0], "half": [0.2, 0.3, 0.4]}
    fu = esm.battery_to_fm_query(uni)
    assert fu["kind"] == "union" and len(fu["parts"]) == 2
    assert all(p["kind"] == "ball" and p["eps"] == 0.4 for p in fu["parts"])
    with pytest.raises(ValueError):
        esm.battery_to_fm_query({"type": "cone"})


def test_adapter_members_match_query_member(esm, smc, cloud):
    """fm.member_indices over the adapted query == the harness's own
    query_member mask -- the two membership definitions must agree exactly."""
    cube, fm = cloud
    queries = [
        {"type": "ball", "center": [0.1, 0.0, 0.0], "eps": 0.4},
        {"type": "box", "center": [0.0, 0.0, 0.0], "half": [0.5, 0.4, 0.3]},
        {"type": "union2", "centers": [[0, 0, 0], [2, 0, 0]], "eps": 0.4},
    ]
    for q in queries:
        got = esm.battery_to_fm_query(q)
        idx = fm.member_indices(got)
        mask_idx = np.nonzero(smc.query_member(q, cube))[0]
        np.testing.assert_array_equal(np.sort(idx), mask_idx), q["type"]
        assert len(idx) > 0                     # the test regions are populated


# ============================ density set-mass path ==============================

def test_density_battery_predictions_counting_measure(esm, cloud):
    """Mock oracle M-tilde(x) = N-hat(x) => density 1 everywhere => every set
    mass must equal the dataset fraction |X|/n (the counting measure)."""
    cube, fm = cloud
    bvf = lambda idx: fm.nhat(cube[np.asarray(idx, np.int64)])  # noqa: E731
    queries = [
        {"type": "ball", "center": [0.0, 0.0, 0.0], "eps": 0.4},
        {"type": "box", "center": [2.0, 0.0, 0.0], "half": [0.5, 0.5, 0.5]},
        {"type": "union2", "centers": [[0, 0, 0], [2, 0, 0]], "eps": 0.4},
    ]
    preds, diags = esm.density_battery_predictions(fm, queries, bvf)
    assert preds.shape == (3,)
    for p, q, dg in zip(preds, queries, diags):
        n_x = len(fm.member_indices(esm.battery_to_fm_query(q)))
        assert dg["n_members"] == n_x and 1 <= dg["n_used"] <= fm.max_members
        assert p == pytest.approx(n_x / fm.n, rel=1e-9)   # density==1 exactly


def test_density_union_additive_by_construction(esm, cloud):
    """With every member evaluated (max_members big), the union mass equals the
    sum of the disjoint parts EXACTLY -- the property the surrogate lacks."""
    cube, _ = cloud
    fm = FullMeasure(cube, eps=0.4, max_members=10_000, seed=0)
    rng = np.random.default_rng(3)
    vals = rng.uniform(0.5, 2.0, len(cube))               # arbitrary density
    bvf = lambda idx: vals[np.asarray(idx, np.int64)] * \
        fm.nhat(cube[np.asarray(idx, np.int64)])          # noqa: E731
    b1 = {"type": "ball", "center": [0, 0, 0], "eps": 0.4}
    b2 = {"type": "ball", "center": [2, 0, 0], "eps": 0.4}
    uni = {"type": "union2", "centers": [b1["center"], b2["center"]], "eps": 0.4}
    (p1, p2, pu), _ = esm.density_battery_predictions(fm, [b1, b2, uni], bvf)
    assert pu == pytest.approx(p1 + p2, rel=1e-12)


# ============================ trained-nhat hook (phase 2b) =======================

def test_fullmeasure_nhat_fn_hook(cloud):
    """FullMeasure(nhat_fn=...): the trained head's predictions replace the
    KDTree kernel count everywhere nhat() is consulted (density, set_mass,
    corrected_moment_z); the floor still applies; member_indices stays
    geometric (KDTree)."""
    cube, fm_kd = cloud
    calls = []

    def mock_nhat(centers):
        centers = np.atleast_2d(centers)
        calls.append(len(centers))
        return np.full(len(centers), 0.01)

    fm_tr = FullMeasure(cube, eps=0.4, seed=0, nhat_fn=mock_nhat)
    q = cube[:5]
    np.testing.assert_allclose(fm_tr.nhat(q), 0.01)
    assert calls, "nhat_fn was not consulted"
    assert not np.allclose(fm_kd.nhat(q), 0.01)        # kdtree path differs
    # density divides by the trained values
    np.testing.assert_allclose(fm_tr.density(np.full(5, 0.02), q), 2.0)
    # floor still applies when the head under-predicts
    fm_lo = FullMeasure(cube, eps=0.4, nhat_fn=lambda c: np.zeros(len(np.atleast_2d(c))))
    np.testing.assert_allclose(fm_lo.nhat(q), fm_lo.floor)
    # member_indices is untouched by the hook (geometry, not mass)
    ball = {"kind": "ball", "center": [0.0, 0.0, 0.0], "eps": 0.4}
    np.testing.assert_array_equal(fm_tr.member_indices(ball),
                                  fm_kd.member_indices(ball))
    # set_mass with density==2 everywhere: mass = 2 * |X|/n
    idx = fm_tr.member_indices(ball)
    got = fm_tr.set_mass(lambda i: np.full(len(i), 0.02), idx)
    assert got["mass"] == pytest.approx(2.0 * len(idx) / fm_tr.n, rel=1e-9)


# ============================ measure-ranked selection ===========================

def test_measure_select_argmax_and_details(esa):
    z_a, z_b, z_c = np.array([1.0]), np.array([2.0]), np.array([3.0])
    calls = []

    def qr_fn(cond):
        calls.append(float(cond[0]))
        return {"q": -abs(float(cond[0]) - 2.1), "n_live": 7}

    out = esa.measure_select([("a", z_a), ("b", z_b), ("c", z_c)], qr_fn)
    assert out["chosen"] == "b"
    assert out["z"] is z_b                       # the WINNING conditioning object
    assert set(out["scores"]) == {"a", "b", "c"}
    assert out["details"]["a"]["n_live"] == 7
    assert calls == [1.0, 2.0, 3.0]              # each candidate scored once


# ============================ CLI contracts ======================================

def test_parsers_accept_new_knobs(esa, esm):
    ns = esa.build_parser().parse_args(
        ["--run-dir", "/tmp/x", "--tasks-json", "/tmp/t.json",
         "--z-inference", "cmoment", "--z-select", "measure"])
    assert ns.z_inference == "cmoment" and ns.z_select == "measure"
    ns0 = esa.build_parser().parse_args(
        ["--run-dir", "/tmp/x", "--tasks-json", "/tmp/t.json"])
    assert ns0.z_select == "off"                 # default = current behavior
    nm = esm.build_parser().parse_args(
        ["--run-dir", "/tmp/x", "--tasks-json", "/tmp/t.json",
         "--predictor", "density"])
    assert nm.predictor == "density"
    nm0 = esm.build_parser().parse_args(
        ["--run-dir", "/tmp/x", "--tasks-json", "/tmp/t.json"])
    assert nm0.predictor == "surrogate"          # default = current behavior
    # phase 2b: trained-denominator variants
    nt = esm.build_parser().parse_args(
        ["--run-dir", "/tmp/x", "--tasks-json", "/tmp/t.json",
         "--predictor", "density_trained"])
    assert nt.predictor == "density_trained"
    na = esa.build_parser().parse_args(
        ["--run-dir", "/tmp/x", "--tasks-json", "/tmp/t.json",
         "--z-inference", "cmoment_trained"])
    assert na.z_inference == "cmoment_trained"


def test_helps_mention_new_knobs():
    pa = subprocess.run([sys.executable, str(SCRIPTS / "eval_smb.py"), "--help"],
                        capture_output=True, text=True, timeout=120)
    assert pa.returncode == 0, pa.stderr
    assert "cmoment" in pa.stdout and "--z-select" in pa.stdout
    pm = subprocess.run([sys.executable, str(SCRIPTS / "eval_smb_measure.py"),
                         "--help"], capture_output=True, text=True, timeout=120)
    assert pm.returncode == 0, pm.stderr
    assert "--predictor" in pm.stdout and "density" in pm.stdout
