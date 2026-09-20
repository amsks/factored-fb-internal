"""Separation-preservation aggregator (no simulator, no checkpoints): the 3-D cell histogram,
TV distance, the per-state D_true / D_pred / R row and the per-phase table."""
import numpy as np
import pandas as pd
import pytest

from scripts.probes.cube_bottleneck import separation_preservation as SP


def test_cube_xyz_from_obs_inverts_the_observation_scaling():
    obs = np.zeros(28); obs[19:22] = [1.0, -2.0, 0.5]        # scaled by 10, centred at (0.425, 0, 0)
    assert np.allclose(SP.cube_xyz_from_obs(obs), [0.525, -0.2, 0.05])


def test_hist3d_is_normalised_clips_outside_points_and_respects_cell_size():
    pts = np.array([[0.30, 0.0, 0.05], [0.30, 0.0, 0.05], [9.0, 9.0, 9.0]])   # last one far outside
    h = SP.hist3d(pts, None, 0.05)
    assert h.sum() == pytest.approx(1.0)
    assert sorted(h[h > 0].tolist()) == pytest.approx([1 / 3, 2 / 3])          # two cells only
    assert len(SP.hist3d(pts, None, 0.05)) > len(SP.hist3d(pts, None, 0.08))   # finer grid, more cells
    # weights: a weighted point counts by its weight
    hw = SP.hist3d(pts, np.array([1.0, 1.0, 0.0]), 0.05)
    assert hw.max() == pytest.approx(1.0)


def test_tv_bounds():
    p = np.array([1.0, 0.0]); q = np.array([0.0, 1.0])
    assert SP.tv(p, q) == pytest.approx(1.0) and SP.tv(p, p) == 0.0
    assert SP.tv(np.array([0.5, 0.5]), q) == pytest.approx(0.5)


def test_discount_weights_normalised_and_geometric():
    w = SP.discount_weights(0.5, 3)
    assert w.sum() == pytest.approx(1.0) and np.allclose(w / w[0], [1, 0.5, 0.25])


def _futures(cell_a, cell_b, H=4, K=2):
    """[2, H, K, 3]: every rollout after a+ sits at cell_a, after a- at cell_b."""
    mc = np.zeros((2, H, K, 3))
    mc[0] = cell_a; mc[1] = cell_b
    return mc


def test_separation_row_fully_preserved_and_lost():
    a, b = np.array([0.30, 0.0, 0.05]), np.array([0.50, 0.2, 0.05])
    pool = np.stack([a, b])
    mc = _futures(a, b)
    # critic puts all mass on the right pool state after each action: D_pred = D_true = 1
    s = SP.separation_row(mc, np.array([[1.0, 0.0], [0.0, 1.0]]), pool, 0.99, 0.05)
    assert s["d_true"] == pytest.approx(1.0) and s["d_pred"] == pytest.approx(1.0)
    assert s["R"] == pytest.approx(1.0)
    assert s["fidelity_plus"] == pytest.approx(0.0) and s["fidelity_minus"] == pytest.approx(0.0)
    # critic predicts the same future for both actions: separation lost, fidelity 0.5 each
    s = SP.separation_row(mc, np.array([[0.5, 0.5], [0.5, 0.5]]), pool, 0.99, 0.05)
    assert s["d_pred"] == pytest.approx(0.0) and s["R"] == pytest.approx(0.0)
    assert s["fidelity_plus"] == pytest.approx(0.5)
    # identical true futures: D_true = 0 and R undefined
    s = SP.separation_row(_futures(a, a), np.array([[1.0, 0.0], [0.0, 1.0]]), pool, 0.99, 0.05)
    assert s["d_true"] == 0.0 and np.isnan(s["R"])


def test_separation_row_discounting_weights_early_steps_more():
    a, b = np.array([0.30, 0.0, 0.05]), np.array([0.50, 0.2, 0.05])
    pool = np.stack([a, b])
    mc = _futures(a, b, H=2, K=1)
    mc[0, 1] = b                       # a+ future: at a for t=0, at b for t=1
    s = SP.separation_row(mc, np.array([[1.0, 0.0], [0.0, 1.0]]), pool, 0.5, 0.05)
    # weights (2/3, 1/3): a+ histogram = (2/3 at a, 1/3 at b); a- = (0, 1) -> TV = 2/3
    assert s["d_true"] == pytest.approx(2 / 3)


def test_phase_table_excludes_small_d_true_and_reports_mean_se_median():
    rows = pd.DataFrame(dict(
        critic=["fb_tdj"] * 5, seed=[0, 0, 1, 1, 1], state=range(5),
        bank=["carrying", "carrying", "carrying", "approach", "approach"],
        d_true=[0.5, 0.02, 0.4, 0.3, 0.6], d_pred=[0.25, 0.01, 0.4, 0.3, 0.3],
        R=[0.5, 0.5, 1.0, 1.0, 0.5], fidelity_plus=[0.2] * 5, fidelity_minus=[0.4] * 5))
    t = SP.phase_table(rows, d_true_min=0.05)
    assert [str(b) for b in t["bank"]] == ["approach", "carrying"]
    c = t[t["bank"] == "carrying"].iloc[0]
    assert c["n"] == 2 and c["n_excluded"] == 1
    assert c["R_mean"] == pytest.approx(0.75) and c["R_median"] == pytest.approx(0.75)
    assert c["R_se"] == pytest.approx(np.std([0.5, 1.0], ddof=1) / np.sqrt(2))
    assert c["d_true_mean"] == pytest.approx(0.45) and c["d_pred_mean"] == pytest.approx(0.325)
    assert c["fidelity_mean"] == pytest.approx(0.3)
    by_seed = SP.phase_table(rows, by_seed=True)
    assert by_seed["n"].sum() == 4 and set(by_seed["seed"]) == {0, 1}


def test_cosine_alignment_on_synthetic_histograms():
    d_true = np.array([0.5, -0.5, 0.0, 0.0])
    assert SP.cosine(d_true, d_true) == pytest.approx(1.0)
    assert SP.cosine(-d_true, d_true) == pytest.approx(-1.0)
    assert SP.cosine(np.array([0.0, 0.0, 1.0, -1.0]), d_true) == pytest.approx(0.0)   # orthogonal
    assert np.isnan(SP.cosine(np.zeros(4), d_true))                                     # zero norm skipped
    a = SP.alignment(np.array([0.1, -0.1, 0.3, -0.3]), d_true, top=2)
    assert a["align"] == pytest.approx(0.1 / (np.sqrt(0.2) * np.sqrt(0.5)))
    assert a["align_top"] == pytest.approx(1.0)          # on the two largest-|d_true| cells it is exact


def test_separation_row_reports_alignment_and_align_table_aggregates():
    a, b = np.array([0.30, 0.0, 0.05]), np.array([0.50, 0.2, 0.05])
    pool = np.stack([a, b])
    mc = _futures(a, b)
    s = SP.separation_row(mc, np.array([[1.0, 0.0], [0.0, 1.0]]), pool, 0.99, 0.05)
    assert s["align"] == pytest.approx(1.0) and s["align_top"] == pytest.approx(1.0)
    s = SP.separation_row(mc, np.array([[0.0, 1.0], [1.0, 0.0]]), pool, 0.99, 0.05)
    assert s["align"] == pytest.approx(-1.0)
    s = SP.separation_row(mc, np.array([[0.5, 0.5], [0.5, 0.5]]), pool, 0.99, 0.05)
    assert np.isnan(s["align"])                              # d_hat = 0: undefined, not zero
    rows = pd.DataFrame(dict(critic=["fb_tdj"] * 4, seed=[0, 0, 1, 1], state=range(4), bank=["carrying"] * 4,
                             align=[0.5, -0.5, 1.0, np.nan], align_top=[0.2, 0.2, 0.2, 0.2]))
    t = SP.align_table(rows)
    r = t.iloc[0]
    assert r["align_n"] == 3 and r["align_undefined"] == 1
    assert r["align_mean"] == pytest.approx(1 / 3) and r["align_median"] == pytest.approx(0.5)
    assert r["align_pos"] == pytest.approx(2 / 3)
    assert r["align_se"] == pytest.approx(np.std([0.5, -0.5, 1.0], ddof=1) / np.sqrt(3))
    vec = {("fb_tdj", 0, "carrying", i): (np.array([1.0, 0.0]), np.array([1.0, 0.0])) for i in range(3)}
    null = SP.permutation_null(vec, n_perm=5)
    assert null.iloc[0]["null_mean"] == pytest.approx(1.0) and null.iloc[0]["null_sd"] == 0.0
