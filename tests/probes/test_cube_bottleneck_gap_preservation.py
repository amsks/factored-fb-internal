"""Aggregator of the gap-preservation figure (no simulator, no checkpoints): per-state gaps,
the per-checkpoint calibration slope, the ratio, quantile binning and the standard error."""
import numpy as np
import pandas as pd
import pytest

from scripts.probes.cube_bottleneck import gap_preservation as GP


def _long(n_states=20, n_cand=6, scale=3.0, seed=0, critic="fb_tdj", flip_state=None, tie_state=None):
    """Synthetic probe rows: q = scale * r + per-state offset (a critic that keeps every gap
    exactly, up to one global affine map). `flip_state` gets its q reversed; `tie_state`
    gets constant returns."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_states):
        r = rng.uniform(0, 10, n_cand)
        if tie_state is not None and i == tie_state:
            r[:] = 2.0
        q = scale * r + rng.normal(0, 50)
        if flip_state is not None and i == flip_state:
            q = -q
        for j in range(n_cand):
            rows.append(dict(critic=critic, seed=0, bank="carrying", state=i, task=0,
                             candidate=f"c{j}", q_pred=q[j], return_true=r[j]))
    return pd.DataFrame(rows)


def test_slope_recovers_the_global_scale_and_ratio_is_one():
    long = _long(scale=3.0)
    slopes = GP.calibration_slopes(long)
    assert slopes[("fb_tdj", 0)] == pytest.approx(1 / 3.0)   # return units per critic unit
    per = GP.attach_ratio(GP.per_state_gaps(long), slopes)
    assert len(per) == 20
    assert np.allclose(per["y"], 1.0)
    assert np.allclose(per["y_raw"], 3.0)
    assert np.allclose(per["y_rel"], 1.0)      # best/worst are the extremes of q too
    assert per["reversed"].sum() == 0 and per["tie"].sum() == 0


def test_reversed_state_gets_negative_ratio_and_tie_is_excluded():
    long = _long(scale=2.0, flip_state=3, tie_state=7)
    per = GP.attach_ratio(GP.per_state_gaps(long), GP.calibration_slopes(long))
    flipped = per[per["state"] == 3].iloc[0]
    assert flipped["delta_hat"] < 0 and flipped["reversed"] == 1 and flipped["y"] < 0
    tie = per[per["state"] == 7].iloc[0]
    assert tie["tie"] == 1 and np.isnan(tie["y"]) and tie["delta_mc"] == 0
    bins = GP.bin_table(per, n_bins=5)
    assert bins["n"].sum() == 19                 # the tie state is dropped before binning


def test_quantile_bins_have_equal_counts_and_are_ordered():
    x = np.array([5.0, 1.0, 1.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0])
    b = GP.quantile_bins(x, 5)
    assert np.bincount(b).tolist() == [2, 2, 2, 2, 2]
    # every value in bin k is <= every value in bin k+1
    for k in range(4):
        assert x[b == k].max() <= x[b == k + 1].min()


def test_bin_table_mean_se_and_reversed_fraction_on_known_values():
    per = pd.DataFrame(dict(
        critic=["fb_tdj"] * 4, seed=[0] * 4, bank=["carrying"] * 4, state=range(4),
        delta_mc=[1.0, 2.0, 3.0, 4.0], delta_hat=[0.5, -1.0, 1.5, 2.0],
        y=[1.0, 3.0, 0.5, -0.5], y_rel=[0.1, 0.2, 0.3, 0.4],
        reversed=[0, 1, 0, 1], tie=[0, 0, 0, 0]))
    bins = GP.bin_table(per, n_bins=2)
    assert len(bins) == 2 and bins["n"].tolist() == [2, 2]
    b0 = bins.iloc[0]
    assert b0["gap_lo"] == 1.0 and b0["gap_hi"] == 2.0 and b0["gap_mean"] == 1.5
    assert b0["y_mean"] == pytest.approx(2.0)
    assert b0["y_se"] == pytest.approx(np.std([1.0, 3.0], ddof=1) / np.sqrt(2))
    assert b0["frac_reversed"] == pytest.approx(0.5)
    b1 = bins.iloc[1]
    assert b1["y_mean"] == pytest.approx(0.0) and b1["y_rel_mean"] == pytest.approx(0.35)


def test_bins_are_per_critic():
    a = _long(scale=1.0, critic="fb_tdj", seed=1)
    b = _long(scale=4.0, critic="density_fb_z", seed=2)
    long = pd.concat([a, b], ignore_index=True)
    per, bins, slopes = GP.aggregate(long, n_bins=5)
    assert slopes[("fb_tdj", 0)] == pytest.approx(1.0)
    assert slopes[("density_fb_z", 0)] == pytest.approx(0.25)
    assert sorted(bins["critic"].unique()) == ["density_fb_z", "fb_tdj"]
    assert (bins.groupby("critic")["n"].sum() == 20).all()
    assert np.allclose(bins["y_mean"], 1.0)



def test_wilson_ci_known_values_and_edges():
    lo, hi = GP.wilson_ci(50, 100)
    assert lo == pytest.approx(0.4038, abs=2e-4) and hi == pytest.approx(0.5962, abs=2e-4)
    lo, hi = GP.wilson_ci(0, 10)
    assert lo == pytest.approx(0.0) and 0 < hi < 0.35
    lo, hi = GP.wilson_ci(10, 10)
    assert hi == pytest.approx(1.0) and 0.65 < lo < 1
    assert all(np.isnan(v) for v in GP.wilson_ci(0, 0))


def test_phase_table_correct_fraction_gap_and_phase_order():
    per = pd.DataFrame(dict(
        critic=["fb_tdj"] * 6, seed=[0, 0, 1, 1, 1, 1], state=range(6),
        bank=["carrying", "carrying", "approach", "approach", "approach", "approach"],
        delta_mc=[10.0, 20.0, 2.0, 4.0, 6.0, 0.0], delta_hat=[3.0, -1.0, 1.0, 1.0, -2.0, 1.0],
        y=[0.1] * 6, y_rel=[0.1] * 6, reversed=[0, 1, 0, 0, 1, 0], tie=[0, 0, 0, 0, 0, 1]))
    t = GP.phase_table(per)
    assert [str(b) for b in t["bank"]] == ["approach", "carrying"]   # phase order
    a, c = t.iloc[0], t.iloc[1]
    assert a["n"] == 3 and a["n_correct"] == 2 and a["frac_correct"] == pytest.approx(2 / 3)  # tie dropped
    assert a["gap_mean"] == pytest.approx(4.0)
    assert a["gap_se"] == pytest.approx(np.std([2.0, 4.0, 6.0], ddof=1) / np.sqrt(3))
    assert (a["ci_lo"], a["ci_hi"]) == GP.wilson_ci(2, 3)
    assert c["n"] == 2 and c["frac_correct"] == pytest.approx(0.5) and c["gap_mean"] == 15.0
    by_seed = GP.phase_table(per, by_seed=True)
    assert by_seed["n"].sum() == 5 and set(by_seed["seed"]) == {0, 1}
