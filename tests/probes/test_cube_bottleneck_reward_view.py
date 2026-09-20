"""Reward-view probe (no simulator, no checkpoints): AUC / d' separation numbers, the
standardised pooling, the discounted scoring of futures, the three-number row and its
critic-gap scale, and the per-phase table."""
import numpy as np
import pandas as pd
import pytest

from scripts.probes.cube_bottleneck import reward_view as RV


def _pairwise_auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    pos, neg = s[y], s[~y]
    return np.mean([(1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg])


def test_auc_matches_pairwise_definition_with_ties():
    rng = np.random.default_rng(0)
    s = np.round(rng.normal(size=60), 1)                 # rounding makes ties
    y = rng.uniform(size=60) < 0.4
    assert RV.auc(s, y) == pytest.approx(_pairwise_auc(s, y))
    assert RV.auc(np.r_[np.ones(5), np.zeros(5)], np.r_[np.ones(5), np.zeros(5)]) == 1.0
    assert RV.auc(np.zeros(10), np.r_[np.ones(5), np.zeros(5)]) == 0.5
    assert np.isnan(RV.auc(np.arange(4), np.ones(4)))    # one class only


def test_dprime_and_standardise():
    s = np.r_[np.full(4, 3.0), np.array([0.0, 2.0, 0.0, 2.0])]
    y = np.r_[np.ones(4), np.zeros(4)]
    sd0 = np.std([0.0, 2.0, 0.0, 2.0], ddof=1)
    assert RV.dprime(s, y) == pytest.approx((3.0 - 1.0) / sd0)
    z = RV.standardise(s, y)
    assert z[~y.astype(bool)].mean() == pytest.approx(0.0) and z[~y.astype(bool)].std(ddof=1) == pytest.approx(1.0)
    assert RV.auc(z, y) == RV.auc(s, y) and RV.dprime(z, y) == pytest.approx(RV.dprime(s, y))
    st = RV.separation_stats(s, y)
    assert st["n1"] == 4 and st["n0"] == 4 and st["auc"] == 1.0 and st["mean1"] == 3.0


def test_discounted_mean_normalised_over_horizon_and_averaged_over_rollouts():
    r = np.zeros((3, 2)); r[0, 0] = 1.0                  # hit at t=0 in one of two rollouts
    g = 0.5
    assert RV.discounted_mean(r, g) == pytest.approx(0.5 * 1.0 / (1 + 0.5 + 0.25))
    assert RV.discounted_mean(np.ones((5, 3)), 0.9) == pytest.approx(1.0)
    assert RV.horizon_sum(0.5, 3) == pytest.approx(1.75)


def test_critic_gap_scale_by_critic():
    assert RV.critic_gap_scale("density_fb_z", 0.99, 100) == 1.0
    assert RV.critic_gap_scale("fb_tdj", 0.99, 100) == pytest.approx(1.0 / RV.horizon_sum(0.99, 100))


def test_gap_row_true_feature_and_critic_gaps():
    H, K = 4, 2
    true_r = np.zeros((2, H, K)); true_r[0] = 1.0        # a+ always at goal, a- never
    rhat = np.zeros((2, H, K)); rhat[0] = 0.3; rhat[1] = 0.1
    g = RV.gap_row(true_r, rhat, q_gap=RV.horizon_sum(0.9, H) * 0.2, gamma=0.9, critic="fb_tdj")
    assert g["true_gap"] == pytest.approx(1.0) and g["feat_gap"] == pytest.approx(0.2)
    assert g["critic_gap"] == pytest.approx(0.2)         # FB's raw Q gap divided by sum gamma^t
    gd = RV.gap_row(true_r, rhat, q_gap=0.2, gamma=0.9, critic="density_fb_z", slope=3.0)
    assert gd["critic_gap_q"] == pytest.approx(0.2) and gd["critic_gap_norm"] == pytest.approx(0.2)
    assert gd["critic_gap"] == pytest.approx(0.6) and gd["feat_gap"] == pytest.approx(0.6)
    assert gd["feat_gap_raw"] == pytest.approx(0.2)


def test_calibration_slope_recovers_a_linear_reward_view():
    rng = np.random.default_rng(3)
    r = (rng.uniform(size=500) < 0.3).astype(float)
    rhat = 4.0 * r + 1.0 + 0.0 * rng.normal(size=500)  # r = 0.25 rhat - 0.25
    assert RV.calibration_slope(rhat, r) == pytest.approx(0.25)
    assert np.isnan(RV.calibration_slope(np.ones(5), r[:5]))


def _per_state():
    rows = []
    rng = np.random.default_rng(1)
    for c in ("fb_tdj", "density_fb_z"):
        for s in (0, 1):
            for b in ("approach", "inhand", "carrying"):
                for i in range(6):
                    t = 0.5 + 0.1 * i
                    f = (0.05 if c == "fb_tdj" else 0.3) * t + 0.001 * rng.normal()
                    q = f + 0.0005 * rng.normal()
                    rows.append(dict(critic=c, seed=s, bank=b, state=i, true_gap=t, feat_gap=f, critic_gap=q))
    return pd.DataFrame(rows)


def test_phase_table_means_signs_and_correlation():
    t = RV.phase_table(_per_state())
    assert len(t) == 6 and set(t["bank"].astype(str)) == {"approach", "inhand", "carrying"}
    fb = t[t["critic"] == "fb_tdj"].iloc[0]
    assert fb["n"] == 12 and fb["true_gap_mean"] == pytest.approx(0.75)
    assert fb["feat_sign_ok"] == 1.0 and fb["critic_sign_ok"] == 1.0
    assert fb["corr_feat_critic"] > 0.9 and fb["spearman_feat_critic"] > 0.8
    assert fb["feat_over_true"] == pytest.approx(0.05, abs=0.02)
    d = t[t["critic"] == "density_fb_z"].iloc[0]
    assert d["critic_over_true"] == pytest.approx(0.3, abs=0.02)
    ts = RV.phase_table(_per_state(), by_seed=True)
    assert len(ts) == 12 and (ts["n"] == 6).all()


def test_phase_table_sign_fraction_counts_negatives():
    p = _per_state()
    p.loc[(p["critic"] == "fb_tdj") & (p["state"] < 3), "critic_gap"] *= -1
    t = RV.phase_table(p)
    assert t[t["critic"] == "fb_tdj"]["critic_sign_ok"].tolist() == pytest.approx([0.5, 0.5, 0.5])
    assert (t[t["critic"] == "density_fb_z"]["critic_sign_ok"] == 1.0).all()
