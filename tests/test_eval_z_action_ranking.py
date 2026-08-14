"""Tests for scripts/rescuing_fb/eval_z_action_ranking.py (spec 28 P0a --
the action-ranking decomposition). Pure numpy; the script's jax pieces are
lazy so the module imports without jax/ogbench (same discipline as
tests/test_eval_density_z_rewards.py).

Pins the four things a wrong number could hide behind:
  (a) the rank statistics (spearman / Mann-Whitney AUC) incl. their CHANCE
      values and tie handling,
  (b) `readouts_from_logits`: both readouts really do share ONE weight
      vector, kg is renormalized over the restricted pool, and the exact
      softmax-prefix property the MC half-pool floor relies on,
  (c) the MC floor's own behaviour (pure-noise ranking => selfcons ~ 0),
  (d) the PRE-REGISTERED control gate: CG1/CG2 blocking, CG3 non-blocking,
      nan treated as FAIL.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "rescuing_fb" / "eval_z_action_ranking.py"


@pytest.fixture(scope="module")
def m():
    spec = importlib.util.spec_from_file_location("eval_z_action_ranking", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============ (a) rank statistics ==================================================

def test_spearman_monotone_and_reversed(m):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert m.spearman(x, 2 * x + 1) == pytest.approx(1.0)
    assert m.spearman(x, -x) == pytest.approx(-1.0)
    # monotone but nonlinear -> still exactly 1 (rank statistic)
    assert m.spearman(x, np.exp(x)) == pytest.approx(1.0)


def test_spearman_constant_is_nan_not_zero(m):
    """A constant readout has an UNDEFINED correlation. Silently returning
    0.0 would read as 'the two readouts disagree' -- exactly the false
    'readout indicted' verdict this probe must not manufacture."""
    assert np.isnan(m.spearman(np.ones(6), np.arange(6.0)))


def test_spearman_matches_scipy_reference_on_ties(m):
    x = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 4.0])
    y = np.array([2.0, 3.0, 3.0, 1.0, 5.0, 5.0])
    # average-rank Pearson, computed independently here
    def rank(v):
        r = np.empty(len(v))
        order = np.argsort(v)
        sv = v[order]
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and sv[j + 1] == sv[i]:
                j += 1
            r[order[i:j + 1]] = np.mean(np.arange(i + 1, j + 2))
            i = j + 1
        return r
    ref = np.corrcoef(rank(x), rank(y))[0, 1]
    assert m.spearman(x, y) == pytest.approx(ref, abs=1e-12)


def test_auc_chance_separation_and_ties(m):
    s = np.array([0.0, 1.0, 2.0, 3.0])
    lab = np.array([0, 0, 1, 1], bool)
    assert m.auc_mw(s, lab) == pytest.approx(1.0)
    assert m.auc_mw(-s, lab) == pytest.approx(0.0)
    # all-tied scores => exactly chance, which is 0.5 (never nan, never 1.0)
    assert m.auc_mw(np.ones(4), lab) == pytest.approx(0.5)
    # one empty class => undefined
    assert np.isnan(m.auc_mw(s, np.zeros(4, bool)))


def test_auc_random_labels_near_chance(m):
    rng = np.random.default_rng(0)
    aucs = [m.auc_mw(rng.normal(size=64), rng.permutation(np.arange(64) < 16))
            for _ in range(200)]
    assert abs(np.mean(aucs) - 0.5) < 0.02


def test_action_labels_actor_unlabelled(m):
    lab = m.action_labels(["traj", "dsrand", "actor", "unif"])
    assert lab.tolist() == [1, 0, -1, 0]


# ============ (b) the two readouts share ONE weight vector =========================

def test_readouts_share_weights(m):
    """q_g and q_z must differ ONLY in the reward vector -- the whole point of
    P0a. Verified by recomputing both from the softmax by hand."""
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(2, 5, 7))
    kg = rng.random(7); kg /= kg.sum()
    rz = rng.normal(size=7)
    out = m.readouts_from_logits(logits, kg, rz)
    w = m.softmax_np(logits, axis=-1)
    assert np.allclose(out["q_g_minhead"], (w @ kg).min(axis=0))
    assert np.allclose(out["q_z_minhead"], (w @ rz).min(axis=0))
    wm = w.min(axis=0)
    assert np.allclose(out["q_g_wmin"], wm @ kg)
    assert np.allclose(out["q_z_wmin"], wm @ rz)


def test_readout_slice_is_exact_subset_softmax(m):
    """The MC half-pool floor rests on this: restricting a softmax to a subset
    and renormalizing == computing the softmax over that subset from scratch,
    and kg (itself a normalized kernel row) is renormalized the same way."""
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(2, 4, 10))
    kg = rng.random(10); kg /= kg.sum()
    rz = rng.normal(size=10)
    sub = slice(0, 5)
    got = m.readouts_from_logits(logits, kg, rz, sub)
    ref = m.readouts_from_logits(logits[..., sub], kg[sub] / kg[sub].sum(), rz[sub])
    for k in got:
        assert np.allclose(got[k], ref[k]), k


def test_span_stats(m):
    q = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    st = m.span_stats(q)
    assert st["span"] == pytest.approx(4.0)
    assert st["iqr"] == pytest.approx(2.0)


# ============ (c) MC floor =========================================================

def test_mc_floor_pure_noise_has_no_self_consistency(m):
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=64), rng.normal(size=64)
    f = m.mc_floor(a, b)
    assert abs(f["selfcons"]) < 0.3
    assert f["snr"] < 12.0


def test_mc_floor_snr_ignores_a_common_mode_offset(m):
    """The correction the g-anchor smoke forced: a constant shift between the
    two halves (which pool points landed where) must NOT be counted as noise --
    the ranking is blind to it. The uncentered variant is kept and MUST see it,
    so the difference between the two stays auditable."""
    rng = np.random.default_rng(6)
    base = np.linspace(0.0, 1.0, 64)
    a = base + 0.001 * rng.normal(size=64)
    b = base + 0.001 * rng.normal(size=64) + 50.0      # pure common mode
    f = m.mc_floor(a, b)
    assert f["snr"] > 100.0, "centered snr must ignore the offset"
    assert f["snr_uncentered"] < 1.0, "uncentered snr must be destroyed by it"
    assert f["selfcons"] > 0.99


def test_mc_floor_real_signal(m):
    rng = np.random.default_rng(4)
    base = np.linspace(0, 10, 64)
    f = m.mc_floor(base + 0.01 * rng.normal(size=64), base + 0.01 * rng.normal(size=64))
    assert f["selfcons"] > 0.99
    assert f["snr"] > 100.0


# ============ (d) the PRE-REGISTERED control gate ==================================

def _summary(**kw):
    base = {"auc_g_vs_unif_median": 0.80, "selfcons_g_median": 0.99,
            "snr_g_median": 20.0, "delta_over_span_g_median": 0.20}
    base.update(kw)
    return base


def test_gate_passes_on_a_healthy_anchor(m):
    ok, rows = m.gate_p0a(_summary())
    assert ok and all(r["ok"] for r in rows)


@pytest.mark.parametrize("key,val", [("auc_g_vs_unif_median", 0.55),
                                     ("selfcons_g_median", 0.10),
                                     ("snr_g_median", 1.2)])
def test_gate_blocks_on_a_dead_instrument(m, key, val):
    ok, rows = m.gate_p0a(_summary(**{key: val}))
    assert not ok
    assert any((not r["ok"]) and r["blocking"] for r in rows)


def test_cg3_is_reported_but_not_blocking(m):
    ok, rows = m.gate_p0a(_summary(delta_over_span_g_median=0.95))
    assert ok, "CG3 must not block -- it gates only the Delta_actor branch"
    cg3 = [r for r in rows if r["check"] == "CG3_delta_over_span_g"][0]
    assert (not cg3["ok"]) and (not cg3["blocking"])


def test_gate_nan_is_a_failure_not_a_pass(m):
    ok, rows = m.gate_p0a(_summary(selfcons_g_median=float("nan")))
    assert not ok


def test_gate_missing_field_fails(m):
    s = _summary()
    del s["snr_g_median"]
    ok, _ = m.gate_p0a(s)
    assert not ok


# ============ pair sampling ========================================================

def test_sample_pairs_on_traj_state_precedes_goal_in_same_episode(m):
    bounds = np.array([[0, 200], [200, 500], [500, 560]])
    rng = np.random.default_rng(5)
    pairs = m.sample_pairs(bounds, 30, 10, 560, rng)
    assert len(pairs) == 40
    assert sum(p["on_traj"] for p in pairs) == 30
    for p in pairs:
        if not p["on_traj"]:
            continue
        s, g = p["s_idx"], p["g_idx"]
        ep = [b for b in bounds if b[0] <= s < b[1]][0]
        assert s < g < ep[1], "on-traj goal must follow s inside the SAME episode"


def test_sample_pairs_rejects_short_episodes(m):
    bounds = np.array([[0, 5], [5, 9]])
    with pytest.raises(AssertionError):
        m.sample_pairs(bounds, 2, 0, 9, np.random.default_rng(0))


# ============ z geometry (collapse vs spread) ======================================

def test_z_geometry_flags_collapse_against_the_sphere_reference(m):
    """A collapsed index must be visibly distinguishable from the sqrt(d)-sphere
    prior it is supposed to live on -- otherwise 'the actor does not move when
    the goal changes' cannot be told apart from 'z(g) never changes'."""
    rng = np.random.default_rng(11)
    d, n = 50, 512
    spread = rng.normal(size=(n, d))
    spread *= np.sqrt(d) / np.linalg.norm(spread, axis=1, keepdims=True)
    g_spread = m.z_geometry(spread, np.random.default_rng(0))
    assert abs(g_spread["cos_mean"]) < 0.1
    assert g_spread["dist_median"] == pytest.approx(np.sqrt(2 * d), rel=0.15)

    collapsed = np.tile(spread[:1], (n, 1)) + 1e-3 * rng.normal(size=(n, d))
    collapsed *= np.sqrt(d) / np.linalg.norm(collapsed, axis=1, keepdims=True)
    g_col = m.z_geometry(collapsed, np.random.default_rng(0))
    assert g_col["cos_mean"] > 0.99
    assert g_col["dist_median"] < 0.05 * g_col["sphere_ref_dist_expected"]
