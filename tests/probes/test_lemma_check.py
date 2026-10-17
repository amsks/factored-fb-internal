"""The linear algebra behind analysis/lemma_check/run.py, on a random successor-like matrix."""
import importlib.util
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
# Load under a distinct name; family.py imports another module named run.
_spec = importlib.util.spec_from_file_location("lemma_check_run", REPO / "analysis" / "lemma_check" / "run.py")
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)


def _mdp(seed=0, n=9):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(n), size=2 * n)              # rows of (1-gamma) M, mass 1
    gamma = 0.8
    m = P / (1 - gamma)
    r = np.zeros(n)
    r[n - 1] = 1.0
    q = m @ r
    return dict(n=n, m_exact=m, r=r, decision=3, gap=float(q[6] - q[7]))


def test_projection_identity_and_split():
    mdp = _mdp()
    rng = np.random.default_rng(1)
    for d in (2, 3, 5):
        B = rng.normal(size=(mdp["n"], d))
        X = lc.ratio_matrix(mdp)
        F = X @ B @ np.linalg.inv(B.T @ B)                # least-squares F given B
        assert abs(lc.trained_gap(mdp, F, B, mdp["r"]) - lc.lemma_gap(mdp, B, mdp["r"])) < 1e-10
        assert abs(lc.lemma_gap(mdp, B, mdp["r"]) + lc.discarded_gap(mdp, B, mdp["r"]) - mdp["gap"]) < 1e-10


def test_lemma_is_linear_in_reward():
    mdp = _mdp(2)
    B = np.random.default_rng(3).normal(size=(mdp["n"], 3))
    r1, r2 = np.random.default_rng(4).uniform(size=(2, mdp["n"]))
    assert abs(lc.lemma_gap(mdp, B, 2 * r1 - r2) - (2 * lc.lemma_gap(mdp, B, r1) - lc.lemma_gap(mdp, B, r2))) < 1e-10


def test_appending_residual_restores_every_reward():
    mdp = _mdp(5)
    B = np.random.default_rng(6).normal(size=(mdp["n"], 3))
    Ba = lc.augmented_features(mdp, B)
    assert Ba.shape[1] == 4
    for r in np.random.default_rng(7).uniform(size=(20, mdp["n"])):
        assert abs(lc.lemma_gap(mdp, Ba, r) - lc.true_gap(mdp, r)) < 1e-10
    assert lc.objective(mdp, Ba) <= lc.objective(mdp, B) + 1e-12
