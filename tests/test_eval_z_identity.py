"""Tests for scripts/rescuing_fb/eval_z_identity.py -- the actor-independent
z(g)-identity retrieval probe (coordinator follow-up, 2026-08-13, spec 24
E1/E2/G4 chain). Pure numpy -- no jax, no env, mirrors this repo's usual
jax-free-at-module-level discipline for the algebra layer.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "rescuing_fb" / "eval_z_identity.py"
sys.path.insert(0, str(REPO / "scripts" / "rescuing_fb"))
from env_geometry import HitSpec  # noqa: E402


@pytest.fixture(scope="module")
def zid():
    spec = importlib.util.spec_from_file_location("eval_z_identity", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _toy_clustered(rng, n_clusters: int, per_cluster: int, z_dim: int = 5,
                   z_noise: float = 0.0, coord_dim: int = 2, coord_spacing: float = 100.0):
    """A toy pool that decouples coordinate space from z-space the way a REAL
    trained encoder does (z is a learned DIRECTION, not the raw coordinate --
    using raw coordinates as z under cosine similarity is invalid: cosine
    depends on angle-from-origin, not Euclidean proximity, so collinear
    coordinate clusters would spuriously "collide"). `n_clusters` coordinate
    clusters, well-separated (spacing `coord_spacing`, collinearity doesn't
    matter here -- coordinate-space comparisons in `retrieval_accuracy` are
    Euclidean via HitSpec.contains, never cosine); each cluster gets ONE
    random unit direction in R^z_dim, shared by all `per_cluster` members
    (+ `z_noise` jitter) -- so same-cluster points are z-close (a "lossless,
    cluster-consistent" toy encoder) and different clusters are z-random
    (generically far apart in a 5+-dim space). Returns (coords, z)."""
    centers = np.arange(n_clusters)[:, None] * coord_spacing * np.ones((1, coord_dim))
    coords = np.repeat(centers, per_cluster, axis=0) + rng.normal(
        scale=0.01, size=(n_clusters * per_cluster, coord_dim))
    cluster_dirs = rng.normal(size=(n_clusters, z_dim))
    cluster_dirs /= np.linalg.norm(cluster_dirs, axis=1, keepdims=True) + 1e-12
    z = np.repeat(cluster_dirs, per_cluster, axis=0) + rng.normal(
        scale=z_noise, size=(n_clusters * per_cluster, z_dim))
    return coords, z


def test_retrieval_perfect_encoding_scores_top1(zid):
    """One point per cluster (no duplicate exists among the OTHER points) --
    top1/top8 must be 0: there is NO correct answer to retrieve, and a
    cluster-consistent-but-injective encoder must not manufacture a false hit."""
    spec = HitSpec.single(0, 2, 0.1)
    rng = np.random.default_rng(0)
    coords, z_pool = _toy_clustered(rng, n_clusters=50, per_cluster=1)
    out = zid.retrieval_accuracy(z_pool, coords, np.arange(50), spec, k_list=(1, 8))
    assert out["top1"] == pytest.approx(0.0)
    assert out["top8"] == pytest.approx(0.0)
    assert out["nearest_wrong_cosine_mean"] is not None


def test_retrieval_collision_encoding_scores_perfect(zid):
    """Two near-duplicate coordinates PER cluster (well inside eps), each
    cluster's z-direction distinct and z-noise negligible -- top1 must be 1.0:
    the in-cluster duplicate is both the coordinate-true AND z-nearest match."""
    spec = HitSpec.single(0, 2, 0.1)
    rng = np.random.default_rng(1)
    coords, z_pool = _toy_clustered(rng, n_clusters=10, per_cluster=2, z_noise=1e-5)
    out = zid.retrieval_accuracy(z_pool, coords, np.arange(20), spec, k_list=(1, 8))
    assert out["top1"] == pytest.approx(1.0)
    assert out["top8"] == pytest.approx(1.0)


def test_retrieval_lossy_encoding_scores_worse_than_lossless(zid):
    """A LOSSY encoder (every cluster collapsed to the SAME z direction,
    losing which cluster a point belongs to) must score a materially higher
    nearest-wrong-cosine than a lossless (cluster-distinct) encoder on the
    SAME coordinate layout -- the concrete behavior this probe exists to
    detect (real z(g) 'is z lossy' question)."""
    spec = HitSpec.single(0, 2, 0.1)
    rng = np.random.default_rng(2)
    coords, z_lossless = _toy_clustered(rng, n_clusters=20, per_cluster=3, z_noise=1e-4)
    n = len(coords)
    # lossy: EVERY point maps to (almost) the SAME single direction, regardless
    # of which cluster/coordinate it came from.
    z_lossy = np.ones((n, 5)) + rng.normal(scale=1e-6, size=(n, 5))
    q = np.arange(n)
    out_lossless = zid.retrieval_accuracy(z_lossless, coords, q, spec, k_list=(1,))
    out_lossy = zid.retrieval_accuracy(z_lossy, coords, q, spec, k_list=(1,))
    assert out_lossy["nearest_wrong_cosine_mean"] > 0.999
    assert out_lossless["nearest_wrong_cosine_mean"] < out_lossy["nearest_wrong_cosine_mean"]


def test_retrieval_self_excluded_not_trivially_perfect(zid):
    """Self-exclusion is load-bearing: without it every query would trivially
    retrieve itself (cosine 1.0, distance 0) and the probe would be vacuous.
    Pin that queries are never scored against their OWN pool row (already
    implicit in the collision/well-separated tests above, but pinned
    explicitly here via a single-point-per-cluster layout where a bug that
    forgot self-exclusion would silently still read top1=1.0)."""
    spec = HitSpec.single(0, 2, 0.05)
    rng = np.random.default_rng(3)
    coords = rng.uniform(-10, 10, size=(30, 2))     # all well-separated, unique
    z_pool = rng.normal(size=(30, 5))                # RANDOM z, unrelated to coords
    out = zid.retrieval_accuracy(z_pool, coords, np.arange(30), spec, k_list=(1,))
    # a random, coordinate-independent z on a well-separated layout must NOT
    # score top1=1.0 (that would only happen if self-match leaked in)
    assert out["top1"] < 1.0


def test_retrieval_rejects_shape_mismatch(zid):
    spec = HitSpec.single(0, 2, 0.1)
    with pytest.raises(AssertionError):
        zid.retrieval_accuracy(np.zeros((5, 3)), np.zeros((4, 2)), [0], spec)


def test_retrieval_rejects_empty_queries(zid):
    spec = HitSpec.single(0, 2, 0.1)
    with pytest.raises(AssertionError):
        zid.retrieval_accuracy(np.zeros((5, 3)), np.zeros((5, 2)), [], spec)


def test_retrieval_deterministic_repeat(zid):
    spec = HitSpec.single(0, 2, 0.1)
    rng = np.random.default_rng(4)
    coords = rng.uniform(-10, 10, size=(40, 2))
    z_pool = rng.normal(size=(40, 6))
    q = np.arange(0, 40, 3)
    out1 = zid.retrieval_accuracy(z_pool.copy(), coords.copy(), q.copy(), spec, k_list=(1, 8))
    out2 = zid.retrieval_accuracy(z_pool.copy(), coords.copy(), q.copy(), spec, k_list=(1, 8))
    assert out1 == out2


# ============================ import / --help / parser defaults ====================

def test_script_help_runs():
    proc = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    for flag in ("--run-dir", "--checkpoint-step", "--pool-size", "--n-queries",
                 "--seed", "--force", "--out"):
        assert flag in proc.stdout, f"{flag} missing from --help"


def test_parser_contract_defaults(zid):
    ap = zid.build_parser()
    ns = ap.parse_args(["--run-dir", "/tmp/x"])
    assert ns.pool_size == 16384
    assert ns.n_queries == 500
    assert ns.seed == 0
    assert ns.force is False


def test_check_density_consistency_reused_not_reimplemented(zid):
    """This script imports the base refusal (measure_density=True required)
    unchanged -- it deliberately does NOT add a z-specific branch (see module
    docstring 'CONTROL'), so a plain DensityMismatchError from the reused
    function must still fire for a non-density checkpoint."""
    with pytest.raises(zid.DensityMismatchError):
        zid.check_density_consistency(False, True, force=False, run_dir="<run>")
