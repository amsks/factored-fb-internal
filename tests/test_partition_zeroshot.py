"""Tests for scripts/rescuing_fb/eval_partition_zeroshot.py (spec 17b sec.3 zero-shot
readout: `PAPER/rescuing-fb/specs/17b-delta-partition-impl.md`, and
`PAPER/rescuing-fb/specs/17-delta-measure.md` sec.1 variant B).

Pure numpy — no env, no checkpoint, no jax (the script stays jax-free at module
level, same discipline as tests/test_eval_zeroshot_rewards.py). Covers:
  (a) the readout algebra `compute_Qr` / `compute_Qr_from_dots` against a
      hand-built numpy reference, including the degenerate constant-reward
      pin (F^Delta sums to zero on the simplex, so Q_r collapses to c/(1-gamma))
      — this is the sign/normalization regression guard the task asked for;
  (b) prototype digest verification: the real canonical file in the repo
      (results/delta_protos/cube_single_K256_s0.npy) must match the pinned
      sha256, a corrupted/permuted copy must be DETECTED and reported, and a
      missing file must fail loudly (not silently regenerate);
  (c) draw_region_centers_with_obs: the same hindsight/min_pos semantics as
      eval_zeroshot_rewards.draw_region_centers, PLUS the source observation;
  (d) the script imports and --help runs (subprocess, no jax needed);
  (e) select_retrieval_candidate: the jax-free half of the --decode retrieval
      selection algorithm (top-1 and top-k-then-nearest-centroid);
  (f) decode-mode wiring: --decode is argparse `choices`-restricted (no
      unknown string can silently reach main()), 'retrieval' and 'qr_greedy'
      route through genuinely DIFFERENT functions (decode_goal_retrieval +
      batched_goal_rollout vs batched_qr_rollout — never both, never neither),
      verified via source inspection of `rollout_for`'s two branches so a
      future refactor that collapses them to one silent fallthrough is caught.
"""
import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "rescuing_fb" / "eval_partition_zeroshot.py"
CANONICAL_PROTOS = REPO / "results" / "delta_protos" / "cube_single_K256_s0.npy"


@pytest.fixture(scope="module")
def pzs():
    spec = importlib.util.spec_from_file_location("eval_partition_zeroshot", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # numpy-only at module level (jax is lazy)
    return mod


@pytest.fixture(scope="module")
def geom():
    sys.path.insert(0, str(REPO / "scripts" / "rescuing_fb"))
    import env_geometry
    return env_geometry


# ============================ (a) readout algebra =================================

def test_compute_Qr_matches_handbuilt_numpy_reference(pzs):
    """K=3 simplex vectors chosen so the reference can be computed by hand:
    p=[0.5,0.3,0.2], phi(x0)=[0.2,0.2,0.6], z_r=[1.0,0.0,2.0], r(x0)=0.4,
    gamma=0.9.
    F^Delta = p - phi(x0) = [0.3, 0.1, -0.4]
    F^Delta . z_r = 0.3*1.0 + 0.1*0.0 + (-0.4)*2.0 = 0.3 - 0.8 = -0.5
    Q_r = (0.4 + (-0.5)) / (1 - 0.9) = (-0.1) / 0.1 = -1.0
    """
    p = np.array([0.5, 0.3, 0.2])
    phi_x0 = np.array([0.2, 0.2, 0.6])
    z_r = np.array([1.0, 0.0, 2.0])
    r_x0 = 0.4
    gamma = 0.9
    q = pzs.compute_Qr(r_x0, p, phi_x0, z_r, gamma)
    assert q == pytest.approx(-1.0, abs=1e-10)

    # independent numpy re-derivation (not the same code path as compute_Qr's
    # internals) as a second check against a copy/transcription error above.
    F_delta = p - phi_x0
    q_ref = (r_x0 + float(np.dot(F_delta, z_r))) / (1.0 - gamma)
    assert q == pytest.approx(q_ref, abs=1e-12)


def test_compute_Qr_batched(pzs):
    """p/phi_x0 as [N,K] batches broadcast correctly against a shared z_r/gamma,
    matching a per-row loop over the scalar formula."""
    rng = np.random.default_rng(0)
    N, K = 10, 6
    logits_p = rng.normal(size=(N, K))
    logits_phi = rng.normal(size=(N, K))
    p = np.exp(logits_p) / np.exp(logits_p).sum(-1, keepdims=True)
    phi_x0 = np.exp(logits_phi) / np.exp(logits_phi).sum(-1, keepdims=True)
    z_r = rng.normal(size=K)
    r_x0 = rng.normal(size=N)
    gamma = 0.95
    q = pzs.compute_Qr(r_x0, p, phi_x0, z_r, gamma)
    assert q.shape == (N,)
    q_loop = np.array([
        (r_x0[i] + float(np.dot(p[i] - phi_x0[i], z_r))) / (1 - gamma)
        for i in range(N)
    ])
    np.testing.assert_allclose(q, q_loop, atol=1e-10)


def test_compute_Qr_degenerate_constant_reward(pzs):
    """DEGENERATE PIN (task's sign/normalization regression guard): if the
    reward is a CONSTANT c over every prototype AND at x0, z_r = c*ones(K), so
    F^Delta . z_r = c*(sum(p) - sum(phi_x0)) = c*(1 - 1) = 0 because both p and
    phi(x0) are on the simplex (sum to 1) -- regardless of K, of p's and
    phi_x0's actual entries, or of gamma. Q_r must equal EXACTLY c/(1-gamma)."""
    rng = np.random.default_rng(1)
    for K in (2, 3, 8, 256):
        for c in (0.0, 1.0, -0.5, 3.7):
            logits_p = rng.normal(size=K)
            logits_phi = rng.normal(size=K)
            p = np.exp(logits_p) / np.exp(logits_p).sum()
            phi_x0 = np.exp(logits_phi) / np.exp(logits_phi).sum()
            z_r = np.full(K, c)
            gamma = 0.9
            q = pzs.compute_Qr(c, p, phi_x0, z_r, gamma)
            assert q == pytest.approx(c / (1.0 - gamma), abs=1e-9), (K, c, q)

    # a NON-simplex p/phi_x0 (e.g. an implementation bug that forgets to
    # normalize) must NOT hit the degenerate identity -- sanity that the test
    # above is actually discriminating, not vacuously true for any p/phi_x0.
    p_bad = np.array([0.5, 0.5, 0.5])       # sums to 1.5, not on the simplex
    phi_bad = np.array([0.2, 0.2, 0.2])     # sums to 0.6
    z_r = np.full(3, 2.0)
    q_bad = pzs.compute_Qr(2.0, p_bad, phi_bad, z_r, 0.9)
    assert q_bad != pytest.approx(2.0 / 0.1, abs=1e-9)


def test_compute_Qr_from_dots_matches_compute_Qr(pzs):
    """The rollout's algebraic shortcut (pre-dotted p.z_r and phi(x0).z_r,
    computed once per candidate anyway) must be EXACTLY equivalent to the
    full-vector formula on random inputs -- proves the two code paths in
    eval_partition_zeroshot.py (compute_Qr for tests/diagnostics,
    compute_Qr_from_dots inside the hot rollout loop) never diverge."""
    rng = np.random.default_rng(2)
    for _ in range(20):
        K = rng.integers(2, 32)
        p = rng.normal(size=K); p = np.exp(p) / np.exp(p).sum()
        phi_x0 = rng.normal(size=K); phi_x0 = np.exp(phi_x0) / np.exp(phi_x0).sum()
        z_r = rng.normal(size=K)
        r_x0 = float(rng.normal())
        gamma = float(rng.uniform(0.8, 0.999))
        q_full = pzs.compute_Qr(r_x0, p, phi_x0, z_r, gamma)
        q_dots = pzs.compute_Qr_from_dots(
            r_x0, float(np.dot(p, z_r)), float(np.dot(phi_x0, z_r)), gamma)
        assert q_full == pytest.approx(q_dots, abs=1e-10)


# ============================ (b) prototype digest verification ===================

def test_canonical_protos_file_matches_pinned_digest(pzs):
    """The real prototype file this task was briefed against
    (results/delta_protos/cube_single_K256_s0.npy) must match the sha256
    baked into CANONICAL_PROTOS_SHA256 -- if this test starts failing, either
    the file was regenerated (the exact failure mode the script guards
    against) or the registry constant is stale."""
    if not CANONICAL_PROTOS.exists():
        pytest.skip("canonical prototype file not present on this checkout")
    rel = "results/delta_protos/cube_single_K256_s0.npy"
    assert rel in pzs.CANONICAL_PROTOS_SHA256
    digest = pzs.file_sha256(CANONICAL_PROTOS)
    assert digest == pzs.CANONICAL_PROTOS_SHA256[rel]
    protos, info = pzs.load_protos_verified(str(CANONICAL_PROTOS))
    assert info["matched"] is True
    assert info["sha256"] == digest
    assert protos.shape == (256, 3)     # K=256, cube-single d_c=3 (19:22:0.4)
    assert protos.dtype == np.float32


def test_protos_digest_mismatch_is_detected(pzs, tmp_path):
    """A permuted copy of the canonical file (same rows, different order --
    exactly the 'silently regenerated or permuted prototype file' failure
    mode the module docstring names) has a DIFFERENT sha256 and must be
    refused when checked against the canonical path's expected digest."""
    if not CANONICAL_PROTOS.exists():
        pytest.skip("canonical prototype file not present on this checkout")
    protos = np.load(CANONICAL_PROTOS)
    permuted = protos[::-1].copy()          # same rows, reversed order
    assert not np.array_equal(permuted, protos)
    bad_path = tmp_path / "cube_single_K256_s0.npy"   # same basename, different dir
    np.save(bad_path, permuted)
    real_digest = pzs.file_sha256(CANONICAL_PROTOS)
    bad_digest = pzs.file_sha256(bad_path)
    assert bad_digest != real_digest
    # explicit expected digest (as if verifying against the canonical registry
    # for a path that doesn't literally resolve under REPO, e.g. a copy in a
    # scratch dir) -- must raise, not silently pass.
    with pytest.raises(AssertionError):
        pzs.verify_protos_file(bad_path, expected_sha256=real_digest)
    # without an expected digest, the mismatch is reported (matched=False is
    # impossible here since no digest was known: matched must be None, i.e.
    # "not checked", not silently "assumed OK").
    info = pzs.verify_protos_file(bad_path, expected_sha256="")
    assert info["matched"] is None
    assert info["sha256"] == bad_digest


def test_protos_missing_file_fails_loudly(pzs, tmp_path):
    missing = tmp_path / "does_not_exist.npy"
    with pytest.raises(FileNotFoundError):
        pzs.verify_protos_file(missing)
    with pytest.raises(FileNotFoundError):
        pzs.load_protos_verified(str(missing))


def test_protos_correct_copy_at_canonical_relpath_matches(pzs, tmp_path):
    """A byte-identical COPY of the canonical file, placed at the same
    repo-relative path structure, matches the registry (sanity: the digest
    check is about file CONTENT, not merely a fixed absolute path -- the
    registry keys on the repo-relative path string, so this test exercises
    the relative-path resolution too by copying to REPO itself under a throwaway
    name is unnecessary; instead check the canonical path directly resolves)."""
    if not CANONICAL_PROTOS.exists():
        pytest.skip("canonical prototype file not present on this checkout")
    info = pzs.verify_protos_file(CANONICAL_PROTOS)
    assert info["matched"] is True
    assert info["expected_sha256"] == pzs.CANONICAL_PROTOS_SHA256[
        "results/delta_protos/cube_single_K256_s0.npy"]


# ============================ (c) region centers with obs =========================

def test_draw_region_centers_with_obs_hindsight_and_min_pos(pzs, geom):
    """Same hindsight/min_pos semantics as
    eval_zeroshot_rewards.draw_region_centers (spec 14 sec.2.1/3.3), PLUS the
    source observation is returned and its coords match the accepted center."""
    spec = geom.HitSpec.parse("0:2:0.4")
    rng0 = np.random.default_rng(0)
    ep0 = rng0.normal(0.0, 0.05, size=(50, 2))
    ep1 = np.linspace(50.0, 60.0, 50)[:, None] * np.ones((1, 2))
    coords_all = np.concatenate([ep0, ep1])
    # obs_all carries the coord in cols [0:2] plus an extra column so we can
    # confirm the FULL observation (not just the coord slice) round-trips.
    obs_all = np.concatenate([coords_all, np.arange(100)[:, None].astype(np.float64)],
                             axis=1)
    terminals = np.zeros(100)
    terminals[49] = terminals[99] = 1.0
    bounds = geom.episode_bounds(terminals)
    relabel = rng0.normal(0.0, 0.05, size=(400, 2))     # covers ep0 only
    rng = np.random.default_rng(3)
    centers, centers_obs, npos, rejected = pzs.draw_region_centers_with_obs(
        obs_all, coords_all, bounds, spec, relabel, rng, k=4, min_pos=5, max_tries=500)
    assert len(centers) == 4
    assert centers_obs.shape == (4, 3)
    assert (npos >= 5).all()
    assert rejected > 0
    for c, o in zip(centers, centers_obs):
        # the returned observation's own coord slice matches the center exactly,
        # up to the float32 storage cast draw_region_centers_with_obs applies
        # (matching the repo's obs dtype convention throughout) -- the center
        # WAS this observation's coord, by construction.
        np.testing.assert_allclose(o[:2], c, atol=1e-6)
        # and the observation is one of the dataset's own rows (visited state).
        assert np.min(np.linalg.norm(obs_all - o, axis=1)) < 1e-6


def test_draw_region_centers_with_obs_no_candidates_returns_empty(pzs, geom):
    """Every start row is rejected (min_pos impossibly high) -> empty arrays,
    not a crash, with the right shapes for downstream code."""
    spec = geom.HitSpec.parse("0:2:0.4")
    coords_all = np.random.default_rng(5).normal(size=(20, 2))
    obs_all = np.concatenate([coords_all, np.zeros((20, 1))], axis=1)
    terminals = np.zeros(20)
    terminals[19] = 1.0
    bounds = geom.episode_bounds(terminals)
    rng = np.random.default_rng(6)
    centers, centers_obs, npos, rejected = pzs.draw_region_centers_with_obs(
        obs_all, coords_all, bounds, spec, coords_all, rng, k=3, min_pos=10**9,
        max_tries=50)
    assert len(centers) == 0
    assert centers_obs.shape == (0, 3)
    assert rejected == 50


# ============================ (d) import + --help ================================

def test_script_help_runs():
    proc = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    for flag in ("--run-dir", "--checkpoint-step", "--families", "--episodes",
                 "--horizon", "--seed", "--out", "--n-candidates",
                 "--expected-protos-sha256"):
        assert flag in proc.stdout, f"{flag} missing from --help"


def test_parser_contract_defaults(pzs):
    ap = pzs.build_parser()
    ns = ap.parse_args(["--run-dir", "/tmp/x"])
    assert ns.episodes == 25 and ns.horizon == 0
    assert ns.tasks == "1,2,3,4,5"
    assert set(ns.families.split(",")) == set(pzs.FAMILIES)
    assert ns.n_candidates == 8
    assert ns.protos_path == "" and ns.expected_protos_sha256 == ""


def test_families_constant_excludes_zinference_only_families(pzs):
    """mixture/linear/reachavoid are deliberately NOT in FAMILIES (module
    docstring explains why) -- pin the set so a future edit that silently
    widens/narrows it gets caught."""
    assert set(pzs.FAMILIES) == {"dirac", "dense", "region"}


# ============================ (e) retrieval selection (jax-free half) =============

def test_select_retrieval_candidate_top1(pzs):
    """k=1 (the run_spec14_jax_table.sh gate protocol default) returns the
    single highest-scoring candidate's index directly -- no averaging."""
    scores = np.array([0.1, 0.9, 0.4, -0.2])
    coords = np.array([[0.0], [5.0], [1.0], [-3.0]])
    obs = np.array([[0.0, 0.0], [5.0, 5.0], [1.0, 1.0], [-3.0, -3.0]])
    chosen, info = pzs.select_retrieval_candidate(scores, coords, obs, retrieval_k=1)
    assert chosen == 1
    assert info["retrieval_k"] == 1
    assert info["n_candidates"] == 4
    assert info["score_chosen"] == pytest.approx(0.9)
    assert info["top1_idx"] == 1


def test_select_retrieval_candidate_topk_nearest_centroid(pzs):
    """k>1: average the top-k candidates' COORDINATES, return the dataset
    candidate NEAREST that average -- constructed so the average sits closer
    to a THIRD point than to either top-k member, pinning the "nearest to the
    centroid" rule (not e.g. "return the better of the top-k")."""
    # top-2 by score are indices 0 (coord=0.0) and 2 (coord=4.0) -> centroid=2.0.
    # index 1 sits exactly at 2.0 (closest), despite NOT being in the top-2 by score.
    scores = np.array([0.9, 0.1, 0.8, 0.05])
    coords = np.array([[0.0], [2.0], [4.0], [10.0]])
    obs = coords.repeat(2, axis=1)   # any obs shape works; unused by the selector math
    chosen, info = pzs.select_retrieval_candidate(scores, coords, obs, retrieval_k=2)
    assert chosen == 1
    assert info["retrieval_k"] == 2
    assert info["top1_idx"] == 0                 # highest scorer alone


def test_select_retrieval_candidate_k_clamped_to_n(pzs):
    """retrieval_k larger than the candidate pool clamps to n, not a crash."""
    scores = np.array([0.5, 0.2])
    coords = np.array([[0.0], [1.0]])
    obs = np.array([[0.0], [1.0]])
    chosen, info = pzs.select_retrieval_candidate(scores, coords, obs, retrieval_k=100)
    assert info["retrieval_k"] == 2
    assert chosen in (0, 1)


def test_select_retrieval_candidate_shape_mismatch_raises(pzs):
    with pytest.raises(AssertionError):
        pzs.select_retrieval_candidate([0.1, 0.2], [[0.0], [1.0], [2.0]], [[0.0], [1.0]])


# ============================ (f) decode-mode wiring (anti-fallthrough) ============

def test_decode_choices_reject_unknown_value():
    """argparse `choices` is the first line of defense: an unrecognized
    --decode value must be a hard parse error, not a silent fallthrough to
    whichever branch happens to be the `else`."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-dir", "/tmp/x", "--decode", "bogus"],
        capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0
    assert "invalid choice" in proc.stderr or "invalid choice" in proc.stdout


def test_decodes_constant_matches_parser_choices(pzs):
    ap = pzs.build_parser()
    ns_default = ap.parse_args(["--run-dir", "/tmp/x"])
    assert ns_default.decode == "retrieval"          # default per the task brief
    ns_qr = ap.parse_args(["--run-dir", "/tmp/x", "--decode", "qr_greedy"])
    assert ns_qr.decode == "qr_greedy"
    assert set(pzs.DECODES) == {"retrieval", "qr_greedy"}


def test_decode_functions_are_distinct_objects(pzs):
    """The two decode modes must be wired to genuinely different callables --
    not the same function under two names, not one calling the other as a
    no-op wrapper."""
    assert pzs.decode_goal_retrieval is not pzs.batched_qr_rollout
    assert pzs.batched_goal_rollout is not pzs.batched_qr_rollout
    assert pzs.decode_goal_retrieval is not pzs.batched_goal_rollout
    names = {pzs.decode_goal_retrieval.__name__, pzs.batched_goal_rollout.__name__,
            pzs.batched_qr_rollout.__name__}
    assert len(names) == 3
    # genuinely different bodies (not e.g. one being `def a(...): return b(...)`
    # with an identical source after the signature line).
    src_retrieval = inspect.getsource(pzs.decode_goal_retrieval)
    src_goal_rollout = inspect.getsource(pzs.batched_goal_rollout)
    src_qr_rollout = inspect.getsource(pzs.batched_qr_rollout)
    assert src_retrieval != src_qr_rollout
    assert src_goal_rollout != src_qr_rollout
    assert "select_retrieval_candidate" in src_retrieval
    assert "select_retrieval_candidate" not in src_qr_rollout


def test_rollout_for_dispatch_wires_each_decode_to_its_own_functions(pzs):
    """Source-inspect main()'s `rollout_for` dispatcher (the ONLY branch point
    between the two decode modes): the qr_greedy branch must call
    batched_qr_rollout and must NOT call decode_goal_retrieval/
    batched_goal_rollout; the retrieval branch must call BOTH
    decode_goal_retrieval and batched_goal_rollout and must NOT call
    batched_qr_rollout. Catches the exact failure mode named in the task
    brief: a mode string that silently falls through to one implementation
    regardless of which value was requested."""
    src = inspect.getsource(pzs.main)
    start = src.index("def rollout_for(")
    end = src.index("\n    results = {", start)
    body = src[start:end]
    qr_branch_start = body.index('if args.decode == "qr_greedy":')
    retrieval_branch_start = body.index('assert args.decode == "retrieval"', qr_branch_start)
    qr_branch = body[qr_branch_start:retrieval_branch_start]
    retrieval_branch = body[retrieval_branch_start:]

    assert "batched_qr_rollout(" in qr_branch
    assert "decode_goal_retrieval(" not in qr_branch
    assert "batched_goal_rollout(" not in qr_branch

    assert "decode_goal_retrieval(" in retrieval_branch
    assert "batched_goal_rollout(" in retrieval_branch
    assert "batched_qr_rollout(" not in retrieval_branch


def test_decode_recorded_in_meta_and_family_records():
    """meta.decode and each family record's decode_info are populated (source
    check: main() must set results['meta']['decode'] and thread `decode_info`
    into every family's per-cell `rec` -- a static grep, since running main()
    needs jax/a checkpoint)."""
    src = SCRIPT.read_text()
    assert '"decode": args.decode' in src
    assert 'rec = {"diag"' in src and '"decode": decode_info' in src
