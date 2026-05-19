import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import gciql_profile as gp
from evals.phase_probe import Thresholds


def test_parse_flags_reads_env_and_seed(tmp_path):
    (tmp_path / "flags.json").write_text(json.dumps(
        {"env_name": "cube-single-play-v0", "seed": 10,
         "agent": "agents/gciql.py"}))
    info = gp.parse_flags(tmp_path)
    assert info["env_name"] == "cube-single-play-v0"
    assert info["seed"] == 10


def _ep(outcome, d, v, grip, cube_z, table_z=0.02):
    T = len(d)
    cube = np.column_stack([np.zeros(T), np.zeros(T), np.asarray(cube_z)])
    return {
        "obs": np.zeros((T, 4), np.float32),
        "d": np.asarray(d, np.float64),
        "V": np.asarray(v, np.float64),
        "grip": np.asarray(grip, np.float64),
        "cube": cube,
        "eff": cube.copy(),                 # effector on the cube
        "goal": np.array([0.0, 0.0, 0.02]),
        "table_z": table_z, "success": outcome == "success",
        "task": "task1", "episode": 0,
    }


def test_episodes_to_frames_t1_t4_funnel():
    # 6 transport steps, V strictly rises as d -> 0  => rho ~ +1
    d = [0.30, 0.25, 0.20, 0.15, 0.10, 0.02]
    v = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    grip = [1.0] * 6
    cube_z = [0.20] * 6                    # lifted (table_z 0.02)
    eps = [_ep("success", d, v, grip, cube_z)]
    vl, cov, fun, vs = gp.episodes_to_frames(eps, Thresholds(),
                                             ref_obs=np.zeros((10, 4)))
    assert set(vl.columns) >= {"task", "episode", "outcome",
                               "rho_V_negd", "n_transport_steps"}
    assert vl.iloc[0]["rho_V_negd"] > 0.99
    assert vl.iloc[0]["outcome"] == "success"
    assert set(cov.columns) >= {"task", "outcome", "region", "nn_dist"}
    assert set(fun.columns) >= {"task", "episode", "furthest_phase",
                                "outcome"}
    assert set(vs.columns) == {"task", "episode", "outcome", "d", "V",
                               "transport"}
    assert len(vs) == 6                       # one row per step
    assert vs["transport"].dtype == bool


def test_episodes_to_frames_attributes_transport_fail_with_real_eff():
    # hand on the cube, cube lifted+gripped >=5 steps, never reaches
    # goal, env success False  ->  classify_phases must say transport_fail
    # (regression: zero-filled eff mislabels this as reach/other)
    T = 6
    cube = np.tile([0.40, 0.10, 0.20], (T, 1)).astype(np.float64)
    eff = cube.copy()                       # effector reaches+secures
    grip = np.ones(T)                       # gripper closed
    d = np.linspace(0.30, 0.10, T)          # cube never delivered
    V = np.linspace(0.0, 1.0, T)
    ep = {
        "obs": np.zeros((T, 4), np.float32),
        "eff": eff, "cube": cube, "grip": grip,
        "d": d, "V": V, "table_z": 0.02,
        "goal": np.array([0.0, 0.0, 0.02]),
        "success": False, "task": "task1", "episode": 0,
    }
    vl, _, fun, _ = gp.episodes_to_frames([ep], Thresholds(),
                                          ref_obs=np.zeros((10, 4)))
    assert vl.iloc[0]["outcome"] == "transport_fail"
    assert fun.iloc[0]["furthest_phase"] == "transport"


def test_build_argparser_has_required_flags():
    ap = gp.build_argparser()
    ns = ap.parse_args([
        "--run-dir", "/x/sd001", "--step", "1000000",
        "--out", "/tmp/o", "--tasks", "1,2,3", "--n-episodes", "4",
        "--dataset-path", "datasets/cube-single-play-v0"])
    assert ns.run_dir == "/x/sd001" and ns.step == 1000000
    assert ns.tasks == "1,2,3" and ns.n_episodes == 4


def _seed_dir(root, s):
    d = root / f"s{s}_final"
    d.mkdir(parents=True)
    pd.DataFrame([
        {"task": "task1", "episode": 0, "outcome": "success",
         "rho_V_negd": 0.6, "V_at_secure": 0.1, "V_at_end": 0.9,
         "n_transport_steps": 6},
        {"task": "task1", "episode": 1, "outcome": "transport_fail",
         "rho_V_negd": 0.0, "V_at_secure": 0.5, "V_at_end": 0.5,
         "n_transport_steps": 6},
    ]).to_parquet(d / "value_landscape.parquet")
    pd.DataFrame([
        {"task": "task1", "outcome": "transport_fail",
         "region": "transport", "nn_dist": 3.0},
        {"task": "task1", "outcome": "success",
         "region": "transport", "nn_dist": 1.0}],
    ).to_parquet(d / "coverage.parquet")
    pd.DataFrame([
        {"task": "task1", "episode": 0, "furthest_phase": "success",
         "outcome": "success"},
        {"task": "task1", "episode": 1, "furthest_phase": "transport",
         "outcome": "transport_fail"}],
    ).to_parquet(d / "phase_funnel.parquet")


def test_gciql_aggregate_and_comparison(tmp_path):
    import importlib
    agg = importlib.import_module("scripts.gciql_profile_aggregate")
    groot = tmp_path / "gciql"
    for s in (1, 2):
        _seed_dir(groot, s)
    # synthetic FB aggregate for the comparison
    fbagg = tmp_path / "fb_aggregate"
    fbagg.mkdir(parents=True)
    pd.DataFrame([
        {"task": "task1", "outcome": "success", "rho_V_negd": 0.13},
        {"task": "task1", "outcome": "transport_fail",
         "rho_V_negd": 0.23}]).to_parquet(
        fbagg / "T1_value_gradient.parquet")
    out = tmp_path / "gagg"
    agg.aggregate(groot, out, fb_aggregate=fbagg)
    assert (out / "story.md").exists()
    assert (out / "T1_value_gradient.parquet").exists()
    cmp = out / "comparison"
    assert (cmp / "fb_vs_gciql.parquet").exists()
    assert (cmp / "comparison.md").exists()
    txt = (cmp / "comparison.md").read_text()
    assert "FB" in txt and "GCIQL" in txt


def _seed_steps(_P, root, s, outcome, n=8):
    import numpy as np
    base = root / f"s{s}_final"
    base.mkdir(parents=True, exist_ok=True)
    ep = 0 if outcome == "success" else 1
    rho = 0.5 if outcome == "success" else 0.0

    def _app(name, df):
        p = base / name
        if p.exists():
            df = pd.concat([pd.read_parquet(p), df], ignore_index=True)
        df.to_parquet(p)

    _app("value_landscape.parquet", pd.DataFrame([
        {"task": "task1", "episode": ep, "outcome": outcome,
         "rho_V_negd": rho, "V_at_secure": 0.1, "V_at_end": 0.9,
         "n_transport_steps": n}]))
    _app("value_steps.parquet", pd.DataFrame(
        {"task": ["task1"] * n, "episode": [ep] * n,
         "outcome": [outcome] * n, "d": np.linspace(0.3, 0.05, n),
         "V": np.linspace(0.0, 1.0, n), "transport": [True] * n}))
    _app("coverage.parquet", pd.DataFrame([
        {"task": "task1", "outcome": outcome, "region": "transport",
         "nn_dist": 2.0}]))
    _app("phase_funnel.parquet", pd.DataFrame([
        {"task": "task1", "episode": ep, "furthest_phase": outcome,
         "outcome": outcome}]))


def test_zscore_v_is_per_call_standardised_and_order_preserving():
    import numpy as np
    from scripts.gciql_profile_aggregate import _zscore_v
    df = pd.DataFrame({"V": [2000.0, 2100.0, 1900.0, 2050.0, 1950.0]})
    out = _zscore_v(df)
    assert abs(float(out["Vz"].mean())) < 1e-9
    assert abs(float(out["Vz"].std(ddof=0)) - 1.0) < 1e-9
    # rank order preserved (so the curve shape is unchanged, only scale)
    assert list(out["Vz"].rank()) == list(df["V"].rank())
    # constant V -> all zeros, no div-by-zero
    z0 = _zscore_v(pd.DataFrame({"V": [5.0, 5.0, 5.0]}))
    assert (z0["Vz"] == 0.0).all()


def test_comparison_emits_four_plots(tmp_path):
    import importlib
    agg = importlib.import_module("scripts.gciql_profile_aggregate")
    groot = tmp_path / "gciql"
    froot = tmp_path / "repr"
    for s in (1, 2):
        _seed_steps(__import__("pathlib").Path, groot, s, "success")
        _seed_steps(__import__("pathlib").Path, groot, s,
                    "transport_fail")
    for s in (1, 2):
        _seed_steps(__import__("pathlib").Path, froot, s, "success")
        _seed_steps(__import__("pathlib").Path, froot, s,
                    "transport_fail")
    fbagg = tmp_path / "fb_aggregate"
    fbagg.mkdir()
    pd.DataFrame([{"task": "task1", "outcome": "success",
                   "rho_V_negd": 0.13},
                  {"task": "task1", "outcome": "transport_fail",
                   "rho_V_negd": 0.23}]).to_parquet(
        fbagg / "T1_value_gradient.parquet")
    out = tmp_path / "gagg"
    agg.aggregate(groot, out, fb_aggregate=fbagg, fb_seed_root=froot)
    cmp = out / "comparison"
    for fn in ("cmp_value_vs_dist.png", "cmp_rho_box.png",
               "cmp_funnel.png", "cmp_coverage.png",
               "cmp_value_rho.png", "fb_vs_gciql.parquet",
               "comparison.md"):
        assert (cmp / fn).exists(), fn
