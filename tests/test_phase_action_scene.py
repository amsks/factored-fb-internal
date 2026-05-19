import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import gciql_profile_aggregate as ga


def test_phase_action_fields_basic():
    # one episode: cube moves +x by 0.1 each step in 'transport'
    df = pd.DataFrame({
        "episode": [0, 0, 0],
        "t": [0, 1, 2],
        "cube_x": [0.0, 0.1, 0.2],
        "cube_y": [0.0, 0.0, 0.0],
        "Vz": [-1.0, 0.0, 1.0],
    })
    grid = (np.linspace(-0.1, 0.3, 9), np.linspace(-0.2, 0.2, 9))
    vz, uv, counts = ga._phase_action_fields(
        df, grid, n_arrow=4, n_min=1)
    assert vz.shape == (8, 8)            # len(edges)-1 cells per axis
    assert np.nanmax(vz) > np.nanmin(vz)
    # net displacement is +x, ~0 y
    us, vs = uv
    assert np.nanmean(us[np.isfinite(us)]) > 0
    assert abs(np.nanmean(vs[np.isfinite(vs)])) < 1e-6
    assert counts.sum() >= 1


def test_phase_action_fields_empty():
    df = pd.DataFrame(columns=["episode", "t", "cube_x", "cube_y", "Vz"])
    grid = (np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    vz, uv, counts = ga._phase_action_fields(
        df, grid, n_arrow=4, n_min=1)
    assert np.isnan(vz).all()
    assert counts.sum() == 0


def test_canon_task_normalizes_both_recorder_labels():
    assert ga._canon_task("task3") == "task3"
    assert ga._canon_task(
        "cube-single-play-singletask-task3-v0") == "task3"


def test_phase_action_scene_smoke(tmp_path, monkeypatch):
    # FB and GCIQL write the `task` column differently (FB: full env
    # id, GCIQL: 'taskN'); the renderer must canonicalize so they pair.
    def _vs(seed_root, task_label):
        d = Path(seed_root) / "s1_final"
        d.mkdir(parents=True)
        rows = []
        for oc in ("success", "transport_fail"):
            for rg in ("reach", "lift", "transport"):
                for t in range(4):
                    rows.append(dict(task=task_label, episode=0, t=t,
                                     outcome=oc, region=rg, d=0.5,
                                     V=float(t),
                                     transport=(rg == "transport"),
                                     cube_x=0.1 + 0.05 * t,
                                     cube_y=0.0))
        pd.DataFrame(rows).to_parquet(d / "value_steps.parquet")

    fb_root = tmp_path / "fb"
    gc_root = tmp_path / "gc"
    _vs(fb_root, "cube-single-play-singletask-task1-v0")  # FB label
    _vs(gc_root, "task1")                                  # GCIQL label
    monkeypatch.setattr(ga, "_decode_support_xy",
                        lambda *a, **k: np.random.RandomState(0)
                        .uniform(0, 0.4, size=(200, 2)))
    out = tmp_path / "phase_scene"
    ga._phase_action_scene(out, str(gc_root), str(fb_root))
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert pngs == ["task1__FB.png", "task1__GCIQL.png"]
    assert (out / "_index.md").exists()
