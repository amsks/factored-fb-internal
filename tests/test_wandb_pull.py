import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import wandb_pull as wp


def test_metric_keys_cube_includes_canonical_eval_and_train():
    keys = wp.metric_keys("cube-single-play-v0")
    assert "eval/reward/eval/success" in keys
    assert "eval/reward/eval/reward" in keys
    assert "eval/reward/cube-single-play-singletask-task1-v0/success" in keys
    assert "eval/reward/cube-single-play-singletask-task5-v0/success" in keys
    assert "train/fb_loss" in keys
    assert "train/orth_loss" in keys
    assert "train/actor_loss" in keys
    assert "train/bc_flow_loss" in keys
    # old nested style must NOT appear
    assert "train/loss/fb" not in keys


def _run(group, tags, created, domain):
    return SimpleNamespace(
        id="x", group=group, tags=tags,
        created_at=created.isoformat().replace("+00:00", "Z"),
        config={"domain": domain, "ortho_coef": 1000, "lr_b": 1e-4, "seed": 1},
    )


def test_filter_runs_by_group_tag_hours_domain():
    now = datetime(2026, 5, 18, tzinfo=timezone.utc)
    recent = _run("g1", ["sweep"], now - timedelta(hours=1), "cube-single-play-v0")
    old = _run("g1", ["sweep"], now - timedelta(hours=99), "cube-single-play-v0")
    wrong_group = _run("g2", ["sweep"], now - timedelta(hours=1), "cube-single-play-v0")
    wrong_domain = _run("g1", ["sweep"], now - timedelta(hours=1), "foo-v0")
    kept = wp.filter_runs(
        [recent, old, wrong_group, wrong_domain],
        groups=["g1"], tags=["sweep"], hours=36,
        domains=["cube-single-play-v0"], now=now,
    )
    assert [r.id for r in kept] == ["x"]
    assert kept[0] is recent
