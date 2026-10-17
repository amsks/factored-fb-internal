"""The vendored phase-labelling code matches its torch source exactly.

Skips when the sibling torch worktree is absent.
"""

import ast
import inspect
import os
import pathlib

import pytest

from scripts.probes.cube_bottleneck import phases

TORCH_ROOT = pathlib.Path(
    os.environ.get("FACTORED_FB_TORCH_WORKTREE", "/home/adityamohan/git/Factored-FB")
)

# (source file, symbol) per vendored name, in phases.py order.
VENDORED = [
    ("evals/phase_probe.py", "Thresholds"),
    ("evals/phase_probe.py", "_max_true_run"),
    ("evals/phase_probe.py", "classify_phases"),
    ("evals/phase_probe.py", "step_signals"),
    ("evals/phase_probe.py", "_target_block"),
    ("evals/phase_probe.py", "episode_goal"),
    ("evals/phase_probe.py", "episode_table_z"),
    ("evals/phase_probe.py", "ensure_manip_env"),
    ("evals/phase_probe.py", "_ensure_mujoco"),
    ("evals/phase_probe.py", "apply_scenario"),
    ("evals/_profile_core.py", "transport_mask"),
    ("scripts/probes/phase_probe_transport_buckets.py", "bucket_row"),
]


def _source_segment(path: pathlib.Path, name: str) -> str:
    """Source text of a top-level def or class, decorators included."""
    src = path.read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if getattr(node, "name", None) == name:
            start = node.lineno
            for d in getattr(node, "decorator_list", []):
                start = min(start, d.lineno)
            lines = src.splitlines(keepends=True)
            return "".join(lines[start - 1 : node.end_lineno])
    raise KeyError(f"{name!r} not found in {path}")


def _require(rel: str) -> pathlib.Path:
    p = TORCH_ROOT / rel
    if not p.exists():
        pytest.skip(
            f"torch worktree source absent: {p} "
            "(set FACTORED_FB_TORCH_WORKTREE to check vendoring)"
        )
    return p


@pytest.mark.parametrize("rel,name", VENDORED, ids=[f"{n}" for _, n in VENDORED])
def test_vendored_symbol_is_byte_identical(rel, name):
    source = _source_segment(_require(rel), name).rstrip("\n")
    vendored = inspect.getsource(getattr(phases, name)).rstrip("\n")
    assert vendored == source, (
        f"{name} has drifted from {rel}. Regenerate the vendored block; do not hand-edit."
    )


def test_n_settle_matches_source():
    """N_SETTLE matches the source constant."""
    src = _require("evals/phase_probe.py").read_text()
    tree = ast.parse(src)
    found = [
        n.value.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "N_SETTLE" for t in n.targets)
    ]
    assert found, "N_SETTLE not found in phase_probe.py"
    assert phases.N_SETTLE == found[0]


def test_scenarios_match_source():
    src = _require("evals/phase_probe.py").read_text()
    tree = ast.parse(src)
    found = [
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "SCENARIOS" for t in n.targets)
    ]
    assert found and phases.SCENARIOS == found[0]


def test_alias_is_the_vendored_class():
    assert phases.PhaseThresholds is phases.Thresholds


def test_thresholds_defaults_are_the_published_ones():
    """Phase thresholds have their published default values."""
    thr = phases.PhaseThresholds()
    assert (thr.eps_reach, thr.delta_lift, thr.k_steps, thr.tau_grip) == (
        0.06,
        0.03,
        5,
        0.5,
    )
