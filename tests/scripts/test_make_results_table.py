"""tests/scripts/test_make_results_table.py -- the paper table's per-run estimator.

OGBench eval.csv files carry one row per TASK per eval step (impls/main.py logs each task_id
separately; eval_aggregate is off by default). score_run must therefore average across tasks
at each step and only then take the mean of the last 3 steps. Keying on step alone lets each
later task row overwrite the earlier one and scores the last task instead of the task mean.
"""
import csv
import statistics as st

import pytest

from scripts.make_results_table import parse_run, score_run

STEPS = [100_000, 200_000, 300_000, 400_000]
# success[step_idx][task_idx]; task 4 (the last one written) is deliberately far from the mean.
SUCCESS = [
    [0.10, 0.20, 0.30, 0.40, 0.90],
    [0.50, 0.60, 0.70, 0.80, 0.10],
    [0.60, 0.70, 0.80, 0.90, 0.20],
    [0.70, 0.80, 0.90, 1.00, 0.30],
]


def _write_eval_csv(path, per_task=True):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "task_id", "success"])
        w.writeheader()
        for step, row in zip(STEPS, SUCCESS):
            if per_task:
                for task_id, s in enumerate(row, start=1):
                    w.writerow({"step": step, "task_id": task_id, "success": s})
            else:
                w.writerow({"step": step, "task_id": -1, "success": st.mean(row)})


def test_score_run_means_over_tasks_then_last_3_steps(tmp_path):
    p = tmp_path / "eval.csv"
    _write_eval_csv(p, per_task=True)
    expected = st.mean(st.mean(SUCCESS[i]) for i in (1, 2, 3))
    assert score_run(str(p)) == pytest.approx(expected)
    # and NOT the last task's curve
    assert score_run(str(p)) != pytest.approx(st.mean(SUCCESS[i][-1] for i in (1, 2, 3)))


def test_score_run_single_row_per_step_unchanged(tmp_path):
    p = tmp_path / "eval.csv"
    _write_eval_csv(p, per_task=False)
    expected = st.mean(st.mean(SUCCESS[i]) for i in (1, 2, 3))
    assert score_run(str(p)) == pytest.approx(expected)


def test_score_run_missing_or_empty(tmp_path):
    assert score_run(str(tmp_path / "nope.csv")) is None
    p = tmp_path / "eval.csv"
    p.write_text("step,task_id,success\n")
    assert score_run(str(p)) is None


@pytest.mark.parametrize("critic", ["density_fb", "density_fb_z", "fb_tdj", "rldp",
                                    "rldp_density", "hilp", "hilp_tdjc", "hilp_density"])
def test_parse_run_knows_tdjtab_critics(critic):
    name = f"cube-double-play-v0_{critic}_flowbc_seed_2"
    assert parse_run(name) == ("cube-double-play-v0", critic, "flowbc", 2)
