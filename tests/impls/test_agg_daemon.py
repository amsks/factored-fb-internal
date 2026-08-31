"""campaign/wandb_agg_daemon.py — the per-seed aggregate must count SEEDS, not jsonl rows.

The daemon's whole job is "once all n_seeds have reported a step, log the mean and the
seed-to-seed std". It read every jsonl row as one seed. A single seed routinely writes
SEVERAL rows at the same step:

  * `impls/main.py::_append_agg` is called inside the `for task_id in adapter.eval_tasks(...)`
    loop, and the OGBench adapter returns one task_id per task unless the domain sets
    `eval_aggregate: true`. `impls/configs/domain/cube_single.yaml` -- the default
    `--domain` -- does not, so that run appends one row PER TASK per eval point;
  * the file is opened in append mode, so a relaunched run adds to what is already there.

So `len(v) >= n_seeds` fired on one seed's task rows and published their spread as
`eval/success_std`, i.e. an error bar over tasks reported as an error bar over seeds.
"""
import json
import os

import numpy as np

from campaign.wandb_agg_daemon import read_group


def _write(d, group, seed, rows):
    with open(os.path.join(d, f"{group}__s{seed}.jsonl"), "a") as f:
        for step, succ in rows:
            f.write(json.dumps(dict(seed=seed, step=step, success_rate=succ,
                                    at_goal_ticks=float("nan"), family="ManiSkill")) + "\n")


GROUP = "density_fb_flowbc__cube-single"


def test_one_seed_with_many_task_rows_counts_as_one_seed(tmp_path):
    d = str(tmp_path)
    # exactly what a 5-task OGBench eval point writes for a single seed
    _write(d, GROUP, 0, [(20000, s) for s in (0.90, 0.10, 0.02, 0.55, 0.03)])
    per_step, _, _ = read_group(d, GROUP)
    assert len(per_step[20000]) == 1, \
        f"5 task rows from one seed counted as {len(per_step[20000])} seeds"
    # the seed's value is its mean over its own tasks
    assert per_step[20000][0] == np.mean([0.90, 0.10, 0.02, 0.55, 0.03])


def test_duplicate_rows_from_a_relaunch_do_not_inflate_the_seed_count(tmp_path):
    d = str(tmp_path)
    _write(d, GROUP, 0, [(20000, 0.4)])
    _write(d, GROUP, 0, [(20000, 0.4)])          # the run was relaunched and re-appended
    _write(d, GROUP, 0, [(20000, 0.4)])
    per_step, _, _ = read_group(d, GROUP)
    assert len(per_step[20000]) == 1


def test_distinct_seeds_are_still_counted_and_ordered(tmp_path):
    d = str(tmp_path)
    _write(d, GROUP, 2, [(20000, 0.3)])
    _write(d, GROUP, 0, [(20000, 0.1)])
    _write(d, GROUP, 1, [(20000, 0.2)])
    per_step, per_dwell, fam = read_group(d, GROUP)
    assert per_step[20000] == [0.1, 0.2, 0.3]     # sorted by seed
    assert len(per_dwell[20000]) == 3 and all(np.isnan(x) for x in per_dwell[20000])
    assert fam == "ManiSkill"


def test_a_nan_success_rate_propagates_rather_than_being_dropped(tmp_path):
    """A domain with no binary success rule writes success_rate=NaN (impls/main.py::
    _append_agg). The aggregate must show a gap, not a number."""
    d = str(tmp_path)
    _write(d, GROUP, 0, [(20000, float("nan"))])
    per_step, _, _ = read_group(d, GROUP)
    assert len(per_step[20000]) == 1 and np.isnan(per_step[20000][0])
