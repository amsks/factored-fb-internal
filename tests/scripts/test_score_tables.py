"""The table estimators in scripts/score_tables.py."""
import json
import math

from scripts.score_tables import BLOCK_TITLES, cell, parse_rows, run_score, score_exorl, write_tables


def test_run_score_is_last_three_evals_of_task_mean(tmp_path):
    p = tmp_path / "eval.csv"
    rows = ["step,success,task_id"]
    for step, succ in [(100, (0.0, 0.0)), (200, (1.0, 0.0)), (300, (1.0, 1.0)), (400, (0.5, 0.5))]:
        rows += [f"{step},{succ[0]},1", f"{step},{succ[1]},2"]
    p.write_text("\n".join(rows))
    score, last = run_score(p)
    assert last == 400
    assert math.isclose(score, 100 * (0.5 + 1.0 + 0.5) / 3)


def test_cell_uses_sample_sd_for_ogbench_and_population_sd_for_exorl():
    m, se, n = cell([1.0, 3.0])
    assert (m, n) == (2.0, 2) and math.isclose(se, math.sqrt(2) / math.sqrt(2))
    m, se, n = cell([1.0, 3.0], pop=True)
    assert math.isclose(se, 1.0 / math.sqrt(2))


def test_exorl_seed_score_averages_tasks(tmp_path):
    lines = [{"domain": "dmc-walker", "seed": s, "exorl_task": t, "episode_return": 100.0 * (s + 1) + i}
             for s in range(2) for i, t in enumerate(["walk", "run"])]
    for name in ["paper_eval_exorl_sqt.jsonl", "paper_eval_exorl_kernels_quadruped_vu.jsonl",
                 "paper_eval_exorl_1mrecipe.jsonl", "paper_eval_exorl_cheetah3_velblock.jsonl"]:
        (tmp_path / name).write_text("\n".join(json.dumps(r) for r in lines))
    (m, se, n), per_seed = score_exorl(str(tmp_path))["Density-FB", "walker"]
    assert n == 2 and [v for v, _ in per_seed] == [100.5, 200.5] and m == 150.5


def _tex():
    def row(t, n):
        return (r"    \texttt{%s} & " % t + " & ".join(r"%d.0\pmse{0.1}" % (k + 1) for k in range(n))
                + r" & 9.0\pmse{0.2} & x \\")
    bench = [t for b in ["OGBench", "MetaWorld", "ManiSkill"] for t in
             {"OGBench": ["cube-single", "cube-double", "scene"], "MetaWorld": ["push", "bin-picking", "box-close"],
              "ManiSkill": ["push-cube", "pick-cube", "stack-cube", "poke-cube", "lift-peg"]}[b]]
    exorl = ["walker", "quadruped", "point-mass", "cheetah"]
    return "\n".join([r"    \toprule", "    " + BLOCK_TITLES["OGBench"] + " & old \\\\"]
                     + [row(t, 6) for t in bench] + [r"    \bottomrule",
                     "        " + BLOCK_TITLES["ExoRL"] + " & old \\\\"]
                     + ["    " + row(t, 3) for t in exorl] + [r"        \bottomrule"])


def test_parse_and_rewrite_tables(tmp_path):
    rows = parse_rows(_tex())
    assert rows["cube-single"][-1] == (9.0, 0.2) and len(rows["cube-single"]) == 7
    assert rows["walker"] == [(1.0, 0.1), (2.0, 0.1), (3.0, 0.1), (9.0, 0.2)]
    p = tmp_path / "t.tex"
    p.write_text(_tex())
    write_tables(p, rows)
    once = p.read_text()
    assert r"\textbf{9.0}" in once and "$+$200.0" in once       # delta vs best baseline 3.0
    write_tables(p, parse_rows(once))
    assert p.read_text() == once                                 # rewriting is idempotent
