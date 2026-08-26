"""scripts/exorl_report.py — read the T3/T4 ExoRL campaign off disk.

    python scripts/exorl_report.py [--root $SCRATCH/factored-fb-exorl] [--partial]

Prints one row per (domain, method) with the final-step metric, per-seed values and the
mean. Pure stdlib and read-only: no jax, so it runs on a login node.

TWO THINGS IT REFUSES TO CONFLATE, because both bit us once:

  * T3 and T4 are DIFFERENT EVALUATIONS, not two arms of one table. T3 withholds the env
    goal and infers the task from reward samples (its eval.csv carries the `inferred_cond*`
    witness columns); T4 is evaluated through the goal frame, which hands the policy the
    target. A T4 number is therefore NOT comparable to the T3 number for the same critic and
    domain -- the goal-given task is the easier one. They print as separate blocks and the
    header of each says which route it used, read off the CSV columns rather than assumed.
  * A run that has not reached `--steps` is not reported at all unless --partial is passed,
    and then it is marked. Mixing a 150k row into a 500k column reads as a bad method.

`cos_neutral` is the constant-reward-fallback witness (impls/utils/eval_cond.py): at 1.0 the
index carried no reward information and the cell is meaningless regardless of its score.
"""
import argparse
import csv
import glob
import os
import statistics

# Names used to split `<domain>_<critic>_<actor>_seed_<n>`. Longest-match, so `ddpgbc_density`
# stays one actor and `dmc-point-mass-maze` is not split on its hyphens.
KNOWN_DOMAINS = ("dmc-point-mass-maze", "dmc-walker", "dmc-cheetah", "hum-humanoid")
KNOWN_CRITICS = ("density_fb", "iql_bilinear", "td_fb", "gciql", "gcivl", "hiql",
                 "psm", "rldp", "crl", "qrl", "vip", "fb")


def cell_rows(root, steps, partial):
    """-> {(table, domain, critic, actor, route): [record]} for every eval.csv under `root`.

    Discovers runs by GLOBBING for eval.csv at any depth and reading identity out of the RUN
    DIRECTORY NAME (`<domain>_<critic>_<actor>_seed_<n>`), not out of the directory nesting.
    The nesting is not stable across launchers: scripts/launch_exorl.sh writes
    `t3/<domain>/<critic>/<run>` while scripts/launch_exorl_paper.sh writes
    `<domain>/<critic>/<run>`, and keying on depth silently reported "no finished runs" for
    the second one. The `t3`/`t4` prefix, when present, only picks the table label.
    """
    out = {}
    for f in glob.glob(os.path.join(root, "**", "eval.csv"), recursive=True):
        rel = os.path.relpath(f, root).split(os.sep)
        run = rel[-2]
        if "_seed_" not in run:
            continue
        table = rel[0] if rel[0] in ("t3", "t4") else "runs"
        head, _, _ = run.rpartition("_seed_")
        # domain and critic are the longest known names that prefix the run dir; the rest is
        # the actor. Matching against the known sets keeps `ddpgbc_density` from being read
        # as actor `density`, and `dmc-point-mass-maze` from being split on its hyphens.
        domain = critic = actor = "?"
        for d in sorted(KNOWN_DOMAINS, key=len, reverse=True):
            if head.startswith(d + "_"):
                domain, tail = d, head[len(d) + 1:]
                for c in sorted(KNOWN_CRITICS, key=len, reverse=True):
                    if tail.startswith(c + "_"):
                        critic, actor = c, tail[len(c) + 1:]
                        break
                else:
                    critic, actor = tail, "?"
                break
        with open(f) as fh:
            rr = list(csv.DictReader(fh))
        if not rr:
            continue
        # NOT simply rr[-1]. CsvLogger APPENDS, so a run that was resumed -- or relaunched
        # onto a directory that already held a finished run -- leaves one file holding two
        # runs back to back, and its literal last row is the NEW run's early progress. Taking
        # rr[-1] then reports a completed cell as unfinished (or, with --partial, reports an
        # early-training number as the result). Prefer the last row that reached `steps`.
        done_rows = [r for r in rr if int(float(r["step"])) == steps]
        last = done_rows[-1] if done_rows else rr[-1]
        step = int(float(last["step"]))
        if step < steps and not partial:
            continue
        route = "zero-shot(reward)" if "inferred_cond" in last else "goal-frame"
        key = (table, domain, critic, actor, route)
        out.setdefault(key, []).append({
            # episode_return is THE ExoRL benchmark metric -- the suite scores the dense
            # reward, and walker/cheetah have no binary success rule at all. `success` exists
            # only on point_mass_maze, where it is a LATCHING any-time flag (within 0.03 of
            # the target at any step of 1000), so it is reported beside the return rather
            # than instead of it.
            "episode_return": float(last["episode_return"]),
            "success": (float(last["success"])
                        if last.get("success") not in ("", None) else None),
            "step": step,
            "cos": float(last.get("inferred_cond_cos_neutral", "nan")),
        })
    return out


def ci95(vals):
    """mean, half-width of the 95% CI (1.96 s/sqrt(n)) -- the convention the paper's
    tab:main uses. Zero half-width at n=1, which the n column is there to disclose."""
    m = statistics.mean(vals)
    if len(vals) < 2:
        return m, 0.0
    return m, 1.96 * statistics.stdev(vals) / (len(vals) ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.environ.get("SCRATCH", ""),
                                                   "factored-fb-exorl"))
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--partial", action="store_true",
                    help="also report runs short of --steps, marked with their step count")
    args = ap.parse_args()

    cells = cell_rows(args.root, args.steps, args.partial)
    if not cells:
        print(f"no finished runs under {args.root} (try --partial)")
        return
    for table in ("t3", "t4", "runs"):
        keys = sorted(k for k in cells if k[0] == table)
        if not keys:
            continue
        label = {"t3": "T3 -- offline, reward-specified",
                 "t4": "T4 -- online density_fb",
                 "runs": "runs under this root"}[table]
        print(f"\n== {label} ==")
        print(f"{'domain':22s} {'method':12s} {'actor':8s} {'eval route':18s} {'n':>2s} "
              f"{'return (mean +- 95% CI)':>26s} {'success':>16s}   seeds (return)")
        for k in keys:
            _, dom, critic, actor, route = k
            recs = cells[k]
            rets = [r["episode_return"] for r in recs]
            m, h = ci95(rets)
            succ = [r["success"] for r in recs if r["success"] is not None]
            sc = "--" if not succ else (f"{ci95(succ)[0]:.3f} +- {ci95(succ)[1]:.3f}"
                                        if len(succ) > 1 else f"{succ[0]:.3f}")
            steps = {r["step"] for r in recs}
            mark = "" if steps == {args.steps} else f"  [PARTIAL: steps {sorted(steps)}]"
            warn = "  [FALLBACK INDEX]" if any(r["cos"] > 0.99 for r in recs) else ""
            print(f"{dom:22s} {critic:12s} {actor:8s} {route:18s} {len(recs):2d} "
                  f"{m:14.2f} +- {h:7.2f} {sc:>16s}   "
                  f"{'/'.join(f'{v:.1f}' for v in rets)}{mark}{warn}")
    print("\nT3 and T4 are different evaluations -- see this file's docstring. Do not read "
          "a T4 row against the T3 row above it.")


if __name__ == "__main__":
    main()
