"""Paper-exact final evaluation for the ExoRL official-parity campaign.

docs/specs/2026-08-28-exorl-official-parity-validation.md work item B.

WHAT THIS IS NOT: a new eval harness. `scripts/reeval_checkpoint.py` already implements
the protocol the RLDP paper describes -- draw N reward-labelled transitions from the
offline dataset (rewards obtained by RE-SIMULATING MuJoCo physics for the target task,
env/exorl_env.py::relabel_rewards), infer z from them, then roll the frozen policy with
the env goal path disabled. `--override exorl_task=<t>` relabels the dataset AND builds
the env for that task together, so the two cannot drift. This module is the DRIVER: it
runs that script once per (domain, task, seed), at the paper's numbers, and reduces the
rows to the replication table the gate is read off.

Keeping the per-cell work in reeval_checkpoint is deliberate. It shares impls.main's own
adapter / Agent.create / restore / evaluate path, so a training run and an eval run can
never disagree about how a checkpoint is built or scored. A standalone reimplementation
would be a second source of truth for exactly the number under dispute.

PROTOCOL KNOBS AND WHERE THEY COME FROM
  --inference-samples 50000   RLDP TrainConfig.num_inference_samples
                              (examples/rldp_train_dmc.py:151). NOTE the flag on the
                              inner script is named `--infer_cond_episodes` but it is a
                              TRANSITION count -- it is passed straight to
                              `train_ds.sample(...)`, which draws that many rows. The
                              name is a wart, not a semantic difference.
  --episodes 50               paper text. The code default is 10; run with --episodes 10
                              to record that as the `eval-10` ablation.
  --seeds 0 1 2 3             4 seeds, mean +- std per cell.
  HILP uses a least-squares z rather than the reward-weighted mean, over 10,000
  transitions; that is a property of the CRITIC's infer_cond, not of this driver, so the
  only thing that changes here is --inference-samples -- which now defaults per critic
  via INFERENCE_SAMPLES rather than relying on the caller to remember.

TASK SETS are RLDP Table 1's row names, taken from env/exorl_env.py::TASKS so this file
cannot disagree with what the env layer will actually build.

    python -m scripts.exorl_paper_eval --root exp/exorl-official \
        --critic rldp_official --actor ddpgbc --steps 5000000 --out exp/paper_eval.jsonl
    python -m scripts.exorl_paper_eval --root exp/exorl-official --report-only \
        --out exp/paper_eval.jsonl
"""
import argparse
import json
import os
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import exorl_env  # noqa: E402

# The four tasks per domain that RLDP Table 1 reports. Read off the env layer's own
# registry, then narrowed to the paper's four where the registry carries more (cheetah
# also defines flip / flip_backward, which Table 1 does not report).
PAPER_TASKS = {
    "dmc-walker": ("stand", "walk", "run", "flip"),
    "dmc-cheetah": ("run", "run_backward", "walk", "walk_backward"),
    "dmc-point-mass-maze": ("reach_top_left", "reach_top_right",
                            "reach_bottom_left", "reach_bottom_right"),
}
# Per-domain hit-spec radii (calibrated; see env/domains/*.yaml). Unused by rldp/fb, read
# by the density arms -- kept in one table so the two cannot drift.
HIT = {"dmc-walker": "0:24:15.9", "dmc-cheetah": "0:17:7.5",
       "dmc-point-mass-maze": "0:2:0.03"}
DISCOUNT = {"dmc-walker": 0.98, "dmc-cheetah": 0.98, "dmc-point-mass-maze": 0.99}


# Each method infers z under its own published protocol, so a single default would score
# one of them wrong. Resolved from the critic rather than left to the caller: the failure
# mode is silent -- HILP scored with RLDP's 50k just returns a slightly different number,
# with nothing in the output saying which protocol produced it.
#   RLDP 50k  examples/rldp_train_dmc.py:151, TrainConfig.num_inference_samples
#   HILP 10k  the official HILP eval draws 10,000 reward-labelled transitions for lstsq
INFERENCE_SAMPLES = {
    "rldp": 50_000, "rldp_official": 50_000, "rldp_density_official": 50_000,
    "rldp_orthoa": 50_000, "rldp_orthob": 50_000,
    "hilp": 10_000, "hilp_density": 10_000,
}


# Architecture-affecting config keys that were ADDED after some runs were trained. A run
# checkpointed before the key existed was trained with the flag off, but today's yaml
# turns it on, so rebuilding the agent from the yaml alone produces a tower the checkpoint
# cannot populate -- flax then fails with a key mismatch on every cell. The run's own
# config.json sits next to its checkpoints and records what it was actually trained with,
# so that is the authority, not the current yaml.
#
# Keys are added here as they appear. Absent from a run's config.json => it predates the
# key => the flag was off for that run.
_ARCH_KEYS_DEFAULT_OFF = ("rldp_official_nets",)


def _legacy_arch_overrides(run_dir):
    """Overrides that pin a pre-existing run to the architecture it was trained with."""
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return []
    try:
        with open(cfg_path) as f:
            recorded = json.load(f).get("config", {})
    except (OSError, ValueError):
        return []          # unreadable config: fall through to the yaml and fail loudly
    out = []
    for key in _ARCH_KEYS_DEFAULT_OFF:
        if key not in recorded:
            out += ["--override", f"{key}=false"]
    return out


def _check_tasks():
    """Fail loudly if the paper's task names are not the env layer's task names. A typo
    here would silently score a different task than the table claims."""
    for dom, tasks in PAPER_TASKS.items():
        known = set(exorl_env.reward_tasks(dom))
        missing = [t for t in tasks if t not in known]
        assert not missing, f"{dom}: {missing} not in env/exorl_env.py TASKS ({sorted(known)})"


def run_cell(args, domain, task, seed):
    run_dir = os.path.join(args.root, domain, args.critic,
                           f"{domain}_{args.critic}_{args.actor}_seed_{seed}")
    # A missing run dir is reported, not skipped silently -- a partially-finished campaign
    # must show up as a hole in the table rather than as a smaller n. Checked AFTER the
    # dry-run print so `--dry-run` shows the full matrix before anything has trained.
    missing = not os.path.isdir(run_dir)
    legacy = _legacy_arch_overrides(run_dir)
    cmd = [args.python, "-m", "scripts.reeval_checkpoint",
           "--critic", args.critic, "--actor", args.actor, "--domain", domain,
           "--seed", str(seed), "--restore_path", run_dir,
           "--restore_epoch", str(args.steps),
           "--eval_episodes", str(args.episodes),
           "--infer_cond_episodes", str(args.inference_samples),
           "--eval_on_inferred_cond",
           "--tag", f"{args.critic}|{domain}|{task}|s{seed}",
           "--out_jsonl", args.out,
           "--override", f"exorl_task={task}",
           "--override", f"discount={DISCOUNT[domain]}",
           "--override", f'measure_hit_spec="{HIT[domain]}"',
           # The training-time actor settings. All four are LOSS-only -- they change no
           # network shape and no eval-time forward pass (eval takes dist.mode()) -- so the
           # restore works without them. They are passed anyway so the eval config is
           # literally the training config: a future knob that IS shape-bearing would
           # otherwise fail as a confusing restore error rather than a config mismatch.
           "--override", "actor.alpha=0.0",
           "--override", "actor.q_normalize=false",
           "--override", "actor.q_noise_std=0.2",
           "--override", "actor.boot_noise_std=0.2"] + legacy
    if legacy and not args.dry_run:
        print(f"LEGACY-ARCH {domain}/{task}/s{seed}: {' '.join(legacy[1::2])} "
              f"(from {os.path.basename(run_dir)}/config.json)", flush=True)
    if args.dry_run:
        print(("MISSING-RUNDIR " if missing else "") + " ".join(cmd))
        return {"domain": domain, "task": task, "seed": seed, "status": "dryrun"}
    if missing:
        print(f"[missing] {run_dir}", flush=True)
        return {"domain": domain, "task": task, "seed": seed, "status": "missing",
                "run_dir": run_dir}
    env = dict(os.environ)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # Same overlay the launcher sets, and for the same reason: reward relabelling
    # re-simulates MuJoCo and this venv's own mujoco is too old for its dm_control.
    env.setdefault("FB_MUJOCO_SITE_PACKAGES",
                   os.path.expanduser("~/git/Factored-FB/.venv/lib/python3.11/site-packages"))
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[FAIL] {domain}/{task}/s{seed} rc={r.returncode}\n{r.stderr[-2000:]}",
              flush=True)
        return {"domain": domain, "task": task, "seed": seed, "status": "failed"}
    print(f"[ok] {domain}/{task}/s{seed}", flush=True)
    return {"domain": domain, "task": task, "seed": seed, "status": "ok"}


def _metric(row):
    """Episode return is the ExoRL benchmark metric; walker/cheetah have no binary
    success rule at all (env/unified.py omits the key), so `success` must never be
    substituted -- a missing success key would otherwise read as a perfect-failure 0.0."""
    for k in ("episode_return", "return", "eval/episode_return"):
        if k in row:
            return float(row[k])
    return None


def report(out_path, critic=None):
    if not os.path.exists(out_path):
        print(f"no results at {out_path}"); return
    cells = {}
    for line in open(out_path):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        tag = str(row.get("tag", ""))
        parts = tag.split("|")
        if len(parts) != 4:
            continue
        c, dom, task, _seed = parts
        if critic and c != critic:
            continue
        v = _metric(row)
        if v is None:
            continue
        cells.setdefault((c, dom, task), []).append(v)
    print(f"{'critic':<24} {'domain':<22} {'task':<20} {'mean':>9} {'std':>8}  n")
    for (c, dom, task), vals in sorted(cells.items()):
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{c:<24} {dom:<22} {task:<20} {statistics.mean(vals):9.1f} {sd:8.1f}  {len(vals)}")
    print("\nCompare against RLDP Table 1 in docs/notes/2026-08-28-rldp-paper-eval.md. "
          "Gate (spec §4): per-task mean within 1 sigma of the official cell for >= 3 of "
          "4 tasks, and no task below 50% of an official mean > 100.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="exp/exorl-official")
    ap.add_argument("--critic", default="rldp_official")
    ap.add_argument("--actor", default="ddpgbc")
    ap.add_argument("--domains", nargs="*", default=sorted(PAPER_TASKS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, required=False, default=5_000_000,
                    help="checkpoint step to restore (params_{steps}.pkl)")
    ap.add_argument("--episodes", type=int, default=50, help="paper text; code default 10")
    ap.add_argument("--inference-samples", type=int, default=None,
                    help="reward-labelled TRANSITIONS for z inference; default resolves "
                         "from the critic (RLDP 50k, HILP 10k)")
    ap.add_argument("--out", default="exp/paper_eval.jsonl")
    ap.add_argument("--python", default="/dev/shm/.venv-jax/bin/python")
    ap.add_argument("--jobs", type=int, default=1,
                    help="cells to evaluate concurrently (one subprocess each)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.inference_samples is None:
        args.inference_samples = INFERENCE_SAMPLES.get(args.critic, 50_000)
        print(f"--inference-samples defaulted to {args.inference_samples} "
              f"for critic {args.critic}", flush=True)

    _check_tasks()
    if args.report_only:
        report(args.out, args.critic); return
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    todo = [(d, t, s) for d in args.domains for t in PAPER_TASKS[d] for s in args.seeds]
    print(f"{len(todo)} cells: {len(args.domains)} domains x 4 tasks x {len(args.seeds)} seeds"
          f" ({args.jobs} at a time)", flush=True)
    # Each cell is an independent subprocess that appends one line to --out; the sequential
    # loop this replaced took hours for a full 32-cell table. ThreadPoolExecutor is the right
    # pool here because the work is entirely in the child processes -- the threads only wait
    # on subprocess.run. Appends are a single short write per cell, done by separate
    # processes, so they do not interleave in practice.
    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            list(ex.map(lambda t: run_cell(args, *t), todo))
    else:
        for dom, task, seed in todo:
            run_cell(args, dom, task, seed)
    if not args.dry_run:
        print(); report(args.out, args.critic)


if __name__ == "__main__":
    main()
