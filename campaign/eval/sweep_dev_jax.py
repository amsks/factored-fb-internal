"""campaign/eval/sweep_dev_jax.py — eval the ALL-JAX dev campaign: every algorithm
(fb, fb_flowbc, gciql, crl, crl_flowbc) x DEV_TASKS, via eval_jax (fixed-seed episodes),
aggregated into a table + per-family means. Runs each (algo, task) eval as a subprocess in
.venv-jax.

Requires (for the new families): metaworld + mani_skill installed in .venv-jax (do this
once the training campaign has finished). Box2D works already. cube-single uses OGBench's
own evaluation and is skipped here (handled separately).

    <fb>/.venv/bin/python campaign/eval/sweep_dev_jax.py \
        --dev_dir /raid/tongle/rl/fb-data/campaign/dev_jax \
        --params params_500000.pkl --n_episodes 100 --n_video 0 \
        --out /raid/tongle/rl/fb-data/campaign/eval_dev
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/raid/tongle/rl/Factored-FB/campaign")
from registry import DEV_TASKS, DATASETS, COLLECTED  # noqa: E402

JAX_PY = "/raid/tongle/rl/Factored-FB/.venv-jax/bin/python"
EVAL_JAX = "/raid/tongle/rl/Factored-FB/campaign/eval/eval_jax.py"
IMPLS = "/raid/tongle/rl/Factored-FB/ogbench/impls"
SHIM = "/raid/tongle/rl/Factored-FB/tools/wandb_mode_shim"
JAX_AGENTS = "/raid/tongle/rl/Factored-FB/campaign/jax_agents"
UNIFIED_CUBE = "/raid/tongle/rl/fb-data/unified/cube-single-play-v0"

ALGO_FILE = {
    "fb": "campaign/jax_agents/fb.py",
    "fb_flowbc": "campaign/jax_agents/fb_flowbc.py",
    "gciql": "agents/gciql.py",
    "crl": "agents/crl.py",
    "crl_flowbc": "tools/wandb_mode_shim/crl_flowbc.py",
}
BOX2D_LOCO = {"b2d-center-ctrl", "b2d-goal-ctrl", "b2d-goal-ctrl-maze", "b2d-center-wall-navigation"}


def data_dir(slug):
    ds = {d["slug"]: d for d in DATASETS}[slug]
    if ds["family"] == "OGBench":
        return UNIFIED_CUBE
    return f"{COLLECTED}/{ds['family']}__{slug}"


def family_of(slug):
    return {d["slug"]: d for d in DATASETS}[slug]["family"]


def run_eval(algo, slug, save_dir, params, n_episodes, n_video, out_dir, seed_base=0):
    env = dict(__import__("os").environ, PYTHONPATH=f"{IMPLS}:{SHIM}:{JAX_AGENTS}",
               JAX_PLATFORMS="", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    cmd = [JAX_PY, EVAL_JAX, "--save_dir", save_dir, "--params", params,
           "--agent_file", ALGO_FILE[algo], "--task", slug, "--data_dir", data_dir(slug),
           "--n_episodes", str(n_episodes), "--seed_base", str(seed_base),
           "--n_video", str(n_video), "--out", out_dir]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       cwd="/raid/tongle/rl/Factored-FB")
    mfp = Path(out_dir) / "metrics.json"
    if mfp.exists():
        return json.loads(mfp.read_text()).get("success_rate")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_dir", default="/raid/tongle/rl/fb-data/campaign/dev_jax")
    ap.add_argument("--params", default="params_500000.pkl")
    ap.add_argument("--n_episodes", type=int, default=100)
    ap.add_argument("--n_video", type=int, default=0)
    ap.add_argument("--out", default="/raid/tongle/rl/fb-data/campaign/eval_dev")
    ap.add_argument("--tasks", default=None, help="comma list to restrict (else DEV_TASKS minus cube)")
    ap.add_argument("--algos", default="fb,fb_flowbc,gciql,crl,crl_flowbc")
    args = ap.parse_args()

    tasks = args.tasks.split(",") if args.tasks else [t for t in DEV_TASKS if t != "cube-single"]
    algos = args.algos.split(",")
    table = {}                                    # table[task][algo] = success
    for slug in tasks:
        table[slug] = {}
        for algo in algos:
            sd = f"{args.dev_dir}/{algo}__{slug}__s0"
            if not Path(sd, args.params).exists():
                print(f"[sweep] MISSING {algo}__{slug} ({args.params})")
                continue
            out_dir = f"{args.out}/{algo}/{slug}"
            sr = run_eval(algo, slug, sd, args.params, args.n_episodes, args.n_video, out_dir)
            table[slug][algo] = sr
            print(f"[sweep] {algo:11s} {slug:24s} success={sr}")

    # report: table (task x algo) + per-family + per-algo means
    fams = {}
    for slug in tasks:
        fams.setdefault(family_of(slug), []).append(slug)
    summary = {"params": args.params, "n_episodes": args.n_episodes, "table": table}
    print(f"\n=== dev eval ({args.params}, {args.n_episodes} eps) ===")
    hdr = "task".ljust(26) + "".join(a.ljust(12) for a in algos)
    print(hdr)
    for slug in tasks:
        row = slug.ljust(26) + "".join((f"{table[slug].get(a):.3f}".ljust(12)
                                        if table[slug].get(a) is not None else "-".ljust(12)) for a in algos)
        print(row)
    print("-" * len(hdr))
    for algo in algos:
        vals = [table[t][algo] for t in tasks if table[t].get(algo) is not None]
        summary.setdefault("algo_mean", {})[algo] = float(np.mean(vals)) if vals else None
    print("MEAN(over tasks)".ljust(26) + "".join(
        (f"{summary['algo_mean'][a]:.3f}".ljust(12) if summary['algo_mean'][a] is not None else "-".ljust(12))
        for a in algos))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[sweep] wrote {args.out}/summary.json")


if __name__ == "__main__":
    main()
