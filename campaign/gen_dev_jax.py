"""campaign/gen_dev_jax.py — emit the ALL-JAX dev-set job list: 5 algorithms
(fb, fb_flowbc, gciql, crl, crl_flowbc) x DEV_TASKS x seeds, all training-only through
campaign/jax_train_unified.py with live wandb logging.

    <fb>/.venv/bin/python campaign/gen_dev_jax.py   ->  campaign/jobs_dev.txt
"""
from pathlib import Path

from registry import DATASETS, DEV_TASKS, NUM_TRAIN_STEPS, SAVE_EVERY, CAMPAIGN_OUT, COLLECTED

FB = "/raid/tongle/rl/Factored-FB"
IMPLS = f"{FB}/third_party/ogbench/impls"
SHIM = f"{FB}/tools/wandb_mode_shim"
JAX_AGENTS = f"{FB}/campaign/jax_agents"
UNIFIED_CUBE = "/raid/tongle/rl/fb-data/unified/cube-single-play-v0"
JOBS = Path(FB) / "campaign" / "jobs_dev.txt"

DEV_SEEDS = [0]                       # first pass: 1 seed to iterate fast (scale up later)

ALGOS = {
    "fb":         ("campaign/jax_agents/fb.py", ""),
    "fb_flowbc":  ("campaign/jax_agents/fb_flowbc.py", ""),
    "gciql":      ("agents/gciql.py", "--set alpha=1.0"),
    "crl":        ("agents/crl.py", "--set alpha=3.0"),
    "crl_flowbc": ("tools/wandb_mode_shim/crl_flowbc.py", "--set bc_coeff=3.0"),
}

_FAM_DIR = {"box2d": "Box2D", "metaworld": "Metaworld", "maniskill": "ManiSkill"}


def _ds_by_slug():
    return {d["slug"]: d for d in DATASETS}


def data_dir(ds):
    if ds["family"] == "OGBench":
        return UNIFIED_CUBE
    return f"{COLLECTED}/{ds['family']}__{ds['slug']}"


def main():
    by_slug = _ds_by_slug()
    lines = []
    for slug in DEV_TASKS:
        ds = by_slug[slug]
        ddir = data_dir(ds)
        for algo, (af, ov) in ALGOS.items():
            for s in DEV_SEEDS:
                save_dir = f"{CAMPAIGN_OUT}/dev_jax/{algo}__{slug}__s{s}"
                cmd = (
                    f"XLA_PYTHON_CLIENT_PREALLOCATE=false "  # on-demand mem: coexist w/ other users
                    f"PYTHONPATH={IMPLS}:{SHIM}:{JAX_AGENTS} "
                    f".venv-jax/bin/python campaign/jax_train_unified.py "
                    f"--data_dir {ddir} --agent_file {af} --save_dir {save_dir} "
                    f"--train_steps {NUM_TRAIN_STEPS} --save_interval {SAVE_EVERY} "
                    f"--seed {s} {ov} "
                    f"--wandb 1 --wandb_project factored-fb-dev "
                    f"--wandb_group {ds['family']} --wandb_name {algo}__{slug}__s{s}"
                )
                lines.append(cmd)
    JOBS.write_text("\n".join(lines) + "\n")
    print(f"[gen_dev_jax] wrote {len(lines)} jobs -> {JOBS}")
    print(f"[gen_dev_jax]   {len(DEV_TASKS)} tasks x {len(ALGOS)} algos x {len(DEV_SEEDS)} seed(s)")


if __name__ == "__main__":
    main()
