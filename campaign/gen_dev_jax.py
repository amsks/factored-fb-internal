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

DEV_SEEDS = [0, 1, 2]                 # 3 seeds (report mean +/- std)
NEWT_PY = "/raid/tongle/rl/newt/.venv/bin/python"   # unified train+eval venv (jax 0.4.38 + all envs)

ALGOS = {
    "fb":         ("campaign/jax_agents/fb.py", ""),
    "fb_flowbc":  ("campaign/jax_agents/fb_flowbc.py", ""),
    "gciql":      ("agents/gciql.py", "--set alpha=1.0"),
    "crl":        ("agents/crl.py", "--set alpha=3.0"),
    "crl_flowbc": ("tools/wandb_mode_shim/crl_flowbc.py", "--set bc_coeff=3.0"),
    "flowbc":     ("campaign/jax_agents/flowbc.py", ""),   # pure flow-matching BC (ablation)
}

_FAM_DIR = {"box2d": "Box2D", "metaworld": "Metaworld", "maniskill": "ManiSkill"}


def _ds_by_slug():
    return {d["slug"]: d for d in DATASETS}


def data_dir(ds):
    if ds["family"] == "OGBench":
        return UNIFIED_CUBE
    return f"{COLLECTED}/{ds['family']}__{ds['slug']}"


RUN_TAG = "v3"                                       # clean relaunch w/ dwell metric; fresh jsonls (v2 was success-only)
AGG_DIR = f"{CAMPAIGN_OUT}/dev_jax_eval/{RUN_TAG}"    # per-seed eval jsonls -> wandb_agg_daemon aggregates


def main():
    by_slug = _ds_by_slug()
    lines = []
    # seed-INNERMOST so a group's 3 seeds are contiguous in the queue -> they co-run and
    # the {algo}__{task}__agg mean+/-var run fills in early (needs all seeds at a step).
    # algo outer, task next -> the first wave still spans all dev tasks for the first algos.
    for algo, (af, ov) in ALGOS.items():
        for slug in DEV_TASKS:
            for s in DEV_SEEDS:
                ds = by_slug[slug]
                ddir = data_dir(ds)
                save_dir = f"{CAMPAIGN_OUT}/dev_jax/{RUN_TAG}/{algo}__{slug}__s{s}"
                cmd = (
                    f"XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl "  # on-demand mem + headless render
                    f"PYTHONPATH={IMPLS}:{SHIM}:{JAX_AGENTS} "
                    f"{NEWT_PY} campaign/jax_train_unified.py "
                    f"--data_dir {ddir} --agent_file {af} --save_dir {save_dir} "
                    f"--train_steps {NUM_TRAIN_STEPS} --save_interval {SAVE_EVERY} "
                    f"--seed {s} {ov} "
                    f"--eval_task {slug} --eval_interval 20000 --eval_episodes 50 "
                    f"--eval_video 4 --eval_video_interval 100000 --agg_dir {AGG_DIR} "
                    f"--wandb 1 --wandb_project factored-fb-dev "
                    f"--wandb_group {algo}__{slug} --wandb_family {ds['family']} "  # seeds share group -> mean+/-var
                    f"--wandb_name {algo}__{slug}__s{s}"
                )
                lines.append(cmd)
    JOBS.write_text("\n".join(lines) + "\n")
    print(f"[gen_dev_jax] wrote {len(lines)} jobs -> {JOBS}")
    print(f"[gen_dev_jax]   {len(DEV_TASKS)} tasks x {len(ALGOS)} algos x {len(DEV_SEEDS)} seed(s)")


if __name__ == "__main__":
    main()
