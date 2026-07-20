"""campaign/gen_jax.py — emit the JAX-baseline job list (gciql, crl, crl_flowbc x
19 datasets x 3 seeds = 171 jobs), each training-only on a unified dataset via
campaign/jax_train_unified.py.

    <fb>/.venv/bin/python campaign/gen_jax.py   ->  campaign/jobs_jax.txt
"""
from pathlib import Path

from registry import (DATASETS, JAX_AGENTS, SEEDS, NUM_TRAIN_STEPS, SAVE_EVERY,
                      CAMPAIGN_OUT, COLLECTED)

FB = "/raid/tongle/rl/Factored-FB"
IMPLS = f"{FB}/third_party/ogbench/impls"
SHIM = f"{FB}/tools/wandb_mode_shim"
UNIFIED_CUBE = "/raid/tongle/rl/fb-data/unified/cube-single-play-v0"
JOBS = Path(FB) / "campaign" / "jobs_jax.txt"


def jax_data_dir(ds):
    if ds["family"] == "OGBench":
        return UNIFIED_CUBE                      # cube-single in unified format (1M transitions)
    return f"{COLLECTED}/{ds['family']}__{ds['slug']}"


def main():
    lines = []
    for ds in DATASETS:
        ddir = jax_data_dir(ds)
        for aname, a in JAX_AGENTS.items():
            overrides = ""
            if "alpha" in a:
                overrides = f"--set alpha={a['alpha']}"
            elif "bc_coeff" in a:
                overrides = f"--set bc_coeff={a['bc_coeff']}"
            for s in SEEDS:
                save_dir = f"{CAMPAIGN_OUT}/jax/{aname}__{ds['slug']}__s{s}"
                cmd = (
                    f"XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 PYTHONPATH={IMPLS}:{SHIM} "
                    f".venv-jax/bin/python campaign/jax_train_unified.py "
                    f"--data_dir {ddir} --agent_file {a['agent_file']} "
                    f"--save_dir {save_dir} --train_steps {NUM_TRAIN_STEPS} "
                    f"--save_interval {SAVE_EVERY} --seed {s} {overrides}"
                )
                lines.append(cmd)
    JOBS.write_text("\n".join(lines) + "\n")
    print(f"[gen_jax] wrote {len(lines)} jobs -> {JOBS}")
    print(f"[gen_jax]   {len(DATASETS)} datasets x {len(JAX_AGENTS)} agents x {len(SEEDS)} seeds")


if __name__ == "__main__":
    main()
