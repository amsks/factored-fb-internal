"""Thin Hydra launcher for the vendored OGBench (JAX) GCIQL track.

This module is PURE ORCHESTRATION and MUST NOT import jax/flax/the OGBench
training code. It translates a Hydra config into OGBench's exact absl CLI
flags and runs `third_party/ogbench/impls/main.py` in the isolated .venv-jax
via subprocess. The two tracks never share a process or environment.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent
OGBENCH_IMPLS = REPO_ROOT / "third_party" / "ogbench" / "impls"


def build_ogbench_argv(cfg: Mapping[str, Any]) -> list[str]:
    """Translate launcher config -> OGBench main.py absl argv (deterministic)."""
    argv = [
        f"--env_name={cfg['env_name']}",
        f"--agent={cfg['agent_file']}",
        f"--seed={cfg['seed']}",
        f"--train_steps={cfg['train_steps']}",
        f"--log_interval={cfg['log_interval']}",
        f"--eval_interval={cfg['eval_interval']}",
        f"--save_interval={cfg['save_interval']}",
        f"--eval_episodes={cfg['eval_episodes']}",
        f"--eval_on_cpu={cfg['eval_on_cpu']}",
        f"--run_group={cfg['run_group']}",
        f"--save_dir={cfg['output_root']}",
    ]
    overrides = cfg.get("agent_overrides") or {}
    for key in sorted(overrides):
        argv.append(f"--agent.{key}={overrides[key]}")
    return argv


def _preflight(cfg: Mapping[str, Any]) -> Path:
    """Fail fast with actionable messages if the JAX track isn't provisioned."""
    if not OGBENCH_IMPLS.is_dir():
        sys.exit(
            f"[run_gciql] vendored OGBench not found: {OGBENCH_IMPLS}\n"
            "Re-vendor it (see third_party/ogbench/PROVENANCE.md)."
        )
    jax_python = Path(str(cfg["jax_python"])).expanduser()
    if not jax_python.exists():
        sys.exit(
            f"[run_gciql] JAX interpreter not found: {jax_python}\n"
            "Mount the NVMe and run `make install-jax` first "
            "(see the spec's Prerequisites section)."
        )
    symlink = Path(str(cfg["ogbench_home_symlink"])).expanduser()
    if not symlink.exists():
        sys.exit(
            f"[run_gciql] {symlink} missing. Create it so OGBench writes "
            "datasets to the NVMe:\n"
            f"  ln -s /mnt/scratch1/ogbench_data {symlink}"
        )
    out_root = Path(str(cfg["output_root"]))
    out_root.mkdir(parents=True, exist_ok=True)
    return jax_python


@hydra.main(version_base="1.3", config_path="configs/gciql",
            config_name="cube_single_state")
def main(cfg: DictConfig) -> None:
    plain = OmegaConf.to_container(cfg, resolve=True)
    argv = build_ogbench_argv(plain)
    jax_python = _preflight(plain)

    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    # OGBench's vendored setup_wandb hardcodes wandb.init(mode='online'), which
    # (wandb >= 0.27) overrides WANDB_MODE/WANDB_DISABLED env vars. Since the
    # vendored code must not be edited, force the mode via an auto-imported
    # sitecustomize shim on PYTHONPATH (no-op unless the flag below is set).
    mode = str(plain.get("wandb_mode", "online"))
    if mode in ("offline", "disabled"):
        env["GCIQL_WANDB_FORCE_MODE"] = mode
        # belt-and-suspenders; "disabled" is not a valid WANDB_MODE value
        env["WANDB_MODE"] = "offline" if mode == "disabled" else mode
        shim_dir = str(REPO_ROOT / "tools" / "wandb_mode_shim")
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = shim_dir + (os.pathsep + existing if existing else "")

    run_meta = {
        "argv": argv,
        # best-effort provenance; empty string if git is unavailable
        "factored_fb_sha": subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "ogbench_pinned": "1d4140997f60c52c6fb0702ec100dc988b18c548",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = Path(str(plain["output_root"])) / "last_run_gciql.json"
    meta_path.write_text(json.dumps(run_meta, indent=2))
    print(f"[run_gciql] {jax_python} main.py {' '.join(argv)}")

    proc = subprocess.run(
        [str(jax_python), "main.py", *argv],
        cwd=str(OGBENCH_IMPLS), env=env,
    )
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
