"""scripts/count_fb_params.py — Parameter-count comparison.

Builds the FB(+FlowBC) agent for two configs the *same* way train.py does
(Hydra compose -> create_ogbench_env for obs_space/action_dim -> make_agent),
then reports a per-submodule trainable-parameter breakdown.

Goal: is the state `L_dim=256` ablation parameter-matched to the pixel DrQ run,
or does the DrQ conv stack add capacity the state ablation never had?

    python scripts/count_fb_params.py

CPU-only; no dataset load. Pixel env needs an EGL/OSMesa MuJoCo backend.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import torch
from hydra import compose, initialize
from omegaconf import open_dict

from envs.ogbench import create_ogbench_env
from train import make_agent


# Target-network copies (_target_*) are non-trainable deep-copies; count them
# separately so the "trainable" figure reflects real model capacity.
def is_target(name: str) -> bool:
    return name.startswith("_target_") or "._target_" in name or "target_" in name.split(".")[0]


def breakdown(model: torch.nn.Module):
    """Group params by the top-level submodule attribute name."""
    groups: dict[str, dict] = {}
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        g = groups.setdefault(top, {"trainable": 0, "frozen": 0})
        g["trainable" if p.requires_grad else "frozen"] += p.numel()
    return groups


def build(overrides):
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    with open_dict(cfg):
        cfg.device = "cpu"
    env, _ = create_ogbench_env(
        cfg.domain, obs_type=cfg.obs_type, seed=0,
        frame_stack=cfg.get("frame_stack", 1),
    )
    obs_space = env.observation_space
    action_dim = env.action_space.shape[0]
    agent = make_agent(cfg, obs_space, action_dim)
    return cfg, agent, obs_space, action_dim


def report(tag, overrides):
    print(f"\n{'='*72}\n{tag}\n  overrides: {overrides}\n{'='*72}")
    cfg, agent, obs_space, action_dim = build(overrides)
    model = agent.model
    print(f"  domain={cfg.domain}  obs_type={cfg.obs_type}  obs_space={tuple(obs_space.shape)}  "
          f"action_dim={action_dim}")
    print(f"  L_dim={cfg.L_dim}  z_dim={cfg.z_dim}  agent={cfg.agent}")

    groups = breakdown(model)
    total_train = total_frozen = 0
    print(f"  {'submodule':<24}{'trainable':>14}{'frozen(target)':>16}")
    print(f"  {'-'*24}{'-'*14}{'-'*16}")
    for name in sorted(groups, key=lambda n: -groups[n]["trainable"]):
        g = groups[name]
        total_train += g["trainable"]
        total_frozen += g["frozen"]
        print(f"  {name:<24}{g['trainable']:>14,}{g['frozen']:>16,}")
    print(f"  {'-'*24}{'-'*14}{'-'*16}")
    print(f"  {'TOTAL':<24}{total_train:>14,}{total_frozen:>16,}")
    return {"tag": tag, "trainable": total_train, "frozen": total_frozen, "groups": groups}


if __name__ == "__main__":
    # State FB+FlowBC ablation with L_dim=256 (matches the fb_state_L256 run:
    # cube_single domain, ortho_coef/lr_b are FB-loss knobs and don't affect param count).
    state = report("STATE  FB+FlowBC  L_dim=256  (fb_state_L256 ablation)",
                   ["domain=cube_single", "L_dim=256"])

    # Pixel FB+FlowBC with the DrQ encoder (visual_cube_single already sets L_dim=256).
    pixel = report("PIXEL  FB+FlowBC  DrQ encoder  (visual_cube_single)",
                   ["domain=visual_cube_single"])

    # Param-matched state ablation: two MLP encoders in the fw/bw slots.
    encmatch = report("STATE  FB+FlowBC  MLP encoders  (cube_single_encmatch ablation)",
                      ["domain=cube_single_encmatch"])

    def online(r):
        return sum(g["trainable"] for n, g in r["groups"].items() if not n.startswith("_target_"))

    print(f"\n{'='*72}\nONLINE (targets excluded) — the matched quantity\n{'='*72}")
    for r in (state, pixel, encmatch):
        print(f"  {r['tag'][:48]:<48}{online(r):>14,}")
    gap = online(encmatch) - online(pixel)
    print(f"  encmatch - pixel : {gap:>+14,}  ({100.0*gap/online(pixel):+.2f}%)")

    d = pixel["trainable"] - state["trainable"]
    print(f"\n{'='*72}\nDIFFERENCE (pixel - state)\n{'='*72}")
    print(f"  state trainable : {state['trainable']:>14,}")
    print(f"  pixel trainable : {pixel['trainable']:>14,}")
    print(f"  delta           : {d:>+14,}  "
          f"({100.0*d/state['trainable']:+.1f}% vs state)")
    enc = sum(pixel["groups"].get(k, {"trainable": 0})["trainable"]
              for k in ("_fw_encoder", "_bw_encoder"))
    print(f"  of which DrQ encoders (_fw_encoder+_bw_encoder): {enc:,}")
