"""Fixed state banks for the cube probes, drawn once per env from the OGBench play dataset (qpos/qvel included).

Phases: P1 pre-contact (contact <= 0.5, effector-cube xy > 0.10), P2 at-contact (contact > 0.5,
not lifted, xy < 0.06), P3 held (contact > 0.5, lifted). P3 is also the held-start bank.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

# Observation layout
CONTACT_IDX = 18
EFF_OBS = slice(12, 15)
CUBE_OBS = slice(19, 22)
XYZ_SCALER = 10.0
CUBE_QPOS_Z = 16

TABLE_Z = 0.02  # cube resting height
LIFT_THRESH = 0.05  # absolute cube z, i.e. lift > delta_lift (0.03)
CONTACT_THRESH = 0.5
P2_XY_MAX = 0.06  # eps_reach
P1_XY_MIN = 0.10
K_STEPS = 5  # PhaseThresholds.k_steps

N_HELD = 64
N_P2 = 64
N_CONTROL = 32
RNG_SEED = 0

DATA_ROOT = Path(os.environ.get("OGBENCH_DATA_ROOT", Path.home() / ".ogbench" / "data"))


def dataset_path(env_name: str) -> Path:
    p = DATA_ROOT / f"{env_name}.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. This must be the SAME npz the methods trained on "
            "(impls/utils/env_utils.make_env_and_datasets -> ogbench), not a re-download."
        )
    return p


def load_raw(env_name: str) -> Dict[str, np.ndarray]:
    d = np.load(dataset_path(env_name))
    for k in ("observations", "actions", "terminals", "qpos", "qvel"):
        if k not in d.files:
            raise KeyError(f"{env_name}.npz lacks {k!r}; add_info columns are required")
    obs = np.asarray(d["observations"], np.float32)
    term = np.asarray(d["terminals"], bool)
    # Episode boundaries
    ends = np.flatnonzero(term)
    starts = np.concatenate([[0], ends[:-1] + 1])
    episode = np.zeros(len(obs), np.int32)
    step = np.zeros(len(obs), np.int32)
    for e, (s, t) in enumerate(zip(starts, ends)):
        episode[s : t + 1] = e
        step[s : t + 1] = np.arange(t + 1 - s)
    return dict(
        observations=obs,
        actions=np.asarray(d["actions"], np.float32),
        qpos=np.asarray(d["qpos"], np.float32),
        qvel=np.asarray(d["qvel"], np.float32),
        episode=episode,
        step=step,
        terminals=term,
    )


def _channels(raw: Dict[str, np.ndarray]):
    obs, qpos = raw["observations"], raw["qpos"]
    contact = obs[:, CONTACT_IDX]
    cube_z = qpos[:, CUBE_QPOS_Z]
    eff_xy = obs[:, EFF_OBS][:, :2] / XYZ_SCALER
    cube_xy = obs[:, CUBE_OBS][:, :2] / XYZ_SCALER
    dxy = np.linalg.norm(eff_xy - cube_xy, axis=1)
    return contact, cube_z, dxy


def _persists(mask: np.ndarray, episode: np.ndarray, k: int) -> np.ndarray:
    """True where `mask` holds for k consecutive steps starting here, within one episode."""
    out = mask.copy()
    for j in range(1, k):
        shifted = np.zeros_like(mask)
        shifted[:-j] = mask[j:] & (episode[j:] == episode[:-j])
        out &= shifted
    return out


def phase_masks(raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    contact, cube_z, dxy = _channels(raw)
    lifted = cube_z > LIFT_THRESH
    in_contact = contact > CONTACT_THRESH
    held = in_contact & lifted
    return dict(
        # Held states must persist k_steps
        P3=_persists(held, raw["episode"], K_STEPS),
        P2=in_contact & ~lifted & (dxy < P2_XY_MAX),
        P1=(~in_contact) & (dxy > P1_XY_MIN),
    )


def sample_bank(
    raw: Dict[str, np.ndarray], mask: np.ndarray, n: int, rng: np.random.Generator
) -> Dict[str, np.ndarray]:
    idx = np.flatnonzero(mask)
    if len(idx) < n:
        raise ValueError(f"only {len(idx)} candidate rows for a bank of {n}")
    sel = np.sort(rng.choice(idx, size=n, replace=False))
    return dict(
        row=sel.astype(np.int64),
        obs=raw["observations"][sel],
        action=raw["actions"][sel],
        qpos=raw["qpos"][sel],
        qvel=raw["qvel"][sel],
        episode=raw["episode"][sel],
        step=raw["step"][sel],
    )


def build(env_name: str, out_dir: Path) -> Dict[str, dict]:
    raw = load_raw(env_name)
    masks = phase_masks(raw)
    rng = np.random.default_rng(RNG_SEED)
    sizes = dict(P3=N_HELD, P2=N_P2, P1=N_CONTROL)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    banks = {}
    # Fixed draw order (P3, P2, P1) from one rng
    for phase in ("P3", "P2", "P1"):
        bank = sample_bank(raw, masks[phase], sizes[phase], rng)
        banks[phase] = bank
        stats[phase] = dict(
            candidates=int(masks[phase].sum()),
            fraction=float(masks[phase].mean()),
            banked=int(len(bank["row"])),
            episodes=int(len(np.unique(bank["episode"]))),
        )
    # Held-start bank is P3
    np.savez(out_dir / "held_bank.npz", **{k: v for k, v in banks["P3"].items()})
    np.savez(
        out_dir / "phase_banks.npz",
        **{f"{p}_{k}": v for p, b in banks.items() for k, v in b.items()},
    )
    stats["_meta"] = dict(
        env=env_name,
        rng_seed=RNG_SEED,
        k_steps=K_STEPS,
        table_z=TABLE_Z,
        lift_thresh=LIFT_THRESH,
        dataset=str(dataset_path(env_name)),
        rows=int(len(raw["observations"])),
        episodes=int(raw["episode"].max() + 1),
    )
    (out_dir / "banks_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--envs", nargs="+", default=["cube-single-play-v0", "cube-double-play-v0"]
    )
    ap.add_argument("--out-root", default="analysis/cube_bottleneck")
    args = ap.parse_args()
    for env_name in args.envs:
        stats = build(env_name, Path(args.out_root) / env_name)
        print(f"[banks] {env_name}")
        for k, v in stats.items():
            print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
