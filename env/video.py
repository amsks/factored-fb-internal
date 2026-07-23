"""Eval video rollouts — adapter-agnostic.

Both adapters already return per-episode frame stacks from `evaluate()` (unified renders
`num_video` of its eval episodes; the OGBench adapter renders `num_video` EXTRA episodes on
top of the scored ones). Nothing consumed them: impls/main.py dropped the frames on the
floor, so the factored tree produced no videos for ANY task. This module is the consumer.

Frame stacks arrive with different shapes per family -- Box2D 80x80, ManiSkill 128x128,
Metaworld 224x224, OGBench 200x400 when a goal frame is stacked on top -- and with different
episode lengths (25 steps for ms-*, 200 for b2d-*). `save_video_grid` normalizes both: pad
every episode to the longest, pad every frame to the largest HxW, then tile into a near-square
grid. That keeps one writer for all four families instead of a per-adapter special case.
"""
import math
import os

import numpy as np


def _stack(ep):
    """One episode's frames -> uint8 (T, H, W, 3), or None if it holds no usable frames."""
    if ep is None or len(ep) == 0:
        return None
    arr = ep if isinstance(ep, np.ndarray) else np.stack([np.asarray(f) for f in ep])
    arr = np.asarray(arr)
    if arr.ndim == 3:                      # a lone (H, W, 3) frame
        arr = arr[None]
    if arr.ndim != 4:
        return None
    if arr.shape[-1] == 1:                 # greyscale -> RGB
        arr = np.repeat(arr, 3, -1)
    elif arr.shape[-1] == 4:               # RGBA -> RGB
        arr = arr[..., :3]
    elif arr.shape[1] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 1, -1)[..., :3]     # (T, C, H, W) -> (T, H, W, C)
    if arr.dtype != np.uint8:              # float renders are 0..1 or already 0..255
        arr = np.asarray(arr, np.float32)
        hi = float(np.nanmax(arr)) if arr.size else 1.0
        arr = np.clip(arr * (255.0 if hi <= 1.0 + 1e-6 else 1.0), 0, 255).astype(np.uint8)
    return arr


def save_video_grid(frames, out_path, fps=30, max_episodes=4):
    """Tile up to `max_episodes` episodes into one mp4. Returns the path, or None if there
    was nothing to write (no frames, or imageio/ffmpeg unavailable -- video is reporting, so
    a failure here must never take down a training run)."""
    eps = [a for a in (_stack(f) for f in (frames or [])) if a is not None][:max_episodes]
    if not eps:
        return None
    try:
        import imageio
    except ImportError:
        return None

    T = max(len(a) for a in eps)
    H = max(a.shape[1] for a in eps)
    W = max(a.shape[2] for a in eps)
    tiles = []
    for a in eps:
        if len(a) < T:                     # freeze the last frame of shorter episodes
            a = np.concatenate([a, np.repeat(a[-1:], T - len(a), 0)])
        ph, pw = H - a.shape[1], W - a.shape[2]
        if ph or pw:                       # letterbox mixed-resolution renders
            a = np.pad(a, ((0, 0), (0, ph), (0, pw), (0, 0)))
        tiles.append(a)

    ncol = int(math.ceil(math.sqrt(len(tiles))))
    nrow = int(math.ceil(len(tiles) / ncol))
    blank = np.zeros_like(tiles[0])
    tiles += [blank] * (nrow * ncol - len(tiles))
    grid = np.concatenate(
        [np.concatenate(tiles[r * ncol:(r + 1) * ncol], axis=2) for r in range(nrow)], axis=1)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    try:
        imageio.mimsave(out_path, grid, fps=fps, macro_block_size=1)
    except Exception as e:                 # missing ffmpeg plugin, unwritable path, ...
        print(f"[video] could not write {out_path}: {type(e).__name__}: {e}", flush=True)
        return None
    return out_path


def log_video(run, path, key="eval/video", step=None):
    """Attach a written mp4 to the wandb run, if there is one. Never raises."""
    if run is None or not path:
        return
    try:
        import wandb
        run.log({key: wandb.Video(path, format="mp4")}, step=step)
    except Exception as e:
        print(f"[video] wandb log failed: {type(e).__name__}: {e}", flush=True)
