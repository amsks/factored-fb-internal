"""eval/video.py — turn rendered rollout frames into files a human can look at.

Every writer takes frames as `[T, H, W, 3]` uint8 (or anything np.asarray can coerce to
that) and returns the written path, or None when there is nothing to write — callers are
in eval paths where a missing video must never take down a training run.

Renders are cheap enough to do inline: Metaworld under EGL on a GH200 costs ~7 ms/frame
against a ~1 ms/step no-render baseline, so a 100-step episode adds well under a second.
The expensive thing is rendering the *metric* episodes, which the rollout loop never does.
"""
from pathlib import Path

import numpy as np


def _as_uint8(frames):
    """-> [T,H,W,3] uint8, or None if there is nothing usable."""
    if frames is None or len(frames) == 0:
        return None
    arr = np.asarray(frames)
    if arr.ndim != 4 or arr.shape[0] == 0:
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _write(arr, out_path, fps):
    import imageio
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # macro_block_size=1: MuJoCo frames are 224x224, not a multiple of the default 16, and
    # imageio would otherwise silently resize them.
    imageio.mimsave(str(out_path), arr, fps=fps, macro_block_size=1)
    return str(out_path)


def save_video(frames, out_path, fps=30):
    """One episode -> one mp4."""
    arr = _as_uint8(frames)
    return None if arr is None else _write(arr, out_path, fps)


def _pad_to(arr, T):
    """Freeze the last frame so episodes that ended early still fill the grid."""
    if len(arr) >= T:
        return arr[:T]
    return np.concatenate([arr, np.repeat(arr[-1:], T - len(arr), axis=0)])


def save_video_grid(episodes, out_path, fps=30, cols=2):
    """Up to `cols*cols` episodes tiled into one mp4, padded to the longest.

    Comparing agents is much easier from one file than from N; this is the default
    artifact for an eval point.
    """
    tiles = [a for a in (_as_uint8(e) for e in episodes) if a is not None]
    if not tiles:
        return None
    n = min(len(tiles), cols * cols)
    tiles = tiles[:n]
    T = max(len(t) for t in tiles)
    H = min(t.shape[1] for t in tiles)
    W = min(t.shape[2] for t in tiles)
    tiles = [_pad_to(t, T)[:, :H, :W] for t in tiles]
    while len(tiles) < cols * cols:                 # blank cells keep the grid rectangular
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[r * cols:(r + 1) * cols], axis=2) for r in range(cols)]
    return _write(np.concatenate(rows, axis=1), out_path, fps)


def save_filmstrip(frames, out_path, n_frames=8):
    """Evenly-spaced frames of one episode as a single wide PNG — static figure material."""
    arr = _as_uint8(frames)
    if arr is None:
        return None
    idx = np.linspace(0, len(arr) - 1, min(n_frames, len(arr))).round().astype(int)
    strip = np.concatenate([arr[i] for i in idx], axis=1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import imageio
    imageio.imwrite(str(out_path), strip)
    return str(out_path)


def save_contrast(success_frames, failure_frames, out_path, fps=30):
    """One successful and one failed episode side by side in a single mp4.

    This is the clip that localises a failure: on a grasp-transport task like bin-picking,
    watching the two diverge says whether the policy never grasped or grasped and dropped —
    a distinction the scalar success rate cannot make.
    """
    a, b = _as_uint8(success_frames), _as_uint8(failure_frames)
    if a is None or b is None:
        return save_video(a if a is not None else b, out_path, fps)
    T = max(len(a), len(b))
    H = min(a.shape[1], b.shape[1])
    W = min(a.shape[2], b.shape[2])
    a, b = _pad_to(a, T)[:, :H, :W], _pad_to(b, T)[:, :H, :W]
    return _write(np.concatenate([a, b], axis=2), out_path, fps)
