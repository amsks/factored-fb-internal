# GCIQL eval-video GIF + FB key scheme — design

**Date:** 2026-05-18
**Status:** Approved (design)

## Problem

GCIQL (vendored OGBench, JAX track) logs eval visualizations to wandb as a
single tiled **mp4** at fps=15 under the key `video`. FB logs eval
visualizations as **GIFs** via `imageio.mimsave(path, arr, format="GIF",
fps=30, loop=0)` and `wandb.Video(gif_path, fps=30, format="gif")` under the
key `eval_video/{task}` (see `evals/ogbench.py:107-119`). Now that GCIQL and
FB log to the same wandb entity/project (`amsks/factored-fb`), the eval media
should use FB's recording format and namespace so the two are consistent in
the same wandb media section.

`third_party/ogbench/` is a byte-for-byte vendored copy and must not be edited
(per its PROVENANCE rules). All behavior changes go through the existing
auto-imported shim `tools/wandb_mode_shim/sitecustomize.py`, which is already
active for the GCIQL child process when `GCIQL_FB_ENV` is set and already
wraps `wandb.log` to mirror eval scalars into FB's key scheme.

## Scope

Chosen scope: **Format + FB key scheme** (not full per-task replication).

- GIF format, fps=30, loop=0 — FB's exact `imageio` recording parameters.
- Keep OGBench's single tiled layout (do not de-tile into per-task clips).
- Additionally log the clip under FB's `eval_video/` namespace.

Out of scope: per-task GIFs under `eval_video/{task}` (would require
intercepting renders before OGBench tiles them — unnecessary for this goal).

## Approach

Approach A: wrap `wandb.Video` and extend the existing `wandb.log` wrapper in
the shim. No vendored edits; no `run_gciql.py` change (the `GCIQL_FB_ENV` gate
already exists). Rejected alternatives: (B) deferred monkeypatch of
`get_wandb_video` — more invasive, unnecessary; (C) let wandb/moviepy encode
the gif — less faithful to FB's `imageio.mimsave` encoder.

## Components

### 1. `wandb.Video` wrapper (new, in shim)

Installed only when the shim is active (`GCIQL_FB_ENV` set; same gate as the
existing behaviors). Replace `wandb.Video` with a factory:

- If `data` is a `numpy.ndarray` **and** `format == 'mp4'` (OGBench's tiled
  eval-clip call signature `wandb.Video(renders, fps=15, format='mp4')`):
  - Convert the array to frames: OGBench's `get_wandb_video` calls
    `reshape_video(...)` which returns shape `(t, c, H, W)` uint8 (confirmed
    in `third_party/ogbench/impls/utils/log_utils.py`); transpose to
    `(t, H, W, c)` for imageio.
  - `imageio.mimsave(<tempfile>.gif, frames, format="GIF", fps=30, loop=0)`
    — FB's exact parameters.
  - Return the **original** `wandb.Video(gif_path, fps=30, format='gif')`.
- Otherwise: passthrough to the original `wandb.Video` unchanged.

The conversion is wrapped in its own `try/except`: on any error, fall back to
the original `wandb.Video(data, fps, format)` (mp4). Training never breaks.

### 2. Re-key in the existing `_log` wrapper (modify)

The shim already wraps `wandb.log` (`_log` / `_fb_keys`) to add FB-scheme
scalar keys. Extend it so that when the logged dict contains a `video` key,
it **additively** also sets `eval_video/all_tasks` to the same Video object.
The original `video` key is kept (consistent with the shim's existing
additive philosophy for scalar mirroring).

Key name `eval_video/all_tasks`: OGBench produces one tiled clip spanning all
tasks; FB's `eval_video/{task}` is per-task and has no single equivalent, so a
fixed `all_tasks` suffix under FB's `eval_video/` namespace is used.

## Data flow

```
OGBench eval
  -> get_wandb_video(renders, n_cols, fps=15)         # vendored, unchanged
  -> wandb.Video(ndarray, fps=15, format='mp4')       # WRAPPED -> GIF fps30 loop0
  -> eval_metrics['video'] = <Video(gif)>             # vendored, unchanged
  -> wandb.log(eval_metrics, step=i)                  # WRAPPED -> also adds
                                                      #   eval_video/all_tasks
  -> wandb
```

Active only when `GCIQL_FB_ENV` is set (the `run_gciql.py` sweep path). For
any unrelated Python process the shim remains fully inert.

## Error handling

- All shim logic stays within the existing top-level `try/except` that makes
  the file inert on import failure.
- The ndarray→GIF conversion has an inner `try/except` that falls back to the
  original mp4 `wandb.Video`. A rendering/encoding problem degrades video
  format but never interrupts the GCIQL run.

## Dependencies

- `imageio` 2.37.3 and `imageio_ffmpeg` 0.6.0 are present in the JAX venv
  (`/dev/shm/.venv-jax`) — verified. Same library/version family FB uses.
- `wandb` 0.27.0 `wandb.Video` accepts both `np.ndarray` and a gif path with
  `format` in `{gif, mp4, webm, ogg}` — verified.

## Testing

Standalone script (no GPU/JAX/network), run with the JAX venv:

1. Set `GCIQL_FB_ENV=cube-single-play-v0`, import the shim.
2. Call `wandb.Video(np.zeros((4,3,16,16), uint8), fps=15, format='mp4')`;
   assert the returned object is backed by a `.gif` (format gif) — i.e., the
   wrapper converted it.
3. Call the wrapped `wandb.log({'video': V, 'evaluation/overall_success': 1.0})`
   with `wandb` in disabled/offline mode (or a stubbed `wandb.log`); assert
   the data passed downstream contains `eval_video/all_tasks` and the original
   `video`, plus the existing scalar mirror keys.
4. Negative: with `GCIQL_FB_ENV` unset, `wandb.Video(ndarray, format='mp4')`
   is unchanged (passthrough) and `wandb.log` adds nothing.

Run offline before relying on the live sweep; the running sweep is unaffected
until its next relaunch (the shim is read at child-process start).

## Rollout

The change is picked up by new GCIQL processes only (sitecustomize is read at
interpreter start). The currently-running sweep (`gciql_state_20260518_201030`)
keeps its mp4 videos until a future relaunch; no restart is forced by this
change.
