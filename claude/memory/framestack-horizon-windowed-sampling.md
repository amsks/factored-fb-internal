---
name: framestack-horizon-windowed-sampling
description: windowed (horizon>1) buffer sampling must frame-stack pixel obs; synthetic-buffer smoke tests miss this
metadata: 
  node_type: memory
  type: project
  originSessionId: 54b70bad-1cae-4220-b346-499c5304e5f4
---

`DictBuffer.sample(B, horizon=h)` (windowed path, `_sample_horizon` in buffers/transition.py) historically ignored `frame_stack` — it gathered raw single frames and returned `[B, h, 3, H, W]` for pixels, while the non-windowed path frame-stacks to 9 channels via `_stack_obs`/`_stack_next_obs`. Only RLDP uses windowed sampling (horizon>1 for its self-predictive loss); FB/PSM use horizon=1. So RLDP pixel training crashed: the 9-channel DrQ encoder got 3-channel input. Fixed (2026-06-03) by frame-stacking each of the B*h window positions in `_sample_horizon`, gated on `self._stack` so the state path stays byte-identical.

**Why:** frame_stack and horizon both index the time axis; the correct windowed-pixel obs is `[B, h, C*frame_stack, H, W]` (each window step is itself a frame-stack).

**How to apply:** Any new agent using windowed sampling on pixels relies on this. When writing pixel smoke tests, build the buffer the REAL way — raw 3-channel frames + `DictBuffer(frame_stack=3, obs_type="pixels")` + a `timestep` key — NOT pre-stacked 9-channel frames with frame_stack=1. The original RLDP pixel smoke test used pre-stacked synthetic frames and gave false confidence (passed while real training crashed). Relates to [[ogbench-drq-augment-host-bound]] (windowed pixel path encodes h positions/step → ~9 FPS, slow at 500k scale).
