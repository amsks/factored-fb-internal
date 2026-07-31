"""eval/ — everything that measures a TRAINED policy, in one place.

The split against env/: `env/` owns WHERE a policy acts (simulators, offline data, the
domain config); `eval/` owns HOW it is measured (rollouts, metrics, videos, checkpoint
restore). Before this package the two were tangled — the rollout loop lived inside
`env/unified.py:UnifiedAdapter.evaluate` while a second, separately-written copy lived in
`campaign/eval/eval_jax.py`, so an in-training number and a post-hoc number could disagree
with nothing to catch it.
"""
from eval.video import save_contrast, save_filmstrip, save_video, save_video_grid

__all__ = ["save_video", "save_video_grid", "save_filmstrip", "save_contrast"]
