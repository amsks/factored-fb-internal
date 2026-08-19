import os, sys
sys.path.insert(0, "/work/11708/amsks/vista/git/Factored-FB")
import numpy as np
from env.maniskill_env import make_maniskill_env

env, _ = make_maniskill_env("ms-push-cube")
env.reset()
u = env.unwrapped
print("unwrapped:", type(u))
print("has render_rgb_array:", hasattr(u, "render_rgb_array"))
for attr in ["_human_render_cameras", "human_render_camera_configs", "scene", "_scene"]:
    print(f"  {attr}: {type(getattr(u, attr, None))}")
cams = getattr(u, "_human_render_cameras", None)
if cams:
    for k, v in cams.items():
        print("  cam", k, type(v))
        cam = getattr(v, "camera", v)
        for f in ["get_local_pose", "set_local_pose", "fovy", "get_intrinsic_matrix", "near", "far"]:
            print("     ", f, hasattr(cam, f))
        try:
            print("      intrinsic:\n", np.asarray(cam.get_intrinsic_matrix()))
        except Exception as e:
            print("      intrinsic err", e)
try:
    img = env.render()
    print("render() ->", np.asarray(img).shape)
except Exception as e:
    print("render failed:", type(e).__name__, e)
