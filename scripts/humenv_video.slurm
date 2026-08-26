#!/bin/bash
# First HumEnv rollout video + a live check of the Meta Motivo agent API.
#
# Three things get established here, none of which need AMASS:
#   1. HumEnv builds and renders under EGL on a gh node.
#   2. The released FB-CPR checkpoint loads and its act()/goal_inference() surface works --
#      that is exactly the interface humenv.bench.GoalEvaluation drives, so this is the
#      dry run for the real eval.
#   3. A goal pose taken from OUR converted offline buffer can be reached-at, which shows
#      the goal arm is runnable before goal_poses.json exists.
#
#   sbatch -A ASC26008 scripts/humenv_video.slurm
#
#SBATCH -J humenv-video
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:40:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
PY="${VENV_HUMENV:-$SCRATCH/.venv-humenv}/bin/python"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0
export HF_HOME="$SCRATCH/humenv-assets/hf"
OUT="$SCRATCH/humenv-video"; mkdir -p "$OUT"
echo "== host $(hostname) =="
$PY - "$OUT" <<'PYEOF'
import os, sys, glob
import subprocess
import numpy as np
out = sys.argv[1]

# The SYSTEM ffmpeg is unusable on the gh compute nodes -- `/usr/bin/ffmpeg` dies with
# "libunwind.so.8: cannot open shared object file" there (it runs fine on the login node,
# which is why a `command -v` check passed). imageio-ffmpeg ships a STATIC binary and is
# installed in the jax venv, so borrow that binary rather than the package.
FFMPEG = os.environ.get(
    "FFMPEG_BIN",
    os.environ["SCRATCH"] + "/.venv-jax-fb/lib/python3.11/site-packages/"
    "imageio_ffmpeg/binaries/ffmpeg-linux-aarch64-v7.0.2")


def write_mp4(path, frames, fps=30):
    """Pipe raw RGB into a working ffmpeg.

    imageio is installed in this venv but imageio-ffmpeg is NOT, so imageio.mimwrite raises
    "Could not find a backend ... with iomode `wI`". ffmpeg itself is on PATH, so feed it
    rawvideo on stdin rather than adding a dependency to a venv other work depends on."""
    f0 = np.asarray(frames[0])
    print(f"  frame0 {f0.shape} {f0.dtype}", flush=True)
    h, w, _ = f0.shape
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in frames:
            p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
        p.stdin.close()
    except BrokenPipeError:
        pass          # ffmpeg died; the return code below is what reports it
    return p.wait()

import gymnasium
from humenv import make_humenv
# FlattenObservation is GoalEvaluation's OWN default wrapper: HumEnv's native observation
# space is a Dict, so without it `obs` is a dict and every downstream shape check is
# meaningless (the first run printed `obs ()`).
env, _ = make_humenv(num_envs=1, state_init="Fall",
                     wrappers=[gymnasium.wrappers.FlattenObservation])
obs, info = env.reset(seed=0)
obs = np.asarray(obs, dtype=np.float32).reshape(-1)
print("env ok: obs", obs.shape, "act", env.action_space.shape, flush=True)

def rollout(policy, n=150, tag="rand"):
    frames = []
    o = np.asarray(env.reset(seed=0)[0], dtype=np.float32).reshape(-1)
    for t in range(n):
        a = policy(o, t)
        o, r, term, trunc, _ = env.step(a)
        o = np.asarray(o, dtype=np.float32).reshape(-1)
        frames.append(env.render())
        if term or trunc:
            o = np.asarray(env.reset(seed=0)[0], dtype=np.float32).reshape(-1)
    p = os.path.join(out, f"humenv-{tag}.mp4")
    rc = write_mp4(p, [np.asarray(f) for f in frames])
    ok = rc == 0 and os.path.exists(p)
    print(f"{'WROTE' if ok else 'FAILED'} {p}  ({len(frames)} frames"
          + (f", {os.path.getsize(p)/1e6:.1f} MB)" if ok else f", ffmpeg rc={rc})"), flush=True)

rollout(lambda o, t: env.action_space.sample(), tag="random")

# ---- the released FB-CPR model, through the same surface GoalEvaluation drives ----
try:
    import torch
    from metamotivo.fb_cpr.huggingface import FBcprModel
    model = FBcprModel.from_pretrained("facebook/metamotivo-S-1")
    model.eval()
    print("checkpoint loaded:", type(model).__name__, flush=True)

    # A goal pose out of OUR converted buffer: the benchmark's distance is defined on
    # obs[:214], and goal_inference expects a full observation-shaped goal.
    # The converted buffer, at the path hum-humanoid.yaml's `data_dir` resolves to. Do NOT
    # invent this path: the first run globbed a made-up directory, found nothing, and the
    # `if goal is not None` below skipped the whole Motivo stage in SILENCE -- the job
    # reported success having tested nothing.
    root = os.environ.get("FB_HUMENV_ROOT", os.environ["SCRATCH"] + "/fb-data/humenv-S1")
    eps = sorted(glob.glob(os.path.join(root, "buffer", "episode_*.npz")))
    if not eps:
        raise FileNotFoundError(f"no episode_*.npz under {root}/buffer -- goal stage cannot run")
    d = np.load(eps[0])
    goal = d["observation"][-1].astype(np.float32)
    print(f"goal pose from {os.path.basename(eps[0])} {goal.shape} "
          f"({len(eps)} episodes available)", flush=True)
    if True:
        # The signature is goal_inference(next_obs), POSITIONAL -- there is no `goal=`
        # kwarg (metamotivo/fb/model.py:150). It is B(next_obs) projected onto the
        # sqrt(d) sphere, i.e. the goal is supplied as an OBSERVATION, which is why taking
        # the last observation of an episode is the right way to build one.
        z = model.goal_inference(torch.tensor(goal, dtype=torch.float32).unsqueeze(0))
        print("goal_inference -> z", tuple(z.shape), float(z.norm()), flush=True)
        def pol(o, t):
            with torch.no_grad():
                a = model.act(torch.tensor(np.asarray(o), dtype=torch.float32).unsqueeze(0), z, mean=True)
            return a.squeeze(0).cpu().numpy()
        rollout(pol, tag="motivo-goal")
except Exception as e:
    import traceback; traceback.print_exc()
    print("MOTIVO STAGE FAILED:", type(e).__name__, e, flush=True)
PYEOF
echo "== done; videos under $OUT =="
ls -la "$OUT" 2>/dev/null
