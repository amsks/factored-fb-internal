#!/bin/bash
# ManiSkill environment for TACC Vista (GH200 / aarch64). Run ON A LOGIN NODE.
#
#   bash scripts/vista_setup_maniskill.sh
#
# WHY A SECOND VENV. ManiSkill pulls torch (via pytorch_kinematics and its own tensor API),
# which this repo deliberately dropped when it went JAX-only. Installing it into
# .venv-jax-fb would also mutate an environment that in-flight training jobs are running
# out of. So ManiSkill gets its own venv and the Metaworld/OGBench venv stays untouched.
#
# WHY SAPIEN COMES FROM GITHUB. ManiSkill needs SAPIEN, and SAPIEN publishes NO aarch64
# wheel to PyPI (x86_64/macOS/Windows only) — a plain `pip install mani_skill` therefore
# fails to resolve on this machine. The aarch64 wheels are built and attached to GitHub
# Releases instead (haosulab/SAPIEN issue #294 tracks publishing them to PyPI). We install
# that wheel first so the later resolve sees `sapien` already satisfied.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${MS_VENV:-$SCRATCH/.venv-ms}"
SAPIEN_VER="${SAPIEN_VER:-3.0.3}"
WHEEL="sapien-${SAPIEN_VER}-cp311-cp311-linux_aarch64.whl"
URL="https://github.com/haosulab/SAPIEN/releases/download/${SAPIEN_VER}/${WHEEL}"
CACHE="${SCRATCH}/wheels"

# Vista login nodes cap a user at ~100 threads; uv's rayon pool otherwise aborts.
export RAYON_NUM_THREADS=4 UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_INSTALLS=4 UV_CONCURRENT_BUILDS=1

echo "[ms-setup] venv = $VENV"
uv venv --python 3.11 "$VENV"
uv pip install --python "$VENV/bin/python" -r "$REPO/requirements.txt"

echo "[ms-setup] fetching SAPIEN $SAPIEN_VER aarch64 wheel"
mkdir -p "$CACHE"
[ -f "$CACHE/$WHEEL" ] || curl -fL --retry 3 -o "$CACHE/$WHEEL" "$URL"
uv pip install --python "$VENV/bin/python" "$CACHE/$WHEEL"

echo "[ms-setup] installing mani_skill (sapien already satisfied from the wheel above)"
uv pip install --python "$VENV/bin/python" mani_skill

"$VENV/bin/python" - <<'PY'
import sapien, mani_skill
print(f"[ms-setup] sapien {sapien.__version__}  mani_skill {mani_skill.__version__}")
PY

echo "[ms-setup] done. Env construction still needs a GPU node — verify with:"
echo "  sbatch -A ASC26008 -p gh-dev -t 00:20:00 --wrap=\"$VENV/bin/python -c 'import gymnasium,mani_skill.envs; e=gymnasium.make(\\\"PickCube-v1\\\"); print(e.reset()[0].shape)'\""
