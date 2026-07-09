#!/usr/bin/env bash
# One-shot install for TACC Vista / linux-aarch64 (Box2D built from source).
# See docs/install.md for details and troubleshooting.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

if ! command -v module >/dev/null 2>&1; then
  echo "[install-vista] 'module' not found; load gcc/swig manually if Box2D build fails."
else
  module load gcc/14.2.0 swig/4.2.1 2>/dev/null || module load swig/4.2.1 2>/dev/null || true
fi

export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

RLMINI="${RLMINI_ENVS_PATH:-}"
if [[ -z "$RLMINI" || ! -d "$RLMINI" ]]; then
  for candidate in \
    "$ROOT/../rlmini-envs" \
    "$ROOT/../rlmini_envs" \
    "/work/11494/calebchuck/vista/rlmini-envs"; do
    if [[ -d "$candidate" ]]; then
      RLMINI="$(cd "$candidate" && pwd -P)"
      break
    fi
  done
fi

if [[ -z "$RLMINI" || ! -d "$RLMINI" ]]; then
  echo "[install-vista] ERROR: rlmini-envs not found."
  echo "  Clone it next to Factored-FB, or set RLMINI_ENVS_PATH=/path/to/rlmini-envs"
  exit 1
fi

# uv path dep in pyproject.toml is ../rlmini-envs relative to this repo.
if [[ ! -d "$ROOT/../rlmini-envs" && "$RLMINI" != "$ROOT/../rlmini-envs" ]]; then
  echo "[install-vista] Linking $ROOT/../rlmini-envs -> $RLMINI"
  ln -sfn "$RLMINI" "$ROOT/../rlmini-envs"
fi

echo "[install-vista] ROOT=$ROOT  RLMINI=$RLMINI  CC=$CC  CXX=$CXX"
uv venv --python 3.11
# shellcheck disable=SC1091
source .venv/bin/activate
make install-vista

python -c "import Box2D; import rlmini_envs; print('[install-vista] Box2D + rlmini_envs OK')"
