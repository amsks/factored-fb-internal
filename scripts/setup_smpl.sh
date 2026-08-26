#!/bin/bash
# Unpack the SMPL body models into the layout humenv/data_preparation expects.
#
#   bash scripts/setup_smpl.sh
#
# Prerequisite (MANUAL, and the only manual step): register at https://smpl.is.tue.mpg.de,
# download SMPL_python_v.1.1.0.zip, and scp it to $SMPL_DIR below. The download is behind a
# login, so it cannot be fetched from the cluster.
#
# Produces, per humenv/data_preparation/README.md step 1:
#   $AMASS_ROOT/models_with_chumpy/SMPL_{NEUTRAL,FEMALE,MALE}.pkl
set -euo pipefail
SMPL_DIR="${SMPL_DIR:-$SCRATCH/humenv-assets/smpl}"
AMASS_ROOT="${AMASS_ROOT:-$SCRATCH/humenv-assets/AMASS}"
ZIP="$SMPL_DIR/SMPL_python_v.1.1.0.zip"

if [ ! -f "$ZIP" ]; then
  echo "MISSING: $ZIP" >&2
  echo "Register at https://smpl.is.tue.mpg.de, download SMPL_python_v.1.1.0.zip, and" >&2
  echo "scp it to $SMPL_DIR/ . Nothing else about this script is manual." >&2
  exit 1
fi

mkdir -p "$AMASS_ROOT/models_with_chumpy"
tmp="$(mktemp -d "$SMPL_DIR/unpack.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
unzip -q "$ZIP" -d "$tmp"

src="$tmp/SMPL_python_v.1.1.0/smpl/models"
[ -d "$src" ] || { echo "unexpected archive layout; looked for $src" >&2; find "$tmp" -name '*.pkl' | head >&2; exit 2; }

cp "$src/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl" "$AMASS_ROOT/models_with_chumpy/SMPL_NEUTRAL.pkl"
cp "$src/basicmodel_f_lbs_10_207_0_v1.1.0.pkl"       "$AMASS_ROOT/models_with_chumpy/SMPL_FEMALE.pkl"
cp "$src/basicmodel_m_lbs_10_207_0_v1.1.0.pkl"       "$AMASS_ROOT/models_with_chumpy/SMPL_MALE.pkl"

echo "SMPL models installed:"
ls -la "$AMASS_ROOT/models_with_chumpy"
echo
echo "Next: the AMASS archives (separate registration, https://amass.is.tue.mpg.de)."
echo "  SMPL-H format, gender based, into $AMASS_ROOT/datasets/"
