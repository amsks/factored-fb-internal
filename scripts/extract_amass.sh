#!/bin/bash
# Extract the AMASS archives that have FINISHED transferring, into the layout
# data_preparation/README.md expects: $AMASS_ROOT/datasets/<DatasetName>/<subject>/*.npz
#
#   bash scripts/extract_amass.sh          # extract whatever is complete, skip the rest
#
# Safe to re-run while `rsync` is still copying: rsync writes in-flight files as
# `.NAME.tar.bz2.XXXXXX` (leading dot), and this only considers names matching *.tar.bz2 /
# *.zip. Each archive is INTEGRITY-CHECKED before extraction, so a file that finished
# copying but arrived corrupt is reported rather than half-extracted, and an archive whose
# target directory already exists is skipped.
set -uo pipefail
A="${AMASS_ROOT:-$SCRATCH/humenv-assets/AMASS}/datasets"
cd "$A" || { echo "no such dir: $A" >&2; exit 2; }
shopt -s nullglob
found=0
for f in *.tar.bz2 *.zip; do
  found=1
  name="${f%%.*}"                       # ACCAD.tar.bz2 -> ACCAD
  if [ -d "$name" ]; then echo "SKIP(extracted) $f -> $name/"; continue; fi
  case "$f" in
    *.tar.bz2)
      if ! bzip2 -t "$f" 2>/dev/null; then echo "SKIP(incomplete/corrupt) $f"; continue; fi
      n=$(tar -tjf "$f" | wc -l)
      tar -xjf "$f" && echo "OK $f -> $name/ ($n entries)" || echo "FAIL extracting $f" ;;
    *.zip)
      if ! unzip -tq "$f" >/dev/null 2>&1; then echo "SKIP(incomplete/corrupt) $f"; continue; fi
      unzip -q "$f" && echo "OK $f -> $name/" || echo "FAIL extracting $f" ;;
  esac
done
[ "$found" = 1 ] || echo "no archives in $A yet"
echo "--- present now:"
ls -d */ 2>/dev/null | tr -d '/' | tr '\n' ' '; echo
echo "--- still transferring (rsync temp files):"
ls -a | grep -E '^\.[A-Za-z]' | tr '\n' ' '; echo
