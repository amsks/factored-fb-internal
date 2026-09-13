#!/usr/bin/env bash
# rldp + rldp_density at ortho_coef=1000 (campaign: 0.01), manipulation envs x 3 seeds, tdjtab protocol.
# HILP has no orthonormality regularizer, so hilp_density is not part of this arm.
# Commands are the manifest lines with save_dir -> exp/ortho1000 and the ortho override appended.
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
GPUS=(0 1 4 5 6 7); SESSION=ortho1000
SEL=$(grep -E '^# (cube_single|cube_double|scene_play|puzzle_3x3)/(rldp|rldp_density)/s[0-2] :: ' scripts/tdjtab_manifest.sh \
      | sed -E 's/^# ([^ ]+) :: (.*)$/\1\t\2/' | sed -E 's/[[:space:]]+## .*$//' | sort -t/ -k3,3 -k1,1)
declare -a Q; for g in "${GPUS[@]}"; do Q+=(""); done
i=0
while IFS=$'\t' read -r id cmd; do
  [ -n "${id:-}" ] || continue
  cmd=$(printf '%s' "$cmd" | sed -E 's/ --save_dir [^ ]+//')
  cmd="$cmd --save_dir exp/ortho1000 --override ortho_coef=1000"
  gi=$((i % ${#GPUS[@]})); gpu=${GPUS[$gi]}; log="logs/ortho1000/$(printf '%s' "$id" | tr '/' '_').log"
  Q[$gi]+="GPU=$gpu; $cmd > $log 2>&1; "
  i=$((i+1))
done <<< "$SEL"
echo "jobs: $i"
tmux new-session -d -s $SESSION -n init
for gi in "${!GPUS[@]}"; do
  tmux new-window -t $SESSION -n "gpu${GPUS[$gi]}" "cd $HOME/git/Factored-FB-seam; ${Q[$gi]} echo DONE; sleep 999999"
done
tmux kill-window -t $SESSION:init || true
