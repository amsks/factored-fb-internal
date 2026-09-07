#!/usr/bin/env bash
# Launch tier-1 tdjtab runs from scripts/tdjtab_manifest.sh, round-robin over GPUs, inside a
# tmux session (one window per GPU, jobs run sequentially within a window).
#
#   bash scripts/launch_tdjtab.sh '<runid regex>' '<gpu list>'
#   e.g. bash scripts/launch_tdjtab.sh 'antmaze_.*/fb_tdj/' '0 1 2 3'
#        bash scripts/launch_tdjtab.sh 'scene_play/(density_fb|rldp_density)/s[01]' '4 5'
#        DRY=1 bash scripts/launch_tdjtab.sh '.' '0 1 2 3 4 5 6 7'      # print the assignment only
#        SESSION=tdjtab2 bash scripts/launch_tdjtab.sh ...              # a second session
#
# Run ids look like <domain>/<critic>/s<seed> (see the manifest header); the regex is matched
# (grep -E) against the id only. Runs whose final params_1000000.pkl already exists are skipped
# (idempotent; partial runs are picked up by the manifest's --auto_resume). Logs go to
# logs/tdjtab/<runid>.log, i.e. logs/tdjtab/<domain>/<critic>/s<seed>.log.
set -euo pipefail
REPO="${REPO:-$HOME/git/Factored-FB-seam}"
MANIFEST="${MANIFEST:-$REPO/scripts/tdjtab_manifest.sh}"
SESSION="${SESSION:-tdjtab}"
LOGROOT="${LOGROOT:-$REPO/logs/tdjtab}"
FILTER="${1:?usage: launch_tdjtab.sh '<runid regex>' '<gpu list>'}"
read -r -a GPU_LIST <<< "${2:?usage: launch_tdjtab.sh '<runid regex>' '<gpu list>'}"
cd "$REPO"

# id<TAB>command, for every manifest line whose id matches the filter.
SEL=$(grep -E '^# [a-z0-9_]+/[a-z0-9_]+/s[0-9] :: ' "$MANIFEST" \
      | sed -E 's/^# ([^ ]+) :: (.*)$/\1\t\2/' | sed -E 's/[[:space:]]+## .*$//' \
      | awk -F'\t' -v re="$FILTER" '$1 ~ re')
# Drop runs whose final checkpoint exists. run dir = <save_dir or exp>/<env_name>_<critic>_flowbc_seed_<s>.
JOBS=""
while IFS=$'\t' read -r id cmd; do
  [ -n "${id:-}" ] || continue
  dom=${id%%/*}; rest=${id#*/}; critic=${rest%%/*}; seed=${id##*/s}
  envn=$(grep -E '^env_name:' "impls/configs/domain/$dom.yaml" | awk '{print $2}')
  sd=$(printf '%s' "$cmd" | grep -oE -- '--save_dir [^ ]+' | awk '{print $2}' || true); sd=${sd:-exp}
  if [ -f "$sd/${envn}_${critic}_flowbc_seed_${seed}/params_1000000.pkl" ]; then
    echo "  skip (done): $id"; continue
  fi
  JOBS+="$id"$'\t'"$cmd"$'\n'
done <<< "$SEL"
N=$(printf '%s' "$JOBS" | grep -c . || true)
echo "$N jobs match /$FILTER/ on GPUs: ${GPU_LIST[*]}"
[ "$N" -gt 0 ] || { echo "nothing to run"; exit 0; }
NG=${#GPU_LIST[@]}
for w in "${!GPU_LIST[@]}"; do
  printf '%s' "$JOBS" | awk -F'\t' -v w="$w" -v n="$NG" -v g="${GPU_LIST[$w]}" 'NF && (NR-1)%n==w {print "  GPU " g " <- " $1}'
done
[ "${DRY:-0}" = "1" ] && exit 0
[ -x /dev/shm/.venv-jax/bin/python ] || { echo "missing /dev/shm/.venv-jax/bin/python (volatile venv; recreate)"; exit 1; }
if tmux has-session -t "=${SESSION}" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists; new windows are added to it"; first=0
else first=1; fi
for w in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$w]}"
  slice=$(printf '%s' "$JOBS" | awk -F'\t' -v w="$w" -v n="$NG" 'NF && (NR-1)%n==w')
  [ -n "$slice" ] || continue
  # One worker script per window: it iterates its slice, exporting GPU for the manifest command.
  wf="$LOGROOT/.worker_gpu${gpu}_$(date +%s).sh"; mkdir -p "$LOGROOT"
  {
    echo "#!/usr/bin/env bash"; echo "cd $REPO"; echo "export GPU=$gpu"
    while IFS=$'\t' read -r id cmd; do
      [ -n "${id:-}" ] || continue
      log="$LOGROOT/$id.log"
      printf 'mkdir -p %q\n' "$(dirname "$log")"
      printf 'echo ">>> %s  [GPU %s] $(date)"\n' "$id" "$gpu"
      printf '%s 2>&1 | tee -a %q\n' "$cmd" "$log"
    done <<< "$slice"
    echo "echo ALL_DONE_GPU_$gpu; sleep 86400"
  } > "$wf"; chmod +x "$wf"
  if [ "$first" -eq 1 ]; then tmux new-session -d -s "$SESSION" -n "gpu$gpu" "bash $wf"; first=0
  else tmux new-window -d -t "=${SESSION}" -n "gpu$gpu" "bash $wf"; fi
done
echo "launched in tmux session '$SESSION'; logs under $LOGROOT/<domain>/<critic>/s<seed>.log"
