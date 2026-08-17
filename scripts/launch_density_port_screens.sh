#!/bin/bash
# Dev screens for docs/specs/2026-08-16-density-fb-bridge-port.md Part A.
#
#   DRYRUN=1 bash scripts/launch_density_port_screens.sh      # print, submit nothing
#   bash scripts/launch_density_port_screens.sh
#
# ONE cell only: mw-pick-place x play-resample v3 is the DEV cell (HPO spec's dev/holdout
# split). Holdout — pnh-1m, prh, ManiSkill, mw-push — is touched exactly once later, with the
# frozen config. Screens are 1 seed: they order arms, they do not measure them.
#
# GRADUATION GATE (collapse spec §4): a 500k final >= 0.55 with no >0.25 drop from that arm's
# own peak. That is why EVAL_INTERVAL=50000 — the collapse IS the observable, so it has to be
# resolved, not sampled at the endpoint. Anything that merely raises the peak without
# flattening the decay has NOT cleared the gate.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

ALLOC="${ALLOC:-ASC26008}"
CODE_DIR="${CODE_DIR:-$PWD}"
VENV="${VENV:-$SCRATCH/.venv-jax-fb}"
SEED="${SEED:-0}"
DRYRUN="${DRYRUN:-0}"
DOM=mw-pick-place
export FB_DATA_ROOT="${FB_DATA_ROOT:-$SCRATCH/fb-data/play-resample}"
MANIFEST="logs/density-port-screens-$(date +%Y%m%d-%H%M%S).txt"

# The repaired protocol every density cell runs under (deconfound spec E3): goal channel
# masked out of BOTH the observation and the emitted goal, no random-goal mass, per-episode
# synthesised eval goals. `p_randomgoal=0` must move its mass to trajgoal or GCDataset's
# sum-to-1 assertion fires at startup.
E3P="obs_mask_dims=[36,37,38] goal_mask_dims=[36,37,38] value_p_randomgoal=0.0 value_p_trajgoal=0.8 eval_goal_mode=synth"
# measure_density_hard is now the config DEFAULT (spec A2) but stays explicit here: the
# override string is what gets pasted into the frozen-config record, and a table cell should
# never depend on a default that could move.
# The hit spec MUST carry literal quotes into the override string. Unquoted,
# `measure_hit_spec=4:7:0.07` is read by OmegaConf as a sexagesimal literal and becomes
# 14820.07 (= 4*3600 + 7*60 + 0.07). Multi-block specs survive unquoted (the comma breaks the
# pattern), so this only bites the single-block case — which is every Metaworld cell.
BASE="measure_density_hard=true measure_hit_spec=\"4:7:0.07\""

submit() {  # submit <tag> <actor> <extra overrides...>
  local tag=$1 actor=$2; shift 2
  local name="DP-$tag-s$SEED"
  local sd="$SCRATCH/factored-fb-density-port/$tag"
  local run="$sd/${DOM}_density_fb_${actor}_seed_${SEED}"
  if [ -f "$run/eval.csv" ] && tail -1 "$run/eval.csv" | tr -d '\0\r' | grep -q ",500000$"; then
    echo "SKIP(done)   $name" | tee -a "$MANIFEST"; return
  fi
  if squeue -h -u "$USER" -n "$name" 2>/dev/null | grep -q .; then
    echo "SKIP(queued) $name" | tee -a "$MANIFEST"; return
  fi
  local ov="$BASE $E3P $*"
  if [ "$DRYRUN" = "1" ]; then
    echo "DRYRUN $name actor=$actor" | tee -a "$MANIFEST"
    echo "        OVERRIDE='$ov'" | tee -a "$MANIFEST"
    echo "        SAVE_DIR=$sd" | tee -a "$MANIFEST"
    return
  fi
  local id
  # Never --dependency-chain: SLURM purges completed jobs and the dependent submission dies.
  until id=$(CRITIC=density_fb ACTOR=$actor DOMAIN=$DOM SEED=$SEED OVERRIDE="$ov" \
             TRAIN_STEPS=500000 EVAL_INTERVAL=50000 SAVE_INTERVAL=100000 \
             EVAL_EPISODES=50 NUM_VIDEO=2 \
             CODE_DIR=$CODE_DIR VENV=$VENV SAVE_DIR="$sd" \
             sbatch -A "$ALLOC" -J "$name" --parsable scripts/vista_train.slurm 2>/dev/null \
             | grep -oE '^[0-9]+' | tail -1) && [ -n "$id" ]; do
    echo "submit refused, waiting 300s ($name)"; sleep 300
  done
  echo "$id $name actor=$actor ov='$ov'" | tee -a "$MANIFEST"
}

# A3 — the ddpgbc extraction (bridge spec 22). Swaps BOTH ends of R1's loop: the actor no
# longer ascends the kernel readout with a flow policy, and the critic target bootstraps at
# that gaussian policy's mode. actor/ddpgbc_density.yaml already carries lr_actor=3.0e-4, so
# this arm bundles A1.
submit a3-ddpgbc ddpgbc_density

# A1 alone on the flow path, for ordering against A3. The S0 screen already ran this exact
# arm (0.367 final on pr pick-place, tag densityhard-lra3e4) — skipped by default rather than
# respent. Set SCREEN_A1=1 to re-run it under the current code (A2 default + A4 logging).
[ "${SCREEN_A1:-0}" = "1" ] && submit a1-flow-lra3e4 flowbc lr_actor=3.0e-4

# ---- Collapse-spec kill-tests (docs/specs/2026-08-16-density-fb-pickplace-collapse.md §4),
# flow actor. Single-change arms preserve the R1/R2/R3 discrimination; the combo arm is the
# graduation candidate (both known helpers at once). actor.q_coeff / actor.bc_coeff are real
# nested keys (read via a_cfg in flowbc.build) — verified, unlike the inert actor.lr.
submit kt1-qc03      flowbc actor.q_coeff=0.3                       # KT-1: weaken actor ascent
submit kt1-qc01      flowbc actor.q_coeff=0.1                       # KT-1
submit kt2-tau001    flowbc tau=0.001                               # KT-2: slow target loop
submit kt2-bc10      flowbc actor.bc_coeff=10.0                     # KT-2: anchor to support
submit kt4-temp2     flowbc measure_density_temp=2.0                # KT-4: smooth readout (watch ESS)
submit kt4-dim128    flowbc measure_density_dim=128                 # KT-4
submit kt3-g098      flowbc discount=0.98                           # KT-3: MECHANISM-SIDE ONLY, never a table cell
submit kt5-split     flowbc measure_hit_spec=\"4:6:0.07,6:7:0.14\"  # KT-5: overrides BASE's spec (last wins)
submit combo-qc03-lr flowbc lr_actor=3.0e-4 actor.q_coeff=0.3       # graduation candidate

# ---- 2026-08-17 audit additions (HP-parity agent): the bridge's own goal-reaching table
# with OUR actor (FQL+BC flow) favors SOFT over hard (cube 0.929 vs 0.915 etc.) — "ship
# hard" was carried by zero-shot/ddpgbc rows. Soft has never run on MW. bc03 completes the
# H3 cross downward and doubles as an R1 kill-test (R1 predicts it worsens the collapse).
submit kt6-soft      flowbc measure_density_hard=false
submit kt7-bc03      flowbc actor.bc_coeff=0.3
# 2026-08-17 round 3: soft graduated (0.900/0.800) alongside qc03 (3-seed 0.900/0.560/0.820,
# seed 1 decays) — the two graduating mechanisms are independent (target smoothing vs ascent
# weakening), so screen their combination as the strongest freeze candidate.
submit combo-qc03-soft flowbc actor.q_coeff=0.3 measure_density_hard=false

echo "== screens submitted; manifest $MANIFEST =="
echo "== gate: 500k final >= 0.55 AND no >0.25 drop from that arm's own peak =="
echo "== read with: bash scripts/density_screen_report.sh =="
