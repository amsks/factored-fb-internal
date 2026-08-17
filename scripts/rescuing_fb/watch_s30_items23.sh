#!/usr/bin/env bash
# Mid-run watcher for the spec-30a Item 2/3 arms (tdi_cd_exact_s0, lsif_cd_phi_s0).
#
# Prints, every WATCH_EVERY seconds, the instruments that tell you whether each arm is
# TRAINING (not whether it is winning -- success is not quoted before 500k, spec 30a):
#
#   TD InfoNCE : tdi_ce_immediate / tdi_ce_bootstrap / tdi_top1 / tdi_w_ess.
#                top1 must lift off the 1/(1+P) = 1/1025 ~ .001 chance line; w_ess falling
#                from ~1025 means the bootstrap label is sharpening. Both CE terms rising
#                without top1 moving = the classifier is not learning.
#   LSIF       : lsif_offdiag / lsif_attract / lsif_m_msq / lsif_diag_mean / lsif_neg_frac.
#                m_msq blowing up is the KNOWN unbounded-linear-attraction failure of the
#                FB/LSIF objective; it is a REPORTABLE property of this control, not
#                something to fix -- adding a scale normalizer would be a third unmatched
#                difference from Density-FB and would destroy the attribution.
#
# It also prints the last eval row (for liveness only) and flags a dead process. It never
# decides anything: no kill gate is defined for these two arms (unlike the P1 screen's
# actor-sensitivity gate) because both are CONTROLS -- a control that trains badly is a
# result, and a control tuned mid-flight is not a control.
#
# Run in tmux:
#   tmux new -d -s s30i23watch 'scripts/rescuing_fb/watch_s30_items23.sh'
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=${OUTROOT:-/home/adityamohan/exp_spec30_i23}/factored-fb
EVERY=${WATCH_EVERY:-1800}

col() {  # col <csv> <header-name>  -> last row's value for that column
  awk -F, -v want="$2" 'NR==1{for(i=1;i<=NF;i++) if($i==want) c=i; next} {v=$c} END{if(c) printf "%s", v; else printf "n/a"}' "$1" 2>/dev/null
}

while true; do
  echo "===== $(date +%F' '%T) ====="
  for tag in tdi_cd_exact_s0 lsif_cd_phi_s0; do
    d=$(compgen -G "$ROOT/*/$tag" | head -1)
    if [ -z "${d:-}" ]; then echo "[$tag] no output dir yet"; continue; fi
    alive=$(pgrep -fc "exp_name=$tag|/$tag" 2>/dev/null || echo 0)
    if [ -f "$d/train.csv" ]; then steps=$(($(wc -l 2>/dev/null < "$d/train.csv") - 1)); else steps=0; fi
    printf '[%s] rows=%s alive=%s\n' "$tag" "$steps" "$alive"
    if [ "$tag" = tdi_cd_exact_s0 ]; then
      printf '    ce_imm=%s ce_boot=%s top1=%s (chance .001) w_ess=%s pos_prob=%s\n' \
        "$(col "$d/train.csv" training/measure/tdi_ce_immediate)" \
        "$(col "$d/train.csv" training/measure/tdi_ce_bootstrap)" \
        "$(col "$d/train.csv" training/measure/tdi_top1)" \
        "$(col "$d/train.csv" training/measure/tdi_w_ess)" \
        "$(col "$d/train.csv" training/measure/tdi_pos_prob)"
      printf '    actor_ce=%s mse=%s\n' \
        "$(col "$d/train.csv" training/actor/tdi_actor_ce)" \
        "$(col "$d/train.csv" training/actor/mse)"
    else
      printf '    offdiag=%s attract=%s m_msq=%s diag_mean=%s neg_frac=%s\n' \
        "$(col "$d/train.csv" training/measure/lsif_offdiag)" \
        "$(col "$d/train.csv" training/measure/lsif_attract)" \
        "$(col "$d/train.csv" training/measure/lsif_m_msq)" \
        "$(col "$d/train.csv" training/measure/lsif_diag_mean)" \
        "$(col "$d/train.csv" training/measure/lsif_neg_frac)"
      printf '    q_mean=%s distill=%s\n' \
        "$(col "$d/train.csv" training/actor/q_mean)" \
        "$(col "$d/train.csv" training/actor/distill_loss)"
    fi
    printf '    [liveness only, NOT a result] last eval step=%s overall=%s\n' \
      "$(col "$d/eval.csv" step)" "$(col "$d/eval.csv" evaluation/overall_success)"
    [ "$alive" = "0" ] && echo "    !! process gone -- check $d and /dev/shm/spec30_i23_logs/$tag.log"
  done
  sleep "$EVERY"
done
