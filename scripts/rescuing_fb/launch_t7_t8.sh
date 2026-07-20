#!/usr/bin/env bash
# T7 (FB conservatism) + T8 (faithful one-step FB) — the two remaining measure-side arms.
# 2026-07-20. Closes the pre-registered claim boundary in
# PAPER/rescuing-fb/specs/04-centered-measure-spec.md §3: with T10-a (0.823, misses its
# 0.85 bar) and T10-b (0.06-0.10, dead band) both in, T7+T8 are what stand between us and
# "no known form of measure-TD grounding supports cube manipulation".
#
# ARMS (3 seeds each, 1M steps, matching the plain-FB anchor's ortho=1000 / lr_b=1e-4):
#   t7_min    fb_pessimism_penalty=0.5  -- with num_parallel=2 this is EXACTLY min-aggregation
#                                          over the twin heads (verified numerically:
#                                          mean - 0.5*|p0-p1| == min for 2 heads). This is the
#                                          canonical FB target the port deviates from by using
#                                          mean (pessimism 0). THE headline T7 test.
#   t7_extra  fb_pessimism_penalty=1.0  -- beyond-min, VC-FB-style extra conservatism. Gives
#                                          T7 a dose-response instead of a single point.
#   t8_zcond  onestep=true onestep_z_cond=true -- SARSA dataset-action bootstrap (a gamma-
#                                          contraction) with F kept z-conditioned, i.e. the
#                                          deep-audit reading of Zheng et al. 2602.11399.
#
# ANCHORS (already measured, no re-run needed):
#   plain FB (mean targets, policy bootstrap) ......... 0.42 @1M
#   OneStepFB z-ZEROED, matched ortho1000/lr_b1e-4 .... 0.058 (10 seeds, bridge_grafts.csv)
#   bilinear GCIQL ..................................... 0.904
# The z-zeroed one-step arm is ALREADY REFUTED at 0.058 -- t8_zcond isolates the single
# disputed bit (z-conditioning of F) against it, which is why we only run the zcond arm.
#
# PRE-REGISTERED READS:
#   t7_min >> 0.42  => the port's mean-aggregation optimism was a real cause; the FB failure
#                      claim needs restating against a conservative FB.
#   t7_min ~ 0.42   => optimism is NOT the bug; closes the VC-FB confound for the paper.
#   t7_extra < t7_min => over-conservatism hurts (expected if 0.5 is already the right dose).
#   t8_zcond >> 0.058 => z-conditioning was the missing piece; one-step FB is a live baseline.
#   t8_zcond ~ 0.058  => one-step FB fails here under BOTH readings => with T10 dead, the
#                      strong negative claim is licensed.
#
# ALWAYS in tmux ([[always-run-in-tmux]]):
#   tmux new -d -s t78 'DRY_RUN=0 bash scripts/rescuing_fb/launch_t7_t8.sh 2>&1 | tee /dev/shm/t78_dispatch.log'
set -uo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=${DRY_RUN:-1}
STEPS=${TRAIN_STEPS:-1000000}
SEEDS=${SEEDS:-0 1 2}

# 8 GPUs free at dispatch. 3 arms x 3 seeds = 9 runs; state-based torch FB is small
# (<5 GB), so GPU 0 carries the one doubled run.
# run_arm <group> <gpus> [VAR=VAL ...] -- arm-specific vars passed explicitly through env(1)
# so they reach the nested `bash state.sh` regardless of shell export semantics.
run_arm() {
    local group="$1" gpus="$2"; shift 2
    echo "=========================================================================="
    echo "[t78] ARM $group  gpus=[$gpus]  seeds=[$SEEDS]  steps=$STEPS  extra=($*)"
    echo "=========================================================================="
    env DRY_RUN="$DRY_RUN" SEEDS="$SEEDS" GPUS="$gpus" TRAIN_STEPS="$STEPS" \
        RUN_GROUP="$group" "$@" bash scripts/agents/fb/state.sh
}

# T7: conservatism dose-response on plain FB.
run_arm fb_state_t7_min   "0 1 2" FB_PESSIMISM=0.5 &
sleep 10
run_arm fb_state_t7_extra "3 4 5" FB_PESSIMISM=1.0 &
sleep 10
# T8: faithful one-step FB.
run_arm fb_state_t8_zcond "6 7 0" MODE=onestep_zcond &

wait
echo "[t78] ALL ARMS DISPATCHED/COMPLETE"
