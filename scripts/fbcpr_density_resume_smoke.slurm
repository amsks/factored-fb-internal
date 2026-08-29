#!/bin/bash
# RESUME SMOKE (2026-08-29): verifies the checkpoint/resume path of
# scripts/fbcpr_density_train_humenv.py. 60k env steps, full-state checkpoint
# every 5000 steps. Kill mid-run with
#   scancel --batch --signal=USR1 <jobid>
# -> python writes work_dir/resume_ckpt/ckpt_*.pt, exits 99, and this script
# resubmits itself; the resumed job must continue from the saved step (check
# step counter, buffer len, and that density_ce does not reset to its
# from-scratch value ~6.9).
# Uses the SAME signal-forward + self-resubmit tail as the production launcher.
#   sbatch scripts/fbcpr_density_resume_smoke.slurm
#SBATCH -J fbcpr-resume-smoke
#SBATCH -A ASC26008
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH --signal=B:USR1@600
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
set -uo pipefail

REPO=/work/11708/amsks/vista/git/Factored-FB
PY=$SCRATCH/.venv-humenv/bin/python
MOTIONS=$SCRATCH/humenv-assets/motions/lists/large1_small1_train_0.1.txt
MOTIONS_ROOT=$SCRATCH/humenv-assets/motions/humenv_amass
WORKDIR=$SCRATCH/runs/fbcpr_density_resume_smoke
MAX_RESUBMITS=3

export PYTHONPATH=$REPO
export MUJOCO_GL=egl
export WANDB_MODE=disabled
export CC=$(command -v gcc)

cd $REPO
# Run python in the background and forward USR1/TERM: with --signal=B:... slurm
# signals only this batch shell, and bash defers traps while a foreground child
# runs, so the trainer must be a background child of an explicit wait.
$PY scripts/fbcpr_density_train_humenv.py \
    --motions $MOTIONS \
    --motions_root $MOTIONS_ROOT \
    --online_parallel_envs 50 \
    --num_env_steps 60000 \
    --num_seed_steps 1000 \
    --update_agent_every 500 \
    --num_agent_updates 50 \
    --buffer_size 100000 \
    --log_every_updates 1000 \
    --checkpoint_every_steps 5000 \
    --work_dir $WORKDIR \
    --resume \
    --device cuda &
PID=$!
trap 'kill -USR1 $PID 2>/dev/null' USR1
trap 'kill -TERM $PID 2>/dev/null' TERM
wait $PID
STATUS=$?
while kill -0 $PID 2>/dev/null; do  # first wait returns early when a trap fires
    wait $PID
    STATUS=$?
done
echo "SMOKE_EXIT_CODE=$STATUS"

# Self-resubmit ONLY on the checkpoint-then-exit code (99), guarded by a counter
# file so a broken resume cannot loop forever.
if [ "$STATUS" -eq 99 ]; then
    COUNT_FILE=$WORKDIR/resume_ckpt/resubmit_count
    N=$(cat "$COUNT_FILE" 2>/dev/null || echo 0)
    if [ "$N" -lt "$MAX_RESUBMITS" ]; then
        echo $((N + 1)) > "$COUNT_FILE"
        echo "Resubmitting (attempt $((N + 1))/$MAX_RESUBMITS)"
        # TACC blocks sbatch on compute nodes; submit via the login node instead.
        # Retry: the dying job still counts toward QOSMaxSubmitJobPerUserLimit
        # until it finishes COMPLETING, so the first attempt can be rejected.
        for attempt in 1 2 3 4 5; do
            if ssh -o StrictHostKeyChecking=no "$SLURM_SUBMIT_HOST" \
                "cd $REPO && sbatch scripts/fbcpr_density_resume_smoke.slurm"; then
                break
            fi
            echo "sbatch attempt $attempt failed; retrying in 60s"
            sleep 60
        done
    else
        echo "Max resubmits ($MAX_RESUBMITS) reached; not resubmitting"
    fi
fi
