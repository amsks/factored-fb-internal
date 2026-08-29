#!/bin/bash
# FULL RUN (2026-08-29): Meta Motivo exact 30M-env-step budget, density-swapped
# FB-CPR on humenv (DensityFBcprAgent), 1 seed, otherwise the metamotivo example
# config verbatim (50 parallel envs, batch 1024, update 50x every 500 steps
# after 50k seed steps, simple model z_dim 256). Real AMASS motions (8471-file
# train list). Model-only checkpoints every 2M steps to work_dir/checkpoint.
#
# RESUME-SUPPORT: full-state checkpoints (model+optimizers+buffers+RNG) every 2M
# steps to work_dir/resume_ckpt, and 10 min before walltime via USR1 (slurm
# signals the batch shell, which forwards to python; python saves, exits 99, and
# this script resubmits itself with --resume, capped by a counter file). A plain
# scancel sends TERM — python also saves on TERM, but the 30s KillWait may
# truncate a large buffer write; the write is atomic (tmp+rename) so the
# previous checkpoint survives. Prefer: scancel --batch --signal=USR1 <jobid>.
#
# CC: environment exports CC=nvc, which breaks triton's runtime build of torch
# 2.13's eager _native kernels. Point CC at gcc.
#   sbatch scripts/fbcpr_density_full.slurm
#SBATCH -J fbcpr-density-full
#SBATCH -A ASC26008
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 47:00:00
#SBATCH --signal=B:USR1@600
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
set -uo pipefail

REPO=/work/11708/amsks/vista/git/Factored-FB
PY=$SCRATCH/.venv-humenv/bin/python
MOTIONS=$SCRATCH/humenv-assets/motions/lists/large1_small1_train_0.1.txt
MOTIONS_ROOT=$SCRATCH/humenv-assets/motions/humenv_amass
WORKDIR=$SCRATCH/factored-fb-humenv/fbcpr-full
MAX_RESUBMITS=8

export PYTHONPATH=$REPO
export MUJOCO_GL=egl
export WANDB_MODE=disabled
export CC=$(command -v gcc)

cd $REPO
# Background + explicit wait: with --signal=B:... slurm signals only this batch
# shell, and bash defers traps while a foreground child runs.
$PY scripts/fbcpr_density_train_humenv.py \
    --seed 0 \
    --motions $MOTIONS \
    --motions_root $MOTIONS_ROOT \
    --num_env_steps 30000000 \
    --checkpoint_every_steps 2000000 \
    --log_every_updates 50000 \
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
echo "FULL_EXIT_CODE=$STATUS"

# Self-resubmit ONLY on the checkpoint-then-exit code (99), guarded by a counter
# file so a broken resume cannot loop forever. Crashes (other nonzero exits) do
# NOT resubmit.
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
                "cd $REPO && sbatch scripts/fbcpr_density_full.slurm"; then
                break
            fi
            echo "sbatch attempt $attempt failed; retrying in 60s"
            sleep 60
        done
    else
        echo "Max resubmits ($MAX_RESUBMITS) reached; not resubmitting"
    fi
fi
