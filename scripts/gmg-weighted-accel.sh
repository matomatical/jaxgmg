#!/bin/bash
#SBATCH --job-name=gmg-wts
#SBATCH --partition=high_priority
#SBATCH --time=8:00:00
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/data/matthew_farrugia_roberts
#SBATCH --output=out/%A_%a-gmg-wts.stdout
#SBATCH --error=out/%A_%a-gmg-wts.stderr
#SBATCH --array 0-6

source jaxgmg.venv/bin/activate
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name accel-mut00 \
        --wandb-entity matthew-farrugia-roberts \
        --wandb-project gmg-wts \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.00 --num-mutate-steps 6 \
        --ued accel --prob-mutate-cheese 0.00;
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name accel-mut01 \
        --wandb-entity matthew-farrugia-roberts \
        --wandb-project gmg-accel-weighted \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.00 --num-mutate-steps 6 \
        --ued accel --prob-mutate-cheese 0.01;
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name accel-mut03 \
        --wandb-entity matthew-farrugia-roberts \
        --wandb-project gmg-accel-weighted \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.00 --num-mutate-steps 6 \
        --ued accel --prob-mutate-cheese 0.03;
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name accel-mut10 \
        --wandb-entity matthew-farrugia-roberts \
        --wandb-project gmg-accel-weighted \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.00 --num-mutate-steps 6 \
        --ued accel --prob-mutate-cheese 0.10;
elif [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name accel-mut30 \
        --wandb-entity matthew-farrugia-roberts \
        --wandb-project gmg-accel-weighted \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.00 --num-mutate-steps 6 \
        --ued accel --prob-mutate-cheese 0.30;
elif [ $SLURM_ARRAY_TASK_ID -eq 5 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name accel-mut50 \
        --wandb-entity matthew-farrugia-roberts \
        --wandb-project gmg-accel-weighted \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.00 --num-mutate-steps 6 \
        --ued accel --prob-mutate-cheese 0.50;
elif [ $SLURM_ARRAY_TASK_ID -eq 6 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name accel-mut70 \
        --wandb-entity matthew-farrugia-roberts \
        --wandb-project gmg-accel-weighted \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.00 --num-mutate-steps 6 \
        --ued accel --prob-mutate-cheese 0.70;
fi
deactivate
