#!/bin/bash
#SBATCH --job-name=gmg-mix
#SBATCH --partition=high_priority
#SBATCH --time=8:00:00
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/data/matthew_farrugia_roberts
#SBATCH --output=out/%A-%a-gmg-mix.stdout
#SBATCH --error=out/%A-%a-gmg-mix.stderr
#SBATCH --array 0-9

source jaxgmg.venv/bin/activate
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name dr-mix01 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.01 --ued dr;
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name plr-mix01 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.01 --ued plr;
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name dr-mix03 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.03 --ued dr;
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name plr-mix03 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.03 --ued plr;
elif [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name dr-mix10 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.10 --ued dr;
elif [ $SLURM_ARRAY_TASK_ID -eq 5 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name plr-mix10 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.10 --ued plr;
elif [ $SLURM_ARRAY_TASK_ID -eq 6 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name dr-mix30 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.30 --ued dr;
elif [ $SLURM_ARRAY_TASK_ID -eq 7 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name plr-mix30 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.30 --ued plr;
elif [ $SLURM_ARRAY_TASK_ID -eq 8 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name dr-mix0.3 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.003 --ued dr;
elif [ $SLURM_ARRAY_TASK_ID -eq 9 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name plr-mix0.3 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.003 --ued plr;
fi
deactivate
