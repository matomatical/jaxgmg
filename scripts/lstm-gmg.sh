#!/bin/bash
#SBATCH --job-name=gmg-lstm
#SBATCH --partition=high_priority
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/data/matthew_farrugia_roberts
#SBATCH --output=out/%A-%a-gmg-lstm.stdout
#SBATCH --error=out/%A-%a-gmg-lstm.stderr
#SBATCH --array 0,1,2,3%4

source jaxgmg.venv/bin/activate
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log \
        --wandb-entity matthew-farrugia-roberts --wandb-project lstm-gmg-1 \
        --num-total-env-steps 20_000_000 \
        --no-env-penalize-time --ued dr \
        --net-cnn-type large --net-rnn-type lstm;
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log \
        --wandb-entity matthew-farrugia-roberts --wandb-project lstm-gmg-1 \
        --num-total-env-steps 20_000_000 \
        --no-env-penalize-time --ued dr \
        --net-cnn-type large --net-rnn-type ff;
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log \
        --wandb-entity matthew-farrugia-roberts --wandb-project lstm-gmg-1 \
        --num-total-env-steps 20_000_000 \
        --no-env-penalize-time --ued plr --plr-regret-estimator PVL \
        --net-cnn-type large --net-rnn-type lstm;
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log \
        --wandb-entity matthew-farrugia-roberts --wandb-project lstm-gmg-1 \
        --num-total-env-steps 20_000_000 \
        --no-env-penalize-time --ued plr --plr-regret-estimator PVL \
        --net-cnn-type large --net-rnn-type ff;
fi
deactivate
