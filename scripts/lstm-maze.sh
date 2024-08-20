#!/bin/bash
#SBATCH --job-name=gmg-maze
#SBATCH --partition=high_priority
#SBATCH --time=10:00:00
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/data/matthew_farrugia_roberts
#SBATCH --output=out/%A-%a-gmg-lstm.stdout
#SBATCH --error=out/%A-%a-gmg-lstm.stderr
#SBATCH --array 0-5

source jaxgmg.venv/bin/activate
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    jaxgmg train minimaze \
        --wandb-log --no-console-log --wandb-name ff-dr \
        --wandb-entity matthew-farrugia-roberts --wandb-project lstm-gmg-2 \
        --num-total-env-steps 100_000_000 \
        --no-env-penalize-time --ued dr \
        --net-cnn-type large --net-rnn-type ff;
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    jaxgmg train minimaze \
        --wandb-log --no-console-log --wandb-name ff-plr \
        --wandb-entity matthew-farrugia-roberts --wandb-project lstm-gmg-2 \
        --num-total-env-steps 100_000_000 \
        --no-env-penalize-time --ued plr --plr-regret-estimator PVL \
        --net-cnn-type large --net-rnn-type ff;
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    jaxgmg train minimaze \
        --wandb-log --no-console-log --wandb-name lstm-dr \
        --wandb-entity matthew-farrugia-roberts --wandb-project lstm-gmg-2 \
        --num-total-env-steps 100_000_000 \
        --no-env-penalize-time --ued dr \
        --net-cnn-type large --net-rnn-type lstm;
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    jaxgmg train minimaze \
        --wandb-log --no-console-log --wandb-name lstm-plr \
        --wandb-entity matthew-farrugia-roberts --wandb-project lstm-gmg-2 \
        --num-total-env-steps 100_000_000 \
        --no-env-penalize-time --ued plr --plr-regret-estimator PVL \
        --net-cnn-type large --net-rnn-type lstm;
elif [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    jaxgmg train minimaze \
        --wandb-log --no-console-log --wandb-name gru-dr \
        --wandb-entity matthew-farrugia-roberts --wandb-project lstm-gmg-2 \
        --num-total-env-steps 100_000_000 \
        --no-env-penalize-time --ued dr \
        --net-cnn-type large --net-rnn-type gru;
elif [ $SLURM_ARRAY_TASK_ID -eq 5 ]; then
    jaxgmg train minimaze \
        --wandb-log --no-console-log --wandb-name gru-plr \
        --wandb-entity matthew-farrugia-roberts --wandb-project lstm-gmg-2 \
        --num-total-env-steps 100_000_000 \
        --no-env-penalize-time --ued plr --plr-regret-estimator PVL \
        --net-cnn-type large --net-rnn-type gru;
fi
deactivate
