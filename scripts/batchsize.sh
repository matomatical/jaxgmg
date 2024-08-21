#!/bin/bash
#SBATCH --job-name=batchsize
#SBATCH --partition=high_priority
#SBATCH --time=10:00:00
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/data/matthew_farrugia_roberts
#SBATCH --output=out/%A-%a-batchsize.stdout
#SBATCH --error=out/%A-%a-batchsize.stderr
#SBATCH --array 0-3

source jaxgmg.venv/bin/activate
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name mb4-env256-ff \
        --wandb-entity matthew-farrugia-roberts --wandb-project batchsize \
        --num-total-env-steps 20_000_000 --ued dr \
        --net-cnn-type large --net-rnn-type ff \
        --num-minibatches-per-epoch 4 --num-parallel-envs 256;
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name mb4-env256-lstm \
        --wandb-entity matthew-farrugia-roberts --wandb-project batchsize \
        --num-total-env-steps 20_000_000 --ued dr \
        --net-cnn-type large --net-rnn-type lstm \
        --num-minibatches-per-epoch 4 --num-parallel-envs 256;
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name mb4-env512-ff \
        --wandb-entity matthew-farrugia-roberts --wandb-project batchsize \
        --num-total-env-steps 20_000_000 --ued dr \
        --net-cnn-type large --net-rnn-type ff \
        --num-minibatches-per-epoch 4 --num-parallel-envs 512;
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name mb4-env512-lstm \
        --wandb-entity matthew-farrugia-roberts --wandb-project batchsize \
        --num-total-env-steps 20_000_000 --ued dr \
        --net-cnn-type large --net-rnn-type lstm \
        --num-minibatches-per-epoch 4 --num-parallel-envs 512;
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name mb4-env1024-ff \
        --wandb-entity matthew-farrugia-roberts --wandb-project batchsize \
        --num-total-env-steps 20_000_000 --ued dr \
        --net-cnn-type large --net-rnn-type ff \
        --num-minibatches-per-epoch 4 --num-parallel-envs 1024;
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name mb4-env1024-lstm \
        --wandb-entity matthew-farrugia-roberts --wandb-project batchsize \
        --num-total-env-steps 20_000_000 --ued dr \
        --net-cnn-type large --net-rnn-type lstm \
        --num-minibatches-per-epoch 4 --num-parallel-envs 1024;
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name mb8-env1024-ff \
        --wandb-entity matthew-farrugia-roberts --wandb-project batchsize \
        --num-total-env-steps 20_000_000 --ued dr \
        --net-cnn-type large --net-rnn-type ff \
        --num-minibatches-per-epoch 8 --num-parallel-envs 1024;
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name mb8-env1024-lstm \
        --wandb-entity matthew-farrugia-roberts --wandb-project batchsize \
        --num-total-env-steps 20_000_000 --ued dr \
        --net-cnn-type large --net-rnn-type lstm \
        --num-minibatches-per-epoch 8 --num-parallel-envs 1024;
fi
deactivate
