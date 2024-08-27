#!/bin/bash
#SBATCH --job-name=lstm-width
#SBATCH --partition=high_priority
#SBATCH --time=36:00:00
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/data/matthew_farrugia_roberts
#SBATCH --output=out/%A-%a-lstm-width.stdout
#SBATCH --error=out/%A-%a-lstm-width.stderr
#SBATCH --array 0-2

source jaxgmg.venv/bin/activate
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name ffn0064-dr \
        --wandb-entity matthew-farrugia-roberts --wandb-project lstm-width \
        --num-total-env-steps 500_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type ff --net-width 64 \
        --prob-shift 0.1 --ued dr;
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name rnn0064-dr \
        --wandb-entity matthew-farrugia-roberts --wandb-project lstm-width \
        --num-total-env-steps 500_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm --net-width 64 \
        --prob-shift 0.1 --ued dr;
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name ffn0512-dr \
        --wandb-entity matthew-farrugia-roberts --wandb-project lstm-width \
        --num-total-env-steps 500_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type ff --net-width 512 \
        --prob-shift 0.1 --ued dr;
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name rnn0512-dr \
        --wandb-entity matthew-farrugia-roberts --wandb-project lstm-width \
        --num-total-env-steps 500_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm --net-width 512 \
        --prob-shift 0.1 --ued dr;
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name ffn1024-dr \
        --wandb-entity matthew-farrugia-roberts --wandb-project lstm-width \
        --num-total-env-steps 500_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type ff --net-width 1024 \
        --prob-shift 0.1 --ued dr;
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name rnn1024-dr \
        --wandb-entity matthew-farrugia-roberts --wandb-project lstm-width \
        --num-total-env-steps 500_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm --net-width 1024 \
        --prob-shift 0.1 --ued dr;
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name ffn0064-plr \
        --wandb-entity matthew-farrugia-roberts --wandb-project lstm-width \
        --num-total-env-steps 500_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type ff --net-width 64 \
        --prob-shift 0.1 --ued plr;
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name rnn0064-plr \
        --wandb-entity matthew-farrugia-roberts --wandb-project lstm-width \
        --num-total-env-steps 500_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm --net-width 64 \
        --prob-shift 0.1 --ued plr;
elif [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name ffn0512-plr \
        --wandb-entity matthew-farrugia-roberts --wandb-project lstm-width \
        --num-total-env-steps 500_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type ff --net-width 512 \
        --prob-shift 0.1 --ued plr;
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name rnn0512-plr \
        --wandb-entity matthew-farrugia-roberts --wandb-project lstm-width \
        --num-total-env-steps 500_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm --net-width 512 \
        --prob-shift 0.1 --ued plr;
elif [ $SLURM_ARRAY_TASK_ID -eq 5 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name ffn1024-plr \
        --wandb-entity matthew-farrugia-roberts --wandb-project lstm-width \
        --num-total-env-steps 500_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type ff --net-width 1024 \
        --prob-shift 0.1 --ued plr;
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name rnn1024-plr \
        --wandb-entity matthew-farrugia-roberts --wandb-project lstm-width \
        --num-total-env-steps 500_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm --net-width 1024 \
        --prob-shift 0.1 --ued plr;
fi
deactivate
