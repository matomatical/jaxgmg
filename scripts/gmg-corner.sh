#!/bin/bash
#SBATCH --job-name=gmg-corner
#SBATCH --partition=high_priority
#SBATCH --time=8:00:00
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/data/matthew_farrugia_roberts
#SBATCH --output=out/%A-%a-gmg-corner.stdout
#SBATCH --error=out/%A-%a-gmg-corner.stderr
#SBATCH --array 0-19

source jaxgmg.venv/bin/activate
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name dr-cs01 \
        --wandb-entity matthew-farrugia-roberts --wandb-project corner-gmg \
        --num-total-env-steps 200_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm \
        --env-corner-size 1 --ued dr;
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name plr-cs01 \
        --wandb-entity matthew-farrugia-roberts --wandb-project corner-gmg \
        --num-total-env-steps 200_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm \
        --env-corner-size 1 --ued plr;
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name dr-cs02 \
        --wandb-entity matthew-farrugia-roberts --wandb-project corner-gmg \
        --num-total-env-steps 200_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm \
        --env-corner-size 2 --ued dr;
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name plr-cs02 \
        --wandb-entity matthew-farrugia-roberts --wandb-project corner-gmg \
        --num-total-env-steps 200_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm \
        --env-corner-size 2 --ued plr;
elif [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name dr-cs03 \
        --wandb-entity matthew-farrugia-roberts --wandb-project corner-gmg \
        --num-total-env-steps 200_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm \
        --env-corner-size 3 --ued dr;
elif [ $SLURM_ARRAY_TASK_ID -eq 5 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name plr-cs03 \
        --wandb-entity matthew-farrugia-roberts --wandb-project corner-gmg \
        --num-total-env-steps 200_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm \
        --env-corner-size 3 --ued plr;
elif [ $SLURM_ARRAY_TASK_ID -eq 6 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name dr-cs04 \
        --wandb-entity matthew-farrugia-roberts --wandb-project corner-gmg \
        --num-total-env-steps 200_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm \
        --env-corner-size 4 --ued dr;
elif [ $SLURM_ARRAY_TASK_ID -eq 7 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name plr-cs04 \
        --wandb-entity matthew-farrugia-roberts --wandb-project corner-gmg \
        --num-total-env-steps 200_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm \
        --env-corner-size 4 --ued plr;
elif [ $SLURM_ARRAY_TASK_ID -eq 8 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name dr-cs05 \
        --wandb-entity matthew-farrugia-roberts --wandb-project corner-gmg \
        --num-total-env-steps 200_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm \
        --env-corner-size 5 --ued dr;
elif [ $SLURM_ARRAY_TASK_ID -eq 9 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name plr-cs05 \
        --wandb-entity matthew-farrugia-roberts --wandb-project corner-gmg \
        --num-total-env-steps 200_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm \
        --env-corner-size 5 --ued plr;
elif [ $SLURM_ARRAY_TASK_ID -eq 10 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name dr-cs06 \
        --wandb-entity matthew-farrugia-roberts --wandb-project corner-gmg \
        --num-total-env-steps 200_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm \
        --env-corner-size 6 --ued dr;
elif [ $SLURM_ARRAY_TASK_ID -eq 11 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name plr-cs06 \
        --wandb-entity matthew-farrugia-roberts --wandb-project corner-gmg \
        --num-total-env-steps 200_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm \
        --env-corner-size 6 --ued plr;
elif [ $SLURM_ARRAY_TASK_ID -eq 12 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name dr-cs07 \
        --wandb-entity matthew-farrugia-roberts --wandb-project corner-gmg \
        --num-total-env-steps 200_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm \
        --env-corner-size 7 --ued dr;
elif [ $SLURM_ARRAY_TASK_ID -eq 13 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name plr-cs07 \
        --wandb-entity matthew-farrugia-roberts --wandb-project corner-gmg \
        --num-total-env-steps 200_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm \
        --env-corner-size 7 --ued plr;
elif [ $SLURM_ARRAY_TASK_ID -eq 14 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name dr-cs08 \
        --wandb-entity matthew-farrugia-roberts --wandb-project corner-gmg \
        --num-total-env-steps 200_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm \
        --env-corner-size 8 --ued dr;
elif [ $SLURM_ARRAY_TASK_ID -eq 15 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name plr-cs08 \
        --wandb-entity matthew-farrugia-roberts --wandb-project corner-gmg \
        --num-total-env-steps 200_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm \
        --env-corner-size 8 --ued plr;
elif [ $SLURM_ARRAY_TASK_ID -eq 16 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name dr-cs09 \
        --wandb-entity matthew-farrugia-roberts --wandb-project corner-gmg \
        --num-total-env-steps 200_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm \
        --env-corner-size 9 --ued dr;
elif [ $SLURM_ARRAY_TASK_ID -eq 17 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name plr-cs09 \
        --wandb-entity matthew-farrugia-roberts --wandb-project corner-gmg \
        --num-total-env-steps 200_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm \
        --env-corner-size 9 --ued plr;
elif [ $SLURM_ARRAY_TASK_ID -eq 18 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name dr-cs10 \
        --wandb-entity matthew-farrugia-roberts --wandb-project corner-gmg \
        --num-total-env-steps 200_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm \
        --env-corner-size 10 --ued dr;
elif [ $SLURM_ARRAY_TASK_ID -eq 19 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name plr-cs10 \
        --wandb-entity matthew-farrugia-roberts --wandb-project corner-gmg \
        --num-total-env-steps 200_000_000 --no-env-penalize-time \
        --env-terminate-after-corner \
        --net-cnn-type large --net-rnn-type lstm \
        --env-corner-size 10 --ued plr;
fi
deactivate
