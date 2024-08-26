#!/bin/bash
#SBATCH --job-name=gmg-mix2
#SBATCH --partition=high_priority
#SBATCH --time=8:00:00
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/data/matthew_farrugia_roberts
#SBATCH --output=out/%A-%a-gmg-mix2.stdout
#SBATCH --error=out/%A-%a-gmg-mix2.stderr
#SBATCH --array 0-14

source jaxgmg.venv/bin/activate
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name dr-mix00 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg2 \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.00 --ued dr;
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name plr-mix00 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg2 \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.00 --ued plr;
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name accel3-mix00 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg2 \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.00 --ued accel --num-mutate-steps 3;
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name accel6-mix00 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg2 \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.00 --ued accel --num-mutate-steps 6;
elif [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name accel9-mix00 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg2 \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.00 --ued accel --num-mutate-steps 9;
elif [ $SLURM_ARRAY_TASK_ID -eq 5 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name dr-mix03 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg2 \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.03 --ued dr;
elif [ $SLURM_ARRAY_TASK_ID -eq 6 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name plr-mix03 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg2 \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.03 --ued plr;
elif [ $SLURM_ARRAY_TASK_ID -eq 7 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name accel3-mix03 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg2 \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.03 --ued accel --num-mutate-steps 3;
elif [ $SLURM_ARRAY_TASK_ID -eq 8 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name accel6-mix03 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg2 \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.03 --ued accel --num-mutate-steps 6;
elif [ $SLURM_ARRAY_TASK_ID -eq 9 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name accel9-mix03 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg2 \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.03 --ued accel --num-mutate-steps 9;
elif [ $SLURM_ARRAY_TASK_ID -eq 10 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name accel1-mix00 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg2 \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.00 --ued accel --num-mutate-steps 1;
elif [ $SLURM_ARRAY_TASK_ID -eq 11 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name accel2-mix00 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg2 \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.00 --ued accel --num-mutate-steps 2;
elif [ $SLURM_ARRAY_TASK_ID -eq 12 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name accel16-mix00 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg2 \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.00 --ued accel --num-mutate-steps 16;
elif [ $SLURM_ARRAY_TASK_ID -eq 13 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name accel32-mix00 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg2 \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.00 --ued accel --num-mutate-steps 32;
elif [ $SLURM_ARRAY_TASK_ID -eq 14 ]; then
    jaxgmg train corner \
        --wandb-log --no-console-log --wandb-name accel64-mix00 \
        --wandb-entity matthew-farrugia-roberts --wandb-project mix-gmg2 \
        --num-total-env-steps 100_000_000 --no-env-penalize-time \
        --net-cnn-type large --net-rnn-type lstm \
        --prob-shift 0.00 --ued accel --num-mutate-steps 64;
fi
deactivate
