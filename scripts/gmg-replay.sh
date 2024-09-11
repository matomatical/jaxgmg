#!/bin/bash
#SBATCH --job-name=gmg-replay
#SBATCH --partition=high_priority
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/data/matthew_farrugia_roberts
#SBATCH --output=out/%A_%a-gmg-replay.stdout
#SBATCH --error=out/%A_%a-gmg-replay.stderr
#SBATCH --array 0-39%10

flags=(
    --no-console-log
    --wandb-log
    --wandb-entity matthew-farrugia-roberts
    --wandb-project gmg-replay
    --env-size 15
    --env-corner-size 1
    --net-cnn-type large
    --net-rnn-type ff
    --net-width 256
    --level-splayer "cheese-and-mouse"
    --plr-regret-estimator "maxmc-actor"
    --plr-robust
    --num-mutate-steps 12
    --plr-proxy-shaping-coeff 0.5
);
prob_shift_array=(0.0003 0.001 0.003 0.01 0.03);
accel_prob_replay_array=(0.7 0.5);
plr_prob_replay_array=(0.5 0.33);

source jaxgmg.venv/bin/activate
for i in {0..4}; do
    prob_shift=${prob_shift_array[$i]}
    j=$((i * 8))
    if [ $SLURM_ARRAY_TASK_ID -eq $((j + 0)) ]; then
        jaxgmg train corner "${flags[@]}" \
            --wandb-name shift:$prob_shift-accel:orig-replay:0.7 \
            --prob-shift $prob_shift \
            --prob-mutate-shift $prob_shift \
            --ued accel \
            --no-plr-proxy-shaping \
            --plr-prob-replay 0.7 \
            --num-total-env-steps 607_000_000 \
            --chain-mutate;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 1)) ]; then
        jaxgmg train corner "${flags[@]}" \
            --wandb-name shift:$prob_shift-accel:prox-replay:0.7 \
            --prob-shift $prob_shift \
            --prob-mutate-shift $prob_shift \
            --ued accel \
            --plr-proxy-shaping \
            --plr-prob-replay 0.7 \
            --num-total-env-steps 607_000_000 \
            --chain-mutate;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 2)) ]; then
        jaxgmg train corner "${flags[@]}" \
            --wandb-name shift:$prob_shift-accel:orig-replay:0.5 \
            --prob-shift $prob_shift \
            --prob-mutate-shift $prob_shift \
            --ued accel \
            --no-plr-proxy-shaping \
            --plr-prob-replay 0.5 \
            --num-total-env-steps 750_000_000 \
            --chain-mutate;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 3)) ]; then
        jaxgmg train corner "${flags[@]}" \
            --wandb-name shift:$prob_shift-accel:prox-replay:0.5 \
            --prob-shift $prob_shift \
            --prob-mutate-shift $prob_shift \
            --ued accel \
            --plr-proxy-shaping \
            --plr-prob-replay 0.5 \
            --num-total-env-steps 750_000_000 \
            --chain-mutate;
    # PLR
    elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 4)) ]; then
        jaxgmg train corner "${flags[@]}" \
            --wandb-name shift:$prob_shift-plr:orig-replay:0.5 \
            --prob-shift $prob_shift \
            --ued plr \
            --no-plr-proxy-shaping \
            --plr-prob-replay 0.5 \
            --num-total-env-steps 500_000_000;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 5)) ]; then
        jaxgmg train corner "${flags[@]}" \
            --wandb-name shift:$prob_shift-plr:prox-replay:0.5 \
            --prob-shift $prob_shift \
            --ued plr \
            --plr-proxy-shaping \
            --plr-prob-replay 0.5 \
            --num-total-env-steps 500_000_000;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 6)) ]; then
        jaxgmg train corner "${flags[@]}" \
            --wandb-name shift:$prob_shift-plr:orig-replay:0.33 \
            --prob-shift $prob_shift \
            --ued plr \
            --no-plr-proxy-shaping \
            --plr-prob-replay 0.33 \
            --num-total-env-steps 750_000_000;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 7)) ]; then
        jaxgmg train corner "${flags[@]}" \
            --wandb-name shift:$prob_shift-plr:prox-replay:0.33 \
            --prob-shift $prob_shift \
            --ued plr \
            --plr-proxy-shaping \
            --plr-prob-replay 0.33 \
            --num-total-env-steps 750_000_000;
    fi
done
deactivate

