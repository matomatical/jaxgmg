#!/bin/bash
#SBATCH --job-name=wspaper-corner
#SBATCH --partition=high_priority
#SBATCH --time=5:00:00
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/data/matthew_farrugia_roberts
#SBATCH --output=out/%A_%a-wspaper-corner.stdout
#SBATCH --error=out/%A_%a-wspaper-corner.stderr
#SBATCH --array 0-119%10

flags=(
    --no-console-log
    --wandb-log
    --wandb-entity matthew-farrugia-roberts
    --wandb-project wspaper-corner
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
seed_array=(0 1 2);
prob_shift_array=(0 0.0003 0.001 0.003 0.01 0.03 0.1 1);

source jaxgmg.venv/bin/activate
for i in {0..2}; do
    seed=${seed_array[$i]}
    for k in {0..7}; do
        prob_shift=${prob_shift_array[$k]}
        j=$((i * 40 + k * 5))
        if [ $SLURM_ARRAY_TASK_ID -eq $((j + 0)) ]; then
            jaxgmg train corner "${flags[@]}" \
                --seed $seed \
                --wandb-name shift:$prob_shift-accel:orig-seed:$seed \
                --prob-shift $prob_shift \
                --prob-mutate-shift $prob_shift \
                --ued accel \
                --no-plr-proxy-shaping \
                --plr-prob-replay 0.5 \
                --num-total-env-steps 750_000_000 \
                --chain-mutate;
        elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 1)) ]; then
            jaxgmg train corner "${flags[@]}" \
                --seed $seed \
                --wandb-name shift:$prob_shift-accel:prox-seed:$seed \
                --prob-shift $prob_shift \
                --prob-mutate-shift $prob_shift \
                --ued accel \
                --plr-proxy-shaping \
                --plr-prob-replay 0.5 \
                --num-total-env-steps 750_000_000 \
                --chain-mutate;
        elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 2)) ]; then
            jaxgmg train corner "${flags[@]}" \
                --seed $seed \
                --wandb-name shift:$prob_shift-plr:orig-seed:$seed \
                --prob-shift $prob_shift \
                --ued plr \
                --no-plr-proxy-shaping \
                --plr-prob-replay 0.33 \
                --num-total-env-steps 750_000_000;
        elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 3)) ]; then
            jaxgmg train corner "${flags[@]}" \
                --seed $seed \
                --wandb-name shift:$prob_shift-plr:prox-seed:$seed \
                --prob-shift $prob_shift \
                --ued plr \
                --plr-proxy-shaping \
                --plr-prob-replay 0.33 \
                --num-total-env-steps 750_000_000;
        elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 4)) ]; then
            jaxgmg train corner "${flags[@]}" \
                --seed $seed \
                --wandb-name shift:$prob_shift-dr-seed:$seed \
                --prob-shift $prob_shift \
                --ued dr \
                --num-total-env-steps 250_000_000;
        fi
    done
done
deactivate

