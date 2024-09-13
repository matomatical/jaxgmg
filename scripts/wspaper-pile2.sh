#!/bin/bash
#SBATCH --job-name=wspaper-pile
#SBATCH --partition=high_priority
#SBATCH --time=5:00:00
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/data/matthew_farrugia_roberts
#SBATCH --output=out/%A_%a-wspaper-pile.stdout
#SBATCH --error=out/%A_%a-wspaper-pile.stderr
#SBATCH --array 0-47%10

flags=(
    --no-console-log
    --wandb-log
    --wandb-entity matthew-farrugia-roberts
    --wandb-project wspaper-pile
    --env-size 15
    --max-cheese-radius 0
    --max-cheese-radius-shift 100
    --no-env-terminate-after-pile
    --net-cnn-type large
    --net-rnn-type ff
    --net-width 256
    --plr-regret-estimator "maxmc-actor"
    --plr-robust
    --num-mutate-steps 12
    --plr-proxy-shaping-coeff 0.5
);
seed_array=(0 1 2);
prob_shift_array=(0 0.00001 0.00003 0.0001 0.0003 0.001 0.003 1);

source jaxgmg.venv/bin/activate
for i in {0..2}; do
    seed=${seed_array[$i]}
    for k in {0..7}; do
        prob_shift=${prob_shift_array[$k]}
        j=$((i * 16 + k * 2))
        if [ $SLURM_ARRAY_TASK_ID -eq $((j + 0)) ]; then
            jaxgmg train pile "${flags[@]}" \
                --seed $seed \
                --wandb-name shift:$prob_shift-accel:orig-id-seed:$seed \
                --prob-shift $prob_shift \
                --prob-mutate-shift $prob_shift \
                --ued accel \
                --no-plr-proxy-shaping \
                --plr-prob-replay 0.5 \
                --num-total-env-steps 750_000_000 \
                --chain-mutate --no-mutate-cheese-on-pile;
        elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 1)) ]; then
            jaxgmg train pile "${flags[@]}" \
                --seed $seed \
                --wandb-name shift:$prob_shift-accel:prox-id-seed:$seed \
                --prob-shift $prob_shift \
                --prob-mutate-shift $prob_shift \
                --ued accel \
                --plr-proxy-shaping \
                --plr-prob-replay 0.5 \
                --num-total-env-steps 750_000_000 \
                --chain-mutate --no-mutate-cheese-on-pile;
        fi
    done
done
deactivate

