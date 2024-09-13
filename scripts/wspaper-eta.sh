#!/bin/bash
#SBATCH --job-name=wspaper-corner-eta
#SBATCH --partition=high_priority
#SBATCH --time=5:00:00
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/data/matthew_farrugia_roberts
#SBATCH --output=out/%A_%a-wspaper-corner-eta.stdout
#SBATCH --error=out/%A_%a-wspaper-corner-eta.stderr
#SBATCH --array 0-80%10

flags=(
    --no-console-log
    --wandb-log
    --wandb-entity matthew-farrugia-roberts
    --wandb-project wspaper-corner-eta
    --env-size 15
    --env-corner-size 1
    --net-cnn-type large
    --net-rnn-type ff
    --net-width 256
    --level-splayer "cheese-and-mouse"
    --plr-regret-estimator "maxmc-actor"
    --plr-robust
    --num-mutate-steps 12
);
seed_array=(0 1 2);
prob_shift=0.003;
eta_array=(0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9 1.0);

source jaxgmg.venv/bin/activate
for i in {0..2}; do
    seed=${seed_array[$i]}
    for k in {0..8}; do
        eta=${eta_array[$k]}
        j=$((i * 27 + k * 3))
        if [ $SLURM_ARRAY_TASK_ID -eq $((j + 0)) ]; then
            jaxgmg train corner "${flags[@]}" \
                --seed $seed \
                --wandb-name shift:$prob_shift-accel:prox-eta:$eta-seed:$seed \
                --prob-shift $prob_shift \
                --prob-mutate-shift $prob_shift \
                --ued accel \
                --plr-proxy-shaping \
                --plr-prob-replay 0.5 \
                --num-total-env-steps 750_000_000 \
                --plr-proxy-shaping-coeff $eta \
                --chain-mutate;
        elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 1)) ]; then
            jaxgmg train corner "${flags[@]}" \
                --seed $seed \
                --wandb-name shift:$prob_shift-accel:prox-id-eta:$eta-seed:$seed \
                --prob-shift $prob_shift \
                --prob-mutate-shift $prob_shift \
                --ued accel \
                --plr-proxy-shaping \
                --plr-prob-replay 0.5 \
                --num-total-env-steps 750_000_000 \
                --plr-proxy-shaping-coeff $eta \
                --chain-mutate --no-mutate-cheese;
        elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 2)) ]; then
            jaxgmg train corner "${flags[@]}" \
                --seed $seed \
                --wandb-name shift:$prob_shift-plr:prox-eta:$eta-seed:$seed \
                --prob-shift $prob_shift \
                --ued plr \
                --plr-proxy-shaping \
                --plr-prob-replay 0.33 \
                --plr-proxy-shaping-coeff $eta \
                --num-total-env-steps 750_000_000;
        fi
    done
done
deactivate

