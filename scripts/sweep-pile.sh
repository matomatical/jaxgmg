#!/bin/bash
#SBATCH --job-name=dbg-dish
#SBATCH --partition=high_priority
#SBATCH --time=5:00:00
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/data/matthew_farrugia_roberts
#SBATCH --output=out/%A_%a-dbg-dish.stdout
#SBATCH --error=out/%A_%a-dbg-dish.stderr
#SBATCH --array 0-20%21

flags=(
    --no-console-log
    --wandb-log
    --wandb-entity matthew-farrugia-roberts
    --wandb-project dbg-dish
    --env-size 15
    --no-env-terminate-after-dish
    --num-channels-cheese 1
    --net-cnn-type large
    --net-rnn-type ff
    --net-width 256
    --plr-regret-estimator "maxmc-actor"
    --plr-robust
    --num-mutate-steps 12
);
seed=42;
plr_proxy_shaping_coeff=0.3;
prob_mutate_shift_array=(1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 1 0);
num_chanels_dish_array=(1 6 11 16);

source jaxgmg.venv/bin/activate
for i in {0..3}; do
    num_channels_dish=${num_channels_dish_array[$i]}
    for j in {0..10}; do
        prob_mutate_shift=${prob_mutate_shift_array[$j]}
        k=$((i * 11 * 7 + j * 7))
        if [ $SLURM_ARRAY_TASK_ID -eq $((k + 0)) ]; then
            jaxgmg train dish "${flags[@]}" \
                --wandb-name dr-alpha:$prob_mutate_shift-seed:$seed \
                --seed $seed \
                --prob-shift $prob_mutate_shift \
                --prob-mutate-shift $prob_mutate_shift \
                --num-channels-dish $num_channels_dish \
                --ued dr \
                --num-total-env-steps 250_000_000;
        elif [ $SLURM_ARRAY_TASK_ID -eq $((k + 1)) ]; then
            jaxgmg train dish "${flags[@]}" \
                --wandb-name plr:orig-alpha:$prob_mutate_shift-seed:$seed \
                --seed $seed \
                --prob-shift $prob_mutate_shift \
                --prob-mutate-shift $prob_mutate_shift \
                --num-channels-dish $num_channels_dish \
                --ued plr \
                --no-plr-proxy-shaping \
                --plr-prob-replay 0.33 \
                --num-total-env-steps 750_000_000;
        elif [ $SLURM_ARRAY_TASK_ID -eq $((k + 2)) ]; then
            jaxgmg train dish "${flags[@]}" \
                --wandb-name plr:prox-alpha:$prob_mutate_shift-seed:$seed \
                --seed $seed \
                --prob-shift $prob_mutate_shift \
                --prob-mutate-shift $prob_mutate_shift \
                --num-channels-dish $num_channels_dish \
                --ued plr \
                --plr-proxy-shaping \
                --plr-proxy-shaping-coeff $plr_proxy_shaping_coeff \
                --plr-prob-replay 0.33 \
                --num-total-env-steps 750_000_000;
        elif [ $SLURM_ARRAY_TASK_ID -eq $((k + 3)) ]; then
            jaxgmg train dish "${flags[@]}" \
                --wandb-name accel:origc-alpha:$prob_mutate_shift-seed:$seed \
                --seed $seed \
                --prob-shift $prob_mutate_shift \
                --prob-mutate-shift $prob_mutate_shift \
                --num-channels-dish $num_channels_dish \
                --ued accel \
                --no-plr-proxy-shaping \
                --plr-prob-replay 0.5 \
                --num-total-env-steps 750_000_000 \
                --chain-mutate --mutate-cheese-on-dish;
        elif [ $SLURM_ARRAY_TASK_ID -eq $((k + 4)) ]; then
            jaxgmg train dish "${flags[@]}" \
                --wandb-name accel:proxc-alpha:$prob_mutate_shift-seed:$seed \
                --seed $seed \
                --prob-shift $prob_mutate_shift \
                --prob-mutate-shift $prob_mutate_shift \
                --num-channels-dish $num_channels_dish \
                --ued accel \
                --plr-proxy-shaping \
                --plr-proxy-shaping-coeff $plr_proxy_shaping_coeff \
                --plr-prob-replay 0.5 \
                --num-total-env-steps 750_000_000 \
                --chain-mutate --mutate-cheese-on-dish;
        elif [ $SLURM_ARRAY_TASK_ID -eq $((k + 5)) ]; then
            jaxgmg train dish "${flags[@]}" \
                --wandb-name accel:origb-alpha:$prob_mutate_shift-seed:$seed \
                --seed $seed \
                --prob-shift $prob_mutate_shift \
                --prob-mutate-shift $prob_mutate_shift \
                --num-channels-dish $num_channels_dish \
                --ued accel \
                --no-plr-proxy-shaping \
                --plr-prob-replay 0.5 \
                --num-total-env-steps 750_000_000 \
                --no-chain-mutate --mutate-cheese-on-dish;
        elif [ $SLURM_ARRAY_TASK_ID -eq $((k + 6)) ]; then
            jaxgmg train dish "${flags[@]}" \
                --wandb-name accel:proxb-alpha:$prob_mutate_shift-seed:$seed \
                --seed $seed \
                --prob-shift $prob_mutate_shift \
                --prob-mutate-shift $prob_mutate_shift \
                --num-channels-dish $num_channels_dish \
                --ued accel \
                --plr-proxy-shaping \
                --plr-proxy-shaping-coeff $plr_proxy_shaping_coeff \
                --plr-prob-replay 0.5 \
                --num-total-env-steps 750_000_000 \
                --no-chain-mutate --mutate-cheese-on-dish;
        fi
    done
done
deactivate

