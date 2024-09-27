#!/bin/bash
#SBATCH --job-name=dish-nch
#SBATCH --partition=high_priority
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/data/matthew_farrugia_roberts
#SBATCH --output=out/%A_%a-dish-nch.stdout
#SBATCH --error=out/%A_%a-dish-nch.stderr
#SBATCH --array 0-209%50

flags=(
    --no-console-log
    --wandb-log
    --wandb-entity matthew-farrugia-roberts
    --wandb-project dish-nch
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
plr_proxy_shaping_coeff=0.3;
seed_array=(0 1);
prob_mutate_shift_array=(3e-3 1e-4 1e-1);
num_channels_dish_array=(1 3 6 12 24);

source jaxgmg.venv/bin/activate
for i in {0..1}; do
    seed=${seed_array[$i]}
    for j in {0..2}; do
        prob_mutate_shift=${prob_mutate_shift_array[$j]}
        for k in {0..4}; do
            num_channels_dish=${num_channels_dish_array[$k]}
            x=$((i * 3 * 5 * 7 + j * 5 * 7 + k * 7))
            if [ $SLURM_ARRAY_TASK_ID -eq $((x + 0)) ]; then
                jaxgmg train dish "${flags[@]}" \
                    --wandb-name ch:$num_channels_dish-algo:dr-alpha:$prob_mutate_shift-seed:$seed \
                    --seed $seed \
                    --prob-shift $prob_mutate_shift \
                    --prob-mutate-shift $prob_mutate_shift \
                    --num-channels-dish $num_channels_dish \
                    --ued dr \
                    --num-total-env-steps 250_000_000;
            elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 1)) ]; then
                jaxgmg train dish "${flags[@]}" \
                    --wandb-name ch:$num_channels_dish-algo:plr:orig-alpha:$prob_mutate_shift-seed:$seed \
                    --seed $seed \
                    --prob-shift $prob_mutate_shift \
                    --prob-mutate-shift $prob_mutate_shift \
                    --num-channels-dish $num_channels_dish \
                    --ued plr \
                    --no-plr-proxy-shaping \
                    --plr-prob-replay 0.33 \
                    --num-total-env-steps 750_000_000;
            elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 2)) ]; then
                jaxgmg train dish "${flags[@]}" \
                    --wandb-name ch:$num_channels_dish-algo:plr:prox-alpha:$prob_mutate_shift-seed:$seed \
                    --seed $seed \
                    --prob-shift $prob_mutate_shift \
                    --prob-mutate-shift $prob_mutate_shift \
                    --num-channels-dish $num_channels_dish \
                    --ued plr \
                    --plr-proxy-shaping \
                    --plr-proxy-shaping-coeff $plr_proxy_shaping_coeff \
                    --plr-prob-replay 0.33 \
                    --num-total-env-steps 750_000_000;
            elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 3)) ]; then
                jaxgmg train dish "${flags[@]}" \
                    --wandb-name ch:$num_channels_dish-algo:accel:origc-alpha:$prob_mutate_shift-seed:$seed \
                    --seed $seed \
                    --prob-shift $prob_mutate_shift \
                    --prob-mutate-shift $prob_mutate_shift \
                    --num-channels-dish $num_channels_dish \
                    --ued accel \
                    --no-plr-proxy-shaping \
                    --plr-prob-replay 0.5 \
                    --num-total-env-steps 750_000_000 \
                    --chain-mutate --mutate-cheese-on-dish;
            elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 4)) ]; then
                jaxgmg train dish "${flags[@]}" \
                    --wandb-name ch:$num_channels_dish-algo:accel:proxc-alpha:$prob_mutate_shift-seed:$seed \
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
            elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 5)) ]; then
                jaxgmg train dish "${flags[@]}" \
                    --wandb-name ch:$num_channels_dish-algo:accel:origb-alpha:$prob_mutate_shift-seed:$seed \
                    --seed $seed \
                    --prob-shift $prob_mutate_shift \
                    --prob-mutate-shift $prob_mutate_shift \
                    --num-channels-dish $num_channels_dish \
                    --ued accel \
                    --no-plr-proxy-shaping \
                    --plr-prob-replay 0.5 \
                    --num-total-env-steps 750_000_000 \
                    --no-chain-mutate --mutate-cheese-on-dish;
            elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 6)) ]; then
                jaxgmg train dish "${flags[@]}" \
                    --wandb-name ch:$num_channels_dish-algo:accel:proxb-alpha:$prob_mutate_shift-seed:$seed \
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
done
deactivate

