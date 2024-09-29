#!/bin/bash
#SBATCH --job-name=dish-nch
#SBATCH --partition=high_priority
#SBATCH --time=6:00:00
#SBATCH --gpus-per-node=1
#SBATCH --array 0-74%25

flags=(
    --no-console-log
    --wandb-log
    --wandb-entity krueger-lab-cambridge
    --wandb-project paper-dish-nch
    --env-size 15
    --no-env-terminate-after-dish
    --num-channels-cheese 1
    --net-cnn-type large
    --net-rnn-type ff
    --net-width 256
    --plr-regret-estimator "maxmc-actor"
    --plr-robust
    --no-plr-proxy-shaping
    --num-mutate-steps 12
    --prob-shift 3e-4
    --prob-mutate-shift 3e-4
    --num-total-env-steps 1_500_000_000
);
seed_array=(10 11 12);
num_channels_dish_array=(1 3 6 12 24);

for i in {0..2}; do
    seed=${seed_array[$i]}
    for j in {0..4}; do
        num_channels_dish=${num_channels_dish_array[$j]}
        x=$((i * 5 * 5 + j * 5))
        if [ $SLURM_ARRAY_TASK_ID -eq $((x + 0)) ]; then
            # DR
            jaxgmg train dish "${flags[@]}" \
                --wandb-name ch:$num_channels_dish-algo:dr-seed:$seed \
                --seed $seed \
                --num-channels-dish $num_channels_dish \
                --ued dr;
        elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 1)) ]; then
            # PLR (robust)
            jaxgmg train dish "${flags[@]}" \
                --wandb-name ch:$num_channels_dish-algo:plr-seed:$seed \
                --seed $seed \
                --num-channels-dish $num_channels_dish \
                --ued plr --plr-prob-replay 0.33;
        elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 2)) ]; then
            # ACCEL (const)
            jaxgmg train dish "${flags[@]}" \
                --wandb-name ch:$num_channels_dish-algo:accel_c-seed:$seed \
                --seed $seed \
                --num-channels-dish $num_channels_dish \
                --ued accel --plr-prob-replay 0.5 \
                    --chain-mutate --mutate-cheese-on-dish;
        elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 3)) ]; then
            # ACCEL (binom)
            jaxgmg train dish "${flags[@]}" \
                --wandb-name ch:$num_channels_dish-algo:accel_b-seed:$seed \
                --seed $seed \
                --num-channels-dish $num_channels_dish \
                --ued accel --plr-prob-replay 0.5 \
                    --no-chain-mutate --mutate-cheese-on-dish;
        elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 4)) ]; then
            # ACCEL (id)
            jaxgmg train dish "${flags[@]}" \
                --wandb-name ch:$num_channels_dish-algo:accel_d-seed:$seed \
                --seed $seed \
                --num-channels-dish $num_channels_dish \
                --ued accel --plr-prob-replay 0.5 \
                    --chain-mutate --no-mutate-cheese-on-dish;
        fi
    done
done
deactivate

