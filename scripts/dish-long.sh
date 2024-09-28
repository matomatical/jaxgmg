#!/bin/bash
#SBATCH --job-name=dish-long
#SBATCH --partition=high_priority
#SBATCH --time=16:00:00
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/data/matthew_farrugia_roberts
#SBATCH --output=out/%A_%a-dish-long.stdout
#SBATCH --error=out/%A_%a-dish-long.stderr
#SBATCH --array 0-19

flags=(
    --no-console-log
    --wandb-log
    --wandb-entity matthew-farrugia-roberts
    --wandb-project dish-long
    --env-size 15
    --no-env-terminate-after-dish
    --num-channels-cheese 1
    --num_channels_dish 6
    --net-cnn-type large
    --net-rnn-type ff
    --net-width 256
    --plr-regret-estimator "maxmc-actor"
    --plr-robust
    --num-mutate-steps 12
    --prob-shift 1e-4
    --prob-mutate-shift 1e-4
    --plr-proxy-shaping-coeff 0.3
    --num-total-env-steps 3_000_000_000
);
seed_array=(0 1);

source jaxgmg.venv/bin/activate
for i in {0..1}; do
    seed=${seed_array[$i]}
    x=$((i * 10))
    if [ $SLURM_ARRAY_TASK_ID -eq $((x + 0)) ]; then
        jaxgmg train dish "${flags[@]}" \
            --wandb-name dr-seed:$seed \
            --seed $seed \
            --ued dr \
            --num-total-env-steps 1_000_000_000;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 1)) ]; then
        jaxgmg train dish "${flags[@]}" \
            --wandb-name plr_orig-seed:$seed \
            --seed $seed \
            --ued plr --plr-prob-replay 0.33 \
            --no-plr-proxy-shaping;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 2)) ]; then
        jaxgmg train dish "${flags[@]}" \
            --wandb-name plr_prox_clip-seed:$seed \
            --seed $seed \
            --ued plr --plr-prob-replay 0.33 \
            --plr-proxy-shaping --clipping;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 3)) ]; then
        jaxgmg train dish "${flags[@]}" \
            --wandb-name plr_prox_noclip-seed:$seed \
            --seed $seed \
            --ued plr --plr-prob-replay 0.33 \
            --plr-proxy-shaping --no-clipping;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 4)) ]; then
        jaxgmg train dish "${flags[@]}" \
            --wandb-name accel_origc-seed:$seed \
            --seed $seed \
            --ued accel --plr-prob-replay 0.5 \
            --no-plr-proxy-shaping \
            --chain-mutate --mutate-cheese-on-dish;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 5)) ]; then
        jaxgmg train dish "${flags[@]}" \
            --wandb-name accel_proxc_clip-seed:$seed \
            --seed $seed \
            --ued accel --plr-prob-replay 0.5 \
            --plr-proxy-shaping --clipping \
            --chain-mutate --mutate-cheese-on-dish;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 6)) ]; then
        jaxgmg train dish "${flags[@]}" \
            --wandb-name accel_proxc_noclip-seed:$seed \
            --seed $seed \
            --ued accel --plr-prob-replay 0.5 \
            --plr-proxy-shaping --no-clipping \
            --chain-mutate --mutate-cheese-on-dish;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 7)) ]; then
        jaxgmg train dish "${flags[@]}" \
            --wandb-name accel_origb-seed:$seed \
            --seed $seed \
            --ued accel --plr-prob-replay 0.5 \
            --no-plr-proxy-shaping \
            --no-chain-mutate --mutate-cheese-on-dish;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 8)) ]; then
        jaxgmg train dish "${flags[@]}" \
            --wandb-name accel_proxb_clip-seed:$seed \
            --seed $seed \
            --ued accel --plr-prob-replay 0.5 \
            --plr-proxy-shaping --clipping \
            --no-chain-mutate --mutate-cheese-on-dish;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 9)) ]; then
        jaxgmg train dish "${flags[@]}" \
            --wandb-name accel_proxb_noclip-seed:$seed \
            --seed $seed \
            --ued accel --plr-prob-replay 0.5 \
            --plr-proxy-shaping --no-clipping \
            --no-chain-mutate --mutate-cheese-on-dish;
    fi
done
deactivate

