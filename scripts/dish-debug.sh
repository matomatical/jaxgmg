#!/bin/bash
#SBATCH --job-name=dbg-dish
#SBATCH --partition=high_priority
#SBATCH --time=5:00:00
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/data/matthew_farrugia_roberts
#SBATCH --output=out/%A_%a-dbg-dish.stdout
#SBATCH --error=out/%A_%a-dbg-dish.stderr
#SBATCH --array 0-20

flags=(
    --no-console-log
    --wandb-log
    --wandb-entity matthew-farrugia-roberts
    --wandb-project dbg-dish
    --env-size 15
    --no-env-terminate-after-dish
    --num-channels-cheese 1
    --num-channels-dish 6
    --net-cnn-type large
    --net-rnn-type ff
    --net-width 256
    --plr-regret-estimator "maxmc-actor"
    --plr-robust
    --num-mutate-steps 12
    --plr-proxy-shaping-coeff 0.3
    --prob-shift 1e-5
    --prob-mutate-shift 1e-5
);
seed_array=(0 1 2);

source jaxgmg.venv/bin/activate
for i in {0..2}; do
    seed=${seed_array[$i]}
    j=$((i * 7))
    if [ $SLURM_ARRAY_TASK_ID -eq $((j + 0)) ]; then
        jaxgmg train dish "${flags[@]}" \
            --seed $seed \
            --wandb-name dr-seed:$seed \
            --ued dr \
            --num-total-env-steps 250_000_000;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 1)) ]; then
        jaxgmg train dish "${flags[@]}" \
            --seed $seed \
            --wandb-name plr:orig-seed:$seed \
            --ued plr \
            --no-plr-proxy-shaping \
            --plr-prob-replay 0.33 \
            --num-total-env-steps 750_000_000;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 2)) ]; then
        jaxgmg train dish "${flags[@]}" \
            --seed $seed \
            --wandb-name plr:prox-seed:$seed \
            --ued plr \
            --plr-proxy-shaping \
            --plr-prob-replay 0.33 \
            --num-total-env-steps 750_000_000;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 3)) ]; then
        jaxgmg train dish "${flags[@]}" \
            --seed $seed \
            --wandb-name accel:orig-c-seed:$seed \
            --ued accel \
            --no-plr-proxy-shaping \
            --plr-prob-replay 0.5 \
            --num-total-env-steps 750_000_000 \
            --chain-mutate --mutate-cheese-on-dish;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 4)) ]; then
        jaxgmg train dish "${flags[@]}" \
            --seed $seed \
            --wandb-name accel:prox-c-seed:$seed \
            --ued accel \
            --plr-proxy-shaping \
            --plr-prob-replay 0.5 \
            --num-total-env-steps 750_000_000 \
            --chain-mutate --mutate-cheese-on-dish;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 5)) ]; then
        jaxgmg train dish "${flags[@]}" \
            --seed $seed \
            --wandb-name accel:orig-binom-seed:$seed \
            --ued accel \
            --no-plr-proxy-shaping \
            --plr-prob-replay 0.5 \
            --num-total-env-steps 750_000_000 \
            --no-chain-mutate --mutate-cheese-on-dish;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 6)) ]; then
        jaxgmg train dish "${flags[@]}" \
            --seed $seed \
            --wandb-name accel:prox-binom-seed:$seed \
            --ued accel \
            --plr-proxy-shaping \
            --plr-prob-replay 0.5 \
            --num-total-env-steps 750_000_000 \
            --no-chain-mutate --mutate-cheese-on-dish;
    fi
done
deactivate

