#!/bin/bash
#SBATCH --job-name=paper-dish
#SBATCH --partition=high_priority
#SBATCH --time=6:00:00
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/data/matthew_farrugia_roberts
#SBATCH --output=out/%A_%a-paper-dish.stdout
#SBATCH --error=out/%A_%a-paper-dish.stderr
#SBATCH --array 0-164%55

flags=(
    --no-console-log
    --wandb-log
    --wandb-entity matthew-farrugia-roberts
    --wandb-project paper-dish
    --env-size 15
    --num-channels-dish 6
    --no-env-terminate-after-dish
    --num-channels-cheese 1
    --net-cnn-type large
    --net-rnn-type ff
    --net-width 256
    --plr-regret-estimator "maxmc-actor"
    --plr-robust
    --num-mutate-steps 12
    --num-total-env-steps 1_500_000_000
);
seed_array=(10 11 12);
prob_shift_array=(0 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 1);

source jaxgmg.venv/bin/activate
for i in {0..2}; do
    seed=${seed_array[$i]}
    for j in {0..10}; do
        prob_shift=${prob_shift_array[$j]}
        x=$((i * 11 * 5 + j * 5))
        if [ $SLURM_ARRAY_TASK_ID -eq $((x + 0)) ]; then
            # DR
            jaxgmg train dish "${flags[@]}" \
                --wandb-name algo:dr-alpha:$prob_shift-seed:$seed \
                --seed $seed \
                --prob-shift $prob_shift --prob-mutate-shift $prob_shift \
                --ued dr;
        elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 1)) ]; then
            # PLR (robust)
            jaxgmg train dish "${flags[@]}" \
                --wandb-name algo:plr-alpha:$prob_shift-seed:$seed \
                --seed $seed \
                --prob-shift $prob_shift --prob-mutate-shift $prob_shift \
                --ued plr --plr-prob-replay 0.33 --no-plr-proxy-shaping;
        elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 2)) ]; then
            # ACCEL (const)
            jaxgmg train dish "${flags[@]}" \
                --wandb-name algo:accelc-alpha:$prob_shift-seed:$seed \
                --seed $seed \
                --prob-shift $prob_shift --prob-mutate-shift $prob_shift \
                --ued accel --plr-prob-replay 0.5 --no-plr-proxy-shaping \
                    --chain-mutate --mutate-cheese-on-dish;
        elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 3)) ]; then
            # ACCEL (binom)
            jaxgmg train dish "${flags[@]}" \
                --wandb-name algo:accelb-alpha:$prob_shift-seed:$seed \
                --seed $seed \
                --prob-shift $prob_shift --prob-mutate-shift $prob_shift \
                --ued accel --no-plr-proxy-shaping --plr-prob-replay 0.5 \
                    --no-chain-mutate --mutate-cheese-on-dish;
        elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 4)) ]; then
            # ACCEL (id)
            jaxgmg train dish "${flags[@]}" \
                --wandb-name algo:accel_d-alpha:$prob_shift-seed:$seed \
                --seed $seed \
                --prob-shift $prob_shift --prob-mutate-shift $prob_shift \
                --ued accel --no-plr-proxy-shaping --plr-prob-replay 0.5 \
                    --chain-mutate --no-mutate-cheese-on-dish;
        fi
    done
done
deactivate

