#!/bin/bash
#SBATCH --job-name=wspaper-corner
#SBATCH --partition=high_priority
#SBATCH --time=5:00:00
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/data/matthew_farrugia_roberts
#SBATCH --output=out/%A_%a-wspaper-corner.stdout
#SBATCH --error=out/%A_%a-wspaper-corner.stderr
#SBATCH --array 0-8

flags=(
    --no-console-log
    --wandb-log
    --wandb-entity krueger-lab-cambridge
    --wandb-project paper-corner-extra-id
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
prob_shift_array=(1e-5 3e-5 1e-4);

source jaxgmg.venv/bin/activate
for i in {0..2}; do
    seed=${seed_array[$i]}
    for j in {0..2}; do
        prob_shift=${prob_shift_array[$j]}
        x=$((i * 3 * 1 + j * 1))
        if [ $SLURM_ARRAY_TASK_ID -eq $((x + 0)) ]; then
            jaxgmg train corner "${flags[@]}" \
                --seed $seed \
                --wandb-name shift:$prob_shift-accel:orig-id-seed:$seed \
                --prob-shift $prob_shift \
                --prob-mutate-shift $prob_shift \
                --ued accel \
                --no-plr-proxy-shaping \
                --plr-prob-replay 0.5 \
                --num-total-env-steps 750_000_000 \
                --chain-mutate --no-mutate-cheese;
        fi
    done
done
deactivate

