#!/bin/bash
#SBATCH --job-name=amp-mut
#SBATCH --partition=high_priority
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/data/matthew_farrugia_roberts
#SBATCH --output=out/%A_%a-amp-mut.stdout
#SBATCH --error=out/%A_%a-amp-mut.stderr
#SBATCH --array 0-34%10

flags=(
    --no-console-log
    --wandb-log
    --wandb-entity matthew-farrugia-roberts
    --wandb-project amp-mut
    --env-size 15
    --env-corner-size 1
    --net-cnn-type large
    --net-rnn-type ff
    --net-width 256
    --num-total-env-steps 500_000_000
    --level-splayer "cheese-and-mouse"
    --plr-regret-estimator "maxmc-actor"
    --plr-robust
    --plr-prob-replay 0.9
    --num-mutate-steps 12
    --plr-proxy-shaping-coeff 0.5
);
prob_shift_array=(0.0001 0.0003 0.001 0.003 0.01 0.03 0.1);

source jaxgmg.venv/bin/activate
for i in {0..6}; do
    prob_shift=${prob_shift_array[$i]}
    j=$((i * 5))
    if [ $SLURM_ARRAY_TASK_ID -eq $((j + 0)) ]; then
        jaxgmg train corner "${flags[@]}" \
            --wandb-name shift:$prob_shift-accel:orig-mut:binom \
            --prob-shift $prob_shift \
            --prob-mutate-shift $prob_shift \
            --ued accel \
            --no-plr-proxy-shaping \
            --no-chain-mutate;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 1)) ]; then
        jaxgmg train corner "${flags[@]}" \
            --wandb-name shift:$prob_shift-accel:orig-mut:chain \
            --prob-shift $prob_shift \
            --prob-mutate-shift $prob_shift \
            --ued accel \
            --no-plr-proxy-shaping \
            --chain-mutate;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 2)) ]; then
        jaxgmg train corner "${flags[@]}" \
            --wandb-name shift:$prob_shift-accel:prox-mut:binom \
            --prob-shift $prob_shift \
            --prob-mutate-shift $prob_shift \
            --ued accel \
            --plr-proxy-shaping \
            --no-chain-mutate;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 3)) ]; then
        jaxgmg train corner "${flags[@]}" \
            --wandb-name shift:$prob_shift-accel:prox-mut:chain \
            --prob-shift $prob_shift \
            --prob-mutate-shift $prob_shift \
            --ued accel \
            --plr-proxy-shaping \
            --chain-mutate;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 4)) ]; then
        jaxgmg train corner "${flags[@]}" \
            --wandb-name shift:$prob_shift-dr \
            --prob-shift $prob_shift \
            --prob-mutate-shift $prob_shift \
            --ued dr;
    fi
done
deactivate

