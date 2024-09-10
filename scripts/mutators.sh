#!/bin/bash
#SBATCH --job-name=mutators
#SBATCH --partition=high_priority
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/data/matthew_farrugia_roberts
#SBATCH --output=out/%A_%a-mutators.stdout
#SBATCH --error=out/%A_%a-mutators.stderr
#SBATCH --array 0-14

flags=(
    --no-console-log
    --wandb-log
    --wandb-entity matthew-farrugia-roberts
    --wandb-project mutators
    --env-size 15
    --env-corner-size 13
    --net-cnn-type large
    --net-rnn-type ff
    --net-width 256
    --num-total-env-steps 500_000_000
    --level-splayer "cheese-and-mouse"
    --ued accel
    --plr-regret-estimator "maxmc-actor"
    --plr-robust
    --plr-prob-replay 0.8
);
num_mutations_array=(03 07 12 32 82);

source jaxgmg.venv/bin/activate
for i in {0..4}; do
    num_mutations=${num_mutations_array[$i]}
    j=$((i * 3))
    if [ $SLURM_ARRAY_TASK_ID -eq $((j + 0)) ]; then
        jaxgmg train corner "${flags[@]}" \
            --wandb-name binom-mut$num_mutations \
            --num-mutate-steps $num_mutations \
            --no-chain-mutate \
            --no-step-mutate;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 1)) ]; then
        jaxgmg train corner "${flags[@]}" \
            --wandb-name chain-mut$num_mutations \
            --num-mutate-steps $num_mutations \
            --chain-mutate \
            --no-step-mutate;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((j + 2)) ]; then
        jaxgmg train corner "${flags[@]}" \
            --wandb-name steps-mut$num_mutations \
            --num-mutate-steps $num_mutations \
            --chain-mutate \
            --step-mutate;
    fi
done
deactivate

