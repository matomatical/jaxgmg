#!/bin/bash
#SBATCH --job-name=amp-pile
#SBATCH --partition=high_priority
#SBATCH --time=14:00:00
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/data/matthew_farrugia_roberts
#SBATCH --output=out/%A_%a-amp-pile.stdout
#SBATCH --error=out/%A_%a-amp-pile.stderr
#SBATCH --array 0-39%10

source jaxgmg.venv/bin/activate
shifts=(0.1 0.01 0.001 0.03 0.003 0.0003 0 1)
for i in {0..7}; do
    shift=${shifts[$i]}
    base_index=$((i * 5))
    if [ $SLURM_ARRAY_TASK_ID -eq $((base_index + 0)) ]; then
        jaxgmg train pile --wandb-log --no-console-log \
            --wandb-entity matthew-farrugia-roberts --wandb-project amp-pile \
            --wandb-name a0-dr_shift$shift \
            --env-size 17 --max-cheese-radius-shift 18 --no-env-terminate-after-pile \
            --num-total-env-steps 400_000_000 --no-env-penalize-time \
            --prob-mutate-shift=$shift --prob-shift=$shift --plr-prob-replay=0.5 \
            --ued dr;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((base_index + 1)) ]; then
        jaxgmg train pile --wandb-log --no-console-log \
            --wandb-entity matthew-farrugia-roberts --wandb-project amp-pile \
            --wandb-name a1-plr_shift$shift \
            --env-size 17 --max-cheese-radius-shift 18 --no-env-terminate-after-pile \
            --num-total-env-steps 800_000_000 --no-env-penalize-time \
            --prob-mutate-shift=$shift --prob-shift=$shift --plr-prob-replay=0.5 \
            --ued plr --plr-regret-estimator maxmc-actor;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((base_index + 2)) ]; then
        jaxgmg train pile --wandb-log --no-console-log \
            --wandb-entity matthew-farrugia-roberts --wandb-project amp-pile \
            --wandb-name a2-accel_$shift \
            --env-size 17 --max-cheese-radius-shift 18 --no-env-terminate-after-pile \
            --num-total-env-steps 800_000_000 --no-env-penalize-time \
            --prob-mutate-shift=$shift --prob-shift=$shift --plr-prob-replay=0.8 \
            --ued accel --plr-regret-estimator maxmc-actor;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((base_index + 3)) ]; then
        jaxgmg train pile --wandb-log --no-console-log \
            --wandb-entity matthew-farrugia-roberts --wandb-project amp-pile \
            --wandb-name a3-plr+proxy_shift$shift \
            --env-size 17 --max-cheese-radius-shift 18 --no-env-terminate-after-pile \
            --num-total-env-steps 800_000_000 --no-env-penalize-time \
            --prob-mutate-shift=$shift --prob-shift=$shift --plr-prob-replay=0.5 \
            --ued plr --plr-proxy-shaping --plr-regret-estimator maxmc-actor;
    elif [ $SLURM_ARRAY_TASK_ID -eq $((base_index + 4)) ]; then
        jaxgmg train pile --wandb-log --no-console-log \
            --wandb-entity matthew-farrugia-roberts --wandb-project amp-pile \
            --wandb-name a4-accel+proxy_shift$shift \
            --env-size 17 --max-cheese-radius-shift 18 --no-env-terminate-after-pile \
            --num-total-env-steps 800_000_000 --no-env-penalize-time \
            --prob-mutate-shift=$shift --prob-shift=$shift --plr-prob-replay=0.8 \
            --ued accel --plr-proxy-shaping --plr-regret-estimator maxmc-actor;
    fi
done
deactivate

