#!/bin/bash
#SBATCH --job-name=eta-schedule
#SBATCH --partition=high_priority
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/data/matthew_farrugia_roberts
#SBATCH --output=out/%A_%a-eta-schedule.stdout
#SBATCH --error=out/%A_%a-eta-schedule.stderr
#SBATCH --array 0-47%24

flags=(
    --no-console-log
    --wandb-log
    --wandb-entity matthew-farrugia-roberts
    --wandb-project eta-schedule
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
    --prob-shift 1e-4
    --prob-mutate-shift 1e-4
    --plr-proxy-shaping
    --num-total-env-steps 750_000_000
);
seed_array=(0 1);
plr_proxy_shaping_coeff_array=(0.3 0.8);
eta_schedule_time_array=(0.4 0.8);

source jaxgmg.venv/bin/activate
for i in {0..1}; do
    seed=${seed_array[$i]}
    for j in {0..1}; do
        eta=${plr_proxy_shaping_coeff_array[$j]}
        for k in {0..1}; do
            time=${eta_schedule_time_array[$k]}
            x=$((i * 2 * 2 * 6 + j * 2 * 6 + k * 6))
            if [ $SLURM_ARRAY_TASK_ID -eq $((x + 0)) ]; then
                jaxgmg train dish "${flags[@]}" \
                    --wandb-name plr_prox_clip-eta:$eta-time:$time-seed:$seed \
                    --seed $seed \
                    --ued plr --plr-prob-replay 0.33 \
                    --eta-schedule --eta-schedule-time $time \
                    --plr-proxy-shaping-coeff $eta --clipping;
            elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 1)) ]; then
                jaxgmg train dish "${flags[@]}" \
                    --wandb-name plr_prox_noclip-eta:$eta-time:$time-seed:$seed \
                    --seed $seed \
                    --ued plr --plr-prob-replay 0.33 \
                    --eta-schedule --eta-schedule-time $time \
                    --plr-proxy-shaping-coeff $eta --no-clipping;
            elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 2)) ]; then
                jaxgmg train dish "${flags[@]}" \
                    --wandb-name accelc_prox_clip-eta:$eta-time:$time-seed:$seed \
                    --seed $seed \
                    --ued accel --plr-prob-replay 0.5 \
                    --eta-schedule --eta-schedule-time $time \
                    --plr-proxy-shaping-coeff $eta --clipping \
                    --chain-mutate --mutate-cheese-on-dish;
            elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 3)) ]; then
                jaxgmg train dish "${flags[@]}" \
                    --wandb-name accelc_prox_noclip-eta:$eta-time:$time-seed:$seed \
                    --seed $seed \
                    --ued accel --plr-prob-replay 0.5 \
                    --eta-schedule --eta-schedule-time $time \
                    --plr-proxy-shaping-coeff $eta --no-clipping \
                    --chain-mutate --mutate-cheese-on-dish;
            elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 4)) ]; then
                jaxgmg train dish "${flags[@]}" \
                    --wandb-name accelb_prox_clip-eta:$eta-time:$time-seed:$seed \
                    --seed $seed \
                    --ued accel --plr-prob-replay 0.5 \
                    --eta-schedule --eta-schedule-time $time \
                    --plr-proxy-shaping-coeff $eta --clipping \
                    --no-chain-mutate --mutate-cheese-on-dish;
            elif [ $SLURM_ARRAY_TASK_ID -eq $((x + 5)) ]; then
                jaxgmg train dish "${flags[@]}" \
                    --wandb-name accelb_prox_noclip-eta:$eta-time:$time-seed:$seed \
                    --seed $seed \
                    --ued accel --plr-prob-replay 0.5 \
                    --eta-schedule --eta-schedule-time $time \
                    --plr-proxy-shaping-coeff $eta --no-clipping \
                    --no-chain-mutate --mutate-cheese-on-dish;
            fi
        done
    done
done
deactivate

