#!/bin/bash
# Script to run jaxgmg train with d6fferent training levels for each corner size

# Specify the array of training levels
train_levels=(1 16 256 2048 4096 16384)

# Loop through each corner size from 1 to 11
for corner_size in {1..11}
do
    # Loop through each specified training level
    for level in "${train_levels[@]}"
    do
        # Running training for current corner size and training level
        jaxgmg train --env-corner-size=$corner_size --num-train-levels=$level --wandb-name="Bugfree13x13_Corner_${corner_size}_Levels_$level"
    done
done
