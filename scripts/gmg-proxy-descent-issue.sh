jaxgmg train corner --net impala --wandb-log --wandb-project descent --wandb-name inf-levels  --env-size 13 --env-corner-size 1 --num-total-env-steps 200000000 --no-fixed-train-levels
jaxgmg train corner --net impala --wandb-log --wandb-project descent --wandb-name 2k-levels  --env-size 13 --env-corner-size 1 --num-total-env-steps 200000000 --fixed-train-levels --num-train-levels 2048
