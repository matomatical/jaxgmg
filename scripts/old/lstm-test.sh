# RELU NETS (FF)
jaxgmg train corner --wandb-log --wandb-project gmg-nets --wandb-name relu-3-128 --env-size 9 --env-corner-size 1 --net relu:3x128  --env-terminate-after-corner
jaxgmg train corner --wandb-log --wandb-project gmg-nets --wandb-name relu-3-256 --env-size 9 --env-corner-size 1 --net relu:3x256  --env-terminate-after-corner
jaxgmg train corner --wandb-log --wandb-project gmg-nets --wandb-name relu-3-512 --env-size 9 --env-corner-size 1 --net relu:3x512  --env-terminate-after-corner
jaxgmg train corner --wandb-log --wandb-project gmg-nets --wandb-name relu-4-256 --env-size 9 --env-corner-size 1 --net relu:4x256  --env-terminate-after-corner
jaxgmg train corner --wandb-log --wandb-project gmg-nets --wandb-name relu-4-512 --env-size 9 --env-corner-size 1 --net relu:4x512  --env-terminate-after-corner
jaxgmg train corner --wandb-log --wandb-project gmg-nets --wandb-name relu-4-1k0 --env-size 9 --env-corner-size 1 --net relu:4x1024 --env-terminate-after-corner
jaxgmg train corner --wandb-log --wandb-project gmg-nets --wandb-name relu-8-1k0 --env-size 9 --env-corner-size 1 --net relu:8x1024 --env-terminate-after-corner

# IMPALA (FF AND LSTM)
jaxgmg train corner --wandb-log --wandb-project gmg-nets --wandb-name impala-small-ff   --env-size 9 --env-corner-size 1 --net impala:small:ff   --env-terminate-after-corner
jaxgmg train corner --wandb-log --wandb-project gmg-nets --wandb-name impala-large-ff   --env-size 9 --env-corner-size 1 --net impala:ff         --env-terminate-after-corner
jaxgmg train corner --wandb-log --wandb-project gmg-nets --wandb-name impala-small-lstm --env-size 9 --env-corner-size 1 --net impala:small:lstm --env-terminate-after-corner
jaxgmg train corner --wandb-log --wandb-project gmg-nets --wandb-name impala-large-lstm --env-size 9 --env-corner-size 1 --net impala:lstm       --env-terminate-after-corner
