# RELU NETS (FF)
jaxgmg train corner --wandb-log --wandb-project gmg-lstm-test --env-size 9 --env-corner-size 1 --net relu:3x128
jaxgmg train corner --wandb-log --wandb-project gmg-lstm-test --env-size 9 --env-corner-size 1 --net relu:3x256
jaxgmg train corner --wandb-log --wandb-project gmg-lstm-test --env-size 9 --env-corner-size 1 --net relu:3x512
jaxgmg train corner --wandb-log --wandb-project gmg-lstm-test --env-size 9 --env-corner-size 1 --net relu:4x256
jaxgmg train corner --wandb-log --wandb-project gmg-lstm-test --env-size 9 --env-corner-size 1 --net relu:4x512

# IMPALA FF
jaxgmg train corner --wandb-log --wandb-project gmg-lstm-test --env-size 9 --env-corner-size 1 --net impala-small:ff
jaxgmg train corner --wandb-log --wandb-project gmg-lstm-test --env-size 9 --env-corner-size 1 --net impala:ff

# IMPALA LSTM
jaxgmg train corner --wandb-log --wandb-project gmg-lstm-test --env-size 9 --env-corner-size 1 --net impala-small:lstm
jaxgmg train corner --wandb-log --wandb-project gmg-lstm-test --env-size 9 --env-corner-size 1 --net impala:lstm

# long running LSTM test
jaxgmg train corner --wandb-log --wandb-project gmg-lstm-test --env-size 9 --env-corner-size 1 --net impala:lstm --num-total-env-steps 200000000 --env-terminate-after-corner
jaxgmg train corner --wandb-log --wandb-project gmg-lstm-test --env-size 9 --env-corner-size 1 --net impala:ff   --num-total-env-steps 200000000 --env-terminate-after-corner
