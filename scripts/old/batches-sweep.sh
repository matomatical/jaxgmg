# Try various minibatch sizes alone
jaxgmg train corner \
    --wandb-log --wandb-project batch-size --wandb-name p32-s256-m1-e5  \
    --env-size 9 --env-corner-size 1 --env-terminate-after-corner \
    --net impala:lstm \
    --num-parallel-envs 32 --num-env-steps-per-cycle 256 \
    --num-minibatches-per-epoch 1 --num-epochs-per-cycle 5
jaxgmg train corner \
    --wandb-log --wandb-project batch-size --wandb-name p32-s256-m2-e5  \
    --env-size 9 --env-corner-size 1 --env-terminate-after-corner \
    --net impala:lstm \
    --num-parallel-envs 32 --num-env-steps-per-cycle 256 \
    --num-minibatches-per-epoch 2 --num-epochs-per-cycle 5
jaxgmg train corner \
    --wandb-log --wandb-project batch-size --wandb-name p32-s256-m4-e5  \
    --env-size 9 --env-corner-size 1 --env-terminate-after-corner \
    --net impala:lstm \
    --num-parallel-envs 32 --num-env-steps-per-cycle 256 \
    --num-minibatches-per-epoch 4 --num-epochs-per-cycle 5
jaxgmg train corner \
    --wandb-log --wandb-project batch-size --wandb-name p32-s256-m8-e5  \
    --env-size 9 --env-corner-size 1 --env-terminate-after-corner \
    --net impala:lstm \
    --num-parallel-envs 32 --num-env-steps-per-cycle 256 \
    --num-minibatches-per-epoch 8 --num-epochs-per-cycle 5
jaxgmg train corner \
    --wandb-log --wandb-project batch-size --wandb-name p32-s256-m16-e5  \
    --env-size 9 --env-corner-size 1 --env-terminate-after-corner \
    --net impala:lstm \
    --num-parallel-envs 32 --num-env-steps-per-cycle 256 \
    --num-minibatches-per-epoch 16 --num-epochs-per-cycle 5

# Try more minibatches with more parallel envs
jaxgmg train corner \
    --wandb-log --wandb-project batch-size --wandb-name p64-s128-m4-e5  \
    --env-size 9 --env-corner-size 1 --env-terminate-after-corner \
    --net impala:lstm \
    --num-parallel-envs 64 --num-env-steps-per-cycle 128 \
    --num-minibatches-per-epoch 4 --num-epochs-per-cycle 5
jaxgmg train corner \
    --wandb-log --wandb-project batch-size --wandb-name p128-s128-m4-e5  \
    --env-size 9 --env-corner-size 1 --env-terminate-after-corner \
    --net impala:lstm \
    --num-parallel-envs 128 --num-env-steps-per-cycle 128 \
    --num-minibatches-per-epoch 4 --num-epochs-per-cycle 5
jaxgmg train corner \
    --wandb-log --wandb-project batch-size --wandb-name p128-s128-m8-e5  \
    --env-size 9 --env-corner-size 1 --env-terminate-after-corner \
    --net impala:lstm \
    --num-parallel-envs 128 --num-env-steps-per-cycle 128 \
    --num-minibatches-per-epoch 8 --num-epochs-per-cycle 5
jaxgmg train corner \
    --wandb-log --wandb-project batch-size --wandb-name p256-s128-m4-e5  \
    --env-size 9 --env-corner-size 1 --env-terminate-after-corner \
    --net impala:lstm \
    --num-parallel-envs 256 --num-env-steps-per-cycle 128 \
    --num-minibatches-per-epoch 4 --num-epochs-per-cycle 5
jaxgmg train corner \
    --wandb-log --wandb-project batch-size --wandb-name p256-s128-m8-e5  \
    --env-size 9 --env-corner-size 1 --env-terminate-after-corner \
    --net impala:lstm \
    --num-parallel-envs 256 --num-env-steps-per-cycle 128 \
    --num-minibatches-per-epoch 8 --num-epochs-per-cycle 5


# That worked well I guess, try more
jaxgmg train corner \
    --wandb-log --wandb-project batch-size --wandb-name b4k-c01  \
    --env-size 9 --env-corner-size  1 --net impala:lstm \
    --num-parallel-envs 256 --num-env-steps-per-cycle 128 \
    --num-minibatches-per-epoch 8 --num-epochs-per-cycle 5
jaxgmg train corner \
    --wandb-log --wandb-project batch-size --wandb-name b4k-c02  \
    --env-size 9 --env-corner-size  2 --net impala:lstm \
    --num-parallel-envs 256 --num-env-steps-per-cycle 128 \
    --num-minibatches-per-epoch 8 --num-epochs-per-cycle 5
jaxgmg train corner \
    --wandb-log --wandb-project batch-size --wandb-name b4k-c03  \
    --env-size 9 --env-corner-size  3 --net impala:lstm \
    --num-parallel-envs 256 --num-env-steps-per-cycle 128 \
    --num-minibatches-per-epoch 8 --num-epochs-per-cycle 5
jaxgmg train corner \
    --wandb-log --wandb-project batch-size --wandb-name b4k-c04  \
    --env-size 9 --env-corner-size  4 --net impala:lstm \
    --num-parallel-envs 256 --num-env-steps-per-cycle 128 \
    --num-minibatches-per-epoch 8 --num-epochs-per-cycle 5
jaxgmg train corner \
    --wandb-log --wandb-project batch-size --wandb-name b4k-c05  \
    --env-size 9 --env-corner-size  5 --net impala:lstm \
    --num-parallel-envs 256 --num-env-steps-per-cycle 128 \
    --num-minibatches-per-epoch 8 --num-epochs-per-cycle 5
jaxgmg train corner \
    --wandb-log --wandb-project batch-size --wandb-name b4k-c06  \
    --env-size 9 --env-corner-size  6 --net impala:lstm \
    --num-parallel-envs 256 --num-env-steps-per-cycle 128 \
    --num-minibatches-per-epoch 8 --num-epochs-per-cycle 5
jaxgmg train corner \
    --wandb-log --wandb-project batch-size --wandb-name b4k-c07  \
    --env-size 9 --env-corner-size  7 --net impala:lstm \
    --num-parallel-envs 256 --num-env-steps-per-cycle 128 \
    --num-minibatches-per-epoch 8 --num-epochs-per-cycle 5

# how about generalisation drift?
jaxgmg train corner \
    --wandb-log --wandb-project batch-size-drift --wandb-name b4k-200M  \
    --env-size 9 --env-corner-size 1 --env-terminate-after-corner \
    --net impala:lstm \
    --num-parallel-envs 256 --num-env-steps-per-cycle 128 \
    --num-minibatches-per-epoch 8 --num-epochs-per-cycle 5 \
    --num-total-env-steps 200000000
