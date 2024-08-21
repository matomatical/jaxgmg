# first test to see if ff outperforms lstm generally
jaxgmg train memory-test \
    --wandb-log --wandb-project lstm-fix --wandb-name relu-ff \
    --net relu
jaxgmg train memory-test \
    --wandb-log --wandb-project lstm-fix --wandb-name relu-lstm \
    --net relu:lstm
jaxgmg train memory-test \
    --wandb-log --wandb-project lstm-fix --wandb-name impala-large-ff \
    --net impala:ff
jaxgmg train memory-test \
    --wandb-log --wandb-project lstm-fix --wandb-name impala-large-lstm \
    --net impala:lstm
# it seems yes but it doesn't do the optimal memory-based policy
# training often doesnt get off the ground or gets stuck in local min


# see if multiple seeds or less entropy regulaisation helps
for seed in 0 1 2 3; do
    for entropycoeff in 0.001 0.0001 0.00001; do
        jaxgmg train memory-test --no-console-log \
            --wandb-log --wandb-project lstm-fix-2 \
            --net relu --seed $seed --ppo-entropy-coeff $entropycoeff;
        jaxgmg train memory-test --no-console-log \
            --wandb-log --wandb-project lstm-fix-2 \
            --net relu:lstm --seed $seed --ppo-entropy-coeff $entropycoeff;
    done
done
# entropy didn't make a difference
# lstm one seems to reasonably-robustly outperform the others, but still both
# fail often


# now i 'fixed' the training loop, let's see if it can remember stuff now...
for seed in 0 1 2 3; do
    jaxgmg train memory-test --no-console-log \
        --wandb-log --wandb-project lstm-fix-3 \
        --net relu --seed $seed;
    jaxgmg train memory-test --no-console-log \
        --wandb-log --wandb-project lstm-fix-3 \
        --net relu:lstm --seed $seed;
done
# doesn't yet, but seems to train just as well (slower)


# ok I got entropy regularisation backwards actually, let's try that again
for entropycoeff in 0.001 0.01 0.1; do
    jaxgmg train memory-test --no-console-log \
        --wandb-log --wandb-project lstm-fix-2 \
        --net relu --ppo-entropy-coeff $entropycoeff;
    jaxgmg train memory-test --no-console-log \
        --wandb-log --wandb-project lstm-fix-2 \
        --net relu:lstm --ppo-entropy-coeff $entropycoeff;
done
# entropy still didn't make a noticeable difference


# how about learning rate?
for learningrate in 1e-5 3e-5 1e-4 3e-4; do
    jaxgmg train memory-test --no-console-log \
        --wandb-log --wandb-project lstm-fix-4 \
        --net relu:lstm --ppo-lr $learningrate;
done
# larger seems to help

jaxgmg train memory-test --no-console-log \
    --wandb-log --wandb-project lstm-fix-4 \
    --net relu:lstm --ppo-lr 3e-4 --num-total-env-steps 20_000_000;
# didn't help

# this worked!
jaxgmg train memory-test --wandb-log \
    --num-total-env-steps 2000000 \
    --net relu:lstm

# confirm it with multiple experiments...!
for seed in 0 1 2 3 4 5 6 7 8 9; do
    jaxgmg train memory-test --wandb-log --wandb-project lstm-fix-5 \
        --num-total-env-steps 2000000 \
        --net relu:lstm --seed $seed;
    jaxgmg train memory-test --wandb-log --wandb-project lstm-fix-5 \
        --num-total-env-steps 2000000 \
        --net relu:4x128 --seed $seed;
done
# yep about 80% of the time the relu:lstm run finds the expected solution

# now let's try with the new architecture config method and GRU options
# (is it suddenly slower somehow?)
for seed in 0 1 2 3 4; do
    jaxgmg train memory-test --wandb-log --wandb-project lstm-fix-6 \
        --num-total-env-steps 2000000 --seed $seed \
        --net-cnn-type mlp --net-rnn-type ff;
    jaxgmg train memory-test --wandb-log --wandb-project lstm-fix-6 \
        --num-total-env-steps 2000000 --seed $seed \
        --net-cnn-type mlp --net-rnn-type lstm;
    jaxgmg train memory-test --wandb-log --wandb-project lstm-fix-6 \
        --num-total-env-steps 2000000 --seed $seed \
        --net-cnn-type mlp --net-rnn-type gru;
done
