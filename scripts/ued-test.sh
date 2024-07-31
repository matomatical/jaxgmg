# plr
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 13 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 2048 --no-train-gifs \
    --wandb-name plr-r.5 --ued plr --plr-prob-replay 0.5 --plr-buffer-size 2048
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 13 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 2048 --no-train-gifs \
    --wandb-name plr-r.8 --ued plr --plr-prob-replay 0.8 --plr-buffer-size 2048
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 13 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 2048 --no-train-gifs \
    --wandb-name plr-8k  --ued plr --plr-prob-replay 0.5 --plr-buffer-size 8192


# dr
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 13 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 2048 --no-train-gifs \
    --wandb-name dr-inf --ued dr
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 13 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 2048 --no-train-gifs \
    --wandb-name dr-2k --ued dr-finite --num-train-levels 2048
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 13 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 2048 --no-train-gifs \
    --wandb-name dr-8k --ued dr-finite --num-train-levels 8192


# plr basic config but with different seeds
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 13 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 2048 --no-train-gifs \
    --wandb-name plr-r.5a --seed 0 --ued plr --plr-prob-replay 0.5 --plr-buffer-size 2048
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 13 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 2048 --no-train-gifs \
    --wandb-name plr-r.5b --seed 1 --ued plr --plr-prob-replay 0.5 --plr-buffer-size 2048
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 13 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 2048 --no-train-gifs \
    --wandb-name plr-r.5c --seed 2 --ued plr --plr-prob-replay 0.5 --plr-buffer-size 2048
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 13 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 2048 --no-train-gifs \
    --wandb-name plr-r.5d --seed 3 --ued plr --plr-prob-replay 0.5 --plr-buffer-size 2048
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 13 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 2048 --no-train-gifs \
    --wandb-name plr-r.5e --seed 4 --ued plr --plr-prob-replay 0.5 --plr-buffer-size 2048


# maybe the plr buffer is way too big? try a way SMALLER buffer?
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 13 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 2048 --no-train-gifs \
    --wandb-name plr-1k  --ued plr --plr-prob-replay 0.5 --plr-buffer-size 1024
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 13 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 2048 --no-train-gifs \
    --wandb-name plr-512  --ued plr --plr-prob-replay 0.5 --plr-buffer-size 512


# tree
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 13 --env-corner-size 1 --env-layout tree --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 2048 --no-train-gifs \
    --wandb-name plr-tree --ued plr --plr-prob-replay 0.5 --plr-buffer-size 2048
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 13 --env-corner-size 1 --env-layout tree --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 2048 --no-train-gifs \
    --wandb-name dr-tree --ued dr


# dr basic config with different seeds
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 13 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 2048 --no-train-gifs \
    --wandb-name dr-inf-a --seed 0 --ued dr
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 13 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 2048 --no-train-gifs \
    --wandb-name dr-inf-b --seed 1 --ued dr
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 13 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 2048 --no-train-gifs \
    --wandb-name dr-inf-c --seed 2 --ued dr
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 13 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 2048 --no-train-gifs \
    --wandb-name dr-inf-d --seed 3 --ued dr
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 13 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 2048 --no-train-gifs \
    --wandb-name dr-inf-e --seed 4 --ued dr


# OK try PLR vs DR again but for longer training times
for i in 6 7 8 9 10; do
    jaxgmg train corner \
        --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
        --env-size 17 --env-corner-size 1 --net impala:lstm --num-total-env-steps 500000000 \
        --num-cycles-per-log 128 --num-cycles-per-eval 128 --num-cycles-per-big-eval 10000 --no-train-gifs \
        --wandb-name plr-long --seed "$i" --ued plr --plr-prob-replay 0.5 --plr-buffer-size 2048;
    jaxgmg train corner \
        --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
        --env-size 17 --env-corner-size 1 --net impala:lstm --num-total-env-steps 500000000 \
        --num-cycles-per-log 128 --num-cycles-per-eval 128 --num-cycles-per-big-eval 10000 --no-train-gifs \
        --wandb-name dr-long --seed "$i" --ued dr;
done


# did: solve levels to get good baselines...
# -> it's slow and they don't make good metrics
    

# try plr on mixture distributions
# 0% shift
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 17 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 10000 --train-gifs \
    --wandb-name corner0-dr   --ued dr  --prob-shift 0.00;
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 17 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 10000 --train-gifs \
    --wandb-name corner0-plr  --ued plr --prob-shift 0.00;
# 1% shift
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 17 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 10000 --train-gifs \
    --wandb-name corner1-dr   --ued dr  --prob-shift 0.01;
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 17 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 10000 --train-gifs \
    --wandb-name corner1-plr  --ued plr --prob-shift 0.01;
# 3% shift
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 17 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 10000 --train-gifs \
    --wandb-name corner3-dr  --ued dr  --prob-shift 0.03;
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 17 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 10000 --train-gifs \
    --wandb-name corner3-plr --ued plr --prob-shift 0.03;
# 10% shift
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 17 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 10000 --train-gifs \
    --wandb-name corner10-dr  --ued dr  --prob-shift 0.10;
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 17 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 10000 --train-gifs \
    --wandb-name corner10-plr --ued plr --prob-shift 0.10;
# 50% shift
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 17 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 10000 --train-gifs \
    --wandb-name corner50-dr  --ued dr  --prob-shift 0.50;
jaxgmg train corner \
    --no-console-log --wandb-log --wandb-entity krueger-lab-cambridge --wandb-project matt-gmg \
    --env-size 17 --env-corner-size 1 --net impala:lstm --num-total-env-steps 20000000 \
    --num-cycles-per-log 8 --num-cycles-per-eval 8 --num-cycles-per-big-eval 10000 --train-gifs \
    --wandb-name corner50-plr --ued plr --prob-shift 0.50;


# todo:
# * implement robust plr / parallel plr
# * have a go at formalism
