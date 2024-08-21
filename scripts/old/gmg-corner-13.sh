# original level distribution (fixed train levels)
jaxgmg train corner --wandb-log --wandb-project gmg-corner-13 --wandb-name cs01-lv2k  --net impala:lstm --env-size 13 --env-corner-size 1  --fixed-train-levels --num-train-levels 2048
jaxgmg train corner --wandb-log --wandb-project gmg-corner-13 --wandb-name cs02-lv2k  --net impala:lstm --env-size 13 --env-corner-size 2  --fixed-train-levels --num-train-levels 2048 
jaxgmg train corner --wandb-log --wandb-project gmg-corner-13 --wandb-name cs03-lv2k  --net impala:lstm --env-size 13 --env-corner-size 3  --fixed-train-levels --num-train-levels 2048 
jaxgmg train corner --wandb-log --wandb-project gmg-corner-13 --wandb-name cs04-lv2k  --net impala:lstm --env-size 13 --env-corner-size 4  --fixed-train-levels --num-train-levels 2048 
jaxgmg train corner --wandb-log --wandb-project gmg-corner-13 --wandb-name cs05-lv2k  --net impala:lstm --env-size 13 --env-corner-size 5  --fixed-train-levels --num-train-levels 2048 
jaxgmg train corner --wandb-log --wandb-project gmg-corner-13 --wandb-name cs06-lv2k  --net impala:lstm --env-size 13 --env-corner-size 6  --fixed-train-levels --num-train-levels 2048 
jaxgmg train corner --wandb-log --wandb-project gmg-corner-13 --wandb-name cs07-lv2k  --net impala:lstm --env-size 13 --env-corner-size 7  --fixed-train-levels --num-train-levels 2048 
jaxgmg train corner --wandb-log --wandb-project gmg-corner-13 --wandb-name cs08-lv2k  --net impala:lstm --env-size 13 --env-corner-size 8  --fixed-train-levels --num-train-levels 2048 
jaxgmg train corner --wandb-log --wandb-project gmg-corner-13 --wandb-name cs09-lv2k  --net impala:lstm --env-size 13 --env-corner-size 9  --fixed-train-levels --num-train-levels 2048 
jaxgmg train corner --wandb-log --wandb-project gmg-corner-13 --wandb-name cs10-lv2k  --net impala:lstm --env-size 13 --env-corner-size 10 --fixed-train-levels --num-train-levels 2048
jaxgmg train corner --wandb-log --wandb-project gmg-corner-13 --wandb-name cs11-lv2k  --net impala:lstm --env-size 13 --env-corner-size 11 --fixed-train-levels --num-train-levels 2048

# infinite level distribution
jaxgmg train corner --wandb-log --wandb-project gmg-corner-13 --wandb-name cs01-lvinf --net impala:lstm --env-size 13 --env-corner-size 1  --no-fixed-train-levels
jaxgmg train corner --wandb-log --wandb-project gmg-corner-13 --wandb-name cs02-lvinf --net impala:lstm --env-size 13 --env-corner-size 2  --no-fixed-train-levels 
jaxgmg train corner --wandb-log --wandb-project gmg-corner-13 --wandb-name cs03-lvinf --net impala:lstm --env-size 13 --env-corner-size 3  --no-fixed-train-levels 
jaxgmg train corner --wandb-log --wandb-project gmg-corner-13 --wandb-name cs04-lvinf --net impala:lstm --env-size 13 --env-corner-size 4  --no-fixed-train-levels 
jaxgmg train corner --wandb-log --wandb-project gmg-corner-13 --wandb-name cs05-lvinf --net impala:lstm --env-size 13 --env-corner-size 5  --no-fixed-train-levels 
jaxgmg train corner --wandb-log --wandb-project gmg-corner-13 --wandb-name cs06-lvinf --net impala:lstm --env-size 13 --env-corner-size 6  --no-fixed-train-levels 
jaxgmg train corner --wandb-log --wandb-project gmg-corner-13 --wandb-name cs07-lvinf --net impala:lstm --env-size 13 --env-corner-size 7  --no-fixed-train-levels 
jaxgmg train corner --wandb-log --wandb-project gmg-corner-13 --wandb-name cs08-lvinf --net impala:lstm --env-size 13 --env-corner-size 8  --no-fixed-train-levels 
jaxgmg train corner --wandb-log --wandb-project gmg-corner-13 --wandb-name cs09-lvinf --net impala:lstm --env-size 13 --env-corner-size 9  --no-fixed-train-levels 
jaxgmg train corner --wandb-log --wandb-project gmg-corner-13 --wandb-name cs10-lvinf --net impala:lstm --env-size 13 --env-corner-size 10 --no-fixed-train-levels
jaxgmg train corner --wandb-log --wandb-project gmg-corner-13 --wandb-name cs11-lvinf --net impala:lstm --env-size 13 --env-corner-size 11 --no-fixed-train-levels

