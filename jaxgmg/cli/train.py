"""
Launcher for training runs.
"""

import jax
import jax.numpy as jnp

from jaxgmg.procgen import maze_generation
from jaxgmg.environments import cheese_in_the_corner
from jaxgmg.environments import cheese_on_a_dish
from jaxgmg.environments import keys_and_chests
from jaxgmg.environments import minigrid_maze
from jaxgmg.environments import lava_land
from jaxgmg.environments import follow_me
from jaxgmg.baselines import train
from jaxgmg.baselines.config import TrainConfig
from jaxgmg.cli import eval_levels

from jaxgmg.environments.base import Level
from jaxgmg.environments.base import MixtureLevelGenerator
from jaxgmg.environments.base import MixtureLevelMutator, IdentityLevelMutator
from jaxgmg.environments.base import ChainLevelMutator, IteratedLevelMutator

from jaxgmg import util


@util.wandb_run
def corner(
    # environment config
    env_size: int = 13,
    env_layout: str = 'blocks',
    env_corner_size: int = 1,
    env_terminate_after_corner: bool = False,
    obs_level_of_detail: int = 0,           # 0 = bool; 1, 3, 4, or 8 = rgb
    img_level_of_detail: int = 1,           # obs_ is for train, img_ for gifs
    env_penalize_time: bool = False,
    # policy config
    net_cnn_type: str = "large",
    net_rnn_type: str = "ff",
    net_width: int = 256,
    # ued config
    ued: str = "plr",                       # dr, dr-finite, plr, accel
    prob_shift: float = 0.0,
    num_train_levels: int = 2048,
    # for plr
    plr_buffer_size: int = 4096,
    plr_temperature: float = 0.1,
    plr_staleness_coeff: float = 0.1,
    plr_prob_replay: float = 0.5, #default 0.5
    plr_regret_estimator: str = "maxmc-actor",
    plr_robust: bool = False,
    # for accel
    num_mutate_steps: int = 12,
    prob_mutate_shift: float = 0.0,
    chain_mutate: bool = True,
    mutate_cheese: bool = True,
    # for proxy augmented methods
    # PPO hyperparameters
    ppo_lr: float = 0.00005,                # learning rate
    ppo_gamma: float = 0.999,               # discount rate
    ppo_clip_eps: float = 0.1,
    ppo_gae_lambda: float = 0.95,
    ppo_entropy_coeff: float = 0.001,
    ppo_critic_coeff: float = 0.5,
    ppo_max_grad_norm: float = 0.5,
    ppo_lr_annealing: bool = False,
    num_minibatches_per_epoch: int = 4,
    num_epochs_per_cycle: int = 5,
    # training dimensions
    num_total_env_steps: int = 20_000_000,
    num_env_steps_per_cycle: int = 128,
    num_parallel_envs: int = 256,
    # logging and evals config
    console_log: bool = True,
    wandb_log: bool = True,
    wandb_project: str = "test",
    wandb_entity: str = None,               # e.g. 'krueger-lab-cambridge'
    wandb_group: str = None,
    wandb_name: str = None,
    log_gifs: bool = False,
    log_imgs: bool = True,
    log_hists: bool = False,
    num_cycles_per_log: int = 32,           #   32 * 32k = roughly  1M steps
    num_cycles_per_eval: int = 32,          #   32 * 32k = roughly  1M steps
    num_cycles_per_gifs: int = 1024,        # 1024 * 32k = roughly 32M steps
    num_cycles_per_big_eval: int = 1024,    # 1024 * 32k = roughly 32M steps
    evals_num_env_steps: int = 512,
    evals_num_levels: int = 256,
    gif_grid_width: int = 16,
    level_splayer: str = 'mouse',           # or 'cheese' or 'cheese-and-mouse'
    # checkpointing
    checkpointing: bool = True,             # keep checkpoints? (default: yes)
    keep_all_checkpoints: bool = False,     # if so: keep all of them? (no)
    max_num_checkpoints: int = 1,           # if not: keep only latest n (=1)
    num_cycles_per_checkpoint: int = 512,
    # other
    seed: int = 42,
):
    config = locals()
    util.print_config(config)


    print("configuring environment...")
    env = cheese_in_the_corner.Env(
        obs_level_of_detail=obs_level_of_detail,
        img_level_of_detail=img_level_of_detail,
        penalize_time=env_penalize_time,
        terminate_after_cheese_and_corner=env_terminate_after_corner,
    )


    print("configuring level generators...")
    maze_generator = maze_generation.get_generator_class_from_name(
        name=env_layout,
    )()
    orig_level_generator = cheese_in_the_corner.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        corner_size=env_corner_size,
    )
    shift_level_generator = cheese_in_the_corner.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        corner_size=env_size-2,
    )
    tree_level_generator = cheese_in_the_corner.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generation.get_generator_class_from_name(
            name='tree',
        )(),
        corner_size=env_corner_size,
    )
    if prob_shift > 0.0:
        print(f"  mixing level generators with {prob_shift=}...")
        train_level_generator = MixtureLevelGenerator(
            level_generator1=orig_level_generator,
            level_generator2=shift_level_generator,
            prob_level1=1.0-prob_shift,
        )
    else:
        train_level_generator = orig_level_generator


    print("configuring level classifier...")
    def classify_level_is_shift(level: Level) -> bool:
        return jnp.logical_or(
            level.cheese_pos[0] != 1,
            level.cheese_pos[1] != 1,
        )


    print("configuring level mutator...")
    # if mutating cheese, mostly stay in the restricted region
    if mutate_cheese:
        biased_cheese_mutator = MixtureLevelMutator(
            mutators=(
                # teleport cheese to the corner or do not move the cheese
                cheese_in_the_corner.CornerCheeseLevelMutator(
                    corner_size=env_corner_size,
                ),
                # teleport cheese to a random position
                cheese_in_the_corner.ScatterCheeseLevelMutator(),
            ),
            mixing_probs=(1-prob_mutate_shift, prob_mutate_shift),
        )
    else:
        # replace the cheese mutation with something else
        biased_cheese_mutator = cheese_in_the_corner.ToggleWallLevelMutator()
    # overall mutations
    if chain_mutate:
        level_mutator = ChainLevelMutator(mutators=(
            # mutate walls (n-2 steps)
            IteratedLevelMutator(
                mutator=cheese_in_the_corner.ToggleWallLevelMutator(),
                num_steps=num_mutate_steps - 2,
            ),
            # maybe scatter mouse (1 step) else another wall toggle
            MixtureLevelMutator(
                mutators=(
                    cheese_in_the_corner.ScatterMouseLevelMutator(
                        transpose_with_cheese_on_collision=False,
                    ),
                    cheese_in_the_corner.ToggleWallLevelMutator(),
                ),
                mixing_probs=(1/2,1/2),
            ),
            # biased scatter cheese (1 step)
            biased_cheese_mutator,
        ))
    else:
        level_mutator = IteratedLevelMutator(
            mutator=MixtureLevelMutator(
                mutators=(
                    cheese_in_the_corner.ToggleWallLevelMutator(),
                    cheese_in_the_corner.ScatterMouseLevelMutator(
                        transpose_with_cheese_on_collision=False,
                    ),
                    biased_cheese_mutator,
                ),
                mixing_probs=(
                    (num_mutate_steps - 2) / num_mutate_steps,
                    1 / num_mutate_steps,
                    1 / num_mutate_steps,
                ),
            ),
            num_steps=num_mutate_steps,
        )


    print("configuring level solver...")
    level_solver = cheese_in_the_corner.LevelSolver(
        env=env,
        discount_rate=ppo_gamma,
    )


    print("configuring level metrics...")
    level_metrics = cheese_in_the_corner.LevelMetrics(
        env=env,
        discount_rate=ppo_gamma,
    )


    print("configuring eval level generators...")
    if prob_shift > 0.0:
        eval_level_generators = {
            "train": train_level_generator,
            "orig": orig_level_generator,
            "shift": shift_level_generator,
            "tree": tree_level_generator,
        }
    else:
        eval_level_generators = {
            "orig": orig_level_generator,
            "shift": shift_level_generator,
            "tree": tree_level_generator,
        }


    print("configuring parser and parsing fixed eval levels...")
    fixed_eval_levels = eval_levels.corner(env_size)


    print("configuring heatmap splayer...")
    splayer_fn = cheese_in_the_corner.splayer_from_name(level_splayer)


    train.run(
        TrainConfig.from_cli(locals()),
        env=env,
        train_level_generator=train_level_generator,
        level_mutator=level_mutator,
        level_solver=level_solver,
        level_metrics=level_metrics,
        eval_level_generators=eval_level_generators,
        fixed_eval_levels=fixed_eval_levels,
        heatmap_splayer_fn=splayer_fn,
        classify_level_is_shift=classify_level_is_shift,
    )


@util.wandb_run
def dish(
    # environment config
    env_size: int = 13,
    env_layout: str = 'blocks',
    env_terminate_after_dish: bool = False,
    num_channels_cheese: int = 1,           # (bool only) num redundant cheese channels
    num_channels_dish: int = 1,             # (bool only) num redundant dish channels
    obs_level_of_detail: int = 0,           # 0 = bool; 1, 3, 4, or 8 = rgb
    img_level_of_detail: int = 1,           # obs_ is for train, img_ for gifs
    env_penalize_time: bool = False,
    # level generator config
    cheese_on_dish: bool = True,
    cheese_on_dish_shift: bool = False,
    # policy config
    net_cnn_type: str = "large",
    net_rnn_type: str = "ff",
    net_width: int = 256,
    # ued config
    ued: str = "plr",
    prob_shift: float = 0.0,
    # for domain randomisation
    num_train_levels: int = 2048,
    # for plr
    plr_buffer_size: int = 4096,
    plr_temperature: float = 0.1,
    plr_staleness_coeff: float = 0.1,
    plr_prob_replay: float = 0.5,
    plr_regret_estimator: str = "maxmc-actor",
    plr_robust: bool = True,
    # for accel
    num_mutate_steps: int = 12,
    prob_mutate_shift: float = 0.0,
    chain_mutate: bool = True,
    mutate_cheese_on_dish: bool = True,
    # for proxy augmented methods
    # PPO hyperparameters
    ppo_lr: float = 0.00005,                # learning rate
    ppo_gamma: float = 0.999,               # discount rate
    ppo_clip_eps: float = 0.1,
    ppo_gae_lambda: float = 0.95,
    ppo_entropy_coeff: float = 0.001,
    ppo_critic_coeff: float = 0.5,
    ppo_max_grad_norm: float = 0.5,
    ppo_lr_annealing: bool = False,
    num_minibatches_per_epoch: int = 4,
    num_epochs_per_cycle: int = 5,
    # training dimensions
    num_total_env_steps: int = 20_000_000,
    num_env_steps_per_cycle: int = 128,
    num_parallel_envs: int = 256,
    # logging and evals config
    console_log: bool = True,
    wandb_log: bool = True,
    wandb_project: str = "test",
    wandb_entity: str = None,               # e.g. 'krueger-lab-cambridge'
    wandb_group: str = None,
    wandb_name: str = None,
    log_gifs: bool = False,
    log_imgs: bool = True,
    log_hists: bool = False,
    num_cycles_per_log: int = 32,           #   32 * 32k = roughly  1M steps
    num_cycles_per_eval: int = 32,          #   32 * 32k = roughly  1M steps
    num_cycles_per_gifs: int = 1024,        # 1024 * 32k = roughly 32M steps
    num_cycles_per_big_eval: int = 1024,    # 1024 * 32k = roughly 32M steps
    evals_num_env_steps: int = 512,
    evals_num_levels: int = 256,
    gif_grid_width: int = 16,
    # checkpointing
    checkpointing: bool = True,             # keep checkpoints? (default: yes)
    keep_all_checkpoints: bool = False,     # if so: keep all of them? (no)
    max_num_checkpoints: int = 1,           # if not: keep only latest n (=1)
    num_cycles_per_checkpoint: int = 512,
    # other
    seed: int = 42,
):
    config = locals()
    util.print_config(config)


    print("configuring environment...")
    if 'oracle' in plr_regret_estimator:
        assert not env_terminate_after_dish, "assumed False hack in scores.py"
    env = cheese_on_a_dish.Env(
        terminate_after_cheese_and_dish=env_terminate_after_dish,
        num_channels_cheese=num_channels_cheese,
        num_channels_dish=num_channels_dish,
        obs_level_of_detail=obs_level_of_detail,
        img_level_of_detail=img_level_of_detail,
        penalize_time=env_penalize_time,
    )


    print("configuring level generators...")
    maze_generator = maze_generation.get_generator_class_from_name(
        name=env_layout,
    )()
    orig_level_generator = cheese_on_a_dish.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        cheese_on_dish=cheese_on_dish,
    )
    shift_level_generator = cheese_on_a_dish.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        cheese_on_dish=cheese_on_dish_shift,  
    )
    if prob_shift > 0.0:
        print(f"  mixing level generators with {prob_shift=}...")
        train_level_generator = MixtureLevelGenerator(
            level_generator1=orig_level_generator,
            level_generator2=shift_level_generator,
            prob_level1=1.0-prob_shift,
        )
    else:
        train_level_generator = orig_level_generator


    print("configuring level classifier...")
    def classify_level_is_shift(level: Level) -> bool:
        return jnp.logical_or(
            level.cheese_pos[0] != level.dish_pos[0],
            level.cheese_pos[1] != level.dish_pos[1],
        )


    print("configuring eval level generators...")
    if prob_shift > 0.0:
        eval_level_generators = {
            "train": train_level_generator,
            "orig": orig_level_generator,
            "shift": shift_level_generator,
        }
    else:
        eval_level_generators = {
            "orig": orig_level_generator,
            "shift": shift_level_generator,
        }


    print("configuring level mutator...")
    if mutate_cheese_on_dish:
        biased_cheese_on_dish_mutator = MixtureLevelMutator(
            mutators=(
                # teleport cheese and dish to new same position
                cheese_on_a_dish.CheeseOnDishLevelMutator(
                    cheese_on_dish=cheese_on_dish,
                ),
                # teleport cheese and dish to new different positions
                cheese_on_a_dish.CheeseOnDishLevelMutator(
                    cheese_on_dish=cheese_on_dish_shift,
                ),
            ),
            mixing_probs=(1-prob_mutate_shift, prob_mutate_shift),
        )
    else:
        # replace this mutation with something else
        biased_cheese_on_dish_mutator = cheese_on_a_dish.ToggleWallLevelMutator()
    # overall
    if chain_mutate:
        level_mutator = ChainLevelMutator(mutators=(
            # mutate walls (n-2 steps)
            IteratedLevelMutator(
                mutator=cheese_on_a_dish.ToggleWallLevelMutator(),
                num_steps=num_mutate_steps - 2,
            ),
            # maybe scatter mouse (1 step) else another wall toggle
            MixtureLevelMutator(
                mutators=(
                    cheese_on_a_dish.ScatterMouseLevelMutator(
                        transpose_with_cheese_on_collision=False,
                        transpose_with_dish_on_collision=False,
                    ),
                    cheese_on_a_dish.ToggleWallLevelMutator(),
                ),
                mixing_probs=(1/2,1/2),
            ),
            # biased reposition cheese/dish (1 step)
            biased_cheese_on_dish_mutator,
        ))
    else:
        # rotate between wall/mouse/cheese mutations uniformly
        level_mutator = IteratedLevelMutator(
            mutator=MixtureLevelMutator(
                mutators=(
                    cheese_on_a_dish.ToggleWallLevelMutator(),
                    cheese_on_a_dish.ScatterMouseLevelMutator(
                        transpose_with_cheese_on_collision=False,
                        transpose_with_dish_on_collision=False,
                    ),
                    biased_cheese_on_dish_mutator,
                ),
                mixing_probs=(
                    (num_mutate_steps - 2) / num_mutate_steps,
                    1 / num_mutate_steps,
                    1 / num_mutate_steps,
                ),
            ),
            num_steps=num_mutate_steps,
        )


    print("configuring level solver...")
    level_solver = cheese_on_a_dish.LevelSolver(
        env=env,
        discount_rate=ppo_gamma,
    )


    print("configuring level metrics...")
    level_metrics = cheese_on_a_dish.LevelMetrics(
        env=env,
        discount_rate=ppo_gamma,
    )


    print("TODO: implement level splayers for heatmap evals...")


    print("configuring parser and parsing fixed eval levels...")
    fixed_eval_levels = eval_levels.dish(env_size)


    train.run(
        TrainConfig.from_cli(locals()),
        env=env,
        train_level_generator=train_level_generator,
        level_mutator=level_mutator,
        level_solver=level_solver,
        level_metrics=level_metrics,
        eval_level_generators=eval_level_generators,
        fixed_eval_levels=fixed_eval_levels,
        heatmap_splayer_fn=None,
        classify_level_is_shift=classify_level_is_shift,
    )


@util.wandb_run
def keys(
    # environment config
    env_size: int = 15,
    env_layout: str = 'blocks',
    env_wall_prob: float = 0.25,
    env_num_keys: int = 3,
    env_num_keys_shift: int = 10,
    env_num_chests: int = 10,
    env_num_chests_shift: int = 3,
    env_baselines: bool = True,             # turn off if too slow
    obs_level_of_detail: int = 0,           # 0 = bool; 1, 3, 4, or 8 = rgb
    img_level_of_detail: int = 1,           # obs_ is for train, img_ for gifs
    env_penalize_time: bool = False,
    #  policy config
    net_cnn_type: str = "large",
    net_rnn_type: str = "ff",
    net_width: int = 256,
    # ued config
    ued: str = "plr",                       # dr, dr-finite, plr, accel
    prob_shift: float = 0.0,
    num_train_levels: int = 2048,
    # for plr
    plr_buffer_size: int = 4096,
    plr_temperature: float = 0.1,
    plr_staleness_coeff: float = 0.1,
    plr_prob_replay: float = 0.5, #default 0.5
    plr_regret_estimator: str = "maxmc-actor",
    plr_robust: bool = False,
    # for accel
    num_mutate_steps: int = 12,
    prob_mutate_shift: float = 0.0,
    chain_mutate: bool = True,
    mutate_keys_ratio: bool = True,
    # for proxy augmented methods
    # PPO hyperparameters
    ppo_lr: float = 0.00005,                # learning rate
    ppo_gamma: float = 0.999,               # discount rate
    ppo_clip_eps: float = 0.1,
    ppo_gae_lambda: float = 0.95,
    ppo_entropy_coeff: float = 0.001,
    ppo_critic_coeff: float = 0.5,
    ppo_max_grad_norm: float = 0.5,
    ppo_lr_annealing: bool = False,
    num_minibatches_per_epoch: int = 4,
    num_epochs_per_cycle: int = 5,
    # training dimensions
    num_total_env_steps: int = 20_000_000,
    num_env_steps_per_cycle: int = 128,
    num_parallel_envs: int = 256,
    # logging and evals config
    console_log: bool = True,
    wandb_log: bool = True,
    wandb_project: str = "keys_demo",
    wandb_entity: str = None,               # e.g. 'krueger-lab-cambridge'
    wandb_group: str = None,
    wandb_name: str = None,
    log_gifs: bool = True,
    log_imgs: bool = True,
    log_hists: bool = False,
    num_cycles_per_log: int = 32,           #   32 * 32k = roughly  1M steps
    num_cycles_per_eval: int = 32,          #   32 * 32k = roughly  1M steps
    num_cycles_per_gifs: int = 1024,        # 1024 * 32k = roughly 32M steps
    num_cycles_per_big_eval: int = 1024,    # 1024 * 32k = roughly 32M steps
    evals_num_env_steps: int = 512,
    evals_num_levels: int = 256,
    gif_grid_width: int = 16,
    # checkpointing
    checkpointing: bool = True,             # keep checkpoints? (default: yes)
    keep_all_checkpoints: bool = False,     # if so: keep all of them? (no)
    max_num_checkpoints: int = 1,           # if not: keep only latest n (=1)
    num_cycles_per_checkpoint: int = 512,
    # other
    seed: int = 42,
):
    config = locals()
    util.print_config(config)


    print("configuring environment...")
    env = keys_and_chests.Env(
        obs_level_of_detail=obs_level_of_detail,
        img_level_of_detail=img_level_of_detail,
        penalize_time=env_penalize_time,
    )


    print("configuring level generators...")
    maze_generator_class = maze_generation.get_generator_class_from_name(
        name=env_layout,
    )
    if env_layout == "blocks":
        maze_generator = maze_generator_class(
            wall_prob=env_wall_prob,
        )
    else:
        maze_generator = maze_generator_class()
    env_num_keys_max = max(env_num_keys, env_num_keys_shift)
    env_num_chests_max = max(env_num_chests, env_num_chests_shift)
    orig_level_generator = keys_and_chests.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        num_keys=env_num_keys,
        num_keys_max=env_num_keys_max,
        num_chests=env_num_chests,
        num_chests_max=env_num_chests_max,
    )
    shift_level_generator = keys_and_chests.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        num_keys=env_num_keys_shift,
        num_keys_max=env_num_keys_max,
        num_chests=env_num_chests_shift,
        num_chests_max=env_num_chests_max,
    )
    if prob_shift > 0.0:
        print(f"  mixing level generators with {prob_shift=}...")
        train_level_generator = MixtureLevelGenerator(
            level_generator1=orig_level_generator,
            level_generator2=shift_level_generator,
            prob_level1=1.0-prob_shift,
        )
    else:
        train_level_generator = orig_level_generator


    print("configuring level classifier...")
    def classify_level_is_shift(level: Level) -> bool:
        num_visible_keys = jnp.sum(~level.hidden_keys)
        num_visible_chests = jnp.sum(~level.hidden_chests)
        return jnp.logical_and(
            num_visible_keys == env_num_keys_shift,
            num_visible_chests == env_num_chests_shift,
        )


    print("configuring eval level generators...")
    if prob_shift > 0.0:
        eval_level_generators = {
            "train": train_level_generator,
            "orig": orig_level_generator,
            "shift": shift_level_generator,
        }
    else:
        eval_level_generators = {
            "orig": orig_level_generator,
            "shift": shift_level_generator,
        }

    print("configuring level mutator...")
    if mutate_keys_ratio:
        biased_keys_mutator = MixtureLevelMutator(
            mutators=(
                # make keys/chests ratio like orig levels
                keys_and_chests.KeysChestsRatioLevelMutator(
                    num_keys=env_num_keys,
                    num_chests=env_num_chests,
                ),
                # make keys/chests ratio like shift levels
                keys_and_chests.KeysChestsRatioLevelMutator(
                    num_keys=env_num_keys_shift,
                    num_chests=env_num_chests_shift,
                ),
            ),
            mixing_probs=(1-prob_mutate_shift, prob_mutate_shift),
        )
    else:
        # replace this mutation with something else
        biased_keys_mutator = keys_and_chests.ToggleWallLevelMutator()
    # overall
    if chain_mutate:
        level_mutator = ChainLevelMutator(mutators=(
            # mutate walls (n-4 steps)
            IteratedLevelMutator(
                mutator=keys_and_chests.ToggleWallLevelMutator(),
                num_steps=num_mutate_steps - 4,
            ),
            # maybe scatter mouse (1 step) else another wall toggle
            MixtureLevelMutator(
                mutators=(
                    keys_and_chests.ScatterMouseLevelMutator(),
                    keys_and_chests.ToggleWallLevelMutator(),
                ),
                mixing_probs=(1/2,1/2),
            ),
            # maybe scatter keys
            MixtureLevelMutator(
                mutators=(
                    keys_and_chests.ScatterKeyLevelMutator(),
                    keys_and_chests.ToggleWallLevelMutator(),
                ),
                mixing_probs=(1/2,1/2),
            ),
            # maybe scatter chest
            MixtureLevelMutator(
                mutators=(
                    keys_and_chests.ScatterChestLevelMutator(),
                    keys_and_chests.ToggleWallLevelMutator(),
                ),
                mixing_probs=(1/2,1/2),
            ),
            # biased scatter keys (1 step)
            biased_keys_mutator,
        ))
    else:
        level_mutator = IteratedLevelMutator(
            mutator=MixtureLevelMutator(
                mutators=(
                    keys_and_chests.ToggleWallLevelMutator(),
                    keys_and_chests.ScatterMouseLevelMutator(),
                    keys_and_chests.ScatterChestLevelMutator(),
                    keys_and_chests.ScatterKeyLevelMutator(),
                    biased_keys_mutator,
                ),
                mixing_probs=(8/12, 1/12, 1/12, 1/12, 1/12),
            ),
            num_steps=num_mutate_steps,
        )


    if env_baselines:
        print("configuring level solver...")
        assert env_num_keys <= env_num_keys_shift
        assert env_num_chests >= env_num_chests_shift
        print("min_keys:", env_num_keys)
        print("min_chests:", env_num_chests_shift)
        level_solver = keys_and_chests.LevelSolverFiltered(
            env=env,
            discount_rate=ppo_gamma,
            min_keys=env_num_keys,
            min_chests=env_num_chests_shift,
        )
    else:
        print("skipping level solver... (set --env-baselines to configure)")
        level_solver = None

    
    if "oracle" in plr_regret_estimator:
        print("assertions guarding the hacks in autocurricula scoring module...")
        assert env_num_keys == 3, "assumed as part of hack"
        assert env_num_chests_shift == 3, "assumed as part of hack"
        assert env.penalize_time == False, "assumed as part of hack"
        assert env.max_steps_in_episode == 128, "assumed as part of hack"


    print("configuring level metrics...")
    level_metrics = keys_and_chests.LevelMetrics(
        env=env,
        discount_rate=ppo_gamma,
    )


    print("TODO: implement level splayers for heatmap evals...")


    print("TODO: configure parser and fixed eval levels...")


    train.run(
        TrainConfig.from_cli(locals()),
        env=env,
        train_level_generator=train_level_generator,
        level_mutator=level_mutator,
        level_solver=level_solver,
        level_metrics=level_metrics,
        eval_level_generators=eval_level_generators,
        fixed_eval_levels={},
        heatmap_splayer_fn=None,
        classify_level_is_shift=classify_level_is_shift,
    )


@util.wandb_run
def minimaze(
    # environment config
    env_size: int = 15,
    env_layout: str = 'noise',
    obs_height: int = 5,
    obs_width: int = 5,
    corner_size: int = 1,
    obs_level_of_detail: int = 0,           # 0 = bool; 1, 3, 4, or 8 = rgb
    img_level_of_detail: int = 1,           # obs_ is for train, img_ for gifs
    env_penalize_time: bool = False,
    # policy config
    net_cnn_type: str = "large",
    net_rnn_type: str = "ff",
    net_width: int = 256,
    # ued config
    ued: str = "plr",                        # dr, dr-finite, plr, accel
    prob_shift: float = 0.0,
    # for domain randomisation
    num_train_levels: int = 2048,
    # for plr
    plr_buffer_size: int = 4096,
    plr_temperature: float = 0.1,
    plr_staleness_coeff: float = 0.1,
    plr_prob_replay: float = 0.5,
    plr_regret_estimator: str = "maxmc-actor",
    plr_robust: bool = True,
    # for accel
    num_mutate_steps: int = 12,
    prob_mutate_shift: float = 0.1,
    # for proxy augmented methods
    # PPO hyperparameters
    ppo_lr: float = 0.00005,                # learning rate
    ppo_gamma: float = 0.999,               # discount rate
    ppo_clip_eps: float = 0.1,
    ppo_gae_lambda: float = 0.95,
    ppo_entropy_coeff: float = 0.001,
    ppo_critic_coeff: float = 0.5,
    ppo_max_grad_norm: float = 0.5,
    ppo_lr_annealing: bool = False,
    num_minibatches_per_epoch: int = 4,
    num_epochs_per_cycle: int = 5,
    # training dimensions
    num_total_env_steps: int = 20_000_000,
    num_env_steps_per_cycle: int = 128,
    num_parallel_envs: int = 256,
    # logging and evals config
    console_log: bool = True,
    wandb_log: bool = True,
    wandb_project: str = "test",
    wandb_entity: str = None,               # e.g. 'krueger-lab-cambridge'
    wandb_group: str = None,
    wandb_name: str = None,
    log_gifs: bool = True,
    log_imgs: bool = True,
    log_hists: bool = False,
    num_cycles_per_log: int = 32,           #   32 * 32k = roughly  1M steps
    num_cycles_per_eval: int = 32,          #   32 * 32k = roughly  1M steps
    num_cycles_per_gifs: int = 1024,        # 1024 * 32k = roughly 32M steps
    num_cycles_per_big_eval: int = 1024,    # 1024 * 32k = roughly 32M steps
    evals_num_env_steps: int = 512,
    evals_num_levels: int = 256,
    gif_grid_width: int = 16,
    # checkpointing
    checkpointing: bool = True,             # keep checkpoints? (default: yes)
    keep_all_checkpoints: bool = False,     # if so: keep all of them? (no)
    max_num_checkpoints: int = 1,           # if not: keep only latest n (=1)
    num_cycles_per_checkpoint: int = 512,
    # other
    seed: int = 42,
):
    config = locals()
    util.print_config(config)


    print("configuring environment...")
    env = minigrid_maze.Env(
        obs_height=obs_height,
        obs_width=obs_width,
        obs_level_of_detail=obs_level_of_detail,
        img_level_of_detail=img_level_of_detail,
        penalize_time=env_penalize_time,
    )


    print("configuring level generators...")
    maze_generator = maze_generation.get_generator_class_from_name(
        name=env_layout,
    )()
    orig_level_generator = minigrid_maze.LevelGenerator(
        maze_generator=maze_generator,
        height=env_size,
        width=env_size,
        corner_size=corner_size,
    )
    shift_level_generator = minigrid_maze.LevelGenerator(
        maze_generator=maze_generator,
        height=env_size,
        width=env_size,
        corner_size=env_size-2,
    )
    if prob_shift > 0.0:
        print(f"  mixing level generators with {prob_shift=}...")
        train_level_generator = MixtureLevelGenerator(
            level_generator1=orig_level_generator,
            level_generator2=shift_level_generator,
            prob_level1=1.0-prob_shift,
        )
    else:
        train_level_generator = orig_level_generator


    print("configuring level classifier...")
    def classify_level_is_shift(level: Level) -> bool:
        return jnp.logical_or(
            level.goal_pos[0] != 1,
            level.goal_pos[1] != 1,
        )


    print("configuring eval level generators...")
    if prob_shift > 0.0:
        eval_level_generators = {
            "train": train_level_generator,
            "orig": orig_level_generator,
            "shift": shift_level_generator,
        }
    else:
        eval_level_generators = {
            "orig": orig_level_generator,
            "shift": shift_level_generator,
        }


    print("configuring level mutator...")
    biased_goal_mutator = MixtureLevelMutator(
        mutators=(
            # teleport goal to the corner
            minigrid_maze.CornerGoalLevelMutator(
                corner_size=corner_size,
            ),
            # teleport cgoal to a random position
            minigrid_maze.CornerGoalLevelMutator(
                corner_size=env_size-2,
            ),
        ),
        mixing_probs=(1-prob_mutate_shift, prob_mutate_shift),
    )
    # overall, rotate between wall/hero/goal mutations uniformly
    level_mutator = IteratedLevelMutator(
        mutator=MixtureLevelMutator(
            mutators=(
                minigrid_maze.ToggleWallLevelMutator(),
                minigrid_maze.ScatterAndSpinHeroLevelMutator(
                    transpose_with_goal_on_collision=False,
                ),
                biased_goal_mutator,
            ),
            mixing_probs=(10/12,1/12,1/12),
        ),
        num_steps=num_mutate_steps,
    )


    print("TODO: implement level solver...")


    print("configuring level metrics...")
    level_metrics = minigrid_maze.LevelMetrics(
        env=env,
        discount_rate=ppo_gamma,
    )


    print("TODO: implement level splayers for heatmap evals...")


    print("configuring parser and parsing fixed eval levels...")
    fixed_eval_levels = eval_levels.minimaze()


    train.run(
        TrainConfig.from_cli(locals()),
        env=env,
        train_level_generator=train_level_generator,
        level_mutator=level_mutator,
        level_solver=None,
        level_metrics=level_metrics,
        eval_level_generators=eval_level_generators,
        fixed_eval_levels=fixed_eval_levels,
        heatmap_splayer_fn=None,
        classify_level_is_shift=classify_level_is_shift,
    )


@util.wandb_run
def memory_test(
    # environment config
    env_size: int = 6,
    obs_height: int = 3,
    obs_width: int = 3,
    obs_level_of_detail: int = 0,
    img_level_of_detail: int = 1,
    env_penalize_time: bool = True,
    # policy config
    net_cnn_type: str = "mlp",
    net_rnn_type: str = "ff",
    net_width: int = 64,
    # curriculum
    ued: str = "dr",
    plr_buffer_size: int = 4096,
    plr_temperature: float = 0.1,
    plr_staleness_coeff: float = 0.1,
    plr_prob_replay: float = 0.5,
    plr_regret_estimator: str = "maxmc-actor",
    plr_robust: bool = True,
    # proxy augmentation
    # PPO hyperparameters
    ppo_lr: float = 0.00005,                # learning rate
    ppo_gamma: float = 0.999,               # discount rate
    ppo_clip_eps: float = 0.1,
    ppo_gae_lambda: float = 0.95,
    ppo_entropy_coeff: float = 0.001,
    ppo_critic_coeff: float = 0.5,
    ppo_max_grad_norm: float = 0.5,
    ppo_lr_annealing: bool = False,
    num_minibatches_per_epoch: int = 4,
    num_epochs_per_cycle: int = 5,
    # training dimensions
    num_total_env_steps: int = 1000_000,
    num_env_steps_per_cycle: int = 64,
    num_parallel_envs: int = 64,
    # logging and evals config
    console_log: bool = True,
    wandb_log: bool = True,
    wandb_project: str = "test",
    wandb_entity: str = None,               # e.g. 'krueger-lab-cambridge'
    wandb_group: str = None,
    wandb_name: str = None,
    log_gifs: bool = True,
    log_imgs: bool = True,
    log_hists: bool = False,
    num_cycles_per_log: int = 32,           #   32 * 32k = roughly  1M steps
    num_cycles_per_eval: int = 32,          #   32 * 32k = roughly  1M steps
    num_cycles_per_gifs: int = 1024,        # 1024 * 32k = roughly 32M steps
    num_cycles_per_big_eval: int = 1024,    # 1024 * 32k = roughly 32M steps
    evals_num_env_steps: int = 512,
    evals_num_levels: int = 256,
    gif_grid_width: int = 16,
    # checkpointing
    checkpointing: bool = False,
    keep_all_checkpoints: bool = False,
    max_num_checkpoints: int = 1,
    num_cycles_per_checkpoint: int = 512,
    # other
    seed: int = 42,
):
    config = locals()
    util.print_config(config)


    print("configuring environment...")
    env = minigrid_maze.Env(
        obs_height=obs_height,
        obs_width=obs_width,
        obs_level_of_detail=obs_level_of_detail,
        img_level_of_detail=img_level_of_detail,
        penalize_time=env_penalize_time,
    )


    print("configuring level generators...")
    orig_level_generator = minigrid_maze.MemoryTestLevelGenerator()


    train.run(
        TrainConfig.from_cli(locals()),
        env=env,
        train_level_generator=orig_level_generator,
        level_mutator=None,
        level_solver=None,
        level_metrics=None,
        eval_level_generators={},
        fixed_eval_levels={},
        heatmap_splayer_fn=None,
        classify_level_is_shift=None,
    )


@util.wandb_run
def follow(
    # environment config
    env_size: int = 15,
    env_layout: str = 'blocks',
    num_beacons: int = 6,
    trustworthy_leader: bool = True,
    trustworthy_leader_shift: bool = False,
    obs_level_of_detail: int = 0,           # 0 = bool; 1, 3, 4, or 8 = rgb
    img_level_of_detail: int = 1,           # obs_ is for train, img_ for gifs
    env_penalize_time: bool = False,
    #  policy config
    net_cnn_type: str = "large",
    net_rnn_type: str = "ff",
    net_width: int = 256,
    # ued config
    ued: str = "plr",                       # dr, dr-finite, plr, accel
    prob_shift: float = 0.0,
    num_train_levels: int = 2048,
    # for plr
    plr_buffer_size: int = 4096,
    plr_temperature: float = 0.1,
    plr_staleness_coeff: float = 0.1,
    plr_prob_replay: float = 0.5, #default 0.5
    plr_regret_estimator: str = "maxmc-actor",
    plr_robust: bool = False,
    # for accel
    num_mutate_steps: int = 12,
    prob_mutate_shift: float = 0.0,
    chain_mutate: bool = True,
    mutate_cheese: bool = True,
    # for proxy augmented methods
    # PPO hyperparameters
    ppo_lr: float = 0.00005,                # learning rate
    ppo_gamma: float = 0.999,               # discount rate
    ppo_clip_eps: float = 0.1,
    ppo_gae_lambda: float = 0.95,
    ppo_entropy_coeff: float = 0.001,
    ppo_critic_coeff: float = 0.5,
    ppo_max_grad_norm: float = 0.5,
    ppo_lr_annealing: bool = False,
    num_minibatches_per_epoch: int = 4,
    num_epochs_per_cycle: int = 5,
    # training dimensions
    num_total_env_steps: int = 20_000_000,
    num_env_steps_per_cycle: int = 128,
    num_parallel_envs: int = 256,
    # logging and evals config
    console_log: bool = True,
    wandb_log: bool = True,
    wandb_project: str = "followme_demo",
    wandb_entity: str = None,               # e.g. 'krueger-lab-cambridge'
    wandb_group: str = None,
    wandb_name: str = None,
    log_gifs: bool = True,
    log_imgs: bool = True,
    log_hists: bool = False,
    num_cycles_per_log: int = 32,           #   32 * 32k = roughly  1M steps
    num_cycles_per_eval: int = 32,          #   32 * 32k = roughly  1M steps
    num_cycles_per_gifs: int = 1024,        # 1024 * 32k = roughly 32M steps
    num_cycles_per_big_eval: int = 1024,    # 1024 * 32k = roughly 32M steps
    evals_num_env_steps: int = 512,
    evals_num_levels: int = 256,
    gif_grid_width: int = 16,
    # checkpointing
    checkpointing: bool = True,             # keep checkpoints? (default: yes)
    keep_all_checkpoints: bool = False,     # if so: keep all of them? (no)
    max_num_checkpoints: int = 1,           # if not: keep only latest n (=1)
    num_cycles_per_checkpoint: int = 512,
    # other
    seed: int = 42,
):
    config = locals()
    util.print_config(config)


    print("configuring environment...")
    env = follow_me.Env(
        obs_level_of_detail=obs_level_of_detail,
        img_level_of_detail=img_level_of_detail,
        penalize_time=env_penalize_time,
    )


    print("configuring level generators...")
    maze_generator = maze_generation.get_generator_class_from_name(
        name=env_layout,
    )()
    orig_level_generator = follow_me.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        num_beacons=num_beacons,
        trustworthy_leader=trustworthy_leader,
    )
    shift_level_generator = follow_me.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        num_beacons=num_beacons,
        trustworthy_leader=trustworthy_leader_shift,
    )
    if prob_shift > 0.0:
        print(f"  mixing level generators with {prob_shift=}...")
        train_level_generator = MixtureLevelGenerator(
            level_generator1=orig_level_generator,
            level_generator2=shift_level_generator,
            prob_level1=1.0-prob_shift,
        )
    else:
        train_level_generator = orig_level_generator


    print("TODO: define level classifier")
    classify_level_is_shift = None


    print("configuring eval level generators...")
    if prob_shift > 0.0:
        eval_level_generators = {
            "train": train_level_generator,
            "orig": orig_level_generator,
            "shift": shift_level_generator,
        }
    else:
        eval_level_generators = {
            "orig": orig_level_generator,
            "shift": shift_level_generator,
        }


    print("TODO: implement level solver...")


    print("TODO: implement level metrics...")


    print("TODO: implement level splayers for heatmap evals...")


    print("TODO: configure parser and fixed eval levels...")


    train.run(
        TrainConfig.from_cli(locals()),
        env=env,
        train_level_generator=train_level_generator,
        level_mutator=None,
        level_solver=None,
        level_metrics=None,
        eval_level_generators=eval_level_generators,
        fixed_eval_levels={},
        heatmap_splayer_fn=None,
        classify_level_is_shift=classify_level_is_shift,
    )


@util.wandb_run
def lava(
    # environment config
    env_size: int = 15,
    env_layout: str = 'blocks',
    num_beacons: int = 6,
    lava_threshold: float = -1.0,
    lava_treshold_shift: float = -0.25,
    obs_level_of_detail: int = 0,           # 0 = bool; 1, 3, 4, or 8 = rgb
    img_level_of_detail: int = 1,           # obs_ is for train, img_ for gifs
    env_penalize_time: bool = False,
    #  policy config
    net_cnn_type: str = "large",
    net_rnn_type: str = "ff",
    net_width: int = 256,
    # ued config
    ued: str = "plr",                       # dr, dr-finite, plr, accel
    prob_shift: float = 0.0,
    num_train_levels: int = 2048,
    # for plr
    plr_buffer_size: int = 4096,
    plr_temperature: float = 0.1,
    plr_staleness_coeff: float = 0.1,
    plr_prob_replay: float = 0.5, #default 0.5
    plr_regret_estimator: str = "maxmc-actor",
    plr_robust: bool = False,
    # for accel
    num_mutate_steps: int = 12,
    prob_mutate_shift: float = 0.0,
    chain_mutate: bool = True,
    mutate_cheese: bool = True,
    # for proxy augmented methods
    # PPO hyperparameters
    ppo_lr: float = 0.00005,                # learning rate
    ppo_gamma: float = 0.999,               # discount rate
    ppo_clip_eps: float = 0.1,
    ppo_gae_lambda: float = 0.95,
    ppo_entropy_coeff: float = 0.001,
    ppo_critic_coeff: float = 0.5,
    ppo_max_grad_norm: float = 0.5,
    ppo_lr_annealing: bool = False,
    num_minibatches_per_epoch: int = 4,
    num_epochs_per_cycle: int = 5,
    # training dimensions
    num_total_env_steps: int = 20_000_000,
    num_env_steps_per_cycle: int = 128,
    num_parallel_envs: int = 256,
    # logging and evals config
    console_log: bool = True,
    wandb_log: bool = True,
    wandb_project: str = "lavaland_demo",
    wandb_entity: str = None,               # e.g. 'krueger-lab-cambridge'
    wandb_group: str = None,
    wandb_name: str = None,
    log_gifs: bool = True,
    log_imgs: bool = True,
    log_hists: bool = False,
    num_cycles_per_log: int = 32,           #   32 * 32k = roughly  1M steps
    num_cycles_per_eval: int = 32,          #   32 * 32k = roughly  1M steps
    num_cycles_per_gifs: int = 1024,        # 1024 * 32k = roughly 32M steps
    num_cycles_per_big_eval: int = 1024,    # 1024 * 32k = roughly 32M steps
    evals_num_env_steps: int = 512,
    evals_num_levels: int = 256,
    gif_grid_width: int = 16,
    # checkpointing
    checkpointing: bool = True,             # keep checkpoints? (default: yes)
    keep_all_checkpoints: bool = False,     # if so: keep all of them? (no)
    max_num_checkpoints: int = 1,           # if not: keep only latest n (=1)
    num_cycles_per_checkpoint: int = 512,
    # other
    seed: int = 42,
):
    config = locals()
    util.print_config(config)


    print("configuring environment...")
    env = lava_land.Env(
        obs_level_of_detail=obs_level_of_detail,
        img_level_of_detail=img_level_of_detail,
        penalize_time=env_penalize_time,
    )


    print("configuring level generators...")
    maze_generator = maze_generation.get_generator_class_from_name(
        name=env_layout,
    )()
    orig_level_generator = lava_land.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        lava_threshold=lava_threshold,
    )
    shift_level_generator = lava_land.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        lava_threshold=lava_treshold_shift,
    )
    if prob_shift > 0.0:
        print(f"  mixing level generators with {prob_shift=}...")
        train_level_generator = MixtureLevelGenerator(
            level_generator1=orig_level_generator,
            level_generator2=shift_level_generator,
            prob_level1=1.0-prob_shift,
        )
    else:
        train_level_generator = orig_level_generator


    print("TODO: define level classifier")
    classify_level_is_shift = None


    print("configuring eval level generators...")
    if prob_shift > 0.0:
        eval_level_generators = {
            "train": train_level_generator,
            "orig": orig_level_generator,
            "shift": shift_level_generator,
        }
    else:
        eval_level_generators = {
            "orig": orig_level_generator,
            "shift": shift_level_generator,
        }


    print("TODO: implement level solver...")


    print("TODO: implement level metrics...")


    print("TODO: implement level splayers for heatmap evals...")


    print("TODO: configure parser and fixed eval levels...")


    train.run(
        TrainConfig.from_cli(locals()),
        env=env,
        train_level_generator=train_level_generator,
        level_mutator=None,
        level_solver=None,
        level_metrics=None,
        eval_level_generators=eval_level_generators,
        fixed_eval_levels={},
        heatmap_splayer_fn=None,
        classify_level_is_shift=classify_level_is_shift,
    )
