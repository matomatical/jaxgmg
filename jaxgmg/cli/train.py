"""
Launcher for training runs.
"""

import jax
import jax.numpy as jnp

from jaxgmg.procgen import maze_generation
from jaxgmg.environments import cheese_in_the_corner
from jaxgmg.environments import cheese_on_a_dish
from jaxgmg.environments import cheese_on_a_pile
from jaxgmg.environments import keys_and_chests
from jaxgmg.environments import minigrid_maze
from jaxgmg.environments import lava_land
from jaxgmg.environments import follow_me
from jaxgmg.baselines import train

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
    ued: str = "plr",                       # dr, dr-finite, plr, plr-parallel
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
    train_proxy_critic: bool = False,
    plr_proxy_shaping: bool = False,
    proxy_name: str = "proxy_corner",
    plr_proxy_shaping_coeff: float = 0.5,
    clipping: bool = False,
    # PPO hyperparameters
    ppo_lr: float = 0.00005,                # learning rate
    ppo_gamma: float = 0.999,               # discount rate
    ppo_clip_eps: float = 0.1,
    ppo_gae_lambda: float = 0.95,
    ppo_entropy_coeff: float = 0.001,
    ppo_critic_coeff: float = 0.5,
    ppo_proxy_critic_coeff: float = 0.5,
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
        print("  mixing level generators with {prob_shift=}...")
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
            #"shift": shift_level_generator,
            "tree": tree_level_generator,
        }


    print("configuring parser and parsing fixed eval levels...")
    if env_size == 15:
        level_parser_15 = cheese_in_the_corner.LevelParser(
            height=15,
            width=15,
        )
        fixed_eval_levels = {
            'sixteen-rooms-corner': level_parser_15.parse("""
                # # # # # # # # # # # # # # # 
                # * . . # . . # . . # . . . #
                # . . . . . . . . . # . . . #
                # . . . # . . # . . . . . . #
                # # . # # # . # # . # # # . #
                # . . . # . . . . . . . . . #
                # . . . . . . # . . # . . . #
                # # # . # . # # . # # # . # #
                # . . . # . . . . . # . . . #
                # . . . # . . # . . . . . . #
                # . # # # # . # # . # . # # #
                # . . . # . . # . . # . . . #
                # . . . . . . # . . . . @ . #
                # . . . # . . . . . # . . . #
                # # # # # # # # # # # # # # #
            """),
            'sixteen-rooms-2-corner': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # * . . # . . . . . # . . . #
                # . . . . . . # . . # . . . #
                # . . . # . . # . . # . . . #
                # # # # # . # # . # # # . # #
                # . . . # . . # . . . . . . #
                # . . . . . . # . . # . . . #
                # # . # # # # # . # # # # # #
                # . . . # . . # . . # . . . #
                # . . . # . . . . . . . . . #
                # # # . # # . # # . # # # # #
                # . . . # . . # . . # . . . #
                # . . . . . . # . . . . @ . #
                # . . . # . . # . . # . . . #
                # # # # # # # # # # # # # # #
            """),
            'labyrinth-2-corner': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # * # . . . . . . . . . . . #
                # . # . # # # # # # # # # . #
                # . # . # . . . . . . . # . #
                # . # . # . # # # # # . # . #
                # . # . # . # . . . # . # . #
                # . . . # . # . # . # . # . #
                # # # # # . # @ # . # . # . #
                # . . . # . # # # . # . # . #
                # . # . # . . . . . # . # . #
                # . # . # # # # # # # . # . #
                # . # . . . . . . . . . # . #
                # . # # # # # # # # # # # . #
                # . . . . . . . . . . . . . #
                # # # # # # # # # # # # # # #
            """),
            'standard-maze': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . . . . . # @ . . . # . . #
                # . # # # . # # # # . # # . #
                # . # . . . . . . . . . . . #
                # . # # # # # # # # . # # # #
                # . . . . . . . . # . . . . #
                # # # # # # # . # # # # # . #
                # . . . . # . . # . . . . . #
                # . # # . . . # # . # # # # #
                # . . # . # . . # . . . # . #
                # # . # . # # . # # # . # . #
                # # . # . . # . . . # . . . #
                # # . # # . # # # . # # # . #
                # . . . # . . * # . # . . . #
                # # # # # # # # # # # # # # #
            """),
            'standard-maze-2': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . . . # . # . . . . # . . #
                # . # . # . # # # # . . . # #
                # . # . . . . . . . . # . . #
                # . # # # # # # # # . # # # #
                # . . . # . . # . # . # . * #
                # # # . # . # # . # . # . . #
                # @ # . # . . . . # . # # . #
                # . # . # # . # # # . . # . #
                # . # . . # . . # # # . # . #
                # . # # . # # . # . # . # . #
                # . # . . . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . . . # . . . # . . . . . #
                # # # # # # # # # # # # # # #
            """),
            'standard-maze-3': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . . . @ # . # . . . . . . #
                # . # # # # . # . # # # # . #
                # . # . . . . # . # . . . . #
                # . . . # # # # . # . # . # #
                # # # . # . . . . # . # . . #
                # . . . # . # # . # . # # . #
                # . # . # . # . . # . . # * #
                # . # . # . # . # # # . # # #
                # . # . . . # . # . # . . . #
                # . # # # . # . # . # # # . #
                # . # . . . # . # . . . # . #
                # . # . # # # . # . # . # . #
                # . # . . . # . . . # . . . #
                # # # # # # # # # # # # # # #
            """),
            'small-corridor': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . . . . . . . . . . . . . #
                # . # . # . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . # . # . # * # . # . # . #
                # @ # # # # # # # # # # # . #
                # . # . # . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . . . . . . . . . . . . . #
                # # # # # # # # # # # # # # #
            """),
            'four-rooms-small': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . . . . . . # . . . . . . #
                # . . . . . . # . . . . . . #
                # . . . . . . # . . . . . . #
                # . . . . . @ # . . . . . . #
                # . . . . . . # . . . . . . #
                # . . . . . . . . . . . . . #
                # # # . # # # # # # . # # # #
                # . . . . . . # . . . . . . #
                # . . . . . . # . . . * . . #
                # . . . . . . . . . . . . . #
                # . . . . . . # . . . . . . #
                # . . . . . . # . . . . . . #
                # . . . . . . # . . . . . . #
                # # # # # # # # # # # # # # #
            """),
            'perfect-maze-15x15': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . # . . . # . # . . . . . #
                # . # . # . # . # . # . # # #
                # * . . # . # . . . # . . . #
                # # # # # . # # # . # # # . #
                # . . . # . . . # . # . # . #
                # . # . # # # . # . # . # . #
                # . # . . . . . # . . . # . #
                # . # # # # # # # # # # # . #
                # . # . . . . . . . . . . . #
                # . # # # # # # # . # # # # #
                # . # . . . . . # . . . . @ #
                # . # . # # # . # # # # # . #
                # . . . # . . . . . . . . . #
                # # # # # # # # # # # # # # #
            """),
        }
    else:
        print("(unsupported size for fixed evals)")
        fixed_eval_levels = {}


    print("configuring heatmap splayer...")
    match level_splayer:
        case 'mouse':
            splayer_fn = cheese_in_the_corner.splay_mouse
        case 'cheese':
            splayer_fn = cheese_in_the_corner.splay_cheese
        case 'cheese-and-mouse':
            splayer_fn = cheese_in_the_corner.splay_cheese_and_mouse 
        case _:
            raise ValueError(f'unknown level splayer {level_splayer!r}')


    train.run(
        seed=seed,
        env=env,
        train_level_generator=train_level_generator,
        level_solver=level_solver,
        level_mutator=level_mutator,
        level_metrics=level_metrics,
        eval_level_generators=eval_level_generators,
        fixed_eval_levels=fixed_eval_levels,
        heatmap_splayer_fn=splayer_fn,
        classify_level_is_shift=classify_level_is_shift,
        net_cnn_type=net_cnn_type,
        net_rnn_type=net_rnn_type,
        net_width=net_width,
        ued=ued,
        prob_shift=prob_shift,
        num_train_levels=num_train_levels,
        plr_buffer_size=plr_buffer_size,
        plr_temperature=plr_temperature,
        plr_staleness_coeff=plr_staleness_coeff,
        plr_prob_replay=plr_prob_replay,
        plr_regret_estimator=plr_regret_estimator,
        plr_robust=plr_robust,
        train_proxy_critic=train_proxy_critic,
        plr_proxy_shaping=plr_proxy_shaping,
        proxy_name=proxy_name,
        plr_proxy_shaping_coeff=plr_proxy_shaping_coeff,
        clipping=clipping,
        eta_schedule=False,
        eta_schedule_time=0.0,
        ppo_lr=ppo_lr,
        ppo_gamma=ppo_gamma,
        ppo_clip_eps=ppo_clip_eps,
        ppo_gae_lambda=ppo_gae_lambda,
        ppo_entropy_coeff=ppo_entropy_coeff,
        ppo_critic_coeff=ppo_critic_coeff,
        ppo_proxy_critic_coeff=ppo_proxy_critic_coeff,
        ppo_max_grad_norm=ppo_max_grad_norm,
        ppo_lr_annealing=ppo_lr_annealing,
        num_minibatches_per_epoch=num_minibatches_per_epoch,
        num_epochs_per_cycle=num_epochs_per_cycle,
        num_total_env_steps=num_total_env_steps,
        num_env_steps_per_cycle=num_env_steps_per_cycle,
        num_parallel_envs=num_parallel_envs,
        console_log=console_log,
        wandb_log=wandb_log,
        log_gifs=log_gifs,
        log_imgs=log_imgs,
        log_hists=log_hists,
        num_cycles_per_log=num_cycles_per_log,
        num_cycles_per_eval=num_cycles_per_eval,
        num_cycles_per_gifs=num_cycles_per_gifs,
        num_cycles_per_big_eval=num_cycles_per_big_eval,
        evals_num_env_steps=evals_num_env_steps,
        evals_num_levels=evals_num_levels,
        gif_grid_width=gif_grid_width,
        checkpointing=checkpointing,
        keep_all_checkpoints=keep_all_checkpoints,
        max_num_checkpoints=max_num_checkpoints,
        num_cycles_per_checkpoint=num_cycles_per_checkpoint,
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
    train_proxy_critic: bool = False,
    plr_proxy_shaping: bool = False,
    proxy_name: str = "proxy_dish",
    plr_proxy_shaping_coeff: float = 0.5,
    clipping: bool = False,
    eta_schedule: bool = False,
    eta_schedule_time: float = 0.4,
    # PPO hyperparameters
    ppo_lr: float = 0.00005,                # learning rate
    ppo_gamma: float = 0.999,               # discount rate
    ppo_clip_eps: float = 0.1,
    ppo_gae_lambda: float = 0.95,
    ppo_entropy_coeff: float = 0.001,
    ppo_critic_coeff: float = 0.5,
    ppo_proxy_critic_coeff: float = 0.5,
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
        print("  mixing level generators with {prob_shift=}...")
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
    if env_size == 15:
        level_parser_15 = cheese_on_a_dish.LevelParser(
            height=15,
            width=15,
        )
        fixed_eval_levels = {
            'sixteen-rooms': level_parser_15.parse("""
                # # # # # # # # # # # # # # # 
                # . . . # . . # . . # . . . #
                # . @ . . . . . . . # . . . #
                # . . . # . . # . . . . . . #
                # # . # # # . # # . # # # . #
                # . . . # . . . . . . . . . #
                # . . . . . . # . . # . . . #
                # # # . # . # # . # # # . # #
                # . . . # . . . . . # . . . #
                # . . . # . . # . . . . . . #
                # . # # # # . # # . # . # # #
                # . . . # . . # . . # . . . #
                # . . . . . . # . . . . b . #
                # . . . # . . . . . # . . . #
                # # # # # # # # # # # # # # #
            """),
            'sixteen-rooms-2': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . . . # . . . . . # . . . #
                # . @ . . . . # . . # . . . #
                # . . . # . . # . . # . . . #
                # # # # # . # # . # # # . # #
                # . . . # . . # . . . . . . #
                # . . . . . . # . . # . . . #
                # # . # # # # # . # # # # # #
                # . . . # . . # . . # . . . #
                # . . . # . . . . . . . . . #
                # # # . # # . # # . # # # # #
                # . . . # . . # . . # . . . #
                # . . . . . . # . . . . b . #
                # . . . # . . # . . # . . . #
                # # # # # # # # # # # # # # #
            """),
            'labyrinth': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . . . . . . . . . . . . . #
                # . # # # # # # # # # # # . #
                # . # . . . . . . . . . # . #
                # . # . # # # # # # # . # . #
                # . # . # . . . . . # . # . #
                # . # . # . # # # . # . # . #
                # . # . # . # b # . # . # . #
                # . # . # . # . # . # . # . #
                # . # . . . # . . . # . # . #
                # . # # # # # # # # # . # . #
                # . . . . . # . . . . . # . #
                # # # # # . # . # # # # # . #
                # @ . . . . # . . . . . . . #
                # # # # # # # # # # # # # # #
            """),
            'labyrinth-2': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # @ # . . . . . . . . . . . #
                # . # . # # # # # # # # # . #
                # . # . # . . . . . . . # . #
                # . # . # . # # # # # . # . #
                # . # . # . # . . . # . # . #
                # . . . # . # . # . # . # . #
                # # # # # . # b # . # . # . #
                # . . . # . # # # . # . # . #
                # . # . # . . . . . # . # . #
                # . # . # # # # # # # . # . #
                # . # . . . . . . . . . # . #
                # . # # # # # # # # # # # . #
                # . . . . . . . . . . . . . #
                # # # # # # # # # # # # # # #
            """),
            'labyrinth-flipped': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . . . . . . . . . . . . . #
                # . # # # # # # # # # # # . #
                # . # . . . . . . . . . # . #
                # . # . # # # # # # # . # . #
                # . # . # . . . . . # . # . #
                # . # . # . # # # . # . # . #
                # . # . # . # b # . # . # . #
                # . # . # . # . # . # . # . #
                # . # . # . . . # . . . # . #
                # . # . # # # # # # # # # . #
                # . # . . . . . # . . . . . #
                # . # # # # # . # . # # # # #
                # . . . . . . . # . . . . @ #
                # # # # # # # # # # # # # # #
            """),
            'standard-maze': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . . . . . # @ . . . # . . #
                # . # # # . # # # # . # # . #
                # . # . . . . . . . . . . . #
                # . # # # # # # # # . # # # #
                # . . . . . . . . # . . . . #
                # # # # # # # . # # # # # . #
                # . . . . # . . # . . . . . #
                # . # # . . . # # . # # # # #
                # . . # . # . . # . . . # . #
                # # . # . # # . # # # . # . #
                # # . # . . # . . . # . . . #
                # # . # # . # # # . # # # . #
                # . . . # . . b # . # . . . #
                # # # # # # # # # # # # # # #
            """),
            'standard-maze-2': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . . . # . # . . . . # . . #
                # . # . # . # # # # . . . # #
                # . # . . . . . . . . # . . #
                # . # # # # # # # # . # # # #
                # . . . # . . # . # . # . b #
                # # # . # . # # . # . # . . #
                # @ # . # . . . . # . # # . #
                # . # . # # . # # # . . # . #
                # . # . . # . . # # # . # . #
                # . # # . # # . # . # . # . #
                # . # . . . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . . . # . . . # . . . . . #
                # # # # # # # # # # # # # # #
            """),
            'standard-maze-3': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . . . @ # . # . . . . . . #
                # . # # # # . # . # # # # . #
                # . # . . . . # . # . . . . #
                # . . . # # # # . # . # . # #
                # # # . # . . . . # . # . . #
                # . . . # . # # . # . # # . #
                # . # . # . # . . # . . # b #
                # . # . # . # . # # # . # # #
                # . # . . . # . # . # . . . #
                # . # # # . # . # . # # # . #
                # . # . . . # . # . . . # . #
                # . # . # # # . # . # . # . #
                # . # . . . # . . . # . . . #
                # # # # # # # # # # # # # # #
            """),
            'small-corridor': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . . . . . . . . . . . . . #
                # . # . # . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . # . # . # b # . # . # . #
                # @ # # # # # # # # # # # . #
                # . # . # . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . . . . . . . . . . . . . #
                # # # # # # # # # # # # # # #
            """),
            'four-rooms-small': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . . . . . . # . . . . . . #
                # . . . . . . # . . . . . . #
                # . . . . . . # . . . . . . #
                # . . . . . @ # . . . . . . #
                # . . . . . . # . . . . . . #
                # . . . . . . . . . . . . . #
                # # # . # # # # # # . # # # #
                # . . . . . . # . . . . . . #
                # . . . . . . # . . . b . . #
                # . . . . . . . . . . . . . #
                # . . . . . . # . . . . . . #
                # . . . . . . # . . . . . . #
                # . . . . . . # . . . . . . #
                # # # # # # # # # # # # # # #
            """),
            'perfect-maze-15x15': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . # . . . # . # . . . . . #
                # . # . # . # . # . # . # # #
                # b . . # . # . . . # . . . #
                # # # # # . # # # . # # # . #
                # . . . # . . . # . # . # . #
                # . # . # # # . # . # . # . #
                # . # . . . . . # . . . # . #
                # . # # # # # # # # # # # . #
                # . # . . . . . . . . . . . #
                # . # # # # # # # . # # # # #
                # . # . . . . . # . . . . @ #
                # . # . # # # . # # # # # . #
                # . . . # . . . . . . . . . #
                # # # # # # # # # # # # # # #
            """),
        }
    else:
        print("(unsupported size for fixed evals)")
        fixed_eval_levels = {}


    train.run(
        seed=seed,
        env=env,
        train_level_generator=train_level_generator,
        level_solver=level_solver,
        level_mutator=level_mutator,
        level_metrics=level_metrics,
        eval_level_generators=eval_level_generators,
        fixed_eval_levels=fixed_eval_levels,
        heatmap_splayer_fn=None,
        classify_level_is_shift=classify_level_is_shift,
        net_cnn_type=net_cnn_type,
        net_rnn_type=net_rnn_type,
        net_width=net_width,
        ued=ued,
        prob_shift=prob_shift,
        num_train_levels=num_train_levels,
        plr_buffer_size=plr_buffer_size,
        plr_temperature=plr_temperature,
        plr_staleness_coeff=plr_staleness_coeff,
        plr_prob_replay=plr_prob_replay,
        plr_regret_estimator=plr_regret_estimator,
        plr_robust=plr_robust,
        train_proxy_critic=train_proxy_critic,
        plr_proxy_shaping=plr_proxy_shaping,
        proxy_name=proxy_name,
        plr_proxy_shaping_coeff=plr_proxy_shaping_coeff,
        clipping=clipping,
        eta_schedule=eta_schedule,
        eta_schedule_time=eta_schedule_time,
        ppo_lr=ppo_lr,
        ppo_gamma=ppo_gamma,
        ppo_clip_eps=ppo_clip_eps,
        ppo_gae_lambda=ppo_gae_lambda,
        ppo_entropy_coeff=ppo_entropy_coeff,
        ppo_critic_coeff=ppo_critic_coeff,
        ppo_proxy_critic_coeff=ppo_proxy_critic_coeff,
        ppo_max_grad_norm=ppo_max_grad_norm,
        ppo_lr_annealing=ppo_lr_annealing,
        num_minibatches_per_epoch=num_minibatches_per_epoch,
        num_epochs_per_cycle=num_epochs_per_cycle,
        num_total_env_steps=num_total_env_steps,
        num_env_steps_per_cycle=num_env_steps_per_cycle,
        num_parallel_envs=num_parallel_envs,
        console_log=console_log,
        wandb_log=wandb_log,
        log_gifs=log_gifs,
        log_imgs=log_imgs,
        log_hists=log_hists,
        num_cycles_per_log=num_cycles_per_log,
        num_cycles_per_eval=num_cycles_per_eval,
        num_cycles_per_gifs=num_cycles_per_gifs,
        num_cycles_per_big_eval=num_cycles_per_big_eval,
        evals_num_env_steps=evals_num_env_steps,
        evals_num_levels=evals_num_levels,
        gif_grid_width=gif_grid_width,
        checkpointing=checkpointing,
        keep_all_checkpoints=keep_all_checkpoints,
        max_num_checkpoints=max_num_checkpoints,
        num_cycles_per_checkpoint=num_cycles_per_checkpoint,
    )


@util.wandb_run
def pile(
    # environment config
    env_size: int = 13,
    env_layout: str = 'blocks',
    # how many objects go on the cheese - not important cuz we want to train
    # with both in the same position
    split_elements_train:int = 0,
    split_elements_shift:int = 0,
    max_cheese_radius: int = 0,
    max_cheese_radius_shift: int = 10,
    # these two are not relevant and can be ignored for now (they may come
    # useful in next implementations)
    max_dish_radius: int = 0,
    max_dish_radius_shift: int= 0,
    # other env stuff
    env_terminate_after_pile: bool = False,
    obs_level_of_detail: int = 0,           # 0 = bool; 1, 3, 4, or 8 = rgb
    img_level_of_detail: int = 1,           # obs_ is for train, img_ for gifs
    env_penalize_time: bool = False,
    # policy config
    net_cnn_type: str = "large",
    net_rnn_type: str = "ff",
    net_width: int = 256,
    # ued config
    ued: str = "plr",                        # dr, dr-finite, plr, plr-parallel
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
    mutate_cheese_on_pile: bool = True,
    # for proxy augmented methods
    train_proxy_critic: bool = False,
    plr_proxy_shaping: bool = False,
    proxy_name: str = "proxy_pile",
    plr_proxy_shaping_coeff: float = 0.5,
    clipping: bool = True,
    # PPO hyperparameters
    ppo_lr: float = 0.00005,                # learning rate
    ppo_gamma: float = 0.999,               # discount rate
    ppo_clip_eps: float = 0.1,
    ppo_gae_lambda: float = 0.95,
    ppo_entropy_coeff: float = 0.001,
    ppo_critic_coeff: float = 0.5,
    ppo_proxy_critic_coeff: float = 0.5,
    ppo_max_grad_norm: float = 0.5,
    ppo_lr_annealing: bool = False,
    num_minibatches_per_epoch: int = 4,
    num_epochs_per_cycle: int = 5,
    # training dimensions
    num_total_env_steps: int = 300_000_000,
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
    env = cheese_on_a_pile.Env(
        terminate_after_cheese_and_pile=env_terminate_after_pile,
        # check this, if it should be split_elements_train or split_elements_shift or 0
        split_object_firstgroup=split_elements_train,
        obs_level_of_detail=obs_level_of_detail,
        img_level_of_detail=img_level_of_detail,
        penalize_time=env_penalize_time,
    )


    print("configuring level generators...")
    maze_generator = maze_generation.get_generator_class_from_name(
        name=env_layout,
    )()
    orig_level_generator = cheese_on_a_pile.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        max_cheese_radius=max_cheese_radius,
        max_dish_radius=max_dish_radius,
        split_elements=split_elements_train,
    )
    shift_level_generator=cheese_on_a_pile.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        max_cheese_radius=max_cheese_radius_shift,
        max_dish_radius=max_dish_radius_shift,
        split_elements=split_elements_shift,
    )
    if prob_shift > 0.0:
        print("  mixing level generators with {prob_shift=}...")
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
            level.cheese_pos[0] != level.napkin_pos[0],
            level.cheese_pos[1] != level.napkin_pos[1],
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
    if mutate_cheese_on_pile:
        biased_cheese_on_pile_mutator = MixtureLevelMutator(
            mutators=(
                # teleport cheese on pile
                cheese_on_a_pile.CheeseonPileLevelMutator(
                    max_cheese_radius=max_cheese_radius,
                    split_elements=split_elements_shift, # split elements shift and train should always be the same -> Will fix that
                ),
                # teleport cheese and pile to a random different position, apart by max_cheese_radius
                cheese_on_a_pile.CheeseonPileLevelMutator(
                    max_cheese_radius=max_cheese_radius_shift,
                    split_elements=split_elements_shift,
                ),
            ),
            mixing_probs=(1-prob_mutate_shift, prob_mutate_shift),
        )
    else:
        # replace this mutation with something else
        biased_cheese_on_pile_mutator = cheese_on_a_pile.ToggleWallLevelMutator()
    # overall
    if chain_mutate:
        level_mutator = ChainLevelMutator(mutators=(
            # mutate walls (n-2 steps)
            IteratedLevelMutator(
                mutator=cheese_on_a_pile.ToggleWallLevelMutator(),
                num_steps=num_mutate_steps - 2,
            ),
            # maybe scatter mouse (1 step) else another wall toggle
            MixtureLevelMutator(
                mutators=(
                    cheese_on_a_pile.ScatterMouseLevelMutator(
                        transpose_with_cheese_on_collision=False,
                        transpose_with_pile_on_collision=False,
                        split_elements=split_elements_shift,
                    ),
                    cheese_on_a_pile.ToggleWallLevelMutator(),
                ),
                mixing_probs=(1/2,1/2),
            ),
            # biased scatter cheese (1 step)
            biased_cheese_on_pile_mutator,
        ))
    else:
        level_mutator = IteratedLevelMutator(
            mutator=MixtureLevelMutator(
                mutators=(
                    cheese_on_a_pile.ToggleWallLevelMutator(),
                    cheese_on_a_pile.ScatterMouseLevelMutator(
                        transpose_with_cheese_on_collision=False,
                        transpose_with_pile_on_collision=False,
                        split_elements=split_elements_shift,
                    ),
                    biased_cheese_on_pile_mutator,
                ),
                mixing_probs=(10/12,1/12,1/12),
            ),
            num_steps=num_mutate_steps,
        )


    print("configuring level solver...")
    level_solver = cheese_on_a_pile.LevelSolver(
        env=env,
        discount_rate=ppo_gamma,
    )


    print("configuring level metrics...")
    level_metrics = cheese_on_a_pile.LevelMetrics(
        env=env,
        discount_rate=ppo_gamma,
    )


    print("TODO: implement level splayers for heatmap evals...")


    print("configuring parser and parsing fixed eval levels...")
    if env_size == 15:
        level_parser_15 = cheese_on_a_pile.LevelParser(
            height=15,
            width=15,
            split_elements=split_elements_train,
        )
        fixed_eval_levels = {
            'sixteen-rooms': level_parser_15.parse("""
                # # # # # # # # # # # # # # # 
                # . . . # . . # . . # . . . #
                # . @ . . . . . . . # . . . #
                # . . . # . . # . . . . . . #
                # # . # # # . # # . # # # . #
                # . . . # . . . . . . . . . #
                # . . . . . . # . . # . . . #
                # # # . # . # # . # # # . # #
                # . . . # . . . . . # . . . #
                # . . . # . . # . . . . . . #
                # . # # # # . # # . # . # # #
                # . . . # . . # . . # . . . #
                # . . . . . . # . . . . b . #
                # . . . # . . . . . # . . . #
                # # # # # # # # # # # # # # #
            """),
            'sixteen-rooms-2': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . . . # . . . . . # . . . #
                # . @ . . . . # . . # . . . #
                # . . . # . . # . . # . . . #
                # # # # # . # # . # # # . # #
                # . . . # . . # . . . . . . #
                # . . . . . . # . . # . . . #
                # # . # # # # # . # # # # # #
                # . . . # . . # . . # . . . #
                # . . . # . . . . . . . . . #
                # # # . # # . # # . # # # # #
                # . . . # . . # . . # . . . #
                # . . . . . . # . . . . b . #
                # . . . # . . # . . # . . . #
                # # # # # # # # # # # # # # #
            """),
            'labyrinth': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . . . . . . . . . . . . . #
                # . # # # # # # # # # # # . #
                # . # . . . . . . . . . # . #
                # . # . # # # # # # # . # . #
                # . # . # . . . . . # . # . #
                # . # . # . # # # . # . # . #
                # . # . # . # b # . # . # . #
                # . # . # . # . # . # . # . #
                # . # . . . # . . . # . # . #
                # . # # # # # # # # # . # . #
                # . . . . . # . . . . . # . #
                # # # # # . # . # # # # # . #
                # @ . . . . # . . . . . . . #
                # # # # # # # # # # # # # # #
            """),
            'labyrinth-2': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # @ # . . . . . . . . . . . #
                # . # . # # # # # # # # # . #
                # . # . # . . . . . . . # . #
                # . # . # . # # # # # . # . #
                # . # . # . # . . . # . # . #
                # . . . # . # . # . # . # . #
                # # # # # . # b # . # . # . #
                # . . . # . # # # . # . # . #
                # . # . # . . . . . # . # . #
                # . # . # # # # # # # . # . #
                # . # . . . . . . . . . # . #
                # . # # # # # # # # # # # . #
                # . . . . . . . . . . . . . #
                # # # # # # # # # # # # # # #
            """),
            'labyrinth-flipped': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . . . . . . . . . . . . . #
                # . # # # # # # # # # # # . #
                # . # . . . . . . . . . # . #
                # . # . # # # # # # # . # . #
                # . # . # . . . . . # . # . #
                # . # . # . # # # . # . # . #
                # . # . # . # b # . # . # . #
                # . # . # . # . # . # . # . #
                # . # . # . . . # . . . # . #
                # . # . # # # # # # # # # . #
                # . # . . . . . # . . . . . #
                # . # # # # # . # . # # # # #
                # . . . . . . . # . . . . @ #
                # # # # # # # # # # # # # # #
            """),
            'standard-maze': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . . . . . # @ . . . # . . #
                # . # # # . # # # # . # # . #
                # . # . . . . . . . . . . . #
                # . # # # # # # # # . # # # #
                # . . . . . . . . # . . . . #
                # # # # # # # . # # # # # . #
                # . . . . # . . # . . . . . #
                # . # # . . . # # . # # # # #
                # . . # . # . . # . . . # . #
                # # . # . # # . # # # . # . #
                # # . # . . # . . . # . . . #
                # # . # # . # # # . # # # . #
                # . . . # . . b # . # . . . #
                # # # # # # # # # # # # # # #
            """),
            'standard-maze-2': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . . . # . # . . . . # . . #
                # . # . # . # # # # . . . # #
                # . # . . . . . . . . # . . #
                # . # # # # # # # # . # # # #
                # . . . # . . # . # . # . b #
                # # # . # . # # . # . # . . #
                # @ # . # . . . . # . # # . #
                # . # . # # . # # # . . # . #
                # . # . . # . . # # # . # . #
                # . # # . # # . # . # . # . #
                # . # . . . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . . . # . . . # . . . . . #
                # # # # # # # # # # # # # # #
            """),
            'standard-maze-3': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . . . @ # . # . . . . . . #
                # . # # # # . # . # # # # . #
                # . # . . . . # . # . . . . #
                # . . . # # # # . # . # . # #
                # # # . # . . . . # . # . . #
                # . . . # . # # . # . # # . #
                # . # . # . # . . # . . # b #
                # . # . # . # . # # # . # # #
                # . # . . . # . # . # . . . #
                # . # # # . # . # . # # # . #
                # . # . . . # . # . . . # . #
                # . # . # # # . # . # . # . #
                # . # . . . # . . . # . . . #
                # # # # # # # # # # # # # # #
            """),
            'small-corridor': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . . . . . . . . . . . . . #
                # . # . # . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . # . # . # b # . # . # . #
                # @ # # # # # # # # # # # . #
                # . # . # . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . # . # . # . # . # . # . #
                # . . . . . . . . . . . . . #
                # # # # # # # # # # # # # # #
            """),
            'four-rooms-small': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . . . . . . # . . . . . . #
                # . . . . . . # . . . . . . #
                # . . . . . . # . . . . . . #
                # . . . . . @ # . . . . . . #
                # . . . . . . # . . . . . . #
                # . . . . . . . . . . . . . #
                # # # . # # # # # # . # # # #
                # . . . . . . # . . . . . . #
                # . . . . . . # . . . b . . #
                # . . . . . . . . . . . . . #
                # . . . . . . # . . . . . . #
                # . . . . . . # . . . . . . #
                # . . . . . . # . . . . . . #
                # # # # # # # # # # # # # # #
            """),
            'perfect-maze-15x15': level_parser_15.parse("""
                # # # # # # # # # # # # # # #
                # . # . . . # . # . . . . . #
                # . # . # . # . # . # . # # #
                # b . . # . # . . . # . . . #
                # # # # # . # # # . # # # . #
                # . . . # . . . # . # . # . #
                # . # . # # # . # . # . # . #
                # . # . . . . . # . . . # . #
                # . # # # # # # # # # # # . #
                # . # . . . . . . . . . . . #
                # . # # # # # # # . # # # # #
                # . # . . . . . # . . . . @ #
                # . # . # # # . # # # # # . #
                # . . . # . . . . . . . . . #
                # # # # # # # # # # # # # # #
            """),
        }
    else:
        print("(unsupported size for fixed evals)")
        fixed_eval_levels = {}


    train.run(
        seed=seed,
        env=env,
        train_level_generator=train_level_generator,
        level_solver=level_solver,
        level_mutator=level_mutator,
        level_metrics=level_metrics,
        eval_level_generators=eval_level_generators,
        fixed_eval_levels=fixed_eval_levels,
        heatmap_splayer_fn=None,
        classify_level_is_shift=classify_level_is_shift,
        net_cnn_type=net_cnn_type,
        net_rnn_type=net_rnn_type,
        net_width=net_width,
        ued=ued,
        prob_shift=prob_shift,
        num_train_levels=num_train_levels,
        plr_buffer_size=plr_buffer_size,
        plr_temperature=plr_temperature,
        plr_staleness_coeff=plr_staleness_coeff,
        plr_prob_replay=plr_prob_replay,
        plr_regret_estimator=plr_regret_estimator,
        plr_robust=plr_robust,
        train_proxy_critic=train_proxy_critic,
        plr_proxy_shaping=plr_proxy_shaping,
        proxy_name=proxy_name,
        plr_proxy_shaping_coeff=plr_proxy_shaping_coeff,
        clipping=clipping,
        eta_schedule=False,
        eta_schedule_time=0.0,
        ppo_lr=ppo_lr,
        ppo_gamma=ppo_gamma,
        ppo_clip_eps=ppo_clip_eps,
        ppo_gae_lambda=ppo_gae_lambda,
        ppo_entropy_coeff=ppo_entropy_coeff,
        ppo_critic_coeff=ppo_critic_coeff,
        ppo_proxy_critic_coeff=ppo_proxy_critic_coeff,
        ppo_max_grad_norm=ppo_max_grad_norm,
        ppo_lr_annealing=ppo_lr_annealing,
        num_minibatches_per_epoch=num_minibatches_per_epoch,
        num_epochs_per_cycle=num_epochs_per_cycle,
        num_total_env_steps=num_total_env_steps,
        num_env_steps_per_cycle=num_env_steps_per_cycle,
        num_parallel_envs=num_parallel_envs,
        console_log=console_log,
        wandb_log=wandb_log,
        log_gifs=log_gifs,
        log_imgs=log_imgs,
        log_hists=log_hists,
        num_cycles_per_log=num_cycles_per_log,
        num_cycles_per_eval=num_cycles_per_eval,
        num_cycles_per_gifs=num_cycles_per_gifs,
        num_cycles_per_big_eval=num_cycles_per_big_eval,
        evals_num_env_steps=evals_num_env_steps,
        evals_num_levels=evals_num_levels,
        gif_grid_width=gif_grid_width,
        checkpointing=checkpointing,
        keep_all_checkpoints=keep_all_checkpoints,
        max_num_checkpoints=max_num_checkpoints,
        num_cycles_per_checkpoint=num_cycles_per_checkpoint,
    )


@util.wandb_run
def keys(
    # environment config
    env_size: int = 15,
    env_layout: str = 'blocks',
    env_num_keys_min: int = 5,
    env_num_keys_max: int = 5,
    env_num_keys_min_shift: int = 15,
    env_num_keys_max_shift: int = 15,
    env_num_chests_min: int = 15,
    env_num_chests_max: int = 15,
    env_num_chests_min_shift: int = 5,
    env_num_chests_max_shift: int = 5,
    obs_level_of_detail: int = 0,           # 0 = bool; 1, 3, 4, or 8 = rgb
    img_level_of_detail: int = 1,           # obs_ is for train, img_ for gifs
    env_penalize_time: bool = False,
    #  policy config
    net_cnn_type: str = "large",
    net_rnn_type: str = "ff",
    net_width: int = 256,
    # ued config
    ued: str = "plr",                       # dr, dr-finite, plr, plr-parallel
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
    train_proxy_critic: bool = False,
    plr_proxy_shaping: bool = False,
    proxy_name: str = "keys",
    plr_proxy_shaping_coeff: float = 0.5,
    clipping: bool = True,
    # PPO hyperparameters
    ppo_lr: float = 0.00005,                # learning rate
    ppo_gamma: float = 0.999,               # discount rate
    ppo_clip_eps: float = 0.1,
    ppo_gae_lambda: float = 0.95,
    ppo_entropy_coeff: float = 0.001,
    ppo_critic_coeff: float = 0.5,
    ppo_proxy_critic_coeff: float = 0.5,
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
    maze_generator = maze_generation.get_generator_class_from_name(
        name=env_layout,
    )()
    orig_level_generator = keys_and_chests.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        num_keys_min=env_num_keys_min,
        num_keys_max=env_num_keys_max,
        num_chests_min=env_num_chests_min,
        num_chests_max=env_num_chests_max,
    )
    shift_level_generator = keys_and_chests.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        num_keys_min=env_num_keys_min_shift,
        num_keys_max=env_num_keys_max_shift,
        num_chests_min=env_num_chests_min_shift,
        num_chests_max=env_num_chests_max_shift,
    )
    if prob_shift > 0.0:
        print("  mixing level generators with {prob_shift=}...")
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
        seed=seed,
        env=env,
        train_level_generator=train_level_generator,
        level_solver=None,
        level_mutator=None,
        level_metrics=None,
        eval_level_generators=eval_level_generators,
        fixed_eval_levels={},
        heatmap_splayer_fn=None,
        classify_level_is_shift=classify_level_is_shift,
        net_cnn_type=net_cnn_type,
        net_rnn_type=net_rnn_type,
        net_width=net_width,
        ued=ued,
        prob_shift=prob_shift,
        num_train_levels=num_train_levels,
        plr_buffer_size=plr_buffer_size,
        plr_temperature=plr_temperature,
        plr_staleness_coeff=plr_staleness_coeff,
        plr_prob_replay=plr_prob_replay,
        plr_regret_estimator=plr_regret_estimator,
        plr_robust=plr_robust,
        train_proxy_critic=train_proxy_critic,
        plr_proxy_shaping=plr_proxy_shaping,
        proxy_name=proxy_name,
        plr_proxy_shaping_coeff=plr_proxy_shaping_coeff,
        clipping=clipping,
        eta_schedule=False,
        eta_schedule_time=0.0,
        ppo_lr=ppo_lr,
        ppo_gamma=ppo_gamma,
        ppo_clip_eps=ppo_clip_eps,
        ppo_gae_lambda=ppo_gae_lambda,
        ppo_entropy_coeff=ppo_entropy_coeff,
        ppo_critic_coeff=ppo_critic_coeff,
        ppo_proxy_critic_coeff=ppo_proxy_critic_coeff,
        ppo_max_grad_norm=ppo_max_grad_norm,
        ppo_lr_annealing=ppo_lr_annealing,
        num_minibatches_per_epoch=num_minibatches_per_epoch,
        num_epochs_per_cycle=num_epochs_per_cycle,
        num_total_env_steps=num_total_env_steps,
        num_env_steps_per_cycle=num_env_steps_per_cycle,
        num_parallel_envs=num_parallel_envs,
        console_log=console_log,
        wandb_log=wandb_log,
        log_gifs=log_gifs,
        log_imgs=log_imgs,
        log_hists=log_hists,
        num_cycles_per_log=num_cycles_per_log,
        num_cycles_per_eval=num_cycles_per_eval,
        num_cycles_per_gifs=num_cycles_per_gifs,
        num_cycles_per_big_eval=num_cycles_per_big_eval,
        evals_num_env_steps=evals_num_env_steps,
        evals_num_levels=evals_num_levels,
        gif_grid_width=gif_grid_width,
        checkpointing=checkpointing,
        keep_all_checkpoints=keep_all_checkpoints,
        max_num_checkpoints=max_num_checkpoints,
        num_cycles_per_checkpoint=num_cycles_per_checkpoint,
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
    ued: str = "plr",                        # dr, dr-finite, plr, plr-parallel
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
    train_proxy_critic: bool = False,
    plr_proxy_shaping: bool = False,
    proxy_name: str = "proxy_corner",
    plr_proxy_shaping_coeff: float = 0.5,
    clipping: bool = True,
    # PPO hyperparameters
    ppo_lr: float = 0.00005,                # learning rate
    ppo_gamma: float = 0.999,               # discount rate
    ppo_clip_eps: float = 0.1,
    ppo_gae_lambda: float = 0.95,
    ppo_entropy_coeff: float = 0.001,
    ppo_critic_coeff: float = 0.5,
    ppo_proxy_critic_coeff: float = 0.5,
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
        print("  mixing level generators with {prob_shift=}...")
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
    level_parser_15 = minigrid_maze.LevelParser(height=15, width=15)
    level_parser_19 = minigrid_maze.LevelParser(height=19, width=19)
    level_parser_21 = minigrid_maze.LevelParser(height=21, width=21)
    fixed_eval_levels = {
        'sixteen-rooms': level_parser_15.parse("""
            # # # # # # # # # # # # # # # 
            # . . . # . . # . . # . . . #
            # . > . . . . . . . # . . . #
            # . . . # . . # . . . . . . #
            # # . # # # . # # . # # # . #
            # . . . # . . . . . . . . . #
            # . . . . . . # . . # . . . #
            # # # . # . # # . # # # . # #
            # . . . # . . . . . # . . . #
            # . . . # . . # . . . . . . #
            # . # # # # . # # . # . # # #
            # . . . # . . # . . # . . . #
            # . . . . . . # . . . . * . #
            # . . . # . . . . . # . . . #
            # # # # # # # # # # # # # # #
        """),
        'sixteen-rooms-2': level_parser_15.parse("""
            # # # # # # # # # # # # # # #
            # . . . # . . . . . # . . . #
            # . > . . . . # . . # . . . #
            # . . . # . . # . . # . . . #
            # # # # # . # # . # # # . # #
            # . . . # . . # . . . . . . #
            # . . . . . . # . . # . . . #
            # # . # # # # # . # # # # # #
            # . . . # . . # . . # . . . #
            # . . . # . . . . . . . . . #
            # # # . # # . # # . # # # # #
            # . . . # . . # . . # . . . #
            # . . . . . . # . . . . * . #
            # . . . # . . # . . # . . . #
            # # # # # # # # # # # # # # #
        """),
        'labyrinth': level_parser_15.parse("""
            # # # # # # # # # # # # # # #
            # . . . . . . . . . . . . . #
            # . # # # # # # # # # # # . #
            # . # . . . . . . . . . # . #
            # . # . # # # # # # # . # . #
            # . # . # . . . . . # . # . #
            # . # . # . # # # . # . # . #
            # . # . # . # * # . # . # . #
            # . # . # . # . # . # . # . #
            # . # . . . # . . . # . # . #
            # . # # # # # # # # # . # . #
            # . . . . . # . . . . . # . #
            # # # # # . # . # # # # # . #
            # > . . . . # . . . . . . . #
            # # # # # # # # # # # # # # #
        """),
        'labyrinth-2': level_parser_15.parse("""
            # # # # # # # # # # # # # # #
            # > # . . . . . . . . . . . #
            # . # . # # # # # # # # # . #
            # . # . # . . . . . . . # . #
            # . # . # . # # # # # . # . #
            # . # . # . # . . . # . # . #
            # . . . # . # . # . # . # . #
            # # # # # . # * # . # . # . #
            # . . . # . # # # . # . # . #
            # . # . # . . . . . # . # . #
            # . # . # # # # # # # . # . #
            # . # . . . . . . . . . # . #
            # . # # # # # # # # # # # . #
            # . . . . . . . . . . . . . #
            # # # # # # # # # # # # # # #
        """),
        'labyrinth-flipped': level_parser_15.parse("""
            # # # # # # # # # # # # # # #
            # . . . . . . . . . . . . . #
            # . # # # # # # # # # # # . #
            # . # . . . . . . . . . # . #
            # . # . # # # # # # # . # . #
            # . # . # . . . . . # . # . #
            # . # . # . # # # . # . # . #
            # . # . # . # * # . # . # . #
            # . # . # . # . # . # . # . #
            # . # . # . . . # . . . # . #
            # . # . # # # # # # # # # . #
            # . # . . . . . # . . . . . #
            # . # # # # # . # . # # # # #
            # . . . . . . . # . . . . < #
            # # # # # # # # # # # # # # #
        """),
        'standard-maze': level_parser_15.parse("""
            # # # # # # # # # # # # # # #
            # . . . . . # > . . . # . . #
            # . # # # . # # # # . # # . #
            # . # . . . . . . . . . . . #
            # . # # # # # # # # . # # # #
            # . . . . . . . . # . . . . #
            # # # # # # # . # # # # # . #
            # . . . . # . . # . . . . . #
            # . # # . . . # # . # # # # #
            # . . # . # . . # . . . # . #
            # # . # . # # . # # # . # . #
            # # . # . . # . . . # . . . #
            # # . # # . # # # . # # # . #
            # . . . # . . * # . # . . . #
            # # # # # # # # # # # # # # #
        """),
        'standard-maze-2': level_parser_15.parse("""
            # # # # # # # # # # # # # # #
            # . . . # . # . . . . # . . #
            # . # . # . # # # # . . . # #
            # . # . . . . . . . . # . . #
            # . # # # # # # # # . # # # #
            # . . . # . . # . # . # . * #
            # # # . # . # # . # . # . . #
            # > # . # . . . . # . # # . #
            # . # . # # . # # # . . # . #
            # . # . . # . . # # # . # . #
            # . # # . # # . # . # . # . #
            # . # . . . # . # . # . # . #
            # . # . # . # . # . # . # . #
            # . . . # . . . # . . . . . #
            # # # # # # # # # # # # # # #
        """),
        'standard-maze-3': level_parser_15.parse("""
            # # # # # # # # # # # # # # #
            # . . . > # . # . . . . . . #
            # . # # # # . # . # # # # . #
            # . # . . . . # . # . . . . #
            # . . . # # # # . # . # . # #
            # # # . # . . . . # . # . . #
            # . . . # . # # . # . # # . #
            # . # . # . # . . # . . # * #
            # . # . # . # . # # # . # # #
            # . # . . . # . # . # . . . #
            # . # # # . # . # . # # # . #
            # . # . . . # . # . . . # . #
            # . # . # # # . # . # . # . #
            # . # . . . # . . . # . . . #
            # # # # # # # # # # # # # # #
        """),
        'small-corridor': level_parser_15.parse("""
            # # # # # # # # # # # # # # #
            # . . . . . . . . . . . . . #
            # . # . # . # . # . # . # . #
            # . # . # . # . # . # . # . #
            # . # . # . # . # . # . # . #
            # . # . # . # . # . # . # . #
            # . # . # . # * # . # . # . #
            # > # # # # # # # # # # # . #
            # . # . # . # . # . # . # . #
            # . # . # . # . # . # . # . #
            # . # . # . # . # . # . # . #
            # . # . # . # . # . # . # . #
            # . # . # . # . # . # . # . #
            # . . . . . . . . . . . . . #
            # # # # # # # # # # # # # # #
        """),
        'four-rooms': level_parser_19.parse("""
            # # # # # # # # # # # # # # # # # # #
            # . . . . . . . . # . . . . . . . . #
            # . . . . . . . . # . . . . . . . . #
            # . . . . . . . . # . . . . . . . . #
            # . . . . . . . . # . . . . . . . . #
            # . . . . . . . . # . . . . . . . . #
            # . . . . . . . ^ # . . . . . . . . #
            # . . . . . . . . # . . . . . . . . #
            # . . . . . . . . . . . . . . . . . #
            # # # # . # # # # # # # # . # # # # #
            # . . . . . . . . # . . . . . . . . #
            # . . . . . . . . # . . . . . . . . #
            # . . . . . . . . # . . . . * . . . #
            # . . . . . . . . . . . . . . . . . #
            # . . . . . . . . # . . . . . . . . #
            # . . . . . . . . # . . . . . . . . #
            # . . . . . . . . # . . . . . . . . #
            # . . . . . . . . # . . . . . . . . #
            # # # # # # # # # # # # # # # # # # #
        """),
        'large-corridor': level_parser_21.parse("""
            # # # # # # # # # # # # # # # # # # # # #
            # . . . . . . . . . . . . . . . . . . . #
            # . # . # . # . # . # . # . # . # . # . #
            # . # . # . # . # . # . # . # . # . # . #
            # . # . # . # . # . # . # . # . # . # . #
            # . # . # . # . # . # . # . # . # . # . #
            # . # . # . # . # . # . # . # . # . # . #
            # . # . # . # . # . # . # . # . # . # . #
            # . # . # . # . # . # . # . # . # . # . #
            # . # . # . # . # . # . # . # . # . # . #
            # > # # # # # # # # # # # # # # # # # . #
            # . # . # . # . # . # * # . # . # . # . #
            # . # . # . # . # . # . # . # . # . # . #
            # . # . # . # . # . # . # . # . # . # . #
            # . # . # . # . # . # . # . # . # . # . #
            # . # . # . # . # . # . # . # . # . # . #
            # . # . # . # . # . # . # . # . # . # . #
            # . # . # . # . # . # . # . # . # . # . #
            # . # . # . # . # . # . # . # . # . # . #
            # . . . . . . . . . . . . . . . . . . . #
            # # # # # # # # # # # # # # # # # # # # #
        """),
        'perfect-maze-15x15': level_parser_15.parse("""
            # # # # # # # # # # # # # # #
            # . # . . . # . # . . . . . #
            # . # . # . # . # . # . # # #
            # * . . # . # . . . # . . . #
            # # # # # . # # # . # # # . #
            # . . . # . . . # . # . # . #
            # . # . # # # . # . # . # . #
            # . # . . . . . # . . . # . #
            # . # # # # # # # # # # # . #
            # . # . . . . . . . . . . . #
            # . # # # # # # # . # # # # #
            # . # . . . . . # . . . . v #
            # . # . # # # . # # # # # . #
            # . . . # . . . . . . . . . #
            # # # # # # # # # # # # # # #
        """),
        'perfect-maze-21x21': level_parser_21.parse("""
            # # # # # # # # # # # # # # # # # # # # #
            # . # . # . . . # . . . # . # . # . . . #
            # . # . # . # # # . # # # . # . # . # . #
            # . # . # . . . # . . . . . # . # . # . #
            # . # . # # # . # . # # # # # . # # # . #
            # . . . . . . . . . . . . . . . . . . . #
            # . # # # . # . # . # # # . # . # . # . #
            # . # . . . # . # . . . # . # . # # # # #
            # # # # # . # # # # # # # # # * # . # . #
            # . . . . . . . . . . . # . . . # . # . #
            # # # . # # # . # # # # # # # # # . # . #
            # . # . # . # . . . . . . . # . . . # . #
            # . # # # . # . # . # # # # # . # # # . #
            # . . . # . . . # . . . . . . . . . . . #
            # . # # # # # . # # # . # # # # # # # . #
            # . . . # . . . . . # . # . . . # . # . #
            # . # # # . # # # . # # # # # . # . # . #
            # . . . . . # . . . . . # . ^ . # . . . #
            # . # # # # # # # . # # # . # # # . # # #
            # . . . . . . . # . . . # . . . . . . . #
            # # # # # # # # # # # # # # # # # # # # #
        """),
    }


    train.run(
        seed=seed,
        env=env,
        train_level_generator=train_level_generator,
        level_solver=None,
        level_mutator=level_mutator,
        level_metrics=level_metrics,
        eval_level_generators=eval_level_generators,
        fixed_eval_levels=fixed_eval_levels,
        heatmap_splayer_fn=None,
        classify_level_is_shift=classify_level_is_shift,
        net_cnn_type=net_cnn_type,
        net_rnn_type=net_rnn_type,
        net_width=net_width,
        ued=ued,
        prob_shift=prob_shift,
        num_train_levels=num_train_levels,
        plr_buffer_size=plr_buffer_size,
        plr_temperature=plr_temperature,
        plr_staleness_coeff=plr_staleness_coeff,
        plr_prob_replay=plr_prob_replay,
        plr_regret_estimator=plr_regret_estimator,
        plr_robust=plr_robust,
        train_proxy_critic=train_proxy_critic,
        plr_proxy_shaping=plr_proxy_shaping,
        proxy_name=proxy_name,
        plr_proxy_shaping_coeff=plr_proxy_shaping_coeff,
        clipping=clipping,
        eta_schedule=False,
        eta_schedule_time=0.0,
        ppo_lr=ppo_lr,
        ppo_gamma=ppo_gamma,
        ppo_clip_eps=ppo_clip_eps,
        ppo_gae_lambda=ppo_gae_lambda,
        ppo_entropy_coeff=ppo_entropy_coeff,
        ppo_critic_coeff=ppo_critic_coeff,
        ppo_proxy_critic_coeff=ppo_proxy_critic_coeff,
        ppo_max_grad_norm=ppo_max_grad_norm,
        ppo_lr_annealing=ppo_lr_annealing,
        num_minibatches_per_epoch=num_minibatches_per_epoch,
        num_epochs_per_cycle=num_epochs_per_cycle,
        num_total_env_steps=num_total_env_steps,
        num_env_steps_per_cycle=num_env_steps_per_cycle,
        num_parallel_envs=num_parallel_envs,
        console_log=console_log,
        wandb_log=wandb_log,
        log_gifs=log_gifs,
        log_imgs=log_imgs,
        log_hists=log_hists,
        num_cycles_per_log=num_cycles_per_log,
        num_cycles_per_eval=num_cycles_per_eval,
        num_cycles_per_gifs=num_cycles_per_gifs,
        num_cycles_per_big_eval=num_cycles_per_big_eval,
        evals_num_env_steps=evals_num_env_steps,
        evals_num_levels=evals_num_levels,
        gif_grid_width=gif_grid_width,
        checkpointing=checkpointing,
        keep_all_checkpoints=keep_all_checkpoints,
        max_num_checkpoints=max_num_checkpoints,
        num_cycles_per_checkpoint=num_cycles_per_checkpoint,
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
    train_proxy_critic: bool = False,
    plr_proxy_shaping: bool = False,
    proxy_name: str = "",
    plr_proxy_shaping_coeff: float = 0.5,
    clipping: bool = True,
    # PPO hyperparameters
    ppo_lr: float = 0.00005,                # learning rate
    ppo_gamma: float = 0.999,               # discount rate
    ppo_clip_eps: float = 0.1,
    ppo_gae_lambda: float = 0.95,
    ppo_entropy_coeff: float = 0.001,
    ppo_critic_coeff: float = 0.5,
    ppo_proxy_critic_coeff: float = 0.5,
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
        seed=seed,
        env=env,
        train_level_generator=orig_level_generator,
        level_solver=None,
        level_mutator=None,
        level_metrics=None,
        eval_level_generators={},
        fixed_eval_levels={},
        heatmap_splayer_fn=None,
        classify_level_is_shift=None,
        net_cnn_type=net_cnn_type,
        net_rnn_type=net_rnn_type,
        net_width=net_width,
        ued=ued,
        prob_shift=0.0,
        num_train_levels=plr_buffer_size,
        plr_buffer_size=plr_buffer_size,
        plr_temperature=plr_temperature,
        plr_staleness_coeff=plr_staleness_coeff,
        plr_prob_replay=plr_prob_replay,
        plr_regret_estimator=plr_regret_estimator,
        plr_robust=plr_robust,
        train_proxy_critic=train_proxy_critic,
        plr_proxy_shaping=plr_proxy_shaping,
        proxy_name=proxy_name,
        plr_proxy_shaping_coeff=plr_proxy_shaping_coeff,
        clipping=clipping,
        eta_schedule=False,
        eta_schedule_time=0.0,
        ppo_lr=ppo_lr,
        ppo_gamma=ppo_gamma,
        ppo_clip_eps=ppo_clip_eps,
        ppo_gae_lambda=ppo_gae_lambda,
        ppo_entropy_coeff=ppo_entropy_coeff,
        ppo_critic_coeff=ppo_critic_coeff,
        ppo_proxy_critic_coeff=ppo_proxy_critic_coeff,
        ppo_max_grad_norm=ppo_max_grad_norm,
        ppo_lr_annealing=ppo_lr_annealing,
        num_minibatches_per_epoch=num_minibatches_per_epoch,
        num_epochs_per_cycle=num_epochs_per_cycle,
        num_total_env_steps=num_total_env_steps,
        num_env_steps_per_cycle=num_env_steps_per_cycle,
        num_parallel_envs=num_parallel_envs,
        console_log=console_log,
        wandb_log=wandb_log,
        log_gifs=log_gifs,
        log_imgs=log_imgs,
        log_hists=log_hists,
        num_cycles_per_log=num_cycles_per_log,
        num_cycles_per_eval=num_cycles_per_eval,
        num_cycles_per_gifs=num_cycles_per_gifs,
        num_cycles_per_big_eval=num_cycles_per_big_eval,
        evals_num_env_steps=evals_num_env_steps,
        evals_num_levels=evals_num_levels,
        gif_grid_width=gif_grid_width,
        checkpointing=checkpointing,
        keep_all_checkpoints=keep_all_checkpoints,
        max_num_checkpoints=max_num_checkpoints,
        num_cycles_per_checkpoint=num_cycles_per_checkpoint,
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
    ued: str = "plr",                       # dr, dr-finite, plr, plr-parallel
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
    train_proxy_critic: bool = False,
    plr_proxy_shaping: bool = False,
    proxy_name: str = "leader_distance",
    plr_proxy_shaping_coeff: float = 0.5,
    clipping: bool = True,
    # PPO hyperparameters
    ppo_lr: float = 0.00005,                # learning rate
    ppo_gamma: float = 0.999,               # discount rate
    ppo_clip_eps: float = 0.1,
    ppo_gae_lambda: float = 0.95,
    ppo_entropy_coeff: float = 0.001,
    ppo_critic_coeff: float = 0.5,
    ppo_proxy_critic_coeff: float = 0.5,
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
        print("  mixing level generators with {prob_shift=}...")
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
        seed=seed,
        env=env,
        train_level_generator=train_level_generator,
        level_solver=None,
        level_mutator=None,
        level_metrics=None,
        eval_level_generators=eval_level_generators,
        fixed_eval_levels={},
        heatmap_splayer_fn=None,
        classify_level_is_shift=classify_level_is_shift,
        net_cnn_type=net_cnn_type,
        net_rnn_type=net_rnn_type,
        net_width=net_width,
        ued=ued,
        prob_shift=prob_shift,
        num_train_levels=num_train_levels,
        plr_buffer_size=plr_buffer_size,
        plr_temperature=plr_temperature,
        plr_staleness_coeff=plr_staleness_coeff,
        plr_prob_replay=plr_prob_replay,
        plr_regret_estimator=plr_regret_estimator,
        plr_robust=plr_robust,
        train_proxy_critic=train_proxy_critic,
        plr_proxy_shaping=plr_proxy_shaping,
        proxy_name=proxy_name,
        plr_proxy_shaping_coeff=plr_proxy_shaping_coeff,
        clipping=clipping,
        eta_schedule=False,
        eta_schedule_time=0.0,
        ppo_lr=ppo_lr,
        ppo_gamma=ppo_gamma,
        ppo_clip_eps=ppo_clip_eps,
        ppo_gae_lambda=ppo_gae_lambda,
        ppo_entropy_coeff=ppo_entropy_coeff,
        ppo_critic_coeff=ppo_critic_coeff,
        ppo_proxy_critic_coeff=ppo_proxy_critic_coeff,
        ppo_max_grad_norm=ppo_max_grad_norm,
        ppo_lr_annealing=ppo_lr_annealing,
        num_minibatches_per_epoch=num_minibatches_per_epoch,
        num_epochs_per_cycle=num_epochs_per_cycle,
        num_total_env_steps=num_total_env_steps,
        num_env_steps_per_cycle=num_env_steps_per_cycle,
        num_parallel_envs=num_parallel_envs,
        console_log=console_log,
        wandb_log=wandb_log,
        log_gifs=log_gifs,
        log_imgs=log_imgs,
        log_hists=log_hists,
        num_cycles_per_log=num_cycles_per_log,
        num_cycles_per_eval=num_cycles_per_eval,
        num_cycles_per_gifs=num_cycles_per_gifs,
        num_cycles_per_big_eval=num_cycles_per_big_eval,
        evals_num_env_steps=evals_num_env_steps,
        evals_num_levels=evals_num_levels,
        gif_grid_width=gif_grid_width,
        checkpointing=checkpointing,
        keep_all_checkpoints=keep_all_checkpoints,
        max_num_checkpoints=max_num_checkpoints,
        num_cycles_per_checkpoint=num_cycles_per_checkpoint,
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
    ued: str = "plr",                       # dr, dr-finite, plr, plr-parallel
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
    train_proxy_critic: bool = False,
    plr_proxy_shaping: bool = False,
    proxy_name: str = "lava",
    plr_proxy_shaping_coeff: float = 0.5,
    clipping: bool = True,
    # PPO hyperparameters
    ppo_lr: float = 0.00005,                # learning rate
    ppo_gamma: float = 0.999,               # discount rate
    ppo_clip_eps: float = 0.1,
    ppo_gae_lambda: float = 0.95,
    ppo_entropy_coeff: float = 0.001,
    ppo_critic_coeff: float = 0.5,
    ppo_proxy_critic_coeff: float = 0.5,
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
        print("  mixing level generators with {prob_shift=}...")
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
        seed=seed,
        env=env,
        train_level_generator=train_level_generator,
        level_solver=None,
        level_mutator=None,
        level_metrics=None,
        eval_level_generators=eval_level_generators,
        fixed_eval_levels={},
        heatmap_splayer_fn=None,
        classify_level_is_shift=classify_level_is_shift,
        net_cnn_type=net_cnn_type,
        net_rnn_type=net_rnn_type,
        net_width=net_width,
        ued=ued,
        prob_shift=prob_shift,
        num_train_levels=num_train_levels,
        plr_buffer_size=plr_buffer_size,
        plr_temperature=plr_temperature,
        plr_staleness_coeff=plr_staleness_coeff,
        plr_prob_replay=plr_prob_replay,
        plr_regret_estimator=plr_regret_estimator,
        plr_robust=plr_robust,
        train_proxy_critic=train_proxy_critic,
        plr_proxy_shaping=plr_proxy_shaping,
        proxy_name=proxy_name,
        plr_proxy_shaping_coeff=plr_proxy_shaping_coeff,
        clipping=clipping,
        eta_schedule=False,
        eta_schedule_time=0.0,
        ppo_lr=ppo_lr,
        ppo_gamma=ppo_gamma,
        ppo_clip_eps=ppo_clip_eps,
        ppo_gae_lambda=ppo_gae_lambda,
        ppo_entropy_coeff=ppo_entropy_coeff,
        ppo_critic_coeff=ppo_critic_coeff,
        ppo_proxy_critic_coeff=ppo_proxy_critic_coeff,
        ppo_max_grad_norm=ppo_max_grad_norm,
        ppo_lr_annealing=ppo_lr_annealing,
        num_minibatches_per_epoch=num_minibatches_per_epoch,
        num_epochs_per_cycle=num_epochs_per_cycle,
        num_total_env_steps=num_total_env_steps,
        num_env_steps_per_cycle=num_env_steps_per_cycle,
        num_parallel_envs=num_parallel_envs,
        console_log=console_log,
        wandb_log=wandb_log,
        log_gifs=log_gifs,
        log_imgs=log_imgs,
        log_hists=log_hists,
        num_cycles_per_log=num_cycles_per_log,
        num_cycles_per_eval=num_cycles_per_eval,
        num_cycles_per_gifs=num_cycles_per_gifs,
        num_cycles_per_big_eval=num_cycles_per_big_eval,
        evals_num_env_steps=evals_num_env_steps,
        evals_num_levels=evals_num_levels,
        gif_grid_width=gif_grid_width,
        checkpointing=checkpointing,
        keep_all_checkpoints=keep_all_checkpoints,
        max_num_checkpoints=max_num_checkpoints,
        num_cycles_per_checkpoint=num_cycles_per_checkpoint,
    )
