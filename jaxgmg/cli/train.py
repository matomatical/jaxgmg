"""
Launcher for training runs.
"""

from typing import Callable

import jax

from jaxgmg.procgen import maze_generation
from jaxgmg.environments import base
from jaxgmg.environments import cheese_in_the_corner
from jaxgmg.environments import cheese_on_a_dish
from jaxgmg.environments import cheese_on_a_pile
from jaxgmg.environments import keys_and_chests
from jaxgmg.environments import minigrid_maze
from jaxgmg.baselines import ppo
from jaxgmg.baselines import networks
from jaxgmg.baselines import evals
from jaxgmg.baselines import autocurricula

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
    # ued config
    ued: str = "plr",                       # dr, dr-finite, plr, plr-parallel
    prob_shift: float = 0.0,
    # for domain randomisation
    num_train_levels: int = 2048,
    # for plr
    plr_buffer_size: int = 2048,
    plr_temperature: float = 0.1,
    plr_staleness_coeff: float = 0.1,
    plr_prob_replay: float = 0.5,
    plr_regret_estimator: str = "PVL",
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
    # training animation dimensions
    train_gifs: bool = True,
    train_gif_grid_width: int = 16,
    # evals config
    num_cycles_per_eval: int = 64,
    num_eval_levels: int = 256,
    num_env_steps_per_eval: int = 512,
    # big evals config
    num_cycles_per_big_eval: int = 1024,    # roughly 9M env steps
    eval_gif_grid_width: int = 16,
    level_splayer: str = 'mouse',           # or 'cheese' or 'cheese-and-mouse'
    # logging
    num_cycles_per_log: int = 64,
    console_log: bool = True,               # whether to log metrics to stdout
    wandb_log: bool = True,                # whether to log metrics to wandb
    wandb_project: str = "test_plr",
    wandb_entity: str = None,               # e.g. 'krueger-lab-cambridge'
    wandb_group: str = None,
    wandb_name: str = None,
    # checkpointing
    checkpointing: bool = True,             # keep checkpoints? (default: yes)
    keep_all_checkpoints: bool = False,     # if so: keep all of them? (no)
    max_num_checkpoints: int = 1,           # if not: keep only latest n (=1)
    num_cycles_per_checkpoint: int = 512,
    # other
    seed: int = 42,
):
    config = locals() # TODO: pass this to w&b instead of using the wrapper
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

    print("TODO: configure parser and fixed eval levels...")

    # splayers for eval levels
    match level_splayer:
        case 'mouse':
            splayer = cheese_in_the_corner.splay_mouse
        case 'cheese':
            splayer = cheese_in_the_corner.splay_cheese
        case 'cheese-and-mouse':
            splayer = cheese_in_the_corner.splay_cheese_and_mouse 
        case _:
            raise ValueError(f'unknown level splayer {level_splayer!r}')

    ppo_training_run(
        # environment-specific stuff
        seed=seed,
        env=env,
        orig_level_generator=orig_level_generator,
        shift_level_generator=shift_level_generator,
        level_solver=level_solver,
        level_metrics=level_metrics,
        splayer=splayer,
        fixed_eval_levels={},
        # non-environment-specific stuff
        net_cnn_type=net_cnn_type,
        net_rnn_type=net_rnn_type,
        ued=ued,
        prob_shift=prob_shift,
        num_train_levels=num_train_levels,
        plr_buffer_size=plr_buffer_size,
        plr_temperature=plr_temperature,
        plr_staleness_coeff=plr_staleness_coeff,
        plr_prob_replay=plr_prob_replay,
        plr_regret_estimator=plr_regret_estimator,
        ppo_lr=ppo_lr,
        ppo_gamma=ppo_gamma,
        ppo_clip_eps=ppo_clip_eps,
        ppo_gae_lambda=ppo_gae_lambda,
        ppo_entropy_coeff=ppo_entropy_coeff,
        ppo_critic_coeff=ppo_critic_coeff,
        ppo_max_grad_norm=ppo_max_grad_norm,
        ppo_lr_annealing=ppo_lr_annealing,
        num_minibatches_per_epoch=num_minibatches_per_epoch,
        num_epochs_per_cycle=num_epochs_per_cycle,
        num_total_env_steps=num_total_env_steps,
        num_env_steps_per_cycle=num_env_steps_per_cycle,
        num_parallel_envs=num_parallel_envs,
        train_gifs=train_gifs,
        train_gif_grid_width=train_gif_grid_width,
        num_cycles_per_eval=num_cycles_per_eval,
        num_eval_levels=num_eval_levels,
        num_env_steps_per_eval=num_env_steps_per_eval,
        num_cycles_per_big_eval=num_cycles_per_big_eval,
        eval_gif_grid_width=eval_gif_grid_width,
        num_cycles_per_log=num_cycles_per_log,
        console_log=console_log,
        wandb_log=wandb_log,
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
    max_cheese_radius: int = 0,
    max_cheese_radius_shift: int = 12,
    obs_level_of_detail: int = 0,           # 0 = bool; 1, 3, 4, or 8 = rgb
    img_level_of_detail: int = 1,           # obs_ is for train, img_ for gifs
    env_penalize_time: bool = False,
    # policy config
    net_cnn_type: str = "large",
    net_rnn_type: str = "ff",
    # ued config
    ued: str = "plr",                       # dr, dr-finite, plr, plr-parallel
    prob_shift: float = 0.0,
    # for domain randomisation
    num_train_levels: int = 2048,
    # for plr
    plr_buffer_size: int = 2048,
    plr_temperature: float = 0.1,
    plr_staleness_coeff: float = 0.1,
    plr_prob_replay: float = 0.5,
    plr_regret_estimator: str = "PVL",      # "PVL" or "absGAE" (todo "maxMC")
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
    # training animation dimensions
    train_gifs: bool = True,
    train_gif_grid_width: int = 16,
    # evals config
    num_cycles_per_eval: int = 64,
    num_eval_levels: int = 256,
    num_env_steps_per_eval: int = 512,
    # big evals config
    num_cycles_per_big_eval: int = 1024,    # roughly 9M env steps
    eval_gif_grid_width: int = 16,
    level_splayer: str = 'mouse',           # or 'cheese' or 'cheese-and-mouse'
    # logging
    num_cycles_per_log: int = 64,
    console_log: bool = True,               # whether to log metrics to stdout
    wandb_log: bool = True,                 # whether to log metrics to wandb
    wandb_project: str = "plr_dish",
    wandb_entity: str = None,
    wandb_group: str = None,
    wandb_name: str = None,
    # checkpointing
    checkpointing: bool = True,             # keep checkpoints? (default: yes)
    keep_all_checkpoints: bool = False,     # if so: keep all of them? (no)
    max_num_checkpoints: int = 1,           # if not: keep only latest n (=1)
    num_cycles_per_checkpoint: int = 512,
    # other
    seed: int = 42,
):
    config = locals() # TODO: pass this to w&b instead of using the wrapper
    util.print_config(config)

    print("configuring environment...")
    env = cheese_on_a_dish.Env(
        terminate_after_cheese_and_dish=env_terminate_after_dish,
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
        max_cheese_radius=max_cheese_radius,  
    )
    shift_level_generator = cheese_on_a_dish.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        max_cheese_radius=max_cheese_radius_shift,  
    )
    
    print("TODO: implement level solver...")
    
    print("configuring level metrics...")
    level_metrics = cheese_on_a_dish.LevelMetrics(
        env=env,
        discount_rate=ppo_gamma,
    )

    print("TODO: implement level splayers for heatmap evals...")
    
    print("TODO: configure parser and fixed eval levels...")
    
    ppo_training_run(
        # environment-specific stuff
        seed=seed,
        env=env,
        orig_level_generator=orig_level_generator,
        shift_level_generator=shift_level_generator,
        level_solver=None,
        level_metrics=level_metrics,
        splayer=None,
        fixed_eval_levels={},
        # non-environment-specific stuff
        net_cnn_type=net_cnn_type,
        net_rnn_type=net_rnn_type,
        ued=ued,
        prob_shift=prob_shift,
        num_train_levels=num_train_levels,
        plr_buffer_size=plr_buffer_size,
        plr_temperature=plr_temperature,
        plr_staleness_coeff=plr_staleness_coeff,
        plr_prob_replay=plr_prob_replay,
        plr_regret_estimator=plr_regret_estimator,
        ppo_lr=ppo_lr,
        ppo_gamma=ppo_gamma,
        ppo_clip_eps=ppo_clip_eps,
        ppo_gae_lambda=ppo_gae_lambda,
        ppo_entropy_coeff=ppo_entropy_coeff,
        ppo_critic_coeff=ppo_critic_coeff,
        ppo_max_grad_norm=ppo_max_grad_norm,
        ppo_lr_annealing=ppo_lr_annealing,
        num_minibatches_per_epoch=num_minibatches_per_epoch,
        num_epochs_per_cycle=num_epochs_per_cycle,
        num_total_env_steps=num_total_env_steps,
        num_env_steps_per_cycle=num_env_steps_per_cycle,
        num_parallel_envs=num_parallel_envs,
        train_gifs=train_gifs,
        train_gif_grid_width=train_gif_grid_width,
        num_cycles_per_eval=num_cycles_per_eval,
        num_eval_levels=num_eval_levels,
        num_env_steps_per_eval=num_env_steps_per_eval,
        num_cycles_per_big_eval=num_cycles_per_big_eval,
        eval_gif_grid_width=eval_gif_grid_width,
        num_cycles_per_log=num_cycles_per_log,
        console_log=console_log,
        wandb_log=wandb_log,
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
    max_cheese_radius_shift: int = 12,
    # these two are not relevant and can be ignored for now (they may come
    # useful in next implementations)
    max_dish_radius: int = 0,
    max_dish_radius_shift: int= 0,
    # other env stuff
    env_terminate_after_dish: bool = True,
    obs_level_of_detail: int = 0,           # 0 = bool; 1, 3, 4, or 8 = rgb
    img_level_of_detail: int = 1,           # obs_ is for train, img_ for gifs
    env_penalize_time: bool = False,
    # policy config
    net_cnn_type: str = "large",
    net_rnn_type: str = "ff",
    # ued config
    ued: str = "plr",                        # dr, dr-finite, plr, plr-parallel
    prob_shift: float = 0.0,
    # for domain randomisation
    num_train_levels: int = 2048,
    # for plr
    plr_buffer_size: int = 2048,
    plr_temperature: float = 0.1,
    plr_staleness_coeff: float = 0.1,
    plr_prob_replay: float = 0.5,
    plr_regret_estimator: str = "PVL",      # "PVL" or "absGAE" (todo "maxMC")
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
    num_total_env_steps: int = 300_000_000,
    num_env_steps_per_cycle: int = 128,
    num_parallel_envs: int = 256,
    # training animation dimensions
    train_gifs: bool = True,
    train_gif_grid_width: int = 16,
    # evals config
    num_cycles_per_eval: int = 64,
    num_eval_levels: int = 256,
    num_env_steps_per_eval: int = 512,
    # big evals config
    num_cycles_per_big_eval: int = 1024,    # roughly 9M env steps
    eval_gif_grid_width: int = 16,
    level_splayer: str = 'mouse',           # or 'cheese' or 'cheese-and-mouse'
    # logging
    num_cycles_per_log: int = 64,
    console_log: bool = True,               # whether to log metrics to stdout
    wandb_log: bool = False,                # whether to log metrics to wandb
    wandb_project: str = "proxy_test_multiple_dish_final",
    wandb_entity: str = None,
    wandb_group: str = None,
    wandb_name: str = None,
    # checkpointing
    checkpointing: bool = True,             # keep checkpoints? (default: yes)
    keep_all_checkpoints: bool = False,     # if so: keep all of them? (no)
    max_num_checkpoints: int = 1,           # if not: keep only latest n (=1)
    num_cycles_per_checkpoint: int = 512,
    # other
    seed: int = 42,
):
    config = locals() # TODO: pass this to w&b instead of using the wrapper
    util.print_config(config)

    print("configuring environment...")
    env = cheese_on_a_pile.Env(
        terminate_after_cheese_and_dish=env_terminate_after_dish,
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
    
    print("TODO: implement level solver...")
    
    print("configuring level metrics...")
    level_metrics = cheese_on_a_pile.LevelMetrics(
        env=env,
        discount_rate=ppo_gamma,
    )
    
    print("TODO: implement level splayers for heatmap evals...")
    
    print("TODO: configure parser and fixed eval levels...")
    
    ppo_training_run(
        # environment-specific stuff
        seed=seed,
        env=env,
        orig_level_generator=orig_level_generator,
        shift_level_generator=shift_level_generator,
        level_solver=None,
        level_metrics=level_metrics,
        splayer=None,
        fixed_eval_levels={},
        # non-environment-specific stuff
        net_cnn_type=net_cnn_type,
        net_rnn_type=net_rnn_type,
        ued=ued,
        prob_shift=prob_shift,
        num_train_levels=num_train_levels,
        plr_buffer_size=plr_buffer_size,
        plr_temperature=plr_temperature,
        plr_staleness_coeff=plr_staleness_coeff,
        plr_prob_replay=plr_prob_replay,
        plr_regret_estimator=plr_regret_estimator,
        ppo_lr=ppo_lr,
        ppo_gamma=ppo_gamma,
        ppo_clip_eps=ppo_clip_eps,
        ppo_gae_lambda=ppo_gae_lambda,
        ppo_entropy_coeff=ppo_entropy_coeff,
        ppo_critic_coeff=ppo_critic_coeff,
        ppo_max_grad_norm=ppo_max_grad_norm,
        ppo_lr_annealing=ppo_lr_annealing,
        num_minibatches_per_epoch=num_minibatches_per_epoch,
        num_epochs_per_cycle=num_epochs_per_cycle,
        num_total_env_steps=num_total_env_steps,
        num_env_steps_per_cycle=num_env_steps_per_cycle,
        num_parallel_envs=num_parallel_envs,
        train_gifs=train_gifs,
        train_gif_grid_width=train_gif_grid_width,
        num_cycles_per_eval=num_cycles_per_eval,
        num_eval_levels=num_eval_levels,
        num_env_steps_per_eval=num_env_steps_per_eval,
        num_cycles_per_big_eval=num_cycles_per_big_eval,
        eval_gif_grid_width=eval_gif_grid_width,
        num_cycles_per_log=num_cycles_per_log,
        console_log=console_log,
        wandb_log=wandb_log,
        checkpointing=checkpointing,
        keep_all_checkpoints=keep_all_checkpoints,
        max_num_checkpoints=max_num_checkpoints,
        num_cycles_per_checkpoint=num_cycles_per_checkpoint,
    )


@util.wandb_run
def keys(
    # environment config
    env_size: int = 13,
    env_layout: str = 'blocks',
    env_num_keys_min: int = 1,
    env_num_keys_max: int = 3,
    env_num_keys_min_shift: int = 9,
    env_num_keys_max_shift: int = 12,
    env_num_chests_min: int = 12,
    env_num_chests_max: int = 24,
    obs_level_of_detail: int = 0,           # 0 = bool; 1, 3, 4, or 8 = rgb
    img_level_of_detail: int = 1,           # obs_ is for train, img_ for gifs
    env_penalize_time: bool = False,
    # policy config
    net_cnn_type: str = "large",
    net_rnn_type: str = "ff",
    # ued config
    ued: str = "plr",                        # dr, dr-finite, plr, plr-parallel
    prob_shift: float = 0.0,
    # for domain randomisation
    num_train_levels: int = 2048,
    # for plr
    plr_buffer_size: int = 2048,
    plr_temperature: float = 0.1,
    plr_staleness_coeff: float = 0.1,
    plr_prob_replay: float = 0.5,
    plr_regret_estimator: str = "PVL",      # "PVL" or "absGAE" (todo "maxMC")
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
    # training animation dimensions
    train_gifs: bool = True,
    train_gif_grid_width: int = 16,
    # evals config
    num_cycles_per_eval: int = 64,
    num_eval_levels: int = 256,
    num_env_steps_per_eval: int = 512,
    # big evals config
    num_cycles_per_big_eval: int = 1024,    # roughly 9M env steps
    eval_gif_grid_width: int = 16,
    level_splayer: str = 'mouse',           # or 'cheese' or 'cheese-and-mouse'
    # logging
    num_cycles_per_log: int = 64,
    console_log: bool = True,               # whether to log metrics to stdout
    wandb_log: bool = False,                # whether to log metrics to wandb
    wandb_project: str = "test",
    wandb_entity: str = None,
    wandb_group: str = None,
    wandb_name: str = None,
    # checkpointing
    checkpointing: bool = True,             # keep checkpoints? (default: yes)
    keep_all_checkpoints: bool = False,     # if so: keep all of them? (no)
    max_num_checkpoints: int = 1,           # if not: keep only latest n (=1)
    num_cycles_per_checkpoint: int = 512,
    # other
    seed: int = 42,
):
    config = locals() # TODO: pass this to w&b instead of using the wrapper
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
        num_chests_min=env_num_chests_min,
        num_chests_max=env_num_chests_max,
    )
    
    print("TODO: implement level solver...")
    
    print("TODO: implement level metrics...")
    
    print("TODO: implement level splayers for heatmap evals...")
    
    print("TODO: configure parser and fixed eval levels...")
    
    ppo_training_run(
        # environment-specific stuff
        seed=seed,
        env=env,
        orig_level_generator=orig_level_generator,
        shift_level_generator=shift_level_generator,
        level_solver=None,
        level_metrics=None,
        splayer=None,
        fixed_eval_levels={},
        # non-environment-specific stuff
        net_cnn_type=net_cnn_type,
        net_rnn_type=net_rnn_type,
        ued=ued,
        prob_shift=prob_shift,
        num_train_levels=num_train_levels,
        plr_buffer_size=plr_buffer_size,
        plr_temperature=plr_temperature,
        plr_staleness_coeff=plr_staleness_coeff,
        plr_prob_replay=plr_prob_replay,
        plr_regret_estimator=plr_regret_estimator,
        ppo_lr=ppo_lr,
        ppo_gamma=ppo_gamma,
        ppo_clip_eps=ppo_clip_eps,
        ppo_gae_lambda=ppo_gae_lambda,
        ppo_entropy_coeff=ppo_entropy_coeff,
        ppo_critic_coeff=ppo_critic_coeff,
        ppo_max_grad_norm=ppo_max_grad_norm,
        ppo_lr_annealing=ppo_lr_annealing,
        num_minibatches_per_epoch=num_minibatches_per_epoch,
        num_epochs_per_cycle=num_epochs_per_cycle,
        num_total_env_steps=num_total_env_steps,
        num_env_steps_per_cycle=num_env_steps_per_cycle,
        num_parallel_envs=num_parallel_envs,
        train_gifs=train_gifs,
        train_gif_grid_width=train_gif_grid_width,
        num_cycles_per_eval=num_cycles_per_eval,
        num_eval_levels=num_eval_levels,
        num_env_steps_per_eval=num_env_steps_per_eval,
        num_cycles_per_big_eval=num_cycles_per_big_eval,
        eval_gif_grid_width=eval_gif_grid_width,
        num_cycles_per_log=num_cycles_per_log,
        console_log=console_log,
        wandb_log=wandb_log,
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
    env_size_shift: int = 21,
    obs_level_of_detail: int = 0,           # 0 = bool; 1, 3, 4, or 8 = rgb
    img_level_of_detail: int = 1,           # obs_ is for train, img_ for gifs
    env_penalize_time: bool = False,
    # policy config
    net_cnn_type: str = "large",
    net_rnn_type: str = "ff",
    # ued config
    ued: str = "plr",                        # dr, dr-finite, plr, plr-parallel
    prob_shift: float = 0.0,
    # for domain randomisation
    num_train_levels: int = 2048,
    # for plr
    plr_buffer_size: int = 2048,
    plr_temperature: float = 0.1,
    plr_staleness_coeff: float = 0.1,
    plr_prob_replay: float = 0.5,
    plr_regret_estimator: str = "PVL",      # "PVL" or "absGAE" (todo "maxMC")
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
    # training animation dimensions
    train_gifs: bool = True,
    train_gif_grid_width: int = 16,
    # evals config
    num_cycles_per_eval: int = 64,
    num_eval_levels: int = 256,
    num_env_steps_per_eval: int = 512,
    # big evals config
    num_cycles_per_big_eval: int = 1024,    # roughly 9M env steps
    eval_gif_grid_width: int = 16,
    level_splayer: str = 'mouse',           # or 'cheese' or 'cheese-and-mouse'
    # logging
    num_cycles_per_log: int = 64,
    console_log: bool = True,               # whether to log metrics to stdout
    wandb_log: bool = False,                # whether to log metrics to wandb
    wandb_project: str = "test",
    wandb_entity: str = None,
    wandb_group: str = None,
    wandb_name: str = None,
    # checkpointing
    checkpointing: bool = True,             # keep checkpoints? (default: yes)
    keep_all_checkpoints: bool = False,     # if so: keep all of them? (no)
    max_num_checkpoints: int = 1,           # if not: keep only latest n (=1)
    num_cycles_per_checkpoint: int = 512,
    # other
    seed: int = 42,
):
    config = locals() # TODO: pass this to w&b instead of using the wrapper
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
    )
    shift_level_generator = minigrid_maze.LevelGenerator(
        maze_generator=maze_generator,
        height=env_size_shift,
        width=env_size_shift,
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

    ppo_training_run(
        # environment-specific stuff
        seed=seed,
        env=env,
        orig_level_generator=orig_level_generator,
        shift_level_generator=shift_level_generator,
        level_solver=None,
        level_metrics=level_metrics,
        splayer=None,
        fixed_eval_levels=fixed_eval_levels,
        # non-environment-specific stuff
        net_cnn_type=net_cnn_type,
        net_rnn_type=net_rnn_type,
        ued=ued,
        prob_shift=prob_shift,
        num_train_levels=num_train_levels,
        plr_buffer_size=plr_buffer_size,
        plr_temperature=plr_temperature,
        plr_staleness_coeff=plr_staleness_coeff,
        plr_prob_replay=plr_prob_replay,
        plr_regret_estimator=plr_regret_estimator,
        ppo_lr=ppo_lr,
        ppo_gamma=ppo_gamma,
        ppo_clip_eps=ppo_clip_eps,
        ppo_gae_lambda=ppo_gae_lambda,
        ppo_entropy_coeff=ppo_entropy_coeff,
        ppo_critic_coeff=ppo_critic_coeff,
        ppo_max_grad_norm=ppo_max_grad_norm,
        ppo_lr_annealing=ppo_lr_annealing,
        num_minibatches_per_epoch=num_minibatches_per_epoch,
        num_epochs_per_cycle=num_epochs_per_cycle,
        num_total_env_steps=num_total_env_steps,
        num_env_steps_per_cycle=num_env_steps_per_cycle,
        num_parallel_envs=num_parallel_envs,
        train_gifs=train_gifs,
        train_gif_grid_width=train_gif_grid_width,
        num_cycles_per_eval=num_cycles_per_eval,
        num_eval_levels=num_eval_levels,
        num_env_steps_per_eval=num_env_steps_per_eval,
        num_cycles_per_big_eval=num_cycles_per_big_eval,
        eval_gif_grid_width=eval_gif_grid_width,
        num_cycles_per_log=num_cycles_per_log,
        console_log=console_log,
        wandb_log=wandb_log,
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
    # training animation dimensions
    train_gifs: bool = True,
    train_gif_grid_width: int = 8,
    # evals config
    num_cycles_per_eval: int = 1024,
    num_eval_levels: int = 8,
    num_env_steps_per_eval: int = 128,
    # logging
    num_cycles_per_log: int = 16,
    console_log: bool = False,              # whether to log metrics to stdout
    wandb_log: bool = False,                # whether to log metrics to wandb
    wandb_project: str = "test",
    wandb_entity: str = None,
    wandb_group: str = None,
    wandb_name: str = None,
    # checkpointing
    checkpointing: bool = False,
    keep_all_checkpoints: bool = False,
    max_num_checkpoints: int = 1,
    num_cycles_per_checkpoint: int = 512,
    # other
    seed: int = 42,
):
    config = locals() # TODO: pass this to w&b instead of using the wrapper
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

    ppo_training_run(
        # environment-specific stuff
        seed=seed,
        env=env,
        orig_level_generator=orig_level_generator,
        shift_level_generator=None,
        level_solver=None,
        level_metrics=None,
        splayer=None,
        fixed_eval_levels={},
        # non-environment-specific stuff
        net_cnn_type=net_cnn_type,
        net_rnn_type=net_rnn_type,
        ued="dr",
        prob_shift=0.0,
        num_train_levels=None,
        plr_buffer_size=None,
        plr_temperature=None,
        plr_staleness_coeff=None,
        plr_prob_replay=None,
        plr_regret_estimator=None,
        ppo_lr=ppo_lr,
        ppo_gamma=ppo_gamma,
        ppo_clip_eps=ppo_clip_eps,
        ppo_gae_lambda=ppo_gae_lambda,
        ppo_entropy_coeff=ppo_entropy_coeff,
        ppo_critic_coeff=ppo_critic_coeff,
        ppo_max_grad_norm=ppo_max_grad_norm,
        ppo_lr_annealing=ppo_lr_annealing,
        num_minibatches_per_epoch=num_minibatches_per_epoch,
        num_epochs_per_cycle=num_epochs_per_cycle,
        num_total_env_steps=num_total_env_steps,
        num_env_steps_per_cycle=num_env_steps_per_cycle,
        num_parallel_envs=num_parallel_envs,
        train_gifs=train_gifs,
        train_gif_grid_width=train_gif_grid_width,
        num_cycles_per_eval=num_cycles_per_eval,
        num_eval_levels=num_eval_levels,
        num_env_steps_per_eval=num_env_steps_per_eval,
        num_cycles_per_log=num_cycles_per_log,
        num_cycles_per_big_eval=1024,
        eval_gif_grid_width=4,
        console_log=console_log,
        wandb_log=wandb_log,
        checkpointing=checkpointing,
        keep_all_checkpoints=keep_all_checkpoints,
        max_num_checkpoints=max_num_checkpoints,
        num_cycles_per_checkpoint=num_cycles_per_checkpoint,
    )


def ppo_training_run(
    # environment-specific stuff
    seed: int,
    env: base.Env,
    orig_level_generator: base.LevelGenerator,
    shift_level_generator: base.LevelGenerator,
    level_solver: base.LevelSolver | None,
    level_metrics: base.LevelMetrics | None,
    splayer: Callable | None,
    fixed_eval_levels: dict[str, base.Level],
    # policy config
    net_cnn_type: str,
    net_rnn_type: str,
    # ued config
    ued: str,
    prob_shift: float,
    # for domain randomisation
    num_train_levels: int,
    # for plr
    plr_buffer_size: int,
    plr_temperature: float,
    plr_staleness_coeff: float,
    plr_prob_replay: float,
    plr_regret_estimator: str,
    # PPO hyperparameters
    ppo_lr: float,
    ppo_gamma: float,
    ppo_clip_eps: float,
    ppo_gae_lambda: float,
    ppo_entropy_coeff: float,
    ppo_critic_coeff: float,
    ppo_max_grad_norm: float,
    ppo_lr_annealing: bool,
    num_minibatches_per_epoch: int,
    num_epochs_per_cycle: int,
    # training dimensions
    num_total_env_steps: int,
    num_env_steps_per_cycle: int,
    num_parallel_envs: int,
    # training animation dimensions
    train_gifs: bool,
    train_gif_grid_width: int,
    # evals config
    num_cycles_per_eval: int,
    num_eval_levels: int,
    num_env_steps_per_eval: int,
    num_cycles_per_big_eval: int,
    eval_gif_grid_width: int,
    # logging
    num_cycles_per_log: int,
    console_log: bool,
    wandb_log: bool,
    # checkpointing
    checkpointing: bool,
    keep_all_checkpoints: bool,
    max_num_checkpoints: int,
    num_cycles_per_checkpoint: int,
):
    rng = jax.random.PRNGKey(seed=seed)
    rng_setup, rng_train = jax.random.split(rng)


    # mixing level distributions
    if prob_shift > 0.0:
        print("mixing level distributions...")
        train_level_generator = base.MixtureLevelGenerator(
            level_generator1=orig_level_generator,
            level_generator2=shift_level_generator,
            prob_level1=1.0-prob_shift,
        )
    else:
        print("using un-mixed level distribution for training.")
        train_level_generator = orig_level_generator
    
    
    print("configuring curriculum...")
    rng_train_levels, rng_setup = jax.random.split(rng_setup)
    if ued == "dr":
        gen = autocurricula.InfiniteDomainRandomisation(
            level_generator=train_level_generator,
        )
        gen_state = gen.init()
    elif ued == "dr-finite":
        train_levels = train_level_generator.vsample(
            rng_train_levels,
            num_levels=num_train_levels,
        )
        gen = autocurricula.FiniteDomainRandomisation()
        gen_state = gen.init(
            levels=train_levels,
        )
    elif ued == "plr":
        gen = autocurricula.PrioritisedLevelReplay(
            level_generator=train_level_generator,
            level_metrics=level_metrics,
            buffer_size=plr_buffer_size,
            temperature=plr_temperature,
            staleness_coeff=plr_staleness_coeff,
            prob_replay=plr_prob_replay,
            regret_estimator=plr_regret_estimator,
        )
        gen_state = gen.init(
            rng=rng_train_levels,
            batch_size_hint=num_parallel_envs,
        )
    elif ued == "plr-parallel":
        gen = autocurricula.ParallelRobustPrioritisedLevelReplay(
            level_generator=train_level_generator,
            level_metrics=level_metrics,
            buffer_size=plr_buffer_size,
            temperature=plr_temperature,
            staleness_coeff=plr_staleness_coeff,
            regret_estimator=plr_regret_estimator,
        )
        gen_state = gen.init(
            rng=rng_train_levels,
            batch_size_hint=num_parallel_envs,
        )
    else:
        raise ValueError(f"unknown UED algorithm: {ued!r}")
    

    print(f"setting up agent with architecture...")
    # select architecture
    net = networks.Impala(
        num_actions=env.num_actions,
        cnn_type=net_cnn_type,
        rnn_type=net_rnn_type,
    )
    print(net)
    # initialise the network
    rng_model_init, rng = jax.random.split(rng)
    rng_example_level, rng = jax.random.split(rng)
    example_level=train_level_generator.sample(rng_example_level)
    net_init_params, net_init_state = net.init_params_and_state(
        rng=rng_model_init,
        obs_type=env.obs_type(level=example_level),
    )


    evals_dict = {}


    print("generating on-distribution eval levels...")
    rng_eval_on_levels, rng_setup = jax.random.split(rng_setup)
    eval_on_levels = orig_level_generator.vsample(
        rng_eval_on_levels,
        num_levels=num_eval_levels,
    )
    if level_solver is not None:
        print("  also solving them...")
        eval_on_benchmark_returns = level_solver.vmap_level_value(
            level_solver.vmap_solve(eval_on_levels),
            eval_on_levels,
        )
        eval_on_level_set = evals.FixedLevelsEvalWithBenchmarkReturns(
            num_levels=num_eval_levels,
            levels=eval_on_levels,
            benchmarks=eval_on_benchmark_returns,
            num_steps=num_env_steps_per_eval,
            discount_rate=ppo_gamma,
            env=env,
        )
    else:
        eval_on_level_set = evals.FixedLevelsEval(
            num_levels=num_eval_levels,
            levels=eval_on_levels,
            num_steps=num_env_steps_per_eval,
            discount_rate=ppo_gamma,
            env=env,
        )
    evals_dict[('on_dist_levels', num_cycles_per_eval)] = eval_on_level_set

    
    if shift_level_generator is not None:
        print("generating off-distribution eval levels...")
        rng_eval_off_levels, rng_setup = jax.random.split(rng_setup)
        eval_off_levels = shift_level_generator.vsample(
            rng_eval_off_levels,
            num_levels=num_eval_levels,
        )
        if level_solver is not None:
            print("  also solving them...")
            eval_off_benchmark_returns = level_solver.vmap_level_value(
                level_solver.vmap_solve(eval_off_levels),
                eval_off_levels,
            )
            eval_off_level_set = evals.FixedLevelsEvalWithBenchmarkReturns(
                num_levels=num_eval_levels,
                num_steps=num_env_steps_per_eval,
                discount_rate=ppo_gamma,
                levels=eval_off_levels,
                benchmarks=eval_off_benchmark_returns,
                env=env,
            )
        else:
            eval_off_level_set = evals.FixedLevelsEval(
                num_levels=num_eval_levels,
                num_steps=num_env_steps_per_eval,
                discount_rate=ppo_gamma,
                levels=eval_off_levels,
                env=env,
            )
        evals_dict[('off_dist_levels', num_cycles_per_eval)] = eval_off_level_set

    
    if fixed_eval_levels:
        print("configuring evals for fixed eval levels...")
        fixed_evals = {
            ('fixed/' + level_name, num_cycles_per_eval):
                evals.SingleLevelEval(
                    num_steps=num_env_steps_per_eval,
                    discount_rate=ppo_gamma,
                    level=level,
                    env=env,
                )
            for level_name, level in fixed_eval_levels.items()
        }
        evals_dict.update(fixed_evals) 

    
    print("configuring rollout recorders for those levels...")
    eval_on_rollouts = evals.AnimatedRolloutsEval(
        num_levels=num_eval_levels,
        levels=eval_on_levels,
        num_steps=env.max_steps_in_episode,
        gif_grid_width=eval_gif_grid_width,
        env=env,
    )
    evals_dict[('on_dist_rollouts', num_cycles_per_big_eval)] = eval_on_rollouts
    if shift_level_generator is not None:
        eval_off_rollouts = evals.AnimatedRolloutsEval(
            num_levels=num_eval_levels,
            levels=eval_off_levels,
            num_steps=env.max_steps_in_episode,
            gif_grid_width=eval_gif_grid_width,
            env=env,
        )
        evals_dict[('off_dist_rollouts', num_cycles_per_big_eval)] = eval_off_rollouts


    print("generating splayed eval level sets for heatmaps...")
    def make_heatmap_evals(level, name):
        if splayer is not None:
            splayset = splayer(level)
            return {
                (name+"_static_heatmap", num_cycles_per_big_eval):
                    evals.ActorCriticHeatmapVisualisationEval(
                        *splayset,
                        env=env,
                    ),
                (name+"_rollout_heatmap", num_cycles_per_big_eval):
                    evals.RolloutHeatmapVisualisationEval(
                        *splayset,
                        env=env,
                        discount_rate=ppo_gamma,
                        num_steps=num_env_steps_per_eval,
                    ),
            }
        else:
            return {}
    evals_dict.update(make_heatmap_evals(
        name="eval_on_0",
        level=jax.tree.map(lambda x: x[0], eval_on_levels),
    ))
    evals_dict.update(make_heatmap_evals(
        name="eval_on_1",
        level=jax.tree.map(lambda x: x[1], eval_on_levels),
    ))
    if shift_level_generator is not None:
        evals_dict.update(make_heatmap_evals(
            name="eval_off_0",
            level=jax.tree.map(lambda x: x[0], eval_off_levels),
        ))
        evals_dict.update(make_heatmap_evals(
            name="eval_off_1",
            level=jax.tree.map(lambda x: x[1], eval_off_levels),
        ))
    
    
    # launch the ppo training run
    # TODO: maybe this function should manage the wandb run?
    ppo.run(
        rng=rng_train,
        # environment
        env=env,
        # level distributions
        gen=gen,
        gen_state=gen_state,
        # actor critic network
        net=net,
        net_init_params=net_init_params,
        net_init_state=net_init_state,
        # evals
        evals_dict=evals_dict,
        # algorithm
        ppo_lr=ppo_lr,
        ppo_gamma=ppo_gamma,
        ppo_clip_eps=ppo_clip_eps,
        ppo_gae_lambda=ppo_gae_lambda,
        ppo_entropy_coeff=ppo_entropy_coeff,
        ppo_critic_coeff=ppo_critic_coeff,
        ppo_max_grad_norm=ppo_max_grad_norm,
        ppo_lr_annealing=ppo_lr_annealing,
        num_minibatches_per_epoch=num_minibatches_per_epoch,
        num_epochs_per_cycle=num_epochs_per_cycle,
        # training dimensions
        num_total_env_steps=num_total_env_steps,
        num_env_steps_per_cycle=num_env_steps_per_cycle,
        num_parallel_envs=num_parallel_envs,
        # training animation dimensions
        train_gifs=train_gifs,
        train_gif_grid_width=train_gif_grid_width,
        # logging
        num_cycles_per_log=num_cycles_per_log,
        console_log=console_log,
        wandb_log=wandb_log,
        # checkpointing
        checkpointing=checkpointing,
        keep_all_checkpoints=keep_all_checkpoints,
        max_num_checkpoints=max_num_checkpoints,
        num_cycles_per_checkpoint=num_cycles_per_checkpoint,
    )
    print("training run complete.")


