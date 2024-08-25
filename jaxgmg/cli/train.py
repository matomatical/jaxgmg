"""
Launcher for training runs.
"""

import jax

from jaxgmg.procgen import maze_generation
from jaxgmg.environments import cheese_in_the_corner
from jaxgmg.environments import cheese_on_a_dish
from jaxgmg.environments import cheese_on_a_pile
from jaxgmg.environments import keys_and_chests
from jaxgmg.environments import minigrid_maze
from jaxgmg.baselines import train

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
    if prob_shift > 0.0:
        print("  mixing level generators with {prob_shift=}...")
        train_level_generator = MixtureLevelGenerator(
            level_generator1=orig_level_generator,
            level_generator2=shift_level_generator,
            prob_level1=1.0-prob_shift,
        )
    else:
        train_level_generator = orig_level_generator
    
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
    
    print("configuring fixed eval levels...")
    fixed_eval_levels = {
        "random0": orig_level_generator.sample(jax.random.key(0)),
        "random1": orig_level_generator.sample(jax.random.key(1)),
        "random2": orig_level_generator.sample(jax.random.key(2)),
        "random3": orig_level_generator.sample(jax.random.key(3)),
    }

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
        level_metrics=level_metrics,
        eval_level_generators=eval_level_generators,
        fixed_eval_levels=fixed_eval_levels,
        heatmap_splayer_fn=splayer_fn,
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
    if prob_shift > 0.0:
        print("  mixing level generators with {prob_shift=}...")
        train_level_generator = MixtureLevelGenerator(
            level_generator1=orig_level_generator,
            level_generator2=shift_level_generator,
            prob_level1=1.0-prob_shift,
        )
    else:
        train_level_generator = orig_level_generator
    
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
    
    print("configuring level metrics...")
    level_metrics = cheese_on_a_dish.LevelMetrics(
        env=env,
        discount_rate=ppo_gamma,
    )

    print("TODO: implement level splayers for heatmap evals...")
    
    print("TODO: configure parser and fixed eval levels...")
    
    train.run(
        seed=seed,
        env=env,
        train_level_generator=train_level_generator,
        level_solver=None,
        level_metrics=level_metrics,
        eval_level_generators=eval_level_generators,
        fixed_eval_levels={},
        heatmap_splayer_fn=None,
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
    if prob_shift > 0.0:
        print("  mixing level generators with {prob_shift=}...")
        train_level_generator = MixtureLevelGenerator(
            level_generator1=orig_level_generator,
            level_generator2=shift_level_generator,
            prob_level1=1.0-prob_shift,
        )
    else:
        train_level_generator = orig_level_generator
    
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
    
    print("configuring level metrics...")
    level_metrics = cheese_on_a_pile.LevelMetrics(
        env=env,
        discount_rate=ppo_gamma,
    )
    
    print("TODO: implement level splayers for heatmap evals...")
    
    print("TODO: configure parser and fixed eval levels...")
    
    train.run(
        seed=seed,
        env=env,
        train_level_generator=train_level_generator,
        level_solver=None,
        level_metrics=level_metrics,
        eval_level_generators=eval_level_generators,
        fixed_eval_levels={},
        heatmap_splayer_fn=None,
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
    if prob_shift > 0.0:
        print("  mixing level generators with {prob_shift=}...")
        train_level_generator = MixtureLevelGenerator(
            level_generator1=orig_level_generator,
            level_generator2=shift_level_generator,
            prob_level1=1.0-prob_shift,
        )
    else:
        train_level_generator = orig_level_generator
    
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
        level_metrics=None,
        eval_level_generators=eval_level_generators,
        fixed_eval_levels={},
        heatmap_splayer_fn=None,
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
    if prob_shift > 0.0:
        print("  mixing level generators with {prob_shift=}...")
        train_level_generator = MixtureLevelGenerator(
            level_generator1=orig_level_generator,
            level_generator2=shift_level_generator,
            prob_level1=1.0-prob_shift,
        )
    else:
        train_level_generator = orig_level_generator
    
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
        level_metrics=level_metrics,
        eval_level_generators=eval_level_generators,
        fixed_eval_levels=fixed_eval_levels,
        heatmap_splayer_fn=None,
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
    # curriculum
    ued: str = "dr",
    plr_buffer_size: int = 2048,
    plr_temperature: float = 0.1,
    plr_staleness_coeff: float = 0.1,
    plr_prob_replay: float = 0.5,
    plr_regret_estimator: str = "PVL",
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

    train.run(
        seed=seed,
        env=env,
        train_level_generator=orig_level_generator,
        level_solver=None,
        level_metrics=None,
        eval_level_generators={},
        fixed_eval_levels={},
        heatmap_splayer_fn=None,
        net_cnn_type=net_cnn_type,
        net_rnn_type=net_rnn_type,
        ued=ued,
        prob_shift=0.0,
        num_train_levels=plr_buffer_size,
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


