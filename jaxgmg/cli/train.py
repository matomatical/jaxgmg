"""
Launcher for training runs.
"""

import jax

from jaxgmg.procgen import maze_generation
from jaxgmg.environments import cheese_in_the_corner
from jaxgmg.baselines import ppo

from jaxgmg import util


@util.wandb_run # inits wandb and syncs the arguments with wandb.config
def corner(
    # environment config
    env_size: int = 9,
    env_layout: str = 'blocks',
    env_corner_size: int = 1,
    env_level_of_detail: int = 0,           # 0 = bool; 1, 3, 4, or 8 = rgb
    # policy config
    net: str = "relu",
    # PPO hyperparameters
    ppo_lr: float = 0.0005,                 # learning rate
    ppo_gamma: float = 0.999,               # discount rate
    ppo_clip_eps: float = 0.2,
    ppo_gae_lambda: float = 0.95,
    ppo_entropy_coeff: float = 0.001,
    ppo_critic_coeff: float = 0.5,
    ppo_max_grad_norm: float = 0.5,
    ppo_lr_annealing: bool = False,
    num_minibatches_per_epoch: int = 1,
    num_epochs_per_cycle: int = 5,
    # training dimensions
    num_total_env_steps: int = 20_000_000,
    num_env_steps_per_cycle: int = 256,
    num_parallel_envs: int = 32,
    fixed_train_levels: bool = False,
    num_train_levels: int = 2048,
    # training animation dimensions
    train_gifs: bool = True,
    train_gif_grid_width: int = 8,
    train_gif_level_of_detail: int = 1,
    # evals config
    num_cycles_per_eval: int = 64,
    num_eval_levels: int = 256,
    num_env_steps_per_eval: int = 512,
    # big evals config
    num_cycles_per_big_eval: int = 1024,    # roughly 9M env steps
    eval_gif_grid_width: int = 16,
    eval_gif_level_of_detail: int = 1,      # 1, 3, 4 or 8
    level_splayer: str = 'mouse',           # or 'cheese' or 'cheese-and-mouse'
    # logging
    num_cycles_per_log: int = 64,
    save_files_to: str = "logs/",
    console_log: bool = True,               # whether to log metrics to stdout
    wandb_log: bool = False,                # whether to log metrics to wandb
    wandb_project: str = "test",
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
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    rng_setup, rng_train = jax.random.split(rng)

    print("setting up environment...")
    env = cheese_in_the_corner.Env(
        obs_level_of_detail=env_level_of_detail,
        penalize_time=False,
    )

    print(f"generating training level distribution...")
    maze_generator = maze_generation.get_generator_class_from_name(
        name=env_layout,
    )()
    train_level_generator = cheese_in_the_corner.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        corner_size=env_corner_size,
    )
    rng_train_levels, rng_setup = jax.random.split(rng_setup)
    if fixed_train_levels:
        train_levels = train_level_generator.vsample(
            rng_train_levels,
            num_levels=num_train_levels,
        )
        train_level_set = ppo.FixedTrainLevelSet(
            num_levels=num_train_levels,
            levels=train_levels,
        )
    else:
        train_level_set = ppo.OnDemandTrainLevelSet(
            level_generator=train_level_generator,
        )

    print(f"generating some eval levels with baselines...")
    level_solver = cheese_in_the_corner.LevelSolver(
        env=env,
        discount_rate=ppo_gamma,
    )
    # on distribution
    rng_eval_on_levels, rng_setup = jax.random.split(rng_setup)
    eval_on_levels = train_level_generator.vsample(
        rng_eval_on_levels,
        num_levels=num_eval_levels,
    )
    eval_on_benchmark_returns = level_solver.vmap_level_value(
        level_solver.vmap_solve(eval_on_levels),
        eval_on_levels,
    )
    eval_on_level_set = ppo.FixedLevelsEval(
        num_levels=num_eval_levels,
        levels=eval_on_levels,
        benchmarks=eval_on_benchmark_returns,
        num_steps=num_env_steps_per_eval,
        discount_rate=ppo_gamma,
        env=env,
    )

    # off distribution
    shift_level_generator = cheese_in_the_corner.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        corner_size=env_size-2,
    )
    rng_eval_off_levels, rng_setup = jax.random.split(rng_setup)
    eval_off_levels = shift_level_generator.vsample(
        rng_eval_off_levels,
        num_levels=num_eval_levels,
    )
    eval_off_benchmark_returns = level_solver.vmap_level_value(
        level_solver.vmap_solve(eval_off_levels),
        eval_off_levels,
    )
    eval_off_level_set = ppo.FixedLevelsEval(
        num_levels=num_eval_levels,
        num_steps=num_env_steps_per_eval,
        discount_rate=ppo_gamma,
        levels=eval_off_levels,
        benchmarks=eval_off_benchmark_returns,
        env=env,
    )
    
    # gif animations from those levels
    eval_on_animation = ppo.AnimatedRolloutsEval(
        num_levels=num_eval_levels,
        levels=eval_on_levels,
        num_steps=env.max_steps_in_episode,
        gif_grid_width=eval_gif_grid_width,
        gif_level_of_detail=eval_gif_level_of_detail,
        env=env,
    )
    eval_off_animation = ppo.AnimatedRolloutsEval(
        num_levels=num_eval_levels,
        levels=eval_off_levels,
        num_steps=env.max_steps_in_episode,
        gif_grid_width=eval_gif_grid_width,
        gif_level_of_detail=eval_gif_level_of_detail,
        env=env,
    )

    # splayed eval levels
    match level_splayer:
        case 'mouse':
            splay = cheese_in_the_corner.splay_mouse
        case 'cheese':
            splay = cheese_in_the_corner.splay_cheese
        case 'cheese-and-mouse':
            splay = cheese_in_the_corner.splay_cheese_and_mouse 
        case _:
            raise ValueError(f'unknown level splayer {level_splayer!r}')
    eval_on_heatmap_0 = ppo.HeatmapVisualisationEval(
        *splay(jax.tree.map(lambda x: x[0], eval_on_levels)),
        num_steps=num_env_steps_per_eval,
        discount_rate=ppo_gamma,
        env=env,
    )
    eval_on_heatmap_1 = ppo.HeatmapVisualisationEval(
        *splay(jax.tree.map(lambda x: x[1], eval_on_levels)),
        num_steps=num_env_steps_per_eval,
        discount_rate=ppo_gamma,
        env=env,
    )
    eval_off_heatmap_0 = ppo.HeatmapVisualisationEval(
        *splay(jax.tree.map(lambda x: x[0], eval_off_levels)),
        num_steps=num_env_steps_per_eval,
        discount_rate=ppo_gamma,
        env=env,
    )
    eval_off_heatmap_1 = ppo.HeatmapVisualisationEval(
        *splay(jax.tree.map(lambda x: x[1], eval_off_levels)),
        num_steps=num_env_steps_per_eval,
        discount_rate=ppo_gamma,
        env=env,
    )

    ppo.run(
        rng=rng_train,
        # environment and level distributions
        env=env,
        train_level_set=train_level_set,
        evals_dict={
            'on_distribution': eval_on_level_set,
            'off_distribution': eval_off_level_set,
        },
        big_evals_dict={
            'on_distribution_animations': eval_on_animation,
            'off_distribution_animations': eval_off_animation,
            'on_distribution_level_0_heatmaps': eval_on_heatmap_0,
            'on_distribution_level_1_heatmaps': eval_on_heatmap_1,
            'off_distribution_eval_level_0_heatmaps': eval_off_heatmap_0,
            'off_distribution_eval_level_1_heatmaps': eval_off_heatmap_1,
        },
        # architecture
        net=net,
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
        train_gif_level_of_detail=train_gif_level_of_detail,
        # evals
        num_cycles_per_eval=num_cycles_per_eval,
        num_cycles_per_big_eval=num_cycles_per_big_eval,
        # logging
        num_cycles_per_log=num_cycles_per_log,
        save_files_to=save_files_to,
        console_log=console_log,
        wandb_log=wandb_log,
        # checkpointing
        checkpointing=checkpointing,
        keep_all_checkpoints=keep_all_checkpoints,
        max_num_checkpoints=max_num_checkpoints,
        num_cycles_per_checkpoint=num_cycles_per_checkpoint,
    )
    # (the decorator finishes the wandb run for us, so no need to do that)
    print("training run complete.")


