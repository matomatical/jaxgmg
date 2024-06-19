"""
Launcher for training runs.
"""

import jax

from jaxgmg.procgen import maze_generation
from jaxgmg.environments import cheese_in_the_corner
# from jaxgmg.environments import cheese_on_a_dish
# from jaxgmg.environments import keys_and_chests
# from jaxgmg.environments import monster_world
# from jaxgmg.environments import lava_land
# from jaxgmg.environments import follow_me
from jaxgmg.baselines import ppo
from jaxgmg.cli import util


@util.wandb_run # inits wandb and syncs the arguments with wandb.config
def corner(
    # environment config
    env_size: int = 9,
    env_layout: str = 'blocks',
    env_corner_size: int = 1,
    env_rgb: bool = False,
    # policy config
    net: str = "relu",
    # PPO hyperparameters
    ppo_lr: float = 1e-4,               # learning rate
    ppo_gamma: float = 0.995,           # discount rate
    ppo_clip_eps: float = 0.2,
    ppo_gae_lambda: float = 0.98,
    ppo_entropy_coeff: float = 1e-3,
    ppo_critic_coeff: float = 0.5,
    ppo_max_grad_norm: float = 0.5,
    ppo_lr_annealing: bool = False,
    num_minibatches_per_epoch: int = 1,
    num_epochs_per_cycle: int = 5,
    # training dimensions
    num_total_env_steps: int = 20_000_000,
    num_env_steps_per_cycle: int = 256,
    num_parallel_envs: int = 32,
    num_train_levels: int = 2048,
    # evaluation dimensions
    num_cycles_per_eval: int = 64,
    num_eval_levels: int = 256,
    num_env_steps_per_eval: int = 1024,
    # console logging
    console_log: bool = True,           # whether to log metrics to stdout
    num_cycles_per_log: int = 64,
    wandb_log: bool = False,            # whether to log metrics to wandb
    # wandb location (forwarded to wandb by the wrapper)
    wandb_project: str = "test",
    wandb_group: str = None,
    wandb_name: str = None,
    # gif animations for training/eval rollouts
    train_gifs: bool = False,
    eval_gifs: bool = False,
    num_cycles_per_gif: int = 256,
    gif_grid_width: int = 16,
    rgb_gifs: bool = False,             # force gifs rgb even if obs are bool
    # output save directory
    save_files_to: str = "logs/",
    # other
    seed: int = 42,
):
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    rng_setup, rng_train = jax.random.split(rng)


    print("setting up environment...")
    env = cheese_in_the_corner.Env(
        rgb=env_rgb,
        penalize_time=False,
    )

    print(f"generating {num_train_levels} training levels...")
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
    train_levels = train_level_generator.vsample(
        rng_train_levels,
        num_levels=num_train_levels,
    )
    
    print(f"generating {num_eval_levels} on-distribution eval levels...")
    rng_eval_on_levels, rng_setup = jax.random.split(rng_setup)
    eval_on_levels = train_level_generator.vsample(
        rng_eval_on_levels,
        num_levels=num_eval_levels,
    )
    
    print(f"generating {num_eval_levels} off-distribution eval levels...")
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

    ppo.run(
        rng=rng_train,
        # environment and level distributions
        env=env,
        train_levels=train_levels,
        eval_levels_dict={
            'on-distribution': eval_on_levels,
            'off-distribution': eval_off_levels,
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
        num_total_env_steps=num_total_env_steps,
        num_env_steps_per_cycle=num_env_steps_per_cycle,
        num_parallel_envs=num_parallel_envs,
        num_cycles_per_eval=num_cycles_per_eval,
        num_env_steps_per_eval=num_env_steps_per_eval,
        # logging
        console_log=console_log,
        wandb_log=wandb_log,
        num_cycles_per_log=num_cycles_per_log,
        train_gifs=train_gifs,
        eval_gifs=eval_gifs,
        num_cycles_per_gif=num_cycles_per_gif,
        gif_grid_width=gif_grid_width,
        rgb_gifs=rgb_gifs,
        save_files_to=save_files_to,
    )
    # (the decorator finishes the wandb run for us, so no need to do that)
    print("training run complete.")

