"""
Launcher for training re-evaluation runs.
"""

import jax

from jaxgmg.procgen import maze_generation
from jaxgmg.environments import cheese_in_the_corner
from jaxgmg.baselines import ppo

from jaxgmg import util


def corner(
    checkpoint_path: str,
    checkpoint_number: int,
    # environment config
    env_size: int = 9,
    env_layout: str = 'blocks',
    env_corner_size: int = 1,
    env_terminate_after_corner: bool = False,
    env_level_of_detail: int = 0,           # 0 = bool; 1, 3, 4, or 8 = rgb
    # policy config
    net: str = 'relu',
    # evals config
    num_cycles_per_eval: int = 64,
    num_eval_levels: int = 256,
    num_env_steps_per_eval: int = 512,
    eval_gif_grid_width: int = 16,
    eval_gif_level_of_detail: int = 1,      # 1, 3, 4 or 8
    level_splayer: str = 'mouse',           # or 'cheese' or 'cheese-and-mouse'
    ppo_gamma: float = 0.999,
    # logging
    save_files_to: str = "evals/",
    # other
    seed: int = 42,
):
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    rng_setup, rng_eval = jax.random.split(rng)

    print("setting up environment...")
    env = cheese_in_the_corner.Env(
        obs_level_of_detail=env_level_of_detail,
        penalize_time=False,
        terminate_after_cheese_and_corner=env_terminate_after_corner,
    )

    print("configuring training level generator...")
    maze_generator = maze_generation.get_generator_class_from_name(
        name=env_layout,
    )()
    train_level_generator = cheese_in_the_corner.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        corner_size=env_corner_size,
    )
    rng_train_level, rng_setup = jax.random.split(rng_setup)
    example_level = train_level_generator.sample(rng_train_level)

    print(f"generating some eval levels with baselines...")
    level_solver = cheese_in_the_corner.LevelSolver(
        env=env,
        discount_rate=ppo_gamma,
    )
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

    # launch the evals
    ppo.eval_checkpoint(
        rng=rng_eval,
        checkpoint_path=checkpoint_path,
        checkpoint_number=checkpoint_number,
        env=env,
        net_spec=net,
        example_level=example_level,
        evals_dict={
            'on_distribution': eval_on_level_set,
            'off_distribution': eval_off_level_set,
            'on_distribution_animations': eval_on_animation,
            'off_distribution_animations': eval_off_animation,
            'on_distribution_level_0_heatmaps': eval_on_heatmap_0,
            'on_distribution_level_1_heatmaps': eval_on_heatmap_1,
            'off_distribution_eval_level_0_heatmaps': eval_off_heatmap_0,
            'off_distribution_eval_level_1_heatmaps': eval_off_heatmap_1,
        },
        save_files_to=save_files_to,
    )
    print("re-evaluation complete.")


