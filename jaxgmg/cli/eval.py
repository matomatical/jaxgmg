"""
Launcher for evaluating checkpoints.
"""

import jax
import jax.numpy as jnp

from jaxgmg.procgen import maze_generation
from jaxgmg.environments import cheese_in_the_corner
from jaxgmg.baselines import evaluate

from jaxgmg.environments.base import Level

from jaxgmg import util


def corner(
    # checkpointing
    checkpoint_folder: str,
    checkpoint_number: int,
    # environment config
    env_size: int = 15,
    env_layout: str = 'blocks',
    env_corner_size: int = 1,
    env_terminate_after_corner: bool = False,
    obs_level_of_detail: int = 0,           # 0 = bool; 1, 3, 4, or 8 = rgb
    img_level_of_detail: int = 1,           # obs_ is for train, img_ for gifs
    env_penalize_time: bool = False,
    ppo_gamma: float = 0.999,
    # policy config
    net_cnn_type: str = "large",
    net_rnn_type: str = "ff",
    net_width: int = 256,
    # evals config
    evals_num_env_steps: int = 512,
    evals_num_levels: int = 256,
    gif_grid_width: int = 16,
    level_splayer: str = 'cheese',           # or 'mouse' or 'cheese-and-mouse'
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


    print("configuring level solver...")
    level_solver = cheese_in_the_corner.LevelSolver(
        env=env,
        discount_rate=ppo_gamma,
    )


    print("configuring eval level generators...")
    eval_level_generators = {
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
            'sixteen-rooms-shifted': level_parser_15.parse("""
                # # # # # # # # # # # # # # # 
                # . . . # . . # . . # . . . #
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
                # . * . . . . # . . . . @ . #
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
        }
    else:
        print("(unsupported size for fixed evals)")
        fixed_eval_levels = {}


    print("configuring heatmap splayer...")
    splayer_fn = cheese_in_the_corner.splayer_from_name(level_splayer)


    evaluate.run(
        seed=seed,
        checkpoint_folder=checkpoint_folder,
        checkpoint_number=checkpoint_number,
        env=env,
        level_solver=level_solver,
        eval_level_generators=eval_level_generators,
        fixed_eval_levels=fixed_eval_levels,
        arbitrary_level_generator=orig_level_generator,
        heatmap_splayer_fn=splayer_fn,
        ppo_gamma=ppo_gamma,
        net_cnn_type=net_cnn_type,
        net_rnn_type=net_rnn_type,
        net_width=net_width,
        evals_num_env_steps=evals_num_env_steps,
        evals_num_levels=evals_num_levels,
        gif_grid_width=gif_grid_width,
    )


