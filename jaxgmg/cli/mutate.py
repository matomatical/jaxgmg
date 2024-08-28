"""
Interactive environment demonstrations.
"""

from typing import Callable
import itertools
import time

import jax
import chex

from jaxgmg.procgen import maze_generation
from jaxgmg.environments import base
from jaxgmg.environments import cheese_in_the_corner
from jaxgmg.environments import cheese_on_a_dish
from jaxgmg.environments import minigrid_maze
from jaxgmg.environments.base import MixtureLevelMutator, IteratedLevelMutator
from jaxgmg import util


# # # 
# HELPER FUNCTION


def mutate_forever(
    rng: chex.PRNGKey,
    env: base.Env,
    level_generator: base.LevelGenerator,
    level_mutator: base.LevelMutator,
    fps: int,
    debug: bool = False,
):
    """
    Helper function to repeatedly mutate and display a level.
    """
    # initial level
    rng_initial_level, rng = jax.random.split(rng)
    level = level_generator.sample(rng_initial_level)

    # render levels
    def render_level(level: base.Level) -> str:
        _, state = env.reset_to_level(level)
        img = env.render_state(state)
        return util.img2str(img)

    # mutation levels
    for i in itertools.count():
        rng_mutate, rng = jax.random.split(rng)
        level = level_mutator.mutate_level(rng_mutate, level)
        img = render_level(level)
        lines = len(str(img).splitlines())
        print(
            "" if (debug or i == 0) else f"\x1b[{lines+2}A",
            f"level after {i} mutations:",
            img,
            sep="\n",
        )
        time.sleep(1/fps)


# # # 
# ENVIRONMENT ENTRY POINTS


def corner(
    height: int                 = 9,
    width: int                  = 9,
    layout: str                 = 'tree',
    corner_size: int            = 1,
    level_of_detail: int        = 8,
    num_mutate_steps: int       = 1,
    prob_mutate_wall: float     = 0.60,
    prob_mutate_step: float     = 0.95,
    prob_mutate_cheese: float   = 0.0,
    transpose: bool             = False,
    fps: float                  = 12.0,
    debug: bool                 = False,
    seed: int                   = 42,
):
    """
    Iterative Cheese in the Corner mutator demo.
    """
    if level_of_detail not in {1,3,4,8}:
        raise ValueError(f"invalid level of detail {level_of_detail}")
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    env = cheese_in_the_corner.Env(img_level_of_detail=level_of_detail)
    level_generator = cheese_in_the_corner.LevelGenerator(
        height=height,
        width=width,
        maze_generator=maze_generation.get_generator_class_from_name(
            name=layout
        )(),
        corner_size=corner_size,
    )
    level_mutator = IteratedLevelMutator(
        mutator=MixtureLevelMutator(
            mutators=(
                cheese_in_the_corner.ToggleWallLevelMutator(),
                cheese_in_the_corner.StepMouseLevelMutator(
                    transpose_with_cheese_on_collision=transpose,
                ),
                cheese_in_the_corner.ScatterMouseLevelMutator(
                    transpose_with_cheese_on_collision=transpose,
                ),
                cheese_in_the_corner.StepCheeseLevelMutator(),
                cheese_in_the_corner.ScatterCheeseLevelMutator(),
            ),
            mixing_probs=(
                prob_mutate_wall,
                (1-prob_mutate_wall)*(1-prob_mutate_cheese)*prob_mutate_step,
                (1-prob_mutate_wall)*(1-prob_mutate_cheese)*(1-prob_mutate_step),
                (1-prob_mutate_wall)*prob_mutate_cheese*prob_mutate_step,
                (1-prob_mutate_wall)*prob_mutate_cheese*(1-prob_mutate_step),
            ),
        ),
        num_steps=num_mutate_steps,
    )
    mutate_forever(
        rng=rng,
        env=env,
        level_generator=level_generator,
        level_mutator=level_mutator,
        fps=fps,
        debug=debug,
    )


def dish(
    height: int                         = 9,
    width: int                          = 9,
    layout: str                         = 'tree',
    level_of_detail: int                = 8,
    max_cheese_radius: int              = 0,
    num_mutate_steps: int               = 1,
    prob_mutate_wall: float             = 0.60,
    prob_mutate_step: float             = 0.95,
    prob_mutate_cheese_or_dish: float   = 0.0,
    fps: float                          = 12.0,
    debug: bool                         = False,
    seed: int                           = 42,
):
    """
    Iterative Cheese on a Dish mutator demo.
    """
    if level_of_detail not in {1,3,4,8}:
        raise ValueError(f"invalid level of detail {level_of_detail}")
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    env = cheese_on_a_dish.Env(img_level_of_detail=level_of_detail)
    level_generator = cheese_on_a_dish.LevelGenerator(
        height=height,
        width=width,
        maze_generator=maze_generation.get_generator_class_from_name(
            name=layout
        )(),
        max_cheese_radius=max_cheese_radius,
    )
    level_mutator = IteratedLevelMutator(
        mutator=MixtureLevelMutator(
            mutators=(
                cheese_on_a_dish.ToggleWallLevelMutator(),
                cheese_on_a_dish.StepMouseLevelMutator(),
                cheese_on_a_dish.ScatterMouseLevelMutator(),
                cheese_on_a_dish.StepDishLevelMutator(),
                cheese_on_a_dish.ScatterDishLevelMutator(),
                cheese_on_a_dish.StepCheeseLevelMutator(),
                cheese_on_a_dish.ScatterCheeseLevelMutator(),
            ),
            mixing_probs=(
                prob_mutate_wall,
                (1-prob_mutate_wall)*(1-prob_mutate_cheese_or_dish)*prob_mutate_step,
                (1-prob_mutate_wall)*(1-prob_mutate_cheese_or_dish)*(1-prob_mutate_step),
                (1-prob_mutate_wall)*prob_mutate_cheese_or_dish/2*prob_mutate_step,
                (1-prob_mutate_wall)*prob_mutate_cheese_or_dish/2*(1-prob_mutate_step),
                (1-prob_mutate_wall)*prob_mutate_cheese_or_dish/2*prob_mutate_step,
                (1-prob_mutate_wall)*prob_mutate_cheese_or_dish/2*(1-prob_mutate_step),
            ),
        ),
        num_steps=num_mutate_steps,
    )
    mutate_forever(
        rng=rng,
        env=env,
        level_generator=level_generator,
        level_mutator=level_mutator,
        fps=fps,
        debug=debug,
    )


def minimaze(
    height: int                     = 9,
    width: int                      = 9,
    obs_height: int                 = 5,
    obs_width: int                  = 5,
    layout: str                     = 'noise',
    level_of_detail: int            = 8,
    num_mutate_steps: int           = 1,
    prob_mutate_wall: float         = 0.60,
    prob_mutate_step_or_turn: float = 0.95,
    prob_mutate_goal: float         = 0.50,
    fps: float                      = 12.0,
    debug: bool                     = False,
    seed: int                       = 42,
):
    """
    Iterative Minigrid Maze mutator demo.
    """
    if level_of_detail not in {1,3,4,8}:
        raise ValueError(f"invalid level of detail {level_of_detail}")
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    env = minigrid_maze.Env(
        obs_height=obs_height,
        obs_width=obs_width,
        img_level_of_detail=level_of_detail,
    )
    level_generator = minigrid_maze.LevelGenerator(
        height=height,
        width=width,
        maze_generator=maze_generation.get_generator_class_from_name(
            name=layout
        )(),
    )
    level_mutator = IteratedLevelMutator(
        mutator=MixtureLevelMutator(
            mutators=(
                minigrid_maze.ToggleWallLevelMutator(),
                minigrid_maze.StepHeroLevelMutator(),
                minigrid_maze.TurnHeroLevelMutator(),
                minigrid_maze.ScatterHeroLevelMutator(),
                minigrid_maze.StepGoalLevelMutator(),
                minigrid_maze.ScatterGoalLevelMutator(),
            ),
            mixing_probs=(
                prob_mutate_wall,
                (1-prob_mutate_wall)*(1-prob_mutate_goal)*prob_mutate_step_or_turn/2,
                (1-prob_mutate_wall)*(1-prob_mutate_goal)*prob_mutate_step_or_turn/2,
                (1-prob_mutate_wall)*(1-prob_mutate_goal)*(1-prob_mutate_step_or_turn),
                (1-prob_mutate_wall)*prob_mutate_goal*prob_mutate_step_or_turn,
                (1-prob_mutate_wall)*prob_mutate_goal*(1-prob_mutate_step_or_turn),
            ),
        ),
        num_steps=num_mutate_steps,
    )
    mutate_forever(
        rng=rng,
        env=env,
        level_generator=level_generator,
        level_mutator=level_mutator,
        fps=fps,
        debug=debug,
    )
    

