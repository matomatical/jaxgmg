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
from jaxgmg.environments import cheese_on_a_pile
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
    prob_mutate_shift: float    = 0.0,
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
    # if mutating cheese, mostly stay in the restricted region
    biased_cheese_mutator = MixtureLevelMutator(
        mutators=(
            cheese_in_the_corner.CornerCheeseLevelMutator(
                corner_size=corner_size,
            ),
            cheese_in_the_corner.ScatterCheeseLevelMutator(),
        ),
        mixing_probs=(1-prob_mutate_shift, prob_mutate_shift),
    )
    # overall, rotate between wall/mouse/cheese mutations uniformly
    level_mutator = IteratedLevelMutator(
        mutator=MixtureLevelMutator(
            mutators=(
                cheese_in_the_corner.ToggleWallLevelMutator(),
                cheese_in_the_corner.ScatterMouseLevelMutator(
                    transpose_with_cheese_on_collision=False,
                ),
                biased_cheese_mutator,
            ),
            mixing_probs=(1/3,1/3,1/3),
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
    max_cheese_radius_shift: int        = 5,
    num_mutate_steps: int               = 1,
    prob_mutate_shift: float            = 0.0,
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
    biased_cheese_on_dish_mutator = MixtureLevelMutator(
        mutators=(
            # teleport cheese and dish to a random position max_cheese_radius
            cheese_on_a_dish.CheeseOnDishLevelMutator(
                max_cheese_radius=max_cheese_radius,
            ),
            # teleport cheese and dish to a random different position, apart by max_cheese_radius
            cheese_on_a_dish.CheeseOnDishLevelMutator(
                max_cheese_radius=max_cheese_radius_shift,
            ),
        ),
        mixing_probs=(1-prob_mutate_shift, prob_mutate_shift),
    )
    # overall, rotate between wall/mouse/cheese mutations uniformly
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
            mixing_probs=(1/3,1/3,1/3),
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


def pile(
    height: int                         = 9,
    width: int                          = 9,
    layout: str                         = 'tree',
    level_of_detail: int                = 8,
    max_cheese_radius: int              = 0,
    min_cheese_radius_shift: int        = 5,
    max_cheese_radius_shift: int        = 5,
    split_elements: int                 = 0,
    num_mutate_steps: int               = 1,
    prob_mutate_shift: float            = 0.3,
    transpose: bool                     = False,
    fps: float                          = 12.0,
    debug: bool                         = False,
    seed: int                           = 42,
):
    """
    Iterative Cheese on a Pile mutator demo.
    """
    if level_of_detail not in {1,3,4,8}:
        raise ValueError(f"invalid level of detail {level_of_detail}")
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    env = cheese_on_a_pile.Env(split_object_firstgroup=split_elements,img_level_of_detail=level_of_detail)
    level_generator = cheese_on_a_pile.LevelGenerator(
        height=height,
        width=width,
        maze_generator=maze_generation.get_generator_class_from_name(
            name=layout
        )(),
        max_cheese_radius=max_cheese_radius,
        split_elements=split_elements,
    )
    print("configuring level mutator...")
    
    biased_cheese_on_pile_mutator = MixtureLevelMutator(
        mutators=(
            # teleport cheese on pile
            cheese_on_a_pile.CheeseonPileLevelMutatorUP(
                min_cheese_radius=min_cheese_radius_shift,
                max_cheese_radius=max_cheese_radius,
                split_elements=split_elements,
            ),
            # teleport cheese and pile to a random different position, apart by max_cheese_radius
            cheese_on_a_pile.CheeseonPileLevelMutatorUP(
                min_cheese_radius=min_cheese_radius_shift,
                max_cheese_radius=max_cheese_radius_shift,
                split_elements=split_elements,
            ),
        ),
        mixing_probs=(1-prob_mutate_shift, prob_mutate_shift),
    )
    # overall, rotate between wall/mouse/cheese mutations uniformly
    level_mutator = IteratedLevelMutator(
        mutator=MixtureLevelMutator(
            mutators=(
                cheese_on_a_pile.ToggleWallLevelMutator(),
                cheese_on_a_pile.ScatterMouseLevelMutator(
                    transpose_with_cheese_on_collision=False,
                    transpose_with_pile_on_collision=False,
                    split_elements=split_elements,
                ),
                biased_cheese_on_pile_mutator,
            ),
            mixing_probs=(1/3,1/3,1/3),
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
    corner_size: int                = 1,
    num_mutate_steps: int           = 1,
    prob_mutate_shift: float        = 0.0,
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
        corner_size= corner_size,
    )
    biased_goal_mutator = MixtureLevelMutator(
        mutators=(
            # teleport cheese to the corner
            minigrid_maze.CornerGoalLevelMutator(
                corner_size=corner_size,
            ),
            # teleport cheese to a random position
            minigrid_maze.CornerGoalLevelMutator(
                corner_size=height-2,
            ),
        ),
        mixing_probs=(1-prob_mutate_shift, prob_mutate_shift),
    )
    level_mutator = IteratedLevelMutator(
        mutator=MixtureLevelMutator(
            mutators=(
                minigrid_maze.ToggleWallLevelMutator(),
                minigrid_maze.ScatterAndSpinHeroLevelMutator(
                    transpose_with_goal_on_collision=False,
                ),
                biased_goal_mutator,
            ),
            mixing_probs=(1/3,1/3,1/3),
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
    

