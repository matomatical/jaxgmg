"""
Demonstrating environment solution functionality.
"""

import time
import jax
import chex
from jaxgmg.procgen import maze_generation
from jaxgmg.environments import base
from jaxgmg.environments import cheese_in_the_corner
from jaxgmg.environments import cheese_on_a_dish
from jaxgmg.environments import keys_and_chests
# from jaxgmg.environments import monster_world
# from jaxgmg.environments import lava_land
# from jaxgmg.environments import follow_me
from jaxgmg import util


def solve_forever(
    rng: chex.PRNGKey,
    env: base.Env,
    level_generator: base.LevelGenerator,
    level_solver: base.LevelSolver,
    fps: float,
    debug: bool,
    partial_level_solver: base.LevelSolver | None = None,
):
    """
    Helper function for solving with a given environment.
    """
    p = partial_level_solver is not None
    while True:
        print("generating level...")
        rng_level, rng = jax.random.split(rng)
        level = level_generator.sample(rng_level)
        obs, state = env.reset_to_level(level)
        print("solving level...") 
        soln = level_solver.solve(level)
        print(f"initial value: {level_solver.level_value(soln, level):.3f}")
        if p:
            partial_soln = partial_level_solver.solve(level)
            print(f"initial value: {partial_level_solver.level_value(partial_soln, level):.3f} (according to partial solver)")

        print("initial state")
        image = util.img2str(obs.image)
        lines = len(str(image).splitlines())
        print(
            image,
            "solving level...     ",
            "^C to quit",
            sep="\n",
        )

        rng_steps, rng = jax.random.split(rng)
        done = False
        R = 0.0
        t = 0
        while not done:
            time.sleep(1/fps)
            rng_step, rng_steps = jax.random.split(rng_steps)
            V = level_solver.state_value(soln, state)
            a = level_solver.state_action(soln, state)
            obs, state, r, done, _ = env.step(rng_step, state, a)
            t = t + 1
            R = R + r * level_solver.discount_rate ** t
            print(
                "" if debug else f"\x1b[{lines+4}A",
                f"action: {a} ({'uldr'[a]})",
                util.img2str(obs.image),
                f"reward: {r:.3f} | return: {R:.3f} | value: {V:.3f} | done: {done}",
                "^C to quit",
                sep="\n",
            )
        if not debug:
            print(f"\x1b[{lines+7+p}A")


def corner(
    height: int                 = 13,
    width: int                  = 13,
    layout: str                 = 'tree',
    corner_size: int            = 1,
    penalize_time: bool         = True,
    max_steps_in_episode: int   = 128,
    discount_rate: float        = 0.995,
    level_of_detail: int        = 8,
    seed: int                   = 42,
    fps: float                  = 8,
    debug: bool                 = False,
):
    """
    Demonstrate optimal solution for random Cheese in the Corner levels.
    """
    util.print_config(locals())

    print("initialising environment, generator, and solver...")
    rng = jax.random.PRNGKey(seed=seed)
    env = cheese_in_the_corner.Env(
        obs_level_of_detail=level_of_detail,
        penalize_time=penalize_time,
        max_steps_in_episode=max_steps_in_episode,
    )
    level_generator = cheese_in_the_corner.LevelGenerator(
        height=height,
        width=width,
        maze_generator=maze_generation.get_generator_class_from_name(
            name=layout,
        )(),
        corner_size=corner_size,
    )
    level_solver = cheese_in_the_corner.LevelSolver(
        env=env,
        discount_rate=discount_rate,
    )

    solve_forever(
        rng=rng,
        env=env,
        level_generator=level_generator,
        level_solver=level_solver,
        fps=fps,
        debug=debug,
    )


def dish(
    height: int                 = 13,
    width: int                  = 13,
    cheese_on_dish: bool        = True,
    layout: str                 = 'tree',
    penalize_time: bool         = True,
    max_steps_in_episode: int   = 128,
    discount_rate: float        = 0.995,
    level_of_detail: int        = 8,
    seed: int                   = 42,
    fps: float                  = 8,
    debug: bool                 = False,
):
    """
    Demonstrate optimal solution for random Cheese on a Dish levels.
    """
    util.print_config(locals())

    print("initialising environment, generator, and solver...")
    rng = jax.random.PRNGKey(seed=seed)
    env = cheese_on_a_dish.Env(
        obs_level_of_detail=level_of_detail,
        penalize_time=penalize_time,
        max_steps_in_episode=max_steps_in_episode,
    )
    level_generator = cheese_on_a_dish.LevelGenerator(
        height=height,
        width=width,
        maze_generator=maze_generation.get_generator_class_from_name(
            name=layout,
        )(),
        cheese_on_dish=cheese_on_dish,
    )
    level_solver = cheese_on_a_dish.LevelSolver(
        env=env,
        discount_rate=discount_rate,
    )

    solve_forever(
        rng=rng,
        env=env,
        level_generator=level_generator,
        level_solver=level_solver,
        fps=fps,
        debug=debug,
    )


def keys(
    height: int                 = 13,
    width: int                  = 13,
    layout: str                 = 'blocks',
    num_keys: int               = 1,
    num_keys_max: int           = 5,
    num_chests: int             = 5,
    num_chests_max: int         = 5,
    penalize_time: bool         = False,
    max_steps_in_episode: int   = 128,
    discount_rate: float        = 0.995,
    level_of_detail: int        = 8,
    seed: int                   = 42,
    fps: float                  = 8,
    debug: bool                 = False,
):
    """
    Demonstrate optimal solution for random Keys and Chests levels.
    """
    util.print_config(locals())

    print("initialising environment, generator, and solver...")
    rng = jax.random.PRNGKey(seed=seed)
    env = keys_and_chests.Env(
        obs_level_of_detail=level_of_detail,
        penalize_time=penalize_time,
        max_steps_in_episode=max_steps_in_episode,
    )
    level_generator = keys_and_chests.LevelGenerator(
        height=height,
        width=width,
        maze_generator=maze_generation.get_generator_class_from_name(
            name=layout,
        )(),
        num_keys=num_keys,
        num_keys_max=num_keys_max,
        num_chests=num_chests,
        num_chests_max=num_chests_max,
    )
    level_solver = keys_and_chests.FullLevelSolver(
        env=env,
        discount_rate=discount_rate,
    )
    partial_level_solver = keys_and_chests.LevelSolverInit(
        env=env,
        discount_rate=discount_rate,
    )

    solve_forever(
        rng=rng,
        env=env,
        level_generator=level_generator,
        level_solver=level_solver,
        partial_level_solver=partial_level_solver,
        fps=fps,
        debug=debug,
    )


