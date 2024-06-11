"""
Interactive environment demonstrations.
"""

import jax
import jax.numpy as jnp
import readchar
import time

from jaxgmg.procgen import maze_generation
from jaxgmg.environments import cheese_in_the_corner
from jaxgmg.environments import cheese_on_a_dish
from jaxgmg.environments import keys_and_chests
from jaxgmg.environments import monster_world
from jaxgmg.environments import lava_land
from jaxgmg.environments import follow_me
from jaxgmg.cli import util


# # # 
# HELPER FUNCTION


def play_forever(rng, env, level_generator, debug=False):
    """
    Helper function for interacting with a given environment.
    """
    while True:
        print("generating level...")
        rng_level, rng = jax.random.split(rng)
        level = level_generator.sample(rng_level)
        obs, state = env.reset_to_level(rng, level)

        print("playing level...")
        print("initial state")
        image = util.img2str(env.get_obs(state))
        lines = len(str(image).splitlines())
        print(
            image,
            "",
            "controls: w = up | a = left | s = down | d = right | q = quit",
            sep="\n",
        )
    
        rng_steps, rng = jax.random.split(rng)
        while True:
            key = readchar.readkey()
            if key == "q":
                print("bye!")
                return
            if key == "r":
                break
            if key not in "wsda":
                continue
            a = "wasd".index(key)
            rng_step, rng_steps = jax.random.split(rng_steps)
            _, state, r, d, _ = env.step(rng_step, state, a)
            print(
                "" if debug else f"\x1b[{lines+4}A",
                f"action: {a} ({'uldr'[a]})",
                util.img2str(env.get_obs(state)),
                f"reward: {r:.2f} done: {d}",
                "controls: w = up | a = left | s = down | d = right | q = quit",
                sep="\n",
            )
            if d:
                break
        if not debug:
            print(f"\x1b[{lines+6}A")


# # # 
# ENVIRONMENT ENTRY POINTS


def corner(
    height: int                 = 13,
    width: int                  = 9,
    layout: str                 = 'tree',
    corner_size: int            = 3,
    seed: int                   = 42,
    debug: bool                 = False,
):
    """
    Interactive Cheese in the Corner environment.
    """
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    env = cheese_in_the_corner.Env(rgb=True)
    level_generator = cheese_in_the_corner.LevelGenerator(
        height=height,
        width=width,
        maze_generator=maze_generation.get_generator_class_from_name(
            name=layout
        )(),
        corner_size=corner_size,
    )
    play_forever(
        rng=rng,
        env=env,
        level_generator=level_generator,
        debug=debug,
    )


def dish(
    height: int                 = 13,
    width: int                  = 9,
    layout: str                 = 'tree',
    max_cheese_radius: int      = 3,
    seed: int                   = 42,
    debug: bool                 = False,
):
    """
    Interactive Cheese on a Dish environment.
    """
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    env = cheese_on_a_dish.Env(rgb=True)
    level_generator = cheese_on_a_dish.LevelGenerator(
        height=height,
        width=width,
        maze_generator=maze_generation.get_generator_class_from_name(
            name=layout
        )(),
        max_cheese_radius=max_cheese_radius,
    )
    play_forever(
        rng=rng,
        env=env,
        level_generator=level_generator,
        debug=debug,
    )


def follow(
    height: int                 = 13,
    width: int                  = 9,
    layout: str                 = 'edges',
    num_beacons: int            = 3,
    trustworthy_leader: bool    = True,
    seed: int                   = 42,
    debug: bool                 = False,
):
    """
    Interactive Follow Me environment.
    """
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    env = follow_me.Env(rgb=True)
    level_generator = follow_me.LevelGenerator(
        height=height,
        width=width,
        maze_generator=maze_generation.get_generator_class_from_name(
            name=layout
        )(),
        num_beacons=num_beacons,
        trustworthy_leader=trustworthy_leader,
    )
    play_forever(
        rng=rng,
        env=env,
        level_generator=level_generator,
        debug=debug,
    )


def keys(
    height: int                 = 13,
    width: int                  = 9,
    layout: str                 = 'tree',
    num_keys_min: int           = 2,
    num_keys_max: int           = 6,
    num_chests_min: int         = 6,
    num_chests_max: int         = 6,
    seed: int                   = 42,
    debug: bool                 = False,
):
    """
    Interactive Keys and Chests environment.
    """
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    env = keys_and_chests.Env(rgb=True)
    level_generator = keys_and_chests.LevelGenerator(
        height=height,
        width=width,
        maze_generator=maze_generation.get_generator_class_from_name(
            name=layout
        )(),
        num_keys_min=num_keys_min,
        num_keys_max=num_keys_max,
        num_chests_min=num_chests_min,
        num_chests_max=num_chests_max,
    )
    play_forever(
        rng=rng,
        env=env,
        level_generator=level_generator,
        debug=debug,
    )


def lava(
    height: int                 = 13,
    width: int                  = 9,
    layout: str                 = 'edges',
    lava_threshold: float       = -0.25,
    seed: int                   = 42,
    debug: bool                 = False,
):
    """
    Interactive Lava Land environment.
    """
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    env = lava_land.Env(rgb=True)
    level_generator = lava_land.LevelGenerator(
        height=height,
        width=width,
        maze_generator=maze_generation.get_generator_class_from_name(
            name=layout
        )(),
        lava_threshold=lava_threshold,
    )
    play_forever(
        rng=rng,
        env=env,
        level_generator=level_generator,
        debug=debug,
    )
    

def monsters(
    height: int                 = 13,
    width: int                  = 9,
    layout: str                 = 'open',
    num_apples: int             = 5,
    num_shields: int            = 5,
    num_monsters: int           = 5,
    monster_optimality: float   = 3,
    seed: int                   = 42,
    debug: bool                 = False,
):
    """
    Interactive Monster World environment.
    """
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    env = monster_world.Env(rgb=True)
    level_generator = monster_world.LevelGenerator(
        height=height,
        width=width,
        maze_generator=maze_generation.get_generator_class_from_name(
            name=layout
        )(),
        num_apples=num_apples,
        num_shields=num_shields,
        num_monsters=num_monsters,
        monster_optimality=monster_optimality,
    )
    play_forever(
        rng=rng,
        env=env,
        level_generator=level_generator,
        debug=debug,
    )
    

