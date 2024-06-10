"""
Demonstrating environment solution functionality.
"""

import jax
from jaxgmg.procgen import maze_generation
from jaxgmg.procgen import maze_solving
from jaxgmg.environments import cheese_in_the_corner
from jaxgmg.environments import cheese_on_a_dish
from jaxgmg.environments import keys_and_chests
from jaxgmg.environments import monster_world
from jaxgmg.environments import lava_land
from jaxgmg.environments import follow_me
from jaxgmg.cli import util


def corner(
    height: int                 = 13,
    width: int                  = 13,
    layout: str                 = 'tree',
    corner_size: int            = 3,
    penalize_time: bool         = True,
    max_steps_in_episode: int   = 128,
    discount_rate: float        = 0.995,
    num_levels: int             = 64,
    seed: int                   = 42,
):
    print(
        "solve-corner: optimal value for a random cheese in the corner level"
    )
    util.print_config(locals())

    print("initialising...")
    rng = jax.random.PRNGKey(seed=seed)
    env = cheese_in_the_corner.Env(
        rgb=False,
        penalize_time=penalize_time,
        max_steps_in_episode=max_steps_in_episode,
    )
    level_generator = cheese_in_the_corner.LevelGenerator(
        height=height,
        width=width,
        maze_generator=maze_generation.get_generator_class_from_name(
            name=layout
        )(),
        corner_size=corner_size,
    )

    print("generating levels...")
    rng_level, rng = jax.random.split(rng)
    levels = level_generator.vsample(rng_level, num_levels)
    rng_reset, rng = jax.random.split(rng)
    obs, state = env.vreset_to_level(rng_reset, levels)

    print("visualising first level (indicative)...")
    img = (
        1 * obs[0,:,:,env.Channel.WALL]
        + 4 * obs[0,:,:,env.Channel.CHEESE]
        + 12 * obs[0,:,:,env.Channel.MOUSE]
    )
    print(util.img2str(img, colormap=util.sweetie16))
    util.print_legend({
        1: "wall",
        4: "cheese",
        12: "mouse",
    }, colormap=util.sweetie16)

    print("solving levels...")
    v = jax.vmap(
        env.optimal_value,
        in_axes=(0,None),
    )(
        levels,
        discount_rate,
    )

    print('optimal values:')
    util.print_histogram(v)
    print('average optimal value:', v.mean())
    print('std dev optimal value:', v.std())


def keys(
    height: int                 = 13,
    width: int                  = 13,
    layout: str                 = 'tree',
    num_keys_min: int           = 2,
    num_keys_max: int           = 4,
    num_chests_min: int         = 4,
    num_chests_max: int         = 4,
    penalize_time: bool         = True,
    max_steps_in_episode: int   = 128,
    discount_rate: float        = 0.995,
    num_levels: int             = 64,
    seed: int                   = 42,
):
    print("solve-keys: optimal value for a random keys and chests level")
    util.print_config(locals())
    if num_keys_max + num_chests_max > 9:
        print(
            f"WARNING: attempting to solve environments with "
            f"{num_keys_max} + {num_chests_max} "
            f"= {num_keys_max + num_chests_max} > 9 keys/chests. "
            "This may take a while."
        )

    print("initialising...")
    rng = jax.random.PRNGKey(seed=seed)
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
    env = keys_and_chests.Env(
        rgb=False,
        penalize_time=penalize_time,
        max_steps_in_episode=max_steps_in_episode,
    )

    print("generating levels...")
    rng_level, rng = jax.random.split(rng)
    levels = level_generator.vsample(rng_level, num_levels)
    rng_reset, rng = jax.random.split(rng)
    obs, state = env.vreset_to_level(rng_reset, levels)
    
    print("visualising first level (indicative)...")
    img = (
        1 * obs[0,:,:,env.Channel.WALL]
        + 2 * obs[0,:,:,env.Channel.CHEST]
        + 4 * obs[0,:,:,env.Channel.KEY]
        + 12 * obs[0,:,:,env.Channel.MOUSE]
    )
    print(util.img2str(img, colormap=util.sweetie16))
    util.print_legend({
        1: "wall",
        2: "chest",
        4: "key",
        12: "mouse",
    }, colormap=util.sweetie16)

    print("solving levels...")
    v = jax.vmap(
        env.optimal_value,
        in_axes=(0,None),
    )(
        levels,
        discount_rate,
    )

    print('optimal values:')
    util.print_histogram(v)
    print('average optimal value:', v.mean())
    print('std dev optimal value:', v.std())

    
