"""
Interactive environment demonstrations.
"""

from typing import Callable
import readchar
import time

import jax
import jax.numpy as jnp
import chex
import einops

from jaxgmg.procgen import maze_generation
from jaxgmg.environments import base
from jaxgmg.environments import cheese_in_the_corner
from jaxgmg.environments import cheese_on_a_dish
from jaxgmg.environments import keys_and_chests
from jaxgmg.environments import follow_me
from jaxgmg.environments import lava_land
from jaxgmg.environments import monster_world
from jaxgmg.environments import minigrid_maze
from jaxgmg import util


# # # 
# HELPER FUNCTION


def play_forever(
    rng: chex.PRNGKey,
    env: base.Env,
    actions: list[str],
    level_generator: base.LevelGenerator,
    record: bool = False,
    split_channels: bool = False,
    debug: bool = False,
):
    """
    Helper function for interacting with a given environment.
    """
    if record: frames = []
    playing = True
    controls = (
        "controls: "
        f"[w {actions[0]}] [a {actions[1]}] "
        f"[s {actions[2]}] [d {actions[3]}] "
        "[r reset] [q quit]"
    )
    def render(img):
        if split_channels:
            h, w, c = img.shape
            refracted = jnp.zeros((3, h, w, c))
            for i in range(c):
                refracted = refracted.at[i%3,:,:,i].set(img[:,:,i])
            padded = jnp.pad(
                refracted,
                pad_width=((0,0),(0,1),(0,1),(0,0)),
                constant_values=0.5,
            )
            stacked = einops.rearrange(padded, 'rgb h w c -> (c h) w rgb')
            padded_again = jnp.pad(
                stacked,
                pad_width=((1,0),(1,0),(0,0)),
                constant_values=0.5,
            )
            return padded_again
        else:
            return img

    while playing:
        print("generating level...")
        rng_level, rng = jax.random.split(rng)
        level = level_generator.sample(rng_level)
        obs, state = env.reset_to_level(level)
        img = render(env.render_state(state))

        print("playing level...")
        print("initial state")
        image = util.img2str(img)
        lines = len(str(image).splitlines())
        print(
            image,
            "",
            controls,
            sep="\n",
        )
        if record: frames.append(obs)
    
        rng_steps, rng = jax.random.split(rng)
        while True:
            key = readchar.readkey()
            if key == "q":
                print("bye!")
                playing = False
                break # will then exit the outer loop
            if key == "r":
                # next level
                break
            if key not in "wsda":
                continue
            a = "wasd".index(key)
            rng_step, rng_steps = jax.random.split(rng_steps)
            obs, state, r, d, _ = env.step(rng_step, state, a)
            img = render(env.render_state(state))
            print(
                "" if debug else f"\x1b[{lines+4}A",
                f"action: {a} ({'uldr'[a]})",
                util.img2str(img),
                f"reward: {r:.2f} done: {d}",
                controls,
                sep="\n",
            )
            if record and not d: frames.append(obs)
            if d:
                break
        if not debug:
            print(f"\x1b[{lines+6}A")
    if record:
        print(f"{len(frames)} frames recorded, saving to './out.gif'...")
        util.save_gif(
            frames=frames,
            path="./out.gif",
            upscale=2,
            fps=8,
            repeat=True,
        )


# # # 
# ENVIRONMENT ENTRY POINTS


def corner(
    height: int                 = 9,
    width: int                  = 9,
    layout: str                 = 'tree',
    corner_size: int            = 3,
    level_of_detail: int        = 8,
    seed: int                   = 42,
    split_channels: bool        = False,
    debug: bool                 = False,
    record: bool                = False,
):
    """
    Interactive Cheese in the Corner environment.
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
    play_forever(
        rng=rng,
        env=env,
        actions=['up', 'left', 'down', 'right'],
        level_generator=level_generator,
        split_channels=split_channels,
        debug=debug,
        record=record,
    )


def dish(
    height: int                 = 9,
    width: int                  = 9,
    layout: str                 = 'tree',
    cheese_on_dish: bool        = True,
    level_of_detail: int        = 8,
    num_channels_cheese: int    = 1,
    num_channels_dish: int      = 1,
    seed: int                   = 42,
    split_channels: bool        = False,
    debug: bool                 = False,
    record: bool                = False,
):
    """
    Interactive Cheese on a Dish environment.
    """
    if level_of_detail not in {0,1,3,4,8}:
        raise ValueError(f"invalid level of detail {level_of_detail}")
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    env = cheese_on_a_dish.Env(
        img_level_of_detail=level_of_detail,
        num_channels_cheese=num_channels_cheese,
        num_channels_dish=num_channels_dish,
    )
    level_generator = cheese_on_a_dish.LevelGenerator(
        height=height,
        width=width,
        maze_generator=maze_generation.get_generator_class_from_name(
            name=layout
        )(),
        cheese_on_dish=cheese_on_dish,
    )
    play_forever(
        rng=rng,
        env=env,
        actions=['up', 'left', 'down', 'right'],
        level_generator=level_generator,
        split_channels=split_channels,
        debug=debug,
        record=record,
    )


def follow(
    height: int                 = 9,
    width: int                  = 9,
    layout: str                 = 'edges',
    num_beacons: int            = 3,
    trustworthy_leader: bool    = True,
    level_of_detail: int        = 8,
    seed: int                   = 42,
    split_channels: bool        = False,
    debug: bool                 = False,
    record: bool                = False,
):
    """
    Interactive Follow Me environment.
    """
    if level_of_detail not in {1,3,4,8}:
        raise ValueError(f"invalid level of detail {level_of_detail}")
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    env = follow_me.Env(img_level_of_detail=level_of_detail)
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
        actions=['up', 'left', 'down', 'right'],
        level_generator=level_generator,
        split_channels=split_channels,
        debug=debug,
        record=record,
    )


def keys(
    height: int                 = 9,
    width: int                  = 9,
    layout: str                 = 'tree',
    num_keys_min: int           = 2,
    num_keys_max: int           = 6,
    num_chests_min: int         = 6,
    num_chests_max: int         = 6,
    level_of_detail: int        = 8,
    seed: int                   = 42,
    split_channels: bool        = False,
    debug: bool                 = False,
    record: bool                = False,
):
    """
    Interactive Keys and Chests environment.
    """
    if level_of_detail not in {1,3,4,8}:
        raise ValueError(f"invalid level of detail {level_of_detail}")
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    env = keys_and_chests.Env(img_level_of_detail=level_of_detail)
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
        actions=['up', 'left', 'down', 'right'],
        level_generator=level_generator,
        split_channels=split_channels,
        debug=debug,
        record=record,
    )


def lava(
    height: int                 = 9,
    width: int                  = 9,
    layout: str                 = 'edges',
    lava_threshold: float       = -0.25,
    level_of_detail: int        = 8,
    seed: int                   = 42,
    split_channels: bool        = False,
    debug: bool                 = False,
    record: bool                = False,
):
    """
    Interactive Lava Land environment.
    """
    if level_of_detail not in {1,3,4,8}:
        raise ValueError(f"invalid level of detail {level_of_detail}")
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    env = lava_land.Env(img_level_of_detail=level_of_detail)
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
        actions=['up', 'left', 'down', 'right'],
        level_generator=level_generator,
        split_channels=split_channels,
        debug=debug,
        record=record,
    )
    

def monsters(
    height: int                 = 9,
    width: int                  = 9,
    layout: str                 = 'open',
    num_apples: int             = 5,
    num_shields: int            = 5,
    num_monsters: int           = 5,
    monster_optimality: float   = 3,
    level_of_detail: int        = 8,
    seed: int                   = 42,
    split_channels: bool        = False,
    debug: bool                 = False,
    record: bool                = False,
):
    """
    Interactive Monster World environment.
    """
    if level_of_detail not in {1,3,4,8}:
        raise ValueError(f"invalid level of detail {level_of_detail}")
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    env = monster_world.Env(img_level_of_detail=level_of_detail)
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
        actions=['up', 'left', 'down', 'right'],
        level_generator=level_generator,
        split_channels=split_channels,
        debug=debug,
        record=record,
    )
    

def minimaze(
    height: int                 = 9,
    width: int                  = 9,
    obs_height: int             = 7,
    obs_width: int              = 7,
    layout: str                 = 'noise',
    level_of_detail: int        = 8,
    seed: int                   = 42,
    split_channels: bool        = False,
    debug: bool                 = False,
    record: bool                = False,
):
    """
    Interactive Minigrid Maze environment.
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
    play_forever(
        rng=rng,
        env=env,
        actions=['forward', 'turn-left', 'wait', 'turn-right'],
        level_generator=level_generator,
        split_channels=split_channels,
        debug=debug,
        record=record,
    )
    

