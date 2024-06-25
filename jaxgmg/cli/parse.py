"""
Testing parsers on ASCII levels.
"""

import jax
import jax.numpy as jnp

from jaxgmg.environments import cheese_in_the_corner
from jaxgmg.environments import cheese_on_a_dish
from jaxgmg.environments import keys_and_chests
from jaxgmg.environments import monster_world
from jaxgmg.environments import lava_land
from jaxgmg.environments import follow_me

from jaxgmg.cli import util


def corner(
    level_of_detail: int = 8,
):
    """
    Test the level parser from the Cheese in the Corner environment.
    """
    test_string = """
        # # # # #
        # . . . #
        # @ # . #
        # . . * #
        # # # # #
    """
    print("test string:", test_string)

    print("parsing...")
    p = cheese_in_the_corner.LevelParser(height=5, width=5)
    level = p.parse(test_string)
    print("level:", level)

    print("rendering...")
    env = cheese_in_the_corner.Env(obs_level_of_detail=level_of_detail)
    obs, state = env.reset_to_level(level)
    print(util.img2str(obs))


def dish(
    level_of_detail: int = 8,
):
    """
    Test the level parser from the Cheese on a Dish environment.
    """
    test_string = """
        # # # # #
        # . . . #
        # @ # d #
        # . . c #
        # # # # #
    """
    print("test string:", test_string)

    print("parsing...")
    p = cheese_on_a_dish.LevelParser(
        height=5,
        width=5,
    )
    level = p.parse(test_string)
    print("level:", level)

    print("rendering...")
    env = cheese_on_a_dish.Env(obs_level_of_detail=level_of_detail)
    obs, state = env.reset_to_level(level)
    print(util.img2str(obs))


def follow(
    level_of_detail: int = 8,
):
    """
    Test the level parser from the Follow Me environment.
    """
    test_string = """
        # # # # # # #
        # @ # . . 1 #
        # . # . # # #
        # . . . . 0 #
        # * # . # # #
        # . # . . 2 #
        # # # # # # #
    """
    print("test string:", test_string)

    print("parsing...")
    p = follow_me.LevelParser(
        height=7,
        width=7,
        num_beacons=3,
        leader_order=(0,1,2),
    )
    level = p.parse(test_string)
    print("level:", level)

    print("rendering...")
    env = follow_me.Env(obs_level_of_detail=level_of_detail)
    obs, state = env.reset_to_level(level)
    print(util.img2str(obs))


def keys(
    level_of_detail: int = 8,
):
    """
    Test the level parser from the Keys and Chests environment.
    """
    test_string = """
        # # # # #
        # . k c #
        # @ # k #
        # k # c #
        # # # # #
    """
    print("test string:", test_string)

    print("parsing...")
    p = keys_and_chests.LevelParser(
        height=5,
        width=5,
        num_keys_max=3,
        num_chests_max=3,
        inventory_map=jnp.arange(3),
    )
    level = p.parse(test_string)
    print("level:", level)

    print("rendering...")
    env = keys_and_chests.Env(obs_level_of_detail=level_of_detail)
    obs, state = env.reset_to_level(level)
    print(util.img2str(obs))


def lava(
    level_of_detail: int = 8,
):
    """
    Test the level parser from the Lava Land environment.
    """
    test_string = """
        # # # # #
        # . . . #
        # @ # . #
        # . X * #
        # # # # #
    """
    print("test string:", test_string)

    print("parsing...")
    p = lava_land.LevelParser(height=5, width=5)
    level = p.parse(test_string)
    print("level:", level)

    print("rendering...")
    env = lava_land.Env(obs_level_of_detail=level_of_detail)
    obs, state = env.reset_to_level(level)
    print(util.img2str(obs))


def monsters(
    level_of_detail: int = 8,
):
    """
    Test the level parser from the Monster World environment.
    """
    test_string = """
        # # # # #
        # s a m #
        # @ # s #
        # . m a #
        # # # # #
    """
    print("test string:", test_string)

    print("parsing...")
    p = monster_world.LevelParser(
        height=5,
        width=5,
        num_apples=2,
        num_monsters=2,
        num_shields=2,
        monster_optimality=3.0,
        inventory_map=jnp.array((0,1)),
    )
    level = p.parse(test_string)
    print("level:", level)

    print("rendering...")
    env = monster_world.Env(obs_level_of_detail=level_of_detail)
    obs, state = env.reset_to_level(level)
    print(util.img2str(obs))


