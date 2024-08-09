"""
Testing splay operations.
"""

import jax
import jax.numpy as jnp
import einops

from jaxgmg.procgen import maze_generation
from jaxgmg.environments import cheese_in_the_corner
# from jaxgmg.environments import cheese_on_a_dish
# from jaxgmg.environments import keys_and_chests
# from jaxgmg.environments import monster_world
# from jaxgmg.environments import lava_land
# from jaxgmg.environments import follow_me

from jaxgmg import util


def corner(
    # environment
    height: int = 7,
    width: int = 7,
    corner_size: int = 1,
    layout: str = 'tree',
    # splayer
    splayer: str = 'mouse', # or cheese or cheese-and-mouse
    # rendering
    level_of_detail: int = 1,
    print_image: bool = True,
    save_image: bool = False,
    # misc
    seed: int = 42,
):
    """
    Test the level splayers from the Cheese in the Corner environment.
    """
    match splayer:
        case 'cheese':
            splayer = cheese_in_the_corner.LevelSplayer.splay_cheese
        case 'mouse':
            splayer = cheese_in_the_corner.LevelSplayer.splay_mouse
        case 'cheese-and-mouse':
            splayer = cheese_in_the_corner.LevelSplayer.splay_cheese_and_mouse
        case _:
            raise ValueError(f"unknown splayer {splayer!r}")
    util.print_config(locals())
    
    print("preparing environment...")
    env = cheese_in_the_corner.Env(
        img_level_of_detail=level_of_detail,
    )

    print("preparing level generator...")
    level_generator = cheese_in_the_corner.LevelGenerator(
        height=height,
        width=width,
        corner_size=corner_size,
        maze_generator=maze_generation.get_generator_class_from_name(
            name=layout,
        )(),
    )

    print("generating a level...")
    rng = jax.random.PRNGKey(seed=seed)
    level = level_generator.sample(rng=rng)
    image = env.render_level(level)
    print(util.img2str(obs))

    print("splaying the level...")
    level_set = splayer(level)
    print("num levels:", level_set.num_levels)
    print("grid shape:", level_set.grid_shape)

    print("generating metamaze visualisation...")
    imgs = jax.vmap(env.render_level)(level_set.levels)
    img = einops.rearrange(
        jnp.zeros((
            *level_set.grid_shape,
            level_of_detail*height,
            level_of_detail*width,
            3,
        )).at[level_set.levels_pos].set(imgs),
        'H W h w c -> (H h) (W w) c',
    )

    if print_image:
        print(util.img2str(img))
    if save_image:
        print("saving image to 'metamaze.png'")
        util.save_image(img, 'metamaze.png')


