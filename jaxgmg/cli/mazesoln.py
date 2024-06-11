"""
Demonstrating maze solution functionality.
"""

import jax
import jax.numpy as jnp
import einops
from jaxgmg.procgen import maze_generation
from jaxgmg.procgen import maze_solving
from jaxgmg.cli import util


def distances(
    layout: str = 'tree',
    height: int = 7,
    width:  int = 7,
    seed: int = 42,
):
    """
    Solve a maze and plot the optimal distance from any source to any
    destination.
    """
    util.print_config(locals())

    print("generating maze...")
    rng = jax.random.PRNGKey(seed=seed)
    maze = maze_generation.get_generator_class_from_name(
        name=layout,
    )().generate(
        key=rng,
        height=height,
        width=width,
    )
    print(util.img2str(maze * .25))

    print("solving maze...")
    dist = maze_solving.maze_distances(maze)

    print("visualising solution...")
    # transform inf -> -inf, [0,max] -> [0.5,1.0] for visualisation
    dist_finite = jnp.nan_to_num(dist, posinf=-jnp.inf) # to allow max
    maxd = dist_finite.max()
    dist_01 = (maxd + jnp.nan_to_num(dist_finite, neginf=-maxd)) / (2*maxd)
    # rearrange into macro/micro maze format
    print(util.img2str(
        einops.rearrange(dist_01, 'H W h w -> (H h) (W w)'),
        colormap=util.viridis,
    ))
    util.print_legend({
        0.0: "wall/unreachable",
        0.5: "distance 0",
        1.0: f"distance {maxd.astype(int)}",
    }, colormap=util.viridis)
    print("the source is represented by the square in the macromaze")
    print("the target is represented by the square in the micromaze")


def directions(
    layout: str = 'tree',
    height: int = 7,
    width:  int = 7,
    stay_action: bool = True,
    seed: int = 42,
):
    """
    Solve a maze and plot the optimal direction to take from any source to
    get to any destination.
    """
    util.print_config(locals())

    print("generating maze...")
    rng = jax.random.PRNGKey(seed=seed)
    maze = maze_generation.get_generator_class_from_name(
        name=layout,
    )().generate(
        key=rng,
        height=height,
        width=width,
    )
    print(util.img2str(maze * .25))

    print("solving maze...")
    soln = maze_solving.maze_optimal_directions(maze, stay_action=stay_action)

    print("visualising directions...")
    # transform {0,1,2,3,4} -> {1,2,3,4,5} and walls -> 0
    soln = (1 + soln) * ~maze * ~maze[:,:,None,None]
    # rearrange into macro/micro maze format
    print(util.img2str(
        einops.rearrange(soln, 'H W h w -> (H h) (W w)'),
        colormap=util.sweetie16,
    ))
    util.print_legend({
        0: "n/a",
        1: "up",
        2: "left",
        3: "down",
        4: "right",
        5: "(stay)",
    }, colormap=util.sweetie16)
    print("the source is represented by the square in the macromaze")
    print("the target is represented by the square in the micromaze")


def distances_and_directions(
    layout: str = 'tree',
    height: int = 7,
    width: int = 7,
    seed: int = 42,
    save_image: bool = False,
    image_upscale: int = 4,
):
    """
    Solve a maze and plot the optimal distance as well as the direction to
    take from any source to get to any destination.
    """
    print("generating maze...")
    rng = jax.random.PRNGKey(seed=seed)
    maze = maze_generation.get_generator_class_from_name(
        name=layout,
    )().generate(
        key=rng,
        height=height,
        width=width,
    )
    print(util.img2str(maze * .25))

    print("solving maze...")
    soln = maze_solving.maze_directional_distances(maze)
    dist = soln[:,:,:,:,4]
    dirs = soln.argmin(axis=4)
    
    print("visualising solution...")
    # transform inf -> -inf, [0,max] -> [0.5,1.0] for visualisation
    dist_finite = jnp.nan_to_num(dist, posinf=-jnp.inf) # to allow max
    maxd = dist_finite.max()
    dist_01 = (maxd + jnp.nan_to_num(dist_finite, neginf=-maxd)) / (2*maxd)
    # rearrange into macro/micro maze format
    dist_vis = einops.rearrange(dist_01, 'H W h w -> (H h) (W w)')
    # print
    dist_rgb = util.viridis(dist_vis)
    print(util.img2str(dist_rgb))
    util.print_legend({
        0.0: "wall/unreachable",
        0.5: "distance 0",
        1.0: f"distance {maxd.astype(int)}",
    }, colormap=util.viridis)

    print("visualising directions...")
    # transform {0,1,2,3,4} -> {8,7,6,5,4} and walls -> 0
    dirs = (8 - dirs) * ~maze * ~maze[:,:,None,None]
    # rearrange into macro/micro maze format
    dirs_vis = einops.rearrange(dirs, 'H W h w -> (H h) (W w)')
    # print
    dirs_rgb = util.sweetie16(dirs_vis)
    print(util.img2str(dirs_rgb))
    util.print_legend({
        0: "n/a",
        8: "up",
        7: "left",
        6: "down",
        5: "right",
        4: "(stay)",
    }, colormap=util.sweetie16)
    print("the source is represented by the square in the macromaze")
    print("the target is represented by the square in the micromaze")

    if save_image:
        both = einops.rearrange(
            jnp.stack((dist_rgb, dirs_rgb)),
            'both h w rgb -> h (both w) rgb',
        )
        print("saving to ./out.png...")
        util.save_image(both, "out.png", upscale=image_upscale)


