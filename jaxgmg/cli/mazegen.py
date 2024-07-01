"""
Demonstrations of maze generation methods.
"""

import jax
import jax.numpy as jnp
import einops

from jaxgmg.procgen import maze_generation
from jaxgmg import util


def tree(
    height: int = 79,
    width: int = 79,
    alt_kruskal_algorithm: bool = False,
    seed: int = 42,
):
    """
    Generate and visualise a random tree maze.
    """
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    gen = maze_generation.TreeMazeGenerator(
        alt_kruskal_algorithm=alt_kruskal_algorithm,
    )
    maze = gen.generate(
        key=rng,
        height=height,
        width=width,
    )
    print(util.img2str(maze * .25))


def edges(
    height: int = 79,
    width: int = 79,
    edge_prob: float = 0.75,
    seed: int = 42,
):
    """
    Generate and visualise a random edge maze.
    """
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    gen = maze_generation.EdgeMazeGenerator(
        edge_prob=edge_prob,
    )
    maze = gen.generate(
        key=rng,
        height=height,
        width=width,
    )
    print(util.img2str(maze * .25))


def noise(
    height: int = 79,
    width: int = 79,
    wall_threshold: float = 0.25,
    cell_size: int = 3,
    num_octaves: int = 1,
    seed: int = 42,
):
    """
    Generate and visualise a maze based on (fractal) Perlin noise.
    """
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    gen = maze_generation.NoiseMazeGenerator(
        wall_threshold=wall_threshold,
        cell_size=cell_size,
        num_octaves=num_octaves,
    )
    maze = gen.generate(
        key=rng,
        height=height,
        width=width,
    )
    print(util.img2str(maze * .25))


def blocks(
    height: int = 79,
    width: int = 79,
    wall_prob: float = 0.25,
    seed: int = 42,
):
    """
    Generate and visualise a random block maze.
    """
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    gen = maze_generation.BlockMazeGenerator(
        wall_prob=wall_prob,
    )
    maze = gen.generate(
        key=rng,
        height=height,
        width=width,
    )
    print(util.img2str(maze * .25))


def open(
    height: int = 79,
    width: int = 79,
    seed: int = 42, # unused
):
    """
    Generate and visualise an open maze.
    """
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    gen = maze_generation.OpenMazeGenerator()
    maze = gen.generate(
        key=rng,
        height=height,
        width=width,
    )
    print(util.img2str(maze * .25))


def mural(
    height: int = 23,
    width: int = 23,
    num_cols: int = 3,
    seed: int = 42,
    save_image: bool = False,
    image_upscale: int = 1,
):
    """
    Generate an image demonstrating the core maze generation algorithms.
    """
    util.print_config(locals())
    rng = jax.random.PRNGKey(seed=seed)

    print("defining maze generators...")
    generators = [
        maze_generation.TreeMazeGenerator(),
        maze_generation.EdgeMazeGenerator(edge_prob=0.75),
        maze_generation.EdgeMazeGenerator(edge_prob=0.85),
        maze_generation.BlockMazeGenerator(wall_prob=0.25),
        maze_generation.NoiseMazeGenerator(cell_size=2, num_octaves=1),
        maze_generation.NoiseMazeGenerator(cell_size=3, num_octaves=1),
        maze_generation.NoiseMazeGenerator(cell_size=8, num_octaves=2),
        maze_generation.OpenMazeGenerator(),
    ]
    
    print("generating mazes...")
    mazes = []
    for i, g in enumerate(generators, 1):
        rng_g, rng = jax.random.split(rng)
        vgenerate = jax.vmap(g.generate, in_axes=(0,None,None))
        rng_gs = jax.random.split(rng_g, num_cols)
        mazes.append(i * vgenerate(rng_gs, height, width))
    
    print("reformatting to one image...")
    mural = jnp.stack(mazes) # -> len(generators) col_width height width
    mural = jnp.pad(mural, ((0,0),(0,0),(0,1),(0,1)))
    mural = einops.rearrange(mural, 'len col h w -> (len h) (col w)')
    mural = mural[:-1,:-1] # remove excess padding
    mural = util.sweetie16(mural) # colorise

    print("printing...")
    print(util.img2str(mural))

    if save_image:
        print("saving to ./out.png...")
        util.save_image(mural, "out.png", upscale=image_upscale)



