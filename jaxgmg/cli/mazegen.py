"""
Demonstrations of maze generation methods.
"""

import jax
from jaxgmg.procgen import maze_generation
from jaxgmg.cli import util


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


