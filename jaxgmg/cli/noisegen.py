"""
Demonstration of noise generation methods.
"""

import jax
from jaxgmg.procgen import noise_generation
from jaxgmg import util


def perlin(
    height: int = 64,
    width:  int = 64,
    num_rows: int = 8,
    num_cols: int = 8,
    seed: int = 42,
):
    """
    Generate and visualise some 2d Perlin noise.
    """
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    noise = noise_generation.generate_perlin_noise(
        key=rng,
        height=height,
        width=width,
        num_rows=num_rows,
        num_cols=num_cols,
    )
    noise_0to1 = (noise + 1) / 2
    print(util.img2str(noise_0to1, colormap=util.viridis))


def fractal(
    height: int = 64,
    width:  int = 64,
    base_num_rows: int = 8,
    base_num_cols: int = 8,
    num_octaves: int = 4,
    seed: int = 42,
):
    """
    Generate and visualise some 2d fractal Perlin noise.
    """
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    noise = noise_generation.generate_fractal_noise(
        key=rng,
        height=height,
        width=width,
        base_num_rows=base_num_rows,
        base_num_cols=base_num_cols,
        num_octaves=num_octaves,
    )
    noise_0to1 = (noise + 1) / 2
    print(util.img2str(noise_0to1, colormap=util.viridis))


