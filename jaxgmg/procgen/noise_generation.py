"""
JAX implementation of 2D perlin noise and associated fractal noise.

Credits:

* Closely based on Craftax's JAX implementation
  (https://github.com/MichaelTMatthews/Craftax/blob/main/craftax/craftax_classic/util/noise.py)
* Which was, in turn, based on Pierre Vigier's numpy implementation
  (https://github.com/pvigier/perlin-numpy)
"""

import jax
import jax.numpy as jnp


def interpolant(t):
    # perlin's 'smootherstep' function
    return t * t * t * (t * (t * 6 - 15) + 10)


def generate_perlin_noise(
    rng,
    noise_height,
    noise_width,
    num_rows,
    num_cols,
    interpolant=interpolant,
):
    cell_height = noise_height // num_rows
    cell_width = noise_width // num_cols
    
    # d[0] = cell_height
    # d[1] = cell_width
    
    t_linear = # TODO rewrite the following 'grid' thing in terms of cells

    delta = (num_rows / noise_height, num_cols / noise_width)
    grid = (
        jnp.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
    )

    # Gradients
    rng, _rng = jax.random.split(rng)
    if override_angles is not None:
        angles = 2 * jnp.pi * override_angles
    else:
        angles = 2 * jnp.pi * jax.random.uniform(_rng, (res[0] + 1, res[1] + 1))
    gradients = jnp.dstack((jnp.cos(angles), jnp.sin(angles)))
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[: -d[0], : -d[1]]
    g10 = gradients[d[0] :, : -d[1]]
    g01 = gradients[: -d[0], d[1] :]
    g11 = gradients[d[0] :, d[1] :]

    # Ramps
    n00 = jnp.sum(jnp.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = jnp.sum(jnp.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = jnp.sum(jnp.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = jnp.sum(jnp.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)

    # Interpolation
    t = interpolant(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return jnp.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def generate_fractal_noise_2d(
    rng,
    shape,
    res,
    octaves=1,
    persistence=0.5,
    lacunarity=2,
    interpolant=interpolant,
    override_angles=None,
):
    noise = jnp.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        rng, _rng = jax.random.split(rng)
        noise += amplitude * generate_perlin_noise_2d(
            _rng,
            shape,
            (frequency * res[0], frequency * res[1]),
            interpolant,
            override_angles=override_angles,
        )
        frequency *= lacunarity
        amplitude *= persistence

    # Normalise
    noise = (noise - noise.min()) / (noise.max() - noise.min())

    return noise


def main():
    rng = jax.random.PRNGKey(0)
    noise = generate_fractal_noise_2d(rng, (64, 64), (1,1), octaves=4,
                                      persistence=0.25, lacunarity=4)
    print(img2str(noise, colormap=viridis))


if __name__ == "__main__":
    main()

