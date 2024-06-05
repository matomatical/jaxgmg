"""
JAX implementation of 2D Perlin noise and associated fractal noise.

This implementation is loosely based on Craftax's JAX implementation
(https://github.com/MichaelTMatthews/Craftax/blob/main/craftax/craftax_classic/util/noise.py,
based in turn on Pierre Vigier's numpy implementation https://github.com/pvigier/perlin-numpy).

Compared to these I refactored the API and algorithm and made some attempt to
improve the readability of the code using additional documentation and the
einops library.

I also added some speculative optimisations:

* Use matrix multiplications instead of element wise product and sum for dot
  product and interpolation steps.
* Reduce number of operations by deferring repeats/broadcasts until
  necessary.

TODO: Profile to see if this leads to a speed-up?
"""

import functools
import jax
import jax.numpy as jnp
import einops


# # # 
# Interpolation functions


def smootherstep(t: float):
    """
    Ken Perlin's 'smootherstep' sigmoidal function.

    A quintic polynomial S : [0,1] -> [0,1] satisfying:

    * S(0) = 0 and S(1) = 1
    * S'(0)  = S'(1)  = 0 (zero first derivative at boundaries),
    * S''(0) = S''(1) = 0 (zero second derivative at boundaries).

    The input, t, should be within the range [0, 1].
    """
    return t * t * t * (t * (6. * t - 15.) + 10.)


# # # 
# Noise algorithms


@functools.partial(
    jax.jit,
    static_argnames=(
        'height',
        'width',
        'num_rows',
        'num_cols',
        'interpolant',
    ),
)
def generate_perlin_noise(
    key,
    height: int,
    width: int,
    num_rows: int,
    num_cols: int,
    interpolant=smootherstep,
):
    """
    Two-dimensional Perlin noise.

    Parameters:

    * key : PRNGKey
        RNG state used to generate angles. Consumed.
    * height : int
        Number of rows in the generated noise grid.
    * width : int
        Number of columns in the generated noise grid.
    * num_rows : int (must divide height)
        Number of rows in the macroscopic grid of cells used to generate the
        noise.
    * num_cols : int (must divide width)
        Number of columns in the macroscopic grid of cells used to generate
        the noise.
    * interpolant : function from [0.,1.] to [0.,1.] (e.g. smootherstep)
        Function used to interpolate between the corners of each cell.

    Returns:

    * noise : float[height, width]
        The noise grid, values between -1.0 and +1.0.

    """
    cell_height = height // num_rows
    cell_width = width // num_cols

    # RANDOM GRADIENTS
    # randomly generate unit vector gradients for each cell vertex
    k_gradients, key = jax.random.split(key)
    angles = jax.random.uniform(
        key=k_gradients,
        shape=(num_rows + 1, num_cols + 1),
        minval=0,
        maxval=2 * jnp.pi,
    )
    vertex_gradients = jnp.dstack((jnp.cos(angles), jnp.sin(angles)))
    # associate to each cell the four gradients of its vertices
    cell_grid_neighbour_gradients = jnp.stack((
            vertex_gradients[:-1,:-1],
            vertex_gradients[1:, :-1],
            vertex_gradients[:-1,1:],
            vertex_gradients[1:, 1:],
        ),
        axis=2,
    ) # -> h_grid w_grid vertex xy
    
    # OFFSETS
    # associate to each point within a cell its offset from each cell vertex
    row_offsets = jnp.arange(cell_height) / cell_height
    col_offsets = jnp.arange(cell_width) / cell_width
    cell_offsets = jnp.dstack(jnp.meshgrid(
        row_offsets,
        col_offsets,
        indexing='ij',
    ))
    one_cell_neighbour_offsets = einops.rearrange(
        cell_offsets,
        'h_cell w_cell xy -> h_cell w_cell 1 xy',
    ) - jnp.array([(0,0),(1,0),(0,1),(1,1)])

    # RAMPS
    # compute for each point the dot products of offset and gradient vectors
    neighbour_gradients_row_vectors = einops.rearrange(
        cell_grid_neighbour_gradients,
        'h_grid w_grid vertex xy -> h_grid 1  w_grid 1  vertex  1 xy',
    )
    neighbour_offsets_col_vectors = einops.rearrange(
        one_cell_neighbour_offsets,
        'h_cell w_cell vertex xy -> 1 h_cell  1 w_cell  vertex  xy 1',
    )
    neighbour_products = einops.rearrange(
        neighbour_gradients_row_vectors @ neighbour_offsets_col_vectors,
        'h_grid h_cell w_grid w_cell vertex 1 1 -> h_grid h_cell w_grid w_cell vertex',
    )

    # INTERPOLATION
    # associate to each point within a cell how to weight its four products
    row_weights = interpolant(row_offsets)
    row_weights_row_vectors = einops.rearrange(
        jnp.stack((1-row_weights, row_weights)),
        't_row h_cell -> h_cell 1 1 t_row',
    )
    col_weights = interpolant(col_offsets)
    col_weights_col_vectors = einops.rearrange(
        jnp.stack((1-col_weights, col_weights)),
        't_col w_cell -> 1 w_cell t_col 1',
    )
    one_cell_interpolation_weights = einops.rearrange(
        col_weights_col_vectors @ row_weights_row_vectors,
        'h_cell w_cell t_col t_row -> h_cell w_cell (t_col t_row)',
    )
    
    # for each point interpolate the four ramp values to get the final value
    neighbour_products_row_vectors = einops.rearrange(
        neighbour_products,
        'h_grid h_cell w_grid w_cell vertex -> h_grid h_cell w_grid w_cell 1 vertex',
    )
    interpolation_weights_col_vectors = einops.rearrange(
        one_cell_interpolation_weights,
        'h_cell w_cell vertex -> 1 h_cell 1 w_cell vertex 1',
    )
    noise = einops.rearrange(
        neighbour_products_row_vectors @ interpolation_weights_col_vectors,
        'h_grid h_cell w_grid w_cell 1 1 -> (h_grid h_cell) (w_grid w_cell)',
    )

    # SCALING
    # scale noise into the desired range [-1, 1].
    # (see https://digitalfreepen.com/2017/06/20/range-perlin-noise.html)
    noise = jnp.sqrt(2) * noise

    return noise


def generate_fractal_noise(
    rng,
    shape,
    res,
    octaves=1,
    persistence=0.5,
    lacunarity=2,
    interpolant=smootherstep,
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
