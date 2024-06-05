"""
JAX implementation of 2D Perlin noise and associated fractal noise.

This implementation is loosely based on Craftax's JAX implementation
(https://github.com/MichaelTMatthews/Craftax/blob/main/craftax/craftax_classic/util/noise.py,
based in turn on Pierre Vigier's numpy implementation https://github.com/pvigier/perlin-numpy).

Compared to these I refactored the API and algorithm and made some attempt to
improve the readability of the code using additional documentation, and added
some optimisations that appear to lead to minor improvements for large sizes
on my CPU (TODO: benchmark on GPU, also try again with matrix multiplications?)
"""

import functools
import jax
import jax.numpy as jnp


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

    # RANDOM ANGLES
    # randomly generate a gradient angle for each cell vertex
    k_gradients, key = jax.random.split(key)
    angles = jax.random.uniform(
        key=k_gradients,
        shape=(num_rows + 1, num_cols + 1),
        minval=0,
        maxval=2 * jnp.pi,
    )
    
    # GRADIENT VECTORS
    # associate to each cell the four gradient vectors of its vertices
    gradients = jnp.dstack((
        jnp.cos(angles),
        jnp.sin(angles),
    ))[:,jnp.newaxis,:,jnp.newaxis,:]
    g00 = gradients[ :-1,:, :-1,:,:]
    g10 = gradients[1:  ,:, :-1,:,:]
    g01 = gradients[ :-1,:,1:  ,:,:]
    g11 = gradients[1:  ,:,1:  ,:,:]
    # shape: num_rows 1 num_cols 1 2
    
    # OFFSETS
    # associate to each point within a cell its offset from each cell vertex
    row_offsets = jnp.arange(cell_height) / cell_height
    col_offsets = jnp.arange(cell_width) / cell_width
    offsets = jnp.dstack(jnp.meshgrid(
        row_offsets,
        col_offsets,
        indexing='ij',
    ))[jnp.newaxis,:,jnp.newaxis,:,:]
    o00 = offsets
    o10 = offsets - jnp.array((1,0))
    o01 = offsets - jnp.array((0,1))
    o11 = offsets - jnp.array((1,1))
    # shape: 1 cell_height 1 cell_width 2
    
    # RAMPS / DOT PRODUCTS
    # compute for each point the dot products of offset and gradient vectors
    r00 = jnp.sum(g00 * o00, axis=-1)
    r10 = jnp.sum(g10 * o10, axis=-1)
    r01 = jnp.sum(g01 * o01, axis=-1)
    r11 = jnp.sum(g11 * o11, axis=-1)
    # shape: num_rows cell_height num_cols cell_width

    # INTERPOLATION WEIGHTS
    # associate to each point some mixing weights for its four dot products
    vtime = interpolant(row_offsets)[jnp.newaxis,:,jnp.newaxis,jnp.newaxis]
    htime = interpolant(col_offsets)[jnp.newaxis,jnp.newaxis,jnp.newaxis,:]
    vtime_comp = 1 - vtime
    htime_comp = 1 - htime
    t00 = vtime_comp * htime_comp
    t10 = vtime      * htime_comp
    t01 = vtime_comp * htime
    t11 = vtime      * htime
    # shape: 1 cell_height 1 cell_width
    
    # MIXING
    # for each point interpolate the four ramp values to get the final value
    noise = t00 * r00 + t10 * r10 + t01 * r01 + t11 * r11
    # shape: num_rows cell_height num_cols cell_width

    # RESHAPE
    # merge the grid and cell axes
    noise = noise.reshape(height, width)
    
    # SCALE
    # scaling by sqrt(2) here makes the final range [-1, 1]
    # see https://digitalfreepen.com/2017/06/20/range-perlin-noise.html
    noise = noise * jnp.sqrt(2)
    
    return noise


@functools.partial(
    jax.jit,
    static_argnames=(
        'height',
        'width',
        'base_num_rows',
        'base_num_cols',
        'num_octaves',
        'interpolant',
    ),
)
def generate_fractal_noise(
    key,
    height: int,
    width: int,
    base_num_rows: int,
    base_num_cols: int,
    num_octaves: int,
    interpolant=smootherstep,
):
    """
    Two-dimensional fractal noise generated based on superimposing Perlin
    noise.

    Parameters:

    * key : PRNGKey
        RNG state used to generate angles. Consumed.
    * height : int
        Number of rows in the generated noise grid.
    * width : int
        Number of columns in the generated noise grid.
    * base_num_rows : int
        Number of rows in the largest macroscopic grid of cells used to
        generate the first layer of noise.
        Must divide height and must be divisble by 2^{num_octaves-1}.
    * base_num_cols : int
        Number of columns in the largest macroscopic grid of cells used to
        generate the first layer of noise.
        Must divide width and must be divisible by 2^{num_octaves-1}.
    * num_octaves : int (>= 1)
        Number of iterations of noise to superimpose.
    * interpolant : function from [0.,1.] to [0.,1.] (e.g. smootherstep)
        Function used to interpolate between the corners of each cell.

    Returns:

    * noise : float[height, width]
        The noise grid, values between -1.0 and +1.0.
    """
    # accumulate noise of increasing frequency/resolution
    noise = jnp.zeros((height, width))
    frequency = 1
    amplitude = 1
    for _ in range(num_octaves):
        k_octave, key = jax.random.split(key)
        noise = noise + amplitude * generate_perlin_noise(
            key=k_octave,
            height=height,
            width=width,
            num_rows=frequency * base_num_rows,
            num_cols=frequency * base_num_cols,
            interpolant=interpolant,
        )
        frequency *= 2
        amplitude *= 0.5

    # scale to the range [-1, 1]
    # total_amplitude = 1.0 + 0.5 + 0.25 + ... + 0.5^{num_octaves-1}
    total_amplitude = 2 - (0.5 ** (num_octaves-1))
    noise = noise / total_amplitude

    return noise
