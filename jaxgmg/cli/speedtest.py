"""
Profiling speed of maze generation methods, level generation methods, and
environment update and render methods.
"""

import time
import tqdm
import numpy as np

import jax
import jax.numpy as jnp
import chex

from jaxgmg.procgen import maze_generation
from jaxgmg.environments import cheese_in_the_corner
from jaxgmg.environments import cheese_on_a_dish
from jaxgmg.environments import follow_me
from jaxgmg.environments import keys_and_chests
from jaxgmg.environments import lava_land
from jaxgmg.environments import monster_world
from jaxgmg.cli import util


# # # 
# Core speedtest methods


def speedtest_mazegen(
    rng : chex.PRNGKey,
    height : int,
    width : int,
    generator : maze_generation.MazeGenerator,
    batch_size : int,
    num_iters : int,
    num_trials : int,
):
    # define a trial
    @jax.jit
    def trial(rng):
        def iterate(_carry, rng_i):
            vgenerate = jax.vmap(generator.generate, in_axes=(0,None,None))
            rng_batch = jax.random.split(rng_i, batch_size)
            mazes = vgenerate(rng_batch, height, width)
            return None, mazes
        _carry, mazes = jax.lax.scan(
            iterate,
            None,
            jax.random.split(rng, num_iters),
        )
        return mazes
    
    # execute and time the trials
    trial_times = []
    for _ in tqdm.trange(num_trials, unit="trials"):
        rng_trial, rng = jax.random.split(rng)
        start_time = time.perf_counter()
        trial(rng=rng_trial)
        end_time = time.perf_counter()
        trial_times.append(end_time - start_time)
    trial_times = np.array(trial_times)

    # summarise the results
    batches_per_second = num_iters / trial_times
    mazes_per_second = batches_per_second * batch_size
    print('trial times:')
    print('  first trial: ', trial_times[0], 'seconds')
    print('  subseq. mean:', trial_times[1:].mean(), 'seconds')
    print('  subseq. stdv:', trial_times[1:].std(), 'seconds')
    print('mazes generated per second:')
    print('  first trial: ', mazes_per_second[0], 'mazes/sec')
    print('  subseq. mean:', mazes_per_second[1:].mean(), 'mazes/sec')
    print('  subseq. stdv:', mazes_per_second[1:].std(), 'mazes/sec')


# # # 
# Entry points for each maze generation method


def mazegen_tree(
    height: int = 13,
    width: int = 13,
    alt_kruskal_algorithm: bool = False,
    seed: int = 42,
    batch_size: int = 32,
    num_iters: int = 512,
    num_trials: int = 32,
):
    """
    Speedtest for tree maze generator.
    """
    util.print_config(locals())
    rng = jax.random.PRNGKey(seed=seed)
    speedtest_mazegen(
        rng=rng,
        height=height,
        width=width,
        generator=maze_generation.TreeMazeGenerator(
            alt_kruskal_algorithm=alt_kruskal_algorithm,
        ),
        batch_size=batch_size,
        num_iters=num_iters,
        num_trials=num_trials,
    )


def mazegen_edges(
    height: int = 13,
    width: int = 13,
    edge_prob: float = 0.75,
    seed: int = 42,
    batch_size: int = 32,
    num_iters: int = 512,
    num_trials: int = 32,
):
    """
    Speedtest for edge maze generator.
    """
    util.print_config(locals())
    rng = jax.random.PRNGKey(seed=seed)
    speedtest_mazegen(
        rng=rng,
        height=height,
        width=width,
        generator=maze_generation.EdgeMazeGenerator(
            edge_prob=edge_prob,
        ),
        batch_size=batch_size,
        num_iters=num_iters,
        num_trials=num_trials,
    )


def mazegen_blocks(
    height: int = 13,
    width: int = 13,
    wall_prob: float = 0.25,
    seed: int = 42,
    batch_size: int = 32,
    num_iters: int = 512,
    num_trials: int = 32,
):
    """
    Speedtest for tree maze generator.
    """
    util.print_config(locals())
    rng = jax.random.PRNGKey(seed=seed)
    speedtest_mazegen(
        rng=rng,
        height=height,
        width=width,
        generator=maze_generation.BlockMazeGenerator(
            wall_prob=wall_prob,
        ),
        batch_size=batch_size,
        num_iters=num_iters,
        num_trials=num_trials,
    )


def mazegen_noise(
    height: int = 13,
    width: int = 13,
    wall_threshold: float = 0.25,
    cell_size: int = 3,
    num_octaves: int = 1,
    seed: int = 42,
    batch_size: int = 32,
    num_iters: int = 512,
    num_trials: int = 32,
):
    """
    Speedtest for noise maze generator.
    """
    util.print_config(locals())
    rng = jax.random.PRNGKey(seed=seed)
    speedtest_mazegen(
        rng=rng,
        height=height,
        width=width,
        generator=maze_generation.NoiseMazeGenerator(
            wall_threshold=wall_threshold,
            cell_size=cell_size,
            num_octaves=num_octaves,
        ),
        batch_size=batch_size,
        num_iters=num_iters,
        num_trials=num_trials,
    )


def mazegen_open(
    height: int = 13,
    width: int = 13,
    seed: int = 42,
    batch_size: int = 32,
    num_iters: int = 512,
    num_trials: int = 32,
):
    """
    Speedtest for open maze generator.
    """
    util.print_config(locals())
    rng = jax.random.PRNGKey(seed=seed)
    speedtest_mazegen(
        rng=rng,
        height=height,
        width=width,
        generator=maze_generation.OpenMazeGenerator(),
        batch_size=batch_size,
        num_iters=num_iters,
        num_trials=num_trials,
    )


