"""
Rank/staleness-based prioritisation of a replay buffer. Used for prioritised
level replay and derivative autocurricula methods.
"""

import jax
import jax.numpy as jnp
from chex import Array


@jax.jit
def plr_replay_probs(
    temperature: float,
    staleness_coeff: float,
    scores: Array,              # float[buffer_size]
    last_visit_times: Array,    # int[buffer_size]
    current_time: int,
) -> Array:                     # float[buffer_size]
    """
    Conditional on sampling from the replay buffer, what is the probability
    of sampling each level in the replay buffer?
    """
    buffer_size, = scores.shape
    # ordinal score-based prioritisation
    ranks = (
        jnp.empty(buffer_size)
            .at[jnp.argsort(scores, descending=True)]
            .set(jnp.arange(1, buffer_size+1))
    )
    tempered_hvals = jnp.pow(1 / ranks, 1 / temperature)
    
    # staleness-aware prioritisation. Both PLR papers (Jiang+2020, Jiang+2021)
    # and both reference implementations define staleness as (c - C_i), so a
    # just-visited level (C_i == current_time) has staleness 0. See BUG-3 in
    # notes/03-bug-log.md.
    staleness = current_time - last_visit_times
    staleness_sum = staleness.sum()
    # Guard the degenerate all-equally-recent case (e.g. at init, before any
    # level has been revisited, every C_i == current_time so the sum is 0):
    # fall back to a uniform staleness distribution rather than dividing 0/0.
    staleness_probs = jnp.where(
        staleness_sum > 0,
        staleness / jnp.where(staleness_sum > 0, staleness_sum, 1),
        1 / buffer_size,
    )

    # probability of replaying each level is a mixture of these
    P_replay = (
        (1-staleness_coeff) * tempered_hvals / tempered_hvals.sum()
        + staleness_coeff * staleness_probs
    )
    return P_replay


