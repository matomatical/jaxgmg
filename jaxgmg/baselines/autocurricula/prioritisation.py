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
    
    # staleness-aware prioritisation
    staleness = 1 + current_time - last_visit_times # TODO: is 1+ correct?

    # probability of replaying each level is a mixture of these
    P_replay = (
        (1-staleness_coeff) * tempered_hvals / tempered_hvals.sum()
        + staleness_coeff * staleness / staleness.sum()
    )
    return P_replay


