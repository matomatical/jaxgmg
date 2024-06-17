import jax
import jax.numpy as jnp
from flax import struct
import chex

import functools
from typing import Tuple, Dict, Any


# # # 
# domain randomisation


@struct.dataclass
class DRState:
    visited: chex.Array # bool[num_levels]


@struct.dataclass
class DR:
    num_levels: int


    @functools.partial(jax.jit, static_argnames=('self',))
    def init(self) -> DRState:
        return DRState(
            visited=jnp.zeros(self.num_levels, dtype=bool),
        )

    
    @functools.partial(jax.jit, static_argnames=('self',))
    def distribution(self, state) -> chex.Array:
        return jnp.ones(self.num_levels) / self.num_levels


    @functools.partial(jax.jit, static_argnames=('self', 'compute_metrics',))
    def update(
        self,
        state: DRState,
        chosen_level_ids: chex.Array,   # index[num_levels]
        gaes: chex.Array,               # float[num_steps, num_levels]
        compute_metrics: bool,
    ) -> Tuple[
        DRState,
        Dict[str, Any],
    ]:
        state = state.replace(
            visited=state.visited.at[chosen_level_ids].set(True),
        )
        if compute_metrics:
            metrics = {
                'visited_proportion': state.visited.mean(),
            }
        else:
            metrics = {}
        return state, metrics


# # # 
# prioritised level replay


@struct.dataclass
class PLRState:
    visited: chex.Array         # bool[num_levels]
    scores: chex.Array          # float[num_levels], masked by visited
    last_visited: chex.Array    # int[num_levels], masked by visited


@struct.dataclass
class PLR:
    num_levels: int
    temperature: float
    staleness_coeff: float


    @functools.partial(jax.jit, static_argnames=('self',))
    def init(self) -> PLRState:
        return PLRState(
            visited=jnp.zeros(self.num_levels, dtype=bool),
            scores=jnp.ones(self.num_levels) * -jnp.inf,
            last_visited=jnp.zeros(self.num_levels, dtype=int),
        )

    
    @functools.partial(jax.jit, static_argnames=('self',))
    def distribution(self, state) -> chex.Array:
        # uniform distribution over new levels
        unvisited = ~state.visited
        P_new = normalise_pmf(unvisited)

        # score-based distribution over replay levels
        ranks = (
            jnp.empty(self.num_levels)
                .at[jnp.argsort(state.scores, descending=True)]
                .set(jnp.arange(self.num_levels))
        )
        hvals = (1 / (1+ranks)) * state.visited
        P_score = normalise_pmf(jnp.pow(hvals, 1/self.temperature))
        
        # staleness distribution over replay levels
        steps_since_visited = (
            state.visited * (1 + state.last_visited.max() - state.last_visited)
        )
        P_staleness = normalise_pmf(steps_since_visited)

        # replay distribution mixes staleness and score distributions
        P_replay = (
            (1-self.staleness_coeff) * P_score
            + self.staleness_coeff * P_staleness
        )

        # full distribution mixes replay and new distributions
        proportion_visited = state.visited.sum() / self.num_levels
        P = (
            proportion_visited * P_replay
            + (1-proportion_visited) * P_new
        )
        return P


    @functools.partial(jax.jit, static_argnames=('self', 'compute_metrics',))
    def update(
        self,
        state: PLRState,
        chosen_level_ids: chex.Array,   # index[num_levels]
        gaes: chex.Array,               # float[num_steps, num_levels]
        compute_metrics: bool,
    ) -> Tuple[
        PLRState,
        Dict[str, Any],
    ]:
        # mark visited levels
        new_visited = state.visited.at[chosen_level_ids].set(True)

        # compute and update score for visited levels
        new_scores = state.scores.at[chosen_level_ids].set(
            jnp.abs(gaes).mean(axis=0), # -> float[num_levels]
        )

        # update visited markers
        new_last_visited = state.last_visited.at[chosen_level_ids].set(
            state.last_visited.max() + 1
        )

        # update state
        state = state.replace(
            visited=new_visited,
            scores=new_scores,
            last_visited=new_last_visited,
        )

        if compute_metrics:
            metrics = {
                'visited_proportion': state.visited.mean(),
            }
        else:
            metrics = {}
        
        return state, metrics


def normalise_pmf(vector, axis=None, eps=1e-10):
    normaliser = jnp.maximum(vector.sum(axis=axis, keepdims=True), eps)
    return vector / normaliser


