"""
Autocurricula stateful level generators.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp

from flax import struct
from chex import PRNGKey, Array

from jaxgmg.environments.base import Level, LevelGenerator

from jaxgmg.baselines.experience import Rollout


# # # 
# Base classes


@struct.dataclass
class CurriculumLevelGenerator:
    """
    Abstract base class for various autocurricula.
    """


    @struct.dataclass
    class State:
        pass

    
    @functools.partial(jax.jit, static_argnames=['self'])
    def init(self) -> State:
        return self.State()


    def get_batch(
        self,
        rng: PRNGKey,
        num_levels: int,
    ) -> tuple[
        State,
        Level, # Level[num_levels]
    ]:
        raise NotImplementedError


    @functools.partial(jax.jit, static_argnames=['self'])
    def update(
        self,
        state: State,
        rollouts: Rollout, # Rollout[num_levels] with Transition[num_steps]
        advantages: Array, # float[num_steps, num_levels]
    ) -> State:
        return state


    @functools.partial(jax.jit, static_argnames=['self'])
    def compute_metrics(self, state: State) -> dict[str, Any]:
        return {}


# # # 
# DOMAIN RANDOMISATION


@struct.dataclass
class FiniteDomainRandomisation(CurriculumLevelGenerator):

    
    @struct.dataclass
    class State(CurriculumLevelGenerator.State):
        levels: Level           # Level[num_levels]
        visit_counts: Array     # int[num_levels]


    @functools.partial(jax.jit, static_argnames=['self'])
    def init(self, levels):
        num_levels = jax.tree.leaves(levels)[0].shape[0]
        return self.State(
            levels=levels,
            visit_counts=jnp.zeros(num_levels, dtype=int),
        )


    @functools.partial(jax.jit, static_argnames=['self', 'num_levels'])
    def get_batch(
        self,
        state: State,
        rng: PRNGKey,
        num_levels: int,
    ) -> tuple[
        State,
        Level, # Level[num_levels]
    ]:
        num_levels_total = jax.tree.leaves(state.levels)[0].shape[0]
        level_ids = jax.random.choice(
            rng,
            num_levels_total,
            (num_levels,),
            replace=(num_levels > num_levels_total),
        )
        levels_batch = jax.tree.map(lambda x: x[level_ids], state.levels)
        new_state = state.replace(
            visit_counts=state.visit_counts.at[level_ids].add(1),
        )
        return new_state, levels_batch


    @functools.partial(jax.jit, static_argnames=['self'])
    def compute_metrics(self, state: State) -> dict[str, Any]:
        return {
            'prop_visited': (state.visit_counts > 0).mean(),
            'avg_visit_count': state.visit_counts.mean(),
        }


@struct.dataclass
class InfiniteDomainRandomisation(CurriculumLevelGenerator):
    level_generator: LevelGenerator


    @functools.partial(jax.jit, static_argnames=['self', 'num_levels'])
    def get_batch(
        self,
        state: CurriculumLevelGenerator.State,
        rng: PRNGKey,
        num_levels: int,
    ) -> tuple[
        CurriculumLevelGenerator.State,
        Level, # Level[num_levels]
    ]:
        levels_batch = self.level_generator.vsample(
            rng,
            num_levels=num_levels,
        )
        return state, levels_batch


