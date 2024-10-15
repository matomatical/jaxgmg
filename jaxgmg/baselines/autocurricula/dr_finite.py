"""
Domain randomisation over a fixed, finite set of levels. Levels are sampled
at creation time and then batches are sampled from this buffer repeatedly.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp

from flax import struct
from chex import PRNGKey, Array

from jaxgmg.environments.base import Level, LevelGenerator, LevelMetrics
from jaxgmg.baselines.experience import Rollout
import jaxgmg.baselines.autocurricula.base as base


@struct.dataclass
class GeneratorState(base.GeneratorState):
    levels: Level           # Level[num_levels]
    visit_counts: Array     # int[num_levels]


@struct.dataclass
class CurriculumGenerator(base.CurriculumGenerator):


    @functools.partial(jax.jit, static_argnames=['self'])
    def init(self, levels):
        num_levels = jax.tree.leaves(levels)[0].shape[0]
        return GeneratorState(
            levels=levels,
            visit_counts=jnp.zeros(num_levels, dtype=int),
        )


    @functools.partial(jax.jit, static_argnames=['self', 'num_levels'])
    def get_batch(
        self,
        state: GeneratorState,
        rng: PRNGKey,
        num_levels: int,
    ) -> tuple[
        GeneratorState,
        Level, # Level[num_levels]
        int,
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
        return new_state, levels_batch, 0


    def batch_type_name(self, batch_type: int) -> str:
        return "generate"


    def should_train(self, cycle: int, batch_type: int) -> bool:
        return True


    @functools.partial(jax.jit, static_argnames=['self'])
    def compute_metrics(self, state: GeneratorState) -> dict[str, Any]:
        return {
            'prop_visited': (state.visit_counts > 0).mean(),
            'avg_visit_count': state.visit_counts.mean(),
        }


