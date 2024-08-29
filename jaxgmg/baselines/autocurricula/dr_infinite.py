"""
Domain randomisation over an entire level generator. Samples new levels from
the generator for each level batch.
"""

import functools

import jax
import jax.numpy as jnp

from flax import struct
from chex import PRNGKey

from jaxgmg.environments.base import Level, LevelGenerator
from jaxgmg.baselines.experience import Rollout
import jaxgmg.baselines.autocurricula.base as base


@struct.dataclass
class CurriculumGenerator(base.CurriculumGenerator):
    level_generator: LevelGenerator


    @functools.partial(jax.jit, static_argnames=['self', 'num_levels'])
    def get_batch(
        self,
        state: base.GeneratorState,
        rng: PRNGKey,
        num_levels: int,
    ) -> tuple[
        base.GeneratorState,
        Level, # Level[num_levels]
        bool,
    ]:
        levels_batch = self.level_generator.vsample(
            rng,
            num_levels=num_levels,
        )
        return state, levels_batch, True


