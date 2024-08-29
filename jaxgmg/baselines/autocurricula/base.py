"""
Autocurricula stateful level generators. Abstract base class defining the API
for specific methods to follow.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp

from flax import struct
from chex import PRNGKey, Array

from jaxgmg.environments.base import Level
from jaxgmg.baselines.experience import Rollout
    

@struct.dataclass
class GeneratorState:
    """
    Abstract base class for the state object for various autocurricula.
    Subclass will add things like level buffers and score tracking data.
    """


@struct.dataclass
class CurriculumGenerator:
    """
    Abstract base class for various autocurricula. Subclass will override the
    `get_batch` method and optionally other methods.
    """


    @functools.partial(jax.jit, static_argnames=['self'])
    def init(self) -> GeneratorState:
        return GeneratorState()


    def get_batch(
        self,
        state: GeneratorState,
        rng: PRNGKey,
        num_levels: int,
    ) -> tuple[
        GeneratorState,
        Level, # Level[num_levels] # TODO: or more levels?
        bool,
    ]:
        """
        Sample a batch of levels from the curriculum.

        Returns:

        * state : GeneratorState
                The updated curriculum state.
        * levels : Level[num_levels]
                The batch of levels.
                NOTE: COULD BE MORE THAN NUM_LEVELS LEVELS IN SOME CASES.
        * curated : bool
                Whether these levels have come from a curated selection and
                should be used for training the policy, or not.
                Either way, the caller should collect experience in these
                levels and pass it back to the curriculum generator via the
                update method.
                But if this flag is True, one can also do training steps in
                these levels, whereas if it is False, one should not.
        """
        raise NotImplementedError


    @functools.partial(jax.jit, static_argnames=['self'])
    def update(
        self,
        state: GeneratorState,
        levels: Level,      # Level[num_levels]
        rollouts: Rollout,  # Rollout[num_levels] with Transition[num_steps]
        advantages: Array,  # float[num_levels, num_steps]
    ) -> GeneratorState:
        return state


    @functools.partial(jax.jit, static_argnames=['self'])
    def compute_metrics(self, state: GeneratorState) -> dict[str, Any]:
        return {}


