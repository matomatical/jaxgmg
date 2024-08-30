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
        int,
    ]:
        """
        Sample a batch of levels from the curriculum.

        Returns:

        * state : GeneratorState
                The updated curriculum state.
        * levels : Level[num_levels]
                The batch of levels.
        * batch_type : int
                Indicates the type of the batch (e.g. a replay batch vs. a
                newly generated batch in PLR).
        """
        raise NotImplementedError

    
    def batch_type_name(self, batch_type: int) -> str:
        """
        You got a batch of this type from the `get_batch` method. This str is
        a name you can use in your logging 
        """
        raise NotImplementedError


    def should_train(self, batch_type: int) -> bool:
        """
        You got a batch of this type from the `get_batch` method . This bool
        tells you if you should do PPO updates (True) or not (False).
        """
        raise NotImplementedError



    @functools.partial(jax.jit, static_argnames=['self'])
    def update(
        self,
        state: GeneratorState,
        levels: Level,                  # Level[num_levels]
        rollouts: Rollout,              # Rollout[num_levels] (num_steps)
        advantages: Array,              # float[num_levels, num_steps]
        proxy_advantages: Array | None, # float[num_levels, num_steps]
    ) -> GeneratorState:
        return state


    @functools.partial(jax.jit, static_argnames=['self'])
    def compute_metrics(self, state: GeneratorState) -> dict[str, Any]:
        return {}


