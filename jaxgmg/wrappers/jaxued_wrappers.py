"""
Wraps one of our custom environments with a JaxUED-conformant API. There are
some minor differences from the JaxUED reference environment:

*   We use our own EnvState type for the state and Level type for levels,
    that are not subclasses of the types defined in the module
    `underspecified_env` and given in the API function signatures
    (actually, the JaxUED reference maze implementation does this too).
*   We use a plain chex.Array jax array type instead of an Observation
    struct.
*   We keep the empty EnvParams type from jaxued, but it's unused (parameters
    are instead provided as fields of the original (wrapped) Env).

Note that we 

Nevertheless `env = UnderspecifiedEnvWrapper(env)` will have the core
properties and methods of a jaxued `UnderspecifiedEnv`:
    
* `params = env.default_params`
* `action_space = env.action_space(params)`
* `obs, state = env.reset_to_level(rng, level, params)`
* `obs, state, reward, done, info = env.step(rng, state, action, params)`
"""

import jax
import functools
from typing import Tuple
import chex
from flax import struct
from gymnax.environments import spaces

from jaxued.environments import UnderspecifiedEnv
from jaxued.environments.underspecified_env import EnvParams

from jaxgmg.environments.base import Env, EnvState, Level


Observation = chex.Array


@struct.dataclass
class UnderspecifiedEnvWrapper(UnderspecifiedEnv):
    """
    Wraps a custom environment to give it the JaxUED 'underspecified
    environment' API.

    The base environment should be a `gmg_environments.base.Env` subclass.
    """
    env: Env
        

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()
    

    def action_space(
        self,
        params: EnvParams, # unused
    ) -> spaces.Discrete:
        return spaces.Discrete(self.env.num_actions)

    
    @functools.partial(jax.jit, static_argnames=('self',))
    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams, # unused
    ) -> Tuple[
        Observation,
        EnvState,
        float,
        bool,
        dict,
    ]:
        return self.env.step(rng, state, action)


    @functools.partial(jax.jit, static_argnames=('self',))
    def reset_env_to_level(
        self,
        rng: chex.PRNGKey,
        level: Level,
        params: EnvParams, # unused
    ) -> Tuple[
        Observation,
        EnvState,
    ]:
        return self.env.reset_to_level(rng, level)
    

@struct.dataclass
class LogWrapperEnvState:
    state: EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


@struct.dataclass
class LogWrapper(UnderspecifiedEnv):
    """
    Wraps an underspecified env class and provides useful logging for episode
    lengths and accumulated return.
    """
    env: UnderspecifiedEnv


    @property
    def default_params(self) -> EnvParams:
        return self.env.default_params
    

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return self.env.action_space(params)


    @functools.partial(jax.jit, static_argnames=('self', 'params'))
    def reset_env_to_level(
        self,
        rng: chex.PRNGKey,
        level: Level,
        params: EnvParams,
    ) -> Tuple[
        Observation,
        EnvState,
    ]:
        obs, state = self.env.reset_to_level(rng, level, params)
        wrapped_state = LogWrapperEnvState(
            state=state,
            episode_returns=0.0,
            episode_lengths=0,
            returned_episode_returns=0.0,
            returned_episode_lengths=0,
            timestep=0,
        )
        return obs, wrapped_state


    @functools.partial(jax.jit, static_argnames=('self', 'params'))
    def step_env(
        self,
        rng: chex.PRNGKey,
        state: LogWrapperEnvState,
        action: int,
        params: EnvParams, # unused
    ) -> Tuple[
        Observation,
        LogWrapperEnvState,
        float,
        bool,
        dict,
    ]:
        # unwrapped step
        unwrapped_state = state.state
        obs, new_unwrapped_state, reward, done, info = self.env.step_env(
            rng=rng,
            state=unwrapped_state,
            action=action,
            params=params,
        )

        # update wrapped state
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        new_wrapped_state = LogWrapperEnvState(
            state=new_unwrapped_state,
            episode_returns=jax.lax.select(
                done,
                0.0,
                new_episode_return,
            ),
            episode_lengths=jax.lax.select(
                done,
                0,
                new_episode_length,
            ),
            returned_episode_returns=jax.lax.select(
                done,
                new_episode_return,
                state.returned_episode_returns,
            ),
            returned_episode_lengths=jax.lax.select(
                done,
                new_episode_length,
                state.returned_episode_lengths,
            ),
            timestep=state.timestep + 1,
        )

        # also put this stuff into the info dictionary
        info["returned_episode"] = done
        info["returned_episode_returns"] = new_wrapped_state.returned_episode_returns
        info["returned_episode_lengths"] = new_wrapped_state.returned_episode_lengths
        info["timestep"] = new_wrapped_state.timestep
        return obs, new_wrapped_state, reward, done, info


