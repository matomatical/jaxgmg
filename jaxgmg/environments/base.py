import functools
from typing import Tuple, Optional, Dict
from flax import struct
import chex

import jax
import jax.numpy as jnp

from jaxgmg.graphics.sprites import LevelOfDetail, spritesheet


@struct.dataclass
class Level:
    """
    Represent a particular environment layout.
    In this case all fields come from subclass.
    """


@struct.dataclass
class EnvState:
    """
    Struct for storing dynamic environment state.

    * `level` (Level)
            A level as defined by the particular environment.
            TODO: maybe this should be its own thing not here?
    * `steps` (int)
            Number of steps since initialisation.
    * `done` (bool)
            True after the level has ended, either based on the specific
            environment's termination condition or a timeout.

    Subclass will add additional fields, and is responsible for updating
    them, but doesn't have to worry about updating these ones.
    """
    level: Level
    steps: int
    done: bool


@struct.dataclass
class Env:
    """
    Stores miscellaneous environment configuration.

    Parameters:

    * max_steps_in_episode: int (default 128)
            Declare an episode terminal after this many steps, regardless of
            whether the episode has been completed.
    * penalize_time : bool (default True)
            If True, all rewards decrease linearly by up to 90% over the
            maximum episode duration.
    * automatically_reset : bool (default True)
            If True, the step method automatically resets the state as the
            level is completed.
    * observation_lod: LevelOfDetail (OK to use raw int 0, 1, 3, 4, or 8)
            The level of detail for observations:
            * If LevelOfDetail.BOOLEAN or int 0, observations come as a
              height by width by num_channels Boolean array with
              environment-specific channels.
            * If LevelOfDetail.RGB_1PX or int 1, observations come as a
              height by width by 3 float RGB array with each sprite
              represented by a single pixel.
            * If LevelOfDetail.RGB_3PX or int 3, observations come as a
              3height by 3width by 3 float RGB array with each sprite
              represented by a 3x3 square of pixels.
            * Similarly for LevelOfDetail.RGB_4PX or int 4.
            * Similarly for LevelOfDetail.RGB_8PX or int 8.
    
    Properties:

    * num_actions : int
            Set by subclass. Number of valid actions in this environment.

    Methods:

    * env.reset_to_level(rng, level) -> (obs, start_state)
    * env.step(rng, state, action) -> (obs, new_state, reward, done, info)
    * env.get_obs(state) -> obs

    Instructions for sublassing: Implement the following methods:

    * _reset(rng, level) -> start_state
    * _step(rng, state, action) -> (new_state, reward, done, info)
    * _get_obs_bool(state) -> obs_bool
    * _get_obs_rgb(state) -> obs_rgb
    """

    max_steps_in_episode: int = 128
    penalize_time: bool = True
    automatically_reset: bool = True
    observation_lod: LevelOfDetail = LevelOfDetail.BOOLEAN


    @property
    def num_actions(self) -> int:
        raise NotImplementedError

    
    @functools.partial(jax.jit, static_argnames=('self',))
    def reset_to_level(
        self,
        rng: chex.PRNGKey,
        level: Level,
    ) -> Tuple[
        chex.Array,
        EnvState,
    ]:
        start_state = self._reset(rng, level)
        obs = self.get_obs(start_state)
        return (obs, start_state)
    

    def _reset(
        self,
        rng: chex.PRNGKey,
        level: Level,
    ) -> EnvState:
        raise NotImplementedError


    @functools.partial(jax.jit, static_argnames=('self',))
    def vreset_to_level(
        self,
        rng: chex.PRNGKey,
        levels: Level,      # * n
    ) -> Tuple[
        chex.Array,         # * n
        EnvState,           # * n
    ]:
        vmapped_reset_to_level = jax.vmap(
            self.reset_to_level,
            in_axes=(0, 0),
            out_axes=(0, 0),
        )
        num_levels = jax.tree.leaves(levels)[0].shape[0]
        return vmapped_reset_to_level(
            jax.random.split(rng, num_levels),
            levels,
        )


    @functools.partial(jax.jit, static_argnames=('self',))
    def step(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: int,
    ) -> Tuple[
        chex.Array,
        EnvState,
        float,
        bool,
        dict,
    ]:
        # defer to subclass to implement actual step
        rng_step, rng = jax.random.split(rng)
        new_state, reward, done, info = self._step(
            rng_step,
            state,
            action,
        )
        
        # track time
        steps = state.steps + 1
        timeout = steps >= self.max_steps_in_episode
        done_or_timeout = jnp.logical_or(done, timeout)
        new_state = new_state.replace(
            steps=steps,
            done=done_or_timeout,
        )

        # optional time penalty to reward
        reward = jax.lax.select(
            self.penalize_time,
            reward * (1.0 - .9 * state.steps / self.max_steps_in_episode),
            reward,
        )

        # (potentially) automatically reset the environment
        rng_reset, rng = jax.random.split(rng)
        reset_state = self._reset(
            rng_reset,
            new_state.level,
        ).replace(
            level=new_state.level,
        )
        new_state = jax.lax.cond( # because pytrees...
            self.automatically_reset & done_or_timeout,
            lambda: reset_state,
            lambda: new_state,
        )

        # render observation
        obs = self.get_obs(new_state)

        return (
            obs,
            new_state,
            reward,
            done_or_timeout,
            info,
        )


    def _step(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: int,
    ) -> Tuple[
        EnvState,
        float,
        bool,
        dict,
    ]:
        raise NotImplementedError

    
    @functools.partial(jax.jit, static_argnames=('self',))
    def vstep(
        self,
        rng: chex.PRNGKey,
        states: EnvState,    # EnvState[n]
        actions: chex.Array, # int[n]
    ) -> Tuple[
        chex.Array,
        EnvState,
        float,
        bool,
        dict,
    ]:
        vmapped_step = jax.vmap(
            self.step,
            in_axes=(0, 0, 0),
            out_axes=(0, 0, 0, 0, 0),
        )
        n, = actions.shape
        return vmapped_step(
            jax.random.split(rng, n),
            states,
            actions,
        )


    @functools.partial(jax.jit, static_argnames=('self', 'force_lod'))
    def get_obs(
        self,
        state: EnvState,
        force_lod: Optional[LevelOfDetail] = None,
    ):
        # override LevelOfDetail
        if force_lod is None:
            lod = self.observation_lod
        else:
            lod = force_lod
        # dispatch to the appropriate renderer
        if lod == LevelOfDetail.BOOLEAN:
            return self._get_obs_bool(state)
        else:
            return self._get_obs_rgb(state, spritesheet(lod))


    def _get_obs_bool(self, state: EnvState) -> chex.Array:
        raise NotImplementedError
    

    def _get_obs_rgb(
        self,
        state: EnvState,
        spritesheet: Dict[str, chex.Array],
    ) -> chex.Array:
        raise NotImplementedError
    
    
    @functools.partial(jax.jit, static_argnames=('self',))
    def optimal_value(
        self,
        levels: Level,
        discount_rate: float,
    ) -> float:
        """
        Compute the optimal return from (the initial state of) a level for a
        given discount rate. Respects time penalties to reward and max
        episode length.

        Parameters:

        * levels : Level
                The level to compute the optimal value for.
        * discount_rate : float
                The discount rate to apply in the formula for computing
                return.
        * The output also depends on the environment's reward function, which
          depends on `self.penalize_time` and `self.max_steps_in_episode`.
        """
        raise NotImplementedError


    @functools.partial(jax.jit, static_argnames=('self',))
    def voptimal_value(
        self,
        levels: Level,          # Level[n]
        discount_rate: float,
    ) -> chex.Array:            # float[n]
        """
        Compute the optimal return from a set of levels for a given discount
        rate. Respects time penalties to reward and max episode length.

        Parameters:

        * levels : Level[n]
                The levels to compute the optimal value for.
        * discount_rate : float
                The discount rate to apply in the formula for computing
                return.
        * The output also depends on the environment's reward function, which
          depends on `self.penalize_time` and `self.max_steps_in_episode`.
        """
        vectorised_optimal_value = jax.vmap(
            self.optimal_value,
            in_axes=(0, None),
            out_axes=0,
        )
        return vectorised_optimal_value(levels, discount_rate)



@struct.dataclass
class LevelGenerator:
    """
    Base class for level generator. Given some maze configuration parameters
    and cheese location parameter, provides a `sample` method that generates
    a random level.
    """


    def sample(
        self,
        rng: chex.PRNGKey,
    ) -> Level:
        """
        Randomly generate a `Level` specification given the parameters
        provided in the constructor of this generator object.

        Subclass should implement this method.
        """
        raise NotImplementedError


    @functools.partial(jax.jit, static_argnames=('self', 'num_levels'))
    def vsample(
        self,
        rng: chex.PRNGKey,
        num_levels: int,
    ) -> Level: # Level[num_levels]
        """
        Randomly generate a pytree of `num_levels` levels.
        """
        vectorised_sample = jax.vmap(
            self.sample,
            in_axes=(0,),
            out_axes=0,
        )
        return vectorised_sample(jax.random.split(rng, num_levels))


@struct.dataclass
class MixtureLevelGenerator:
    level_generator1: LevelGenerator
    level_generator2: LevelGenerator
    prob_level1: float


    def sample(
        self,
        rng: chex.PRNGKey,
    ) -> Level:
        # which level generator's sample should we use?
        rngmix, rng = jax.random.split(rng)
        which = jax.random.bernoulli(rngmix, p=self.prob_level1)

        # generate a level from each generator
        rng1, rng2 = jax.random.split(rng)
        level1 = self.level_generator1.sample(rng1)
        level2 = self.level_generator2.sample(rng2)

        # choose the chosen level
        chosen_level = jax.tree.map(
            lambda leaf1, leaf2: jax.lax.select(which, leaf1, leaf2),
            level1,
            level2
        )

        return chosen_level


