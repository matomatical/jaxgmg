"""
Abstract base classes for jaxgmg environment.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

import chex
from jaxtyping import PyTree

from jaxgmg.graphics import LevelOfDetail, load_spritesheet


# # # 
# Basic structs


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
class Observation:
    """
    Represent an observation from the environment. Subclass may add
    one or more fields.
    """


# # #
# Environment base class


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
    * obs_level_of_detail: LevelOfDetail (OK to use raw int 0, 1, 3, 4, or 8)
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
    * img_level_of_detail: LevelOfDetail (OK to use raw int 0, 1, 3, 4, or 8)
            The level of detail for rendered states. Same options as
            obs_level_of_detail.
    
    Properties:

    * num_actions : int
            Set by subclass. Number of valid actions in this environment.

    Methods:

    * obs_type(level) -> obs_type
    * env.reset_to_level(level) -> (obs, start_state)
    * env.step(rng, state, action) -> (obs, new_state, reward, done, info)
    * env.get_obs(state) -> obs
    * env.render_state(state) -> obs

    Instructions for sublassing: Implement the following methods:

    * @property num_actions(self) -> num_actions
    * obs_type(self, level) -> obs_type
    * _reset(self, level) -> start_state
    * _step(self, rng, state, action) -> (new_state, reward, done, info)
    * _render_obs_bool(self, state) -> obs_bool
    * _render_obs_rgb(self, state, spritesheet) -> obs_rgb
    * _render_state_bool(self, state) -> img_bool
    * _render_state_rgb(self, state, spritesheet) -> img_rgb
    """
    max_steps_in_episode: int = 128
    penalize_time: bool = True
    automatically_reset: bool = True
    obs_level_of_detail: LevelOfDetail = LevelOfDetail.BOOLEAN
    img_level_of_detail: LevelOfDetail = LevelOfDetail.RGB_1PX


    @property
    def num_actions(self) -> int:
        raise NotImplementedError


    def obs_type(self, level: Level) -> PyTree[jax.ShapeDtypeStruct]:
        raise NotImplementedError


    # methods supplied by base class

    
    def _reset(
        self,
        level: Level,
    ) -> EnvState:
        raise NotImplementedError


    def _step(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: int,
    ) -> tuple[
        EnvState,
        float,
        bool,
        dict,
    ]:
        raise NotImplementedError

    
    def _render_obs_bool(self, state: EnvState) -> Observation:
        raise NotImplementedError
    

    def _render_obs_rgb(
        self,
        state: EnvState,
        spritesheet: dict[str, chex.Array],
    ) -> Observation:
        raise NotImplementedError
    

    def _render_state_bool(self, state: EnvState) -> chex.Array:
        return NotImplementedError
    

    def _render_state_rgb(
        self,
        state: EnvState,
        spritesheet: dict[str, chex.Array],
    ) -> chex.Array:
        raise NotImplementedError
    

    # public API

    
    @functools.partial(jax.jit, static_argnames=('self',))
    def reset_to_level(
        self,
        level: Level,
    ) -> tuple[
        Observation,
        EnvState,
    ]:
        start_state = self._reset(level)
        obs = self.get_obs(start_state)
        return (obs, start_state)
    

    @functools.partial(jax.jit, static_argnames=('self',))
    def step(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: int,
    ) -> tuple[
        Observation,
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
        if self.penalize_time:
            penalty = (1.0 - .9 * state.steps / self.max_steps_in_episode)
            reward = reward * penalty
            if 'proxy_rewards' in info:
                info['proxy_rewards'] = {
                    k: r * penalty for k, r in info['proxy_rewards'].items()
                }

        # (potentially) automatically reset the environment
        rng_reset, rng = jax.random.split(rng)
        reset_state = self._reset(new_state.level)
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


    @functools.partial(jax.jit, static_argnames=('self',))
    def get_obs(
        self,
        state: EnvState,
    ) -> Observation:
        # dispatch to the appropriate renderer
        if self.obs_level_of_detail == LevelOfDetail.BOOLEAN:
            return self._render_obs_bool(state)
        else:
            spritesheet = load_spritesheet(self.obs_level_of_detail)
            return self._render_obs_rgb(state, spritesheet)
    

    @functools.partial(jax.jit, static_argnames=('self',))
    def render_state(
        self,
        state: EnvState,
    ) -> chex.Array: # float[h, w, rgb] or bool[h, w, c]
        # dispatch to the appropriate renderer
        if self.img_level_of_detail == LevelOfDetail.BOOLEAN:
            return self._render_state_bool(state)
        else:
            spritesheet = load_spritesheet(self.img_level_of_detail)
            return self._render_state_rgb(state, spritesheet)


    @functools.partial(jax.jit, static_argnames=('self',))
    def render_level(
        self,
        level: Level,
    ) -> chex.Array: # float[h, w, rgb] or bool[h, w, c]
        state = self._reset(level)
        image = self.render_state(state)
        return image


    # pre-vectorised methods
        

    @functools.partial(jax.jit, static_argnames=('self',))
    def vreset_to_level(
        self,
        levels: Level,      # Level[n]
    ) -> tuple[
        Observation,        # Observation[n]
        EnvState,           # Level[n]
    ]:
        vmapped_reset_to_level = jax.vmap(
            self.reset_to_level,
            in_axes=(0,),
            out_axes=(0, 0),
        )
        return vmapped_reset_to_level(levels)


    @functools.partial(jax.jit, static_argnames=('self',))
    def vstep(
        self,
        rng: chex.PRNGKey,
        states: EnvState,       # EnvState[n]
        actions: chex.Array,    # int[n]
    ) -> tuple[
        Observation,            # Observation[n]
        EnvState,               # EnvState[n]
        chex.Array,             # float[n]
        chex.Array,             # bool[n]
        chex.ArrayTree,         # dict[n]
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


# # # 
# Level generator


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


# # # 
# Level generator combinators


@struct.dataclass
class MixtureLevelGenerator(LevelGenerator):
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


# # # 
# Level mutation


@struct.dataclass
class LevelMutator:
    

    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(
        self,
        rng: chex.PRNGKey,
        level: Level,
    ) -> Level:
        raise NotImplementedError


    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_levels(
        self,
        rng: chex.PRNGKey,
        levels: Level,      # Level[num_levels]
    ) -> Level:             # Level[num_levels]
        num_levels, *_ = jax.tree.leaves(levels)[0].shape
        vmapped_mutate_level = jax.vmap(self.mutate_level, in_axes=(0, 0))
        return vmapped_mutate_level(
            jax.random.split(rng, num_levels),
            levels,
        )


# # # 
# Level mutator combinators


@struct.dataclass
class MixtureLevelMutator(LevelMutator):
    mutators: tuple[LevelMutator, ...]


    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        rng_mutate, rng_choice = jax.random.split(rng)
        # perform each type of mutation
        rng_mutates = jax.random.split(rng_mutate, len(self.mutators))
        levels = [
            child.mutate_level(rng_, level)
            for child, rng_ in zip(self.mutators, rng_mutates)
        ]
        which = jax.random.choice(rng_choice, len(self.mutators))
        return jax.tree.map(
            lambda *xs: jax.lax.select_n(which, *xs),
            *levels,
        )


@struct.dataclass
class IteratedLevelMutator(LevelMutator):
    mutator: LevelMutator
    num_steps: int


    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(
        self,
        rng: chex.PRNGKey,
        level: Level,
    ) -> Level:
        level, _ = jax.lax.scan(
            lambda level, rng: (self.mutator.mutate_level(rng, level), None),
            level,
            jax.random.split(rng, self.num_steps),
        )
        return level


# # # 
# Level parsing


@struct.dataclass
class LevelParser:
    """
    Level parser for an environment. Given some parameters (e.g. determining
    expected level shape), provides a `parse` method that converts an ASCII
    depiction of a level into a Level struct. Also provides a `parse_batch`
    method that parses a list of level strings into a single vectorised Level
    PyTree.

    Subclasses must implement the `parse` method, this superclass implements
    `parse_batch` based on that.
    """
    

    def parse(self, level_str: str) -> Level:
        """
        Convert an ASCII string depiction of a level into a Level struct.
        Implemented in subclass.
        """
        raise NotImplementedError


    def parse_batch(self, level_strs: list[str]) -> Level: # Level[n]
        """
        Convert a list of ASCII string depiction of length `num_levels`
        into a vectorised `Level[num_levels]` PyTree. See `parse` method for
        the details of the string depiction.
        """
        # parse levels
        levels = [self.parse(level_str) for level_str in level_strs]
        # stack into a single Level object
        return jax.tree.map(lambda *xs: jax.numpy.stack(xs), *levels)


# # # 
# Level solving


@struct.dataclass
class LevelSolution:
    """
    Represent a solution to some level. All fields come from subclass.
    """


@struct.dataclass
class LevelSolver:
    env: Env
    discount_rate: float


    @functools.partial(jax.jit, static_argnames=('self',))
    def solve(self, level: Level) -> LevelSolution:
        raise NotImplementedError


    @functools.partial(jax.jit, static_argnames=('self',))
    def state_value(self, soln: LevelSolution, state: EnvState) -> float:
        raise NotImplementedError


    @functools.partial(jax.jit, static_argnames=('self',))
    def state_action(self, soln: LevelSolution, state: EnvState) -> int:
        raise NotImplementedError

    
    @functools.partial(jax.jit, static_argnames=('self',))
    def state_action_values(
        self,
        soln: LevelSolution,
        state: EnvState,
    ) -> chex.Array: # float[4]
        raise NotImplementedError

    
    @functools.partial(jax.jit, static_argnames=('self',))
    def level_value(self, soln: LevelSolution, level: Level) -> float:
        state = self.env._reset(level)
        return self.state_value(soln, state)


    # pre-vectorised methods
        

    @functools.partial(jax.jit, static_argnames=('self',))
    def vmap_solve(
        self,
        levels: Level, # Level[n]
    ) -> LevelSolution: # LevelSolution[n]
        vectorised_solve = jax.vmap(
            self.solve,
        )
        return vectorised_solve(levels)
    
    
    @functools.partial(jax.jit, static_argnames=('self',))
    def vmap_level_value(
        self,
        solns: LevelSolution,   # LevelSolution[n]
        levels: Level,          # Level[n]
    ) -> float:                 # float[n]
        vectorised_level_value = jax.vmap(
            self.level_value,
        )
        return vectorised_level_value(solns, levels)


# # # 
# Level complexity metrics


@struct.dataclass
class LevelMetrics:
    env: Env
    discount_rate: float


    @functools.partial(jax.jit, static_argnames=('self',))
    def compute_metrics(self, levels: Level) -> dict[str, Any]:
        raise NotImplementedError


