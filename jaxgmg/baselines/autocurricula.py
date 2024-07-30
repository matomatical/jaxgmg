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
        levels: Level,      # Level[num_levels]
        rollouts: Rollout,  # Rollout[num_levels] with Transition[num_steps]
        advantages: Array,  # float[num_levels, num_steps]
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


# # # 
# PRIORITISED LEVEL REPLAY


@struct.dataclass
class PrioritisedLevelReplay(CurriculumLevelGenerator):
    level_generator: LevelGenerator
    buffer_size: int
    temperature: float
    staleness_coeff: float
    prob_replay: float
    
    
    @struct.dataclass
    class State(CurriculumLevelGenerator.State):
        level_buffer: Level                 # Level[buffer_size]
        last_scores: Array                  # float[buffer_size]
        last_visit_time: Array                 # int[buffer_size]
        first_visit_time: Array                # int[buffer_size]
        num_replay_batches: int
        prev_batch_was_replay: bool
        prev_batch_level_ids: Array         # int[num_levels]
    # TODO:
    # * consider wrapping first four into an inner TaggedLevel struct...
    # * add the prev P_replay distribution, and its components, to save
    #   computation during update and to facilitate logging these metrics


    @functools.partial(jax.jit, static_argnames=['self'])
    def init(self, rng: PRNGKey, default_score: float = 0.0):
        # seed the level buffer with random levels with some default score.
        # initially we replay these in lieu of levels we have actual scores
        # for, but over time we replace them with real replay levels.
        return self.State(
            level_buffer=self.level_generator.vsample(
                rng=rng,
                num_levels=self.buffer_size,
            ),
            last_scores=jnp.ones(self.buffer_size) * default_score,
            last_visit_time=jnp.zeros(self.buffer_size, dtype=int),
            first_visit_time=jnp.zeros(self.buffer_size, dtype=int),
            num_replay_batches=0,
            prev_batch_was_replay=False,
            prev_batch_level_ids=jnp.arange(0), # size changes to batch size
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
        # spawn a batch of completely new levels
        rng_new, rng = jax.random.split(rng)
        new_levels = self.level_generator.vsample(
            rng=rng_new,
            num_levels=num_levels,
        )
        
        # spawn a batch of replay levels
        rng_replay, rng = jax.random.split(rng)
        assert num_levels <= self.buffer_size,
        replay_level_ids = jax.random.choice(
            rng=rng_replay,
            a=self.buffer_size,
            shape=(num_levels,),
            p=self._P_replay(state),
            replace=False, # increase diversity, easier updating
        )
        replay_levels = jax.tree.map(
            lambda x: x[replay_level_ids],
            state.level_buffer,
        )

        # decide which batch to use by flipping a biased coin
        rng_coin, rng = jax.random.split(rng)
        replay_choice = jax.random.bernoulli(
            key=rng_coin,
            p=self.prob_replay,
        )
        # select those levels
        chosen_next_state, chosen_levels = jax.tree.map(
            lambda r, n: jnp.where(replay_choice, r, n),
            replay_levels,
            new_levels,
        )

        # record information required for update in the state
        next_state_new = state.replace(
            prev_state_was_replay=replay_choice,
            prev_batch_level_ids=replay_level_ids,
        )
        return chosen_next_state, chosen_levels


    def _P_replay(self, state: State) -> Array: # float[self.buffer_size]
        """
        Conditional on sampling from the replay buffer, compute the
        probability of sampling each level in the replay buffer?
        """
        # ordinal score-based prioritisation
        ranks = (
            jnp.empty(self.buffer_size)
                .at[jnp.argsort(state.last_scores, descending=True)]
                .set(jnp.arange(1, self.buffer_size+1))
        )
        tempered_hvals = jnp.pow(1 / ranks, 1 / self.temperature)
        
        # staleness-aware prioritisation
        staleness = 1 + state.num_replay_batches - state.last_visit_time

        # probability of replaying each level is a mixture of these
        P_replay = (
            (1-self.staleness_coeff) * tempered_hvals / tempered_hvals.sum()
            + self.staleness_coeff * staleness / staleness.sum()
        )
        return P_replay
    

    @functools.partial(jax.jit, static_argnames=['self'])
    def update(
        self,
        state: State,
        levels: Level,      # Level[num_levels]
        rollouts: Rollout,  # Rollout[num_levels] with Transition[num_steps]
        advantages: Array,  # float[num_levels, num_steps]
    ) -> State:
        # estimate scores of these levels from the rollouts
        batch_scores = self._scores(rollouts, advantages)
    
        # perform both a replay-type update and a new-type update
        replay_next_state = self._replay_update(state, scores)
        new_next_state = self._new_update(state, levels, scores)
        # and keep the result corresponding to the previous batch's type
        next_state = jax.tree.map(
            lambda r, n: jnp.where(state.prev_batch_was_replay, r, n),
            replay_next_state,
            new_next_state,
        )
        return next_state

        
    def _score(
        self,
        rollouts: Rollout,  # Rollout[num_levels] with Transition[num_steps]
        advantages: Array,  # float[num_levels, num_steps]
    ) -> Array:             # float[num_levels]
        # L1 ppo value loss or something
        scores = jnp.abs(advantages).mean(axis=1)
        # TODO: test other scoring functions like maxmc
        return scores


    def _replay_update(
        self,
        state: State,
        scores: Array,
    ) -> State:
        """
        Conditional on the previous batch being a replay batch, update the
        state.
        """
        # replace the scores of the replayed level ids with the new scores
        # and mark those levels as just visited
        return state.replace(
            last_scores=state.last_scores
                .at[state.prev_batch_level_ids]
                .set(scores),
            last_visit_time=state.last_visit_time
                .at[state.prev_batch_level_ids]
                .set(state.num_replay_batches + 1),
            num_replay_batches=state.num_replay_batches + 1,
        )


    def _new_update(
        self,
        state: State,
        levels: Level,  # Level[num_levels]
        scores: Array,  # float[num_levels]
    ) -> State:
        """
        Conditional on the previous batch being a new batch (not a replay
        batch), update the state.
        """
        num_levels, = scores.shape

        # identify the num_levels levels with lowest replay potential
        P_replay = self._P_replay(state)
        _, worst_level_ids = jax.lax.top_k(-P_replay, k=num_levels)

        # concatenate the low-potential levels + their last score and timing
        # data with the new levels + their new scores and the time now
        time_now = jnp.full(num_levels, state.num_replay_batches, dtype=int)
        (
            candidate_levels,
            candidate_scores,
            candidate_last_visit_time,
            candidate_first_visit_time,
        ) = jax.tree.map(
            lambda l, r: jnp.concatenate((l, r), axis=0),
            (
                state.level_buffer[worst_level_ids],
                state.last_scores[worst_level_ids],
                state.last_visit_time[worst_level_ids],
                state.first_visit_time[worst_level_ids],
            ),
            (levels, scores, time_now, time_now),
        )

        # of these 2*num_levels levels, which num_levels have highest scores?
        _, best_level_ids = jax.lax.top_k(candidate_scores, k=num_levels)
        
        # use those levels to replace the lowest-potential levels
        return state.replace
            level_buffer=state.level_buffer
                .at[worst_level_ids]
                .set(candidate_levels[best_level_ids]),
            last_scores=state.last_scores
                .at[worst_level_ids]
                .set(candidate_scores[best_level_ids]),
            last_visit_time=state.last_visit_time
                .at[worst_level_ids]
                .set(candidate_last_visit_time[best_level_ids]),
            first_visit_time=state.first_visit_time
                .at[worst_level_ids]
                .set(candidate_first_visit_time[best_level_ids]),
        )


    @functools.partial(jax.jit, static_argnames=['self'])
    def compute_metrics(self, state: State) -> dict[str, Any]:
        # TODO: more sophisticated level complexity metrics that also work
        # for levels without this particular variable.
        initial_mouse_pos_x = state.level_buffer.initial_mouse_pos[:, 0]
        initial_mouse_pos_y = state.level_buffer.initial_mouse_pos[:, 1]
        return {
            'scoring': {
                'avg_scores': state.last_score.mean(),
                'scores_hist': state.last_score,
            },
            'visit_patterns': {
                'num_replay_batches': state.num_replay_batches,
                'avg_last_visit_time': state.last_visit_time.mean(),
                'avg_first_visit_time': state.first_visit_time.mean(),
                'last_visit_time_hist': state.last_visit_time,
                'first_visit_time_hist': state.first_visit_time,
                'prev_batch_level_ids_hist': state.prev_batch_level_ids,
            },
            'level_buffer_contents': {
                'avg_mouse_spawn_x': initial_mouse_pos_x.mean(),
                'avg_mouse_spawn_y': initial_mouse_pos_y.mean(),
                'mouse_spawn_x_hist': initial_mouse_pos_x,
                'mouse_spawn_y_hist': initial_mouse_pos_y,
            }
        }

