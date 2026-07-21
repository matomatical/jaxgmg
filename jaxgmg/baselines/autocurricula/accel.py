"""
ACCEL level replay stateful level generator.

* Like PLR, maintains a buffer with the most replayable levels over a level
  generator.
* Unlike PLR, also sometimes samples perturbations of levels in the buffer
  to search locally for even more replayable levels.

ACCEL stands for Adversarially Compounding Complexity by Editing Levels and
the algorithm is from [1].

[1] Parker-Holder, Jiang et al., "2022, Evolving Curricula with Regret-Based
    Environment Design".
"""

import enum
import functools
from typing import Any

import jax
import jax.numpy as jnp

from flax import struct
from chex import PRNGKey, Array

from jaxgmg.environments.base import Level, LevelGenerator, LevelMutator
from jaxgmg.environments.base import LevelMetrics, LevelSolver
from jaxgmg.baselines.experience import Rollout
from jaxgmg.baselines.autocurricula import base
from jaxgmg.baselines.autocurricula import buffer


class BatchType(enum.IntEnum):
    GENERATE = 0
    REPLAY = 1
    MUTATE = 2


@struct.dataclass
class AnnotatedLevel(buffer.AnnotatedLevel):
    # ACCEL tracks how many times each buffered level has been mutated/replayed
    num_mutations: int
    num_replays: int


@struct.dataclass
class GeneratorState(base.GeneratorState):
    buffer: AnnotatedLevel              # AnnotatedLevel[buffer_size]
    # state machine and variables
    prev_batch_type: BatchType
    prev_batch_level_ids: Array         # int[num_levels]
    prev_batch_mutate_counts: Array     # int[num_levels]
    # step tracking 
    num_generate_batches: int
    num_replay_batches: int
    num_mutate_batches: int
    

@struct.dataclass
class CurriculumGenerator(base.CurriculumGenerator):
    level_generator: LevelGenerator
    level_metrics: LevelMetrics | None
    level_mutator: LevelMutator
    # replay buffer
    buffer_size: int
    temperature: float
    staleness_coeff: float
    # replay dynamics
    robust: bool
    prob_replay: float
    # scoring
    scoring_method: str
    discount_rate: float
    # solver for oracle-based scoring methods (unused otherwise)
    level_solver: LevelSolver | None = None


    @functools.partial(jax.jit, static_argnames=['self', 'batch_size_hint'])
    def init(
        self,
        rng: PRNGKey,
        default_score: float = 0.0,
        batch_size_hint: int = 0,
    ):
        # seed the level buffer with random levels with a default score.
        # initially we replay these in lieu of levels we have actual scores
        # for, but over time we replace them with real replay levels.
        initial_counts = jnp.zeros(self.buffer_size, dtype=int)
        # initialise the state with the above information in the level buffer
        # and additional information needed to maintain the state of the PLR
        # algorithm
        return GeneratorState(
            buffer=AnnotatedLevel(
                **buffer.seed_annotations(
                    level_generator=self.level_generator,
                    rng=rng,
                    buffer_size=self.buffer_size,
                    default_score=default_score,
                ),
                num_replays=initial_counts,
                num_mutations=initial_counts,
            ),
            prev_batch_type=BatchType.GENERATE, # white lie
            prev_batch_level_ids=jnp.arange(batch_size_hint),
            prev_batch_mutate_counts=jnp.zeros(batch_size_hint, dtype=int),
            num_generate_batches=0,
            num_replay_batches=0,
            num_mutate_batches=0,
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
        bool,
    ]:
        # generate a batch of new levels
        rng_generate, rng = jax.random.split(rng)
        generated_levels = self.level_generator.vsample(
            rng=rng_generate,
            num_levels=num_levels,
        )
        
        # sample a batch of replay levels
        rng_replay, rng = jax.random.split(rng)
        P_replay = buffer.replay_probs(
            buffer=state.buffer,
            temperature=self.temperature,
            staleness_coeff=self.staleness_coeff,
            current_time=state.num_replay_batches,
        )
        replay_level_ids, replay_levels = buffer.sample_replay(
            rng=rng_replay,
            buffer=state.buffer,
            probs=P_replay,
            num_levels=num_levels,
        )

        # mutate the previous(! not current?) batch of replay levels
        prev_replay_levels, prev_mutate_counts = jax.tree.map(
            lambda x: x[state.prev_batch_level_ids],
            (state.buffer.level, state.buffer.num_mutations),
        )
        rng_mutate, rng = jax.random.split(rng)
        mutated_levels = self.level_mutator.mutate_levels(
            rng=rng_mutate,
            levels=prev_replay_levels,
        )
        mutate_counts = prev_mutate_counts + 1

        # decide which batch to use by consulting the state machine:
        transition_probs = jnp.array([
            # generate -> generate (1-p) or replay (p)
            [1-self.prob_replay, self.prob_replay, 0],
            # replay -> mutate (always)
            [0, 0, 1],
            # mutate -> generate (1-p) or replay (p)
            [1-self.prob_replay, self.prob_replay, 0],
        ], dtype=float)
        rng_choice, rng = jax.random.split(rng)
        batch_type = jax.random.choice(
            key=rng_choice,
            a=3,
            p=transition_probs[state.prev_batch_type],
        )

        # select the appropriate batch to return
        chosen_levels = jax.tree.map(
            lambda g, r, m: jax.lax.select_n(batch_type, g, r, m),
            generated_levels,
            replay_levels,
            mutated_levels,
        )

        # record information required for update in the state
        next_state = state.replace(
            prev_batch_type=batch_type,
            prev_batch_level_ids=replay_level_ids,
            prev_batch_mutate_counts=mutate_counts,
        )

        return next_state, chosen_levels, batch_type

    
    def batch_type_name(self, batch_type: int) -> str:
        match batch_type:
            case 0:
                return "generate"
            case 1:
                return "replay"
            case 2:
                return "mutate"
            case _:
                raise ValueError(f"Invalid batch type {batch_type!r}")


    def should_train(self, batch_type: int) -> bool:
        if not self.robust:
            return True
        else:
            return (batch_type == 1)


    @functools.partial(jax.jit, static_argnames=['self'])
    def update(
        self,
        state: GeneratorState,
        levels: Level,                  # Level[num_levels]
        rollouts: Rollout,              # Rollout[num_levels] (num_steps)
        advantages: Array,              # float[num_levels, num_steps]
    ) -> GeneratorState:
        # perform all possible kinds of update
        generate_next_state = self._generate_update(
            state=state,
            rollouts=rollouts,
            advantages=advantages,
            levels=levels,
        )
        replay_next_state = self._replay_update(
            state=state,
            rollouts=rollouts,
            advantages=advantages,
            levels=levels,
        )
        mutate_next_state = self._mutate_update(
            state=state,
            rollouts=rollouts,
            advantages=advantages,
            levels=levels,
        )

        # keep the result corresponding to the previous batch's type
        next_state = jax.tree.map(
            lambda g, r, m: jax.lax.select_n(state.prev_batch_type, g, r, m),
            generate_next_state,
            replay_next_state,
            mutate_next_state,
        )
        return next_state

        
    def _generate_update(
        self,
        state: GeneratorState,
        rollouts: Rollout,
        advantages: Array,
        levels: Level, # Level[num_levels]
    ) -> GeneratorState:
        """
        Conditional on the previous batch being a generate batch, update the
        state.
        """
        state = state.replace(
            num_generate_batches=state.num_generate_batches + 1,
        )
        state = self._buffer_insert_update(
            state=state,
            rollouts=rollouts,
            advantages=advantages,
            levels=levels,
        )
        return state


    def _replay_update(
        self,
        state: GeneratorState,
        rollouts: Rollout,
        advantages: Array,
        levels: Level, # Level[num_levels]
    ) -> GeneratorState:
        """
        Conditional on the previous batch being a replay batch, update the
        state.
        """
        # update the max returns (running max over replays of these levels)
        new_max_returns = buffer.batch_max_returns(rollouts, self.discount_rate)
        max_max_returns = buffer.merge_max_returns(
            buffer=state.buffer,
            ids=state.prev_batch_level_ids,
            new_max_returns=new_max_returns,
        )
        # compute the scores of these levels from the rollouts
        scores = buffer.score_batch(
            scoring_method=self.scoring_method,
            rollouts=rollouts,
            advantages=advantages,
            discount_rate=self.discount_rate,
            levels=levels,
            level_solver=self.level_solver,
            max_ever_returns=max_max_returns,
        )

        # replace the scores of the replayed level ids with the new scores
        # and mark those levels as just visited
        state = state.replace(
            num_replay_batches=state.num_replay_batches + 1,
            buffer=buffer.record_replay(
                buffer=state.buffer,
                ids=state.prev_batch_level_ids,
                scores=scores,
                max_ever_returns=max_max_returns,
                visit_time=state.num_replay_batches + 1,
            ),
        )
        return state


    def _mutate_update(
        self,
        state: GeneratorState,
        rollouts: Rollout,
        advantages: Array,
        levels: Level, # Level[num_levels]
    ) -> GeneratorState:
        """
        Conditional on the previous batch being a mutate batch, update the
        state.
        """
        state = state.replace(
            num_mutate_batches=state.num_mutate_batches + 1,
        )
        state = self._buffer_insert_update(
            state=state,
            rollouts=rollouts,
            advantages=advantages,
            levels=levels,
        )
        return state


    def _buffer_insert_update(
        self,
        state: GeneratorState,
        rollouts: Rollout,      # Rollout[num_levels] w/ Transition[num_steps]
        advantages: Array,      # float[num_levels, num_steps]
        levels: Level,          # Level[num_levels]
    ) -> GeneratorState:
        # initialise the max returns and compute the initial scores
        max_returns = buffer.batch_max_returns(rollouts, self.discount_rate)
        scores = buffer.score_batch(
            scoring_method=self.scoring_method,
            rollouts=rollouts,
            advantages=advantages,
            discount_rate=self.discount_rate,
            levels=levels,
            level_solver=self.level_solver,
            max_ever_returns=max_returns,
        )

        # annotate the levels we're trying to add to the buffer
        num_levels, = scores.shape
        time_now = jnp.full(num_levels, state.num_replay_batches, dtype=int)
        count_zero = jnp.zeros(num_levels, dtype=int)
        mutate_counts = (
            # prepared values if this is a mutate batch, else zeros
            state.prev_batch_mutate_counts
            * (state.prev_batch_type == BatchType.MUTATE)
        )
        challengers = AnnotatedLevel(
            level=levels,
            last_score=scores,
            last_visit_time=time_now,
            max_ever_return=max_returns,
            first_visit_time=time_now,
            num_replays=count_zero,
            num_mutations=mutate_counts,
        )

        # run the tournament: challengers try to displace the lowest-priority
        # incumbents. Unlike PLR, ACCEL recomputes the replay distribution here
        # (rather than caching it from get_batch) to evict against.
        P_replay = buffer.replay_probs(
            buffer=state.buffer,
            temperature=self.temperature,
            staleness_coeff=self.staleness_coeff,
            current_time=state.num_replay_batches,
        )
        return state.replace(
            buffer=buffer.insert_by_score(
                buffer=state.buffer,
                challengers=challengers,
                eviction_probs=P_replay,
                num_levels=num_levels,
            ),
        )


    @functools.partial(jax.jit, static_argnames=['self'])
    def compute_metrics(self, state: GeneratorState) -> dict[str, Any]:
        if self.level_metrics is not None:
            P_replay = buffer.replay_probs(
                buffer=state.buffer,
                temperature=self.temperature,
                staleness_coeff=self.staleness_coeff,
                current_time=state.num_replay_batches,
            )
            buffer_metrics = self.level_metrics.compute_metrics(
                levels=state.buffer.level,
                weights=P_replay,
            )
        else:
            buffer_metrics = {}
        return {
            **buffer_metrics,
            'scoring': {
                'avg_scores': state.buffer.last_score.mean(),
                'scores_hist': state.buffer.last_score,
            },
            'visit_patterns': {
                'num_generate_batches': state.num_generate_batches,
                'num_replay_batches': state.num_replay_batches,
                'num_mutate_batches': state.num_mutate_batches,
                'avg_last_visit_time': state.buffer.last_visit_time.mean(),
                'avg_first_visit_time': state.buffer.first_visit_time.mean(),
                'last_visit_time_hist': state.buffer.last_visit_time,
                'first_visit_time_hist': state.buffer.first_visit_time,
                'prev_batch_level_ids_hist': state.prev_batch_level_ids,
                'mutate_count_avg': state.buffer.num_mutations.mean(),
                'mutate_count_hist': state.buffer.num_mutations,
            },
        }


