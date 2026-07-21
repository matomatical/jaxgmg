"""
Prioritised level replay stateful level generator. Maintains a buffer with
the most replayable levels over a level generator.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp

from flax import struct
from chex import PRNGKey, Array

from jaxgmg.environments.base import Level, LevelGenerator, LevelMetrics
from jaxgmg.baselines.experience import Rollout
from jaxgmg.baselines.autocurricula import base
from jaxgmg.baselines.autocurricula import buffer


# a PLR buffer entry is exactly the shared annotation struct
AnnotatedLevel = buffer.AnnotatedLevel


@struct.dataclass
class GeneratorState(base.GeneratorState):
    buffer: AnnotatedLevel              # AnnotatedLevel[buffer_size]
    num_replay_batches: int
    num_generate_batches: int
    prev_P_replay: Array                # float[buffer_size]
    prev_batch_was_replay: bool
    prev_batch_level_ids: Array         # int[num_levels]
    

@struct.dataclass
class CurriculumGenerator(base.CurriculumGenerator):
    level_generator: LevelGenerator
    level_metrics: LevelMetrics | None
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
        # initialise the state with the above information in the level buffer
        # and additional information needed to maintain the state of the PLR
        # algorithm
        return GeneratorState(
            buffer=AnnotatedLevel(**buffer.seed_annotations(
                level_generator=self.level_generator,
                rng=rng,
                buffer_size=self.buffer_size,
                default_score=default_score,
            )),
            num_replay_batches=0,
            num_generate_batches=0,
            prev_P_replay=jnp.zeros(self.buffer_size),
            prev_batch_was_replay=False,
            prev_batch_level_ids=jnp.arange(batch_size_hint),
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
        # spawn a batch of completely new levels
        rng_new, rng = jax.random.split(rng)
        new_levels = self.level_generator.vsample(
            rng=rng_new,
            num_levels=num_levels,
        )
        
        # spawn a batch of replay levels
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

        # decide which batch to use by flipping a biased coin
        rng_coin, rng = jax.random.split(rng)
        replay_choice = jax.random.bernoulli(
            key=rng_coin,
            p=self.prob_replay,
        )
        # select those levels
        chosen_levels = jax.tree.map(
            lambda r, n: jnp.where(replay_choice, r, n),
            replay_levels,
            new_levels,
        )

        # record information required for update in the state
        next_state = state.replace(
            prev_P_replay=P_replay,
            prev_batch_was_replay=replay_choice,
            prev_batch_level_ids=replay_level_ids,
        )
        return next_state, chosen_levels, replay_choice.astype(int)


    def batch_type_name(self, batch_type: int) -> str:
        match batch_type:
            case 0:
                return "generate"
            case 1:
                return "replay"
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
        # perform both a replay-type update and a new-type update
        replay_next_state = self._replay_update(
            state,
            rollouts=rollouts,
            advantages=advantages,
            levels=levels,
        )
        new_next_state = self._new_update(
            state,
            rollouts=rollouts,
            advantages=advantages,
            levels=levels,
        )
        # and keep the result corresponding to the previous batch's type
        next_state = jax.tree.map(
            lambda r, n: jnp.where(state.prev_batch_was_replay, r, n),
            replay_next_state,
            new_next_state,
        )
        return next_state

        
    def _replay_update(
        self,
        state: GeneratorState,
        rollouts: Rollout,
        advantages: Array,
        levels: Level,  # Level[num_levels]
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
            max_ever_returns=max_max_returns,
        )
        # replace the scores of the replayed level ids with the new scores
        # and mark those levels as just visited
        return state.replace(
            buffer=buffer.record_replay(
                buffer=state.buffer,
                ids=state.prev_batch_level_ids,
                scores=scores,
                max_ever_returns=max_max_returns,
                visit_time=state.num_replay_batches + 1,
            ),
            num_replay_batches=state.num_replay_batches + 1,
        )


    def _new_update(
        self,
        state: GeneratorState,
        rollouts: Rollout,
        advantages: Array,
        levels: Level,  # Level[num_levels]
    ) -> GeneratorState:
        """
        Conditional on the previous batch being a new batch (not a replay
        batch), update the state.
        """
        # initialise the max returns and compute the initial scores
        max_returns = buffer.batch_max_returns(rollouts, self.discount_rate)
        scores = buffer.score_batch(
            scoring_method=self.scoring_method,
            rollouts=rollouts,
            advantages=advantages,
            discount_rate=self.discount_rate,
            levels=levels,
            max_ever_returns=max_returns,
        )

        # annotate the levels we're trying to add to the buffer
        num_levels, = scores.shape
        time_now = jnp.full(num_levels, state.num_replay_batches, dtype=int)
        challengers = AnnotatedLevel(
            level=levels,
            last_score=scores,
            last_visit_time=time_now,
            first_visit_time=time_now,
            max_ever_return=max_returns,
        )

        # run the tournament: challengers try to displace the lowest-priority
        # incumbents (evicting against the distribution we sampled this batch
        # with, cached in prev_P_replay).
        return state.replace(
            buffer=buffer.insert_by_score(
                buffer=state.buffer,
                challengers=challengers,
                eviction_probs=state.prev_P_replay,
                num_levels=num_levels,
            ),
            num_generate_batches=state.num_generate_batches + 1,
        )


    @functools.partial(jax.jit, static_argnames=['self'])
    def compute_metrics(self, state: GeneratorState) -> dict[str, Any]:
        if self.level_metrics is not None:
            buffer_metrics = self.level_metrics.compute_metrics( 
                levels=state.buffer.level,
                weights=state.prev_P_replay,
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
                'num_replay_batches': state.num_replay_batches,
                'avg_last_visit_time': state.buffer.last_visit_time.mean(),
                'avg_first_visit_time': state.buffer.first_visit_time.mean(),
                'last_visit_time_hist': state.buffer.last_visit_time,
                'first_visit_time_hist': state.buffer.first_visit_time,
                'prev_batch_level_ids_hist': state.prev_batch_level_ids,
            },
        }


