"""
Prioritised level replay stateful level generator (parallel and robust
version).

* (PLR) Maintains a buffer with the most replayable levels over a level
  generator.
* (Parallel) Returns level batches with 'replay' levels and also 'new' levels
  so that experience collection can happen with both at once.
* (Robust) The 'new' levels are appended to the batch of the requested size,
  so that only the 'replay' levels are used for policy updates (both are used
  for replay buffer updates).

TODO: find a better, less hacky API for robustness...
"""

raise NotImplementedError
"""
Until I find the right API for this, I am decommissioning this module. I have
moved on from this API in the baselines/train script. Here is some relevant
old code from the training loop:

        ... (get batch call that might return more levels)

        # NOTE: get_batch may return num_levels levels or a number of
        # additional levels (e.g. in the case of parallel robust PLR,
        # 2*num_levels levels are returned). The contract is that we
        # should do rollouts and update UED in all of them, but we should
        # only train in the first num_levels of them.
        # TODO: more fine-grained logging.

        ... collect experience, update UED method)
            # TODO: steps per second is now wrong, doesn't account for actual
            # levels generated and simulated... use size of rollouts
            # TODO: split up each kind of rollouts/metrics?
            # TODO: count the number of env steps total along with the number
            # of env steps used for training
        
        # isolate valid levels for ppo updating
        valid_rollouts, valid_advantages = jax.tree.map(
            lambda x: x[:num_parallel_envs],
            (rollouts, advantages),
        )
        ... (and then pass these to ppo update)
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
from jaxgmg.baselines.autocurricula.prioritisation import plr_replay_probs
from jaxgmg.baselines.autocurricula.scores import plr_compute_scores


@struct.dataclass
class AnnotatedLevel:
    level: Level
    last_score: float
    last_visit_time: int
    first_visit_time: int


@struct.dataclass
class GeneratorState(base.GeneratorState):
    buffer: AnnotatedLevel              # AnnotatedLevel[buffer_size]
    num_batches: int
    prev_batch_replay_level_ids: Array  # int[2*num_levels]


@struct.dataclass
class CurriculumGenerator(base.CurriculumGenerator):
    level_generator: LevelGenerator
    level_metrics: LevelMetrics | None
    buffer_size: int
    temperature: float
    staleness_coeff: float
    regret_estimator: str       # "absGAE", "PVL", todo: "maxMC"


    @functools.partial(jax.jit, static_argnames=['self', 'batch_size_hint'])
    def init(
        self,
        rng: PRNGKey,
        default_score: float = 0.0,
        batch_size_hint: int = 0,
    ):
        # seed the level buffer with random levels with some default score.
        # initially we replay these in lieu of levels we have actual scores
        # for, but over time we replace them with real replay levels.
        return GeneratorState(
            buffer=AnnotatedLevel(
                level=self.level_generator.vsample(
                    rng=rng,
                    num_levels=self.buffer_size,
                ),
                last_score=jnp.ones(self.buffer_size) * default_score,
                last_visit_time=jnp.zeros(self.buffer_size, dtype=int),
                first_visit_time=jnp.zeros(self.buffer_size, dtype=int),
            ),
            num_batches=0,
            prev_batch_replay_level_ids=jnp.arange(batch_size_hint),
        )


    @functools.partial(jax.jit, static_argnames=['self', 'num_levels'])
    def get_batch(
        self,
        state: GeneratorState,
        rng: PRNGKey,
        num_levels: int,
    ) -> tuple[
        GeneratorState,
        Level, # Level[2*num_levels]
    ]:
        # spawn a batch of completely new levels
        rng_new, rng = jax.random.split(rng)
        new_levels = self.level_generator.vsample(
            rng=rng_new,
            num_levels=num_levels,
        )
        
        # spawn a batch of replay levels
        rng_replay, rng = jax.random.split(rng)
        P_replay = plr_replay_probs(
            temperature=self.temperature,
            staleness_coeff=self.staleness_coeff,
            scores=state.buffer.last_score,
            last_visit_times=state.buffer.last_visit_time,
            current_time=state.num_batches,
        )
        assert num_levels <= self.buffer_size
        replay_level_ids = jax.random.choice(
            key=rng_replay,
            a=self.buffer_size,
            shape=(num_levels,),
            p=P_replay,
            replace=False, # increase diversity, easier updating
        )
        replay_levels = jax.tree.map(
            lambda x: x[replay_level_ids],
            state.buffer.level,
        )

        # combine them into a single array of levels
        combined_levels = jax.tree.map(
            lambda r, n: jnp.concatenate([r, n], axis=0),
            replay_levels,
            new_levels,
        )
        # record information required for update in the state
        next_state = state.replace(
            prev_batch_replay_level_ids=replay_level_ids,
        )
        return next_state, combined_levels
 

    @functools.partial(jax.jit, static_argnames=['self'])
    def update(
        self,
        state: GeneratorState,
        levels: Level,      # Level[2*num_levels]
        rollouts: Rollout,  # Rollout[2*num_levels] with Transition[num_steps]
        advantages: Array,  # float[2*num_levels, num_steps]
    ) -> GeneratorState:
        # estimate scores of all the levels from the rollouts
        scores = plr_compute_scores(
            regret_estimator=self.regret_estimator,
            rollouts=rollouts,
            advantages=advantages,
            level=levels,
        )
        num_levels = scores.shape[0] // 2

        # perform replay-type update with the replay half of the levels
        scores_replay = scores[:num_levels]
        state_replay_updated = state.replace(
            buffer=state.buffer.replace(
                last_score=state.buffer.last_score
                    .at[state.prev_batch_replay_level_ids]
                    .set(scores_replay),
                last_visit_time=state.buffer.last_visit_time
                    .at[state.prev_batch_replay_level_ids]
                    .set(state.num_batches + 1),
            ),
            num_batches=state.num_batches + 1,
        )

        # perform new-type update with the new half of the levels
        scores_new = scores[num_levels:]
        levels_new = jax.tree.map(lambda x: x[num_levels:], levels)
        state_replay_and_new_updated = self._new_update(
            state=state_replay_updated,
            levels=levels_new,
            scores=scores_new,
        )
        
        return state_replay_and_new_updated


    def _new_update(
        self,
        state: GeneratorState,
        levels: Level,  # Level[num_levels]
        scores: Array,  # float[num_levels]
    ) -> GeneratorState:
        num_levels, = scores.shape

        # for parallel PLR we need to recompute the replay probabilities
        # since the buffer has changed by the time we do this update
        P_replay = plr_replay_probs(
            temperature=self.temperature,
            staleness_coeff=self.staleness_coeff,
            scores=state.buffer.last_score,
            last_visit_times=state.buffer.last_visit_time,
            current_time=state.num_batches,
        )

        # identify the num_levels levels with lowest replay potential
        _, worst_level_ids = jax.lax.top_k(
            -P_replay,
            k=num_levels,
        )

        # extract these low potential levels and concatenate them with the
        # new levels we're considering adding to the buffer
        # (together with required score and timing data)
        time_now = jnp.full(num_levels, state.num_batches, dtype=int)
        challengers = AnnotatedLevel(
            level=levels,
            last_score=scores,
            last_visit_time=time_now,
            first_visit_time=time_now,
        )
        candidate_buffer = jax.tree.map(
            lambda b, c: jnp.concatenate((b[worst_level_ids], c), axis=0),
            state.buffer,
            challengers,
        )

        # of these 2*num_levels levels, which num_levels have highest scores?
        _, best_level_ids = jax.lax.top_k(
            candidate_buffer.last_score,
            k=num_levels,
        )
        
        # use those levels to replace the lowest-potential levels
        return state.replace(
            buffer=jax.tree.map(
                lambda b, c: b.at[worst_level_ids].set(c[best_level_ids]),
                state.buffer,
                candidate_buffer,
            ),
        )


    @functools.partial(jax.jit, static_argnames=['self'])
    def compute_metrics(self, state: GeneratorState) -> dict[str, Any]:
        if self.level_metrics is not None:
            P_replay = plr_replay_probs(
                temperature=self.temperature,
                staleness_coeff=self.staleness_coeff,
                scores=state.buffer.last_score,
                last_visit_times=state.buffer.last_visit_time,
                current_time=state.num_batches,
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
                'num_replay_batches': state.num_batches,
                'avg_last_visit_time': state.buffer.last_visit_time.mean(),
                'avg_first_visit_time': state.buffer.first_visit_time.mean(),
                'last_visit_time_hist': state.buffer.last_visit_time,
                'first_visit_time_hist': state.buffer.first_visit_time,
                'prev_batch_level_ids_hist': state.prev_batch_replay_level_ids,
            },
        }


