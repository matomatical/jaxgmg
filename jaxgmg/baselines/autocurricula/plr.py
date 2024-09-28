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

from jaxgmg.baselines import experience
from jaxgmg.baselines.autocurricula.prioritisation import plr_replay_probs
from jaxgmg.baselines.autocurricula.scores import plr_compute_scores

from jaxgmg.environments.base import Level, LevelGenerator, LevelMetrics
from jaxgmg.baselines.experience import Rollout
from jaxgmg.baselines.autocurricula import base


@struct.dataclass
class AnnotatedLevel:
    level: Level
    last_score: float
    last_visit_time: int
    # information for maxmc methods
    max_ever_return: float
    max_ever_proxy_return: float    # proxy returns for maxmc
    # additional annotations for metrics
    first_visit_time: int


@struct.dataclass
class GeneratorState(base.GeneratorState):
    buffer: AnnotatedLevel              # AnnotatedLevel[buffer_size]
    num_replay_batches: int
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
    proxy_shaping: bool
    proxy_name: str | None
    proxy_shaping_coeff: float | None
    clipping: bool


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
        initial_levels = self.level_generator.vsample(
            rng=rng,
            num_levels=self.buffer_size,
        )
        initial_scores = jnp.ones(self.buffer_size) * default_score
        initial_time = jnp.zeros(self.buffer_size, dtype=int)
        # initialise the state with the above information in the level buffer
        # and additional information needed to maintain the state of the PLR
        # algorithm
        return GeneratorState(
            buffer=AnnotatedLevel(
                level=initial_levels,
                last_score=initial_scores,
                last_visit_time=initial_time,
                first_visit_time=initial_time,
                max_ever_return=jnp.zeros(self.buffer_size),
                max_ever_proxy_return=jnp.zeros(self.buffer_size),
            ),
            num_replay_batches=0,
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
        P_replay = plr_replay_probs(
            temperature=self.temperature,
            staleness_coeff=self.staleness_coeff,
            scores=state.buffer.last_score,
            last_visit_times=state.buffer.last_visit_time,
            current_time=state.num_replay_batches,
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
        proxy_advantages: Array | None, # float[num_levels, num_steps]
        step: int, # for eta schedule
    ) -> GeneratorState:
        # perform both a replay-type update and a new-type update
        replay_next_state = self._replay_update(
            state,
            rollouts=rollouts,
            advantages=advantages,
            proxy_advantages=proxy_advantages,
            levels=levels,
            step=step,
        )
        new_next_state = self._new_update(
            state,
            rollouts=rollouts,
            advantages=advantages,
            proxy_advantages=proxy_advantages,
            levels=levels,
            step=step,
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
        proxy_advantages: Array,
        levels: Level,  # Level[num_levels]
        step: int,
    ) -> GeneratorState:
        """
        Conditional on the previous batch being a replay batch, update the
        state.
        """
        # update the max returns
        new_max_returns = jax.vmap(
            experience.compute_maximum_return,
            in_axes=(0,0,None),
        )(
            rollouts.transitions.reward,
            rollouts.transitions.done,
            self.discount_rate,
        )
        old_max_returns = state.buffer.max_ever_return[
            state.prev_batch_level_ids
        ]
        max_max_returns = jnp.maximum(
            new_max_returns,
            old_max_returns,
        )
        # update the proxy max returns
        new_proxy_max_returns = jax.vmap(
            experience.compute_maximum_return,
            in_axes=(0,0,None),
        )(
            rollouts.transitions.info['proxy_rewards'][self.proxy_name],
            rollouts.transitions.done,
            self.discount_rate,
        )
        old_proxy_max_returns = state.buffer.max_ever_proxy_return[
            state.prev_batch_level_ids
        ]
        max_proxy_max_returns = jnp.maximum(
            new_proxy_max_returns,
            old_proxy_max_returns,
        )
        # compute the scores of these levels from the rollouts
        scores = plr_compute_scores(
            scoring_method=self.scoring_method,
            rollouts=rollouts,
            max_ever_returns=max_max_returns,
            advantages=advantages,
            discount_rate=self.discount_rate,
            proxy_shaping=self.proxy_shaping,
            proxy_name=self.proxy_name,
            proxy_shaping_coeff=self.proxy_shaping_coeff,
            max_ever_proxy_returns=max_proxy_max_returns,
            proxy_advantages=proxy_advantages,
            levels=levels,
            clipping=self.clipping,
            step=step,
        )
        # replace the scores of the replayed level ids with the new scores
        # and mark those levels as just visited
        return state.replace(
            buffer=state.buffer.replace(
                last_score=state.buffer.last_score
                    .at[state.prev_batch_level_ids]
                    .set(scores),
                last_visit_time=state.buffer.last_visit_time
                    .at[state.prev_batch_level_ids]
                    .set(state.num_replay_batches + 1),
                max_ever_return=state.buffer.max_ever_return
                    .at[state.prev_batch_level_ids]
                    .set(max_max_returns),
                max_ever_proxy_return=state.buffer.max_ever_proxy_return
                    .at[state.prev_batch_level_ids]
                    .set(max_proxy_max_returns),
            ),
            num_replay_batches=state.num_replay_batches + 1,
        )


    def _new_update(
        self,
        state: GeneratorState,
        rollouts: Rollout,
        advantages: Array,
        proxy_advantages: Array,
        levels: Level,  # Level[num_levels]
        step: int,
    ) -> GeneratorState:
        """
        Conditional on the previous batch being a new batch (not a replay
        batch), update the state.
        """
        # initialise the max returns
        max_returns = jax.vmap(
            experience.compute_maximum_return,
            in_axes=(0,0,None),
        )(
            rollouts.transitions.reward,
            rollouts.transitions.done,
            self.discount_rate,
        )
        # initialise the proxy max returns
        proxy_max_returns = jax.vmap(
            experience.compute_maximum_return,
            in_axes=(0,0,None),
        )(
            rollouts.transitions.info['proxy_rewards'][self.proxy_name],
            rollouts.transitions.done, 
            self.discount_rate,
        )

        # compute the initial scores from these rollouts
        scores = plr_compute_scores(
            scoring_method=self.scoring_method,
            rollouts=rollouts,
            max_ever_returns=max_returns,
            advantages=advantages,
            discount_rate=self.discount_rate,
            proxy_shaping=self.proxy_shaping,
            proxy_name=self.proxy_name,
            proxy_shaping_coeff=self.proxy_shaping_coeff,
            max_ever_proxy_returns=proxy_max_returns,
            proxy_advantages=proxy_advantages,
            levels=levels,
            clipping=self.clipping,
            step=step,
        )

        # on to updating the buffer...
        num_levels, = scores.shape

        # identify the num_levels levels with lowest replay potential
        _, worst_level_ids = jax.lax.top_k(
            -state.prev_P_replay,
            k=num_levels,
        )

        # annotate the levels we're trying to add to the buffer
        time_now = jnp.full(num_levels, state.num_replay_batches, dtype=int)
        challengers = AnnotatedLevel(
            level=levels,
            last_score=scores,
            last_visit_time=time_now,
            first_visit_time=time_now,
            max_ever_return=max_returns,
            max_ever_proxy_return=proxy_max_returns,
        )

        # concatenate the low-potential levels and the challenger levels
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


