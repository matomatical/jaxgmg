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
from jaxgmg.environments.base import LevelSolver, LevelSolution

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
        # TODO: optionally return a benchmark dict for evals?
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
    regret_estimator: str                   # "absGAE", "PVL", todo: "maxMC"
    level_solver: None | LevelSolver        # note: only used for metrics


    @struct.dataclass
    class State(CurriculumLevelGenerator.State):
        @struct.dataclass
        class AnnotatedLevel:
            level: Level
            last_score: float
            last_visit_time: int
            solution: None | LevelSolution  # note: only used for metrics
            first_visit_time: int           # note: only used for metrics
        buffer: AnnotatedLevel              # AnnotatedLevel[buffer_size]
        num_replay_batches: int
        prev_P_replay: Array                # float[buffer_size]
        prev_batch_was_replay: bool
        prev_batch_level_ids: Array         # int[num_levels]


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
        # if solution method is given, solve the levels for metrics
        if self.level_solver is not None:
            initial_level_solutions = self.level_solver.vmap_solve(
                levels=initial_levels,
            )
        else:
            initial_level_solutions = None
        # initialise the state with the above information in the level buffer
        # and additional information needed to maintain the state of the PLR
        # algorithm
        return self.State(
            buffer=self.State.AnnotatedLevel(
                level=initial_levels,
                solution=initial_level_solutions,
                last_score=initial_scores,
                last_visit_time=initial_time,
                first_visit_time=initial_time,
            ),
            num_replay_batches=0,
            prev_P_replay=jnp.zeros(self.buffer_size),
            prev_batch_was_replay=False,
            prev_batch_level_ids=jnp.arange(batch_size_hint),
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
        P_replay = self._compute_P_replay(state)
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
        return next_state, chosen_levels


    def _compute_P_replay(self, state: State) -> Array: # float[buffer_size]
        """
        Conditional on sampling from the replay buffer, compute the
        probability of sampling each level in the replay buffer?
        """
        # ordinal score-based prioritisation
        scores = state.buffer.last_score
        ranks = (
            jnp.empty(self.buffer_size)
                .at[jnp.argsort(scores, descending=True)]
                .set(jnp.arange(1, self.buffer_size+1))
        )
        tempered_hvals = jnp.pow(1 / ranks, 1 / self.temperature)
        
        # staleness-aware prioritisation
        staleness = 1 + state.num_replay_batches - state.buffer.last_visit_time

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
        scores = self._compute_scores(rollouts, advantages)
    
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

        
    def _compute_scores(
        self,
        rollouts: Rollout,  # Rollout[num_levels] with Transition[num_steps]
        advantages: Array,  # float[num_levels, num_steps]
    ) -> Array:             # float[num_levels]
        match self.regret_estimator.lower():
            case "absgae":
                return jnp.abs(advantages).mean(axis=1)
            case "pvl":
                return jnp.maximum(advantages, 0).mean(axis=1)
            case "maxmc":
                raise NotImplementedError # TODO
            case _:
                raise ValueError("Invalid return estimator.")


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
            buffer=state.buffer.replace(
                last_score=state.buffer.last_score
                    .at[state.prev_batch_level_ids]
                    .set(scores),
                last_visit_time=state.buffer.last_visit_time
                    .at[state.prev_batch_level_ids]
                    .set(state.num_replay_batches + 1),
            ),
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
        _, worst_level_ids = jax.lax.top_k(
            -state.prev_P_replay,
            k=num_levels,
        )

        # annotate the levels we're trying to add to the buffer
        time_now = jnp.full(num_levels, state.num_replay_batches, dtype=int)
        # (if solutions are enabled, solve the levels too)
        if self.level_solver is not None:
            level_solutions = self.level_solver.vmap_solve(levels)
        else:
            level_solutions = None
        challengers = self.State.AnnotatedLevel(
            level=levels,
            solution=level_solutions,
            last_score=scores,
            last_visit_time=time_now,
            first_visit_time=time_now,
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
    def compute_metrics(self, state: State) -> dict[str, Any]:
        # basic PLR diagnostic metrics
        metrics = {
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
        # TODO: enable arbitrary domain specific complexity measures such as
        # shortest path length, number of blocks, etc... solutions is not the
        # right abstraction?
        if self.level_solver is not None:
            optimal_values = self.level_solver.vmap_level_value(
                solns=state.buffer.solution,
                levels=state.buffer.level,
            )
            metrics['buffer_contents'] = {
                'avg_optimal_values': optimal_values.mean(),
                'wavg_optimal_values': state.prev_P_replay @ optimal_values,
                'optimal_values_hist': optimal_values,
            }
        # TODO: remove these metrics in favor of more sophisticated level
        # complexity metrics that also work for levels without this
        # particular variable...
        initial_mouse_pos_x = state.buffer.level.initial_mouse_pos[:, 0]
        initial_mouse_pos_y = state.buffer.level.initial_mouse_pos[:, 1]
        metrics['level_buffer_contents'] = {
            'avg_mouse_spawn_x': initial_mouse_pos_x.mean(),
            'avg_mouse_spawn_y': initial_mouse_pos_y.mean(),
            'wavg_mouse_spawn_x': state.prev_P_replay @ initial_mouse_pos_x,
            'wavg_mouse_spawn_y': state.prev_P_replay @ initial_mouse_pos_y,
            'mouse_spawn_x_hist': initial_mouse_pos_x,
            'mouse_spawn_y_hist': initial_mouse_pos_y,
        }
        return metrics


