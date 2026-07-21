"""
Reusable building blocks for score-prioritised level-replay buffers, shared by
PLR and ACCEL (and available to future methods of the same family).

A "buffer" here is a batch of `AnnotatedLevel` entries: a level plus the
bookkeeping a replay curriculum needs -- its most recent replayability score,
when it was first/last visited (for staleness), and the maximum return ever
observed on it (for max-Monte-Carlo regret estimators). Methods that want extra
per-level bookkeeping (e.g. ACCEL's mutation/replay counts) subclass
`AnnotatedLevel`; every function here operates on the buffer generically -- by
common field name and via `jax.tree.map` -- so the extra fields ride along
untouched.

The pieces that *differ* between methods (PLR's replay/generate coin flip vs
ACCEL's generate/replay/mutate state machine) stay in their own modules. This
library is only the shared buffer mechanics they are both expressed in terms of:

* `AnnotatedLevel`   -- the buffer-entry struct (common fields).
* `seed_annotations` -- initial annotations for a fresh buffer from a generator.
* `replay_probs`     -- P(replay each level) from score rank + staleness.
* `sample_replay`    -- draw a batch of distinct levels by replay priority.
* `batch_max_returns`-- max discounted return per level across its rollout.
* `score_batch`      -- replayability scores for a batch of rollouts.
* `merge_max_returns`-- running max of stored vs. freshly-observed returns.
* `record_replay`    -- write back scores/returns/visit-time for replayed levels.
* `insert_by_score`  -- top-k tournament: evict lowest-priority incumbents for
                        higher-scoring challengers.

None of these are individually `jax.jit`-ed: they are traced inside the already
jitted `CurriculumGenerator` methods that call them.
"""

import jax
import jax.numpy as jnp

from flax import struct
from chex import PRNGKey, Array

from jaxgmg.baselines import experience
from jaxgmg.baselines.experience import Rollout
from jaxgmg.baselines.autocurricula.prioritisation import plr_replay_probs
from jaxgmg.baselines.autocurricula.scores import plr_compute_scores

from jaxgmg.environments.base import Level, LevelGenerator, LevelSolver


@struct.dataclass
class AnnotatedLevel:
    """
    A buffer entry: a level plus the replay-curriculum bookkeeping for it.

    Methods needing extra per-level state subclass this (fields are appended);
    the functions in this module only ever read/write the fields below, so
    subclass fields are preserved untouched.
    """
    level: Level
    last_score: float
    last_visit_time: int
    # additional annotation for metrics
    first_visit_time: int
    # information for maxmc methods
    max_ever_return: float


def seed_annotations(
    level_generator: LevelGenerator,
    rng: PRNGKey,
    buffer_size: int,
    default_score: float = 0.0,
) -> dict:
    """
    The common initial annotations for a fresh buffer: `buffer_size` sampled
    levels tagged with a default score, zero visit times, and zero max-ever
    return. These placeholder entries are replayed until real scored levels
    displace them.

    Returned as a dict so a caller can splat it into its own `AnnotatedLevel`
    (sub)class alongside any method-specific fields:

        buffer = AnnotatedLevel(**seed_annotations(gen, rng, size), num_replays=...)
    """
    initial_levels = level_generator.vsample(rng=rng, num_levels=buffer_size)
    initial_times = jnp.zeros(buffer_size, dtype=int)
    return dict(
        level=initial_levels,
        last_score=jnp.ones(buffer_size) * default_score,
        last_visit_time=initial_times,
        first_visit_time=initial_times,
        max_ever_return=jnp.zeros(buffer_size),
    )


def replay_probs(
    buffer: AnnotatedLevel,
    temperature: float,
    staleness_coeff: float,
    current_time: int,
) -> Array:                             # float[buffer_size]
    """
    The replay-sampling distribution over the buffer: a mixture of score-rank
    prioritisation and staleness. See `prioritisation.plr_replay_probs`.
    """
    return plr_replay_probs(
        temperature=temperature,
        staleness_coeff=staleness_coeff,
        scores=buffer.last_score,
        last_visit_times=buffer.last_visit_time,
        current_time=current_time,
    )


def sample_replay(
    rng: PRNGKey,
    buffer: AnnotatedLevel,
    probs: Array,                       # float[buffer_size]
    num_levels: int,
) -> tuple[
    Array,                              # int[num_levels]  (the sampled ids)
    Level,                             # Level[num_levels] (the sampled levels)
]:
    """
    Draw `num_levels` *distinct* level ids from the buffer with probability
    `probs` (typically from `replay_probs`), and gather the levels at those ids.
    Sampling without replacement increases batch diversity and keeps buffer
    updates simple.
    """
    buffer_size, = buffer.last_score.shape
    assert num_levels <= buffer_size
    ids = jax.random.choice(
        key=rng,
        a=buffer_size,
        shape=(num_levels,),
        p=probs,
        replace=False,
    )
    levels = jax.tree.map(lambda x: x[ids], buffer.level)
    return ids, levels


def batch_max_returns(
    rollouts: Rollout,                  # Rollout[num_levels] w/ Transition[num_steps]
    discount_rate: float,
) -> Array:                             # float[num_levels]
    """
    The maximum discounted return achieved within each level's rollout -- the
    running-max estimate of the optimal return used by maxMC regret estimators.
    """
    return jax.vmap(
        experience.compute_maximum_return,
        in_axes=(0, 0, None),
    )(
        rollouts.transitions.reward,
        rollouts.transitions.done,
        discount_rate,
    )


def merge_max_returns(
    buffer: AnnotatedLevel,
    ids: Array,                         # int[num_levels]
    new_max_returns: Array,             # float[num_levels]
) -> Array:                             # float[num_levels]
    """
    Elementwise max of this rollout's max return with the buffer's stored
    max-ever return, for the levels at `ids`. (Used when a level is replayed:
    its max-ever return can only grow.)
    """
    return jnp.maximum(new_max_returns, buffer.max_ever_return[ids])


def oracle_returns(
    level_solver: LevelSolver | None,
    scoring_method: str,
    levels: Level,                      # Level[num_levels]
) -> Array:                             # float[num_levels]
    """
    The analytically-computed optimal return for each level, for ORACLE scoring
    methods (the `oracle-actor` regret estimator). Computed by solving each
    level with the provided, pre-configured `level_solver`.

    `scoring_method` is static, so the branch resolves at trace time: non-oracle
    methods don't need a solver and get a cheap zero placeholder (the scorer
    ignores it). This is where the oracle re-solving lives -- see the note in
    `scores.regret_oracle_actor` on eventually caching it per buffer entry.
    """
    if "oracle" in scoring_method.lower():
        assert level_solver is not None, \
            "oracle scoring methods require a configured level_solver"
        solutions = level_solver.vmap_solve(levels)
        return level_solver.vmap_level_value(solutions, levels)
    # placeholder for non-oracle methods (the scorer does not read it)
    num_levels = jax.tree.leaves(levels)[0].shape[0]
    return jnp.zeros(num_levels)


def score_batch(
    scoring_method: str,
    rollouts: Rollout,                  # Rollout[num_levels] w/ Transition[num_steps]
    advantages: Array,                  # float[num_levels, num_steps]
    discount_rate: float,
    levels: Level,                      # Level[num_levels]
    max_ever_returns: Array,            # float[num_levels]
    level_solver: LevelSolver | None = None,
) -> Array:                             # float[num_levels]
    """
    Replayability scores for a batch of rollouts under the named scoring method
    (usually a regret estimator). See `scores.plr_compute_scores`. Oracle
    methods additionally solve each level via `level_solver`; other methods
    ignore it.
    """
    return plr_compute_scores(
        scoring_method=scoring_method,
        rollouts=rollouts,
        max_ever_returns=max_ever_returns,
        advantages=advantages,
        discount_rate=discount_rate,
        oracle_returns=oracle_returns(level_solver, scoring_method, levels),
    )


def record_replay(
    buffer: AnnotatedLevel,
    ids: Array,                         # int[num_levels]
    scores: Array,                      # float[num_levels]
    max_ever_returns: Array,            # float[num_levels]
    visit_time: int,
) -> AnnotatedLevel:
    """
    Write back the results of replaying the levels at `ids`: their new scores,
    running-max returns, and the current time (marking them just-visited).
    Non-replayed entries and any subclass-specific fields are left untouched.
    """
    return buffer.replace(
        last_score=buffer.last_score.at[ids].set(scores),
        max_ever_return=buffer.max_ever_return.at[ids].set(max_ever_returns),
        last_visit_time=buffer.last_visit_time.at[ids].set(visit_time),
    )


def insert_by_score(
    buffer: AnnotatedLevel,
    challengers: AnnotatedLevel,        # AnnotatedLevel[num_levels]
    eviction_probs: Array,             # float[buffer_size]
    num_levels: int,
) -> AnnotatedLevel:
    """
    Top-k tournament insertion. Evict the `num_levels` incumbents with the
    lowest `eviction_probs` (i.e. lowest replay priority), pool those slots with
    the `num_levels` `challengers`, and keep whichever `num_levels` of the pool
    have the highest score. Generic over the annotation (sub)struct.

    `eviction_probs` is passed in rather than recomputed so the caller controls
    which priorities to evict against -- PLR reuses the distribution it sampled
    the batch with; ACCEL recomputes a fresh one.
    """
    # identify the num_levels lowest-priority incumbents (the eviction slots)
    _, worst_level_ids = jax.lax.top_k(-eviction_probs, k=num_levels)
    # pool those incumbents with the challengers
    candidate_buffer = jax.tree.map(
        lambda b, c: jnp.concatenate((b[worst_level_ids], c), axis=0),
        buffer,
        challengers,
    )
    # of these 2*num_levels entries, keep the num_levels highest scoring
    _, best_level_ids = jax.lax.top_k(candidate_buffer.last_score, k=num_levels)
    # write them into the evicted slots
    return jax.tree.map(
        lambda b, c: b.at[worst_level_ids].set(c[best_level_ids]),
        buffer,
        candidate_buffer,
    )
