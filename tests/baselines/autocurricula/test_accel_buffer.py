"""
Tests for ACCEL's buffer update mechanics in
``jaxgmg.baselines.autocurricula.accel.CurriculumGenerator``. ACCEL reuses the
shared buffer library (tested in ``test_buffer.py``) but wires it into a 3-way
generate/replay/mutate state machine and adds mutation/replay bookkeeping —
this file covers that wiring (issue #7 in ``notes/02-cleanup-plan.md``: accel.py
was previously a near-untested near-duplicate of plr.py).

As in ``test_plr_buffer.py`` we drive the real methods with integer "levels"
(the ``pvl`` scorer ignores the level payload) and hand-built single-episode
rollouts, and use ``pvl`` scoring so each level's score is set via its
advantages. ``staleness_coeff=0`` makes the recomputed eviction priority a pure
function of score rank, so eviction targets the lowest-score levels.
"""

import numpy as np

import jax
import jax.numpy as jnp
import pytest

from jaxgmg.baselines import experience
from jaxgmg.baselines.autocurricula import accel
from jaxgmg.baselines.autocurricula.accel import BatchType


NS = 3            # steps per rollout
GAMMA = 0.9


def make_gen(scoring_method, buffer_size):
    return accel.CurriculumGenerator(
        level_generator=None, level_metrics=None, level_mutator=None,
        buffer_size=buffer_size, temperature=1.0, staleness_coeff=0.0,
        robust=False, prob_replay=0.5,
        scoring_method=scoring_method, discount_rate=GAMMA,
    )


def make_buffer(levels, scores, last_visit=None, max_ever=None, num_mutations=None):
    n = len(levels)
    if last_visit is None:
        last_visit = np.zeros(n, dtype=int)
    if max_ever is None:
        max_ever = np.zeros(n)
    if num_mutations is None:
        num_mutations = np.zeros(n, dtype=int)
    return accel.AnnotatedLevel(
        level=jnp.asarray(levels),
        last_score=jnp.asarray(scores, dtype=float),
        last_visit_time=jnp.asarray(last_visit, dtype=int),
        first_visit_time=jnp.asarray(last_visit, dtype=int),
        max_ever_return=jnp.asarray(max_ever, dtype=float),
        num_replays=jnp.zeros(n, dtype=int),
        num_mutations=jnp.asarray(num_mutations, dtype=int),
    )


def make_state(buffer, prev_batch_type, prev_batch_level_ids,
               prev_batch_mutate_counts=None,
               num_generate_batches=0, num_replay_batches=0, num_mutate_batches=0):
    prev_batch_level_ids = jnp.asarray(prev_batch_level_ids)
    if prev_batch_mutate_counts is None:
        prev_batch_mutate_counts = jnp.zeros_like(prev_batch_level_ids)
    return accel.GeneratorState(
        buffer=buffer,
        prev_batch_type=prev_batch_type,
        prev_batch_level_ids=prev_batch_level_ids,
        prev_batch_mutate_counts=jnp.asarray(prev_batch_mutate_counts),
        num_generate_batches=num_generate_batches,
        num_replay_batches=num_replay_batches,
        num_mutate_batches=num_mutate_batches,
    )


def make_rollout(rewards):
    rewards = jnp.asarray(rewards, dtype=float)
    nl = rewards.shape[0]
    dones = jnp.zeros((nl, NS), dtype=bool).at[:, -1].set(True)
    transitions = experience.Transition(
        env_state=None, obs=None, net_state=None, prev_action=None,
        value=jnp.zeros((nl, NS)), action=None,
        log_prob=None, reward=rewards, done=dones, info={},
    )
    return experience.Rollout(transitions=transitions, final_value=jnp.zeros(nl))


def const_adv(scores):
    return jnp.asarray([[s] * NS for s in scores], dtype=float)


# --- _replay_update: max-ever return + staleness bookkeeping -------------- #

def test_replay_update_tracks_max_ever_and_stamps_visit():
    gen = make_gen('maxmc-actor', buffer_size=4)
    state = make_state(
        make_buffer([0, 1, 2, 3], scores=[0.0] * 4, max_ever=[0.5, 0.9, 0.0, 0.0]),
        prev_batch_type=BatchType.REPLAY,
        prev_batch_level_ids=[0, 1],           # slots 0,1 were replayed
        num_replay_batches=5,
    )
    this_return = GAMMA ** (NS - 1)            # reward 1 on last step
    ns = gen._replay_update(
        state, rollouts=make_rollout([[0, 0, 1], [0, 0, 1]]),
        advantages=const_adv([0.0, 0.0]), levels=jnp.asarray([0, 1]),
    )
    max_ever = np.asarray(ns.buffer.max_ever_return)
    assert max_ever[0] == pytest.approx(this_return)   # 0.5 -> gamma^2
    assert max_ever[1] == pytest.approx(0.9)           # 0.9 wins
    assert max_ever[2] == pytest.approx(0.0) and max_ever[3] == pytest.approx(0.0)
    # replayed slots stamped num_replay_batches + 1 = 6; clock advances
    visit = np.asarray(ns.buffer.last_visit_time)
    assert visit[0] == 6 and visit[1] == 6
    assert visit[2] == 0 and visit[3] == 0
    assert int(ns.num_replay_batches) == 6


# --- _generate_update: the tournament (recomputed eviction priority) ------ #

def test_generate_update_strong_challenger_evicts_lowest_score():
    gen = make_gen('pvl', buffer_size=4)
    # staleness_coeff=0 -> eviction priority is pure score rank, so the two
    # lowest-score incumbents (ids 2 and 3) are the eviction candidates.
    state = make_state(
        make_buffer([0, 1, 2, 3], scores=[10.0, 8.0, 1.0, 0.5]),
        prev_batch_type=BatchType.GENERATE,
        prev_batch_level_ids=[0, 1],
        num_generate_batches=7,
    )
    ns = gen._generate_update(
        state, rollouts=make_rollout([[0, 0, 0], [0, 0, 0]]),
        advantages=const_adv([5.0, 0.2]),      # challenger scores 5.0, 0.2
        levels=jnp.asarray([100, 101]),
    )
    levels = np.asarray(ns.buffer.level)
    assert set(int(x) for x in levels) == {0, 1, 2, 100}
    assert int(ns.num_generate_batches) == 8
    # a generate-batch challenger enters with a zero mutation count
    assert np.asarray(ns.buffer.num_mutations)[levels == 100].tolist() == [0]


# --- _mutate_update: mutation-count propagation (accel-specific) ---------- #

def test_mutate_update_propagates_mutation_counts():
    gen = make_gen('pvl', buffer_size=2)
    # both challengers outscore the (empty-ish) buffer, so both enter; the
    # prepared per-level mutation counts [3, 4] must ride in with them.
    state = make_state(
        make_buffer([0, 1], scores=[0.0, 0.0]),
        prev_batch_type=BatchType.MUTATE,
        prev_batch_level_ids=[0, 1],
        prev_batch_mutate_counts=[3, 4],
        num_mutate_batches=2,
    )
    ns = gen._mutate_update(
        state, rollouts=make_rollout([[0, 0, 0], [0, 0, 0]]),
        advantages=const_adv([5.0, 6.0]), levels=jnp.asarray([100, 101]),
    )
    assert int(ns.num_mutate_batches) == 3
    # the entering challengers carry the propagated mutation counts {3, 4}
    assert set(int(x) for x in np.asarray(ns.buffer.num_mutations)) == {3, 4}


def test_buffer_insert_zeroes_mutation_counts_when_not_a_mutate_batch():
    # same prepared counts, but a GENERATE batch must NOT propagate them.
    gen = make_gen('pvl', buffer_size=2)
    state = make_state(
        make_buffer([0, 1], scores=[0.0, 0.0]),
        prev_batch_type=BatchType.GENERATE,
        prev_batch_level_ids=[0, 1],
        prev_batch_mutate_counts=[3, 4],
    )
    ns = gen._generate_update(
        state, rollouts=make_rollout([[0, 0, 0], [0, 0, 0]]),
        advantages=const_adv([5.0, 6.0]), levels=jnp.asarray([100, 101]),
    )
    assert set(int(x) for x in np.asarray(ns.buffer.num_mutations)) == {0}


# --- update(): the 3-way state-machine dispatch --------------------------- #

@pytest.mark.parametrize("batch_type,counter", [
    (BatchType.GENERATE, "num_generate_batches"),
    (BatchType.REPLAY, "num_replay_batches"),
    (BatchType.MUTATE, "num_mutate_batches"),
])
def test_update_dispatches_on_prev_batch_type(batch_type, counter):
    gen = make_gen('pvl', buffer_size=4)
    state = make_state(
        make_buffer([0, 1, 2, 3], scores=[10.0, 8.0, 1.0, 0.5]),
        prev_batch_type=batch_type,
        prev_batch_level_ids=[0, 1],
        num_generate_batches=10, num_replay_batches=20, num_mutate_batches=30,
    )
    ns = gen.update(
        state, levels=jnp.asarray([100, 101]),
        rollouts=make_rollout([[0, 0, 0], [0, 0, 0]]),
        advantages=const_adv([5.0, 0.2]),
    )
    # exactly the counter for the previous batch type advances by one; the
    # other two are left at their base values.
    expected = {"num_generate_batches": 10, "num_replay_batches": 20,
                "num_mutate_batches": 30}
    expected[counter] += 1
    for name, value in expected.items():
        assert int(getattr(ns, name)) == value
