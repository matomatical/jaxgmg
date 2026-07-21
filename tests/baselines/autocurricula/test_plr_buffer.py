"""
Tests for the PLR replay-buffer update mechanics in
``jaxgmg.baselines.autocurricula.plr.CurriculumGenerator`` — the buffer
"tournament" that decides which levels stay in the curriculum
(``_new_update``), and the replay bookkeeping that tracks max-ever return and
staleness (``_replay_update``). Test-strategy item #7 / issue #7 in
``notes/02-cleanup-plan.md`` (this logic is duplicated in ``accel.py``).

We drive the real methods with lightweight stand-ins: integer arrays as
"levels" (the ``pvl`` / ``maxmc-actor`` scorers never read the ``Level`` payload)
and hand-built single-episode rollouts. ``pvl`` scoring lets us set each level's
score directly via its advantages.
"""

import numpy as np

import jax
import jax.numpy as jnp
import pytest

from jaxgmg.baselines import experience
from jaxgmg.baselines.autocurricula import plr


NS = 3            # steps per rollout
GAMMA = 0.9


def make_gen(scoring_method, buffer_size):
    return plr.CurriculumGenerator(
        level_generator=None, level_metrics=None, buffer_size=buffer_size,
        temperature=1.0, staleness_coeff=0.0, robust=False, prob_replay=0.5,
        scoring_method=scoring_method, discount_rate=GAMMA,
    )


def make_buffer(levels, scores, last_visit=None, max_ever=None):
    n = len(levels)
    if last_visit is None:
        last_visit = np.zeros(n, dtype=int)
    if max_ever is None:
        max_ever = np.zeros(n)
    return plr.AnnotatedLevel(
        level=jnp.asarray(levels),
        last_score=jnp.asarray(scores, dtype=float),
        last_visit_time=jnp.asarray(last_visit, dtype=int),
        first_visit_time=jnp.asarray(last_visit, dtype=int),
        max_ever_return=jnp.asarray(max_ever, dtype=float),
        max_ever_proxy_return=jnp.zeros(n),
    )


def make_rollout(rewards):
    # rewards: float[num_levels, NS]. Single episode each (done on last step).
    rewards = jnp.asarray(rewards, dtype=float)
    nl = rewards.shape[0]
    dones = jnp.zeros((nl, NS), dtype=bool).at[:, -1].set(True)
    transitions = experience.Transition(
        env_state=None, obs=None, net_state=None, prev_action=None,
        value=jnp.zeros((nl, NS)), proxy_value=jnp.zeros((nl, NS)), action=None,
        log_prob=None, reward=rewards, done=dones,
        info={'proxy_rewards': {'x': jnp.zeros((nl, NS))}},
    )
    return experience.Rollout(
        transitions=transitions,
        final_value=jnp.zeros(nl), final_proxy_value=jnp.zeros(nl),
    )


def const_adv(scores):
    # advantages whose pvl score (mean of positive part) equals each given value.
    return jnp.asarray([[s] * NS for s in scores], dtype=float)


# --- _new_update: the buffer tournament ----------------------------------- #

def test_strong_challenger_displaces_weak_buffer_level():
    gen = make_gen('pvl', buffer_size=4)
    state = plr.GeneratorState(
        buffer=make_buffer([0, 1, 2, 3], scores=[10.0, 8.0, 1.0, 0.5]),
        num_replay_batches=0, num_generate_batches=0,
        # lowest replay potential = ids 2 and 3 -> the eviction candidates.
        prev_P_replay=jnp.asarray([0.4, 0.3, 0.2, 0.1]),
        prev_batch_was_replay=False, prev_batch_level_ids=jnp.arange(2),
    )
    # challenger 100 scores 5.0 (beats buffer levels 2 and 3); 101 scores 0.2.
    ns = gen._new_update(
        state, rollouts=make_rollout([[0, 0, 0], [0, 0, 0]]),
        advantages=const_adv([5.0, 0.2]),
        levels=jnp.asarray([100, 101]), scoring_method_override=None,
    )
    buf_levels = set(int(x) for x in np.asarray(ns.buffer.level))
    # 100 (strong) enters, 3 (worst score among candidates) is evicted;
    # 101 (weak challenger) is rejected; untouched levels 0,1 remain.
    assert buf_levels == {0, 1, 2, 100}
    assert 101 not in buf_levels and 3 not in buf_levels
    assert ns.num_generate_batches == 1


def test_weak_challengers_are_all_rejected():
    gen = make_gen('pvl', buffer_size=4)
    state = plr.GeneratorState(
        buffer=make_buffer([0, 1, 2, 3], scores=[10.0, 8.0, 3.0, 2.0]),
        num_replay_batches=0, num_generate_batches=0,
        prev_P_replay=jnp.asarray([0.4, 0.3, 0.2, 0.1]),   # candidates: ids 2,3
        prev_batch_was_replay=False, prev_batch_level_ids=jnp.arange(2),
    )
    ns = gen._new_update(
        state, rollouts=make_rollout([[0, 0, 0], [0, 0, 0]]),
        advantages=const_adv([0.1, 0.0]),   # both weak
        levels=jnp.asarray([100, 101]), scoring_method_override=None,
    )
    # candidate levels 2 (3.0) and 3 (2.0) both beat the challengers, so the
    # buffer is unchanged.
    assert set(int(x) for x in np.asarray(ns.buffer.level)) == {0, 1, 2, 3}


def test_eviction_targets_lowest_potential_not_lowest_score():
    # A low-*score* level with high replay potential is protected; a
    # higher-score level with low potential is the one at risk.
    gen = make_gen('pvl', buffer_size=4)
    state = plr.GeneratorState(
        # level 3 has the lowest score (0.5) but HIGH potential (0.3);
        # levels 0,1 have the lowest potential (0.1, 0.2) -> eviction candidates.
        buffer=make_buffer([0, 1, 2, 3], scores=[1.0, 2.0, 10.0, 0.5]),
        num_replay_batches=0, num_generate_batches=0,
        prev_P_replay=jnp.asarray([0.1, 0.2, 0.4, 0.3]),
        prev_batch_was_replay=False, prev_batch_level_ids=jnp.arange(2),
    )
    ns = gen._new_update(
        state, rollouts=make_rollout([[0, 0, 0], [0, 0, 0]]),
        advantages=const_adv([5.0, 0.0]),
        levels=jnp.asarray([100, 101]), scoring_method_override=None,
    )
    buf_levels = set(int(x) for x in np.asarray(ns.buffer.level))
    # strong challenger 100 enters by evicting level 0 (score 1.0, lowest
    # potential); the lowest-*score* level 3 is untouched because its potential
    # was not among the lowest.
    assert 100 in buf_levels
    assert 3 in buf_levels                    # protected despite lowest score
    assert 0 not in buf_levels                # evicted (lowest potential)


# --- _replay_update: max-ever return + staleness bookkeeping -------------- #

def test_replay_update_tracks_max_ever_return_monotonically():
    gen = make_gen('maxmc-actor', buffer_size=4)
    state = plr.GeneratorState(
        buffer=make_buffer(
            [0, 1, 2, 3], scores=[0.0, 0.0, 0.0, 0.0],
            max_ever=[0.5, 0.9, 0.0, 0.0],
        ),
        num_replay_batches=0, num_generate_batches=0,
        prev_P_replay=jnp.asarray([0.25, 0.25, 0.25, 0.25]),
        prev_batch_was_replay=True,
        prev_batch_level_ids=jnp.asarray([0, 1]),   # slots 0,1 were replayed
    )
    # each replayed rollout collects reward 1 on the last step -> return gamma^2.
    this_return = GAMMA ** (NS - 1)
    ns = gen._replay_update(
        state, rollouts=make_rollout([[0, 0, 1], [0, 0, 1]]),
        advantages=const_adv([0.0, 0.0]),
        levels=jnp.asarray([0, 1]), scoring_method_override=None,
    )
    max_ever = np.asarray(ns.buffer.max_ever_return)
    # slot 0: old 0.5 < gamma^2 (~0.81) -> updates up; slot 1: old 0.9 wins.
    assert max_ever[0] == pytest.approx(this_return)
    assert max_ever[1] == pytest.approx(0.9)
    # never decreases, and dominates this rollout's return for replayed slots
    # (loose tol: buffer arrays are float32).
    assert max_ever[0] >= this_return - 1e-5
    assert max_ever[1] >= this_return - 1e-5
    # untouched slots stay at their initial value
    assert max_ever[2] == pytest.approx(0.0) and max_ever[3] == pytest.approx(0.0)


def test_replay_update_marks_visited_and_advances_clock():
    gen = make_gen('maxmc-actor', buffer_size=4)
    state = plr.GeneratorState(
        buffer=make_buffer([0, 1, 2, 3], scores=[0.0, 0.0, 0.0, 0.0],
                           last_visit=[0, 0, 0, 0]),
        num_replay_batches=5, num_generate_batches=0,
        prev_P_replay=jnp.asarray([0.25, 0.25, 0.25, 0.25]),
        prev_batch_was_replay=True,
        prev_batch_level_ids=jnp.asarray([1, 3]),   # replayed slots 1 and 3
    )
    ns = gen._replay_update(
        state, rollouts=make_rollout([[0, 0, 1], [0, 0, 1]]),
        advantages=const_adv([0.0, 0.0]),
        levels=jnp.asarray([1, 3]), scoring_method_override=None,
    )
    visit = np.asarray(ns.buffer.last_visit_time)
    # replayed slots stamped with num_replay_batches + 1 = 6; others unchanged.
    assert visit[1] == 6 and visit[3] == 6
    assert visit[0] == 0 and visit[2] == 0
    assert int(ns.num_replay_batches) == 6
