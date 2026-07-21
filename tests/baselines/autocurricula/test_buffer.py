"""
Tests for the shared level-replay buffer library
``jaxgmg.baselines.autocurricula.buffer`` — the reusable operations that PLR and
ACCEL are both written in terms of (issue #7 in ``notes/02-cleanup-plan.md``).

These test the primitives directly and algorithm-agnostically. In particular
they check the "generic over the annotation struct" property: operations must
preserve subclass-specific fields (which ACCEL relies on for its mutation/replay
counts). We use a small ``Extended`` subclass with a ``tag`` field for that, and
integer arrays as stand-in "levels".
"""

import numpy as np

import jax
import jax.numpy as jnp
import pytest
from flax import struct

from jaxgmg.baselines import experience
from jaxgmg.baselines.autocurricula import buffer


GAMMA = 0.9
NS = 3            # steps per rollout


@struct.dataclass
class Extended(buffer.AnnotatedLevel):
    # a subclass field, to test that operations preserve extra annotations
    tag: jnp.ndarray


def make_buffer(levels, scores, last_visit=None, max_ever=None, tag=None, cls=buffer.AnnotatedLevel):
    n = len(levels)
    if last_visit is None:
        last_visit = np.zeros(n, dtype=int)
    if max_ever is None:
        max_ever = np.zeros(n)
    fields = dict(
        level=jnp.asarray(levels),
        last_score=jnp.asarray(scores, dtype=float),
        last_visit_time=jnp.asarray(last_visit, dtype=int),
        first_visit_time=jnp.asarray(last_visit, dtype=int),
        max_ever_return=jnp.asarray(max_ever, dtype=float),
    )
    if cls is Extended:
        fields["tag"] = jnp.asarray(
            np.zeros(n, dtype=int) if tag is None else tag
        )
    return cls(**fields)


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


# --- seed_annotations ----------------------------------------------------- #

class _StubGenerator:
    """Minimal level generator: 'levels' are just their indices."""
    def vsample(self, rng, num_levels):
        return jnp.arange(num_levels)


def test_seed_annotations_shapes_and_defaults():
    fields = buffer.seed_annotations(
        level_generator=_StubGenerator(),
        rng=jax.random.PRNGKey(0),
        buffer_size=5,
        default_score=0.3,
    )
    assert set(fields) == {
        "level", "last_score", "last_visit_time",
        "first_visit_time", "max_ever_return",
    }
    assert np.asarray(fields["last_score"]).tolist() == pytest.approx([0.3] * 5)
    assert np.asarray(fields["last_visit_time"]).tolist() == [0] * 5
    assert np.asarray(fields["first_visit_time"]).tolist() == [0] * 5
    assert np.asarray(fields["max_ever_return"]).tolist() == [0.0] * 5
    # splats straight into an AnnotatedLevel (or a subclass with extra fields)
    entry = Extended(**fields, tag=jnp.arange(5))
    assert np.asarray(entry.tag).tolist() == [0, 1, 2, 3, 4]


# --- sample_replay -------------------------------------------------------- #

def test_sample_replay_is_without_replacement():
    buf = make_buffer([10, 11, 12, 13], scores=[0, 0, 0, 0])
    probs = jnp.ones(4) / 4
    ids, levels = buffer.sample_replay(
        rng=jax.random.PRNGKey(1), buffer=buf, probs=probs, num_levels=4,
    )
    # a full-size draw without replacement is a permutation of all ids
    assert sorted(np.asarray(ids).tolist()) == [0, 1, 2, 3]
    # gathered levels match the buffer at those ids
    assert np.asarray(levels).tolist() == np.asarray(buf.level)[np.asarray(ids)].tolist()


def test_sample_replay_respects_probabilities():
    buf = make_buffer([10, 11, 12, 13], scores=[0, 0, 0, 0])
    probs = jnp.asarray([0.0, 0.0, 1.0, 0.0])   # only id 2 has mass
    ids, levels = buffer.sample_replay(
        rng=jax.random.PRNGKey(2), buffer=buf, probs=probs, num_levels=1,
    )
    assert np.asarray(ids).tolist() == [2]
    assert np.asarray(levels).tolist() == [12]


# --- batch_max_returns / merge_max_returns -------------------------------- #

def test_batch_max_returns_is_discounted_max():
    # reward 1 on the last of NS steps -> discounted return gamma^(NS-1).
    rollouts = make_rollout([[0, 0, 1], [1, 0, 0]])
    out = np.asarray(buffer.batch_max_returns(rollouts, GAMMA))
    assert out[0] == pytest.approx(GAMMA ** (NS - 1))
    assert out[1] == pytest.approx(GAMMA ** 0)


def test_merge_max_returns_takes_running_max():
    buf = make_buffer([0, 1, 2, 3], scores=[0] * 4, max_ever=[0.5, 0.9, 0.0, 0.0])
    merged = np.asarray(buffer.merge_max_returns(
        buffer=buf, ids=jnp.asarray([0, 1]), new_max_returns=jnp.asarray([0.7, 0.2]),
    ))
    assert merged[0] == pytest.approx(0.7)    # new 0.7 > stored 0.5
    assert merged[1] == pytest.approx(0.9)    # stored 0.9 > new 0.2


# --- record_replay -------------------------------------------------------- #

def test_record_replay_writes_ids_and_preserves_others_and_subclass_fields():
    buf = make_buffer(
        [0, 1, 2, 3], scores=[0, 0, 0, 0], last_visit=[0, 0, 0, 0],
        max_ever=[0.5, 0.9, 0.1, 0.2], tag=[7, 7, 7, 7], cls=Extended,
    )
    out = buffer.record_replay(
        buffer=buf, ids=jnp.asarray([1, 3]),
        scores=jnp.asarray([2.0, 3.0]),
        max_ever_returns=jnp.asarray([1.0, 0.95]),
        visit_time=6,
    )
    assert np.asarray(out.last_score).tolist() == [0.0, 2.0, 0.0, 3.0]
    assert np.asarray(out.max_ever_return).tolist() == pytest.approx([0.5, 1.0, 0.1, 0.95])
    assert np.asarray(out.last_visit_time).tolist() == [0, 6, 0, 6]
    # untouched fields (and the subclass tag) ride along unchanged
    assert np.asarray(out.first_visit_time).tolist() == [0, 0, 0, 0]
    assert np.asarray(out.tag).tolist() == [7, 7, 7, 7]


# --- insert_by_score (the tournament) ------------------------------------- #

def test_insert_by_score_strong_challenger_evicts_lowest_priority():
    buf = make_buffer([0, 1, 2, 3], scores=[10.0, 8.0, 1.0, 0.5])
    challengers = make_buffer([100, 101], scores=[5.0, 0.2])
    # lowest eviction priority = ids 2 and 3 -> the candidate slots.
    out = buffer.insert_by_score(
        buffer=buf, challengers=challengers,
        eviction_probs=jnp.asarray([0.4, 0.3, 0.2, 0.1]), num_levels=2,
    )
    levels = set(int(x) for x in np.asarray(out.level))
    # strong challenger 100 (score 5) beats candidates 2 (1.0) and 3 (0.5);
    # of the pool the top-2 scores are level 2 (1.0) and level 100 (5.0).
    assert levels == {0, 1, 2, 100}
    assert 101 not in levels and 3 not in levels


def test_insert_by_score_evicts_by_potential_not_by_score():
    # a low-score level with high eviction priority is protected.
    buf = make_buffer([0, 1, 2, 3], scores=[1.0, 2.0, 10.0, 0.5])
    challengers = make_buffer([100, 101], scores=[5.0, 0.0])
    # ids 0,1 have the lowest priority -> candidates; level 3 (lowest score)
    # has high priority so it is NOT a candidate.
    out = buffer.insert_by_score(
        buffer=buf, challengers=challengers,
        eviction_probs=jnp.asarray([0.1, 0.2, 0.4, 0.3]), num_levels=2,
    )
    levels = set(int(x) for x in np.asarray(out.level))
    assert 100 in levels                 # strong challenger enters
    assert 3 in levels                   # protected despite lowest score
    assert 0 not in levels               # evicted (lowest priority)


def test_insert_by_score_preserves_subclass_fields():
    buf = make_buffer([0, 1, 2, 3], scores=[10.0, 8.0, 1.0, 0.5],
                      tag=[0, 0, 0, 0], cls=Extended)
    challengers = make_buffer([100, 101], scores=[5.0, 0.2],
                              tag=[1, 1], cls=Extended)
    out = buffer.insert_by_score(
        buffer=buf, challengers=challengers,
        eviction_probs=jnp.asarray([0.4, 0.3, 0.2, 0.1]), num_levels=2,
    )
    levels = np.asarray(out.level)
    tags = np.asarray(out.tag)
    # the entering challenger (level 100) carries its tag=1; incumbents keep 0.
    assert tags[levels == 100].tolist() == [1]
    assert set(tags[levels != 100].tolist()) == {0}
