"""
Tests for ``jaxgmg.baselines.autocurricula.prioritisation.plr_replay_probs`` —
the rank + staleness mixture at the heart of Prioritised Level Replay.

Test-strategy item #7 in ``notes/02-cleanup-plan.md``. This is a pure function,
so we pin it with hand-computed golden distributions and the monotonicity
properties PLR relies on (higher score ⇒ more replay; staler ⇒ more replay).

Reference: given ordinal ranks (rank 1 = highest score),
    tempered_hvals = (1/rank) ** (1/temperature)
    staleness      = 1 + current_time - last_visit_time
    P = (1-c) * tempered/Σtempered  +  c * staleness/Σstaleness
"""

import numpy as np

import jax.numpy as jnp
import pytest

from jaxgmg.baselines.autocurricula import prioritisation as prio


def probs(scores, temperature=1.0, staleness_coeff=0.0, last_visit=None, current_time=0):
    scores = jnp.asarray(scores, dtype=float)
    if last_visit is None:
        last_visit = jnp.zeros_like(scores, dtype=int)
    return np.asarray(prio.plr_replay_probs(
        temperature=temperature,
        staleness_coeff=staleness_coeff,
        scores=scores,
        last_visit_times=jnp.asarray(last_visit, dtype=int),
        current_time=current_time,
    ))


# --- golden distributions -------------------------------------------------- #

def test_rank_based_golden_temperature_one():
    # scores [1,3,2,0] -> ranks [3,1,2,4] -> tempered [1/3,1,1/2,1/4]
    # normalised (sum = 25/12): [4/25, 12/25, 6/25, 3/25]
    p = probs([1.0, 3.0, 2.0, 0.0], temperature=1.0, staleness_coeff=0.0)
    np.testing.assert_allclose(p, [0.16, 0.48, 0.24, 0.12], rtol=1e-6)
    assert p.sum() == pytest.approx(1.0)
    # the highest score gets the largest probability
    assert np.argmax(p) == 1


def test_pure_staleness_golden():
    # staleness_coeff=1 -> P = staleness / sum(staleness).
    # last_visit [0,1,2,3], current 3 -> staleness [4,3,2,1] -> [.4,.3,.2,.1]
    p = probs([0, 0, 0, 0], staleness_coeff=1.0, last_visit=[0, 1, 2, 3], current_time=3)
    np.testing.assert_allclose(p, [0.4, 0.3, 0.2, 0.1], rtol=1e-6)
    # the least-recently-visited level (smallest last_visit) is most probable
    assert np.argmax(p) == 0


def test_staleness_offset_is_one_not_zero():
    # Characterizes the `1 + current - last_visit` offset (there's a TODO in the
    # source questioning it): a just-visited level (last_visit == current) still
    # has staleness 1, not 0, so it keeps nonzero staleness weight.
    p = probs([0, 0], staleness_coeff=1.0, last_visit=[5, 5], current_time=5)
    np.testing.assert_allclose(p, [0.5, 0.5], rtol=1e-6)   # both staleness 1


# --- normalisation --------------------------------------------------------- #

@pytest.mark.parametrize("c", [0.0, 0.1, 0.5, 1.0])
def test_probs_sum_to_one(c):
    rng = np.random.default_rng(0)
    for _ in range(10):
        n = 6
        scores = rng.normal(size=n)
        last_visit = rng.integers(0, 20, size=n)
        p = probs(scores, temperature=0.3, staleness_coeff=c,
                  last_visit=last_visit, current_time=20)
        assert p.sum() == pytest.approx(1.0)
        assert np.all(p >= 0)


# --- monotonicity properties PLR relies on -------------------------------- #

def test_higher_score_gets_more_replay_probability():
    # staleness_coeff=0: replay prob is strictly decreasing in rank.
    p = probs([0.0, 1.0, 2.0, 3.0, 4.0], temperature=1.0, staleness_coeff=0.0)
    # scores increasing -> probabilities should be increasing
    assert np.all(np.diff(p) > 0)


def test_staler_levels_get_more_replay_probability():
    # staleness_coeff=1: replay prob strictly decreasing in last_visit_time.
    p = probs([0, 0, 0, 0], staleness_coeff=1.0,
              last_visit=[10, 7, 3, 0], current_time=10)
    # last_visit decreasing (staler) -> probabilities increasing
    assert np.all(np.diff(p) > 0)


def test_lower_temperature_concentrates_on_top_rank():
    # As temperature falls, mass concentrates on the highest-score level.
    top = []
    for temp in (2.0, 1.0, 0.5, 0.2):
        p = probs([1.0, 2.0, 3.0], temperature=temp, staleness_coeff=0.0)
        top.append(p[np.argmax(p)])
    assert np.all(np.diff(top) > 0)          # monotonically sharper


def test_staleness_coeff_linearly_mixes_the_two_distributions():
    scores = [1.0, 3.0, 2.0, 0.0]
    last_visit = [0, 1, 2, 3]
    rank_only = probs(scores, staleness_coeff=0.0, last_visit=last_visit, current_time=3)
    stale_only = probs(scores, staleness_coeff=1.0, last_visit=last_visit, current_time=3)
    mixed = probs(scores, staleness_coeff=0.25, last_visit=last_visit, current_time=3)
    np.testing.assert_allclose(mixed, 0.75 * rank_only + 0.25 * stale_only, rtol=1e-6)


# --- tie-breaking (characterization) -------------------------------------- #

def test_equal_scores_broken_by_index_not_uniform():
    # Ranks come from argsort, so equal scores are NOT tied: the earlier index
    # gets the better rank (and thus more probability). Documented behaviour.
    p = probs([5.0, 5.0, 5.0], temperature=1.0, staleness_coeff=0.0)
    # ranks [1,2,3] -> tempered [1, 1/2, 1/3] -> normalised
    np.testing.assert_allclose(p, np.array([1, 1 / 2, 1 / 3]) / (1 + 1 / 2 + 1 / 3), rtol=1e-6)
    assert p[0] > p[1] > p[2]
