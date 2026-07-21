"""
Characterization tests for the regret-score primitives in
``jaxgmg.baselines.autocurricula.scores`` — the science core of the
autocurriculum (PLR/ACCEL) baselines.

Test-strategy item #6 in ``notes/02-cleanup-plan.md``. These pin the *live*
regret definitions (``scores.py`` L296-600) so the coming refactor — deleting
the dead ``plr_compute_scores_old`` (L627-1243, no callers), extracting a shared
PLR buffer, and cleaning up ``regret_oracle_actor`` — cannot silently change the
numbers the paper's estimators produce.

Strategy: the two dispatchers (``plr_compute_scores`` / ``plr_compute_score``)
eat big vmapped rollout PyTrees, but every regret *definition* is a small pure
function over 1-D arrays. So we test the primitives directly against
hand-computed golden values, and cover the ``match`` dispatch with one light
smoke test (the dispatcher returns each primitive's value verbatim).

The oracle-latest estimator (``regret_oracle_actor``) is now pure arithmetic
(``oracle_return - realised_return``); the oracle optimal return is computed by
``buffer.oracle_returns`` from a configured ``LevelSolver`` (issue #11), and the
BUG-1 discount off-by-one is fixed (the solver reports ``gamma^(d-1)``), so an
optimal agent has exactly zero oracle regret.
"""

import math

import numpy as np

import jax
import jax.numpy as jnp
import pytest

from jaxgmg.baselines import experience
from jaxgmg.baselines.autocurricula import scores
from jaxgmg.baselines.autocurricula import buffer
from jaxgmg.environments import cheese_in_the_corner as corner


GAMMA = 0.9


# --- replayability / PVL (GAE-based) -------------------------------------- #

def test_l1_value_loss_and_pvl_golden():
    adv = jnp.asarray([2.0, -1.0, 3.0, -4.0, 0.0])
    # l1 = mean(|adv|) = (2+1+3+4+0)/5
    assert float(scores.l1_value_loss(adv)) == pytest.approx(10.0 / 5)
    # pvl = mean(relu(adv)) = (2+0+3+0+0)/5
    assert float(scores.regret_pvl(adv)) == pytest.approx(5.0 / 5)


def test_pvl_never_exceeds_l1_and_is_zero_when_all_nonpositive():
    rng = np.random.default_rng(0)
    for _ in range(20):
        adv = jnp.asarray(rng.normal(size=8))
        assert float(scores.regret_pvl(adv)) <= float(scores.l1_value_loss(adv)) + 1e-6
    all_neg = jnp.asarray([-1.0, -2.0, -0.5, 0.0])
    assert float(scores.regret_pvl(all_neg)) == pytest.approx(0.0)


# --- maxmc-actor (the paper's default estimator) -------------------------- #

def test_maxmc_actor_equals_max_ever_minus_average_return():
    rewards = jnp.asarray([0.0, 0.0, 1.0])
    dones = jnp.asarray([False, False, True])
    max_ever = 1.0
    # single episode: average return discounts the terminal reward by gamma^2.
    avg = float(experience.compute_average_return(rewards, dones, GAMMA))
    assert avg == pytest.approx(GAMMA ** 2)
    got = float(scores.regret_maxmc_actor(rewards, dones, GAMMA, max_ever))
    assert got == pytest.approx(max_ever - GAMMA ** 2)
    # and it is exactly max_ever - compute_average_return by definition
    assert got == pytest.approx(max_ever - avg)


def test_maxmc_actor_monotone_in_max_ever():
    rewards = jnp.asarray([0.0, 1.0])
    dones = jnp.asarray([False, True])
    r_lo = float(scores.regret_maxmc_actor(rewards, dones, GAMMA, 1.0))
    r_hi = float(scores.regret_maxmc_actor(rewards, dones, GAMMA, 2.0))
    assert r_hi == pytest.approx(r_lo + 1.0)


# --- maxmc-paper / maxmc-initial (critic-based, undiscounted / initial) ---- #

def test_maxmc_paper_and_initial_golden():
    values = jnp.asarray([0.5, 0.4, 0.3])
    max_ever = 1.0
    # paper: max_ever - mean(values)
    assert float(scores.regret_maxmc_paper(values, max_ever)) == pytest.approx(1.0 - 0.4)
    # initial: max_ever - values[0]
    assert float(scores.regret_maxmc_initial(values, max_ever)) == pytest.approx(1.0 - 0.5)


# --- maxmc-critic vs maxmc-critic-balanced -------------------------------- #
#
# Both estimate max_ever - mean_t(gamma^t V_t). They must AGREE on a single
# episode, and DIVERGE on a multi-episode rollout of unequal lengths: the plain
# `critic` averages over all steps (biased toward longer episodes) while
# `balanced` averages the per-episode means. This pins the documented bias that
# `_balanced` was written to fix.

def test_critic_and_balanced_agree_on_single_episode():
    values = jnp.asarray([1.0, 1.0, 1.0])
    dones = jnp.asarray([False, False, True])
    max_ever = 2.0
    disc_mean = (1.0 + GAMMA + GAMMA ** 2) / 3
    critic = float(scores.regret_maxmc_critic(values, dones, GAMMA, max_ever))
    balanced = float(scores.regret_maxmc_critic_balanced(values, dones, GAMMA, max_ever))
    assert critic == pytest.approx(max_ever - disc_mean)
    assert balanced == pytest.approx(max_ever - disc_mean)
    assert critic == pytest.approx(balanced)


def test_critic_and_balanced_diverge_on_unequal_episodes():
    # episode 1 length 2, episode 2 length 3.
    values = jnp.asarray([1.0, 1.0, 1.0, 1.0, 1.0])
    dones = jnp.asarray([False, True, False, False, True])
    max_ever = 2.0
    s1 = 1.0 + GAMMA                       # sum gamma^t V over episode 1 (T1=2)
    s2 = 1.0 + GAMMA + GAMMA ** 2          # episode 2 (T2=3)
    # plain critic: average discounted value over ALL steps
    critic = float(scores.regret_maxmc_critic(values, dones, GAMMA, max_ever))
    assert critic == pytest.approx(max_ever - (s1 + s2) / 5)
    # balanced: mean of per-episode averages
    balanced = float(scores.regret_maxmc_critic_balanced(values, dones, GAMMA, max_ever))
    assert balanced == pytest.approx(max_ever - (s1 / 2 + s2 / 3) / 2)
    # and they genuinely differ (the bias `_balanced` corrects)
    assert abs(critic - balanced) > 1e-3


# --- oracle-actor (the oracle-latest estimator) --------------------------- #

def _corridor_corner_level(distance, width=None):
    # 3 x width grid, open middle row; mouse at left, cheese `distance` cells to
    # the right along the corridor. A fixed `width` lets several such levels
    # (different distances) stack into one batch.
    w = distance + 3 if width is None else width
    wall = np.ones((3, w), dtype=bool)
    wall[1, 1:w - 1] = False               # open corridor
    return corner.Level(
        wall_map=jnp.asarray(wall),
        cheese_pos=jnp.asarray([1, 1 + distance]),
        initial_mouse_pos=jnp.asarray([1, 1]),
    )


def test_oracle_actor_is_oracle_minus_realised_return():
    # regret_oracle_actor now takes the oracle optimal return directly and
    # subtracts the realised (average) return of the rollout.
    oracle_return = 0.7
    # zero realised reward -> regret is exactly the oracle term
    regret0 = float(scores.regret_oracle_actor(
        oracle_return=oracle_return,
        rewards=jnp.zeros(4), dones=jnp.asarray([False, False, False, True]),
        discount_rate=GAMMA,
    ))
    assert regret0 == pytest.approx(oracle_return)
    # a realised reward of 1 on the index-1 step -> avg return gamma^1
    regret1 = float(scores.regret_oracle_actor(
        oracle_return=oracle_return,
        rewards=jnp.asarray([0.0, 1.0]), dones=jnp.asarray([False, True]),
        discount_rate=GAMMA,
    ))
    assert regret1 == pytest.approx(oracle_return - GAMMA ** 1)


def test_oracle_returns_via_solver_is_gamma_pow_d_minus_1():
    # end-to-end plumbing: buffer.oracle_returns solves a batch of levels with
    # the configured corner solver and returns gamma^(d-1) (BUG-1 fixed).
    env = corner.Env(
        max_steps_in_episode=128, penalize_time=False, automatically_reset=False,
    )
    solver = corner.LevelSolver(env=env, discount_rate=GAMMA)
    ds = (1, 2, 3)
    levels = jax.tree.map(
        lambda *xs: jnp.stack(xs),
        *[_corridor_corner_level(d, width=6) for d in ds],
    )
    out = np.asarray(buffer.oracle_returns(solver, "oracle-actor", levels))
    assert out == pytest.approx([GAMMA ** (d - 1) for d in ds])


def test_oracle_returns_placeholder_for_non_oracle_methods():
    # non-oracle methods don't need a solver: a zero placeholder, no solve.
    levels = jax.tree.map(
        lambda *xs: jnp.stack(xs),
        _corridor_corner_level(1, width=6), _corridor_corner_level(2, width=6),
    )
    out = np.asarray(buffer.oracle_returns(None, "maxmc-actor", levels))
    assert out.tolist() == [0.0, 0.0]


def test_oracle_actor_optimal_agent_has_zero_regret():
    # end-to-end: an optimal agent's realised return equals the (BUG-1-fixed)
    # oracle return, so its oracle-latest regret is exactly 0.
    d = 2
    oracle_return = GAMMA ** (d - 1)    # what the corner solver reports
    rewards = jnp.asarray([0.0, 1.0])   # optimal: reward on the arrival step
    dones = jnp.asarray([False, True])
    regret = float(scores.regret_oracle_actor(
        oracle_return=oracle_return, rewards=rewards, dones=dones,
        discount_rate=GAMMA,
    ))
    assert regret == pytest.approx(0.0)


# --- dispatcher wiring (light smoke test) --------------------------------- #

def _make_rollout(rewards, dones, values):
    transitions = experience.Transition(
        env_state=None, obs=None, net_state=None, prev_action=None,
        value=jnp.asarray(values), action=None,
        log_prob=None, reward=jnp.asarray(rewards),
        done=jnp.asarray(dones, dtype=bool), info={},
    )
    return experience.Rollout(
        transitions=transitions, final_value=None,
    )


@pytest.mark.parametrize("method", ["maxmc-actor", "pvl", "maxmc-paper"])
def test_dispatcher_returns_primitive(method):
    rewards = [0.0, 0.0, 1.0]
    dones = [False, False, True]
    values = [0.5, 0.4, 0.3]
    advantages = jnp.asarray([0.2, -0.1, 0.4])
    max_ever = 1.0
    rollout = _make_rollout(rewards, dones, values)

    dispatched = float(scores.plr_compute_score(
        scoring_method=method,
        rollout=rollout,
        max_ever_return=max_ever,
        advantages=advantages,
        discount_rate=GAMMA,
        oracle_return=0.0,
    ))

    if method == "maxmc-actor":
        expected = float(scores.regret_maxmc_actor(
            jnp.asarray(rewards), jnp.asarray(dones, bool), GAMMA, max_ever))
    elif method == "pvl":
        expected = float(scores.regret_pvl(advantages))
    else:  # maxmc-paper
        expected = float(scores.regret_maxmc_paper(jnp.asarray(values), max_ever))
    assert dispatched == pytest.approx(expected)
