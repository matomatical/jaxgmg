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
smoke test (``proxy_shaping=False`` returns the primitive's value verbatim).

The oracle-latest estimator (``regret_oracle_actor``) carries the same
**BUG-1** discount off-by-one as the ``LevelSolver`` oracles
(``notes/03-bug-log.md``): it values the optimum at ``gamma^d`` while an optimal
agent realises ``gamma^(d-1)``, so an optimal agent shows *negative* regret.
Pinned here as ``xfail``.
"""

import math

import numpy as np

import jax
import jax.numpy as jnp
import pytest

from jaxgmg.baselines import experience
from jaxgmg.baselines.autocurricula import scores
from jaxgmg.environments import cheese_in_the_corner as corner
from jaxgmg.procgen import maze_solving


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

def _corridor_corner_level(distance):
    # 3 x (distance+3) grid, open middle row; mouse at left, cheese `distance`
    # cells to the right along the corridor.
    w = distance + 3
    wall = np.ones((3, w), dtype=bool)
    wall[1, 1:w - 1] = False               # open corridor
    return corner.Level(
        wall_map=jnp.asarray(wall),
        cheese_pos=jnp.asarray([1, 1 + distance]),
        initial_mouse_pos=jnp.asarray([1, 1]),
    )


def test_oracle_actor_oracle_term_is_gamma_pow_d():
    # With zero realised reward, regret == the oracle term == gamma^d, and we
    # cross-check d against maze_distances independently.
    for d in (1, 2, 3):
        level = _corridor_corner_level(d)
        dist = float(np.asarray(maze_solving.maze_distances(level.wall_map))[1, 1, 1, 1 + d])
        assert dist == d
        rewards = jnp.zeros(4)
        dones = jnp.asarray([False, False, False, True])
        regret = float(scores.regret_oracle_actor(
            level=level, rewards=rewards, dones=dones,
            discount_rate=GAMMA, proxy_oracle=False,
        ))
        assert regret == pytest.approx(GAMMA ** d)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG-1 (notes/03-bug-log.md) at the estimator level: regret_oracle_actor "
        "values the optimum at gamma^d, but an optimal agent that reaches the "
        "goal in d steps realises gamma^(d-1) (reward on the arrival step). So "
        "an optimal agent gets NEGATIVE oracle regret instead of 0. When BUG-1 "
        "is fixed, drop this xfail."
    ),
)
def test_oracle_actor_optimal_agent_has_zero_regret():
    d = 2
    level = _corridor_corner_level(d)
    # optimal agent arrives in d steps: reward on the final (arrival) step.
    rewards = jnp.asarray([0.0, 1.0])
    dones = jnp.asarray([False, True])
    regret = float(scores.regret_oracle_actor(
        level=level, rewards=rewards, dones=dones,
        discount_rate=GAMMA, proxy_oracle=False,
    ))
    assert regret == pytest.approx(0.0)


# --- dispatcher wiring (light smoke test) --------------------------------- #

def _make_rollout(rewards, dones, values):
    transitions = experience.Transition(
        env_state=None, obs=None, net_state=None, prev_action=None,
        value=jnp.asarray(values), proxy_value=None, action=None,
        log_prob=None, reward=jnp.asarray(rewards),
        done=jnp.asarray(dones, dtype=bool), info={},
    )
    return experience.Rollout(
        transitions=transitions, final_value=None, final_proxy_value=None,
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
        level=None,
        clipping=False,
    ))

    if method == "maxmc-actor":
        expected = float(scores.regret_maxmc_actor(
            jnp.asarray(rewards), jnp.asarray(dones, bool), GAMMA, max_ever))
    elif method == "pvl":
        expected = float(scores.regret_pvl(advantages))
    else:  # maxmc-paper
        expected = float(scores.regret_maxmc_paper(jnp.asarray(values), max_ever))
    assert dispatched == pytest.approx(expected)
