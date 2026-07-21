"""
Oracle cross-check for Keys-and-Chests: does ``FullLevelSolver``'s
enumerate-and-argmax oracle report the correct optimal discounted return?

Test-strategy item #3 in ``notes/02-cleanup-plan.md``. The combinatorial
*primitives* (permutations / associations) are covered in
``procgen/test_combinatorix.py``; this file covers the **oracle that consumes
them** to value a level.

Two independent angles:

1. **Cardinality** of the enumerated plan set. The oracle evaluates the full
   cartesian product ``permutations(K, N) x permutations(C, N) x
   associations(N)`` where ``N = min(K, C)`` — including the documented 21,600
   plans for ``K=3, C=10``.
2. **Golden oracle values** on hand-built corridor levels whose optimal path is
   a straight line of ``MOVE_RIGHT``s, cross-checked against the return actually
   collected by stepping ``env.step`` with that hardcoded optimal action
   sequence (fully independent of the oracle's own simulation).

As with the corner oracle, this surfaces the **BUG-1** discount off-by-one
(``notes/03-bug-log.md``): the oracle discounts a chest's reward by
``gamma ** (cumulative distance to that chest)``, but the realised return
discounts by ``gamma ** (distance - 1)`` because the reward lands on the
*arrival* step. So the keys oracle also over-discounts by one factor of gamma
per chest. Pinned as ``xfail`` here too (``test_oracle_value_should_equal_..``).
"""

import math

import numpy as np

import jax
import jax.numpy as jnp
import pytest

from jaxgmg.environments import keys_and_chests as kc
from jaxgmg.procgen import combinatorix
from jaxgmg.baselines import experience


GAMMA = 0.9
RIGHT = int(kc.Action.MOVE_RIGHT)


def catalan(n):
    return math.comb(2 * n, n) // (n + 1)


# --- cardinality of the enumerated plan set ------------------------------- #
#
# _plan() forms sequences_of_keys = permutations(K, N),
# sequences_of_chests = permutations(C, N), sequences_of_which =
# associations(N), and vmaps the evaluator over their cartesian product.

def plan_set_shapes(K, C):
    N = min(K, C)
    n_keys = np.asarray(combinatorix.permutations(K, N)).shape[0]
    n_chests = np.asarray(combinatorix.permutations(C, N)).shape[0]
    n_which = np.asarray(combinatorix.associations(N)).shape[0]
    return n_keys, n_chests, n_which


def test_plan_set_cardinality_matches_closed_form():
    for K in range(1, 5):
        for C in range(1, 5):
            N = min(K, C)
            n_keys, n_chests, n_which = plan_set_shapes(K, C)
            assert n_keys == math.perm(K, N)
            assert n_chests == math.perm(C, N)
            assert n_which == catalan(N)


def test_plan_set_cardinality_k3_c10_is_21600():
    # documented example: K=3 keys, C=10 chests -> N=min(3,10)=3.
    # perm(3,3)=6, perm(10,3)=720, Catalan(3)=5 -> 6*720*5 = 21600 plans.
    n_keys, n_chests, n_which = plan_set_shapes(3, 10)
    assert (n_keys, n_chests, n_which) == (6, 720, 5)
    assert n_keys * n_chests * n_which == 21600


# --- golden corridor levels ----------------------------------------------- #
#
# Each level's optimal policy is a straight run of MOVE_RIGHT, so we can
# hardcode the optimal action sequence and compute both the oracle's discount
# convention (gamma ** cumulative_distance) and the realised return
# (gamma ** (distance - 1)) by hand.


def make_env(penalize_time=False):
    # no time penalty (clean gamma^d form), no auto-reset (single episode).
    return kc.Env(
        max_steps_in_episode=128,
        penalize_time=penalize_time,
        automatically_reset=False,
    )


def parse_level(level_str, height, width, num_keys, num_chests):
    p = kc.LevelParser(
        height=height,
        width=width,
        num_keys_max=num_keys,       # exactly as many slots as items -> none hidden
        num_chests_max=num_chests,
        inventory_map=jnp.arange(num_keys),
    )
    return p.parse(level_str)


def rollout_return(env, level, actions, gamma):
    """Step `actions` through env.step and return the realised discounted return."""
    _obs, state = env.reset_to_level(level)
    rng = jax.random.PRNGKey(0)
    rewards, dones = [], []
    for a in actions:
        rng, rng_step = jax.random.split(rng)
        _obs, state, reward, done, _info = env.step(rng_step, state, int(a))
        rewards.append(float(reward))
        dones.append(bool(done))
        if done:
            break
    realised = float(experience.compute_average_return(
        rewards=jnp.asarray(rewards),
        dones=jnp.asarray(dones, dtype=bool),
        discount_rate=gamma,
    ))
    return realised, rewards, dones


# name, level_str, (height, width), num_keys, num_chests,
#   optimal MOVE_RIGHT count,
#   oracle value (gamma^cumulative-dist per chest),
#   realised value (gamma^(dist-1) per chest)
GOLDEN = [
    (
        # 1 key, 1 chest: @ -> k(d1) -> c(d2). chest reward at cumulative dist 2.
        "one_key_one_chest",
        """
        # # # # #
        # @ k c #
        # # # # #
        """,
        (3, 5), 1, 1, 2,
        GAMMA ** 2,                       # oracle: gamma^2
        GAMMA ** 1,                       # realised: chest opens on step index 1
    ),
    (
        # 2 keys, 2 chests: @ -> k -> k -> c(d3) -> c(d4).
        "two_keys_two_chests",
        """
        # # # # # # #
        # @ k k c c #
        # # # # # # #
        """,
        (3, 7), 2, 2, 4,
        GAMMA ** 3 + GAMMA ** 4,          # oracle
        GAMMA ** 2 + GAMMA ** 3,          # realised (opens at t=2, t=3)
    ),
    (
        # 1 key, 2 chests: must pick which chest to open. Right chest is closer
        # after grabbing the key (@ -> k(d1) -> c_right(d2)) than the left one
        # (@ -> k(d1) -> back to c_left, d4). Oracle must argmax to gamma^2.
        "one_key_two_chests_pick_nearer",
        """
        # # # # # # #
        # c . @ k c #
        # # # # # # #
        """,
        (3, 7), 1, 2, 2,
        GAMMA ** 2,                       # oracle picks the reachable-sooner chest
        GAMMA ** 1,                       # realised
    ),
]


@pytest.fixture
def env():
    return make_env()


@pytest.fixture
def solver(env):
    return kc.FullLevelSolver(env=env, discount_rate=GAMMA)


@pytest.mark.parametrize("case", GOLDEN, ids=[c[0] for c in GOLDEN])
def test_golden_optimal_rollout_collects_expected_return(env, case):
    _name, s, (h, w), nk, nc, n_right, _oracle, realised_expected = case
    level = parse_level(s, h, w, nk, nc)
    actions = [RIGHT] * n_right
    realised, rewards, dones = rollout_return(env, level, actions, GAMMA)
    # every chest that can be opened is opened exactly once
    assert sum(rewards) == pytest.approx(min(nk, nc))
    assert dones[-1] is True
    assert realised == pytest.approx(realised_expected, rel=1e-6)


@pytest.mark.parametrize("case", GOLDEN, ids=[c[0] for c in GOLDEN])
def test_oracle_level_value_matches_gamma_pow_cumulative_distance(solver, case):
    # Pin the CURRENT oracle behaviour: it enumerates plans and returns the max
    # sum of gamma^(cumulative distance) over opened chests. (See off-by-one.)
    _name, s, (h, w), nk, nc, _n_right, oracle_expected, _realised = case
    level = parse_level(s, h, w, nk, nc)
    soln = solver.solve(level)
    value = float(solver.level_value(soln, level))
    assert value == pytest.approx(oracle_expected, rel=1e-6)


@pytest.mark.parametrize("case", GOLDEN, ids=[c[0] for c in GOLDEN])
@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG-1 (notes/03-bug-log.md) recurs in the keys oracle: "
        "FullLevelSolver discounts each chest reward by gamma^(cumulative "
        "distance to the chest), but the realised return discounts by "
        "gamma^(distance-1) since the reward lands on the arrival step. So the "
        "oracle over-discounts by one factor of gamma per chest. When BUG-1 is "
        "fixed across all envs, drop this xfail."
    ),
)
def test_oracle_value_should_equal_realised_return(env, solver, case):
    _name, s, (h, w), nk, nc, n_right, _oracle, _realised = case
    level = parse_level(s, h, w, nk, nc)
    realised, _rewards, _dones = rollout_return(env, level, [RIGHT] * n_right, GAMMA)
    soln = solver.solve(level)
    value = float(solver.level_value(soln, level))
    assert value == pytest.approx(realised, rel=1e-6)
