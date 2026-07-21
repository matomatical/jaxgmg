"""
Oracle cross-check for Cheese-in-the-Corner: does the ``LevelSolver`` optimum
agree with the return an optimal policy actually collects by stepping through
``env.step``?

This is test-strategy item #2 in ``notes/02-cleanup-plan.md`` and validates the
oracle-latest regret estimator end-to-end. We deliberately disable the time
penalty (``penalize_time=False``) so the clean closed form γ^d applies, matching
the oracle's documented validity constraints (cardinal actions, no time
penalty, γ ∈ (0,1), episode long enough).

It also surfaces a subtle off-by-one: see
``test_oracle_value_should_equal_realised_return``.
"""

import numpy as np

import jax
import jax.numpy as jnp
import pytest

from jaxgmg.environments import cheese_in_the_corner as corner
from jaxgmg.procgen import maze_generation, maze_solving
from jaxgmg.baselines import experience


GAMMA = 0.9
_MOVES = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1)}


@pytest.fixture
def env():
    # no time penalty, no auto-reset: a single clean optimal episode per level.
    return corner.Env(
        max_steps_in_episode=128,
        penalize_time=False,
        automatically_reset=False,
    )


@pytest.fixture
def solver(env):
    return corner.LevelSolver(env=env, discount_rate=GAMMA)


def sample_levels(count=8, size=7, seed=0):
    # Tree mazes are spanning trees of the interior, so every open cell is
    # reachable from every other -> the cheese is always reachable and the
    # mouse->cheese distance is always finite.
    gen = corner.LevelGenerator(
        height=size, width=size,
        maze_generator=maze_generation.TreeMazeGenerator(),
        corner_size=1,
    )
    return [gen.sample(k) for k in jax.random.split(jax.random.PRNGKey(seed), count)]


def optimal_rollout(env, level):
    """
    Step an optimal (shortest-path-to-cheese) policy through ``env.step``.

    Returns ``(rewards, dones)`` for the single episode (auto-reset disabled).
    """
    actions = np.asarray(maze_solving.maze_optimal_directions(level.wall_map))
    cheese = np.asarray(level.cheese_pos)
    _obs, state = env.reset_to_level(level)
    rng = jax.random.PRNGKey(0)
    rewards, dones = [], []
    for _ in range(env.max_steps_in_episode):
        mr, mc = int(state.mouse_pos[0]), int(state.mouse_pos[1])
        a = int(actions[mr, mc, cheese[0], cheese[1]])
        rng, rng_step = jax.random.split(rng)
        _obs, state, reward, done, _info = env.step(rng_step, state, a)
        rewards.append(float(reward))
        dones.append(bool(done))
        if done:
            break
    return rewards, dones


def mouse_cheese_distance(level):
    dist = np.asarray(maze_solving.maze_distances(level.wall_map))
    mr, mc = np.asarray(level.initial_mouse_pos)
    cr, cc = np.asarray(level.cheese_pos)
    return dist[mr, mc, cr, cc]


# --- the realised return is exactly gamma^(d-1) --------------------------- #

def test_optimal_policy_reaches_cheese_in_d_steps(env):
    for level in sample_levels(count=8):
        d = mouse_cheese_distance(level)
        rewards, dones = optimal_rollout(env, level)
        # reward 1 delivered exactly once, on the final (arrival) step.
        assert sum(rewards) == pytest.approx(1.0)
        assert dones[-1] is True
        assert len(rewards) == int(d)          # arrives in d steps


def test_realised_return_is_gamma_pow_d_minus_1(env):
    # this is the actual regret target: compute_average_return over the rollout.
    for level in sample_levels(count=8):
        d = int(mouse_cheese_distance(level))
        rewards, dones = optimal_rollout(env, level)
        realised = float(experience.compute_average_return(
            rewards=jnp.asarray(rewards),
            dones=jnp.asarray(dones, dtype=bool),
            discount_rate=GAMMA,
        ))
        assert realised == pytest.approx(GAMMA ** (d - 1), rel=1e-6)


# --- the oracle equals the realised optimal return (BUG-1 fixed) ---------- #

def test_oracle_level_value_is_gamma_pow_d_minus_1(env, solver):
    # The oracle discounts the cheese reward by gamma^(d-1), matching the
    # arrival-step index (d-1) of the reward (BUG-1 fixed, was gamma^d).
    for level in sample_levels(count=8):
        d = int(mouse_cheese_distance(level))
        soln = solver.solve(level)
        value = float(solver.level_value(soln, level))
        assert value == pytest.approx(GAMMA ** (d - 1), rel=1e-6)


def test_oracle_value_should_equal_realised_return(env, solver):
    for level in sample_levels(count=8):
        rewards, dones = optimal_rollout(env, level)
        realised = float(experience.compute_average_return(
            rewards=jnp.asarray(rewards),
            dones=jnp.asarray(dones, dtype=bool),
            discount_rate=GAMMA,
        ))
        soln = solver.solve(level)
        value = float(solver.level_value(soln, level))
        assert value == pytest.approx(realised, rel=1e-6)


# --- edge case: unreachable cheese ---------------------------------------- #

def build_isolated_cheese_level():
    # 7x7: cheese walled into the top-left corner, mouse in the open interior.
    h = w = 7
    wall = np.ones((h, w), dtype=bool)
    wall[1:-1, 1:-1] = False           # open interior
    wall[1, 1] = False                 # cheese cell open...
    wall[1, 2] = True                  # ...but boxed in by walls
    wall[2, 1] = True
    return corner.Level(
        wall_map=jnp.asarray(wall),
        cheese_pos=jnp.asarray([1, 1]),
        initial_mouse_pos=jnp.asarray([3, 3]),
    )


def test_unreachable_cheese_has_zero_value_and_zero_return(env, solver):
    level = build_isolated_cheese_level()
    # distance is infinite
    assert not np.isfinite(mouse_cheese_distance(level))
    # oracle value is zero (gamma^inf == 0)
    soln = solver.solve(level)
    assert float(solver.level_value(soln, level)) == 0.0
    # and the agent collects nothing
    rewards, dones = optimal_rollout(env, level)
    assert sum(rewards) == 0.0
