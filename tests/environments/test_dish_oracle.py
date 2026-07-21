"""
Oracle cross-check for Cheese-on-a-Dish: does the ``LevelSolver`` optimum agree
with the return an optimal policy actually collects by stepping through
``env.step``?

Mirror of ``tests/environments/test_corner_oracle.py`` (test-strategy item #2 in
``notes/02-cleanup-plan.md``). The dish is a *proxy* goal: by default
``terminate_after_cheese_and_dish=False``, so the episode ends on EITHER cheese
OR dish. Stepping onto the dish therefore ends the episode *before* the cheese,
so the true-reward oracle must route around the dish — the solver treats the
dish cell as a wall whenever ``dish_pos != cheese_pos``. We check that barrier
behaviour explicitly (``test_dish_acts_as_barrier``), alongside the same
``gamma^(d-1)`` realised-return / ``gamma^d`` oracle characterization and the
**BUG-1** discount off-by-one (``notes/03-bug-log.md``) as the corner oracle.

As with corner, we disable the time penalty (``penalize_time=False``) so the
clean closed form applies, and auto-reset (one clean episode per level).
"""

import numpy as np

import jax
import jax.numpy as jnp
import pytest

from jaxgmg.environments import cheese_on_a_dish as dish
from jaxgmg.procgen import maze_solving
from jaxgmg.baselines import experience


GAMMA = 0.9


@pytest.fixture
def env():
    return dish.Env(
        max_steps_in_episode=128,
        penalize_time=False,
        automatically_reset=False,
    )


@pytest.fixture
def solver(env):
    return dish.LevelSolver(env=env, discount_rate=GAMMA)


def _open(h, w, r0, r1, c0, c1):
    # all-wall grid with an open rectangle [r0:r1, c0:c1) (border stays walled).
    wall = np.ones((h, w), dtype=bool)
    wall[r0:r1, c0:c1] = False
    return wall


def _level(wall, cheese, dish_pos, mouse):
    return dish.Level(
        wall_map=jnp.asarray(np.asarray(wall, dtype=bool)),
        cheese_pos=jnp.asarray(cheese),
        dish_pos=jnp.asarray(dish_pos),
        initial_mouse_pos=jnp.asarray(mouse),
    )


def _barrier_wall_map(level):
    # Replicate the solver: the dish cell becomes a wall iff dish != cheese, so
    # the optimal path to the cheese cannot pass through (and be ended by) it.
    wall = np.asarray(level.wall_map).copy()
    d = tuple(int(x) for x in np.asarray(level.dish_pos))
    c = tuple(int(x) for x in np.asarray(level.cheese_pos))
    if d != c:
        wall[d] = True
    return jnp.asarray(wall)


def mouse_cheese_distance(level):
    dist = np.asarray(maze_solving.maze_distances(_barrier_wall_map(level)))
    mr, mc = np.asarray(level.initial_mouse_pos)
    cr, cc = np.asarray(level.cheese_pos)
    return dist[mr, mc, cr, cc]


def optimal_rollout(env, level):
    """Step a shortest-path-to-cheese (dish-avoiding) policy through env.step."""
    actions = np.asarray(maze_solving.maze_optimal_directions(_barrier_wall_map(level)))
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


# Reachable levels: 4x5 grid, open rectangle rows 1-2 x cols 1-3. mouse (1,1),
# cheese (1,3), reached in d=2 along the top row.
_WALL_2x3 = _open(4, 5, 1, 3, 1, 4)
REACHABLE = [
    # dish parked off the optimal path -> it is a barrier but doesn't block.
    ("dish_off_path", _level(_WALL_2x3, cheese=(1, 3), dish_pos=(2, 3), mouse=(1, 1)), 2),
    # cheese-on-dish (non-distinguishing): dish == cheese, so NOT a barrier;
    # arriving collects both at once.
    ("cheese_on_dish", _level(_WALL_2x3, cheese=(1, 3), dish_pos=(1, 3), mouse=(1, 1)), 2),
]


# --- the realised return is exactly gamma^(d-1) --------------------------- #

@pytest.mark.parametrize("case", REACHABLE, ids=[c[0] for c in REACHABLE])
def test_optimal_policy_reaches_cheese_in_d_steps(env, case):
    _name, level, d = case
    assert int(mouse_cheese_distance(level)) == d
    rewards, dones = optimal_rollout(env, level)
    # cheese reward 1 delivered exactly once, on the final (arrival) step.
    assert sum(rewards) == pytest.approx(1.0)
    assert dones[-1] is True
    assert len(rewards) == d


@pytest.mark.parametrize("case", REACHABLE, ids=[c[0] for c in REACHABLE])
def test_realised_return_is_gamma_pow_d_minus_1(env, case):
    _name, level, d = case
    rewards, dones = optimal_rollout(env, level)
    realised = float(experience.compute_average_return(
        rewards=jnp.asarray(rewards),
        dones=jnp.asarray(dones, dtype=bool),
        discount_rate=GAMMA,
    ))
    assert realised == pytest.approx(GAMMA ** (d - 1), rel=1e-6)


# --- the oracle equals the realised optimal return (BUG-1 fixed) ---------- #

@pytest.mark.parametrize("case", REACHABLE, ids=[c[0] for c in REACHABLE])
def test_oracle_level_value_is_gamma_pow_d_minus_1(solver, case):
    # The oracle discounts the cheese reward by gamma^(d-1), matching the
    # arrival-step index (BUG-1 fixed, was gamma^d). Same as corner/keys.
    _name, level, d = case
    soln = solver.solve(level)
    value = float(solver.level_value(soln, level))
    assert value == pytest.approx(GAMMA ** (d - 1), rel=1e-6)


@pytest.mark.parametrize("case", REACHABLE, ids=[c[0] for c in REACHABLE])
def test_oracle_value_should_equal_realised_return(env, solver, case):
    _name, level, _d = case
    rewards, dones = optimal_rollout(env, level)
    realised = float(experience.compute_average_return(
        rewards=jnp.asarray(rewards),
        dones=jnp.asarray(dones, dtype=bool),
        discount_rate=GAMMA,
    ))
    soln = solver.solve(level)
    value = float(solver.level_value(soln, level))
    assert value == pytest.approx(realised, rel=1e-6)


# --- the distinctive dish behaviour: dish-as-barrier ---------------------- #

def test_dish_acts_as_barrier(env, solver):
    # 3x5 grid, single open corridor (row 1, cols 1-3): mouse (1,1), dish (1,2),
    # cheese (1,3). The dish sits on the ONLY path to the cheese.
    wall = _open(3, 5, 1, 2, 1, 4)
    level = _level(wall, cheese=(1, 3), dish_pos=(1, 2), mouse=(1, 1))

    # Since terminate_after_cheese_and_dish is False, stepping onto the dish ends
    # the episode before the cheese, so the cheese is unreachable to the
    # true-reward oracle and its value is 0.
    assert not np.isfinite(mouse_cheese_distance(level))
    soln = solver.solve(level)
    assert float(solver.level_value(soln, level)) == 0.0

    # Walking toward the cheese (straight line on the un-modified map) steps onto
    # the dish, which terminates the episode with no cheese reward.
    actions = np.asarray(maze_solving.maze_optimal_directions(level.wall_map))
    _obs, state = env.reset_to_level(level)
    a = int(actions[1, 1, 1, 3])
    _obs, state, reward, done, _info = env.step(jax.random.PRNGKey(0), state, a)
    assert bool(state.got_dish) is True
    assert bool(state.got_cheese) is False
    assert bool(done) is True
    assert float(reward) == 0.0


# --- edge case: unreachable cheese ---------------------------------------- #

def test_unreachable_cheese_has_zero_value_and_zero_return(env, solver):
    # 7x7: cheese boxed into the top-left corner, mouse in the open interior,
    # dish elsewhere in the open (its position is irrelevant here).
    h = w = 7
    wall = np.ones((h, w), dtype=bool)
    wall[1:-1, 1:-1] = False
    wall[1, 1] = False       # cheese cell open...
    wall[1, 2] = True        # ...but boxed in
    wall[2, 1] = True
    level = _level(wall, cheese=(1, 1), dish_pos=(3, 4), mouse=(3, 3))

    assert not np.isfinite(mouse_cheese_distance(level))
    soln = solver.solve(level)
    assert float(solver.level_value(soln, level)) == 0.0
    rewards, _dones = optimal_rollout(env, level)
    assert sum(rewards) == 0.0
