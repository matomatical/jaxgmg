"""
Tests for ``jaxgmg.procgen.maze_solving`` — the JAX Floyd--Warshall all-pairs
shortest-path solver that every oracle in the library depends on.

Coverage:

* golden distances on hand-laid mazes (corridors, detours, unreachable);
* invariants: symmetry, zero self-distance, walls are ``inf``, triangle
  inequality, non-negative integers;
* a cross-check against an independent BFS reference over many generated mazes;
* the **border invariant** — distances are correct only because of the
  mandatory 1-cell wall border, which masks the wrap-around in the flattened
  neighbour indexing. This is a load-bearing, easy-to-miss coupling
  (``notes/00-codebase-model.md`` §6.1), so we pin it with a dedicated test;
* the directional-distance and optimal-direction derivatives, including the
  documented up/left/down/right tie-breaking and the ``stay_action`` edge cases.
"""

import numpy as np
from collections import deque

import jax.numpy as jnp

from jaxgmg.procgen import maze_solving


# --- independent reference implementation --------------------------------- #

def reference_distances(grid):
    """
    Trusted all-pairs shortest paths via plain BFS from every open cell.

    Deliberately written in straightforward NumPy with explicit bounds checks
    (no index wrap-around) so it cannot share a bug with the JAX solver. Walls
    are unreachable and have infinite distance even to themselves, matching the
    documented convention of ``maze_distances``.
    """
    grid = np.asarray(grid, dtype=bool)
    h, w = grid.shape
    dist = np.full((h, w, h, w), np.inf)
    for si in range(h):
        for sj in range(w):
            if grid[si, sj]:
                continue
            d = np.full((h, w), np.inf)
            d[si, sj] = 0
            q = deque([(si, sj)])
            while q:
                ci, cj = q.popleft()
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < h and 0 <= nj < w \
                            and not grid[ni, nj] and d[ni, nj] == np.inf:
                        d[ni, nj] = d[ci, cj] + 1
                        q.append((ni, nj))
            dist[si, sj] = d
    return dist


def bordered(interior):
    """Wrap a 2D interior wall pattern in a 1-cell wall border."""
    interior = np.asarray(interior, dtype=bool)
    ih, iw = interior.shape
    grid = np.ones((ih + 2, iw + 2), dtype=bool)
    grid[1:-1, 1:-1] = interior
    return grid


def solve(grid):
    """Run the solver and return a plain NumPy array."""
    return np.asarray(maze_solving.maze_distances(jnp.asarray(grid, dtype=bool)))


# --- golden distances on hand-laid mazes ---------------------------------- #

def test_self_distance_zero_for_open_cell():
    # 3x3 with a single open interior cell.
    grid = bordered([[False]])
    dist = solve(grid)
    assert dist[1, 1, 1, 1] == 0.0


def test_walls_have_infinite_distance_including_to_self():
    grid = bordered([[False]])
    dist = solve(grid)
    # every border cell is a wall: infinite distance from anything (incl self)
    assert np.isinf(dist[0, 0, 0, 0])          # wall to itself
    assert np.isinf(dist[0, 0, 1, 1])          # wall to open cell
    assert np.isinf(dist[1, 1, 0, 0])          # open cell to wall


def test_straight_corridor_distances_equal_column_difference():
    # interior is a single open row of length 5 -> a 1x5 corridor.
    grid = bordered([[False, False, False, False, False]])
    dist = solve(grid)
    row = 1
    for a in range(1, 6):
        for b in range(1, 6):
            assert dist[row, a, row, b] == abs(a - b), (a, b)


def test_open_room_is_manhattan_distance():
    # a fully open 3x3 interior room: with no obstacles, shortest path length
    # is the Manhattan distance.
    grid = bordered(np.zeros((3, 3), dtype=bool))
    dist = solve(grid)
    for ai in range(1, 4):
        for aj in range(1, 4):
            for bi in range(1, 4):
                for bj in range(1, 4):
                    assert dist[ai, aj, bi, bj] == abs(ai - bi) + abs(aj - bj)


def test_obstacle_forces_detour():
    # 5x5 interior with a wall wall down the middle column except the bottom,
    # forcing a path around it.
    #   interior (3x3):
    #     . # .
    #     . # .
    #     . . .
    interior = [[0, 1, 0],
                [0, 1, 0],
                [0, 0, 0]]
    grid = bordered(interior)
    dist = solve(grid)
    # top-left interior (1,1) to top-right interior (1,3): straight line is
    # blocked by the wall, must go down 2, right 2, up 2 = 6 steps.
    assert dist[1, 1, 1, 3] == 6
    # cross-check the whole thing against BFS too.
    np.testing.assert_array_equal(dist, reference_distances(grid))


def test_unreachable_region_is_infinite():
    # two open cells separated by a solid wall column -> mutually unreachable.
    #   interior (1x3): . # .
    interior = [[0, 1, 0]]
    grid = bordered(interior)
    dist = solve(grid)
    assert np.isinf(dist[1, 1, 1, 3])
    assert np.isinf(dist[1, 3, 1, 1])
    # but each is reachable from itself
    assert dist[1, 1, 1, 1] == 0.0
    assert dist[1, 3, 1, 3] == 0.0


# --- invariants over generated mazes -------------------------------------- #

def test_symmetry(make_mazes, generator_name):
    for grid in make_mazes(name=generator_name, count=6, size=7):
        dist = solve(grid)
        # dist[i,j,k,l] == dist[k,l,i,j]
        swapped = np.transpose(dist, (2, 3, 0, 1))
        np.testing.assert_array_equal(dist, swapped)


def test_self_distance_and_walls(make_mazes, generator_name):
    for grid in make_mazes(name=generator_name, count=6, size=7):
        grid_np = np.asarray(grid)
        dist = solve(grid)
        h, w = grid_np.shape
        for i in range(h):
            for j in range(w):
                if grid_np[i, j]:
                    assert np.isinf(dist[i, j, i, j])   # wall: inf to self
                else:
                    assert dist[i, j, i, j] == 0.0      # open: zero to self


def test_values_are_nonnegative_integers_or_inf(make_mazes, generator_name):
    for grid in make_mazes(name=generator_name, count=6, size=7):
        dist = solve(grid)
        finite = dist[np.isfinite(dist)]
        assert np.all(finite >= 0)
        assert np.all(finite == np.round(finite))


def test_triangle_inequality(make_mazes, generator_name):
    for grid in make_mazes(name=generator_name, count=4, size=7):
        dist = solve(grid)
        n = dist.shape[0] * dist.shape[1]
        d = dist.reshape(n, n)
        # d[a,b] <= d[a,m] + d[m,b] for all a, b, m  (inf arithmetic is fine
        # under numpy: inf + x == inf, and x <= inf always holds)
        via = d[:, :, None] + d[None, :, :]      # via[a,m,b]
        best_via = via.min(axis=1)               # min over m
        # allow exact equality; shortest path is itself a "via" of its own nodes
        assert np.all(d <= best_via + 1e-9)


def test_cross_check_against_bfs(make_mazes, generator_name):
    for grid in make_mazes(name=generator_name, count=8, size=7):
        got = solve(grid)
        ref = reference_distances(grid)
        # compare with inf handled (array_equal treats inf==inf as True)
        np.testing.assert_array_equal(got, ref)


# --- the border invariant ------------------------------------------------- #

def test_cross_check_holds_on_larger_bordered_maze(make_mazes):
    # one larger maze per generator, to be sure the border masking scales.
    for name in ("open", "blocks", "edges", "tree"):
        for grid in make_mazes(name=name, count=2, size=11):
            np.testing.assert_array_equal(solve(grid), reference_distances(grid))


def test_border_invariant_is_load_bearing():
    """
    The solver is correct *only* because of the mandatory 1-cell wall border.

    ``maze_distances`` builds the neighbour graph by offsetting flattened
    indices by +/-1 and +/-w. At the grid edges those offsets wrap around (e.g.
    the last cell of a row becomes "adjacent" to the first cell of the next
    row). The border invariant says: because every edge cell is a wall, those
    spurious wrap edges always touch a wall and are masked to ``inf`` by the
    ``grid | grid[...]`` step — so they never corrupt interior distances.

    This test guards that coupling. A *borderless* fully-open grid triggers the
    wrap and yields distances that are provably too short, while the same
    interior *with* a border matches the BFS reference exactly. If someone ever
    changes the neighbour indexing to handle borders explicitly, this test will
    flip and force a deliberate decision rather than a silent correctness bug.
    """
    interior = np.zeros((3, 3), dtype=bool)

    # WITH border: exactly correct.
    with_border = bordered(interior)
    np.testing.assert_array_equal(
        solve(with_border),
        reference_distances(with_border),
    )

    # WITHOUT border: wrap-around makes some distances spuriously short.
    no_border = interior
    got = solve(no_border)
    ref = reference_distances(no_border)
    assert not np.array_equal(got, ref), (
        "borderless maze unexpectedly matched BFS — the wrap-around the border "
        "invariant protects against may have been changed; revisit the invariant"
    )
    # Concretely: opposite corners of a 3x3 open grid are 4 apart, but the
    # wrap makes the solver believe they are adjacent.
    assert ref[0, 0, 2, 2] == 4
    assert got[0, 0, 2, 2] < 4


# --- maze_directional_distances ------------------------------------------- #

def test_directional_distances_shape_and_stay_channel(make_mazes):
    for grid in make_mazes(name="blocks", count=4, size=7):
        dist = solve(grid)
        dir_dist = np.asarray(maze_solving.maze_directional_distances(jnp.asarray(grid)))
        h, w = np.asarray(grid).shape
        assert dir_dist.shape == (h, w, h, w, 5)
        # channel 4 ("stay") must be exactly the plain distance matrix.
        np.testing.assert_array_equal(dir_dist[:, :, :, :, 4], dist)


def test_directional_distances_match_neighbour_distance(make_mazes):
    # moving in a direction from source s should give the distance-to-target
    # measured from the cell you step into (inf if that cell is a wall / oob).
    offsets = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1)}  # up,left,down,right
    for grid in make_mazes(name="blocks", count=4, size=7):
        grid_np = np.asarray(grid)
        dist = solve(grid)
        dir_dist = np.asarray(maze_solving.maze_directional_distances(jnp.asarray(grid)))
        h, w = grid_np.shape
        for si in range(h):
            for sj in range(w):
                for a, (di, dj) in offsets.items():
                    ni, nj = si + di, sj + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        expected = dist[ni, nj]            # distance from neighbour
                    else:
                        expected = np.full((h, w), np.inf)
                    np.testing.assert_array_equal(dir_dist[si, sj, :, :, a], expected)


# --- maze_optimal_directions ---------------------------------------------- #

# action index -> (drow, dcol)
_MOVES = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1)}


def test_optimal_direction_tie_break_order():
    # source in the middle of an open room; an equidistant target reachable
    # equally via "up" and "left" must break the tie to "up" (index 0), since
    # ties resolve in up, left, down, right order.
    grid = bordered(np.zeros((3, 3), dtype=bool))
    actions = np.asarray(maze_solving.maze_optimal_directions(jnp.asarray(grid)))
    # source (2,2) (centre of interior), target (1,1) (diagonally up-left):
    # up then left, or left then up — both optimal; tie-break picks up == 0.
    assert actions[2, 2, 1, 1] == 0


def test_following_optimal_directions_reaches_target(make_mazes):
    # the strongest end-to-end check: walking the optimal-direction field from
    # any open source to any reachable open target arrives in exactly the
    # shortest-path number of steps.
    for grid in make_mazes(name="blocks", count=4, size=7):
        grid_np = np.asarray(grid)
        dist = solve(grid)
        actions = np.asarray(maze_solving.maze_optimal_directions(jnp.asarray(grid)))
        h, w = grid_np.shape
        opens = [(i, j) for i in range(h) for j in range(w) if not grid_np[i, j]]
        for (si, sj) in opens:
            for (ti, tj) in opens:
                d = dist[si, sj, ti, tj]
                if not np.isfinite(d):
                    continue
                ci, cj = si, sj
                for step in range(int(d)):
                    a = int(actions[ci, cj, ti, tj])
                    dci, dcj = _MOVES[a]
                    ci, cj = ci + dci, cj + dcj
                    assert not grid_np[ci, cj], "stepped into a wall"
                assert (ci, cj) == (ti, tj), \
                    f"from {(si, sj)} to {(ti, tj)}: arrived at {(ci, cj)} after {d} steps"


def test_optimal_directions_stay_action_edge_cases():
    interior = [[0, 1, 0]]            # two open cells separated by a wall
    grid = bordered(interior)
    actions = np.asarray(
        maze_solving.maze_optimal_directions(jnp.asarray(grid), stay_action=True)
    )
    # source == target -> stay (4)
    assert actions[1, 1, 1, 1] == 4
    # wall source -> stay (4)
    assert actions[0, 0, 1, 1] == 4
    # wall target -> stay (4)
    assert actions[1, 1, 0, 0] == 4
    # unreachable target -> stay (4)
    assert actions[1, 1, 1, 3] == 4


def test_optimal_directions_no_stay_action_is_in_range(make_mazes):
    # without the stay action, all entries are in {0,1,2,3} (edge cases are
    # documented as arbitrary, but must still be a valid move index).
    for grid in make_mazes(name="blocks", count=3, size=7):
        actions = np.asarray(maze_solving.maze_optimal_directions(jnp.asarray(grid)))
        assert actions.min() >= 0 and actions.max() <= 3
