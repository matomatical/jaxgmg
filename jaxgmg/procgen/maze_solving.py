import functools
import jax
import jax.numpy as jnp
import einops


@jax.jit
def maze_distances(grid):
    """
    All pairs shortest paths lengths.

    Parameters:

    * grid: bool[h, w]
            Array of walls, as returned by generate functions.

    Returns:

    * dist: float[h, w, h, w]
            Array of distance arrays. The first two axes specify the source
            square and the final two axes specify the destination square. The
            value is the length in horizontal/vertical steps of the shortest
            path not including walls from the source to the destination.

            A value of `inf` indicates that there is no path from the
            source to the destination.
    Notes:

    * The dist matrix is symmetric so the distinction between source and
      destination is arbitrary.
    * By convention, walls have infinite distance even from themselves.
    * The approach uses Floyd--Warshall algorithm, which is very simple, but
      not the fastest approach (at O(h^3w^3 operations)). The algorithm from
      Seidel, 1995, "On the all-pairs-shortest-path problem in unweighted
      undirected graphs" would be faster and probably jittable (I think
      minimax uses that algorithm). But for small maps, it probably doesn't
      make a big difference.
    """
    h, w = grid.shape
    n = h * w
    idx = jnp.arange(n)

    # initial distance matrix: zero to self, one to neighbours, inf otherwise
    dist_init = (
        jnp.full((n, n), fill_value=jnp.inf)
            .at[idx, idx].set(0.)               # distance zero to self
            .at[idx, idx-w].set(1.)             # distance one to neighbour above
            .at[idx, idx-1].set(1.)             # distance one to neighbour left
            .at[idx, idx+w].set(1.)             # distance one to neighbour down
            .at[idx, idx+1].set(1.)             # distance one to neighbour right
    )
    dist_init = jnp.where(
        (grid | grid[:,:,jnp.newaxis,jnp.newaxis]).reshape(n,n),
        jnp.inf,
        dist_init,
    )

    # floyd--warshall algorithm: consider paths via each node in turn
    def _relax_via_node(k, dist):
        return jnp.minimum(dist, dist[:,(k,)] + dist[(k,),:])
    dist = jax.lax.fori_loop(0, n, _relax_via_node, dist_init)

    # transform back into the square format for the return
    dist = einops.rearrange(
        dist,
        '(H W) (h w) -> H W h w',
        H=h, W=w, h=h, w=w,
    )

    return dist
    

@jax.jit
def maze_directional_distances(grid):
    """
    All pairs distance after moving in each direction (or staying still).

    Parameters:
    
    * grid: bool[h, w]
            Array of walls, as returned by generate functions.

    Returns:

    * dir_dists: int[h, w, h, w, 5]
            Array of direction arrays. The first two axes specify the source
            square and the final two axes specify the target square. The
            final axis has four or five values corresponding to:
            * 0: the distance upon moving up (that is, decrementing row)
            * 1: the distance upon moving left (that is, decrementing column)
            * 2: the distance upon moving down (that is, incrementing row)
            * 3: the distance upon moving right (that is, incrementing column)
            * 5: the distance upon not moving from the source square.
            
            In the case of an unreachable destination, or upon moving into a
            wall, the value is infinite. Otherwise, the value is the number
            of steps required to reach the target from the corresponding
            square.
    
    Notes:

    * Unlike for `maze_distances`, the distance distributions are not
      symmetric. However, they are probably symmetric up to some permutation
      of the actions corresponding to travelling in reverse along optimal
      paths or something. I haven't thought about it enough to say.
    """
    # solve maze
    dist = maze_distances(grid)

    # pad maze so every source has all neighbours
    dist_padded = jnp.pad(
        dist,
        (
            (1,1), (1,1), # source rows and columns
            (0,0), (0,0), # no need to pad destinations
        ),
        constant_values=jnp.inf,
    )

    dir_dist = jnp.stack((
        # up: get the distance matrix for the above source
        dist_padded[  :-2, 1:-1, :, :],
        # left: get the distance matrix for the left source
        dist_padded[ 1:-1,  :-2, :, :],
        # down: get the distance matrix for the below source
        dist_padded[ 2:  , 1:-1, :, :],
        # right: get the distance matrix for the right source
        dist_padded[ 1:-1, 2:  , :, :],
        # stay: use the original distance matrix
        dist,
    ))
    
    # rearrange into required format
    dir_dist = einops.rearrange(dir_dist, 'd H W h w -> H W h w d')

    return dir_dist


@jax.jit
def maze_optimal_directions(grid, stay_action=False):
    """
    All pairs shortest paths optimal policy. Breaks ties in up, left, down,
    right, order.

    Parameters:

    * grid: bool[h, w]
            Array of walls, as returned by generate functions.

    Returns:

    * action: int[h, w, h, w]
            Array of action arrays. The first two axes specify the source
            square and the final two axes specify the destination square.
            The value is an integer from 0 to 3+`stay_action` as follows:

            * 0 (meaning up, that is, decrement row)
            * 1 (meaning left, that is, decrement column)
            * 2 (meaning down, that is, increment row)
            * 3 (meaning right, that is, increment column)
            * (if `stay_action`) 4 (meaning stay at the same position)

            If `stay_action` is True, then the stay action is optimal if and
            only if the source is equal to the target and not a wall. If stay
            action is False, then in such cases the action is arbitrary.
            
            In the case of an unreachable target or if the source is a wall,
            an arbitrary action is returned.
    
    Notes:

    * Unlike for `maze_distances`, the action matrix is not symmetric, so the
      distinction between source and target axes is crucial. This is an easy
      error to make because the indexing will work fine. Beware.
    * Because of the ways in which ties are broken, the path from source A to
      target B might not be the reverse of the path from source B to target A.
    * I think a faster approach could be made to use the APSP methods
      described in Seidel, 1995, "On the all-pairs-shortest-path problem in
      unweighted undirected graphs". But for small maps, it probably doesn't
      make a big difference.
    """
    # find the cost for each action
    dir_dist = maze_directional_distances(grid)
    
    # (dynamically) correct for stay_action flag
    # it will always have the lowest cost by at most 1 (when source = target)
    # so, incrementing by 1 will sabotage it in the argmin operation below
    # (1 is enough because ties are broken in array order and stay is last)
    dir_dist = dir_dist.at[:,:,:,:,4].add(1-stay_action)

    # find the lowest-cost action
    actions = jnp.argmin(dir_dist, axis=4)

    return actions

