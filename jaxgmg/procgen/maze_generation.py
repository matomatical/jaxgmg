"""
JAX-compatible functions for generating and solving different kinds of mazes.
"""

import functools
import jax
import jax.numpy as jnp
import einops
from flax import struct

from jaxgmg.procgen import noise_generation


def get_generator_function(name):
    """
    Maps a string descriptor of a maze generation method. Useful for handling
    command line arguments.

    Parameters:

    * name : str
        * 'kruskal' or 'tree' for `generate_tree_maze`
        * 'edge' for `generate_edge_maze`
        * 'block' or 'blocks' for `generate_block_maze`
        * 'open' or 'empty' for `generate_open_maze`

    Returns:

    * the above-listed maze generation function
    """
    if name.lower() in {"kruskal", "tree"}:
        return generate_tree_maze
    elif name.lower() in {"edges", "bernoulli"}:
        return generate_edge_maze
    elif name.lower() in {"block", "blocks"}:
        return generate_block_maze
    elif name.lower() in {"noise"}:
        return generate_noise_maze
    elif name.lower() in {"open", "empty"}:
        return generate_open_maze
    else:
        raise ValueError(f"Unknown maze generation method {name!r}")


# # # 
# GENERATING MAZES

@struct.dataclass
class MazeGenerator:
    """
    Abstract base class for maze generation.

    Parameters:

    * height : int (>= 3)
        Maze height (number of rows in grid).
    * width : int (>= 3)
        Maze width (number of columns in grid).
    """
    height: int
    width: int


@struct.dataclass
class TreeMazeGenerator(MazeGenerator):
    """
    Generate tree mazes ('perfect mazes' or 'acyclic mazes') using Kruskal's
    algorithm for finding a random spanning tree of a grid graph.

    Parameters:

    * height : int (>= 3)
        Maze height (number of rows in grid).
    * width : int (>= 3)
        Maze width (number of columns in grid).
    * alt_kruskal_algorithm : bool (default False)
        Whether to use an alternative implementation of Kruskal's algorithm
        (probably slower after JAX acceleration).
    """
    alt_kruskal_algorithm: bool = False

    def __post_init__(self):
        assert self.height >= 3, "height must be at least 3"
        assert self.width >= 3, "width must be at least 3"
        assert self.height % 2 == 1, "height must be odd"
        assert self.width % 2 == 1,  "width must be odd"


    @functools.partial(jax.jit, static_argnames=('self',))
    def generate(self, key):
        """
        Generate a `height` by `width` binary gridworld with a 1-wall thick
        border and a random acyclic maze in the centre.

        Consider the 'junction' squares
            
            (1,1), (1,3), ..., (1,w-1), (3,1), ..., (height-1,width-1).
        
        These squares form the nodes of a grid graph. This function
        constructs a random spanning tree of this grid graph using Kruskal's
        algorithm, and returns the corresponding binary matrix.

        Parameters:

        * key : PRNGKey
                RNG state (will be consumed)

        Returns:

        * grid : bool[height, width]
                the binary grid (True indicates a wall, False indicates a
                path)


        """
        # assign each 'junction' in the grid an integer node id
        H, W = self.height // 2, self.width // 2
        nodes = jnp.arange(H * W)
        ngrid = nodes.reshape((H, W))

        # an edge between each pair of nodes (represented as a node id pair)
        # note: there are (H-1)W + H(W-1) = 2HW - H - W edges
        h_edges = jnp.stack((ngrid[:,:-1].flatten(), ngrid[:,1:].flatten()))
        v_edges = jnp.stack((ngrid[:-1,:].flatten(), ngrid[1:,:].flatten()))
        edges = jnp.concatenate((h_edges, v_edges), axis=-1).transpose()

        # kruskal's random spanning tree algorithm
        if self.alt_kruskal_algorithm:
            include_edges = self._kruskal_alt(key, nodes, edges)
        else:
            include_edges = self._kruskal(key, nodes, edges)

        # finally, generate the grid array
        grid = jnp.ones((self.height, self.width), dtype=bool)
        grid = grid.at[1::2,1::2].set(False)        # carve out junctions
        include_edges_ijs = jnp.rint(jnp.stack((    # carve out edges
            include_edges // W,
            include_edges % W,
        )).mean(axis=-1) * 2 + 1).astype(int)
        grid = grid.at[tuple(include_edges_ijs)].set(False)

        return grid


    def _kruskal(self, key, nodes, edges):
        """
        Kruskal's random spanning tree algorithm for an unweighted connected
        graph.

        Parameters:

        * key : PRNGKey
            Random state (consumed).
        * nodes : int[n]
            Labels of the nodes of the graph. Should be unique.
        * edges : int[m,2]
            Label pairs for the edges available. Pairs should be unique.

        Returns:
            
        * include_edges : int[n-1,2]
            Subset of edges included in the random spanning tree.

        Note: if the node labels are not unique or the graph is not
        connected, some of the entries in `include_edges` may become invalid.
        """
        # initially each node is in its own subtree
        initial_parents = nodes

        # randomly shuffling the edges creates a random spanning tree
        edges = jax.random.permutation(key, edges, axis=0)

        # for each edge we decide whether to include or skip it;
        # track connected subtrees with a simple union-find data structure
        def try_edge(parents, edge):
            u, v = edge
            pu = parents[u]
            pv = parents[v]
            include_edge = (pu != pv)
            new_parents = jax.lax.select(
                include_edge & (parents == pv),
                jnp.full_like(parents, pu),
                parents,
            )
            return new_parents, include_edge
        
        _final_parents, include_edge_mask = jax.lax.scan(
            try_edge,
            initial_parents,
            edges,
        )

        # extract the pairs corresponding to the `n-1` included edges
        include_edges = edges[
            jnp.where(include_edge_mask, size=(nodes.size-1))
        ]
        return include_edges


    def _kruskal_alt(self, key, nodes, edges):
        """
        Alternative implementation of Kruskal's algorithm that is faster in
        theory but seems to be much slower in practice when acelerated and
        running in parallel on a GPU.
        """
        # initially each node is in its own subtree
        initial_parents = nodes

        # randomly shuffling the edges creates a random spanning tree
        edges = jax.random.permutation(key, edges, axis=0)

        # for each edge we decide whether to include it or skip it; tracking
        # connected subtrees with a sophisticated union-find data structure.

        def _find(x, parent):
            """
            Finds the root of x, while updating parents so that parent[i]
            points one step closer to the root of i for next time.
            """
            px = parent[x]
            def _find_body_fun(args):
                x, px, parents = args
                ppx = parents[px]
                return px, ppx, parents.at[x].set(ppx)
            root, _, parent = jax.lax.while_loop(
                lambda args: args[0] != args[1],
                _find_body_fun,
                (x, px, parent),
            )
            return root, parent

        def _union(root_x, root_y, parents):
            """
            Updates the root of x to be the root of y.
            """
            return parents.at[root_x].set(root_y)

        def try_edge(parents, edge):
            u, v = edge
            ru, parents = _find(u, parents)
            rv, parents = _find(v, parents)
            include_edge = (ru != rv)
            parents = jax.lax.cond(
                include_edge,
                _union,
                lambda rx, ry, ps: ps,
                ru,
                rv,
                parents,
            )
            return parents, include_edge

        _final_parents, include_edge_mask = jax.lax.scan(
            try_edge,
            initial_parents,
            edges,
        )

        # extract the pairs corresponding to the `n-1` included edges
        include_edges = edges[
            jnp.where(include_edge_mask, size=(nodes.size-1))
        ]
        return include_edges


@struct.dataclass
class EdgeMazeGenerator(MazeGenerator):
    """
    Generate edge mazes (random subgraph of a grid graph).

    Parameters:

    * height : int (>= 3)
        Maze height (number of rows in grid).
    * width : int (>= 3)
        Maze width (number of columns in grid).
    * edge_prob : float (probability, default 0.75).
        Independent probability for each edge between grid nodes to be
        available in the maze.
        Default of 0.75 leads to mazes that are usually mostly connected
        (except around some edges or corners).
    """
    edge_prob: float = 0.75

    def __post_init__(self):
        assert self.height >= 3, "height must be at least 3"
        assert self.width >= 3, "width must be at least 3"
        assert self.height % 2 == 1, "height must be odd"
        assert self.width % 2 == 1,  "width must be odd"
        assert 0.0 <= self.edge_prob <= 1.0, "edge_prob must be a probability"


    @functools.partial(jax.jit, static_argnames=('self',))
    def generate(self, key):
        """
        Generate a `height` by `width` binary gridworld including a
        1-wall-thick border around the perimiter and a grid graph with random
        edges in the center.
        
        Consider the 'junction' squares

            (1,1), (1,3), ..., (1,w-1), (3,1), ..., (h-1,w-1).

        These squares form the nodes of a grid graph. This function
        constructs a random subgraph of this grid graph with each edge
        included independently with probability `edge_prob`, and returns the
        corresponding binary matrix.

        Parameters:

        * key : PRNGKey
                RNG state (will be consumed).

        Returns:

        * grid : bool[height, width]
                The binary grid (True indicates a wall, False indicates a
                path).
        """
        H = self.height // 2
        W = self.width // 2
        grid = jnp.ones((self.height, self.width), dtype=bool)

        # junctions
        grid = grid.at[1:-1:2,1:-1:2].set(False)
        
        # edges
        kh, kv = jax.random.split(key)
        grid = grid.at[2:-2:2,1:-1:2].set(~jax.random.bernoulli(
            key=kh,
            p=self.edge_prob,
            shape=(H-1, W),
        ))
        grid = grid.at[1:-1:2,2:-2:2].set(~jax.random.bernoulli(
            key=kv,
            p=self.edge_prob,
            shape=(H, W-1),
        ))

        return grid


@struct.dataclass
class NoiseMazeGenerator(MazeGenerator):
    """
    Generate noise mazes (with walls locations determined by Perlin noise).

    Parameters:

    * height : int (>= 3)
        Maze height (number of rows in grid).
    * width : int (>= 3)
        Maze width (number of columns in grid).
    * wall_threshold : float (between -1.0 and 1.0, default 0.25)
        The noise threshold above which a wall spawns.
    * cell_size : int (>= 2, default 2)
        Width and height of the gradient grid for the noise (in the case of
        multiple octaves, this is the size of the largest grid, and should be
        repeatedly divisible by two).
    """
    wall_threshold: float = 0.25
    cell_size : int = 2
    num_octaves : int = 1


    def __post_init__(self):
        assert self.height >= 3, "height must be at least 3"
        assert self.width >= 3, "width must be at least 3"
        assert -1.0 <= self.wall_threshold <= 1.0, "invalid threshold"
        assert self.cell_size >= 1, "cell size must be at least 2"
        assert self.num_octaves >= 1, "num octaves must be positive"


    @functools.partial(jax.jit, static_argnames=('self',))
    def generate(self, key):
        """
        Generate a `height` by `width` binary gridworld including a
        1-wall-thick border around the perimiter and an arrangement of walls
        in the center determined by Perlin noise.

        Parameters:

        * key : PRNGKey
                RNG state (will be consumed)

        Returns:

        * grid : bool[height, width]
                the binary grid (True indicates a wall, False indicates a path)
        """
        # make noise size a multiple of cell_size larger than the interior
        interior_height = self.height - 2
        interior_width = self.width - 2
        
        def next_multiple(number, divisor):
            quotient = (number // divisor) + ((number % divisor) > 0)
            next_multiple = quotient * divisor
            return next_multiple, quotient
        noise_height, num_cols = next_multiple(interior_height, self.cell_size)
        noise_width, num_rows = next_multiple(interior_width, self.cell_size)

        # generate noise
        noise = noise_generation.generate_fractal_noise(
            key=key,
            height=noise_height,
            width=noise_width,
            base_num_rows=num_rows,
            base_num_cols=num_cols,
            num_octaves=self.num_octaves,
        )

        # determine walls location in interior
        interior_walls = (
            noise[:interior_height,:interior_width] > self.wall_threshold
        )

        # build the grid
        grid = jnp.ones((self.height, self.width), dtype=bool)
        grid = grid.at[1:-1,1:-1].set(interior_walls)

        return grid


@struct.dataclass
class BlockMazeGenerator(MazeGenerator):
    """
    Generate block mazes (mazes with walls placed independently at random).

    Parameters:

    * height : int (>= 3)
            Maze height (number of rows in grid).
    * width : int (>= 3)
            Maze width (number of columns in grid).
    * wall_prob : float (probability, default 0.25)
    """
    wall_prob : float = 0.25


    def __post_init__(self):
        assert self.height >= 3, "height must be at least 3"
        assert self.width >= 3, "width must be at least 3"
        assert 0.0 <= self.wall_prob <= 1.0, "wall_prob must be a probability"


    @functools.partial(jax.jit, static_argnames=('self',))
    def generate(self, key):
        """
        Generate a `height` by `width` binary gridworld including a
        1-wall-thick border around the perimeter and randomly placed walls
        within the interior.

        Parameters:

        * key : PRNGKey
                Unused (here for compatibility with other methods)
        Returns:

        * grid : bool[height, width]
                The binary grid (True indicates a wall, False indicates a
                path)
        """
        grid = jnp.ones((self.height, self.width), dtype=bool)

        grid = grid.at[1:-1,1:-1].set(jax.random.bernoulli(
            key=key,
            p=self.wall_prob,
            shape=(self.height-2, self.width-2),
        ))

        return grid


@struct.dataclass
class OpenMazeGenerator(MazeGenerator):
    """
    Generate open 'mazes' (mazes with no walls actually...).

    Parameters:

    * height : int (>= 3)
            Maze height (number of rows in grid).
    * width : int (>= 3)
            Maze width (number of columns in grid).
    """


    def __post_init__(self):
        assert self.height >= 3, "height must be at least 3"
        assert self.width >= 3, "width must be at least 3"


    @functools.partial(jax.jit, static_argnames=('self',))
    def generate(self, key):
        """
        Generate a `height` by `width` binary gridworld including a
        1-wall-thick border around the perimeter and an empry interior.

        Parameters:

        * key : PRNGKey
                Unused (here for compatibility with other methods)
        Returns:

        * grid : bool[height, width]
                The binary grid (True indicates a wall, False indicates a
                path)
        """
        # empty gridworld
        grid = jnp.zeros((self.height, self.width), dtype=bool)
        # fill in border
        grid = grid.at[::self.height-1,:].set(True)
        grid = grid.at[:,::self.width-1].set(True)

        return grid


# # # 
# SOLVING MAZES

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

