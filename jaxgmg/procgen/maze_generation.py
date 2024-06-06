"""
JAX-compatible functions for generating and solving different kinds of mazes.
"""

import functools
import jax
import jax.numpy as jnp
import einops
from flax import struct

from jaxgmg.procgen import noise_generation


def get_generator_class(name):
    """
    Maps a string descriptor to a maze generator struct class.

    Parameters:

    * name : str ("tree", "edges", "blocks", "noise", or "open")
            The name of the generator.

    Returns:

    * MazeGenerator : subclass of MazeGenerator
            The class of the maze generator.
    """

    if name.lower() == "tree":
        return TreeMazeGenerator
    elif name.lower() == "edges":
        return EdgeMazeGenerator
    elif name.lower() == "blocks":
        return BlockMazeGenerator
    elif name.lower() == "noise":
        return NoiseMazeGenerator
    elif name.lower() == "open":
        return OpenMazeGenerator
    else:
        raise ValueError(f"Unknown maze generator {name!r}")


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

    @functools.partial(jax.jit, static_argnames=('self',))
    def generate(self, key):
        """
        Generate a `height` by `width` binary gridworld with a 1-wall thick
        border and a maze of some kind in the interior.
        """
        raise NotImplementedError


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
    Generate noise mazes (with walls locations determined by Perlin noise
    or associated fractal noise).

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


