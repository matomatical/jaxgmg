"""
Parameterised environment and level generator for cheese in the corner
problem. Key components are as follows.

Structs:

* The `Level` struct represents a particular maze layout and cheese/mouse
  spawn position.
* The `EnvState` struct represents a specific dynamic state of the
  environment.

Classes:

* `Env` class, provides `reset`, `step`, and `render` methods in a
  gymnax-style interface (see `base` module for specifics of the interface).
* `LevelGenerator` class, provides `sample` method for randomly sampling a
  level.
* `LevelMutator` class, provides `mutate` method for mutating an existing
  level with configurable mutations.
* `LevelParser` class, provides a `parse` and `parse_batch` method for
  designing Level structs based on ASCII depictions.
"""

from typing import Tuple
import enum
import functools

import jax
import jax.numpy as jnp
import chex
import einops
from flax import struct

from jaxgmg.procgen import maze_generation
from jaxgmg.environments import base
from jaxgmg.environments import spritesheet


@struct.dataclass
class Level(base.Level):
    """
    Represent a particular environment layout:

    * wall_map : bool[h, w]
            Maze layout (True = wall)
    * cheese_pos : index[2]
            Coordinates of cheese position (index into `wall_map`)
    * initial_mouse_pos : index[2]
            Coordinates of initial mouse position (index into `wall_map`)
    """
    wall_map: chex.Array
    cheese_pos: chex.Array
    initial_mouse_pos: chex.Array


@struct.dataclass
class EnvState(base.EnvState):
    """
    Dynamic environment state within a particular level.

    * mouse_pos : index[2]
            Current coordinates of the mouse. Initialised to
            `level.initial_mouse_pos`.
    * got_cheese : bool
            Whether the mouse has already gotten the cheese.
    """
    mouse_pos: chex.Array
    got_cheese: bool


class Env(base.Env):
    """
    Cheese in the Corner environment.

    In this environment the agent controls a mouse navigating a grid-based
    maze. The mouse must must navigate the maze to the cheese, normally
    located near the top left corner.

    There are four available actions which deterministically move the mouse
    one grid square up, right, down, or left respectively.
    * If the mouse would hit a wall it remains in place.
    * If the mouse hits the cheese, the agent gains reward and the episode
      ends.

    Observations come in one of two formats:

    * Boolean: a H by W by C bool array where each channel represents the
      presence of one type of thing (walls, mouse, cheese).
    * Pixels: an 8H by 8W by 3 array of RGB float values where each 8 by 8
      tile corresponds to one grid square.
    """
    class Action(enum.IntEnum):
        """
        The environment has a discrete action space of size 4 with the following
        meanings.
        """
        MOVE_UP     = 0
        MOVE_LEFT   = 1
        MOVE_DOWN   = 2
        MOVE_RIGHT  = 3


    @property
    def num_actions(self) -> int:
        return len(Env.Action)


    class Channel(enum.IntEnum):
        """
        The observations returned by the environment are an `h` by `w` by
        `channel` Boolean array, where the final dimensions 0 through 4
        indicate the following:

        * `WALL`:   True in the locations where there is a wall.
        * `MOUSE`:  True in the one location the mouse occupies.
        * `CHEESE`: True in the one location the cheese occupies.
        """
        WALL    = 0
        MOUSE   = 1
        CHEESE  = 2

    
    def _reset(
        self,
        rng: chex.PRNGKey,
        level: Level,
    ) -> EnvState:
        return EnvState(
            mouse_pos=level.initial_mouse_pos,
            got_cheese=False,
            level=level,
            steps=0,
            done=False,
        )
        

    def _step(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: int,
    ) -> Tuple[
        EnvState,
        float,
        bool,
        dict,
    ]:
        # update mouse position
        steps = jnp.array((
            (-1,  0),   # up
            ( 0, -1),   # left
            (+1,  0),   # down
            ( 0, +1),   # right
        ))
        ahead_pos = state.mouse_pos + steps[action]
        hit_wall = state.level.wall_map[ahead_pos[0], ahead_pos[1]]
        state = state.replace(
            mouse_pos=jax.lax.select(
                hit_wall,
                state.mouse_pos,
                ahead_pos,
            )
        )

        # check if mouse got to cheese
        got_cheese = (state.mouse_pos == state.level.cheese_pos).all()
        state = state.replace(got_cheese=state.got_cheese | got_cheese)

        # reward and done
        reward = got_cheese.astype(float)
        done = state.got_cheese

        return (
            state,
            reward,
            done,
            {},
        )

    
    @functools.partial(jax.jit, static_argnames=('self',))
    def _get_obs_bool(self, state: EnvState) -> chex.Array:
        """
        Return a boolean grid observation.
        """
        H, W = state.level.wall_map.shape
        C = len(Env.Channel)
        obs = jnp.zeros((H, W, C), dtype=bool)

        # render walls
        obs = obs.at[:, :, Env.Channel.WALL].set(state.level.wall_map)

        # render mouse
        obs = obs.at[
            state.mouse_pos[0],
            state.mouse_pos[1],
            Env.Channel.MOUSE,
        ].set(True)
        
        # render cheese
        obs = obs.at[
            state.level.cheese_pos[0],
            state.level.cheese_pos[1],
            Env.Channel.CHEESE,
        ].set(~state.got_cheese)

        return obs


    @functools.partial(jax.jit, static_argnames=('self',))
    def _get_obs_rgb(self, state: EnvState) -> chex.Array:
        """
        Return an RGB observation, which is also a human-interpretable image.
        """
        # get the boolean grid representation of the state
        obs = self._get_obs_bool(state)
        H, W, _C = obs.shape

        # find out, for each position, which object to render
        # (for each position pick the first true index top-down this list)
        sprite_priority_vector_grid = jnp.stack([
            # one object
            obs[:, :, Env.Channel.WALL],
            obs[:, :, Env.Channel.MOUSE],
            obs[:, :, Env.Channel.CHEESE],
            # no objects, 'default' (always true)
            jnp.ones((H, W), dtype=bool),
        ])
        chosen_sprites = jnp.argmax(sprite_priority_vector_grid, axis=0)

        # put the corresponding sprite into each square
        spritemap = jnp.stack([
            # one object
            spritesheet.WALL,
            spritesheet.MOUSE,
            spritesheet.CHEESE,
            # no objects
            spritesheet.PATH,
        ])[chosen_sprites]
        image = einops.rearrange(
            spritemap,
            'h w th tw rgb -> (h th) (w tw) rgb',
        )

        return image
    

    @functools.partial(jax.jit, static_argnames=('self',))
    def optimal_value(
        self,
        level: Level,
        discount_rate: float,
    ) -> float:
        """
        Compute the optimal return from a given level (initial state) for a
        given discount rate. Respects time penalties to reward and max
        episode length.

        Parameters:

        * level : Level
                The level to compute the optimal value for. Depends on the
                wall configuration, the initial agent location, and the
                location of each key and chest.
        * discount_rate : float
                The discount rate to apply in the formula for computing
                return.
        * The output also depends on the environment's reward function, which
          depends on `self.penalize_time` and `self.max_steps_in_episode`.

        Notes:

        * With a steep discount rate or long episodes, this algorithm might
          run into minor numerical issues where small contributions to the
          return from late into the episode are lost.
        * For VERY large mazes, solving the level will be pretty slow.

        TODO:

        * Solving the mazes currently uses all pairs shortest paths
          algorithm, which is not efficient enough to work for very large
          mazes. If we wanted to solve very large mazes, we could by changing
          to a single source shortest path algorithm.
        * Support computing the return from arbitrary states. This would be
          not very difficult, requires taking note of `got_cheese` flag
          and current time, and computing distance from mouse's current
          position rather than initial position.
        """
        # compute distance between mouse and cheese
        dist = maze_generation.maze_distances(level.wall_map)
        optimal_dist = dist[
            level.initial_mouse_pos[0],
            level.initial_mouse_pos[1],
            level.cheese_pos[0],
            level.cheese_pos[1],
        ]

        # reward when we get the cheese is 1
        reward = 1
        
        # maybe we apply an optional time penalty
        penalized_reward = jnp.where(
            self.penalize_time,
            (1.0 - 0.9 * optimal_dist / self.max_steps_in_episode) * reward,
            reward,
        )
        
        # mask out rewards beyond the end of the episode
        episodes_still_valid = (optimal_dist < self.max_steps_in_episode)
        valid_reward = penalized_reward * episodes_still_valid

        # discount the reward
        discounted_reward = (discount_rate ** optimal_dist) * valid_reward

        return discounted_reward


@struct.dataclass
class LevelGenerator(base.LevelGenerator):
    """
    Level generator for Cheese in the Corner environment. Given some maze
    configuration parameters and cheese location parameter, provides a
    `sample` method that generates a random level.

    * height : int,(>= 3, odd)
            the number of rows in the grid representing the maze
            (including top and bottom boundary rows)
    * width : int (>= 3, odd)
            the number of columns in the grid representing the maze
            (including left and right boundary rows)
    * layout : str ('tree', 'bernoulli', 'blocks', 'noise', or 'open')
            specifies the maze generation method to use (see module
            `maze_generation` for details)
    * corner_size : int (>=1, <=width, <=height):
            the cheese will spawn within a square of this width located in
            the top left corner.
    """
    height: int = 13
    width: int = 13
    layout : str = 'tree'
    corner_size: int = 1
    
    def __post_init__(self):
        # validate layout
        assert self.layout in {'tree', 'edges', 'blocks', 'open', 'noise'}
        # validate dimensions
        assert self.height >= 3
        assert self.width >= 3
        if self.layout == 'tree' or self.layout == 'edges':
            assert self.height % 2 == 1, "height must be odd for this layout"
            assert self.width % 2 == 1,  "width must be odd for this layout"
        # validate corner size
        assert self.corner_size >= 1
        assert self.corner_size <= self.width - 2


    @functools.partial(jax.jit, static_argnames=('self',))
    def sample(self, rng: chex.PRNGKey) -> Level:
        """
        Randomly generate a `Level` specification given the parameters
        provided in the constructor of this generator object.
        """
        # construct a random maze
        rng_walls, rng = jax.random.split(rng)
        wall_map = maze_generation.get_generator_function(self.layout)(
            key=rng_walls,
            h=self.height,
            w=self.width,
        )

        # sample spawn positions by sampling from a list of coordinate pairs
        coords = einops.rearrange(
            jnp.indices((self.height, self.width)),
            'c h w -> (h w) c',
        )
        no_wall = ~wall_map.flatten()
        
        # cheese spawn in top left corner region
        in_corner = jnp.logical_and(
            coords[:, 0] < self.corner_size + 1, # first `corner_size+1` rows
            coords[:, 1] < self.corner_size + 1, # first `corner_size+1` cols
            # (+1s account for boundary which is masked out by `no_wall` mask)
        ).flatten()

        rng_spawn_cheese, rng = jax.random.split(rng)
        cheese_pos = coords[jax.random.choice(
            key=rng_spawn_cheese,
            a=coords.shape[0],
            shape=(),
            p=no_wall & in_corner,
        )]
        
        # mouse spawn in some remaining valid position
        no_cheese = jnp.ones_like(wall_map).at[
            cheese_pos[0],
            cheese_pos[1],
        ].set(False).flatten()

        rng_spawn_mouse, rng = jax.random.split(rng)
        initial_mouse_pos = coords[jax.random.choice(
            key=rng_spawn_mouse,
            a=coords.shape[0],
            shape=(),
            p=no_wall & no_cheese,
        )]

        return Level(
            wall_map=wall_map,
            cheese_pos=cheese_pos,
            initial_mouse_pos=initial_mouse_pos,
        )


@struct.dataclass
class LevelMutator(base.LevelMutator):
    """
    Configurable level mutator. Provides a 'mutate' method that transforms a
    level into a slightly different level, with the configured mutation
    operations.

    Parameters:

    * prob_wall_spawn : float
            Probability that a given interior (non-border) wall will despawn
            during a mutation.
    * prob_wall_despawn : float
            Probability that a given blank space (with no wall, also no
            cheese or mouse) will spawn a wall during a mutation.
    * mouse_scatter : bool
            If true, then each mutation will resample a new initial mouse
            position. If false, then the initial mouse position will only be
            changed if `max_mouse_steps > 0`.
    * max_mouse_steps : int (>= 0)
            If positive and `mouse_scatter` is false, then sample this many
            directions and apply them to the initial mouse position.
            (If the position would move into the cheese or a wall, the move
            is not applied).
    * cheese_scatter : bool
            If true, then each mutation will resample a new cheese position
            within the top left `corner_size` by `corner_size` region of the
            map. If false, then the cheese position will only be changed if
            `max_cheese_steps > 0`.
    * max_cheese_steps : int (>= 0)
            If positive and `cheese_scatter` is false, then sample this many
            directions and apply them to the cheese position.
            (If the position would move into the mouse, into a wall, or out
            of the top left `corner_size` by `corner_size` region, the move
            is not applied).
    * corner_size : int (>= 1)
            The size of the region in the top left corner to which the cheese
            is confined. Starts from 1 and doesn't count the border squares.
            For example, a value of 1 means the cheese will be confined to
            the position (1, 1), or a value of 2 means the cheese will be
            confined to the positions (1, 1), (1, 2), (2, 1), (2, 2).
    
    Notes:

    * The wall mutations are applied first, then the mouse position
      mutations, then the cheese position mutations.
    * As long as the input level initially has distinct locations for all
      walls, the cheese, and the initial mouse location, and the cheese is in
      the requested corner region, the mutation function should work as
      expected. However, if these invariants are violated, then it's possible
      that the mutation will crash or return an invalid level.
      * For example, this could happen if the entire corner region is covered
        by walls and so there is nowhere to scatter the cheese.
    """
    prob_wall_spawn: float
    prob_wall_despawn: float
    mouse_scatter: bool
    max_mouse_steps: int
    cheese_scatter: bool
    max_cheese_steps: int
    corner_size: int


    @functools.partial(jax.jit, static_argnames=('self',))
    def mutate(self, rng: chex.PRNGKey, level: Level):
        # toggle walls
        if self.prob_wall_spawn > 0 or self.prob_wall_despawn > 0:
            rng_walls, rng = jax.random.split(rng)
            level = self._toggle_walls(rng_walls, level)

        # moving mouse spawn location
        if self.mouse_scatter:
            rng_mouse, rng = jax.random.split(rng)
            level = self._scatter_mouse(rng_mouse, level)
        elif self.max_mouse_steps > 0:
            rng_mouse, rng = jax.random.split(rng)
            def _step(level, rng_step):
                return self._step_mouse(rng_step, level), None
            level, _ = jax.lax.scan(
                _step,
                level,
                jax.random.split(rng_mouse, self.max_mouse_steps),
                unroll=True,
            )

        # move cheese location
        if self.cheese_scatter:
            rng_cheese, rng = jax.random.split(rng)
            level = self._scatter_cheese(rng_cheese, level)
        elif self.max_cheese_steps > 0:
            rng_cheese, rng = jax.random.split(rng)
            def _step(level, rng_step):
                return self._step_cheese(rng_step, level), None
            level, _ = jax.lax.scan(
                _step,
                level,
                jax.random.split(rng_cheese, self.max_cheese_steps),
                unroll=True,
            )

        return level


    def _toggle_walls(self, rng, level):
        # decide which walls to toggle
        h, w = level.wall_map.shape
        prob_wall_toggle = jnp.where(
            level.wall_map[1:-1,1:-1],
            self.prob_wall_despawn,
            self.prob_wall_spawn,
        )
        walls_to_toggle = jax.random.bernoulli(
            key=rng,
            p=prob_wall_toggle,
            shape=(h-2, w-2),
        )
        walls_to_toggle = jnp.pad(
            walls_to_toggle,
            pad_width=1,
            constant_values=False,
        )
        
        # don't toggle the place where the cheese or the mouse is
        walls_to_toggle = walls_to_toggle.at[
            (level.cheese_pos[0], level.initial_mouse_pos[0]),
            (level.cheese_pos[1], level.initial_mouse_pos[1]),
        ].set(False)

        # toggle!
        new_wall_map = jnp.logical_xor(level.wall_map, walls_to_toggle)

        return level.replace(wall_map=new_wall_map)

    
    def _scatter_mouse(self, rng, level):
        coords = einops.rearrange(
            jnp.indices(level.wall_map.shape),
            'c h w -> (h w) c',
        )

        # avoid walls and cheese
        no_wall = ~level.wall_map.flatten()
        no_cheese = jnp.ones_like(level.wall_map).at[
            level.cheese_pos[0],
            level.cheese_pos[1],
        ].set(False).flatten()
        
        # resample a new mouse position
        new_initial_mouse_pos = coords[jax.random.choice(
            key=rng,
            a=coords.shape[0],
            shape=(),
            p=no_wall & no_cheese,
        )]

        return level.replace(initial_mouse_pos=new_initial_mouse_pos)


    def _scatter_cheese(self, rng, level):
        coords = einops.rearrange(
            jnp.indices(level.wall_map.shape),
            'c h w -> (h w) c',
        )

        # identify the top corner region
        in_corner = jnp.logical_and(
            coords[:, 0] < self.corner_size + 1, # first `corner_size+1` rows
            coords[:, 1] < self.corner_size + 1, # first `corner_size+1` cols
            # (+1s account for boundary which is masked out by `no_wall` mask)
        ).flatten()

        # avoid walls and mouse
        no_wall = ~level.wall_map.flatten()
        no_mouse = jnp.ones_like(level.wall_map).at[
            level.initial_mouse_pos[0],
            level.initial_mouse_pos[1],
        ].set(False).flatten()
        
        # resample a new cheese position
        new_cheese_pos = coords[jax.random.choice(
            key=rng,
            a=coords.shape[0],
            shape=(),
            p=no_wall & no_mouse & in_corner,
        )]

        return level.replace(cheese_pos=new_cheese_pos)


    def _step_mouse(self, rng, level):
        # choose a random direction
        steps = jnp.array((
            (-1,  0),   # up
            ( 0, -1),   # left
            (+1,  0),   # down
            ( 0, +1),   # right
        ))
        direction = jax.random.choice(rng, 4)

        # check it's clear of obstacles
        try_pos = level.initial_mouse_pos + steps[direction]
        hit_wall = level.wall_map[try_pos[0], try_pos[1]]
        hit_cheese = (try_pos == level.cheese_pos).all()

        # if so, move
        new_initial_mouse_pos = jax.lax.select(
            hit_wall | hit_cheese,
            level.initial_mouse_pos,
            try_pos,
        )

        return level.replace(initial_mouse_pos=new_initial_mouse_pos)


    def _step_cheese(self, rng, level):
        # choose a random direction
        steps = jnp.array((
            (-1,  0),   # up
            ( 0, -1),   # left
            (+1,  0),   # down
            ( 0, +1),   # right
        ))
        direction = jax.random.choice(rng, 4)

        # check it's clear of obstacles and inside the corner
        try_pos = level.cheese_pos + steps[direction]
        hit_wall = level.wall_map[try_pos[0], try_pos[1]]
        hit_mouse = (try_pos == level.initial_mouse_pos).all()
        beyond_corner = (try_pos > self.corner_size).any()

        # if so, move
        new_cheese_pos = jax.lax.select(
            hit_wall | hit_mouse | beyond_corner,
            level.cheese_pos,
            try_pos,
        )
        
        return level.replace(cheese_pos=new_cheese_pos)


@struct.dataclass
class LevelParser:
    """
    Level parser for Cheese in the Corner environment. Given some parameters
    determining level shape, provides a `parse` method that converts an ASCII
    depiction of a level into a Level struct. Also provides a `parse_batch`
    method that parses a list of level strings into a single vectorised Level
    PyTree object.

    * height (int, >= 3, odd):
            The number of rows in the grid representing the maze
            (including top and bottom boundary rows)
    * width (int, >= 3, odd):
            The number of columns in the grid representing the maze
            (including left and right boundary rows)
    * char_map : optional, dict{str: int}
            The keys in this dictionary are the symbols the parser will look
            to define the location of the walls and each of the items. The
            default map is as follows:
            * The character '#' maps to `Env.Channel.WALL`.
            * The character '@' maps to `Env.Channel.MOUSE`.
            * The character '*' maps to `Env.Channel.CHEESE`.
            * The character '.' maps to `len(Env.Channel)`, i.e. none of the
              above, representing the absence of an item.
    """
    height: int
    width: int
    char_map = {
        '#': Env.Channel.WALL,
        '@': Env.Channel.MOUSE,
        '*': Env.Channel.CHEESE,
        '.': len(Env.Channel), # PATH
    }


    def parse(self, level_str):
        """
        Convert an ASCII string depiction of a level into a Level struct.
        For example:

        >>> p = LevelParser(height=5, width=5)
        >>> p.parse('''
        ... # # # # #
        ... # . . . #
        ... # @ # . #
        ... # . . * #
        ... # # # # #
        ... ''')
        Level(
            wall_map=Array([
                [1,1,1,1,1],
                [1,0,0,0,1],
                [1,0,1,0,1],
                [1,0,0,0,1],
                [1,1,1,1,1],
            ], dtype=bool),
            cheese_pos=Array([3, 3], dtype=int32),
            initial_mouse_pos=Array([2, 1], dtype=int32),
        )
        """
        # parse into grid of IntEnum elements
        level_grid = [
            [self.char_map[e] for e in line.split()]
            for line in level_str.strip().splitlines()
        ]
        assert len(level_grid) == self.height, "wrong height"
        assert all([len(r) == self.width for r in level_grid]), "wrong width"
        level_map = jnp.asarray(level_grid)
        
        # extract wall map
        wall_map = (level_map == Env.Channel.WALL)
        assert wall_map[0,:].all(), "top border incomplete"
        assert wall_map[:,0].all(), "left border incomplete"
        assert wall_map[-1,:].all(), "bottom border incomplete"
        assert wall_map[:,-1].all(), "right border incomplete"
        
        # extract cheese position
        cheese_map = (level_map == Env.Channel.CHEESE)
        assert cheese_map.sum() == 1, "there must be exactly one cheese"
        cheese_pos = jnp.concatenate(
            jnp.where(cheese_map, size=1)
        )

        # extract mouse spawn position
        mouse_spawn_map = (level_map == Env.Channel.MOUSE)
        assert mouse_spawn_map.sum() == 1, "there must be exactly one mouse"
        initial_mouse_pos = jnp.concatenate(
            jnp.where(mouse_spawn_map, size=1)
        )

        return Level(
            wall_map=wall_map,
            cheese_pos=cheese_pos,
            initial_mouse_pos=initial_mouse_pos,
        )


    def parse_batch(self, level_strs):
        """
        Convert a list of ASCII string depiction of length `num_levels`
        into a vectorised `Level[num_levels]` PyTree. See `parse` method for
        the details of the string depiction.
        """
        levels = [self.parse(level_str) for level_str in level_strs]
        return Level(
            wall_map=jnp.stack([l.wall_map for l in levels]),
            cheese_pos=jnp.stack([l.cheese_pos for l in levels]),
            initial_mouse_pos=jnp.stack([l.initial_mouse_pos for l in levels]),
        )
