"""
Parameterised environment and level generator for lava land problem.
Key components are as follows.

Structs:

* The `Level` struct represents a particular world layout with trees, lava,
  and cheese/mouse spawn positions.
* The `EnvState` struct represents a specific dynamic state of the
  environment.

Classes:

* `Env` class, provides `reset`, `step`, and `render` methods in a
  gymnax-style interface (see `base` module for specifics of the interface).
* `LevelGenerator` class, provides `sample` method for randomly sampling a
  level.
* `LevelParser` class, provides a `parse` and `parse_batch` method for
  designing Level structs based on ASCII depictions.
"""

import enum
import functools

import jax
import jax.numpy as jnp
import einops
from flax import struct

import chex
from jaxtyping import PyTree

from jaxgmg.procgen import maze_generation as mg
from jaxgmg.procgen import maze_solving
from jaxgmg.procgen import noise_generation
from jaxgmg.environments import base


@struct.dataclass
class Level(base.Level):
    """
    Represent a particular environment layout:

    * wall_map : bool[h, w]
            Maze layout (True = wall here)
    * lava_map : bool[h, w]
            Lava layout (True = lava here)
    * cheese_pos : index[2]
            Coordinates of cheese position (index into `wall_map`)
    * initial_mouse_pos : index[2]
            Coordinates of initial mouse position (index into `wall_map`)
    """
    wall_map: chex.Array
    lava_map: chex.Array
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


@struct.dataclass
class Observation(base.Observation):
    """
    Observation for partially observable Maze environment.

    * image : bool[h, w, c] or float[h, w, rgb]
            The contents of the state. Comes in one of two formats:
            * Boolean: a H by W by C bool array where each channel represents
              the presence of one type of thing (walls, lava, mouse, cheese).
            * Pixels: an D.H by D.W by 3 array of RGB float values where each
              D by D tile corresponds to one grid square. (D is level of
              detail.)

    Observations come in one of two formats:

    * Pixels: an 8H by 8W by 3 array of RGB float values where each 8 by 8
      tile corresponds to one grid square.
    """
    image: chex.Array


class Channel(enum.IntEnum):
    """
    The observations returned by the environment are an `h` by `w` by
    `channel` Boolean array, where the final dimensions 0 through 4
    indicate the following:

    * `WALL`:   True in the locations where there is a wall.
    * `LAVA`:   True in the locations where there is lava.
    * `MOUSE`:  True in the one location the mouse occupies.
    * `CHEESE`: True in the one location the cheese occupies.
    """
    WALL    = 0
    LAVA    = 1
    MOUSE   = 2
    CHEESE  = 3
    

class Action(enum.IntEnum):
    """
    The environment has a discrete action space of size 4 with the following
    meanings.
    """
    MOVE_UP     = 0
    MOVE_LEFT   = 1
    MOVE_DOWN   = 2
    MOVE_RIGHT  = 3


class Env(base.Env):
    """
    Lava Land environment.

    In this environment the agent controls a mouse navigating a grid-based
    world. The mouse must must navigate the world to the cheese. Sometimes,
    the world contains lava tiles.

    There are four available actions which deterministically move the mouse
    one grid square up, right, down, or left respectively.
    * If the mouse would hit a wall it remains in place.
    * If the mouse hits the cheese, the agent gains reward and the episode
      ends.
    * If the mouse hits lava, it gets a negative reward (TODO: dies?)
    """


    @property
    def num_actions(self) -> int:
        return len(Action)


    def obs_type( self, level: Level) -> PyTree[jax.ShapeDtypeStruct]:
        # TODO: only works for boolean observations...
        H, W = level.wall_map.shape
        C = len(Channel)
        return Observation(
            image=jax.ShapeDtypeStruct(
                shape=(H, W, C),
                dtype=bool,
            ),
        )


    def _reset(
        self,
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
    ) -> tuple[
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

        # check if mouse is on lava
        on_lava = state.level.lava_map[
            state.mouse_pos[0],
            state.mouse_pos[1],
        ]

        # reward and done
        reward = got_cheese.astype(float) - on_lava.astype(float)
        done = state.got_cheese

        return (
            state,
            reward,
            done,
            {},
        )

    
    @functools.partial(jax.jit, static_argnames=('self',))
    def _render_obs_bool(self, state: EnvState) -> Observation:
        """
        Return a boolean grid observation.
        """
        H, W = state.level.wall_map.shape
        C = len(Channel)
        image = jnp.zeros((H, W, C), dtype=bool)

        # render walls
        image = image.at[:, :, Channel.WALL].set(state.level.wall_map)
        
        # render lava
        image = image.at[:, :, Channel.LAVA].set(state.level.lava_map)

        # render mouse
        image = image.at[
            state.mouse_pos[0],
            state.mouse_pos[1],
            Channel.MOUSE,
        ].set(True)
        
        # render cheese
        image = image.at[
            state.level.cheese_pos[0],
            state.level.cheese_pos[1],
            Channel.CHEESE,
        ].set(~state.got_cheese)

        return Observation(image=image)


    @functools.partial(jax.jit, static_argnames=('self',))
    def _render_obs_rgb(
        self,
        state: EnvState,
        spritesheet: dict[str, chex.Array],
    ) -> Observation:
        """
        Return an RGB observation based on a grid of tiles from the given
        spritesheet.
        """
        # get the boolean grid representation of the state
        image_bool = self._render_obs_bool(state).image
        H, W, _C = image_bool.shape

        # find out, for each position, which object to render
        # (for each position pick the first true index top-down this list)
        sprite_priority_vector_grid = jnp.stack([
            # two objects
            image_bool[:, :, Channel.LAVA] & image_bool[:, :, Channel.MOUSE],
            # one object
            image_bool[:, :, Channel.WALL],
            image_bool[:, :, Channel.LAVA],
            image_bool[:, :, Channel.MOUSE],
            image_bool[:, :, Channel.CHEESE],
            # no objects, 'default' (always true)
            jnp.ones((H, W), dtype=bool),
        ])
        chosen_sprites = jnp.argmax(sprite_priority_vector_grid, axis=0)

        # put the corresponding sprite into each square
        spritemap = jnp.stack([
            # two objects
            spritesheet['MOUSE_ON_LAVA'],
            # one object
            spritesheet['TREE'],
            spritesheet['LAVA'],
            spritesheet['MOUSE'],
            spritesheet['CHEESE'],
            # no objects
            spritesheet['GRASS'],
        ])[chosen_sprites]

        image_rgb = einops.rearrange(
            spritemap,
            'h w th tw rgb -> (h th) (w tw) rgb',
        )
        return Observation(image=image_rgb)
    

    @functools.partial(jax.jit, static_argnames=('self',))
    def _render_state_bool(
        self,
        state: EnvState,
    ) -> chex.Array:
        return self._render_obs_bool(state).image


    @functools.partial(jax.jit, static_argnames=('self',))
    def _render_state_rgb(
        self,
        state: EnvState,
        spritesheet: dict[str, chex.Array],
    ) -> chex.Array:
        return self._render_obs_rgb(state, spritesheet).image


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

        For Lava Land, the optimal policy is to take the shortest path to the
        cheese that also avoids lava tiles. If there is no such path, the
        best one can do is simply avoid lava and get a score of zero (it is
        never worth crossing lava to get to the cheese, this will always lead
        to negative net reward).

        Parameters:

        * level : Level
                The level to compute the optimal value for.
        * discount_rate : float
                The discount rate to apply in the formula for computing
                return.
        * The output also depends on the environment's reward function, which
          depends on `self.penalize_time` and `self.max_steps_in_episode`.

        Notes:

        * In the rare case of a level in which the mouse spawns surrounded
          by immediately adjacent lava tiles in all four directions, the
          optimal value is actually negative (because there is no 'stay'
          action) but this function will return 0 for simplicity.
        * With a steep discount rate or long episodes, this algorithm might
          run into minor numerical issues where small contributions to the
          return from late into the episode are lost.
        * For VERY large mazes, solving the level will be pretty slow.

        TODO:

        * Solving the mazes currently uses all pairs shortest paths
          algorithm, which is not efficient enough to work for very large
          mazes. If we wanted to solve very large mazes, we could by changing
          to a single source shortest path algorithm.
        * Supporting a generalisation 
        * Support computing the return from arbitrary states. This would be
          not very difficult, requires taking note of `got_cheese` flag
          and current time, and computing distance from mouse's current
          position rather than initial position.
        """
        # compute distance between mouse and cheese
        dist = maze_solving.maze_distances(level.wall_map | level.lava_map)
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
    Level generator for Lava Land environment. Given some maze configuration
    parameters and cheese location parameter, provides a `sample` method that
    generates a random level.

    * height : int,(>= 3, odd)
            the number of rows in the grid representing the maze
            (including top and bottom boundary rows)
    * width : int (>= 3, odd)
            the number of columns in the grid representing the maze
            (including left and right boundary rows)
    * maze_generator : maze_generation.MazeGenerator
            Provides the maze generation method to use (see module
            `maze_generation` for details).
            The default is a tree maze generator using Kruskal's algorithm.
    * lava_threshold : float (-1.0 to +1.0, default -0.25):
            unoccupied tiles will spawn lava where perlin noise falls below
            this threshold (-1.0 means never, +1.0 means always)
    """
    height: int = 13
    width: int = 13
    maze_generator : mg.MazeGenerator = mg.TreeMazeGenerator()
    lava_threshold: int = -0.25
    
    def __post_init__(self):
        assert -1.0 <= self.lava_threshold <= 1.0


    @functools.partial(jax.jit, static_argnames=('self',))
    def sample(self, rng: chex.PRNGKey) -> Level:
        """
        Randomly generate a `Level` specification given the parameters
        provided in the constructor of this generator object.
        """
        # construct a random maze
        rng_walls, rng = jax.random.split(rng)
        wall_map = self.maze_generator.generate(
            key=rng_walls,
            height=self.height,
            width=self.width,
        )

        # sample spawn positions by sampling from a list of coordinate pairs
        coords = einops.rearrange(
            jnp.indices((self.height, self.width)),
            'c h w -> (h w) c',
        )
        no_wall_map = ~wall_map
        no_wall_mask = no_wall_map.flatten()
        
        # spawn cheese in some valid position
        rng_spawn_cheese, rng = jax.random.split(rng)
        cheese_pos = jax.random.choice(
            key=rng_spawn_cheese,
            a=coords,
            axis=0,
            p=no_wall_mask,
        )
        
        # spawn mouse in some remaining valid position
        no_cheese_map = jnp.ones_like(wall_map).at[
            cheese_pos[0],
            cheese_pos[1],
        ].set(False)

        rng_spawn_mouse, rng = jax.random.split(rng)
        initial_mouse_pos = jax.random.choice(
            key=rng_spawn_mouse,
            a=coords,
            axis=0,
            p=no_wall_mask & no_cheese_map.flatten(),
        )

        # spawn lava independently in each remaining position
        no_mouse_map = jnp.ones_like(wall_map).at[
            initial_mouse_pos[0],
            initial_mouse_pos[1],
        ].set(False)

        rng_lava, rng = jax.random.split(rng)
        noise_height = max(4, 2**(self.height - 1).bit_length())
        noise_width = max(4, 2**(self.width - 1).bit_length())
        raw_lava_map = noise_generation.generate_perlin_noise(
            key=rng_lava,
            height=noise_height,
            width=noise_width,
            num_cols=noise_height//4,
            num_rows=noise_width//4,
        )[:self.height,:self.width] < self.lava_threshold
        lava_map = (
            raw_lava_map
            & no_wall_map
            & no_cheese_map
            & no_mouse_map
        )

        return Level(
            wall_map=wall_map,
            lava_map=lava_map,
            cheese_pos=cheese_pos,
            initial_mouse_pos=initial_mouse_pos,
        )


@struct.dataclass
class LevelParser(base.LevelParser):
    """
    Level parser for Lava Land environment. Given some parameters determining
    level shape, provides a `parse` method that converts an ASCII depiction
    of a level into a Level struct. Also provides a `parse_batch` method that
    parses a list of level strings into a single vectorised Level PyTree
    object.

    * height (int, >= 3):
            The number of rows in the grid representing the maze
            (including top and bottom boundary rows)
    * width (int, >= 3):
            The number of columns in the grid representing the maze
            (including left and right boundary rows)
    * char_map : optional, dict{str: int}
            The keys in this dictionary are the symbols the parser will look
            to define the location of the walls and each of the items. The
            default map is as follows:
            * The character '#' maps to `Channel.WALL`.
            * The character 'X' maps to `Channel.LAVA`.
            * The character '@' maps to `Channel.MOUSE`.
            * The character '*' maps to `Channel.CHEESE`.
            * The character '.' maps to `len(Channel)`, i.e. none of the
              above, representing the absence of an item.
    """
    height: int
    width: int
    char_map = {
        '#': Channel.WALL,
        'X': Channel.LAVA,
        '@': Channel.MOUSE,
        '*': Channel.CHEESE,
        '.': len(Channel), # PATH
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
        ... # . X * #
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
            lava_map=Array([
                [0,0,0,0,0],
                [0,0,0,0,0],
                [0,0,0,0,0],
                [0,0,1,0,0],
                [0,0,0,0,0],
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
        wall_map = (level_map == Channel.WALL)
        assert wall_map[0,:].all(), "top border incomplete"
        assert wall_map[:,0].all(), "left border incomplete"
        assert wall_map[-1,:].all(), "bottom border incomplete"
        assert wall_map[:,-1].all(), "right border incomplete"
        
        # extract lava map
        lava_map = (level_map == Channel.LAVA)
        
        # extract cheese position
        cheese_map = (level_map == Channel.CHEESE)
        assert cheese_map.sum() == 1, "there must be exactly one cheese"
        cheese_pos = jnp.concatenate(
            jnp.where(cheese_map, size=1)
        )

        # extract mouse spawn position
        mouse_spawn_map = (level_map == Channel.MOUSE)
        assert mouse_spawn_map.sum() == 1, "there must be exactly one mouse"
        initial_mouse_pos = jnp.concatenate(
            jnp.where(mouse_spawn_map, size=1)
        )

        return Level(
            wall_map=wall_map,
            lava_map=lava_map,
            cheese_pos=cheese_pos,
            initial_mouse_pos=initial_mouse_pos,
        )


