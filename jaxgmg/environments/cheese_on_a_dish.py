"""
Parameterised environment and level generator for cheese on a dish
problem. Key components are as follows.

Structs:

* The `Level` struct represents a particular maze layout and
  cheese/mouse/dish spawn position.
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
from jaxgmg.environments import base


@struct.dataclass
class Level(base.Level):
    """
    Represent a particular environment layout:

    * wall_map : bool[h, w]
            Maze layout (True = wall)
    * cheese_pos : index[2]
            Coordinates of cheese position (index into `wall_map`)
    * dish_pos : index[2]
            Coordinates of dish position (index into `wall_map`)
    * initial_mouse_pos : index[2]
            Coordinates of initial mouse position (index into `wall_map`)
    """
    wall_map: chex.Array
    cheese_pos: chex.Array
    dish_pos: chex.Array
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
    * got_dish : bool
            Whether the mouse has already gotten the dish.
    """
    mouse_pos: chex.Array
    got_cheese: bool
    got_dish: bool


class Env(base.Env):
    """
    Cheese on a Dish environment.

    In this environment the agent controls a mouse navigating a grid-based
    maze. The mouse must must navigate the maze to the cheese, normally
    co-located with a dish.

    There are four available actions which deterministically move the mouse
    one grid square up, right, down, or left respectively.
    * If the mouse would hit a wall it remains in place.
    * If the mouse hits the dish, the dish is removed.
    * If the mouse hits the cheese, the agent gains reward and the episode
      ends.

    Observations come in one of two formats:

    * Boolean: a H by W by C bool array where each channel represents the
      presence of one type of thing (walls, mouse, cheese, dish).
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


    def obs_type( self, level: Level) -> PyTree[jax.ShapeDtypeStruct]:
        H, W = level.wall_map.shape
        C = len(Env.Channel)
        return jax.ShapeDtypeStruct(
            shape=(H, W, C),
            dtype=bool,
        )


    class Channel(enum.IntEnum):
        """
        The observations returned by the environment are an `h` by `w` by
        `channel` Boolean array, where the final dimensions 0 through 4
        indicate the following:

        * `WALL`:   True in the locations where there is a wall.
        * `MOUSE`:  True in the one location the mouse occupies.
        * `CHEESE`: True in the one location the cheese occupies.
        * `DISH`:   True in the one location the dish occupies.
        """
        WALL    = 0
        MOUSE   = 1
        CHEESE  = 2
        DISH    = 3

    
    def _reset(
        self,
        level: Level,
    ) -> EnvState:
        return EnvState(
            mouse_pos=level.initial_mouse_pos,
            got_cheese=False,
            got_dish=False,
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
        
        # check if mouse got to dish
        got_dish = (state.mouse_pos == state.level.dish_pos).all()
        state = state.replace(got_dish=state.got_dish | got_dish)

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
        
        # render dish
        obs = obs.at[
            state.level.dish_pos[0],
            state.level.dish_pos[1],
            Env.Channel.DISH,
        ].set(~state.got_dish)

        return obs


    @functools.partial(jax.jit, static_argnames=('self',))
    def _get_obs_rgb(
        self,
        state: EnvState,
        spritesheet: dict[str, chex.Array],
    ) -> chex.Array:
        """
        Return an RGB observation based on a grid of tiles from the given
        spritesheet.
        """
        # get the boolean grid representation of the state
        obs = self._get_obs_bool(state)
        H, W, _C = obs.shape

        # find out, for each position, which object to render
        # (for each position pick the first true index top-down this list)
        sprite_priority_vector_grid = jnp.stack([
            # two objects
            obs[:, :, Env.Channel.CHEESE] & obs[:, :, Env.Channel.DISH],
            # one object
            obs[:, :, Env.Channel.WALL],
            obs[:, :, Env.Channel.MOUSE],
            obs[:, :, Env.Channel.CHEESE],
            obs[:, :, Env.Channel.DISH],
            # no objects, 'default' (always true)
            jnp.ones((H, W), dtype=bool),
        ])
        chosen_sprites = jnp.argmax(sprite_priority_vector_grid, axis=0)

        # put the corresponding sprite into each square
        spritemap = jnp.stack([
            # two objects
            spritesheet['CHEESE_ON_DISH'],
            # one object
            spritesheet['WALL'],
            spritesheet['MOUSE'],
            spritesheet['SMALL_CHEESE'],
            spritesheet['DISH'],
            # no objects
            spritesheet['PATH'],
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
                The level to compute the optimal value for.
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
        dist = maze_solving.maze_distances(level.wall_map)
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
    Level generator for Cheese on a Dish environment. Given some maze
    configuration parameters and cheese location parameter, provides a
    `sample` method that generates a random level.

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
    * max_cheese_radius : int (>=0)
            the cheese will spawn within this many steps away from the
            location of the dish.
    """
    height: int = 13
    width: int = 13
    maze_generator : mg.MazeGenerator = mg.TreeMazeGenerator()
    max_cheese_radius: int = 0
    
    def __post_init__(self):
        # validate cheese radius
        assert self.max_cheese_radius >= 0


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
        
        # mouse spawns in some random valid position
        no_wall = ~wall_map.flatten()

        rng_spawn_mouse, rng = jax.random.split(rng)
        initial_mouse_pos = jax.random.choice(
            key=rng_spawn_mouse,
            a=coords,
            axis=0,
            p=no_wall,
        )
        
        # dish spawns in some remaining valid position
        no_mouse = jnp.ones_like(wall_map).at[
            initial_mouse_pos[0],
            initial_mouse_pos[1],
        ].set(False).flatten()

        rng_spawn_dish, rng = jax.random.split(rng)
        dish_pos = jax.random.choice(
            key=rng_spawn_dish,
            a=coords,
            axis=0,
            p=no_wall & no_mouse,
        )

        # cheese spawns in some remaining valid position near the dish
        distance_to_dish = maze_solving.maze_distances(wall_map)[
            dish_pos[0],
            dish_pos[1],
        ]
        near_dish = (distance_to_dish <= self.max_cheese_radius).flatten()

        rng_spawn_cheese, rng = jax.random.split(rng)
        cheese_pos = jax.random.choice(
            key=rng_spawn_cheese,
            a=coords,
            axis=0,
            p=no_wall & no_mouse & near_dish,
        )

        return Level(
            wall_map=wall_map,
            initial_mouse_pos=initial_mouse_pos,
            dish_pos=dish_pos,
            cheese_pos=cheese_pos,
        )


@struct.dataclass
class LevelParser:
    """
    Level parser for Cheese on a Dish environment. Given some parameters
    determining level shape, provides a `parse` method that converts an ASCII
    depiction of a level into a Level struct. Also provides a `parse_batch`
    method that parses a list of level strings into a single vectorised Level
    PyTree object.

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
            * The character '#' maps to `Env.Channel.WALL`.
            * The character '@' maps to `Env.Channel.MOUSE`.
            * The character 'd' maps to `Env.Channel.DISH`.
            * The character 'c' maps to `Env.Channel.CHEESE`.
            * The character 'b' maps to `len(Env.Channel)`, i.e. none of the
              above, representing *both* the cheese and the dish.
            * The character '.' maps to `len(Env.Channel)+1`, i.e. none of
              the above, representing the absence of an item.
    """
    height: int
    width: int
    char_map = {
        '#': Env.Channel.WALL,
        '@': Env.Channel.MOUSE,
        'd': Env.Channel.DISH,
        'c': Env.Channel.CHEESE,
        'b': len(Env.Channel),   # BOTH
        '.': len(Env.Channel)+1, # PATH
        # TODO: I should switch these to using a standalone enum...
    }


    def parse(self, level_str):
        """
        Convert an ASCII string depiction of a level into a Level struct.
        For example:

        >>> p = LevelParser(height=5, width=5)
        >>> p.parse('''
        ... # # # # #
        ... # . . . #
        ... # @ # d #
        ... # . . c #
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
            dish_pos=Array([2, 3], dtype=int32),
            initial_mouse_pos=Array([2, 1], dtype=int32),
        )
        >>> p.parse('''
        ... # # # # #
        ... # . . @ #
        ... # . # # #
        ... # . . b #
        ... # # # # #
        ... ''')
        Level(
            wall_map=Array([
                [1,1,1,1,1],
                [1,0,0,0,1],
                [1,0,1,1,1],
                [1,0,0,0,1],
                [1,1,1,1,1],
            ], dtype=bool),
            cheese_pos=Array([3, 3], dtype=int32),
            dish_pos=Array([3, 3], dtype=int32),
            initial_mouse_pos=Array([1, 3], dtype=int32),
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
        cheese_map = (
            (level_map == Env.Channel.CHEESE)
            | (level_map == len(Env.Channel)) # both dish and cheese
        )
        assert cheese_map.sum() == 1, "there must be exactly one cheese"
        cheese_pos = jnp.concatenate(
            jnp.where(cheese_map, size=1)
        )
        
        # extract dish position
        dish_map = (
            (level_map == Env.Channel.DISH)
            | (level_map == len(Env.Channel)) # both dish and cheese
        )
        assert dish_map.sum() == 1, "there must be exactly one dish"
        dish_pos = jnp.concatenate(
            jnp.where(dish_map, size=1)
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
            dish_pos=dish_pos,
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
            dish_pos=jnp.stack([l.dish_pos for l in levels]),
            initial_mouse_pos=jnp.stack([l.initial_mouse_pos for l in levels]),
        )
