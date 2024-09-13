"""
Parameterised environment and level generator for cheese in the corner
problem. Key components are as follows.

Structs:

* The `Level` struct represents a particular maze layout and cheese/mouse
  spawn position.
* The `EnvState` struct represents a specific dynamic state of the
  environment.
* The `LevelSolution` struct represents a solution to a given maze layout.

Classes:

* `Env` class, provides `reset`, `step`, and `render` methods in a
  gymnax-style interface (see `base` module for specifics of the interface).
* `LevelGenerator` class, provides `sample` method for randomly sampling a
  level.
* `LevelParser` class, provides a `parse` and `parse_batch` method for
  designing Level structs based on ASCII depictions.
* `LevelSolver` class, provides a `solve` method for levels and further
  methods to query the solution for the optimal value of the level and the
  value or optimal action from any state within that level.
* `LevelSplayer` classes, providing static `splay` methods for transforming a
  level into sets of similar levels.
"""

import enum
import functools
from typing import Any

import jax
import jax.numpy as jnp
import einops
from flax import struct

import chex
from jaxtyping import PyTree

from jaxgmg.procgen import maze_generation as mg
from jaxgmg.procgen import maze_solving
from jaxgmg.environments import base

from jaxgmg import util
import jax.numpy as jnp
from jax import lax


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
    * got_corner : bool
            Whether the mouse has already visited the corner.
    """
    mouse_pos: chex.Array
    got_cheese: bool
    got_corner: bool


@struct.dataclass
class Observation(base.Observation):
    """
    Observation for partially observable Maze environment.

    * image : bool[h, w, c] or float[h, w, rgb]
            The contents of the state. Comes in one of two formats:
            * Boolean: a H by W by C bool array where each channel represents
              the presence of one type of thing (walls, mouse, cheese).
            * Pixels: an D.H by D.W by 3 array of RGB float values where each
              D by D tile corresponds to one grid square. (D is level of
              detail.)
    """
    image: chex.Array


class Action(enum.IntEnum):
    """
    The environment has a discrete action space of size 4 with the following
    meanings.
    """
    MOVE_UP     = 0
    MOVE_LEFT   = 1
    MOVE_DOWN   = 2
    MOVE_RIGHT  = 3


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


@struct.dataclass
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
    """
    terminate_after_cheese_and_corner: bool = False


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
            got_corner=False,
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
        got_cheese_first_time = got_cheese & ~state.got_cheese
        state = state.replace(got_cheese=state.got_cheese | got_cheese)
        
        # check if mouse got to corner
        got_corner = (state.mouse_pos[0] == 1) & (state.mouse_pos[1] == 1)
        got_corner_first_time = got_corner & ~state.got_corner
        state = state.replace(got_corner=state.got_corner | got_corner)

        # rewards
        reward = got_cheese_first_time.astype(float)
        proxy_reward = got_corner_first_time.astype(float)
        
        # end of episode
        if self.terminate_after_cheese_and_corner:
            done = state.got_cheese & state.got_corner
        else:
            done = state.got_cheese

        return (
            state,
            reward,
            done,
            {
                'proxy_rewards': {
                    'proxy_corner': proxy_reward,
                },
            },
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
            # one object
            image_bool[:, :, Channel.WALL],
            image_bool[:, :, Channel.MOUSE],
            image_bool[:, :, Channel.CHEESE],
            # no objects, 'default' (always true)
            jnp.ones((H, W), dtype=bool),
        ])
        chosen_sprites = jnp.argmax(sprite_priority_vector_grid, axis=0)

        # put the corresponding sprite into each square
        spritemap = jnp.stack([
            # one object
            spritesheet['WALL'],
            spritesheet['MOUSE'],
            spritesheet['CHEESE'],
            # no objects
            spritesheet['PATH'],
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


# # # 
# Level generation


@struct.dataclass
class LevelGenerator(base.LevelGenerator):
    """
    Level generator for Cheese in the Corner environment. Given some maze
    configuration parameters and cheese location parameter, provides a
    `sample` method that generates a random level.

    * height : int,(>= 3, odd)
            The number of rows in the grid representing the maze
            (including top and bottom boundary rows)
    * width : int (>= 3, odd)
            The number of columns in the grid representing the maze
            (including left and right boundary rows)
    * maze_generator : maze_generation.MazeGenerator
            Provides the maze generation method to use (see module
            `maze_generation` for details).
            The default is a tree maze generator using Kruskal's algorithm.
    * corner_size : int (>=1, <=width, <=height):
            The cheese will spawn within a square of this width located in
            the top left corner.
    """
    height: int = 13
    width: int = 13
    maze_generator : mg.MazeGenerator = mg.TreeMazeGenerator()
    corner_size: int = 1
    
    def __post_init__(self):
        assert self.corner_size >= 1


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
        
        # cheese spawn in top left corner region
        corner_mask = (
            # first `corner_size` rows not including border
              (coords[:, 0] >= 1)
            & (coords[:, 0] <= self.corner_size)
            # first `corner_size` cols not including border
            & (coords[:, 1] >= 1)
            & (coords[:, 1] <= self.corner_size)
        ).flatten()
        # ... not on a wall (unless that's the only option)
        cheese_mask = corner_mask & no_wall_mask
        cheese_mask = cheese_mask | (~(cheese_mask.any()) & corner_mask)
        rng_spawn_cheese, rng = jax.random.split(rng)
        cheese_pos = jax.random.choice(
            key=rng_spawn_cheese,
            a=coords,
            axis=0,
            p=cheese_mask,
        )
        # ... in case there *was* a wall there , remove it
        wall_map = wall_map.at[cheese_pos[0], cheese_pos[1]].set(False)
        
        # mouse spawn in some remaining valid position
        mouse_mask = no_wall_map.at[
            cheese_pos[0],
            cheese_pos[1],
        ].set(False).flatten()

        rng_spawn_mouse, rng = jax.random.split(rng)
        initial_mouse_pos = jax.random.choice(
            key=rng_spawn_mouse,
            a=coords,
            axis=0,
            p=mouse_mask,
        )

        return Level(
            wall_map=wall_map,
            cheese_pos=cheese_pos,
            initial_mouse_pos=initial_mouse_pos,
        )


# # # 
# Level mutation


@struct.dataclass
class ToggleWallLevelMutator(base.LevelMutator):


    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape
        # TODO: assuming (h-2)*(w-2) > 2 or something
        
        # which walls are available to toggle?
        valid_map = jnp.ones((h, w), dtype=bool)
        # exclude border
        valid_map = valid_map.at[(0, h-1), :].set(False)
        valid_map = valid_map.at[:, (0, w-1)].set(False)
        # exclude current cheese and mouse spawn positions
        valid_map = valid_map.at[
            (level.cheese_pos[0], level.initial_mouse_pos[0]),
            (level.cheese_pos[1], level.initial_mouse_pos[1]),
        ].set(False)
        valid_mask = valid_map.flatten()

        # pick a random valid position
        coords = einops.rearrange(jnp.indices((h, w)), 'c h w -> (h w) c')
        toggle_pos = jax.random.choice(
            key=rng,
            a=coords,
            axis=0,
            p=valid_mask,
        )

        # toggle the wall there
        hit_wall = level.wall_map[toggle_pos[0], toggle_pos[1]]
        new_wall_map = level.wall_map.at[
            toggle_pos[0],
            toggle_pos[1],
        ].set(~hit_wall)

        return level.replace(wall_map=new_wall_map)


@struct.dataclass
class StepMouseLevelMutator(base.LevelMutator):
    transpose_with_cheese_on_collision: bool


    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape
        assert h > 3, "level too small to step mouse"
        assert w > 3, "level too small to step mouse"

        # move the mouse in a random direction (within bounds)
        steps = jnp.array((
            (-1,  0),   # up
            ( 0, -1),   # left
            (+1,  0),   # down
            ( 0, +1),   # right
        ))
        valid_mask = jnp.array((
            level.initial_mouse_pos[0] >= 2,
            level.initial_mouse_pos[1] >= 2,
            level.initial_mouse_pos[0] <= h-3,
            level.initial_mouse_pos[1] <= w-3,
        ))
        chosen_step = jax.random.choice(
            key=rng,
            a=steps,
            p=valid_mask,
        )
        new_initial_mouse_pos = level.initial_mouse_pos + chosen_step

        # carve through walls
        new_wall_map = level.wall_map.at[
            new_initial_mouse_pos[0],
            new_initial_mouse_pos[1],
        ].set(False)

        # resolve collision with cheese
        hit_cheese = (new_initial_mouse_pos == level.cheese_pos).all()
        if self.transpose_with_cheese_on_collision:
            # transpose mouse with cheese
            new_cheese_pos = jax.lax.select(
                hit_cheese,
                level.initial_mouse_pos,
                level.cheese_pos,
            )
        else:
            # mutation fails to move cheese
            new_cheese_pos = level.cheese_pos
            new_initial_mouse_pos = jax.lax.select(
                hit_cheese,
                level.initial_mouse_pos,
                new_initial_mouse_pos,
            )

        return level.replace(
            wall_map=new_wall_map,
            initial_mouse_pos=new_initial_mouse_pos,
            cheese_pos=new_cheese_pos,
        )


@struct.dataclass
class ScatterMouseLevelMutator(base.LevelMutator):
    transpose_with_cheese_on_collision: bool


    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape

        # teleport the mouse to a random location within bounds
        rng_row, rng_col = jax.random.split(rng)
        new_mouse_row = jax.random.choice(
            key=rng_row,
            a=jnp.arange(1, h-1),
        )
        new_mouse_col = jax.random.choice(
            key=rng_col,
            a=jnp.arange(1, w-1),
        )
        new_initial_mouse_pos = jnp.array((
            new_mouse_row,
            new_mouse_col,
        ))

        # carve through walls
        new_wall_map = level.wall_map.at[
            new_initial_mouse_pos[0],
            new_initial_mouse_pos[1],
        ].set(False)

        # resolve collision with cheese
        hit_cheese = (new_initial_mouse_pos == level.cheese_pos).all()
        if self.transpose_with_cheese_on_collision:
            # transpose mouse with cheese
            new_cheese_pos = jax.lax.select(
                hit_cheese,
                level.initial_mouse_pos,
                level.cheese_pos,
            )
        else:
            # leave both in current positions
            new_cheese_pos = level.cheese_pos
            new_initial_mouse_pos = jax.lax.select(
                hit_cheese,
                level.initial_mouse_pos,
                new_initial_mouse_pos,
            )

        return level.replace(
            wall_map=new_wall_map,
            initial_mouse_pos=new_initial_mouse_pos,
            cheese_pos=new_cheese_pos,
        )


@struct.dataclass
class StepCheeseLevelMutator(base.LevelMutator):
    # TODO: restrict to a designated region
    # TODO: make it optional to transpose with the mouse


    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape
        assert h > 3, "level too small to step cheese"
        assert w > 3, "level too small to step cheese"

        # move the cheese in a random direction (within bounds)
        steps = jnp.array((
            (-1,  0),   # up
            ( 0, -1),   # left
            (+1,  0),   # down
            ( 0, +1),   # right
        ))
        valid_mask = jnp.array((
            level.cheese_pos[0] >= 2,
            level.cheese_pos[1] >= 2,
            level.cheese_pos[0] <= h-3,
            level.cheese_pos[1] <= w-3,
        ))
        chosen_step = jax.random.choice(
            key=rng,
            a=steps,
            p=valid_mask,
        )
        new_cheese_pos = level.cheese_pos + chosen_step

        # carve through walls
        new_wall_map = level.wall_map.at[
            new_cheese_pos[0],
            new_cheese_pos[1],
        ].set(False)

        # upon collision with mouse, transpose cheese with mouse
        hit_mouse = (new_cheese_pos == level.initial_mouse_pos).all()
        new_initial_mouse_pos = jax.lax.select(
            hit_mouse,
            level.cheese_pos,
            level.initial_mouse_pos,
        )

        return level.replace(
            wall_map=new_wall_map,
            initial_mouse_pos=new_initial_mouse_pos,
            cheese_pos=new_cheese_pos,
        )


@struct.dataclass
class ScatterCheeseLevelMutator(base.LevelMutator):
    # TODO: restrict to a designated region


    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape

        # teleport the cheese to a random location within bounds
        rng_row, rng_col = jax.random.split(rng)
        new_cheese_row = jax.random.choice(
            key=rng_row,
            a=jnp.arange(1, h-1),
        )
        new_cheese_col = jax.random.choice(
            key=rng_col,
            a=jnp.arange(1, w-1),
        )
        new_cheese_pos = jnp.array((
            new_cheese_row,
            new_cheese_col,
        ))

        # carve through walls
        new_wall_map = level.wall_map.at[
            new_cheese_pos[0],
            new_cheese_pos[1],
        ].set(False)

        # upon collision with mouse, transpose cheese with mouse
        hit_mouse = (new_cheese_pos == level.initial_mouse_pos).all()
        new_initial_mouse_pos = jax.lax.select(
            hit_mouse,
            level.cheese_pos,
            level.initial_mouse_pos,
        )

        return level.replace(
            wall_map=new_wall_map,
            initial_mouse_pos=new_initial_mouse_pos,
            cheese_pos=new_cheese_pos,
        )


@struct.dataclass
class CornerCheeseLevelMutator(base.LevelMutator):
    corner_size: int


    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape
        row_max = min(h - 2, self.corner_size)
        col_max = min(w - 2, self.corner_size)

        # teleport the cheese to a random location within the corner region
        rng_row, rng_col = jax.random.split(rng)
        new_cheese_row = jax.random.choice(
            key=rng_row,
            a=jnp.arange(1, row_max+1), # 1 to row_max inclusive
        )
        new_cheese_col = jax.random.choice(
            key=rng_col,
            a=jnp.arange(1, col_max+1), # 1 to col_max inclusive
        )
        new_cheese_pos = jnp.array((
            new_cheese_row,
            new_cheese_col,
        ))

        # carve through walls
        new_wall_map = level.wall_map.at[
            new_cheese_row,
            new_cheese_col,
        ].set(False)

        #if cheese is not in the corner, leave it there -> ADDING THIS TO SEE IF THIS IS THE PROBLEM FOR ACCEL PERFORMING POORLY
        
        # corner_pos = jnp.array([1,1])
        # cheese_hit_corner = (level.cheese_pos == corner_pos).all()
        # new_cheese_pos = jax.lax.select(
        #     cheese_hit_corner,
        #     new_cheese_pos,
        #     level.cheese_pos,
        # )
        
        
            
        # upon collision with mouse, transpose cheese with mouse
        hit_mouse = (new_cheese_pos == level.initial_mouse_pos).all()
        new_initial_mouse_pos = jax.lax.select(
            hit_mouse,
            level.cheese_pos,
            level.initial_mouse_pos,
        )
        

        return level.replace(
            wall_map=new_wall_map,
            initial_mouse_pos=new_initial_mouse_pos,
            cheese_pos=new_cheese_pos,
        )
    
@struct.dataclass
class FixedCheeseLevelMutator(base.LevelMutator):

    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        
        return level.replace(
            wall_map=level.wall_map,
            initial_mouse_pos=level.initial_mouse_pos,
            cheese_pos=level.cheese_pos,
        )
    

# # # 
# Level parsing


@struct.dataclass
class LevelParser(base.LevelParser):
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
            * The character '#' maps to `Channel.WALL`.
            * The character '@' maps to `Channel.MOUSE`.
            * The character '*' maps to `Channel.CHEESE`.
            * The character '.' maps to `len(Channel)`, i.e. none of the
              above, representing the absence of an item.
    """
    height: int
    width: int
    char_map = {
        '#': Channel.WALL,
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
        wall_map = (level_map == Channel.WALL)
        assert wall_map[0,:].all(), "top border incomplete"
        assert wall_map[:,0].all(), "left border incomplete"
        assert wall_map[-1,:].all(), "bottom border incomplete"
        assert wall_map[:,-1].all(), "right border incomplete"
        
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
            cheese_pos=cheese_pos,
            initial_mouse_pos=initial_mouse_pos,
        )


# # # 
# Level solving


@struct.dataclass
class LevelSolution(base.LevelSolution):
    level: Level
    directional_distance_to_cheese: chex.Array


@struct.dataclass
class LevelSolutionProxies(base.LevelSolutionProxies):
    level: Level
    #you have a dictionary of proxies, and have an entry for each proxy. so create a dict of proxies, where each entry has a name for a proxy and a corresponding chex.array
    directional_distance_to_proxies: dict[str, chex.Array]


@struct.dataclass
class LevelSolver(base.LevelSolver):


    @functools.partial(jax.jit, static_argnames=('self',))
    def solve(self, level: Level) -> LevelSolution:
        """
        Compute the distance from each possible mouse position to the cheese
        position a given level. From this information one can easy compute
        the optimal action or value from any state of this level.

        Parameters:

        * level : Level
                The level to compute the optimal value for.

        Returns:

        * soln : LevelSolution
                The necessary precomputed (directional) distances for later
                computing optimal values and actions from states.

        TODO:

        * Solving the mazes currently uses all pairs shortest paths
          algorithm, which is not efficient enough to work for very large
          mazes. If we wanted to solve very large mazes, we could by changing
          to a single source shortest path algorithm.
        """
        # compute distance between mouse and cheese
        dir_dist = maze_solving.maze_directional_distances(level.wall_map)
        dir_dist_to_cheese = dir_dist[
            :,
            :,
            level.cheese_pos[0],
            level.cheese_pos[1],
            :,
        ]

        return LevelSolution(
            level=level,
            directional_distance_to_cheese=dir_dist_to_cheese,
        )


    @functools.partial(jax.jit, static_argnames=('self',)) #proxies is a list with a name of strings for various proxies
    def solve_proxy(self, level: Level) -> LevelSolutionProxies:
        """
        Compute the distance from each possible mouse position to the cheese
        position a given level. From this information one can easy compute
        the optimal action or value from any state of this level.

        Parameters:

        * level : Level
                The level to compute the optimal value for.

        Returns:

        * soln : LevelSolution
                The necessary precomputed (directional) distances for later
                computing optimal values and actions from states.

        TODO:

        * Solving the mazes currently uses all pairs shortest paths
          algorithm, which is not efficient enough to work for very large
          mazes. If we wanted to solve very large mazes, we could by changing
          to a single source shortest path algorithm.
        """
        proxies = ['proxy_corner'] #where you define your proxies...
        # compute distance between mouse and cheese
        dir_dist = maze_solving.maze_directional_distances(level.wall_map)
        # calculate the distance for each proxy
        proxy_directions = {}
        # first, get the name of each proxy
        for proxy_name in proxies:
            if proxy_name == 'proxy_corner':
                dir_dist_to_corner = dir_dist[
                    :,
                    :,
                    1,
                    1,
                    :,
                ]
                proxy_directions[proxy_name] = dir_dist_to_corner
            else:
                raise ValueError(f"Proxy {proxy_name} not recognized") #corner is the only proxy in this environment
        return LevelSolutionProxies(
            level=level,
            directional_distance_to_proxies=proxy_directions,
        )
    

    @functools.partial(jax.jit, static_argnames=('self',))
    def state_value(self, soln: LevelSolution, state: EnvState) -> float:
        """
        Optimal return value from a given state.

        Parameters:

        * soln : LevelSolution
                The output of `solve` method for this level.
        * state : EnvState
                The state to compute the value for.

        Return:

        * value : float
                The optimal value of this state.
        """
        # steps to get to the cheese: look up in distance cache
        optimal_dist = soln.directional_distance_to_cheese[
            state.mouse_pos[0],
            state.mouse_pos[1],
            4, # stay here
        ]

        # reward when we get to the cheese is 1 iff the cheese is still there
        reward = (1.0 - state.got_cheese)
        # maybe we apply a time penalty
        time_of_reward = state.steps + optimal_dist
        penalty = (1.0 - 0.9 * time_of_reward / self.env.max_steps_in_episode)
        penalized_reward = jnp.where(
            self.env.penalize_time,
            penalty * reward,
            reward,
        )
        # mask out rewards beyond the end of the episode
        episode_still_valid = time_of_reward < self.env.max_steps_in_episode
        valid_reward = penalized_reward * episode_still_valid

        # discount the reward
        discounted_reward = (self.discount_rate**optimal_dist) * valid_reward

        return discounted_reward


    @functools.partial(jax.jit, static_argnames=('self',))
    def state_value_proxies(self,soln: LevelSolutionProxies, state: EnvState) -> dict[str, float]:
        """
        Optimal return value from a given state.

        Parameters:

        * soln : LevelSolutionProxies
                The output of `solve` method for this level for the proxies.
        * state : EnvState
                The state to compute the value for.
        
        Return:

        * dict of rewards for each proxy: dict[str, float]
                The optimal value of this state for each proxy.
        """

        proxy_rewards = {}
        for proxy_name, proxy_directions in soln.directional_distance_to_proxies.items():
            if proxy_name == 'proxy_corner':
                optimal_dist = proxy_directions[
                    state.mouse_pos[0],
                    state.mouse_pos[1],
                    4, # stay here
                ]
                # reward when we get to the corner is 1 iff the corner is still there
                reward = (1.0 - state.got_corner)
                # maybe we apply a time penalty
                time_of_reward = state.steps + optimal_dist
                penalty = (1.0 - 0.9 * time_of_reward / self.env.max_steps_in_episode)
                penalized_reward = jnp.where(
                    self.env.penalize_time,
                    penalty * reward,
                    reward,
                )
                # mask out rewards beyond the end of the episode
                episode_still_valid = time_of_reward < self.env.max_steps_in_episode
                valid_reward = penalized_reward * episode_still_valid

                # discount the reward
                discounted_reward = (self.discount_rate**optimal_dist) * valid_reward
                proxy_rewards[proxy_name] = discounted_reward
            else:
                raise ValueError(f"Proxy {proxy_name} not recognized") #corner is the only proxy in this environment, did not implement any other
        
        return proxy_rewards


    @functools.partial(jax.jit, static_argnames=('self',))
    def state_action_values(
        self,
        soln: LevelSolution,
        state: EnvState,
    ) -> chex.Array: # float[4]
        """
        Optimal return value from a given state.

        Parameters:

        * soln : LevelSolution
                The output of `solve` method for this level.
        * state : EnvState
                The state to compute the value for.
            
        Notes:

        * With a steep discount rate or long episodes, this algorithm might
          run into minor numerical issues where small contributions to the
          return from late into the episode are lost.
        """
        # steps to get to the cheese for adjacent squares: look up in cache
        dir_dists = soln.directional_distance_to_cheese[
            state.mouse_pos[0],
            state.mouse_pos[1],
        ] # -> float[5] (up left down right stay)
        # steps after taking each action, taking collisions into account:
        # replace inf values with stay-still values
        action_dists = jnp.where(
            jnp.isinf(dir_dists[:4]),
            dir_dists[4],
            dir_dists[:4],
        )

        # reward when we get to the cheese is 1 iff the cheese is still there
        reward = (1.0 - state.got_cheese)
        # maybe we apply a time penalty
        times_of_reward = state.steps + action_dists
        penalties = (
            1.0 - 0.9 * times_of_reward / self.env.max_steps_in_episode
        )
        penalized_rewards = jnp.where(
            self.env.penalize_time,
            penalties * reward,
            reward,
        )
        # mask out rewards beyond the end of the episode
        episode_still_valids = times_of_reward < self.env.max_steps_in_episode
        valid_rewards = penalized_rewards * episode_still_valids

        # discount the reward
        discounted_rewards = (
            (self.discount_rate ** action_dists) * valid_rewards
        )

        return discounted_rewards


    @functools.partial(jax.jit, static_argnames=('self',))
    def state_action(self, soln: LevelSolution, state: EnvState) -> int:
        """
        Optimal action from a given state.

        Parameters:

        * soln : LevelSolution
                The output of `solve` method for this level.
        * state : EnvState
                The state to compute the optimal action for.
            
        Return:

        * action : int                      # TODO: use the Action enum?
                An optimal action from the given state.
                
        Notes:

        * If there are multiple equally optimal actions, this method will
          return the first according to the order up (0), left (1), down (2),
          or right (3).
        * As a special case of this, if the cheese is unreachable, the
          returned action will be up (0).
        * If the cheese is on the current square, the returned action is
          arbitrary, and in fact it might even be suboptimal, since if there
          is a wall the optimal action is to move into that wall.
        * If the cheese has already been gotten then there is no more reward
          available, but this method will still direct the mouse towards the
          cheese position.
        * If the cheese is too far away to reach by the end of the episode,
          this method will still direct the mouse towards the cheese.

        TODO: 

        * Make all environments have a 'stay action' will simplify these
          solutions a fair bit. The mouse could stay when on the cheese, or
          when the cheese is unreachable, or when the cheese is already
          gotten.
        """
        action = jnp.argmin(soln.directional_distance_to_cheese[
            state.mouse_pos[0],
            state.mouse_pos[1],
            :4,
        ])
        return action


# # # 
# Splay functions
# Note that these functions are not jittable.


def splay_mouse(level: Level):
    free_map = ~level.wall_map
    free_map = free_map.at[
        level.cheese_pos[0],
        level.cheese_pos[1],
    ].set(False)

    # assemble level batch
    num_levels = free_map.sum()
    levels = Level(
        wall_map=einops.repeat(
            level.wall_map,
            'h w -> n h w',
            n=num_levels,
        ),
        cheese_pos=einops.repeat(
            level.cheese_pos,
            'c -> n c',
            n=num_levels,
        ),
        initial_mouse_pos=jnp.stack(jnp.where(free_map), axis=1),
    )
    
    # remember how to put the levels back together into a grid
    levels_pos = jnp.where(free_map)
    grid_shape = free_map.shape
    
    return (
        levels,
        num_levels,
        levels_pos,
        grid_shape,
    )


def splay_cheese(level: Level):
    free_map = ~level.wall_map
    free_map = free_map.at[
        level.initial_mouse_pos[0],
        level.initial_mouse_pos[1],
    ].set(False)

    # assemble level batch
    num_levels = free_map.sum()
    levels = Level(
        wall_map=einops.repeat(
            level.wall_map,
            'h w -> n h w',
            n=num_levels,
        ),
        cheese_pos=jnp.stack(jnp.where(free_map), axis=1),
        initial_mouse_pos=einops.repeat(
            level.initial_mouse_pos,
            'c -> n c',
            n=num_levels,
        ),
    )
    
    # remember how to put the levels back together into a grid
    levels_pos = jnp.where(free_map)
    grid_shape = free_map.shape
    
    return (
        levels,
        num_levels,
        levels_pos,
        grid_shape,
    )


def splay_cheese_and_mouse(level: Level):
    free_map = ~level.wall_map
    # macromaze
    free_metamap = free_map[None,None,:,:] & free_map[:,:,None,None]
    # remove mouse/cheese clashes
    free_pos = jnp.where(free_map)
    clash_pos = free_pos + free_pos # concatenate tuples
    free_metamap = free_metamap.at[clash_pos].set(False)
    # rearrange into appropriate order(s)
    free_metamap_hhww = einops.rearrange(
        free_metamap,
        'H W h w -> H h W w',
    )

    # assemble level batch
    num_spaces = free_map.sum()
    num_levels = (num_spaces - 1) * num_spaces
    # cheese/mouse spawn locations
    cheese1, mouse1, cheese2, mouse2 = jnp.where(free_metamap_hhww)
    levels = Level(
        wall_map=einops.repeat(
            level.wall_map,
            'h w -> n h w',
            n=num_levels,
        ),
        cheese_pos=jnp.stack((cheese1, cheese2), axis=1),
        initial_mouse_pos=jnp.stack((mouse1, mouse2), axis=1),
    )

    # remember how to put the levels back together into a grid
    free_metamap_grid = einops.rearrange(
        free_metamap_hhww[1:-1,:,1:-1,:],
        'H h W w -> (H h) (W w)',
    )
    levels_pos = jnp.where(free_metamap_grid)
    grid_shape = free_metamap_grid.shape
    
    return (
        levels,
        num_levels,
        levels_pos,
        grid_shape,
    )


# # # 
# Level complexity metrics


@struct.dataclass
class LevelMetrics(base.LevelMetrics):


    @functools.partial(jax.jit, static_argnames=('self',))
    def compute_metrics(
        self,
        levels: Level,          # Level[num_levels]
        weights: chex.Array,    # float[num_levels]
    ) -> dict[str, Any]:        # metrics
        # basics
        num_levels, h, w = levels.wall_map.shape
        dists = jax.vmap(maze_solving.maze_distances)(levels.wall_map)
        
        # num walls (excluding border)
        inner_wall_maps = levels.wall_map[:,1:-1,1:-1]
        num_walls = jnp.sum(inner_wall_maps, axis=(1,2))
        prop_walls = num_walls / ((h-2) * (w-2) - 2)

        # avg wall location, cheese location, mouse location
        wall_map = levels.wall_map
        mouse_map = jnp.zeros_like(levels.wall_map).at[
            levels.initial_mouse_pos[:, 0],
            levels.initial_mouse_pos[:, 1],
        ].set(True)
        cheese_map = jnp.zeros_like(levels.wall_map).at[
            levels.cheese_pos[0],
            levels.cheese_pos[1],
        ].set(True)
        
        # cheese distance from top left corner
        cheese_dists = dists[
            jnp.arange(num_levels),
            1,
            1,
            levels.cheese_pos[:, 0],
            levels.cheese_pos[:, 1],
        ]
        cheese_dists_finite = jnp.nan_to_num(
            cheese_dists,
            posinf=(h-2)*(w-2)/2,
        )
        avg_cheese_dist = cheese_dists_finite.mean()

        #buffer
        cheese_in_corner = (levels.cheese_pos[:, 0] == 1) & (levels.cheese_pos[:, 1] == 1)
        cheese_in_corner_avg = cheese_in_corner.mean()

        # shortest path length and solvability
        opt_dists = dists[
            jnp.arange(num_levels),
            levels.initial_mouse_pos[:, 0],
            levels.initial_mouse_pos[:, 1],
            levels.cheese_pos[:, 0],
            levels.cheese_pos[:, 1],
        ]
        solvable = ~jnp.isposinf(opt_dists)
        opt_dists_solvable = solvable * opt_dists
        opt_dists_finite = jnp.nan_to_num(opt_dists, posinf=(h-2)*(w-2)/2)
        
        # rendered levels in a grid
        def render_level(level):
            state = self.env._reset(level)
            rgb = self.env.render_state(state)
            return rgb
        _, top_16_level_ids = jax.lax.top_k(weights, k=16)
        top_16_levels = jax.tree.map(
            lambda leaf: leaf[top_16_level_ids],
            levels,
        )
        rendered_levels = jax.vmap(render_level)(top_16_levels)
        rendered_levels_pad = jnp.pad(
            rendered_levels,
            pad_width=((0, 0), (0, 1), (0, 1), (0, 0)),
        )
        rendered_levels_grid = einops.rearrange(
            rendered_levels_pad,
            '(level_h level_w) h w c -> (level_h h) (level_w w) c',
            level_w=4,
        )[:-1,:-1] # cut off last pixel of padding

        return {
            'layout': {
                'levels16_img': rendered_levels_grid,
                # number of walls
                'num_walls_hist': num_walls,
                'num_walls_avg': num_walls.mean(),
                'num_walls_wavg': num_walls @ weights,
                # proportion of walls
                'prop_walls_hist': prop_walls,
                'prop_walls_avg': prop_walls.mean(),
                'prop_walls_wavg': prop_walls @ weights,
                # # superimposed layout and position maps
                # 'wall_map_avg_img': util.viridis(wall_map.mean(axis=0)),
                # 'wall_map_wavg_img': util.viridis(jnp.einsum('lhw,l->hw', wall_map, weights)),
                # 'mouse_map_avg_img': util.viridis(mouse_map.mean(axis=0)),
                # 'mouse_map_wavg_img': util.viridis(jnp.einsum('lhw,l->hw', mouse_map, weights)),
                # 'cheese_map_avg_img': util.viridis(cheese_map.mean(axis=0)),
                # 'cheese_map_wavg_img': util.viridis(jnp.einsum('lhw,l->hw', cheese_map, weights)),
            },
            'distances': {
                # solvability
                'solvable_num': solvable.sum(),
                'solvable_avg': solvable.mean(),
                'solvable_wavg': solvable @ weights,
                # optimal dist mouse to cheese
                'mouse-cheese_dist_finite_hist': opt_dists_finite,
                'mouse-cheese_dist_finite_avg': opt_dists_finite.mean(),
                'mouse-cheese_dist_finite_wavg': (opt_dists_finite @ weights),
                'mouse-cheese_dist_solvable_avg': opt_dists_solvable.sum() / solvable.sum(),
                'mouse-cheese_dist_solvable_wavg': (opt_dists_solvable @ weights) / (solvable @ weights),
                # optimal dist from cheese to corner
                'cheese-corner_dist_hist': cheese_dists_finite,
                'cheese-corne_dist_avg': cheese_dists_finite.mean(),
                'cheese-corne_dist_wavg': cheese_dists_finite @ weights,
                #buffer metrics
                'levels_cheese_is_in_corner_avg':  cheese_in_corner_avg
            },
        }

