"""
Parameterised environment and level generator for keys and chests problem.
Key components are as follows.

Structs:

* The `Level` struct represents a particular maze layout and key/chest/mouse
  spawn position.
* The `EnvState` struct represents a specific dynamic state of the
  environment.

Classes:

* `Env` class, provides `reset`, `step`, and `render` methods in a
  gymnax-style interface (see `base` module for specifics of the interface).
* `LevelGenerator` class, provides `sample` method for randomly sampling a
  level from a configurable level distribution.
* `LevelParser` class, provides a `parse` and `parse_batch` method for
  designing Level structs based on ASCII depictions.
"""

import enum
import functools
import itertools

from typing import Any

import jax
import jax.numpy as jnp
import einops
from flax import struct
import chex
from jaxtyping import PyTree

from jaxgmg.procgen import maze_generation as mg
from jaxgmg.procgen import maze_solving, combinatorix
from jaxgmg.environments import base


@struct.dataclass
class Level(base.Level):
    """
    Represent a particular environment layout:

    * wall_map : bool[h, w]
            Maze layout (True = wall)
    * keys_pos : index[k, 2]
            List of coordinates of keys (index into `wall_map`)
    * chests_pos : index[c, 2]
            List of coordinates of chests (index into `wall_map`)
    * initial_mouse_pos : index[2]
            Coordinates of initial mouse position (index into `wall_map`)
    * inventory_map : index[k]
            Coordinates of inventory (index into width)
    * hidden_keys : bool[k]
            True for keys that are actually unused within the level.
    * hidden_chests : bool[c]
            True for chests that are actually unused within the level.
    """
    wall_map: chex.Array
    keys_pos: chex.Array
    chests_pos: chex.Array
    initial_mouse_pos: chex.Array
    inventory_map: chex.Array
    hidden_keys: chex.Array
    hidden_chests: chex.Array


@struct.dataclass
class EnvState(base.EnvState):
    """
    Dynamic environment state within a particular level.

    * mouse_pos : index[2]
            Current coordinates of the mouse. Initialised to
            `level.initial_mouse_pos`.
    * got_keys : bool[k]
            Mask tracking which keys have already been collected (True).
            Initially all False.
    * used_keys : bool[k]
            Mask tracking which keys have already been collected and then
            spent to open a chest (True). Initially all False.
    * got_chests : bool[c]
            Mask tracking which chests have already been opened (True).
            Initially all False.
    """
    mouse_pos: chex.Array
    got_keys: jax.Array
    used_keys: jax.Array
    got_chests: jax.Array


@struct.dataclass
class Observation(base.Observation):
    """
    Observation for partially observable Maze environment.

    * image : bool[h, w, c] or float[h, w, rgb]
            The contents of the state. Comes in one of two formats:
            * Boolean: a H by W by C bool array where each channel represents
              the presence of one type of thing (wall, mouse, key in world,
              chest, key in inventory).
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
    `channel` Boolean array, where the final dimensions 0 through 4 indicate
    the following:

    * `WALL`:   True in the locations where there is a wall.
    * `MOUSE`:  True in the one location the mouse occupies.
    * `KEY`:    True in locations occupied by an uncollected key.
    * `CHEST`:  True in locations occupied by an unopened chest.
    * `INV`:    True in a number of random locations corresponding to the
                number of previously-collected but as-yet-unused keys.
    """
    WALL  = 0
    MOUSE = 1
    KEY   = 2
    CHEST = 3
    INV   = 4


class Env(base.Env):
    """
    Keys and Chests environment.

    In this environment the agent controls a mouse navigating a grid-based
    maze. The mouse must pick up keys located throught the maze and then use
    them to open chests.

    There are four available actions which deterministically move the mouse
    one grid square up, right, down, or left respectively.
    * If the mouse would hit a wall it remains in place.
    * If the mouse hits a key, the key is removed from the grid and stored in
      the mouse's inventory.
    * If the mouse hits a chest and has at least one key in its inventory
      then the mouse opens the chest, reward is delivered, and the key is
      spent. If the mouse doesn't have any keys it passes through the chest.
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


    @functools.partial(jax.jit, static_argnames=('self',))
    def _reset(
        self,
        level: Level,
    ) -> EnvState:
        """
        See reset_to_level method of Underspecified
        """
        num_keys, _2 = level.keys_pos.shape
        num_chests, _2 = level.chests_pos.shape
        state = EnvState(
            mouse_pos=level.initial_mouse_pos,
            got_keys=level.hidden_keys,
            used_keys=level.hidden_keys,
            got_chests=level.hidden_chests,
            level=level,
            steps=0,
            done=False,
        )
        return state
        

    @functools.partial(jax.jit, static_argnames=('self',))
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

        # interact with keys
        pickup_keys = (
            # select keys in the same location as the mouse
            (state.mouse_pos == state.level.keys_pos).all(axis=1)
            # filter for keys the mouse hasn't yet picked up
            & (~state.got_keys)
        )
        state = state.replace(got_keys=state.got_keys ^ pickup_keys)

        # interact with chests
        available_keys = (state.got_keys & ~state.used_keys)
        open_chests = (
            # select chests in the same location as the mouse:
            (state.mouse_pos == state.level.chests_pos).all(axis=1)
            # filter for chests the mouse hasn't yet picked up
            & (~state.got_chests)
            # mask this whole thing by whether the mouse currently has a key
            & available_keys.any()
        )
        the_used_key = jnp.argmax(available_keys) # finds first True if any

        state = state.replace(
            got_chests=state.got_chests | open_chests,
            used_keys=state.used_keys.at[the_used_key].set(
                state.used_keys[the_used_key] | open_chests.any()
            ),
        )
        
        # reward for each chest just opened
        reward = open_chests.sum().astype(float)
        
        # check progress
        # TODO: consider reachability
        available_keys = (~state.level.hidden_keys).sum()
        available_chests = (~state.level.hidden_chests).sum()
        #keys_collected_excess =  ## complete there
        chests_collectable = jnp.minimum(available_keys, available_chests)
        chests_collected = (state.got_chests ^ state.level.hidden_chests).sum()
        done = (chests_collected == chests_collectable)

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

        # render mouse
        image = image.at[
            state.mouse_pos[0],
            state.mouse_pos[1],
            Channel.MOUSE,
        ].set(True)

        # render keys that haven't been picked up
        image = image.at[
            state.level.keys_pos[:, 0],
            state.level.keys_pos[:, 1],
            Channel.KEY,
        ].set(~state.got_keys)
        
        # render chests that haven't been opened
        image = image.at[
            state.level.chests_pos[:, 0],
            state.level.chests_pos[:, 1],
            Channel.CHEST,
        ].set(~state.got_chests)

        # render keys that have been picked up but haven't been used
        image = image.at[
            0,
            state.level.inventory_map,
            Channel.INV,
        ].set(state.got_keys & ~state.used_keys)

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
            # multiple objects
            image_bool[:, :, Channel.MOUSE] & image_bool[:, :, Channel.CHEST],
            image_bool[:, :, Channel.WALL] & image_bool[:, :, Channel.INV],
            # one object
            image_bool[:, :, Channel.WALL],
            image_bool[:, :, Channel.MOUSE],
            image_bool[:, :, Channel.KEY],
            image_bool[:, :, Channel.CHEST],
            # no objects, 'default' (always true)
            jnp.ones((H, W), dtype=bool),
        ])
        chosen_sprites = jnp.argmax(sprite_priority_vector_grid, axis=0)

        # put the corresponding sprite into each square
        spritemap = jnp.stack([
            # multiple objects
            spritesheet['MOUSE_ON_CHEST'],
            spritesheet['KEY_ON_WALL'],
            # one object
            spritesheet['WALL'],
            spritesheet['MOUSE'],
            spritesheet['KEY'],
            spritesheet['CHEST'],
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
# Level generator


@struct.dataclass
class LevelGenerator(base.LevelGenerator):
    """
    Level generator for Keys and Chests environment. Given some maze
    configuration parameters and key/chest sparsity parameters, provides a
    `sample` method that generates a random level.

    * height (int, >= 3):
            the number of rows in the grid representing the maze
            (including top and bottom boundary rows)
    * width (int, >= 3):
            the number of columns in the grid representing the maze
            (including left and right boundary rows)
    * maze_generator : maze_generation.MazeGenerator
            Provides the maze generation method to use (see module
            `maze_generation` for details).
            The default is a tree maze generator using Kruskal's algorithm.
    * num_keys : int (>= 0)
            the number of keys to randomly place in each generated maze.
    * num_keys_max : int (>0, >= num_keys)
            determines the shape of the key-related arrays in the level
            struct.
    * num_chests : int (>= 0)
            the number of chests to randomly place in each generated maze.
    * num_chests_max : int (>-, >= num_chests)
            determines the shape of the chest-related arrays in the level
            struct.
    """
    height: int = 13
    width: int = 13
    maze_generator : mg.MazeGenerator = mg.TreeMazeGenerator()
    num_keys: int = 1
    num_keys_max: int = 4
    num_chests: int = 4
    num_chests_max: int = 4

    def __post_init__(self):
        assert self.num_keys >= 0
        assert self.num_keys_max > 0
        assert self.num_keys_max >= self.num_keys
        assert self.num_chests >= 0
        assert self.num_chests_max > 0
        assert self.num_chests_max >= self.num_chests
        assert self.num_keys_max <= self.width, "need width for inventory"
        # TODO: somehow prevent or handle too many walls to spawn all items?

    
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

        # spawn random mouse pos, keys pos, chests pos
        rng_spawn, rng = jax.random.split(rng)
        coords = einops.rearrange(
            jnp.indices((self.height, self.width)),
            'c h w -> (h w) c',
        )
        all_pos = coords[jax.random.choice(
            key=rng_spawn,
            a=coords.shape[0],
            shape=(1 + self.num_keys_max + self.num_chests_max,),
            p=~wall_map.flatten(),
            replace=False,
        )]
        initial_mouse_pos = all_pos[0]
        keys_pos = all_pos[1:1+self.num_keys_max]
        chests_pos = all_pos[1+self.num_keys_max:]
    
        # decide random positions for keys to display in
        rng_inventory, rng = jax.random.split(rng)
        inventory_map = jax.random.choice(
            key=rng_inventory,
            a=self.width,
            shape=(self.num_keys_max,),
            replace=False,
        )

        # hide keys after the given number
        hidden_keys = (jnp.arange(self.num_keys_max) >= self.num_keys)
        
        # hide chests after the given number
        hidden_chests = (jnp.arange(self.num_chests_max) >= self.num_chests)
    
        return Level(
            wall_map=wall_map,
            initial_mouse_pos=initial_mouse_pos,
            keys_pos=keys_pos,
            chests_pos=chests_pos,
            inventory_map=inventory_map,
            hidden_keys=hidden_keys,
            hidden_chests=hidden_chests,
        )


# # #
# Level Mutation


@struct.dataclass
class ToggleWallLevelMutator(base.LevelMutator):

    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape
        
        # which walls are available to toggle?
        valid_map = jnp.ones((h, w), dtype=bool)
        # exclude border
        valid_map = valid_map.at[(0, h-1), :].set(False)
        valid_map = valid_map.at[:, (0, w-1)].set(False)
        # exclude keys/chests/mouse positions
        valid_map = valid_map.at[
            level.chests_pos[:, 0],
            level.chests_pos[:, 1],
        ].set(False)
        valid_map = valid_map.at[
            level.keys_pos[:, 0],
            level.keys_pos[:, 1],
        ].set(False)
        valid_map = valid_map.at[
            level.initial_mouse_pos[0],
            level.initial_mouse_pos[1],
        ].set(False)
        

        # pick a random valid position
        valid_mask = valid_map.flatten()
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
class KeysChestsRatioLevelMutator(base.LevelMutator):
    num_keys: int
    num_chests: int

    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        # hide all but the requested number of keys
        num_keys_max, = level.hidden_keys.shape
        hidden_keys = (jnp.arange(num_keys_max) >= self.num_keys)
        # hide all but the requested number of chests
        num_chests_max, = level.hidden_chests.shape
        hidden_chests = (jnp.arange(num_chests_max) >= self.num_chests)
        return level.replace(
            hidden_keys=hidden_keys,
            hidden_chests=hidden_chests,
        )


@struct.dataclass
class ScatterMouseLevelMutator(base.LevelMutator):

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

        # fail if we hit a chest (even if it is hidden)
        hit_chest = (
            (new_initial_mouse_pos == level.chests_pos).all(axis=1)
        ).any()
        new_chests_pos = level.chests_pos
        new_initial_mouse_pos = jax.lax.select(
            hit_chest,
            level.initial_mouse_pos,
            new_initial_mouse_pos
        )

        # fail if we hit a key (even if it is hidden)
        hit_key = (
            (new_initial_mouse_pos == level.keys_pos).all(axis=1)
        ).any()
        new_keys_pos = level.keys_pos
        new_initial_mouse_pos = jax.lax.select(
            hit_key,
            level.initial_mouse_pos,
            new_initial_mouse_pos
        )
        
        return level.replace(
            wall_map=new_wall_map,
            initial_mouse_pos=new_initial_mouse_pos,
            keys_pos=new_keys_pos,
            chests_pos=new_chests_pos,
        )


@struct.dataclass
class ScatterKeyLevelMutator(base.LevelMutator):

    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape
        rng_row, rng_col, rng_key = jax.random.split(rng, num=3)

        # pick a random key
        selected_key = jax.random.choice(
            key=rng_key,
            a=level.keys_pos.shape[0],
            p=~level.hidden_keys,
        )
        old_key_pos = level.keys_pos[selected_key]
        
        # teleport to a random location within bounds
        new_key_row = jax.random.choice(
            key=rng_row,
            a=jnp.arange(1, h-1),
        )
        new_key_col = jax.random.choice(
            key=rng_col,
            a=jnp.arange(1, w-1),
        )
        new_key_pos = jnp.array((
            new_key_row,
            new_key_col,
        ))
        
        # carve through walls
        new_wall_map = level.wall_map.at[
            new_key_pos[0],
            new_key_pos[1],
        ].set(False)

        # fail if hit a chest (even a hidden one)
        hit_chest = (
           (new_key_pos == level.chests_pos).all(axis=1)
        ).any()

        new_key_pos = jax.lax.select(
            hit_chest,
            old_key_pos,
            new_key_pos,
        )

        # fail if hit existing key (even a hidden one)
        hit_key = (
           (new_key_pos == level.keys_pos).all(axis=1)
        ).any()
        new_key_pos = jax.lax.select(
            hit_key,
            old_key_pos,
            new_key_pos,
        )

        # fail if hit mouse
        hit_mouse =  (new_key_pos == level.initial_mouse_pos).all()
        new_key_pos = jax.lax.select(
            hit_mouse,
            old_key_pos,
            new_key_pos,
        )

        # update keys pos array with location of new key
        new_keys_pos = level.keys_pos.at[selected_key].set(new_key_pos)

        return level.replace(
            wall_map=new_wall_map,
            keys_pos=new_keys_pos,
        )


@struct.dataclass
class ScatterChestLevelMutator(base.LevelMutator):

    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape
        rng_row, rng_col, rng_chest = jax.random.split(rng, num=3)
        
        # pick a random chest
        selected_chest = jax.random.choice(
            key=rng_chest,
            a=level.chests_pos.shape[0],
            p=~level.hidden_chests,
        )
        old_chest_pos = level.chests_pos[selected_chest]
        
        # teleport the chest to a random location within bounds
        new_chest_row = jax.random.choice(
            key=rng_row,
            a=jnp.arange(1, h-1),
        )
        new_chest_col = jax.random.choice(
            key=rng_col,
            a=jnp.arange(1, w-1),
        )
        new_chest_pos = jnp.array((
            new_chest_row,
            new_chest_col,
        ))

        # fail if hit existing chest (even a hidden one)
        hit_chest = (
           (new_chest_pos == level.chests_pos).all(axis=1)
        ).any()
        new_chest_pos = jax.lax.select(
            hit_chest,
            old_chest_pos,
            new_chest_pos,
        )

        # fail if hit key (even a hidden one)
        hit_key = (
           (new_chest_pos == level.keys_pos).all(axis=1)
        ).any()
        new_chest_pos = jax.lax.select(
            hit_key,
            old_chest_pos,
            new_chest_pos,
        )

        # fail if hit mouse spawn
        hit_mouse = (new_chest_pos == level.initial_mouse_pos).all()
        new_chest_pos = jax.lax.select(
            hit_mouse,
            old_chest_pos,
            new_chest_pos,
        )

        # carve through walls
        new_wall_map = level.wall_map.at[
            new_chest_pos[0],
            new_chest_pos[1],
        ].set(False)
        
        # update keys pos array with location of new key
        new_chests_pos = level.chests_pos.at[selected_chest].set(new_chest_pos)

        return level.replace(
            wall_map=new_wall_map,
            chests_pos=new_chests_pos,
        )


# # # 
# Level parsing


@struct.dataclass
class LevelParser(base.LevelParser):
    """
    Level parser for Keys and Chests environment. Given some parameters
    determining level shape, provides a `parse` method that converts an
    ASCII depiction of a level into a Level struct. Also provides a
    `parse_batch` method that parses a list of level strings into a single
    vectorised Level PyTree object.

    * height (int, >= 3):
            The number of rows in the grid representing the maze
            (including top and bottom boundary rows)
    * width (int, >= 3):
            The number of columns in the grid representing the maze
            (including left and right boundary rows)
    * num_keys_max : int (>0, >= num_keys_min, <= width)
            The largest number of keys that might appear in the level.
            Note: Cannot exceed width as inventory is shown along top row.
    * num_chests_max : int (>-, >= num_chests_min)
            the largest number of chests that might appear in the level.
    * inventory_map : int[num_keys_max] (all are < width)
            The indices into the top row where successive keys are stored.
    * char_map : optional, dict{str: int}
            The keys in this dictionary are the symbols the parser will look
            to define the location of the walls and each of the items. The
            default map is as follows:
            * The character '#' maps to `Channel.WALL`.
            * The character '@' maps to `Channel.MOUSE`.
            * The character 'k' maps to `Channel.KEY`.
            * The character 'c' maps to `Channel.CHEST`.
            * The character '.' maps to `len(Channel)`, i.e. none of the
              above, representing the absence of an item.
    """
    height: int
    width: int
    num_keys_max: int
    num_chests_max: int
    inventory_map: chex.Array
    char_map = {
        '#': Channel.WALL,
        '@': Channel.MOUSE,
        'k': Channel.KEY,
        'c': Channel.CHEST,
        '.': len(Channel), # PATH
    }


    def parse(self, level_str):
        """
        Convert an ASCII string depiction of a level into a Level struct.
        For example:

        >>> p = LevelParser(height=5,width=5,num_keys_max=3,num_chests_max=3)
        >>> p.parse('''
        ... # # # # #
        ... # . k c #
        ... # @ # k #
        ... # k # c #
        ... # # # # #
        ... ''')
        Level(
            wall_map=Array([
                [1,1,1,1,1],
                [1,0,0,0,1],
                [1,0,1,0,1],
                [1,0,1,0,1],
                [1,1,1,1,1],
            ], dtype=bool),
            keys_pos=Array([[1, 2], [2, 3], [3, 1]], dtype=int32),
            chests_pos=Array([[1, 3], [3, 3], [0, 0]], dtype=int32),
            initial_mouse_pos=Array([2, 1], dtype=int32),
            inventory_map=Array([0, 1, 2], dtype=int32),
            hidden_keys=Array([False, False, False], dtype=bool),
            hidden_chests=Array([False, False,  True], dtype=bool),
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

        # extract key positions and number
        key_map = (level_map == Channel.KEY)
        num_keys = key_map.sum()
        assert num_keys <= self.num_keys_max, "too many keys"
        keys_pos = jnp.stack(
            jnp.where(key_map, size=self.num_keys_max),
            axis=1,
        )
        hidden_keys = (jnp.arange(self.num_keys_max) >= num_keys)
        
        # extract chest positions and number
        chest_map = (level_map == Channel.CHEST)
        num_chests = chest_map.sum()
        assert num_chests <= self.num_chests_max, "too many chests"
        chests_pos = jnp.stack(
            jnp.where(chest_map, size=self.num_chests_max),
            axis=1,
        )
        hidden_chests = (jnp.arange(self.num_chests_max) >= num_chests)

        # extract mouse spawn position
        mouse_spawn_map = (level_map == Channel.MOUSE)
        assert mouse_spawn_map.sum() == 1, "there must be exactly one mouse"
        initial_mouse_pos = jnp.concatenate( # cat for the mouse ;3
            jnp.where(mouse_spawn_map, size=1)
        )

        return Level(
            wall_map=wall_map,
            keys_pos=keys_pos,
            chests_pos=chests_pos,
            initial_mouse_pos=initial_mouse_pos,
            inventory_map=jnp.asarray(self.inventory_map),
            hidden_keys=hidden_keys,
            hidden_chests=hidden_chests,
        )


# # # 
# Level solving (level only)


@struct.dataclass
class LevelSolutionInit(base.LevelSolution):
    value: float # just cache the value


@struct.dataclass
class LevelSolverInit(base.LevelSolver):
    """
    Just solves the level from the initial state; does not implement the
    state-based solution methods.
    """


    @functools.partial(jax.jit, static_argnames=('self',))
    def solve(self, level: Level) -> LevelSolutionInit:
        value = _evaluate_all_visitation_sequences(
            level=level,
            discount_rate=self.discount_rate,
            penalize_time=self.env.penalize_time,
            max_steps_in_episode=self.env.max_steps_in_episode,
        )
        return LevelSolutionInit(
            value=value,
        )


    @functools.partial(jax.jit, static_argnames=('self',))
    def level_value(self, soln: LevelSolutionInit, level: Level) -> float:
        return soln.value
    
    @functools.partial(jax.jit, static_argnames=('self',))
    def state_value(self, soln: LevelSolutionInit, state: EnvState) -> float:
        raise NotImplementedError("Use LevelSolverFull")

    @functools.partial(jax.jit, static_argnames=('self',))
    def state_action_values(
        self,
        soln: LevelSolutionInit,
        state: EnvState,
    ) -> chex.Array: # float[4]
        raise NotImplementedError("Use LevelSolverFull")

    @functools.partial(jax.jit, static_argnames=('self',))
    def state_action(self, soln: LevelSolutionInit, state: EnvState) -> int:
        raise NotImplementedError("Use LevelSolverFull")


@struct.dataclass
class LevelSolverFiltered(LevelSolverInit):
    """
    Just solves the level from the initial state; does not implement the
    state-based solution methods. The difference here is that this one does
    some extra work to save computation by taking advantage of knowledge that
    the level either has a small number of keys or a small number of chests.

    THIS WILL SILENTLY BREAK IF YOU PASS IT A LEVEL THAT DOES NOT HAVE THE
    EXPECTED FORMAT, WHICH IS:
    * EITHER the first min_keys keys are non-hidden, the rest are hidden
    * OR the first min_chests chests are non-hidden, the rest are hidden
    """
    min_keys: int
    min_chests: int


    @functools.partial(jax.jit, static_argnames=('self',))
    def solve(self, level: Level) -> LevelSolutionInit:
        # solve after filtering out hidden keys
        value_filtered_keys = _evaluate_all_visitation_sequences(
            level=level.replace(
                keys_pos=level.keys_pos[:self.min_keys],
                hidden_keys=level.hidden_keys[:self.min_keys],
                # inventory map?
            ),
            discount_rate=self.discount_rate,
            penalize_time=self.env.penalize_time,
            max_steps_in_episode=self.env.max_steps_in_episode,
        )
        # solve after filtering out hidden chests
        value_filtered_chests = _evaluate_all_visitation_sequences(
            level=level.replace(
                chests_pos=level.chests_pos[:self.min_chests],
                hidden_chests=level.hidden_chests[:self.min_chests],
            ),
            discount_rate=self.discount_rate,
            penalize_time=self.env.penalize_time,
            max_steps_in_episode=self.env.max_steps_in_episode,
        )
        # which one was right? hidden_keys[i] marks slot i as HIDDEN, so the
        # number of *real* keys is the count of non-hidden slots. (See BUG-2 in
        # notes/03-bug-log.md.)
        num_real_keys = (~level.hidden_keys).sum()
        value = jnp.where(
            num_real_keys == self.min_keys,
            value_filtered_keys,
            value_filtered_chests,
        )
        return LevelSolutionInit(
            value=value,
        )
    

@jax.jit
def _evaluate_all_visitation_sequences(
    level: Level,
    discount_rate: float,
    penalize_time: bool,
    max_steps_in_episode: int,
) -> float:
    """
    Enumerate all 'plausibly-optimal' sequences of key/chest collection,
    evaluate each one, then return the optimal plan's value.

    This one assumes we don't know statically which keys/chests are hidden. If
    you have that information statically, you can save time by avoiding
    enumerating paths that visit hidden keys/chests. An easy way to do that is by
    passing this function a 'filtered level' that has only the parts you know
    to be non-hidden (e.g. truncate level.keys_pos and level.hidden_keys to the
    non-hidden parts).
    """
    K, _2 = level.keys_pos.shape
    C, _2 = level.chests_pos.shape
    N = min(K, C)

    # compute the abstract distance graph: the distance between the mouse,
    # each key, and each chest
    pos = jnp.concatenate((
        level.initial_mouse_pos[jnp.newaxis],
        level.keys_pos,
        level.chests_pos,
    ))
    dists = maze_solving.maze_distances(level.wall_map)[
        pos[:,0],
        pos[:,1],
        pos[:,[0]],
        pos[:,[1]],
    ]
    # hidden keys/chests are unreachable, give them infinite distance (this
    # means any visitation sequence that uses them will get 0 reward for it)
    hidden = jnp.concatenate((
        jnp.array([False]),
        level.hidden_keys,
        level.hidden_chests,
    ))
    dists = jnp.where(
        hidden | hidden[jnp.newaxis],
        jnp.inf,
        dists,
    )

    # enumerate visitation sequences that could plausibly be optimal
    sequences_of_keys = combinatorix.permutations(K, N)
    sequences_of_chests = combinatorix.permutations(C, N)
    sequences_of_which = combinatorix.associations(N).astype(bool)
    # (the cartesian product of these three sets of sequences gives the
    # full set of visitation sequences)

    # vmap the evaluation function over the cartesian triple product of
    # the above arrays, i.e. over all combinations of one sequence of
    # keys, one sequence of chests, and one interleaving sequence.
    # ev : *etc, int[   N], int[   N], bool[   2N], *etc. -> float[]
    # v1 : *etc, int[   N], int[   N], bool[Z, 2N], *etc. -> float[Z]
    # v2 : *etc, int[   N], int[Y, N], bool[Z, 2N], *etc. -> float[Y, Z]
    # v3 : *etc, int[X, N], int[Y, N], bool[Z, 2N], *etc. -> float[X, Y, Z]
    ev = _evaluate_visitation_sequence
    v1 = jax.vmap(ev, in_axes=(None, None, None, None, 0, None, None, None))
    v2 = jax.vmap(v1, in_axes=(None, None, None, 0, None, None, None, None))
    v3 = jax.vmap(v2, in_axes=(None, None, 0, None, None, None, None, None))
    values = v3(
        dists,
        K,
        sequences_of_keys,
        sequences_of_chests,
        sequences_of_which,
        discount_rate,
        penalize_time,
        max_steps_in_episode,
    ) # -> float[X, Y, Z]

    # report the best value available
    return values.max()


@jax.jit
def _evaluate_visitation_sequence(
    dists: chex.Array,
    num_keys: int,
    sequence_of_keys: chex.Array,
    sequence_of_chests: chex.Array,
    sequence_of_keys_or_chests: chex.Array,
    discount_rate: float,
    penalize_time: bool,
    max_steps_in_episode: int,
) -> float:
    """
    Simulate a rollout from the initial state of the level and compute how much
    reward is accumulated. Let K = num_keys, C = num_chests, N = min(K, C).

    The details of the level:

    * dists: float[1+K+C, 1+K+C]
        Distance matrix. The rows/columns represent, in order, the mouse,
        key1, ..., keyK, chest1, ..., chestC. The type is 'float' even
        though most distances are integers because some pairs are
        unreachable and for these the distance is stored as floating
        point infinity.
        It is assumed that this distance matrix corresponds to that of the
        level's wall map (but that it has been precomputed, for efficiency
        reasons).
    * num_keys: int
        Needed to compute chest indexes into dists array. Should be equal to K
        as above. Doesn't need to be static for this usage.
    
    The visitation sequence:

    * sequence_of_keys: int[N]
        A sequence of key indices, in the range 0, ..., K-1, for indexing
        into state key arrays.
    * sequence_of_chests: int[N]
        A sequence of chest indices, in the range 0, ..., C-1, for
        indexing into state chest arrays.
    * sequence_of_keys_or_chests: bool[2*N]
        A sequence of flags describing how to interleave the key sequence
        and the chest sequence. A value of 'False' indicates to go to the
        next key. A value of 'True' indicates to go to the next chest.
    
    The configuration of the reward function:

    * discount_rate: float
    * penalize_time: bool
    * max_steps_in_episode: int
    """
    @struct.dataclass
    class SimulationState:
        # mouse state
        current_node: int           # index into D
        num_keys_in_inv: int
        # visitation sequence state
        keys_visited: int           # index into sequence_of_keys
        chests_visited: int         # index into sequence_of_chests
        # evaluation state
        cumulative_distance: float  # float to handle infinities
        cumulative_reward: float
    
    # initialise simulation state based on current environment state
    initial_simulation_state = SimulationState(
        current_node=0,
        num_keys_in_inv=0,
        keys_visited=0,
        chests_visited=0,
        cumulative_distance=0.0,
        cumulative_reward=0.0,
    )
    
    # simulate one step of the plan/visitation sequence
    def step_simulation(simulation_state, chest_step):
        key_step = ~chest_step

        # where to next?
        next_key = sequence_of_keys[simulation_state.keys_visited]
        next_chest = sequence_of_chests[simulation_state.chests_visited]
        next_node = jnp.where(
            key_step,
            1 + next_key,       # transform for indexing into dists
            1 + num_keys + next_chest, # "
        )
        
        # logic for key step:
        get_key = key_step

        # logic for chest step:
        hit_chest = chest_step
        has_key = (simulation_state.num_keys_in_inv > 0)
        open_chest = has_key & hit_chest
            
        # logic for inventory
        new_num_keys_in_inv = (
            simulation_state.num_keys_in_inv
            + get_key       # increment if we picked up a key
            - open_chest    # decrement if we unlocked a chest
        )
        
        # how many maze steps?
        distance = dists[
            simulation_state.current_node,
            next_node,
        ]
        new_cumulative_distance = (
            simulation_state.cumulative_distance + distance
        )

        # compute reward delivered at this step
        raw_reward = open_chest.astype(float)
        # discount reward since it comes in the future
        discount_factor = jnp.where(
            jnp.isinf(new_cumulative_distance),
            0.0, # even if discount rate is 1.0, no reward from inf dist
            discount_rate ** new_cumulative_distance,
        )
        discounted_reward = raw_reward * discount_factor
        # modify reward based on environment configuration
        penalty_factor = jnp.where(
            penalize_time,
            # optional linearly decaying penalty factor
            1.0 - .9 * new_cumulative_distance / max_steps_in_episode,
            # or, no penalty
            1.0,
        )
        truncation_factor = jnp.where(
            new_cumulative_distance >= max_steps_in_episode,
            # if the simulation has exceeded allowed time, zero reward
            0.0,
            # else, full reward
            1.0,
        )
        reward = discounted_reward * penalty_factor * truncation_factor

        # update the carry
        new_simulation_state = SimulationState(
            current_node=next_node,
            num_keys_in_inv=new_num_keys_in_inv,
            keys_visited=simulation_state.keys_visited + key_step,
            chests_visited=simulation_state.chests_visited + chest_step,
            cumulative_distance=new_cumulative_distance,
            cumulative_reward=simulation_state.cumulative_reward + reward,
        )
        return new_simulation_state, None

    # scan this function over the steps of the plan
    final_simulation_state, _ = jax.lax.scan(
        step_simulation,
        initial_simulation_state,
        sequence_of_keys_or_chests,
    )
    return final_simulation_state.cumulative_reward


# # # 
# Level solving (full, slow)

# This code is a more complex implementation of the above, that works from any
# state. The cost is that you need to do the expensive search from each state.
# I haven't thought about whether it is possible to make it more efficient yet,
# for example maybe if you pre-compute some data the first time then you can
# have faster queries later.


@struct.dataclass
class FullLevelSolution(base.LevelSolution):
    level: Level
    directional_distances: chex.Array


@struct.dataclass
class FullLevelSolver(base.LevelSolver):


    @functools.partial(jax.jit, static_argnames=('self',))
    def solve(self, level: Level) -> FullLevelSolution:
        # compute distance between mouse and cheese
        dd = maze_solving.maze_directional_distances(level.wall_map)
        return FullLevelSolution(
            level=level,
            directional_distances=dd,
        )

    
    def _evaluate_visitation_sequence(
        self,
        dists: chex.Array,
        state: EnvState,
        sequence_of_keys: chex.Array,           # int[Combinations(K, N), N], index into dists
        sequence_of_chests: chex.Array,         # int[Combinations(C, N), N], index into dists
        sequence_of_keys_or_chests: chex.Array, # bool[Catalan(N), 2*N]
    ) -> float:
        """
        Simulate a rollout from the current state and compute how much value
        it creates.
        
        Inputs (let K = num keys, C = num chests, N = min(K, C)):

        * dists: float[1+K+C, 1+K+C]
            Distance matrix. The rows columns represent, in order, the mouse,
            key1, ..., keyK, chest1, ..., chestC. The type is 'float' even
            though most distances are integers because some pairs are
            unreachable and for these the distance is stored as floating
            point infinity.
        * state: EnvState
            The state from which the plan to be evaluated is to begin. Used
            for checking which keys/chests have already been collected, for
            example.
            Note that the dists could have been computed from the state, but
            are pre-computed for efficiency reasons. It is assumed that the
            dists correspond to this state.
        * sequence_of_keys: int[N]
            A sequence of key indices, in the range 0, ..., K-1, for indexing
            into state key arrays.
        * sequence_of_chests: int[N]
            A sequence of chest indices, in the range 0, ..., C-1, for
            indexing into state chest arrays.
        * sequence_of_keys_or_chests: bool[2*N]
            A sequence of flags describing how to interleave the key sequence
            and the chest sequence. A value of 'False' indicates to go to the
            next key. A value of 'True' indicates to go to the next chest.
        
        The final three arguments represent a 'visitation sequence' or
        'plan'. Given such a plan and a starting state we can simulate the
        mouse's path through the environment and figure out how much
        discounted reward it would get. This function does that.
        """
        K, _2 = state.level.keys_pos.shape
        
        @struct.dataclass
        class SimulationState:
            # mouse state
            current_node: int           # index into D
            num_keys_in_inv: int
            # visitation sequence state
            keys_visited: int           # index into sequence_of_keys
            chests_visited: int         # index into sequence_of_chests
            # evaluation state
            cumulative_distance: float  # float to handle infinities
            cumulative_reward: float
        
        # initialise simulation state based on current environment state
        initial_simulation_state = SimulationState(
            current_node=0,
            num_keys_in_inv=jnp.sum(state.got_keys & ~state.used_keys),
            keys_visited=0,
            chests_visited=0,
            cumulative_distance=0.0,
            cumulative_reward=0.0,
        )
        
        # simulate one step of the plan/visitation sequence
        def step_simulation(simulation_state, chest_step):
            key_step = ~chest_step

            # where to next?
            next_key = sequence_of_keys[simulation_state.keys_visited]
            next_chest = sequence_of_chests[simulation_state.chests_visited]
            next_node = jnp.where(
                key_step,
                1 + next_key,       # transform for indexing into dists
                1 + K + next_chest, # "
            )
            
            # logic for key step:
            skip_key = (
                key_step
                & state.got_keys[next_key]
                & ~state.level.hidden_keys[next_key]
            )
            get_key = (
                key_step
                & ~skip_key
                & ~state.level.hidden_keys[next_key]
            )

            # logic for chest step:
            skip_chest = (
                chest_step
                & state.got_chests[next_chest]
                & ~state.level.hidden_chests[next_chest]
            )
            hit_chest = (
                chest_step
                & ~skip_chest
                & ~state.level.hidden_chests[next_chest]
            )
            has_key = (simulation_state.num_keys_in_inv > 0)
            open_chest = has_key & hit_chest
                
            # logic for inventory
            new_num_keys_in_inv = (
                simulation_state.num_keys_in_inv
                + get_key       # increment if we picked up a key
                - open_chest    # decrement if we unlocked a chest
            )
            
            # how many maze steps?
            distance = dists[
                simulation_state.current_node,
                next_node,
            ]
            new_cumulative_distance = jnp.where(
                skip_key | skip_chest,
                simulation_state.cumulative_distance,
                simulation_state.cumulative_distance + distance,
            )
            next_node = jnp.where(
                skip_key | skip_chest,
                simulation_state.current_node,
                next_node,
            )

            # compute reward delivered at this step
            raw_reward = open_chest.astype(float)
            # discount reward since it comes in the future
            discount_factor = self.discount_rate ** new_cumulative_distance
            discounted_reward = raw_reward * discount_factor
            # modify reward based on environment configuration
            virtual_time = state.steps + new_cumulative_distance
            penalty_factor = jnp.where(
                self.env.penalize_time,
                # optional linearly decaying penalty factor
                1.0 - .9 * virtual_time / self.env.max_steps_in_episode,
                # or, no penalty
                1.0,
            )
            truncation_factor = jnp.where(
                virtual_time >= self.env.max_steps_in_episode,
                # if the simulation has exceeded allowed time, zero reward
                0.0,
                # else, full reward
                1.0,
            )
            reward = discounted_reward * penalty_factor * truncation_factor

            # update the carry
            new_simulation_state = SimulationState(
                current_node=next_node,
                num_keys_in_inv=new_num_keys_in_inv,
                keys_visited=simulation_state.keys_visited + key_step,
                chests_visited=simulation_state.chests_visited + chest_step,
                cumulative_distance=new_cumulative_distance,
                cumulative_reward=simulation_state.cumulative_reward + reward,
            )
            return new_simulation_state, new_simulation_state

        # scan this function over the steps of the plan
        final_simulation_state, trace = jax.lax.scan(
            step_simulation,
            initial_simulation_state,
            sequence_of_keys_or_chests,
        )
        return final_simulation_state.cumulative_reward


    def _plan(
        self,
        soln: FullLevelSolution,
        state: EnvState,
    ) -> tuple[
        # value
        float,
        # visitation sequence / plan
        tuple[
            chex.Array,
            chex.Array,
            chex.Array,
        ],
    ]:
        """
        Enumerate all 'plausibly-optimal' sequences of key/chest collection,
        evaluate each one, then return the optimal plan and its value.
        """
        K, _2 = state.level.keys_pos.shape
        C, _2 = state.level.chests_pos.shape
        N = min(K, C)

        # compute the abstract distance graph: the distance between the mouse,
        # each key, and each chest
        pos = jnp.concatenate((
            state.mouse_pos[jnp.newaxis],
            state.level.keys_pos,
            state.level.chests_pos,
        ))
        dists = soln.directional_distances[
            pos[:,0],
            pos[:,1],
            pos[:,[0]],
            pos[:,[1]],
            4, # distance from here (rather than after moving in a direction)
        ]

        # enumerate visitation sequences that could plausibly be optimal
        sequences_of_keys = combinatorix.permutations(K, N)
        sequences_of_chests = combinatorix.permutations(C, N)
        sequences_of_which = combinatorix.associations(N).astype(bool)
        # (the cartesian product of these three sets of sequences gives the
        # full set of visitation sequences)

        # vmap the evaluation function over the cartesian triple product of
        # the above arrays, i.e. over all combinations of one sequence of
        # keys, one sequence of chests, and one interleaving sequence.
        ev = functools.partial(
            self._evaluate_visitation_sequence,
            dists,
            state,
        )
        v1 = jax.vmap(ev, in_axes=(None, None, 0)) # :   N,   N, Z 2N ->     Z
        v2 = jax.vmap(v1, in_axes=(None, 0, None)) # :   N, Y N, Z 2N ->   Y Z
        v3 = jax.vmap(v2, in_axes=(0, None, None)) # : X N, Y N, Z 2N -> X Y Z

        # apply the vmapped function to get an array of values
        values = v3(
            sequences_of_keys,
            sequences_of_chests,
            sequences_of_which,
        ) # -> float[X, Y, Z]
        # multidimensional argmax (note: breaks ties by lowest index)
        i, j, k = jnp.unravel_index(
            jnp.argmax(values),
            values.shape,
        )
        
        value = values[i, j, k]
        plan = (
            sequences_of_keys[i],
            sequences_of_chests[j],
            sequences_of_which[k],
        )
        return value, plan


    @functools.partial(jax.jit, static_argnames=('self',))
    def state_value(self, soln: FullLevelSolution, state: EnvState) -> float:
        # find the value of the best plan from this state using the helper
        value, _plan = self._plan(soln, state)
        return value


    @functools.partial(jax.jit, static_argnames=('self',))
    def state_action_values(
        self,
        soln: FullLevelSolution,
        state: EnvState,
    ) -> chex.Array: # float[4]
        # TODO: I guess it requires to simulate each action and then evaluate
        # the resulting states separately? this is not necessary for now...!
        raise NotImplementedError("TODO")


    @functools.partial(jax.jit, static_argnames=('self',))
    def state_action(self, soln: FullLevelSolution, state: EnvState) -> int:
        """
        Optimal action from a given state.

        Parameters:

        * soln : FullLevelSolution
                The output of `solve` method for this level.
        * state : EnvState
                The state to compute the optimal action for.
            
        Return:

        * action : int
                An optimal action from the given state.
                
        Notes:

        * If there are multiple equally optimal actions, this method will
          systematically prefer one or another in a complex way depending on
          the implementation. Currently, it finds the first optimal plan
          based on the order the plans happen to be enumerated, and the first
          optimal action for carrying out the first step of that plan
          (according to the order up (0), left (1), down (2), or right (3)).
        * As a special case of this, if there is no achievable value, the
          best plan would be the first plan, which might involve moving the
          mouse towards keys or chests that are unreachable, disabled or have
          already been collected.
        """
        # find an optimal plan from this state using the helper
        _value, plan = self._plan(soln, state)
        sequence_of_keys, sequence_of_chests, sequence_of_keys_or_chests = plan

        # identify the first not-yet-taken step of the plan
        N, = sequence_of_keys.shape
        skip_keys_mask = state.got_keys[sequence_of_keys]
        skip_chests_mask = state.got_chests[sequence_of_chests]
        skip_which_mask = (jnp.zeros(2*N, dtype=bool)
            .at[jnp.where(sequence_of_keys_or_chests, size=N)]
            .set(skip_chests_mask)
            .at[jnp.where(~sequence_of_keys_or_chests, size=N)]
            .set(skip_keys_mask)
        )
        next_key = jnp.argmin(skip_keys_mask) # id of first False (no skip)
        next_chest = jnp.argmin(skip_chests_mask)
        next_which = jnp.argmin(skip_which_mask)

        # identify the target position to execute this step
        target_pos = jnp.where(
            sequence_of_keys_or_chests[next_which],
            state.level.chests_pos[sequence_of_chests[next_chest]],
            state.level.keys_pos[sequence_of_keys[next_key]],
        )

        # choose the action that steps the mouse towards that position
        action = jnp.argmin(soln.directional_distances[
            state.mouse_pos[0],
            state.mouse_pos[1],
            target_pos[0],
            target_pos[1],
            :4, # only consider up/left/down/right, not 'stay' dimension
        ])
        return action


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
        

        def count_reachable_keys_and_chests(level):
            # solve the maze
            dists = maze_solving.maze_distances(level.wall_map)
            # count reachable keys
            keys_dists = dists[
                level.initial_mouse_pos[0],
                level.initial_mouse_pos[1],
                level.keys_pos[:, 0],
                level.keys_pos[:, 1],
            ]
            reachable_keys = ~jnp.isinf(keys_dists)
            num_reachable_keys = jnp.sum(reachable_keys & ~level.hidden_keys)
            # count reachable chests
            chests_dists = dists[
                level.initial_mouse_pos[0],
                level.initial_mouse_pos[1],
                level.chests_pos[:, 0],
                level.chests_pos[:, 1],
            ]
            reachable_chests = ~jnp.isinf(chests_dists)
            num_reachable_chests = jnp.sum(reachable_chests & ~level.hidden_chests)
            return num_reachable_keys, num_reachable_chests
        num_reachable_keys, num_reachable_chests = jax.vmap(count_reachable_keys_and_chests)(levels)
        

        def count_visible_keys_and_chests(level):
            num_visible_keys = jnp.sum(~level.hidden_keys)
            num_visible_chests = jnp.sum(~level.hidden_chests)
            return num_visible_keys, num_visible_chests
        num_visible_keys, num_visible_chests = jax.vmap(count_visible_keys_and_chests)(levels)

        # num walls (excluding border)
        inner_wall_maps = levels.wall_map[:,1:-1,1:-1]
        num_walls = jnp.sum(inner_wall_maps, axis=(1,2))

        # rendered levels in a grid
        def render_level(level):
            state = self.env._reset(level)
            rgb = self.env.render_state(state)
            return rgb
        _, top_64_level_ids = jax.lax.top_k(weights, k=64)
        top_64_levels = jax.tree.map(
            lambda leaf: leaf[top_64_level_ids],
            levels,
        )
        rendered_levels = jax.vmap(render_level)(top_64_levels)
        rendered_levels_pad = jnp.pad(
            rendered_levels,
            pad_width=((0, 0), (0, 1), (0, 1), (0, 0)),
        )
        rendered_levels_grid = einops.rearrange(
            rendered_levels_pad,
            '(level_h level_w) h w c -> (level_h h) (level_w w) c',
            level_w=8,
        )[:-1,:-1] # cut off last pixel of padding

        return {
            'layout': {
                'levels64_img': rendered_levels_grid,
                # number of walls
                'num_walls_avg': num_walls.mean(),
            },
            'counts': {
                'num_visible_keys_avg': num_visible_keys.mean(),
                'num_visible_chests_avg': num_visible_chests.mean(),
                'num_reachable_keys_avg': num_reachable_keys.mean(),
                'num_reachable_chests_avg': num_reachable_chests.mean(),
            },
        }


