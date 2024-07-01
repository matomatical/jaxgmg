"""
Parameterised environment and level generator for 'follow me' problem. Key
components are as follows.

Structs:

* The `Level` struct represents a particular maze layout and beacon/mouse
  spawn position.
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

from typing import Tuple, Dict
import enum
import functools

import jax
import jax.numpy as jnp
import chex
import einops
from flax import struct

from jaxgmg.procgen import maze_generation as mg
from jaxgmg.procgen import maze_solving
from jaxgmg.environments import base


@struct.dataclass
class Level(base.Level):
    """
    Represent a particular environment layout:

    * wall_map : bool[h, w]
            Maze layout (True = wall)
    * beacons_pos: index[b, 2]
            Coordinates of the three beacon (index into `wall_map`)
    * initial_leader_pos : index[2]
            Coordinates of initial leader position (index into `wall_map`)
    * leader_order : index[b]
            Order in which the leader visits the beacons.
    * dir_map : int[h, w, b]
            Cached solution to the maze. The first two axes specify the
            leader location. The third axis specifies a beacon index. The
            value is an integer indicating the optimal direction for the
            leader to move to get to that beacon, namely (0) move up, (1)
            move left, (2) move down, (3) move right, or (4) stay put.
    * initial_mouse_pos : index[2]
            Coordinates of initial mouse position (index into `wall_map`)
    
    TODO: Consider a version with a generic number of beacons?
    """
    wall_map: chex.Array
    beacons_pos: chex.Array
    initial_leader_pos: chex.Array
    leader_order: chex.Array
    dir_map: chex.Array
    initial_mouse_pos: chex.Array


@struct.dataclass
class EnvState(base.EnvState):
    """
    Dynamic environment state within a particular level.

    * leader_pos : index[2]
            Current coordinates of the leader. Initialised to
            `level.initial_leader_pos`.
    * leader_next_beacon_id : int
            Index into leader beacon order of the next beacon the leader
            intends to move towards.
    * mouse_pos : index[2]
            Current coordinates of the mouse. Initialised to
            `level.initial_mouse_pos`.
    * mouse_next_beacon_id : int
            Index into beacon order of the next beacon the mouse is supposed
            to move towards for the reward (the visibly active beacon).
    """
    leader_pos: chex.Array
    leader_next_beacon_id: int
    mouse_pos: chex.Array
    mouse_next_beacon_id: int


class Env(base.Env):
    """
    Follow me environment.

    In this environment the agent controls a mouse navigating a grid-based
    maze. The mouse must must navigate the maze to a range of beacons that
    visibly activate in a repeating sequence. There is a second mouse in the
    maze, the 'leader', that also pursues the beacons, usually in the same
    order as they activate, so the mouse can follow the leader to solve the
    task. However, in some configurations, the leader's beacon visit order is
    distinct from the beacon activation order.

    There are four available actions which deterministically move the mouse
    one grid square up, right, down, or left respectively.
    * If the mouse would hit a wall it remains in place.
    * The mouse can pass through beacons and the leader.

    When the mouse hits an active beacon, it receives reward, the beacon
    deactivates, and the next beacon in the sequence activates.

    Observations come in one of two formats:

    * Boolean: a H by W by C bool array where each channel represents the
      presence of one type of thing (walls, mouse, leader, beacon, active
      beacon).
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
        * `LEADER`: True in the one location the leader occupies.
        * `BEACON`: True in the locations occupied by a beacon.
        * `ACTIVE`: True in the one location occupied by the currently active
                    beacon.
        """
        WALL    = 0
        MOUSE   = 1
        LEADER  = 2
        BEACON  = 3
        ACTIVE  = 4

    
    def _reset(
        self,
        level: Level,
    ) -> EnvState:
        return EnvState(
            leader_pos=level.initial_leader_pos,
            leader_next_beacon_id=0,
            mouse_pos=level.initial_mouse_pos,
            mouse_next_beacon_id=0,
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
        # constants
        steps = jnp.array((
            (-1,  0),   # up
            ( 0, -1),   # left
            (+1,  0),   # down
            ( 0, +1),   # right
            ( 0,  0),   # stay
        ))
        num_beacons, _2 = state.level.beacons_pos.shape

        # update leader position
        leader_action = state.level.dir_map[
            state.leader_pos[0],
            state.leader_pos[1],
            state.leader_next_beacon_id,
        ]
        # (cached action will never hit walls or use 'stay' action, just go)
        next_leader_pos = state.leader_pos + steps[leader_action]
        state = state.replace(
            leader_pos=state.leader_pos + steps[leader_action]
        )

        # check if leader got to its next beacon
        leader_next_beacon_pos = (
            state.level.beacons_pos[
                state.level.leader_order[state.leader_next_beacon_id]
            ]
        )
        leader_got_beacon = (state.leader_pos == leader_next_beacon_pos).all()
        # if so, increment next beacon id for leader (NOTE: WITH WRAPAROUND)
        state = state.replace(leader_next_beacon_id=
            (state.leader_next_beacon_id + leader_got_beacon) % num_beacons,
        )
        
        # update mouse position
        ahead_pos = state.mouse_pos + steps[action]
        hit_wall = state.level.wall_map[ahead_pos[0], ahead_pos[1]]
        state = state.replace(
            mouse_pos=jax.lax.select(
                hit_wall,
                state.mouse_pos,
                ahead_pos,
            )
        )

        # check if mouse got to next beacon
        next_beacon_pos = state.level.beacons_pos[state.mouse_next_beacon_id]
        got_next_beacon = (state.mouse_pos == next_beacon_pos).all()
        # if so, increment next beacon id for mouse (NOTE: NO WRAPAROUND)
        state = state.replace(mouse_next_beacon_id=
            (state.mouse_next_beacon_id + got_next_beacon),
        )

        # reward and done
        reward = got_next_beacon.astype(float)
        done = (state.mouse_next_beacon_id == num_beacons)

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
        
        # render leader
        obs = obs.at[
            state.leader_pos[0],
            state.leader_pos[1],
            Env.Channel.LEADER,
        ].set(True)
        
        # render beacons
        obs = obs.at[
            state.level.beacons_pos[:, 0],
            state.level.beacons_pos[:, 1],
            Env.Channel.BEACON,
        ].set(True)

        # render active beacon
        active_beacon_pos = state.level.beacons_pos[state.mouse_next_beacon_id]
        # note: this active_beacon_pos is invalidated after id exceeds range,
        # but we don't draw in that case anyway so it's fine.
        obs = obs.at[
            active_beacon_pos[0],
            active_beacon_pos[1],
            Env.Channel.ACTIVE,
        ].set(~state.done)
        
        return obs


    @functools.partial(jax.jit, static_argnames=('self',))
    def _get_obs_rgb(
        self,
        state: EnvState,
        spritesheet: Dict[str, chex.Array],
    ) -> chex.Array:
        """
        Return an RGB observation based on a grid of tiles from the given
        spritesheet.
        """
        # get the boolean grid representation of the state
        obs = self._get_obs_bool(state)
        H, W, _C = obs.shape
        C = Env.Channel

        # find out, for each position, which object to render
        inactive = obs[:,:,C.BEACON] & ~obs[:,:,C.ACTIVE]
        both_mice = obs[:,:,C.MOUSE] & obs[:,:,C.LEADER]
        # note: mouse & active beacon is not possible
        # (for each position pick the first true index top-down this list)
        sprite_priority_vector_grid = jnp.stack([
            # combinations
            inactive & both_mice,
            obs[:,:,C.LEADER] & obs[:,:,C.ACTIVE],
            obs[:,:,C.LEADER] & inactive,
            obs[:,:,C.MOUSE] & inactive,
            both_mice,
            # individual entities
            obs[:,:,C.ACTIVE],
            inactive,
            obs[:,:,C.LEADER],
            obs[:,:,C.MOUSE],
            obs[:,:,C.WALL],
            # no objects, 'default' (always true)
            jnp.ones((H, W), dtype=bool),
        ])
        chosen_sprites = jnp.argmax(sprite_priority_vector_grid, axis=0)

        # put the corresponding sprite into each square
        spritemap = jnp.stack([
            # combinations
            spritesheet['BOTH_MICE_ON_BEACON_OFF'],
            spritesheet['LEADER_ON_BEACON_ON'],
            spritesheet['LEADER_ON_BEACON_OFF'],
            spritesheet['MOUSE_ON_BEACON_OFF'],
            spritesheet['MOUSE_ON_LEADER'],
            # individual entities
            spritesheet['BEACON_ON'],
            spritesheet['BEACON_OFF'],
            spritesheet['LEADER_MOUSE'],
            spritesheet['MOUSE'],
            spritesheet['WALL'],
            # no objects
            spritesheet['PATH'],
        ])[chosen_sprites]
        image = einops.rearrange(
            spritemap,
            'h w th tw rgb -> (h th) (w tw) rgb',
        )

        return image


@struct.dataclass
class LevelGenerator(base.LevelGenerator):
    """
    Level generator for Cheese on a Dish environment. Given some maze
    configuration parameters and cheese location parameter, provides a
    `sample` method that generates a random level.

    * height : int,(>= 3, odd)
            The number of rows in the grid representing the maze
            (including top and bottom boundary rows).
    * width : int (>= 3, odd)
            The number of columns in the grid representing the maze
            (including left and right boundary rows).
    * maze_generator : maze_generation.MazeGenerator
            Provides the maze generation method to use (see module
            `maze_generation` for details).
            The default is a tree maze generator using Kruskal's algorithm.
    * num_beacons : int (>=0)
            This many beacons will spawn throughout the maze.
    * trustworthy_leader : bool (default True)
            Whether the leader follows the beacons in the correct order.
    """
    height : int = 13
    width : int = 13
    num_beacons : int = 3
    maze_generator : mg.MazeGenerator = mg.EdgeMazeGenerator()
    trustworthy_leader : bool = True

    def __post_init(self):
        assert num_beacons >= 1, "num_beacons must be positive"


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
        
        # spawn random mouse pos, leader pos, beacons pos
        rng_spawn, rng = jax.random.split(rng)
        coords = einops.rearrange(
            jnp.indices((self.height, self.width)),
            'c h w -> (h w) c',
        )
        all_pos = jax.random.choice(
            key=rng_spawn,
            a=coords,
            shape=(2 + self.num_beacons,),
            axis=0,
            p=~wall_map.flatten(),
            replace=False,
        )
        initial_mouse_pos = all_pos[0]
        initial_leader_pos = all_pos[1]
        beacons_pos = all_pos[2:]

        # solve maze for leader and cache solution
        dir_map = maze_solving.maze_optimal_directions(
            wall_map,
            stay_action=True,
        )[
            :,
            :,
            beacons_pos[:, 0],
            beacons_pos[:, 1],
        ]

        # determine leader order
        rng_leader_order, rng = jax.random.split(rng)
        if self.trustworthy_leader:
            leader_order = jnp.arange(self.num_beacons)
        else:
            leader_order = jax.random.permutation(
                key=rng_leader_order,
                x=self.num_beacons,
            )

        # done
        return Level(
            wall_map=wall_map,
            beacons_pos=beacons_pos,
            initial_leader_pos=initial_leader_pos,
            leader_order=leader_order,
            dir_map=dir_map,
            initial_mouse_pos=initial_mouse_pos,
        )


@struct.dataclass
class LevelParser:
    """
    Level parser for Follow Me environment. Given some parameters determining
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
    * num_beacons : int
            The number of beacons in the level.
    * leader_order : arraylike of type int of length num_beacons
            The order in which the leader follows the beacons.
    * char_map : optional, dict{str: int}
            The keys in this dictionary are the symbols the parser will look
            to define the location of the walls and each of the items. The
            default map is as follows:
            * The character '#' maps to `Env.Channel.WALL`.
            * The character '@' maps to `Env.Channel.MOUSE`.
            * The character '*' maps to `Env.Channel.LEADER`.
            * The charcters '0' through '9' map to 'Env.Channel.BEACON`.
            * The character '.' maps to `len(Env.Channel)`, i.e. none of the
              above, representing the absence of an item.
            The beacon order is determined by the lexicographic order of the
            symbols used that map to `Env.Channel.BEACON`.
    """
    height: int
    width: int
    num_beacons: int
    leader_order: tuple
    char_map = {
        '#': Env.Channel.WALL,
        '@': Env.Channel.MOUSE,
        '*': Env.Channel.LEADER,
        # the following all map to BEACON, the lexicographic order is used to
        # determine the beacon order.
        '0': Env.Channel.BEACON, '1': Env.Channel.BEACON,
        '2': Env.Channel.BEACON, '3': Env.Channel.BEACON,
        '4': Env.Channel.BEACON, '5': Env.Channel.BEACON,
        '6': Env.Channel.BEACON, '7': Env.Channel.BEACON,
        '8': Env.Channel.BEACON, '9': Env.Channel.BEACON,
        '.': len(Env.Channel), # PATH
    }


    def parse(self, level_str):
        """
        Convert an ASCII string depiction of a level into a Level struct.
        For example:

        >>> p = LevelParser(
        ...     height=7,
        ...     width=7,
        ...     num_beacons=3,
        ...     leader_order=(0,1,2),
        ... )
        >>> p.parse('''
        ... # # # # # # #
        ... # @ # . . 1 #
        ... # . # . # # #
        ... # . . . . 0 #
        ... # * # . # # #
        ... # . # . . 2 #
        ... # # # # # # #
        ... ''')
        Level(
            wall_map=Array([
                [1,1,1,1,1,1,1],
                [1,0,1,0,0,0,1],
                [1,0,1,0,1,1,1],
                [1,0,0,0,0,0,1],
                [1,0,1,0,1,1,1],
                [1,0,1,0,0,0,1],
                [1,1,1,1,1,1,1],
            ], dtype=bool),
            beacons_pos=Array([[3,5],[1,5],[5,5]]),
            initial_leader_pos=Array([4,1]),
            leader_order=Array([0,1,2]),
            dir_map=Array([[[...]]]),
            initial_mouse_pos=Array([1,1]),
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
    
        # extract beacon positions and number
        beacon_map = (level_map == Env.Channel.BEACON)
        num_beacons = beacon_map.sum()
        assert num_beacons == self.num_beacons, "wrong number of beacons"
        unordered_beacons_pos = jnp.stack(
            jnp.where(beacon_map, size=self.num_beacons),
            axis=1,
        )
        # extract beacon order!
        ord_map = jnp.asarray([
            [ord(e) for e in line.split()]
            for line in level_str.strip().splitlines()
        ])
        beacon_order = jnp.argsort(ord_map[
            unordered_beacons_pos[:,0],
            unordered_beacons_pos[:,1],
        ])
        beacons_pos = unordered_beacons_pos[beacon_order]
        
        # extract leader spawn position
        leader_spawn_map = (level_map == Env.Channel.LEADER)
        assert leader_spawn_map.sum() == 1, "there must be exactly one leader"
        initial_leader_pos = jnp.concatenate(
            jnp.where(leader_spawn_map, size=1)
        )

        # extract mouse spawn position
        mouse_spawn_map = (level_map == Env.Channel.MOUSE)
        assert mouse_spawn_map.sum() == 1, "there must be exactly one mouse"
        initial_mouse_pos = jnp.concatenate(
            jnp.where(mouse_spawn_map, size=1)
        )
        
        # solve the wall map for the leader
        dir_map = maze_solving.maze_optimal_directions(
            wall_map,
            stay_action=True,
        )[
            :,
            :,
            beacons_pos[:, 0],
            beacons_pos[:, 1],
        ]

        return Level(
            wall_map=wall_map,
            beacons_pos=beacons_pos,
            initial_leader_pos=initial_leader_pos,
            leader_order=jnp.asarray(self.leader_order),
            dir_map=dir_map,
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
            wall_map=jnp.stack(
                [l.wall_map for l in levels]
            ),
            beacons_pos=jnp.stack(
                [l.beacons_pos for l in levels]
            ),
            initial_leader_pos=jnp.stack(
                [l.initial_leader_pos for l in levels]
            ),
            leader_order=jnp.stack(
                [l.leader_order for l in levels]
            ),
            initial_mouse_pos=jnp.stack(
                [l.initial_mouse_pos for l in levels]
            ),
            dir_map=jnp.stack(
                [l.dir_map for l in levels]
            ),
        )
