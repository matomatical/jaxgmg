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

        # Calculate the number of keys collected (excluding hidden keys)
        keys_collected = (state.got_keys & ~state.level.hidden_keys).sum()

        # Calculate the number of keys used (excluding hidden keys)
        keys_used = (state.used_keys & ~state.level.hidden_keys).sum()

        # Calculate the number of keys currently in inventory
        keys_in_inventory = keys_collected - keys_used

        # Calculate the number of chests remaining to be opened (excluding hidden chests)
        chests_remaining = ((~state.got_chests) & ~state.level.hidden_chests).sum()

        # Calculate the excess keys collected beyond what is needed to open remaining chests



        return (
            state,
            reward,
            done,
            {
                'proxy_rewards': {
                    'keys': keys_in_inventory,
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


    @functools.partial(jax.jit, static_argnames=('self',))
    def optimal_value(
        self,
        level: Level,
        discount_rate: float,
    ) -> float:
        """
        Compute the optimal return from a given level (initial state) for a
        given discount rate. Respects time penalties to the reward and max
        episode length.

        The approach is simplistic---brute force & vectorised evaluation of
        all Hamiltonian paths through the abstract key/chest network and
        select the best one. This will be enough for small numbers of keys
        and chests, but will not scale well in time or memory requirements.

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
        * With many keys and chests, the algorithm mighy be slow to compile
          and run and may use a lot of memory. This is because it relies on
          a brute force approach of statically generating and then processing
          a factorial number of key/chest visitation sequences.
        
        TODO:

        * Support computing the return from arbitrary states. This would be
          not too much harder, requiring I think just the following:
          * Use the current mouse position instead of the initial one.
          * Mask out keys and chests that have already been collected.
          * Initialise the simulation using the current number of
            collected/unused keys (rather than 0).
          * Decrease time remaining from max by current number of steps.
        * Find a more clever way to generate only feasibly optimal solutions,
          such as by considering only combinations of the maximum available
          number of (non-hidden) chests and the correponding number of
          (non-hidden) keys, and in an order that also ensures the chests are
          visited when the inventory is nonempty (like balanced parentheses).
        * Generally there is a lot more redundant computation going on here,
          for example the return over subsequences are repeatedly computed
          many times, it should be possible to speed things up with a dynamic
          programming approach, and this might even be somewhat jittable.
        
        Having said all that---seems like this will be enough for now.
        """
        # compute distances between the mouse, and each key, and each chest
        dist = maze_solving.maze_distances(level.wall_map)
        pos = jnp.concatenate((
            level.initial_mouse_pos[jnp.newaxis],
            level.keys_pos,
            level.chests_pos,
        ))
        D = dist[pos[:,0],pos[:,1],pos[:,[0]],pos[:,[1]]]

        # (statically) generate a vector of possible visitation sequences
        # * each is a vector of indices into the above distance matrix
        # * the vectors are 1-based because index 0 in the distance matrix
        #   represents the mouse, the implicit start of all visitation
        #   sequences
        num_keys, _2 = level.keys_pos.shape
        num_chests, _2 = level.chests_pos.shape
        visitation_sequences = jnp.array(
            list(itertools.permutations(range(1, 1+num_keys+num_chests))),
            dtype=int,
        )

        # step through each visitation sequence in parallel, computing the
        # return value for that sequence
        num_sequences, _ = visitation_sequences.shape
        initial_carry = (
            jnp.zeros(num_sequences),            # path length so far
            jnp.zeros(num_sequences),            # return so far
            jnp.zeros(num_sequences, dtype=int), # idx of prev visited thing
            jnp.zeros(num_sequences, dtype=int), # inventory num keys so far
        )
        def _step(carry, visit_id):
            length, value, last_visit_id, inventory = carry
            
            # keep track of how many steps
            dist = D[last_visit_id, visit_id]
            new_length = length + dist

            # if we hit a chest, and we have a key, provide reward (1)
            hit_chest = (visit_id >= 1 + num_keys)
            hit_real_chest = (
                hit_chest
                & ~level.hidden_chests[visit_id-1-num_keys]
                # note: invalid index if hit_chest is false, ok because of &
            )
            has_key = (inventory > 0)
            reward = (hit_real_chest & has_key).astype(float)
            # optional time penalty to reward
            penalized_reward = jnp.where(
                self.penalize_time,
                (1.0 - 0.9 * new_length / self.max_steps_in_episode) * reward,
                reward,
            )
            # mask out rewards beyond the end of the episode
            episodes_still_valid = (new_length < self.max_steps_in_episode)
            valid_reward = penalized_reward * episodes_still_valid
            # contribute to return
            discounted_reward = (discount_rate ** new_length) * valid_reward
            new_value = value + discounted_reward

            # if we hit a key, gain a key; if we hit a chest, use a key
            hit_key = (visit_id >= 1) & (visit_id <= num_keys)
            hit_real_key = (
                hit_key
                & ~level.hidden_keys[visit_id-1]
                # note: invalid index if hit_key is false, ok because of &
            )
            new_inventory = (
                inventory
                + hit_real_key
                - (hit_real_chest & has_key)
            )

            new_carry = (
                new_length,
                new_value,
                visit_id,
                new_inventory,
            )
            return new_carry, None

        final_carry, _ = jax.lax.scan(
            _step,
            initial_carry,
            visitation_sequences.T,
        )
        
        # the highest return among all of these sequences must be the optimal
        # return
        _, values, _, _ = final_carry
        
        return values.max()


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
    * num_keys_min : int (>= 0)
            the smallest number of keys to randomly place in each generated
            maze. the actual number will be uniformly random between this and
            num_keys_max (inclusive).
    * num_keys_max : int (>0, >= num_keys_min)
            the largest number of keys to randomly place in each generated
            maze. the actual number will be uniformly random between this and
            num_keys_min (inclusive).
            this one also determines the shape of the key-related arrays in
            the level struct.
    * num_chests_min : int (>= 0)
            the smallest number of chests to randomly place in each generated
            maze. the actual number will be uniformly random between this and
            num_chests_max (inclusive).
    * num_chests_max : int (>-, >= num_chests_min)
            the largest number of chests to randomly place in each generated
            maze. the actual number will be uniformly random between this and
            num_chests_min (inclusive).
            this one also determines the shape of the chest-related arrays in
            the level struct.
    """
    height: int = 13
    width: int = 13
    maze_generator : mg.MazeGenerator = mg.TreeMazeGenerator()
    num_keys_min: int = 1
    num_keys_max: int = 4
    num_chests_min: int = 4
    num_chests_max: int = 4

    def __post_init__(self):
        assert self.num_keys_min >= 0
        assert self.num_keys_max > 0
        assert self.num_keys_max >= self.num_keys_min
        assert self.num_chests_min >= 0
        assert self.num_chests_max > 0
        assert self.num_chests_max >= self.num_chests_min
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

        # decide randomly how many keys to use
        rng_hidden_keys, rng = jax.random.split(rng)
        num_keys = jax.random.randint(
            key=rng_hidden_keys,
            shape=(),
            minval=self.num_keys_min,
            maxval=self.num_keys_max+1, # exclusive
        )
        # hide keys after that many
        hidden_keys = (jnp.arange(self.num_keys_max) >= num_keys)
        
        # decide randomly how many chests to use
        rng_hidden_chests, rng = jax.random.split(rng)
        num_chests = jax.random.randint(
            key=rng_hidden_chests,
            shape=(),
            minval=self.num_chests_min,
            maxval=self.num_chests_max+1, # exclusive
        )
        # hide chests after that many
        hidden_chests = (jnp.arange(self.num_chests_max) >= num_chests)
    
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


