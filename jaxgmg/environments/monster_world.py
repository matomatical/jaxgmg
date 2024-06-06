"""
Parameterised environment and level generator for monster gridworld problem.
Key components are as follows.

Structs:

* The `Level` struct represents a particular maze layout and monster/shields
  /apple/mouse spawn positions.
* The `EnvState` struct represents a specific dynamic state of the
  environment.

Classes:

* `Env` class, provides `reset`, `step`, and `render` methods in a
  gymnax-style interface (see `base` module for specifics of the interface).
* `LevelGenerator` class, provides `sample` method for randomly sampling a
  level from a configurable level distribution.
* `LevelMutator` class, provides `mutate` method for mutating an existing
  level with configurable 
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
    * apples_pos : index[a, 2]
            Coordinates of the apples (index into `wall_map`)
    * shield_pos : index[s, 2]
            List of coordinates of shields (index into `wall_map`)
    * initial_monsters_pos : index[m, 2]
            List of initial coordinates of monsters (index into `wall_map`)
    * initial_mouse_pos : index[2]
            Coordinates of initial mouse position (index into `wall_map`)
    * dist_map : int[h,w,h,w,5]
            Cached solution to the maze. The first two axes specify the
            square the monster is currently on. The second two axes specify
            the target location (mouse location). The values along the final
            axis encode the distance to the target if the source were to
            (0) move up, (1) move left, (2) move down, (3) move right, or
            (4) stay put. The monsters decide in which direction to move
            using this vector according to a softmax distribution.
    * monster_optimality : float (positive)
            The inverse temperature for the monster softmax distribution.
    * inventory_map : index[s]
            Coordinates of inventory (index into width)
    """
    wall_map: chex.Array
    apples_pos: chex.Array
    shields_pos: chex.Array
    initial_monsters_pos: chex.Array
    initial_mouse_pos: chex.Array
    dist_map: chex.Array
    monster_optimality: float
    inventory_map: chex.Array


@struct.dataclass
class EnvState(base.EnvState):
    """
    Dynamic environment state within a particular level.

    * mouse_pos : index[2]
            Current coordinates of the mouse. Initialised to
            `level.initial_mouse_pos`.
    * monsters_pos : index[m, 2]
            Current coordinates of the monsters. Initialised to
            `level.initial_monsters_pos`.
    * got_monsters : bool[s]
            Mask tracking which monsters have already been defeated (True).
            Initially all False.
    * got_shields : bool[s]
            Mask tracking which shields have already been collected (True).
            Initially all False.
    * used_shields : bool[s]
            Mask tracking which shields have already been collected and then
            spent to defend against a monster (True). Initially all False.
    * got_apples : bool[a]
            Mask tracking which apples have already been collected (True).
            Initially all False.
    """
    mouse_pos: chex.Array
    monsters_pos: chex.Array
    got_monsters: chex.Array
    got_shields: chex.Array
    used_shields: chex.Array
    got_apples: chex.Array


class Env(base.Env):
    """
    Monster gridworld environment.

    In this environment the agent controls a mouse navigating a grid world.
    The mouse must pick up shields located throught the world to defend
    against monsters, and collect apples.

    There are four available actions which deterministically move the mouse
    one grid square up, right, down, or left respectively.
    * If the mouse would hit a wall it remains in place.
    * If the mouse hits a shield, the shield is removed from the grid and
      stored in the mouse's inventory.
    * If the mouse hits a monster, if there is at least one shield in its
      inventory, the shield is spent and the monster is destroyed. Otherwise,
      the agent gets -1 reward.
    * If the mouse hits an apple, it gets +1 reward and the apple is removed
      from the grid.

    The episode ends when the mouse has collected all apples and defeated the
    maximum possible number of monsters (all monsters, or one monster for
    each shield if there are fewer shields than monsters).

    Observations come in one of two formats:

    * Boolean: a H by W by C bool array where each channel represents the
      presence of one type of thing (wall, mouse, apple, shield in world,
      monster, shield in inventory).
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


    class Channel(enum.IntEnum):
        """
        The observations returned by the environment are an `h` by `w` by
        `channel` Boolean array, where the final dimensions 0 through 4 indicate
        the following:

        * WALL:     True in the locations where there is a wall.
        * MOUSE:    True in the one location the mouse occupies.
        * APPLE:   True in locations occupied by an apple.
        * SHIELD:   True in locations occupied by a shield.
        * MONSTER:  True in locations occupied by a monster.
        * INV:      True in a number of locations along the top row
                    corresponding to the number of previously-collected but
                    as-yet-unused shields.
        """
        WALL = 0
        MOUSE = 1
        APPLE = 2
        MONSTER = 3
        SHIELD = 4
        INV = 5


    @property
    def num_actions(self) -> int:
        return len(Env.Action)

    
    @functools.partial(jax.jit, static_argnames=('self',))
    def _reset(
        self,
        rng: chex.PRNGKey,
        level: Level,
    ) -> EnvState:
        """
        See reset_to_level method of UnderspecifiedEnv.
        """
        num_shields, _2 = level.shields_pos.shape
        num_monsters, _2 = level.initial_monsters_pos.shape
        num_apples, _2 = level.apples_pos.shape
        state = EnvState(
            mouse_pos=level.initial_mouse_pos,
            monsters_pos=level.initial_monsters_pos,
            got_monsters=jnp.zeros(num_monsters, dtype=bool),
            got_shields=jnp.zeros(num_shields, dtype=bool),
            used_shields=jnp.zeros(num_shields, dtype=bool),
            got_apples=jnp.zeros(num_apples, dtype=bool),
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

        # move the monsters
        # choose monster directions
        monster_action_distances = state.level.dist_map[
            state.monsters_pos[:, 0],
            state.monsters_pos[:, 1],
            state.mouse_pos[0],
            state.mouse_pos[1],
        ]
        monster_action_logits = (
            - state.level.monster_optimality * monster_action_distances
        )
        rng_monster, rng = jax.random.split(rng)
        monster_actions = jax.random.categorical(
            key=rng_monster,
            logits=monster_action_logits,
            axis=-1,
        )
        # candidate new monsters positions
        monster_steps = jnp.array((
            (-1,  0),   # up
            ( 0, -1),   # left
            (+1,  0),   # down
            ( 0, +1),   # right
            ( 0,  0),   # stay
        ))[monster_actions]
        monsters_ahead_pos = state.monsters_pos + monster_steps
        # wall collision detection can happen now since walls don't move
        monsters_hit_wall = state.level.wall_map[
            monsters_ahead_pos[:,0],
            monsters_ahead_pos[:,1],
        ]
        # but move the monsters one at a time so they don't collide, and
        # in a random order so that there are no predictable patterns
        initial_monsters_pos = state.monsters_pos
        rng_priority, rng = jax.random.split(rng)
        random_monster_order = jax.random.permutation(
            key=rng_priority,
            x=state.monsters_pos.shape[0],
        )
        def _move_one_monster(monsters_pos, monster_idx):
            monster_ahead_pos = monsters_ahead_pos[monster_idx]
            monster_hit_wall = monsters_hit_wall[monster_idx]
            # check collision for any monsters with current monster positions
            monster_hit_monsters = jnp.any(
                jnp.all(
                    monsters_pos == monster_ahead_pos,
                    axis=1,
                )
                & ~state.got_monsters
            )
            monster_hit_another_monster = monster_hit_monsters.any()
            # move this monster
            monsters_pos = monsters_pos.at[monster_idx].set(
                jnp.where(
                    monster_hit_wall | monster_hit_another_monster,
                    monsters_pos[monster_idx],
                    monster_ahead_pos,
                )
            )
            return monsters_pos, None
        final_monsters_pos, _ = jax.lax.scan(
            _move_one_monster,
            initial_monsters_pos,
            random_monster_order,
        )
        # finally, save the updated monster positions into the state
        state = state.replace(
            monsters_pos=final_monsters_pos
        )

        # pick up shields
        pickup_shields = (
            # select shields in the same location as the mouse:
            (state.mouse_pos == state.level.shields_pos).all(axis=1)
            # filter for shields the mouse hasn't yet picked up:
            & (~state.got_shields)
        )
        state = state.replace(got_shields=state.got_shields ^ pickup_shields)

        # interact with monsters
        colliding_monsters = (
            # select monsters in the same location as the mouse:
            (state.mouse_pos == state.monsters_pos).all(axis=1)
            # filter for monsters the mouse hasn't yet defeated
            & (~state.got_monsters)
        )
        available_shields = (state.got_shields & ~state.used_shields)
        # if the mouse has a shield, defeat monster and consume shield
        got_monsters = colliding_monsters & available_shields.any()
        the_used_shield = jnp.argmax(available_shields) # first true if any
        state = state.replace(
            got_monsters=state.got_monsters | got_monsters,
            used_shields=state.used_shields.at[the_used_shield].set(
                state.used_shields[the_used_shield] | got_monsters.any()
            ),
            monsters_pos=jnp.where(
                got_monsters[:, jnp.newaxis],
                jnp.array([0,0]),
                state.monsters_pos,
            ),
        )
        # if the mouse has no shield, take a hit
        got_hit_by_monsters = colliding_monsters & ~available_shields.any()
        
        # interact with apples
        got_apples = (
            # select apples in the same location as the mouse
            (state.mouse_pos == state.level.apples_pos).all(axis=1)
            # filter for apples the mouse hasn't yet collected
            & (~state.got_apples)
        )
        state = state.replace(
            got_apples=state.got_apples | got_apples,
        )

        # reward for getting apples minus getting hit
        reward = (got_apples.sum() - got_hit_by_monsters.sum()).astype(float)

        # done if we got all apples and killed max monsters possible
        max_kills = min(state.got_monsters.size, state.got_shields.size)
        done = state.got_apples.all() & (state.got_monsters.sum() == max_kills)
        
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

        # render apples if not yet gotten
        obs = obs.at[
            state.level.apples_pos[:, 0],
            state.level.apples_pos[:, 1],
            Env.Channel.APPLE,
        ].set(~state.got_apples)

        # render monsters that haven't been killed
        obs = obs.at[
            state.monsters_pos[:, 0],
            state.monsters_pos[:, 1],
            Env.Channel.MONSTER,
        ].set(~state.got_monsters)
        
        # render shields that haven't been picked up
        obs = obs.at[
            state.level.shields_pos[:, 0],
            state.level.shields_pos[:, 1],
            Env.Channel.SHIELD,
        ].set(~state.got_shields)
        
        # render shields that have been picked up but haven't been used
        obs = obs.at[
            0,
            state.level.inventory_map,
            Env.Channel.INV,
        ].set(state.got_shields & ~state.used_shields)

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
            # multiple objects
            obs[:, :, Env.Channel.WALL] & obs[:, :, Env.Channel.INV],
            obs[:, :, Env.Channel.MONSTER] & obs[:, :, Env.Channel.APPLE],
            obs[:, :, Env.Channel.MONSTER] & obs[:, :, Env.Channel.SHIELD],
            obs[:, :, Env.Channel.MONSTER] & obs[:, :, Env.Channel.MOUSE],
            # one object
            obs[:, :, Env.Channel.WALL],
            obs[:, :, Env.Channel.MOUSE],
            obs[:, :, Env.Channel.APPLE],
            obs[:, :, Env.Channel.MONSTER],
            obs[:, :, Env.Channel.SHIELD],
            # no objects, 'default' (always true)
            jnp.ones((H, W), dtype=bool),
        ])
        chosen_sprites = jnp.argmax(sprite_priority_vector_grid, axis=0)

        # put the corresponding sprite into each square
        spritemap = jnp.stack([
            # multiple objects
            spritesheet.SHIELD_ON_WALL,
            spritesheet.MONSTER_ON_APPLE,
            spritesheet.MONSTER_ON_SHIELD,
            spritesheet.MOUSE_ON_MONSTER,
            # one object
            spritesheet.WALL,
            spritesheet.MOUSE,
            spritesheet.APPLE,
            spritesheet.MONSTER,
            spritesheet.SHIELD,
            # no objects
            spritesheet.PATH,
        ])[chosen_sprites]
        image = einops.rearrange(
            spritemap,
            'h w th tw rgb -> (h th) (w tw) rgb',
        )

        return image

        
@struct.dataclass
class LevelGenerator(base.LevelGenerator):
    """
    Level generator for Keys and Chests environment. Given some maze
    configuration parameters and key/chest sparsity parameters, provides a
    `sample` method that generates a random level.

    * height : int (>= 3, odd)
            the number of rows in the grid representing the maze
            (including top and bottom boundary rows)
    * width : int (>= 3, odd)
            the number of columns in the grid representing the maze
            (including left and right boundary rows)
    * layout : str ('open', 'tree', 'bernoulli', 'blocks', or 'noise')
            specifies the maze generation method to use (see module
            `maze_generation` for details)
    * num_shields : int
            the number of shields to randomly place in each generated maze
    * num_monsters : int
            the number of monsters to randomly place in each generated maze
    * num_apples : int
            the number of apples to randomly place in each generated maze
    * monster_optimality : float (positive, default 10)
            inverse temperature for the monster to step towards the player
    """
    height: int                 = 13
    width: int                  = 13
    layout: str                 = 'open'
    num_shields: int            = 5
    num_monsters: int           = 5
    num_apples: int             = 5
    monster_optimality: float   = 3
    
    def __post_init__(self):
        # validate layout
        assert self.layout in {'tree', 'edges', 'blocks', 'open', 'noise'}
        # validate dimensions
        assert self.height >= 3
        assert self.width >= 3
        if self.layout == 'tree' or self.layout == 'edges':
            assert self.height % 2 == 1, "height must be odd for this layout"
            assert self.width % 2 == 1,  "width must be odd for this layout"
        # validate shields
        assert self.num_shields > 0
        assert self.num_shields <= self.width, "not enough space for inventory"
        # validate monsters
        assert self.num_monsters > 0
        assert self.monster_optimality >= 0, "negative inverse temperature"
        # validate apples
        assert self.num_apples > 0
    

    @functools.partial(jax.jit, static_argnames=('self',))
    def sample(self, rng: chex.PRNGKey) -> Level:
        """
        Randomly generate a `Level` specification given the parameters
        provided in the constructor of this generator object.
        """
        # construct the wall map
        rng_walls, rng = jax.random.split(rng)
        wall_map = maze_generation.get_generator_function(self.layout)(
            key=rng_walls,
            h=self.height,
            w=self.width,
        )
        
        # spawn random mouse, apple, shield and monster positions
        rng_spawn, rng = jax.random.split(rng)
        coords = einops.rearrange(
            jnp.indices((self.height, self.width)),
            'c h w -> (h w) c',
        )
        num_all = 1 + self.num_apples + self.num_shields + self.num_monsters
        all_pos = jax.random.choice(
            key=rng_spawn,
            a=coords,
            axis=0,
            shape=(num_all,),
            p=~wall_map.flatten(),
            replace=False,
        )
        initial_mouse_pos = all_pos[0]
        apples_pos = all_pos[1:1+self.num_apples]
        shields_pos = all_pos[1+self.num_apples:num_all-self.num_monsters]
        initial_monsters_pos = all_pos[num_all-self.num_monsters:num_all]

        # solve the map and cache the solution for the monsters
        dist_map = maze_generation.maze_directional_distances(wall_map)

        # decide random positions for shield display
        rng_inventory, rng = jax.random.split(rng)
        inventory_map = jax.random.choice(
            key=rng_inventory,
            a=self.width,
            shape=(self.num_shields,),
            replace=False,
        )
    
        return Level(
            wall_map=wall_map,
            initial_mouse_pos=initial_mouse_pos,
            apples_pos=apples_pos,
            shields_pos=shields_pos,
            initial_monsters_pos=initial_monsters_pos,
            dist_map=dist_map,
            monster_optimality=self.monster_optimality,
            inventory_map=inventory_map,
        )


@struct.dataclass
class LevelMutator(base.LevelMutator):
    """
    Configurable level mutator. Provides a 'mutate' method that transforms a
    level into a slightly different level, with the configured mutation
    operations.

    Parameters:

    * prob_wall_spawn : float (probability)
            Probability that a given interior (non-border) wall will despawn
            during a mutation.
    * prob_wall_despawn : float (probability)
            Probability that a given blank space (with no wall, shield,
            monster, apple, or mouse) will spawn a wall during a mutation.
    * prob_scatter : float (probability)
            Probability that a given entity (shield/monster/mouse/apple
            spawn) will have its position randomised during a mutation.
    * max_steps : int (>= 0)
            For each entity, sample this many directions and apply them to
            the position (unless that would result in a collision with
            another entity or with a wall).
    * monster_optimality_step : float
            The monster optimality is an inverse temperature parameter. Each
            mutation may randomly leave the optimality the same, increase it
            by this amount, or decrease it by this amount (to a minimum of
            zero), each with equal probability.
    
    Notes:

    * The wall mutations are applied first, then the scatter mutation, then
      the step mutations are applied (stepwise, within each step the order is
      mouse spawn, apples, shields, then monsters).
    * The step mutations are implemented using a fully unrolled loop, which
      should be slightly faster at the cost of slightly larger functions to
      compile. This might become an issue if very large number of steps are
      requested.
    """
    prob_wall_spawn: float
    prob_wall_despawn: float
    prob_scatter: float
    max_steps: int
    monster_optimality_step: float


    @functools.partial(jax.jit, static_argnames=('self',))
    def mutate(self, rng: chex.PRNGKey, level: Level):
        # toggle walls
        if self.prob_wall_spawn > 0 or self.prob_wall_despawn > 0:
            rng_walls, rng = jax.random.split(rng)
            level = self._toggle_walls(rng_walls, level)

        # moving entity spawn locations
        if self.prob_scatter:
            rng_scatter, rng = jax.random.split(rng)
            level = self._scatter(rng_scatter, level)
        
        # moving entities locally
        if self.max_steps > 0:
            rng_steps, rng = jax.random.split(rng)
            def _step(level, rng_step):
                return self._step_entities(rng_step, level), None
            level, _ = jax.lax.scan(
                _step,
                level,
                jax.random.split(rng_steps, self.max_steps),
                unroll=True,
            )

        # tweak monster optimality
        if self.monster_optimality_step > 0:
            rng_optimality, rng = jax.random.split(rng)
            level = self._tweak_optimality(rng_optimality, level)

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
        
        # don't toggle the place where there are entities
        entities_pos = jnp.concatenate((
            level.initial_mouse_pos[jnp.newaxis],
            level.apples_pos,
            level.shields_pos,
            level.initial_monsters_pos,
        ))
        walls_to_toggle = walls_to_toggle.at[
            entities_pos[:,0],
            entities_pos[:,1],
        ].set(False)

        # toggle!
        new_wall_map = jnp.logical_xor(level.wall_map, walls_to_toggle)

        return level.replace(wall_map=new_wall_map)


    def _scatter(self, rng, level):
        num_apples = level.apples_pos.shape[0]
        num_shields = level.shields_pos.shape[0]
        num_monsters = level.initial_monsters_pos.shape[0]
        num_entities = 1 + num_apples + num_shields + num_monsters
        all_pos = jnp.concatenate((
            level.initial_mouse_pos[jnp.newaxis],
            level.apples_pos,
            level.shields_pos,
            level.initial_monsters_pos,
        ))
        
        # randomly decide a subset of entities to move
        rng_subset, rng = jax.random.split(rng)
        entities_to_move = jax.random.bernoulli(
            key=rng_subset,
            p=self.prob_scatter,
            shape=(num_entities,),
        )

        # move them one at a time, avoiding collisions
        initial_available_map = (~level.wall_map).at[
            all_pos[:, 0],
            all_pos[:, 1],
        ].set(False)
        coords = einops.rearrange(
            jnp.indices(level.wall_map.shape),
            'c h w -> (h w) c',
        )

        def _move(available_map, rng_pos_and_move):
            rng, pos, move = rng_pos_and_move
            available_map = available_map.at[
                pos[0],
                pos[1],
            ].set(True)
            try_pos = jax.random.choice(
                key=rng,
                a=coords,
                axis=0,
                p=available_map.flatten(),
            )
            new_pos = jnp.where(
                move,
                try_pos,
                pos,
            )
            available_map = available_map.at[
                new_pos[0],
                new_pos[1],
            ].set(False)
            return available_map, new_pos

        _, mut_pos = jax.lax.scan(
            _move,
            initial_available_map,
            (
                jax.random.split(rng, num_entities),
                all_pos,
                entities_to_move,
            ),
        )

        # extract from unified position array
        mut_initial_mouse_pos = mut_pos[0]
        mut_apples_pos = mut_pos[1:1+num_apples]
        mut_shields_pos = mut_pos[1+num_apples:num_entities-num_monsters]
        mut_initial_monsters_pos = mut_pos[num_entities-num_monsters:]
    
        return level.replace(
            initial_mouse_pos=mut_initial_mouse_pos,
            apples_pos=mut_apples_pos,
            shields_pos=mut_shields_pos,
            initial_monsters_pos=mut_initial_monsters_pos,
        )


    def _step_entities(self, rng, level):
        num_apples = level.apples_pos.shape[0]
        num_shields = level.shields_pos.shape[0]
        num_monsters = level.initial_monsters_pos.shape[0]
        num_entities = 1 + num_apples + num_shields + num_monsters
        
        # unified array of all current positions
        all_pos = jnp.concatenate((
            level.initial_mouse_pos[jnp.newaxis],
            level.apples_pos,
            level.shields_pos,
            level.initial_monsters_pos,
        ))

        # randomly sampled candidate position for each entity
        steps = jnp.array((
            (-1,  0),   # up
            ( 0, -1),   # left
            (+1,  0),   # down
            ( 0, +1),   # right
        ))
        directions = jax.random.choice(
            key=rng,
            a=4,
            shape=(num_entities,),
        )
        try_pos = all_pos + steps[directions]

        # execute the steps, one by one, avoiding collisions
        initial_collision_map = level.wall_map.at[
            all_pos[:,0],
            all_pos[:,1],
        ].set(True)

        def _step(collision_map, step):
            old_pos, try_pos = step
            hit_wall_or_another_entity = collision_map[
                try_pos[0],
                try_pos[1],
            ]
            new_pos = jnp.where(
                hit_wall_or_another_entity,
                old_pos,
                try_pos,
            )
            collision_map = collision_map.at[
                (old_pos[0], new_pos[0]),
                (old_pos[1], new_pos[1]),
            ].set(jnp.array((False, True))) # if same pos, 'True' will stick
            return collision_map, new_pos
        
        _, mut_pos = jax.lax.scan(
            _step,
            initial_collision_map,
            (all_pos, try_pos),
            unroll=True,
        )

        # extract positions for different entities from unified results array
        mut_initial_mouse_pos = mut_pos[0]
        mut_apples_pos = mut_pos[1:1+num_apples]
        mut_shields_pos = mut_pos[1+num_apples:num_entities-num_monsters]
        mut_initial_monsters_pos = mut_pos[num_entities-num_monsters:]
    
        return level.replace(
            initial_mouse_pos=mut_initial_mouse_pos,
            apples_pos=mut_apples_pos,
            shields_pos=mut_shields_pos,
            initial_monsters_pos=mut_initial_monsters_pos,
        )
    

    def _tweak_optimality(self, rng, level):
        step = jax.random.choice(
            key=rng,
            a=jnp.array((
                -self.monster_optimality_step,
                0,
                self.monster_optimality_step,
            )),
        )
        new_monster_optimality = jnp.maximum(
            0,
            level.monster_optimality + step
        )
        return level.replace(
            monster_optimality=new_monster_optimality,
        )
