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
    #dish_positions: chex.Array
    initial_mouse_pos: chex.Array
    fork_pos: chex.Array
    spoon_pos: chex.Array
    glass_pos: chex.Array
    mug_pos: chex.Array
    napkin_pos: chex.Array


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
    got_fork: bool
    got_spoon: bool
    got_glass: bool
    got_mug: bool
    got_napkin: bool

    #got_items_groupone: bool
    #got_items_grouptwo: bool
    #collected_dishes: chex.Array


@struct.dataclass
class Observation(base.Observation):
    """
    Observation for partially observable Maze environment.

    * image : bool[h, w, c] or float[h, w, rgb]
            The contents of the state. Comes in one of two formats:
            * Boolean: a H by W by C bool array where each channel represents
              the presence of one type of thing (walls, mouse, cheese, dish,
              etc.).
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
    * `DISH`:   True in the one location the dish occupies.
    """
    WALL    = 0
    MOUSE   = 1
    CHEESE  = 2
    DISH    = 3
    FORK    = 4
    SPOON   = 5
    GLASS   = 6
    MUG     = 7
    NAPKIN  = 8

    
@struct.dataclass
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
    """
    terminate_after_cheese_and_pile: bool = False
    split_object_firstgroup: int = 6

    @property
    def num_actions(self) -> int:
        return len(Action)


    def obs_type( self, level: Level) -> PyTree[jax.ShapeDtypeStruct]:
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
            got_dish=False,
            got_fork=False,
            got_spoon=False,
            got_glass=False,
            got_mug=False,
            got_napkin=False,
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
            #got_dish=False,
            #got_fork=False,
            # got_spoon=False,
            # got_glass=False,
            # got_mug=False,
            # got_napkin=False,
            # got_table=False,
            # got_oven=False,

        got_cheese = (state.mouse_pos == state.level.cheese_pos).all()
        got_cheese_first_time = got_cheese & ~state.got_cheese
        state = state.replace(got_cheese=state.got_cheese | got_cheese)
        
        got_dish = (state.mouse_pos == state.level.dish_pos).all()
        got_fork = (state.mouse_pos == state.level.fork_pos).all()
        got_spoon = (state.mouse_pos == state.level.spoon_pos).all()
        got_glass = (state.mouse_pos == state.level.glass_pos).all()
        got_mug = (state.mouse_pos == state.level.mug_pos).all()
        got_napkin = (state.mouse_pos == state.level.napkin_pos).all()

        got_dish_first_time = got_dish & ~state.got_dish
        state = state.replace(got_dish=state.got_dish | got_dish)

        got_fork_first_time = got_fork & ~state.got_fork
        state = state.replace(got_fork=state.got_fork | got_fork)

        got_spoon_first_time = got_spoon & ~state.got_spoon
        state = state.replace(got_spoon=state.got_spoon | got_spoon)

        got_glass_first_time = got_glass & ~state.got_glass
        state = state.replace(got_glass=state.got_glass | got_glass)

        got_mug_first_time = got_mug & ~state.got_mug
        state = state.replace(got_mug=state.got_mug | got_mug)

        got_napkin_first_time = got_napkin & ~state.got_napkin
        state = state.replace(got_napkin=state.got_napkin | got_napkin)
        
        got_objects = [got_dish, got_fork, got_spoon, got_glass, got_mug, got_napkin]

        first_time_objects = [got_dish_first_time, got_fork_first_time, got_spoon_first_time, got_glass_first_time, got_mug_first_time, got_napkin_first_time]

        state_objects = [state.got_dish, state.got_fork, state.got_spoon, state.got_glass, state.got_mug, state.got_napkin ]

        objects_positions = [state.level.dish_pos, state.level.fork_pos, state.level.spoon_pos, state.level.glass_pos, state.level.mug_pos, state.level.napkin_pos]


        # reward and done
        reward = got_cheese_first_time.astype(float)
        first_group_got_objects = got_objects[:self.split_object_firstgroup]
        first_group_first_time_objects= first_time_objects[:self.split_object_firstgroup]
        first_group_state_objects= state_objects[:self.split_object_firstgroup]
        first_group_objects_positions= objects_positions[:self.split_object_firstgroup]
        if self.split_object_firstgroup > len(got_objects) - 1:
            second_group_got_objects = []
            second_group_first_time_objects = []
            second_group_objects_positions = []
            second_group_state_objects = []
        else:
            second_group_got_objects= got_objects[self.split_object_firstgroup:]
            second_group_first_time_objects= first_time_objects[self.split_object_firstgroup:]
            second_group_state_objects= state_objects[self.split_object_firstgroup:]
            second_group_objects_positions= objects_positions[self.split_object_firstgroup:]
        

        #proxy_first_group= first_group_first_time_objects[0].asytype(float)
        if len(second_group_first_time_objects) > 0:
            proxy_pile = got_napkin_first_time.astype(float)
            got_pile_before_cheese = got_napkin_first_time & ~state.got_cheese
            got_cheese_before_pile = state.got_cheese & ~state.got_napkin
        else:
            proxy_pile = got_cheese_first_time.astype(float)
            got_pile_before_cheese = 0
            got_cheese_before_pile = 0


        #proxy_reward_dish = got_dish_first_time.astype(float)

        #got_dish_after_cheese = state.got_dish & state.got_cheese
        #got_cheese_after_dish = state.got_cheese & state.got_dish

        #proxy_cheese_second = reward * got_dish_after_cheese
        #proxy_dish_second = proxy_reward_dish * got_cheese_after_dish
        
        proxy_cheese_first = reward * got_cheese_before_pile
        proxy_pile_first = proxy_pile * got_pile_before_cheese
        
        if self.terminate_after_cheese_and_pile:
            if len(second_group_first_time_objects) > 0:
                done = state.got_cheese & state.got_napkin
            else:
                done = state.got_cheese
        else:
            done = state.got_cheese | state.got_napkin

        cheese_rate = jax.lax.cond(jnp.any(state.got_cheese), lambda _: 1.0, lambda _: 0.0, operand=None)
        pile_rate = jax.lax.cond(jnp.any(state.got_napkin), lambda _: 1.0, lambda _: 0.0, operand=None)

        return (
            state,
            reward,
            done,
            {
                'proxy_rewards': {
                    'proxy_pile': proxy_pile,
                    'proxy_pile_first': proxy_pile_first,
                    'proxy_cheese_first': proxy_cheese_first,
                    'cheese_solve_rate': cheese_rate, 
                    'pile_solve_rate': pile_rate,
                     #'proxy_cheese_second': proxy_cheese_second,
                     #'proxy_dish_second': proxy_dish_second,
                },
            },
        )

    
    @functools.partial(jax.jit, static_argnames=('self',))
    def _render_obs_bool(self, state: EnvState) -> chex.Array:
        """
        Return a boolean grid observation.
        """
        H, W = state.level.wall_map.shape
        C = len(Channel)
        obs = jnp.zeros((H, W, C), dtype=bool)

        # render walls
        obs = obs.at[:, :, Channel.WALL].set(state.level.wall_map)

        # render mouse
        obs = obs.at[
            state.mouse_pos[0],
            state.mouse_pos[1],
            Channel.MOUSE,
        ].set(True)
        
        # render cheese
        obs = obs.at[
            state.level.cheese_pos[0],
            state.level.cheese_pos[1],
            Channel.CHEESE,
        ].set(~state.got_cheese)
        
        # render dish
        obs = obs.at[
            state.level.dish_pos[0],
            state.level.dish_pos[1],
            Channel.DISH,
        ].set(~state.got_dish)

        #render other objects
        obs = obs.at[
            state.level.fork_pos[0],
            state.level.fork_pos[1],
            Channel.FORK,
        ].set(~state.got_fork)

        obs = obs.at[
            state.level.spoon_pos[0],
            state.level.spoon_pos[1],
            Channel.SPOON,
        ].set(~state.got_spoon)

        obs = obs.at[
            state.level.glass_pos[0],
            state.level.glass_pos[1],
            Channel.GLASS,
        ].set(~state.got_glass)

        obs = obs.at[
            state.level.mug_pos[0],
            state.level.mug_pos[1],
            Channel.MUG,
        ].set(~state.got_mug)

        obs = obs.at[
            state.level.napkin_pos[0],
            state.level.napkin_pos[1],
            Channel.NAPKIN,
        ].set(~state.got_napkin)
        
        return Observation(image=obs)


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
        obs = self._render_obs_bool(state).image
        H, W, _C = obs.shape

        # find out, for each position, which object to render
        # (for each position pick the first true index top-down this list)
        sprite_priority_vector_grid = jnp.stack([
            #seven objects
            obs[:, :, Channel.CHEESE]
                & obs[:, :, Channel.DISH]
                & obs[:, :, Channel.FORK]
                & obs[:, :, Channel.SPOON]
                & obs[:, :, Channel.GLASS]
                & obs[:, :, Channel.MUG]
                & obs[:, :, Channel.NAPKIN],
            #six objects
            obs[:, :, Channel.CHEESE]
                & obs[:, :, Channel.DISH]
                & obs[:, :, Channel.FORK]
                & obs[:, :, Channel.SPOON]
                & obs[:, :, Channel.GLASS]
                & obs[:, :, Channel.MUG],
            obs[:, :, Channel.DISH]
                & obs[:, :, Channel.FORK]
                & obs[:, :, Channel.SPOON]
                & obs[:, :, Channel.GLASS]
                & obs[:, :, Channel.MUG]
                & obs[:, :, Channel.NAPKIN],
            #five objects
            obs[:, :, Channel.CHEESE]
                & obs[:, :, Channel.DISH]
                & obs[:, :, Channel.FORK]
                & obs[:, :, Channel.SPOON]
                & obs[:, :, Channel.GLASS],
            obs[:, :, Channel.NAPKIN]
                & obs[:, :, Channel.MUG]
                & obs[:, :, Channel.GLASS]
                & obs[:, :, Channel.SPOON]
                & obs[:, :, Channel.FORK],
            #four objects
            obs[:, :, Channel.CHEESE]
                & obs[:, :, Channel.DISH]
                & obs[:, :, Channel.FORK]
                & obs[:, :, Channel.SPOON],
            obs[:, :, Channel.NAPKIN]
                & obs[:, :, Channel.MUG]
                & obs[:, :, Channel.GLASS]
                & obs[:, :, Channel.SPOON],
            #three objects
            obs[:, :, Channel.CHEESE]
                & obs[:, :, Channel.DISH]
                & obs[:, :, Channel.FORK],
            obs[:, :, Channel.NAPKIN]
                & obs[:, :, Channel.MUG]
                & obs[:, :, Channel.GLASS],
            # two objects
            obs[:, :, Channel.CHEESE] & obs[:, :, Channel.DISH],
            obs[:, :, Channel.NAPKIN] & obs[:, :, Channel.MUG],
            # one object
            obs[:, :, Channel.WALL],
            obs[:, :, Channel.MOUSE],
            obs[:, :, Channel.CHEESE],
            obs[:, :, Channel.DISH],
            obs[:, :, Channel.FORK],
            obs[:, :, Channel.SPOON],
            obs[:, :, Channel.GLASS],
            obs[:, :, Channel.MUG],
            obs[:, :, Channel.NAPKIN],
            # no objects, 'default' (always true)
            jnp.ones((H, W), dtype=bool),
        ])
        chosen_sprites = jnp.argmax(sprite_priority_vector_grid, axis=0)

        # put the corresponding sprite into each square
        spritemap = jnp.stack([
            #seven objects
            spritesheet['CHEESE'],
            #six objects
            spritesheet['CHEESE'],
            spritesheet['BEACON_OFF'],
            #five objects
            spritesheet['CHEESE'],
            spritesheet['BEACON_OFF'],
            #four objects
            spritesheet['CHEESE'],
            spritesheet['BEACON_OFF'],
            #three objects
            spritesheet['CHEESE'],
            spritesheet['BEACON_OFF'],
            # two objects
            spritesheet['CHEESE'],
            spritesheet['BEACON_OFF'],
            # one object
            spritesheet['WALL'],
            spritesheet['MOUSE'],
            spritesheet['CHEESE'],
            spritesheet['BEACON_OFF'],
            spritesheet['BEACON_OFF'],
            spritesheet['BEACON_OFF'],
            spritesheet['BEACON_OFF'],
            spritesheet['BEACON_OFF'],
            spritesheet['BEACON_OFF'],
            # no objects
            spritesheet['PATH'],
        ])[chosen_sprites]

        image = einops.rearrange(
            spritemap,
            'h w th tw rgb -> (h th) (w tw) rgb',
        )
        return Observation(image=image)


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
    max_dish_radius: int = 0
    split_elements: int = 0 # how many items should be placed with the cheese, max is six
    
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
        
        # pile spawns in some remaining valid position
        no_mouse = jnp.ones_like(wall_map).at[
            initial_mouse_pos[0],
            initial_mouse_pos[1],
        ].set(False).flatten()

        final_spawn = []

        #first group position
        rng_spawn_first, rng = jax.random.split(rng)
        napkin_pos = jax.random.choice(
            key=rng_spawn_first,
            a=coords,
            axis=0,
            p=no_wall & no_mouse,
        )
        for i in range(6-self.split_elements):
            pos = napkin_pos
            final_spawn.append(pos)


        distance_to_napkin = maze_solving.maze_distances(wall_map)[
            napkin_pos[0],
            napkin_pos[1],
        ]

        near_napkin = (distance_to_napkin <= self.max_cheese_radius).flatten()


        rng_spawn_cheese, rng = jax.random.split(rng)
        cheese_pos = jax.random.choice(
            key=rng_spawn_cheese,
            a=coords,
            axis=0,
            p=no_wall & no_mouse & near_napkin,
            
        )


        for i in range(self.split_elements):
            pos = cheese_pos
            final_spawn.append(pos)

        final_spawn.reverse()


        return Level(
            wall_map=wall_map,
            initial_mouse_pos=initial_mouse_pos,
            cheese_pos=cheese_pos,
            dish_pos=final_spawn[0],
            fork_pos = final_spawn[1],
            spoon_pos = final_spawn[2],
            glass_pos = final_spawn[3],
            mug_pos = final_spawn[4],
            napkin_pos = final_spawn[5],
        )

#Level Mutation 

@struct.dataclass
class ToggleWallLevelMutator(base.LevelMutator):

    @functools.partial(jax.jit, static_argnames=('self',))
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape

        valid_map = jnp.ones((h, w), dtype=bool)
        valid_map = valid_map.at[(0, h-1), :].set(False)
        valid_map = valid_map.at[:, (0, w-1)].set(False)

        # exclude the cheese, mouse and pile positions
        valid_map = valid_map.at[
            (level.cheese_pos[0], level.napkin_pos[0], level.initial_mouse_pos[0]),
            (level.cheese_pos[1], level.napkin_pos[1], level.initial_mouse_pos[1]),
        ].set(False)
        valid_mask= valid_map.flatten()

        coords = einops.rearrange(jnp.indices((h, w)), 'c h w -> (h w) c')
        toggle_pos = jax.random.choice(
            key=rng,
            a=coords,
            axis=0,
            p=valid_mask,
        )

        hit_wall = level.wall_map[toggle_pos[0], toggle_pos[1]]
        new_wall_map = level.wall_map.at[
            toggle_pos[0],
            toggle_pos[1],
        ].set(~hit_wall)

        return level.replace(wall_map=new_wall_map)


@struct.dataclass
class StepMouseLevelMutator(base.LevelMutator):
    transpose_with_cheese_on_collision: bool = False
    transpose_with_pile_on_collision: bool = False
    split_elements: int = 0

    @functools.partial(jax.jit, static_argnames=('self',))
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape
        assert h > 3 and w > 3 # need enough space to move the mouse

        steps = jnp.array((
            (-1,  0),   # up
            ( 0, -1),   # left
            (+1,  0),   # down
            ( 0, +1),   # right
        ))
        valid_mask = jnp.array((
            level.initial_mouse_pos[0] > 1,  # up
            level.initial_mouse_pos[1] > 1,  # left
            level.initial_mouse_pos[0] < h-2,  # down
            level.initial_mouse_pos[1] < w-2,  # right
        ))
        chosen_step = jax.random.choice(
            key=rng,
            a=steps,
            p=valid_mask,
        )
        new_initial_mouse_pos = level.initial_mouse_pos + chosen_step

        new_wall_map = level.wall_map.at[
            new_initial_mouse_pos[0],
            new_initial_mouse_pos[1],
        ].set(False)

        hit_cheese = (new_initial_mouse_pos == level.cheese_pos).all()
        hit_pile = (new_initial_mouse_pos == level.napkin_pos).all()
        if self.transpose_with_cheese_on_collision:
            new_cheese_pos = jax.lax.select(
                hit_cheese,
                level.initial_mouse_pos,
                level.cheese_pos,
            )
        else:
            new_cheese_pos = level.cheese_pos
            new_initial_mouse_pos = jax.lax.select(
                hit_cheese,
                level.initial_mouse_pos,
                new_initial_mouse_pos,
            )
        if self.transpose_with_pile_on_collision:
            new_napkin_pos = jax.lax.select(
                hit_pile,
                level.initial_mouse_pos,
                level.napkin_pos,
            )
        else:
            new_napkin_pos = level.napkin_pos
            new_initial_mouse_pos = jax.lax.select(
                hit_pile,
                level.initial_mouse_pos,
                new_initial_mouse_pos,
            )

        final_spawn = jnp.array([
            level.dish_pos,
            level.fork_pos,
            level.spoon_pos,
            level.glass_pos,
            level.mug_pos,
            level.napkin_pos,
        ])

        # reassign the positions accordingly
        cheese_indices = jnp.arange(self.split_elements)
        napkin_indices = jnp.arange(6 - self.split_elements, 6)

        cheese_mask = jnp.zeros(6, dtype=bool).at[cheese_indices].set(True)
        napkin_mask = jnp.zeros(6, dtype=bool).at[napkin_indices].set(True)

         
        final_spawn = jnp.where(cheese_mask[:, None], new_cheese_pos, final_spawn)
        final_spawn = jnp.where(napkin_mask[:, None], new_napkin_pos, final_spawn)

        return level.replace(
            wall_map=new_wall_map,
            initial_mouse_pos=new_initial_mouse_pos,
            cheese_pos=new_cheese_pos,
            dish_pos=final_spawn[0],
            fork_pos = final_spawn[1],
            spoon_pos = final_spawn[2],
            glass_pos = final_spawn[3],
            mug_pos = final_spawn[4],
            napkin_pos = final_spawn[5],
        )
        



        

@struct.dataclass
class ScatterMouseLevelMutator(base.LevelMutator):
    transpose_with_cheese_on_collision: bool = False
    transpose_with_pile_on_collision: bool = False
    split_elements: int = 0

    @functools.partial(jax.jit, static_argnames=('self',))
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape
        assert h > 3 and w > 3

        rng_row, rng_col = jax.random.split(rng)
        new_mouse_row = jax.random.choice(
            key=rng_row,
            a=jnp.arange(1, h-1),
        )
        new_mouse_col = jax.random.choice(
            key=rng_col,
            a=jnp.arange(1, w-1),
        )
        new_initial_mouse_pos = jnp.array((new_mouse_row, new_mouse_col))

        new_wall_map = level.wall_map.at[
            new_initial_mouse_pos[0],
            new_initial_mouse_pos[1],
        ].set(False)

        hit_cheese = (new_initial_mouse_pos == level.cheese_pos).all()
        hit_pile = (new_initial_mouse_pos == level.napkin_pos).all()
        if self.transpose_with_cheese_on_collision:
            new_cheese_pos = jax.lax.select(
                hit_cheese,
                level.initial_mouse_pos,
                level.cheese_pos,
            )
        else:
            new_cheese_pos = level.cheese_pos
            new_initial_mouse_pos = jax.lax.select(
                hit_cheese,
                level.initial_mouse_pos,
                new_initial_mouse_pos,
            )
        if self.transpose_with_pile_on_collision:
            new_napkin_pos = jax.lax.select(
                hit_pile,
                level.initial_mouse_pos,
                level.napkin_pos,
            )
        else:
            new_napkin_pos = level.napkin_pos
            new_initial_mouse_pos = jax.lax.select(
                hit_pile,
                level.initial_mouse_pos,
                new_initial_mouse_pos,
            )
        
        final_spawn = jnp.array([
            level.dish_pos,
            level.fork_pos,
            level.spoon_pos,
            level.glass_pos,
            level.mug_pos,
            level.napkin_pos,
        ])

        # reassign the positions accordingly
        cheese_indices = jnp.arange(self.split_elements)
        napkin_indices = jnp.arange(6 - self.split_elements, 6)

        cheese_mask = jnp.zeros(6, dtype=bool).at[cheese_indices].set(True)
        napkin_mask = jnp.zeros(6, dtype=bool).at[napkin_indices].set(True)

         
        final_spawn = jnp.where(cheese_mask[:, None], new_cheese_pos, final_spawn)
        final_spawn = jnp.where(napkin_mask[:, None], new_napkin_pos, final_spawn)

        return level.replace(
            wall_map=new_wall_map,
            initial_mouse_pos=new_initial_mouse_pos,
            cheese_pos=new_cheese_pos,
            dish_pos=final_spawn[0],
            fork_pos = final_spawn[1],
            spoon_pos = final_spawn[2],
            glass_pos = final_spawn[3],
            mug_pos = final_spawn[4],
            napkin_pos = final_spawn[5],
        )


@struct.dataclass
class StepCheeseLevelMutator(base.LevelMutator):
    transpose_with_mouse_on_collision: bool = False
    transpose_with_pile_on_collision: bool = False
    split_elements: int = 0

    @functools.partial(jax.jit, static_argnames=('self',))
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape
        assert h > 3 and w > 3

        steps = jnp.array((
            (-1,  0),   # up
            ( 0, -1),   # left
            (+1,  0),   # down
            ( 0, +1),   # right
        ))

        valid_mask = jnp.array((
            level.cheese_pos[0] > 1,  # up
            level.cheese_pos[1] > 1,  # left
            level.cheese_pos[0] < h-2,  # down
            level.cheese_pos[1] < w-2,  # right
        ))

        chosen_step = jax.random.choice(
            key=rng,
            a=steps,
            p=valid_mask,
        )
        new_cheese_pos = level.cheese_pos + chosen_step

        new_wall_map = level.wall_map.at[
            new_cheese_pos[0],
            new_cheese_pos[1],
        ].set(False)

        hit_mouse = (new_cheese_pos == level.initial_mouse_pos).all()
        hit_pile = (new_cheese_pos == level.napkin_pos).all()
        if self.transpose_with_mouse_on_collision:
            new_initial_mouse_pos = jax.lax.select(
                hit_mouse,
                level.cheese_pos,
                level.initial_mouse_pos,
            )
        else:
            new_initial_mouse_pos = level.initial_mouse_pos
            new_cheese_pos = jax.lax.select(
                hit_mouse,
                level.cheese_pos,
                new_cheese_pos,
            )
        if self.transpose_with_pile_on_collision:
            new_napkin_pos = jax.lax.select(
                hit_pile,
                level.cheese_pos,
                level.napkin_pos,
            )
        else:
            new_napkin_pos = level.napkin_pos
            new_cheese_pos = jax.lax.select(
                hit_pile,
                level.cheese_pos,
                new_cheese_pos,
            )
        
        final_spawn = jnp.array([
            level.dish_pos,
            level.fork_pos,
            level.spoon_pos,
            level.glass_pos,
            level.mug_pos,
            level.napkin_pos,
        ])


        # reassign the positions accordingly
        cheese_indices = jnp.arange(self.split_elements)
        napkin_indices = jnp.arange(6 - self.split_elements, 6)

        cheese_mask = jnp.zeros(6, dtype=bool).at[cheese_indices].set(True)
        napkin_mask = jnp.zeros(6, dtype=bool).at[napkin_indices].set(True)

         
        final_spawn = jnp.where(cheese_mask[:, None], new_cheese_pos, final_spawn)
        final_spawn = jnp.where(napkin_mask[:, None], new_napkin_pos, final_spawn)

        return level.replace(
            wall_map=new_wall_map,
            initial_mouse_pos=new_initial_mouse_pos,
            cheese_pos=new_cheese_pos,
            dish_pos=final_spawn[0],
            fork_pos = final_spawn[1],
            spoon_pos = final_spawn[2],
            glass_pos = final_spawn[3],
            mug_pos = final_spawn[4],
            napkin_pos = final_spawn[5],
        )

@struct.dataclass
class CheeseonPileLevelMutator(base.LevelMutator):
    split_elements: int = 0
    max_cheese_radius: int = 0

    @functools.partial(jax.jit, static_argnames=('self',))
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape
        assert h > 3 and w > 3
        coords = einops.rearrange(jnp.indices((h, w)), 'c h w -> (h w) c')
        # eliminate the border from coords
        border = jnp.zeros((h, w), dtype=bool)
        border = border.at[(0, h-1), :].set(True)
        border = border.at[:, (0, w-1)].set(True)
        border = border.flatten()


        
        
        
        rng_row, rng_col = jax.random.split(rng)
        new_cheese_row = jax.random.choice(
            key=rng_row,
            a=jnp.arange(1, h-1),
        )
        new_cheese_col = jax.random.choice(
            key=rng_col,
            a=jnp.arange(1, w-1),
        )
        new_cheese_pos = jnp.array((new_cheese_row, new_cheese_col))
        
        #carve the wall
        new_wall_map = level.wall_map.at[
            new_cheese_pos[0],
            new_cheese_pos[1],
        ].set(False)

        distance_to_cheese = maze_solving.maze_distances(new_wall_map)[
            new_cheese_pos[0],
            new_cheese_pos[1],
        ]

        near_cheese = (distance_to_cheese <= self.max_cheese_radius).flatten()

        rng_spawn_pile, rng = jax.random.split(rng)
        new_napkin_pos = jax.random.choice(
            key=rng_spawn_pile,
            a =coords,
            p = near_cheese & ~border
        )
    

        #carve the wall
        new_wall_map = new_wall_map.at[
            new_napkin_pos[0],
            new_napkin_pos[1],
        ].set(False)

        napkin_hit_mouse = (new_napkin_pos == level.initial_mouse_pos).all()
        cheese_hit_mouse = (new_cheese_pos == level.initial_mouse_pos).all()

        new_initial_mouse_pos = jax.lax.select(
            napkin_hit_mouse,
            level.napkin_pos, 
            level.initial_mouse_pos,
        )

        new_initial_mouse_pos = jax.lax.select(
            cheese_hit_mouse,
            level.cheese_pos, 
            level.initial_mouse_pos,
        )

        final_spawn = []
        for i in range(6-self.split_elements):
            pos = new_napkin_pos
            final_spawn.append(pos)
        
        for i in range(self.split_elements):
            pos = new_cheese_pos
            final_spawn.append(pos)
        
        final_spawn.reverse()


        return level.replace(
            wall_map=new_wall_map,
            initial_mouse_pos=new_initial_mouse_pos,
            cheese_pos=new_cheese_pos,
            dish_pos=final_spawn[0],
            fork_pos = final_spawn[1],
            spoon_pos = final_spawn[2],
            glass_pos = final_spawn[3],
            mug_pos = final_spawn[4],
            napkin_pos = final_spawn[5],
        )


@struct.dataclass
class ScatterCheeseLevelMutator(base.LevelMutator):
    transpose_with_mouse_on_collision: bool = False
    transpose_with_pile_on_collision: bool = False
    split_elements: int = 0

    @functools.partial(jax.jit, static_argnames=('self',))
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape
        assert h > 3 and w > 3

        rng_row, rng_col = jax.random.split(rng)
        new_cheese_row = jax.random.choice(
            key=rng_row,
            a=jnp.arange(1, h-1),
        )
        new_cheese_col = jax.random.choice(
            key=rng_col,
            a=jnp.arange(1, w-1),
        )
        new_cheese_pos = jnp.array((new_cheese_row, new_cheese_col))

        new_wall_map = level.wall_map.at[
            new_cheese_pos[0],
            new_cheese_pos[1],
        ].set(False)

        hit_mouse = (new_cheese_pos == level.initial_mouse_pos).all()
        hit_pile = (new_cheese_pos == level.napkin_pos).all()
        if self.transpose_with_mouse_on_collision:
            new_initial_mouse_pos = jax.lax.select(
                hit_mouse,
                level.cheese_pos,
                level.initial_mouse_pos,
            )
        else:
            new_initial_mouse_pos = level.initial_mouse_pos
            new_cheese_pos = jax.lax.select(
                hit_mouse,
                level.cheese_pos,
                new_cheese_pos,
            )
        if self.transpose_with_pile_on_collision:
            new_napkin_pos = jax.lax.select(
                hit_pile,
                level.cheese_pos,
                level.napkin_pos,
            )
        else:
            new_napkin_pos = level.napkin_pos
            new_cheese_pos = jax.lax.select(
                hit_pile,
                level.cheese_pos,
                new_cheese_pos,
            )
        
        final_spawn = []
        for i in range(6-self.split_elements):
            pos = new_napkin_pos
            final_spawn.append(pos)
        for i in range(self.split_elements):
            pos = new_cheese_pos
            final_spawn.append(pos)
        final_spawn.reverse()

        return level.replace(
            wall_map=new_wall_map,
            initial_mouse_pos=new_initial_mouse_pos,
            cheese_pos=new_cheese_pos,
            dish_pos=final_spawn[0],
            fork_pos = final_spawn[1],
            spoon_pos = final_spawn[2],
            glass_pos = final_spawn[3],
            mug_pos = final_spawn[4],
            napkin_pos = final_spawn[5],
        )

@struct.dataclass
class StepPileLevelMutator(base.LevelMutator):
    transpose_with_mouse_on_collision: bool = False
    transpose_with_cheese_on_collision: bool = False
    split_elements: int = 0

    @functools.partial(jax.jit, static_argnames=('self',))
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape
        assert h > 3 and w > 3

        steps = jnp.array((
            (-1,  0),   # up
            ( 0, -1),   # left
            (+1,  0),   # down
            ( 0, +1),   # right
        ))

        valid_mask = jnp.array((
            level.napkin_pos[0] > 1,  # up
            level.napkin_pos[1] > 1,  # left
            level.napkin_pos[0] < h-2,  # down
            level.napkin_pos[1] < w-2,  # right
        ))

        chosen_step = jax.random.choice(
            key=rng,
            a=steps,
            p=valid_mask,
        )
        new_napkin_pos = level.napkin_pos + chosen_step

        new_wall_map = level.wall_map.at[
            new_napkin_pos[0],
            new_napkin_pos[1],
        ].set(False)

        hit_mouse = (new_napkin_pos == level.initial_mouse_pos).all()
        if self.transpose_with_mouse_on_collision:
            new_initial_mouse_pos = jax.lax.select(
                hit_mouse,
                level.napkin_pos,
                level.initial_mouse_pos,
            )
        else:
            new_initial_mouse_pos = level.initial_mouse_pos
            new_napkin_pos = jax.lax.select(
                hit_mouse,
                level.napkin_pos,
                new_napkin_pos,
            )
        
        #not doing it for cheese since pile and cheese can have the same position

        final_spawn = []
        for i in range(6-self.split_elements):
            pos = new_napkin_pos
            final_spawn.append(pos)
        for i in range(self.split_elements):
            pos = new_cheese_pos
            final_spawn.append(pos)
        final_spawn.reverse()

        return level.replace(
            wall_map=new_wall_map,
            initial_mouse_pos=new_initial_mouse_pos,
            cheese_pos=level.cheese_pos,
            dish_pos=final_spawn[0],
            fork_pos = final_spawn[1],
            spoon_pos = final_spawn[2],
            glass_pos = final_spawn[3],
            mug_pos = final_spawn[4],
            napkin_pos = final_spawn[5],
        )

@struct.dataclass
class ScatterPileLevelMutator(base.LevelMutator):

    transpose_with_mouse_on_collision: bool = False
    transpose_with_cheese_on_collision: bool = False
    split_elements: int = 0

    @functools.partial(jax.jit, static_argnames=('self',))
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape
        assert h > 3 and w > 3

        rng_row, rng_col = jax.random.split(rng)
        new_napkin_row = jax.random.choice(
            key=rng_row,
            a=jnp.arange(1, h-1),
        )
        new_napkin_col = jax.random.choice(
            key=rng_col,
            a=jnp.arange(1, w-1),
        )
        new_napkin_pos = jnp.array((new_napkin_row, new_napkin_col))

        new_wall_map = level.wall_map.at[
            new_napkin_pos[0],
            new_napkin_pos[1],
        ].set(False)

        hit_mouse = (new_napkin_pos == level.initial_mouse_pos).all()
        if self.transpose_with_mouse_on_collision:
            new_initial_mouse_pos = jax.lax.select(
                hit_mouse,
                level.napkin_pos,
                level.initial_mouse_pos,
            )
        else:
            new_initial_mouse_pos = level.initial_mouse_pos
            new_napkin_pos = jax.lax.select(
                hit_mouse,
                level.napkin_pos,
                new_napkin_pos,
            )
        
        #not doing it for cheese since pile and cheese can have the same position

        final_spawn = []
        for i in range(6-self.split_elements):
            pos = new_napkin_pos
            final_spawn.append(pos)
        for i in range(self.split_elements):
            pos = new_cheese_pos
            final_spawn.append(pos)
        final_spawn.reverse()

        return level.replace(
            wall_map=new_wall_map,
            initial_mouse_pos=new_initial_mouse_pos,
            cheese_pos=level.cheese_pos,
            dish_pos=final_spawn[0],
            fork_pos = final_spawn[1],
            spoon_pos = final_spawn[2],
            glass_pos = final_spawn[3],
            mug_pos = final_spawn[4],
            napkin_pos = final_spawn[5],
        )

@struct.dataclass
class MoveObjectsPileLevelMutator(base.LevelMutator):
    
        @functools.partial(jax.jit, static_argnames=('self'))
        def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
            h, w = level.wall_map.shape
            assert h > 3 and w > 3
    
    
            final_spawn = jnp.array([
                level.dish_pos,
                level.fork_pos,
                level.spoon_pos,
                level.glass_pos,
                level.mug_pos,
                level.napkin_pos,
            ])
            
            rng, rng_split = jax.random.split(rng)

            # Generate a random number between 0 and 6 for splitting among cheese and pile
            new_split_element = jax.random.randint(rng_split, (), 0, 7)

            cheese_mask = jnp.arange(6) < new_split_element

            # Create the array with objects moved  around
            final_spawn = jnp.where(
                cheese_mask[:, jnp.newaxis],
                level.cheese_pos,
                level.napkin_pos
            )

            return level.replace(
                wall_map=level.wall_map,
                initial_mouse_pos= level.initial_mouse_pos,
                cheese_pos=level.cheese_pos,
                dish_pos= final_spawn[0],
                fork_pos = final_spawn[1],
                spoon_pos = final_spawn[2],
                glass_pos = final_spawn[3],
                mug_pos = final_spawn[4],
                napkin_pos = final_spawn[5],
            )




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
        dish_map = jnp.zeros_like(levels.wall_map).at[
            levels.dish_pos[0],
            levels.dish_pos[1],
        ].set(True)
        
        # cheese - dish distance

        cheese_pile_dists = dists[
            jnp.arange(num_levels),
            levels.cheese_pos[:, 0],
            levels.cheese_pos[:, 1],
            levels.napkin_pos[:, 0],
            levels.napkin_pos[:, 1],
        ]

        cheese_pile_dists_finite = jnp.nan_to_num(
            cheese_pile_dists,
            posinf=(h-2)*(w-2)/2,
        )

        avg_cheese_pile_dist = cheese_pile_dists_finite.mean()

        # cheese_dists = dists[
        #     jnp.arange(num_levels),
        #     1,
        #     1,
        #     levels.cheese_pos[:, 0],
        #     levels.cheese_pos[:, 1],
        # ]
        # cheese_dists_finite = jnp.nan_to_num(
        #     cheese_dists,
        #     posinf=(h-2)*(w-2)/2,
        # )
        # avg_cheese_dist = cheese_dists_finite.mean()

        # shortest path length and solvability - mouse to cheese
        opt_dists_cheese = dists[
            jnp.arange(num_levels),
            levels.initial_mouse_pos[:, 0],
            levels.initial_mouse_pos[:, 1],
            levels.cheese_pos[:, 0],
            levels.cheese_pos[:, 1],
        ]
        solvable = ~jnp.isposinf(opt_dists_cheese)
        opt_dists_solvable_cheese = solvable * opt_dists_cheese
        opt_dists_finite_cheese = jnp.nan_to_num(opt_dists_cheese, posinf=(h-2)*(w-2)/2)

        # shortest path length and solvability - mouse to pile
        opt_dists_pile = dists[
            jnp.arange(num_levels),
            levels.initial_mouse_pos[:, 0],
            levels.initial_mouse_pos[:, 1],
            levels.napkin_pos[:, 0],
            levels.napkin_pos[:, 1],
        ]
        solvable_pile = ~jnp.isposinf(opt_dists_pile)
        opt_dists_solvable_pile = solvable_pile * opt_dists_pile
        opt_dists_finite_pile = jnp.nan_to_num(opt_dists_pile, posinf=(h-2)*(w-2)/2)

        #buffer
        cheese_on_pile= (levels.cheese_pos[:, 0] == levels.napkin_pos[:, 0]) & (levels.cheese_pos[:, 1] == levels.napkin_pos[:, 1])
        cheese_on_pile_avg =  cheese_on_pile.mean()

        #tracking elements on pile
        objects_on_cheese = (
            (levels.dish_pos == levels.cheese_pos).all(axis=1) +
            (levels.fork_pos == levels.cheese_pos).all(axis=1) +
            (levels.spoon_pos == levels.cheese_pos).all(axis=1) +
            (levels.glass_pos == levels.cheese_pos).all(axis=1) +
            (levels.mug_pos == levels.cheese_pos).all(axis=1) 
            ).mean()
        objects_on_pile = (
            (levels.dish_pos == levels.napkin_pos).all(axis=1) +
            (levels.fork_pos == levels.napkin_pos).all(axis=1) +
            (levels.spoon_pos == levels.napkin_pos).all(axis=1) +
            (levels.glass_pos == levels.napkin_pos).all(axis=1) +
            (levels.mug_pos == levels.napkin_pos).all(axis=1) 
            ).mean()
        
        dish_napkin_same = jnp.all(jnp.all(levels.dish_pos == levels.napkin_pos, axis=1))


        objects_on_cheese, objects_on_pile = jax.lax.cond(
            dish_napkin_same,
            lambda: (jnp.float32(6), jnp.float32(0)),
            lambda: (objects_on_cheese, objects_on_pile)
        )


        # rendered levels in a grid
        def render_level(level):
            state = self.env._reset(level)
            rgb = self.env.render_state(state)
            return rgb
        rendered_levels = jax.vmap(render_level)(levels)
        rendered_levels_pad = jnp.pad(
            rendered_levels,
            pad_width=((0, 0), (0, 1), (0, 1), (0, 0)),
        )
        rendered_levels_grid = einops.rearrange(
            rendered_levels_pad,
            '(level_h level_w) h w c -> (level_h h) (level_w w) c',
            level_w=64,
        )[:-1,:-1] # cut off last pixel of padding

        return {
            'layout': {
                'levels_img': rendered_levels_grid,
                # number of walls
                'num_walls_hist': num_walls,
                'num_walls_avg': num_walls.mean(),
                'num_walls_wavg': num_walls @ weights,
                # proportion of walls
                'prop_walls_hist': prop_walls,
                'prop_walls_avg': prop_walls.mean(),
                'prop_walls_wavg': prop_walls @ weights,
                # superimposed layout and position maps
                'wall_map_avg_img': util.viridis(wall_map.mean(axis=0)),
                'wall_map_wavg_img': util.viridis(jnp.einsum('lhw,l->hw', wall_map, weights)),
                'mouse_map_avg_img': util.viridis(mouse_map.mean(axis=0)),
                'mouse_map_wavg_img': util.viridis(jnp.einsum('lhw,l->hw', mouse_map, weights)),
                'cheese_map_avg_img': util.viridis(cheese_map.mean(axis=0)),
                'cheese_map_wavg_img': util.viridis(jnp.einsum('lhw,l->hw', cheese_map, weights)),
                'pile_map_avg_img': util.viridis(dish_map.mean(axis=0)),
                'pile_map_wavg_img': util.viridis(jnp.einsum('lhw,l->hw', dish_map, weights)),

            },
            'distances': {
                # solvability
                'solvable_num': solvable.sum(),
                'solvable_avg': solvable.mean(),
                'solvable_wavg': solvable @ weights,
                # optimal dist mouse to cheese
                'mouse-cheese_dist_finite_hist': opt_dists_finite_cheese,
                'mouse-cheese_dist_finite_avg': opt_dists_finite_cheese.mean(),
                'mouse-cheese_dist_finite_wavg': (opt_dists_finite_cheese @ weights),
                'mouse-cheese_dist_solvable_avg': opt_dists_solvable_cheese.sum() / solvable.sum(),
                'mouse-cheese_dist_solvable_wavg': (opt_dists_solvable_cheese @ weights) / (solvable @ weights),
                # optimal dist from mouse to pile
                'mouse-pile_dist_finite_hist': opt_dists_finite_pile,
                'mouse-pile_dist_finite_avg': opt_dists_finite_pile.mean(),
                'mouse-pile_dist_finite_wavg': (opt_dists_finite_pile @ weights),
                'mouse-pile_dist_solvable_avg': opt_dists_solvable_pile.sum() / solvable_pile.sum(),
                'mouse-pile_dist_solvable_wavg': (opt_dists_solvable_pile @ weights) / (solvable_pile @ weights),
                # avg cheese-dish distance
                'cheese-pile_dist_hist': cheese_pile_dists_finite,
                'cheese-pile_dist_avg': avg_cheese_pile_dist,
                'cheese-pile_dist_wavg': cheese_pile_dists_finite @ weights,
                # elements on cheese and pile
                '#objects_on_cheese': objects_on_cheese,
                '#objects_on_pile': objects_on_pile,

                #buffer metrics
                'levels_cheese_is_on_pile_avg':  cheese_on_pile_avg 
            },  
        }
