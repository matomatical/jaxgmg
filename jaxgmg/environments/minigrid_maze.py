"""
Parameterised environment and level generator for Maze problem. Key
components are as follows.

Structs:

* The `Level` struct represents a particular maze layout and goal/agent
  spawn positions.
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
* TODO: document also splay methods and metrics class.
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
    * goal_pos : index[2]
            Coordinates of goal position (index into `wall_map`)
    * initial_hero_pos : index[2]
            Coordinates of initial hero position (index into `wall_map`)
    * initial_hero_dir : index
            Direction hero is facing, in WASD/counterclockwise order:
            * 0: up
            * 1: left
            * 2: down
            * 3: right
    """
    wall_map: chex.Array
    goal_pos: chex.Array
    initial_hero_pos: chex.Array
    initial_hero_dir: int


@struct.dataclass
class EnvState(base.EnvState):
    """
    Dynamic environment state within a particular level.

    * hero_pos : index[2]
            Current coordinates of the hero. Initialised to
            `level.initial_hero_pos`.
    * hero_dir : int
            Current orientation of the hero. Initialised to
            `level.initial_hero_dir`.
    * got_goal : bool
            Whether the hero has already gotten the goal.
    """
    hero_pos: chex.Array
    hero_dir: int
    got_goal: bool


@struct.dataclass
class Observation(base.Observation):
    """
    Observation for partially observable Maze environment.

    * image : bool[h, w, c] or float[h, w, rgb]
            The contents of the maze ahead of the agent.
    * orientation: int
            The direction the hero is facing.
    """
    image: chex.Array
    orientation: int


@struct.dataclass
class Env(base.Env):
    """
    Maze environment.

    In this environment the agent controls a hero navigating a grid-based
    maze. The hero must must navigate the maze to the goal, normally
    located near the top left corner.

    There are four available actions which deterministically move the hero
    one grid square up, right, down, or left respectively.
    * If the hero would hit a wall it remains in place.
    * If the hero hits the goal, the agent gains reward and the episode
      ends.

    Observations come in one of two formats:

    * Boolean: a H by W by C bool array where each channel represents the
      presence of one type of thing (walls, hero, goal).
    * Pixels: an 8H by 8W by 3 array of RGB float values where each 8 by 8
      tile corresponds to one grid square.
    """
    obs_height: int = 7
    obs_width: int = 7 # TODO: needs to be odd
    terminate_after_goal: bool = True


    class Action(enum.IntEnum):
        """
        The environment has a discrete action space of size 4 with the following
        meanings.
        """
        MOVE_FORWARD = 0
        TURN_LEFT    = 1
        STAY_STILL   = 2
        TURN_RIGHT   = 3


    @property
    def num_actions(self) -> int:
        return len(Env.Action)
    

    def obs_type(self, level: Level) -> PyTree[jax.ShapeDtypeStruct]:
        # TODO: this will not work for RGB observations...
        # TODO: remove dependence on level
        # TODO: i need to adapt the networks to work with obs structs...?
        C = len(Env.Channel)
        return Observation(
            image=jax.ShapeDtypeStruct(
                shape=(self.obs_height, self.obs_width, C),
                dtype=bool,
            ),
            orientation=jax.ShapeDtypeStruct(
                shape=(),
                dtype=int,
            ),
        )


    class Channel(enum.IntEnum):
        """
        The observations returned by the environment are an `h` by `w` by
        `channel` Boolean array, where the final dimensions 0 through 4
        indicate the following:

        * `WALL`:   True in the locations where there is a wall.
        * `HERO`:  True in the one location the hero occupies.
        * `GOAL`: True in the one location the goal occupies.
        """
        WALL    = 0
        HERO   = 1
        GOAL  = 2

    
    def _reset(
        self,
        level: Level,
    ) -> EnvState:
        return EnvState(
            hero_pos=level.initial_hero_pos,
            hero_dir=level.initial_hero_dir,
            got_goal=False,
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

        # note: turn left/turn right/step actions are mutually exclusive so
        # we can implement them independently

        # turn actions
        turn_left = (action == Env.Action.TURN_LEFT)
        turn_right = (action == Env.Action.TURN_RIGHT)
        new_dir = (state.hero_dir + turn_left - turn_right) % 4
        # move forward action
        move_forward = (action == Env.Action.MOVE_FORWARD)
        steps = jnp.array((
            (-1,  0),   # up
            ( 0, -1),   # left
            (+1,  0),   # down
            ( 0, +1),   # right
        ))
        ahead_pos = state.hero_pos + move_forward * steps[state.hero_dir]
        # collision detection
        hit_wall = state.level.wall_map[ahead_pos[0], ahead_pos[1]]
        new_pos = jax.lax.select(hit_wall, state.hero_pos, ahead_pos)
        # note: these are no-ops if action is STAY_STILL
        # update state
        state = state.replace(
            hero_pos=new_pos,
            hero_dir=new_dir,
        )

        # check if hero got to goal
        got_goal = (state.hero_pos == state.level.goal_pos).all()
        got_goal_first_time = got_goal & ~state.got_goal
        state = state.replace(got_goal=state.got_goal | got_goal)
        
        # rewards
        reward = got_goal_first_time.astype(float)
        
        # end of episode
        if self.terminate_after_goal:
            done = state.got_goal
        else:
            done = False

        return (
            state,
            reward,
            done,
            {},
        )

    
    @functools.partial(jax.jit, static_argnames=('self',))
    def _render_state_bool(self, state: EnvState) -> chex.Array:
        """
        Render a boolean grid image of the current state.
        """
        H, W = state.level.wall_map.shape
        C = len(Env.Channel)
        image = jnp.zeros((H, W, C), dtype=bool)

        # render walls
        image = image.at[:, :, Env.Channel.WALL].set(state.level.wall_map)

        # render hero
        image = image.at[
            state.hero_pos[0],
            state.hero_pos[1],
            Env.Channel.HERO,
        ].set(True)
        
        # render goal
        image = image.at[
            state.level.goal_pos[0],
            state.level.goal_pos[1],
            Env.Channel.GOAL,
        ].set(~state.got_goal)

        return image

    
    @functools.partial(jax.jit, static_argnames=('self',))
    def _render_state_rgb(
        self,
        state: EnvState,
        spritesheet: dict[str, chex.Array],
    ) -> chex.Array: # float[h, w, 3]
        """
        Render an RGB image of the current state based on a grid of tiles
        from the given spritesheet.

        Note: it's dimmed outside of visible region.
        """
        # get the boolean grid representation of the state
        image_bool = self._render_state_bool(state)
        H, W, _C = image_bool.shape

        # find out, for each position, which object to render
        # (for each position pick the first true index top-down this list)
        sprite_priority_vector_grid = jnp.stack([
            # one object
            image_bool[:, :, Env.Channel.WALL],
            image_bool[:, :, Env.Channel.HERO],
            image_bool[:, :, Env.Channel.GOAL],
            # no objects, 'default' (always true)
            jnp.ones((H, W), dtype=bool),
        ])
        chosen_sprites = jnp.argmax(sprite_priority_vector_grid, axis=0)

        # put the corresponding sprite into each square
        spritemap = jnp.stack([
            # one object
            spritesheet['WALL'],
            spritesheet['MOUSE'],
            spritesheet['GOAL'],
            # no objects
            spritesheet['PATH'],
        ])[chosen_sprites] # -> h w th tw rgb

        # identify currently-visible region
        i = state.hero_pos[0]
        j = state.hero_pos[1]
        h = self.obs_height
        w = self.obs_width
        M = max(h, w, 2) - 2 # maximum padding required
        mask = jnp.zeros((M+H+M, M+W+M), dtype=bool)
        fov = jnp.ones((h, w), dtype=bool)
        mask = jax.lax.select_n(
            state.hero_dir,
            jax.lax.dynamic_update_slice(mask, fov,   (M+i+1-h,  M+j-w//2)),
            jax.lax.dynamic_update_slice(mask, fov.T, (M+i-w//2, M+j+1-h )),
            jax.lax.dynamic_update_slice(mask, fov,   (M+i,      M+j-w//2)),
            jax.lax.dynamic_update_slice(mask, fov.T, (M+i-w//2, M+j     )),
        )
        mask = mask[M:M+H, M:M+W]

        # use it to visibly highlight sprites in that region vs. outside
        spritemap = jnp.where(
            mask.reshape(H, W, 1, 1, 1),
            0.2 + 0.8 * spritemap,
            spritemap,
        )
        
        # rearrange into required form
        image_rgb = einops.rearrange(
            spritemap,
            'h w th tw rgb -> (h th) (w tw) rgb',
        )
        return image_rgb


    @functools.partial(jax.jit, static_argnames=('self',))
    def _render_obs_bool(self, state: EnvState) -> chex.Array:
        """
        Return an observation with a boolean grid image.
        """
        # render the full image
        image = self._render_state_bool(state)
        
        # pad to allow slicing
        h = self.obs_height
        w = self.obs_width
        M = max(h, w, 2) - 2 # maximum padding required
        image = jnp.pad(
            array=image,
            pad_width=((M, M), (M, M), (0, 0)),
            mode='edge',
        )

        # construct an oriented slice of the state in front of the hero
        i = state.hero_pos[0]
        j = state.hero_pos[1]
        C = len(Env.Channel)
        # four possible slices
        case_facing_up = jnp.rot90(
            jax.lax.dynamic_slice(image, (M+i+1-h, M+j-w//2, 0), (h, w, C)),
            k=0,
        )
        case_facing_left = jnp.rot90(
            jax.lax.dynamic_slice(image, (M+i-w//2, M+j+1-h, 0), (w, h, C)),
            k=3,
        )
        case_facing_down = jnp.rot90(
            jax.lax.dynamic_slice(image, (M+i, M+j-w//2, 0), (h, w, C)),
            k=2,
        )
        case_facing_right = jnp.rot90(
            jax.lax.dynamic_slice(image, (M+i-w//2, M+j, 0), (w, h, C)),
            k=1,
        )
        # chosen slice
        image = jax.lax.select_n(
            state.hero_dir,
            case_facing_up,
            case_facing_left,
            case_facing_down,
            case_facing_right,
        )
        return Observation(
            image=image,
            orientation=state.hero_dir,
        )
    

    @functools.partial(jax.jit, static_argnames=('self',))
    def _render_obs_rgb(
        self,
        state: EnvState,
        spritesheet: dict[str, chex.Array],
    ) -> Observation:
        # get the boolean grid version of the observation
        image_bool = self._render_obs_bool(state).image
        H, W, _C = image_bool.shape

        # find out, for each position, which object to render
        # (for each position pick the first true index top-down this list)
        sprite_priority_vector_grid = jnp.stack([
            # one object
            image_bool[:, :, Env.Channel.WALL],
            image_bool[:, :, Env.Channel.HERO],
            image_bool[:, :, Env.Channel.GOAL],
            # no objects, 'default' (always true)
            jnp.ones((H, W), dtype=bool),
        ])
        chosen_sprites = jnp.argmax(sprite_priority_vector_grid, axis=0)

        # put the corresponding sprite into each square
        spritemap = jnp.stack([
            # one object
            spritesheet['WALL'],
            spritesheet['MOUSE'],
            spritesheet['GOAL'],
            # no objects
            spritesheet['PATH'],
        ])[chosen_sprites] # -> h w th tw rgb

        # rearrange into required form
        image_rgb = einops.rearrange(
            spritemap,
            'h w th tw rgb -> (h th) (w tw) rgb',
        )
        return image_rgb


# # # 
# Level generation


@struct.dataclass
class LevelGenerator(base.LevelGenerator):
    """
    Level generator for Maze environment. Given some maze configuration
    parameters and goal location parameter, provides a `sample` method that
    generates a random level.

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
    """
    height: int = 13
    width: int = 13
    maze_generator : mg.MazeGenerator = mg.TreeMazeGenerator()
    

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
        
        # spawn goal in some random valid position
        no_wall = ~wall_map.flatten()
        rng_spawn_goal, rng = jax.random.split(rng)
        goal_pos = jax.random.choice(
            key=rng_spawn_goal,
            a=coords,
            axis=0,
            p=no_wall,
        )

        # spawn hero in some random remaining valid position
        no_goal = jnp.ones_like(wall_map).at[
            goal_pos[0],
            goal_pos[1],
        ].set(False).flatten()
        rng_spawn_hero_pos, rng = jax.random.split(rng)
        initial_hero_pos = jax.random.choice(
            key=rng_spawn_hero_pos,
            a=coords,
            axis=0,
            p=no_wall & no_goal,
        )

        # spawn hero with some random orientation
        rng_spawn_hero_dir, rng = jax.random.split(rng)
        initial_hero_dir = jax.random.choice(
            key=rng_spawn_hero_dir,
            a=4,
        )

        return Level(
            wall_map=wall_map,
            goal_pos=goal_pos,
            initial_hero_pos=initial_hero_pos,
            initial_hero_dir=initial_hero_dir,
        )


# # # 
# Level parsing


@struct.dataclass
class LevelParser(base.LevelParser):
    """
    Level parser for Maze environment. Given some parameters determining
    level shape, provides a `parse` method that converts an ASCII depiction
    of a level into a Level struct. Also provides a `parse_batch` method that
    parses a list of level strings into a single vectorised Level PyTree.

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
            * The characters '^', '<', 'v', and '>' map to `Env.Channel.HERO`
              (in initial orientations up, left, down, right respectively).
            * The character '*' maps to `Env.Channel.GOAL`.
            * The character '.' maps to `len(Env.Channel)`, i.e. none of the
              above, representing the absence of an item.
    """
    height: int
    width: int
    char_map = {
        '#': Env.Channel.WALL,
        '^': Env.Channel.HERO,
        '<': Env.Channel.HERO,
        'v': Env.Channel.HERO,
        '>': Env.Channel.HERO,
        '*': Env.Channel.GOAL,
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
        ... # < # . #
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
            goal_pos=Array([3, 3], dtype=int32),
            initial_hero_pos=Array([2, 1], dtype=int32),
            initial_hero_dir=1,
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
        
        # extract goal position
        goal_map = (level_map == Env.Channel.GOAL)
        assert goal_map.sum() == 1, "there must be exactly one goal"
        goal_pos = jnp.concatenate(
            jnp.where(goal_map, size=1)
        )

        # extract hero spawn position
        hero_spawn_map = (level_map == Env.Channel.HERO)
        assert hero_spawn_map.sum() == 1, "there must be exactly one hero"
        initial_hero_pos = jnp.concatenate(
            jnp.where(hero_spawn_map, size=1)
        )

        # extract hero direction
        if '^' in level_str:
            initial_hero_dir = 0
        elif '<' in level_str:
            initial_hero_dir = 1
        elif 'v' in level_str:
            initial_hero_dir = 2
        else: # '>' in level_str:
            initial_hero_dir = 3

        return Level(
            wall_map=wall_map,
            goal_pos=goal_pos,
            initial_hero_pos=initial_hero_pos,
            initial_hero_dir=initial_hero_dir,
        )


# # # 
# Level solving


@struct.dataclass
class LevelSolution(base.LevelSolution):
    level: Level
    directional_distance_to_goal: chex.Array


@struct.dataclass
class LevelSolver(base.LevelSolver):


    @functools.partial(jax.jit, static_argnames=('self',))
    def solve(self, level: Level) -> LevelSolution:
        """
        Compute the distance from each possible hero position to the goal
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
        # compute distance between hero and goal
        dir_dist = maze_solving.maze_directional_distances(level.wall_map)
        dir_dist_to_goal = dir_dist[
            :,
            :,
            level.goal_pos[0],
            level.goal_pos[1],
            :,
        ]

        return LevelSolution(
            level=level,
            directional_distance_to_goal=dir_dist_to_goal,
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
        # steps to get to the goal: look up in distance cache
        optimal_dist = soln.directional_distance_to_goal[
            state.hero_pos[0],
            state.hero_pos[1],
            4, # stay here
        ]

        # reward when we get to the goal is 1 iff the goal is still there
        reward = (1.0 - state.got_goal)
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
        # steps to get to the goal for adjacent squares: look up in cache
        dir_dists = soln.directional_distance_to_goal[
            state.hero_pos[0],
            state.hero_pos[1],
        ] # -> float[5] (up left down right stay)
        # steps after taking each action, taking collisions into account:
        # replace inf values with stay-still values
        action_dists = jnp.where(
            jnp.isinf(dir_dists[:4]),
            dir_dists[4],
            dir_dists[:4],
        )

        # reward when we get to the goal is 1 iff the goal is still there
        reward = (1.0 - state.got_goal)
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

        * action : int                      # TODO: use the Env.Action enum?
                An optimal action from the given state.
                
        Notes:

        * If there are multiple equally optimal actions, this method will
          return the first according to the order up (0), left (1), down (2),
          or right (3).
        * As a special case of this, if the goal is unreachable, the
          returned action will be up (0).
        * If the goal is on the current square, the returned action is
          arbitrary, and in fact it might even be suboptimal, since if there
          is a wall the optimal action is to move into that wall.
        * If the goal has already been gotten then there is no more reward
          available, but this method will still direct the hero towards the
          goal position.
        * If the goal is too far away to reach by the end of the episode,
          this method will still direct the hero towards the goal.

        TODO: 

        * Make all environments have a 'stay action' will simplify these
          solutions a fair bit. The hero could stay when on the goal, or
          when the goal is unreachable, or when the goal is already
          gotten.
        """
        action = jnp.argmin(soln.directional_distance_to_goal[
            state.hero_pos[0],
            state.hero_pos[1],
            :4,
        ])
        return action


# # # 
# Splay functions
# Note that these functions are not jittable.


def splay_hero(level: Level):
    free_map = ~level.wall_map
    free_map = free_map.at[
        level.goal_pos[0],
        level.goal_pos[1],
    ].set(False)

    # assemble level batch
    num_levels = free_map.sum()
    levels = Level(
        wall_map=einops.repeat(
            level.wall_map,
            'h w -> n h w',
            n=num_levels,
        ),
        goal_pos=einops.repeat(
            level.goal_pos,
            'c -> n c',
            n=num_levels,
        ),
        initial_hero_pos=jnp.stack(jnp.where(free_map), axis=1),
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


def splay_goal(level: Level):
    free_map = ~level.wall_map
    free_map = free_map.at[
        level.initial_hero_pos[0],
        level.initial_hero_pos[1],
    ].set(False)

    # assemble level batch
    num_levels = free_map.sum()
    levels = Level(
        wall_map=einops.repeat(
            level.wall_map,
            'h w -> n h w',
            n=num_levels,
        ),
        goal_pos=jnp.stack(jnp.where(free_map), axis=1),
        initial_hero_pos=einops.repeat(
            level.initial_hero_pos,
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


def splay_goal_and_hero(level: Level):
    free_map = ~level.wall_map
    # macromaze
    free_metamap = free_map[None,None,:,:] & free_map[:,:,None,None]
    # remove hero/goal clashes
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
    # goal/hero spawn locations
    goal1, hero1, goal2, hero2 = jnp.where(free_metamap_hhww)
    levels = Level(
        wall_map=einops.repeat(
            level.wall_map,
            'h w -> n h w',
            n=num_levels,
        ),
        goal_pos=jnp.stack((goal1, goal2), axis=1),
        initial_hero_pos=jnp.stack((hero1, hero2), axis=1),
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

        # avg wall location, goal location, hero location
        wall_map = levels.wall_map
        hero_map = jnp.zeros_like(levels.wall_map).at[
            levels.initial_hero_pos[:, 0],
            levels.initial_hero_pos[:, 1],
        ].set(True)
        goal_map = jnp.zeros_like(levels.wall_map).at[
            levels.goal_pos[0],
            levels.goal_pos[1],
        ].set(True)
        
        # goal distance from top left corner
        goal_dists = dists[
            jnp.arange(num_levels),
            1,
            1,
            levels.goal_pos[:, 0],
            levels.goal_pos[:, 1],
        ]
        goal_dists_finite = jnp.nan_to_num(
            goal_dists,
            posinf=(h-2)*(w-2)/2,
        )
        avg_goal_dist = goal_dists_finite.mean()

        # shortest path length and solvability
        opt_dists = dists[
            jnp.arange(num_levels),
            levels.initial_hero_pos[:, 0],
            levels.initial_hero_pos[:, 1],
            levels.goal_pos[:, 0],
            levels.goal_pos[:, 1],
        ]
        solvable = ~jnp.isposinf(opt_dists)
        opt_dists_solvable = solvable * opt_dists
        opt_dists_finite = jnp.nan_to_num(opt_dists, posinf=(h-2)*(w-2)/2)
        
        # rendered levels in a grid
        def render_level(level):
            state = self.env._reset(level)
            rgb = self.env.get_obs(state, force_lod=1)
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
                'hero_map_avg_img': util.viridis(hero_map.mean(axis=0)),
                'hero_map_wavg_img': util.viridis(jnp.einsum('lhw,l->hw', hero_map, weights)),
                'goal_map_avg_img': util.viridis(goal_map.mean(axis=0)),
                'goal_map_wavg_img': util.viridis(jnp.einsum('lhw,l->hw', goal_map, weights)),
            },
            'distances': {
                # solvability
                'solvable_num': solvable.sum(),
                'solvable_avg': solvable.mean(),
                'solvable_wavg': solvable @ weights,
                # optimal dist hero to goal
                'hero_dist_finite_hist': opt_dists_finite,
                'hero_dist_finite_avg': opt_dists_finite.mean(),
                'hero_dist_finite_wavg': (opt_dists_finite @ weights),
                'hero_dist_solvable_avg': opt_dists_solvable.sum() / solvable.sum(),
                'hero_dist_solvable_wavg': (opt_dists_solvable @ weights) / (solvable @ weights),
                # optimal dist from goal to hero
                'goal_dist_hist': goal_dists_finite,
                'goal_dist_avg': goal_dists_finite.mean(),
                'goal_dist_wavg': goal_dists_finite @ weights,
            },
        }

