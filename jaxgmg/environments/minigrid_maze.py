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
    * got_proxy : bool
            Whether the hero has already gotten the proxy goal.
    """
    hero_pos: chex.Array
    hero_dir: int
    got_goal: bool
    got_proxy: bool


@struct.dataclass
class Observation(base.Observation):
    """
    Observation for partially observable Maze environment.

    * image : bool[h, w, c] or float[h, w, rgb]
            The contents of the maze ahead of the agent.
    * orientation: float[4]
            One-hot encoding of the direction the hero is facing (dimensions
            correspond to up, left, down, right).
    """
    image: chex.Array
    orientation: chex.Array


class Action(enum.IntEnum):
    """
    The environment has a discrete action space of size 4 with the following
    meanings.
    """
    MOVE_FORWARD = 0
    TURN_LEFT    = 1
    STAY_STILL   = 2
    TURN_RIGHT   = 3


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


    @property
    def num_actions(self) -> int:
        return len(Action)
    

    def obs_type(self, level: Level) -> PyTree[jax.ShapeDtypeStruct]:
        # TODO: this will not work for RGB observations...
        C = len(Channel)
        return Observation(
            image=jax.ShapeDtypeStruct(
                shape=(self.obs_height, self.obs_width, C),
                dtype=bool,
            ),
            orientation=jax.ShapeDtypeStruct(
                shape=(4,),
                dtype=float,
            ),
        )


    def _reset(
        self,
        level: Level,
    ) -> EnvState:
        return EnvState(
            hero_pos=level.initial_hero_pos,
            hero_dir=level.initial_hero_dir,
            got_goal=False,
            got_proxy=False,
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
        turn_left = (action == Action.TURN_LEFT)
        turn_right = (action == Action.TURN_RIGHT)
        new_dir = (state.hero_dir + turn_left - turn_right) % 4
        # move forward action
        move_forward = (action == Action.MOVE_FORWARD)
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

        #check if hero got to proxy goal
        proxy_pos = jnp.array([0, 0])
        got_proxy_corner = (state.hero_pos == proxy_pos).all()
        got_proxy_first_time = got_proxy_corner & ~state.got_proxy
        state = state.replace(got_proxy=state.got_proxy | got_proxy_corner)

        
        # rewards
        reward = got_goal_first_time.astype(float)
        proxy_reward = got_proxy_first_time.astype(float)
        
        # end of episode
        if self.terminate_after_goal:
            done = state.got_goal
        else:
            done = state.got_goal & state.got_proxy

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
    def _render_state_bool(self, state: EnvState) -> chex.Array:
        """
        Render a boolean grid image of the current state.
        """
        H, W = state.level.wall_map.shape
        C = len(Channel)
        image = jnp.zeros((H, W, C), dtype=bool)

        # render walls
        image = image.at[:, :, Channel.WALL].set(state.level.wall_map)

        # render hero
        image = image.at[
            state.hero_pos[0],
            state.hero_pos[1],
            Channel.HERO,
        ].set(True)
        
        # render goal
        image = image.at[
            state.level.goal_pos[0],
            state.level.goal_pos[1],
            Channel.GOAL,
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
            image_bool[:, :, Channel.WALL],
            image_bool[:, :, Channel.HERO],
            image_bool[:, :, Channel.GOAL],
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
        C = len(Channel)
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
        # encode orientation
        orientation = jnp.eye(4)[state.hero_dir]
        return Observation(
            image=image,
            orientation=orientation,
        )
    

    @functools.partial(jax.jit, static_argnames=('self',))
    def _render_obs_rgb(
        self,
        state: EnvState,
        spritesheet: dict[str, chex.Array],
    ) -> Observation:
        # get the boolean grid version of the observation
        obs_bool = self._render_obs_bool(state)
        image_bool = obs_bool.image
        H, W, _C = image_bool.shape

        # find out, for each position, which object to render
        # (for each position pick the first true index top-down this list)
        sprite_priority_vector_grid = jnp.stack([
            # one object
            image_bool[:, :, Channel.WALL],
            image_bool[:, :, Channel.HERO],
            image_bool[:, :, Channel.GOAL],
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
        return Observation(
            image=image_rgb,
            orientation=obs_bool.orientation,
        )


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
    corner_size: int = 1
    

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

        no_wall_map = ~wall_map
        no_wall_mask = no_wall_map.flatten()
        # sample spawn positions by sampling from a list of coordinate pairs
        coords = einops.rearrange(
            jnp.indices((self.height, self.width)),
            'c h w -> (h w) c',
        )
        corner_mask = (
            # first `corner_size` rows not including border
              (coords[:, 0] >= 1)
            & (coords[:, 0] <= self.corner_size)
            # first `corner_size` cols not including border
            & (coords[:, 1] >= 1)
            & (coords[:, 1] <= self.corner_size)
        ).flatten()

        cheese_mask = corner_mask & no_wall_mask
        cheese_mask = cheese_mask | (~(cheese_mask.any()) & corner_mask)
        rng_spawn_cheese, rng = jax.random.split(rng)
        goal_pos = jax.random.choice(
            key=rng_spawn_cheese,
            a=coords,
            axis=0,
            p=cheese_mask,
        )
        # ... in case there *was* a wall there , remove it
        wall_map = wall_map.at[goal_pos[0], goal_pos[1]].set(False)
        no_wall_map = ~wall_map
        no_wall_mask = no_wall_map.flatten()
        

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
            p=no_wall_mask & no_goal,
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


@struct.dataclass
class MemoryTestLevelGenerator(base.LevelGenerator):
    """
    Probe level generator for Maze environment.
    """


    @functools.partial(jax.jit, static_argnames=('self',))
    def sample(self, rng: chex.PRNGKey) -> Level:
        # fixed wall map
        wall_map = jnp.array(
            [
                [ 1, 1, 1, 1, 1, 1, 1,],
                [ 1, 0, 0, 0, 0, 0, 1,],
                [ 1, 0, 1, 0, 1, 0, 1,],
                [ 1, 0, 1, 0, 1, 0, 1,],
                [ 1, 0, 1, 1, 1, 0, 1,],
                [ 1, 1, 1, 1, 1, 1, 1,],
            ],
            dtype=bool,
        )

        # random goal spawn position
        goal_pos = jax.random.choice(
            key=rng,
            a=jnp.array([[4, 1], [4, 5]]),
            axis=0,
        )

        # fixed hero spawn position and orientation
        initial_hero_pos = jnp.array([3, 3])
        initial_hero_dir = 1

        return Level(
            wall_map=wall_map,
            goal_pos=goal_pos,
            initial_hero_pos=initial_hero_pos,
            initial_hero_dir=initial_hero_dir,
        )

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
            level.goal_pos[0],
            level.goal_pos[1],
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
            state.hero_pos[0],
            state.hero_pos[1],
            4, # stay here
        ]

        # reward when we get to the cheese is 1 iff the cheese is still there
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
                    state.hero_pos[0],
                    state.hero_pos[1],
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

        # reward when we get to the cheese is 1 iff the cheese is still there
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
            state.goal_pos[0],
            state.goal_pos[1],
            :4,
        ])
        return action


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
        # exclude current hero and goal spawn positions
        valid_map = valid_map.at[
            (level.goal_pos[0], level.initial_hero_pos[0]),
            (level.goal_pos[1], level.initial_hero_pos[1]),
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
class StepHeroLevelMutator(base.LevelMutator):
    transpose_with_goal_on_collision: bool


    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape
        assert h > 3 and w > 3, "level too small"

        # move the hero in a random direction (within bounds)
        steps = jnp.array((
            (-1,  0),   # up
            ( 0, -1),   # left
            (+1,  0),   # down
            ( 0, +1),   # right
        ))
        valid_mask = jnp.array((
            level.initial_hero_pos[0] >= 2,
            level.initial_hero_pos[1] >= 2,
            level.initial_hero_pos[0] <= h-3,
            level.initial_hero_pos[1] <= w-3,
        ))
        chosen_step = jax.random.choice(
            key=rng,
            a=steps,
            p=valid_mask,
        )
        new_initial_hero_pos = level.initial_hero_pos + chosen_step

        # carve through walls
        new_wall_map = level.wall_map.at[
            new_initial_hero_pos[0],
            new_initial_hero_pos[1],
        ].set(False)

        # resolve collision with goal
        hit_goal = (new_initial_hero_pos == level.goal_pos).all()
        if self.transpose_with_goal_on_collision:
            # transpose hero with goal
            new_goal_pos = jax.lax.select(
                hit_goal,
                level.initial_hero_pos,
                level.goal_pos,
            )
        else:
            # mutation fails to move hero
            new_goal_pos = level.goal_pos
            new_initial_hero_pos = jax.lax.select(
                hit_goal,
                level.initial_hero_pos,
                new_initial_hero_pos,
            )

        return level.replace(
            wall_map=new_wall_map,
            initial_hero_pos=new_initial_hero_pos,
            goal_pos=new_goal_pos,
        )


@struct.dataclass
class TurnHeroLevelMutator(base.LevelMutator):


    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        # turn the hero to face a random direction (not the current direction)
        valid_mask = jnp.ones(4, dtype=int).at[level.initial_hero_dir].set(0)
        new_initial_hero_dir = jax.random.choice(
            key=rng,
            a=4,
            p=valid_mask,
        )
        return level.replace(initial_hero_dir=new_initial_hero_dir)


@struct.dataclass
class ScatterAndSpinHeroLevelMutator(base.LevelMutator):
    transpose_with_goal_on_collision: bool


    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape
        rng_scatter, rng_spin = jax.random.split(rng)

        # teleport the hero to a random location within bounds
        rng_row, rng_col = jax.random.split(rng_scatter)
        new_hero_row = jax.random.choice(
            key=rng_row,
            a=jnp.arange(1, h-1),
        )
        new_hero_col = jax.random.choice(
            key=rng_col,
            a=jnp.arange(1, w-1),
        )
        new_initial_hero_pos = jnp.array((
            new_hero_row,
            new_hero_col,
        ))

        # turn the hero to face a random direction
        new_initial_hero_dir = jax.random.choice(
            key=rng_spin,
            a=4,
        )

        # carve through walls
        new_wall_map = level.wall_map.at[
            new_initial_hero_pos[0],
            new_initial_hero_pos[1],
        ].set(False)

        # resolve collision with goal
        hit_goal = (new_initial_hero_pos == level.goal_pos).all()
        if self.transpose_with_goal_on_collision:
            # transpose hero with goal
            new_goal_pos = jax.lax.select(
                hit_goal,
                level.initial_hero_pos,
                level.goal_pos,
            )
        else:
            # mutation fails to move hero
            new_goal_pos = level.goal_pos
            new_initial_hero_pos = jax.lax.select(
                hit_goal,
                level.initial_hero_pos,
                new_initial_hero_pos,
            )

        return level.replace(
            wall_map=new_wall_map,
            initial_hero_pos=new_initial_hero_pos,
            initial_hero_dir=new_initial_hero_dir,
            goal_pos=new_goal_pos,
        )


@struct.dataclass
class StepGoalLevelMutator(base.LevelMutator):


    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape
        assert h > 3 and w > 3, "level too small"

        # move the goal in a random direction (within bounds)
        steps = jnp.array((
            (-1,  0),   # up
            ( 0, -1),   # left
            (+1,  0),   # down
            ( 0, +1),   # right
        ))
        valid_mask = jnp.array((
            level.goal_pos[0] >= 2,
            level.goal_pos[1] >= 2,
            level.goal_pos[0] <= h-3,
            level.goal_pos[1] <= w-3,
        ))
        chosen_step = jax.random.choice(
            key=rng,
            a=steps,
            p=valid_mask,
        )
        new_goal_pos = level.goal_pos + chosen_step

        # carve through walls
        new_wall_map = level.wall_map.at[
            new_goal_pos[0],
            new_goal_pos[1],
        ].set(False)

        # upon collision with hero, transpose goal with hero
        hit_hero = (new_goal_pos == level.initial_hero_pos).all()
        new_initial_hero_pos = jax.lax.select(
            hit_hero,
            level.goal_pos,
            level.initial_hero_pos,
        )

        return level.replace(
            wall_map=new_wall_map,
            initial_hero_pos=new_initial_hero_pos,
            goal_pos=new_goal_pos,
        )


@struct.dataclass
class ScatterGoalLevelMutator(base.LevelMutator):


    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape

        # teleport the goal to a random location within bounds
        rng_row, rng_col = jax.random.split(rng)
        new_goal_row = jax.random.choice(
            key=rng_row,
            a=jnp.arange(1, h-1),
        )
        new_goal_col = jax.random.choice(
            key=rng_col,
            a=jnp.arange(1, w-1),
        )
        new_goal_pos = jnp.array((
            new_goal_row,
            new_goal_col,
        ))

        # carve through walls
        new_wall_map = level.wall_map.at[
            new_goal_pos[0],
            new_goal_pos[1],
        ].set(False)

        # upon collision with hero, transpose goal with hero
        hit_hero = (new_goal_pos == level.initial_hero_pos).all()
        new_initial_hero_pos = jax.lax.select(
            hit_hero,
            level.goal_pos,
            level.initial_hero_pos,
        )

        return level.replace(
            wall_map=new_wall_map,
            initial_hero_pos=new_initial_hero_pos,
            goal_pos=new_goal_pos,
        )

@struct.dataclass
class CornerGoalLevelMutator(base.LevelMutator):
    corner_size: int = 1

    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape
        row_max = min(h - 2, self.corner_size)
        col_max = min(w - 2, self.corner_size)

        # teleport the cheese to a random location within the corner region
        rng_row, rng_col = jax.random.split(rng)
        new_goal_row = jax.random.choice(
            key=rng_row,
            a=jnp.arange(1, row_max+1), # 1 to row_max inclusive
        )
        new_goal_col = jax.random.choice(
            key=rng_col,
            a=jnp.arange(1, col_max+1), # 1 to col_max inclusive
        )
        new_goal_pos = jnp.array((
            new_goal_row,
            new_goal_col,
        ))

        # teleport the goal to a random location within bound

        # carve through walls
        new_wall_map = level.wall_map.at[
            new_goal_pos[0],
            new_goal_pos[1],
        ].set(False)

        # upon collision with hero, transpose goal with hero
        hit_hero = (new_goal_pos == level.initial_hero_pos).all()
        new_initial_hero_pos = jax.lax.select(
            hit_hero,
            level.goal_pos,
            level.initial_hero_pos,
        )

        return level.replace(
            wall_map=new_wall_map,
            initial_hero_pos=new_initial_hero_pos,
            goal_pos=new_goal_pos,
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
            * The character '#' maps to `Channel.WALL`.
            * The characters '^', '<', 'v', and '>' map to `Channel.HERO`
              (in initial orientations up, left, down, right respectively).
            * The character '*' maps to `Channel.GOAL`.
            * The character '.' maps to `len(Channel)`, i.e. none of the
              above, representing the absence of an item.
    """
    height: int
    width: int
    char_map = {
        '#': Channel.WALL,
        '^': Channel.HERO,
        '<': Channel.HERO,
        'v': Channel.HERO,
        '>': Channel.HERO,
        '*': Channel.GOAL,
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
        wall_map = (level_map == Channel.WALL)
        assert wall_map[0,:].all(), "top border incomplete"
        assert wall_map[:,0].all(), "left border incomplete"
        assert wall_map[-1,:].all(), "bottom border incomplete"
        assert wall_map[:,-1].all(), "right border incomplete"
        
        # extract goal position
        goal_map = (level_map == Channel.GOAL)
        assert goal_map.sum() == 1, "there must be exactly one goal"
        goal_pos = jnp.concatenate(
            jnp.where(goal_map, size=1)
        )

        # extract hero spawn position
        hero_spawn_map = (level_map == Channel.HERO)
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
# Level complexity metrics


@struct.dataclass
class LevelMetrics(base.LevelMetrics):


    @functools.partial(jax.jit, static_argnames=('self',))
    def compute_metrics(
        self,
        levels: Level,          # Level[num_levels]
        weights: chex.Array,    # float[num_levels]
    ) -> dict[str, Any]:        # metrics
        """
        TODO: This is copied from Cheese in the Corner. There is more to do
        like measure the distribution of initial directions and to measure
        whether the goal is visible in the initial state or not.
        """
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
                # # superimposed layout and position maps
                # 'wall_map_avg_img': util.viridis(wall_map.mean(axis=0)),
                # 'wall_map_wavg_img': util.viridis(jnp.einsum('lhw,l->hw', wall_map, weights)),
                # 'hero_map_avg_img': util.viridis(hero_map.mean(axis=0)),
                # 'hero_map_wavg_img': util.viridis(jnp.einsum('lhw,l->hw', hero_map, weights)),
                # 'goal_map_avg_img': util.viridis(goal_map.mean(axis=0)),
                # 'goal_map_wavg_img': util.viridis(jnp.einsum('lhw,l->hw', goal_map, weights)),
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


