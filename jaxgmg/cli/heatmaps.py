"""
Test heatmap visualisations of splayed level sets.
"""

import jax
import jax.numpy as jnp
import einops

from jaxgmg.procgen import maze_generation
from jaxgmg.environments import cheese_in_the_corner
# from jaxgmg.environments import cheese_on_a_dish
# from jaxgmg.environments import keys_and_chests
# from jaxgmg.environments import monster_world
# from jaxgmg.environments import lava_land
# from jaxgmg.environments import follow_me

from jaxgmg.cli import util


def heatmap_demo(level_set, states, level_solver, inverse_temperature):

    print("solving the levels...")
    # note: intentially does not take advantage of splay structure
    solns = level_solver.vmap_solve(level_set.levels)

    print("="*56)
    print("evaluating the states...")
    values = jax.vmap(level_solver.state_value)(solns, states)
    print(util.img2str(
        jnp.zeros(level_set.grid_shape).at[level_set.levels_pos].set(values),
        colormap=util.viridis,
    ))
    util.print_legend({
        0.00: "value 0.00",
        0.25: "value 0.25",
        0.50: "value 0.50",
        0.75: "value 0.75",
        1.00: "value 1.00",
    }, colormap=util.viridis)
    
    print("="*56)
    print("calculating the best action for each state...")
    actions = jax.vmap(level_solver.state_action)(solns, states)
    print(util.img2str(
        jnp.zeros(level_set.grid_shape, dtype=int)
            .at[level_set.levels_pos].set(actions+1),
        colormap=util.sweetie16,
    ))
    util.print_legend({
        0: "invalid position",
        1: "up",
        2: "left",
        3: "down",
        4: "right",
    }, colormap=util.sweetie16)
    
    print("="*56)
    print("calculating action values for each state...")
    action_values = jax.vmap(level_solver.state_action_values)(solns, states)
    action_values_vis = (
        jnp.full((4, 4, *level_set.grid_shape, 3), 0.5)
            .at[(0, 1, *level_set.levels_pos)]
                .set(util.viridis(action_values[:,0]))
            .at[(1, 0, *level_set.levels_pos)]
                .set(util.viridis(action_values[:,1]))
            .at[(2, 1, *level_set.levels_pos)]
                .set(util.viridis(action_values[:,2]))
            .at[(1, 2, *level_set.levels_pos)]
                .set(util.viridis(action_values[:,3]))
    )
    print(util.img2str(
        einops.rearrange(action_values_vis, 'y x h w c -> (h y) (w x) c'),
    ))
    util.print_legend({
        0.00: "value 0.00",
        0.25: "value 0.25",
        0.50: "value 0.50",
        0.75: "value 0.75",
        1.00: "value 1.00",
    }, colormap=util.viridis)
    print("within each diamond the four points represent the value of ")
    print("moving in the analogous direction")
    
    print("="*56)
    print("calculating softmax action distribution for each state...")
    action_distribution = jax.nn.softmax(
        inverse_temperature * action_values,
        axis=1,
    )
    action_dist_vis = (
        jnp.full((4, 4, *level_set.grid_shape, 3), 0.5)
            .at[(0, 1, *level_set.levels_pos)]
                .set(util.viridis(action_distribution[:,0]))
            .at[(1, 0, *level_set.levels_pos)]
                .set(util.viridis(action_distribution[:,1]))
            .at[(2, 1, *level_set.levels_pos)]
                .set(util.viridis(action_distribution[:,2]))
            .at[(1, 2, *level_set.levels_pos)]
                .set(util.viridis(action_distribution[:,3]))
    )
    print(util.img2str(
        einops.rearrange(action_dist_vis, 'y x h w c -> (h y) (w x) c'),
    ))
    util.print_legend({
        0.00: "probability 0.00",
        0.50: "probability 0.50",
        1.00: "probability 1.00",
    }, colormap=util.viridis)
    print("within each diamond the four points represent the probability of ")
    print("moving in the analogous direction")
    
    print("="*56)
    print("synthesizing into a single color image...")
    action_mixes_rgb = (
        action_distribution @ util.sweetie16(jnp.arange(1,5))
    )
    print(util.img2str(
        jnp.zeros(level_set.grid_shape+(3,))
            .at[:,:].set(util.sweetie16(0))
            .at[level_set.levels_pos].set(action_mixes_rgb)
    ))
    util.print_legend({
        0: "invalid position",
        1: "up",
        2: "left",
        3: "down",
        4: "right",
    }, colormap=util.sweetie16)
    print("these colours are mixed with the distribution weights")

    
def corner(
    # environment
    height: int = 7,
    width: int = 7,
    corner_size: int = 1,
    layout: str = 'tree',
    # splayer
    splayer: str = 'mouse',
    # reward/return (for solving)
    max_steps_in_episode: int = 128,
    penalize_time: bool = True,
    discount_rate: float = 0.9,
    inverse_temperature: float = 100,
    # misc
    seed: int = 42,
):
    """
    Test heatmap generation on the Cheese in the Corner environment.
    """
    match splayer:
        case 'cheese':
            splayer = cheese_in_the_corner.LevelSplayer.splay_cheese
        case 'mouse':
            splayer = cheese_in_the_corner.LevelSplayer.splay_mouse
        case 'cheese-and-mouse':
            splayer = cheese_in_the_corner.LevelSplayer.splay_cheese_and_mouse
        case _:
            raise ValueError(f"unknown splayer {splayer!r}")
    util.print_config(locals())

    print("preparing environment...")
    env = cheese_in_the_corner.Env(
        max_steps_in_episode=max_steps_in_episode,
        penalize_time=penalize_time,
        obs_level_of_detail=1,
    )

    print("preparing level generator...")
    level_generator = cheese_in_the_corner.LevelGenerator(
        height=height,
        width=width,
        corner_size=corner_size,
        maze_generator=maze_generation.get_generator_class_from_name(
            name=layout,
        )(),
    )

    print("preparing level solver...")
    level_solver = cheese_in_the_corner.LevelSolver(
        env=env,
        discount_rate=discount_rate,
    )

    print("generating a level...")
    rng = jax.random.PRNGKey(seed=seed)
    level = level_generator.sample(rng=rng)
    obs, state = env.reset_to_level(level)
    print(util.img2str(obs))

    print("splaying the level...")
    level_set = splayer(level)
    states = jax.vmap(env._reset)(level_set.levels)
    print("num levels:", level_set.num_levels)

    heatmap_demo(
        level_set=level_set,
        states=states,
        level_solver=level_solver,
        inverse_temperature=inverse_temperature,
    )


