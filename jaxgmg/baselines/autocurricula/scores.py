import functools

import jax
import jax.numpy as jnp
from chex import Array
from jaxgmg.environments.base import Level

from jaxgmg.baselines.experience import Rollout
from jaxgmg.environments import cheese_in_the_corner
from jaxgmg.environments import keys_and_chests
from jaxgmg.environments import cheese_on_a_dish
from jaxgmg.environments import cheese_on_a_pile
from jaxgmg.environments import minigrid_maze
from jaxgmg.baselines import experience



@functools.partial(jax.jit, static_argnames=["regret_estimator"])
def plr_compute_scores(
    regret_estimator: str,
    rollouts: Rollout,              # Rollout[num_levels] with Transition[num_steps]
    advantages: Array,              # float[num_levels, num_steps]
    proxy_advantages: Array | None, # optional float[num_levels, num_steps]
    levels: Level,                  # Level[num_levels]
) -> Array:                         # float[num_levels]
    """
    Compute 'score' for level prioritisation, using a named scoring method.
    
    Inputs:

    * regret_estimator : str (static)
            One of a specific number of regret estimation methods. This
            determines the type of score quantity that is computed. Not all
            of these are actually estimators of regret, that's a historical
            name, we should probably change it to the more generic 'method'.
            See below for a list of methods.
    * rollouts : Rollout[num_levels] with Transition[num_steps].
            The experience data from which the score should be computed.
    * advantages : float[num_levels, num_steps]
            When training with PPO, we probably have already computed the
            GAEs from the rollouts. In order to skip computing them again,
            provide them here.
    * proxy_advantages : optional float[num_levels, num_steps] or None
            When training with PPO, we probably have already computed the
            GAEs of the proxy reward from the rollouts. In order to skip
            computing them again, provide them here.
    * levels : Level[num_levels]
            The levels probably shouldn't be needed for computing the scores,
            as they are meant to be based on the rollouts. However, for
            ORACLE versions of the scores (used for evaluating estimators)
            they are provided here.

    Returns:

    * scores : float[num_levels]
            One score for each level.
    
    Methods (not case sensitive):

    * 'absGAE': L1 value loss or absolute value of the GAE. Used in the
      original PLR paper.
    * 'PVL': poxitive value loss, or max(GAE, 0). Used in the Robust PLR
      paper.
    * Methods under development:
      * TODO: document the various oracle and proxy-shaped regret
        definitions...
    * Methods not yet implemented:
      * 'MaxMC' (TODO): Maximum Monte-Carlo estimate of the regret. Requires
        tracking the highest return ever achieved for a level. Not yet
        implemented.

    Dev notes: This function is seriously in need of some refactoring after
    the various deadlines are over and we have a chance.
    
    * We should make providing advantages of both types optional and have a
      default so it is actually optional.
    * We should be doing this with vmap rather than manually mapping
      everything across the levels axis. Not sure why I didn't realise that
      immediately.
    * It's unsustainable to define a separate method for each environment and
      proxy. Rethink that.
    * We should actually implement MaxMC. This probably involves having the
      UED methods track the max return achieved in the episodes.
    """
    match regret_estimator.lower():
        case "absgae":
            return jnp.abs(advantages).mean(axis=1)
        case "pvl":
            return jnp.maximum(advantages, 0).mean(axis=1)
        case "regret_diff_pvl":
            pvl = jnp.maximum(advantages, 0).mean(axis=1)
            proxy_pvl = jnp.maximum(proxy_advantages, 0).mean(axis=1)
            return pvl - proxy_pvl
        case "proxy_regret_corner":
            # true_reward = rollouts.transitions.reward.sum(axis=1)
            # proxy_reward = rollouts.transitions.info['proxy_rewards']['corner'].sum(axis=1)
            env = cheese_in_the_corner.Env(
                obs_level_of_detail=0,
                penalize_time=False,
                terminate_after_cheese_and_corner=False,
            )
            level_solver = cheese_in_the_corner.LevelSolver(
                env=env,
                discount_rate=0.999,
            )
            eval_on_benchmark_returns = level_solver.vmap_level_value(
                level_solver.vmap_solve(levels),
                levels
            )
            eval_of_proxy_benchmark_returns = level_solver.vmap_level_value_proxy(
                level_solver.vmap_solve_proxies(levels),
                levels,
            )
            eval_off_level_set = experience.compute_rollout_metrics(
                rollouts=rollouts,
                discount_rate=0.999,
                benchmark_returns=eval_on_benchmark_returns,
                benchmark_proxies=eval_of_proxy_benchmark_returns,
                )
            true_reward = eval_off_level_set['lvl_avg_return_hist']
            proxy_reward = eval_off_level_set['proxy_corner']['lvl_avg_return_hist']
            return jnp.maximum(true_reward - proxy_reward, 0)
        case "true_regret_corner":
            env = cheese_in_the_corner.Env(
                obs_level_of_detail=0,
                penalize_time=False,
                terminate_after_cheese_and_corner=False,
            )
            level_solver = cheese_in_the_corner.LevelSolver(
                env=env,
                discount_rate=0.999,
            )
            eval_on_benchmark_returns = level_solver.vmap_level_value(
                level_solver.vmap_solve(levels),
                levels
            )
            eval_of_proxy_benchmark_returns = level_solver.vmap_level_value_proxy(
                level_solver.vmap_solve_proxies(levels),
                levels,
            )
            eval_off_level_set = experience.compute_rollout_metrics(
                rollouts=rollouts,
                discount_rate=0.999,
                benchmark_returns=eval_on_benchmark_returns,
                benchmark_proxies=eval_of_proxy_benchmark_returns,
                )
            regret_true_reward = eval_off_level_set['lvl_benchmark_regret_hist']
            return jnp.maximum(regret_true_reward,0)
        case "relative_true_regret_corner":
            env = cheese_in_the_corner.Env(
                obs_level_of_detail=0,
                penalize_time=False,
                terminate_after_cheese_and_corner=False,
            )
            level_solver = cheese_in_the_corner.LevelSolver(
                env=env,
                discount_rate=0.999,
            )
            eval_on_benchmark_returns = level_solver.vmap_level_value(
                level_solver.vmap_solve(levels),
                levels
            )
            eval_of_proxy_benchmark_returns = level_solver.vmap_level_value_proxy(
                level_solver.vmap_solve_proxies(levels),
                levels,
            )
            eval_off_level_set = experience.compute_rollout_metrics(
                rollouts=rollouts,
                discount_rate=0.999,
                benchmark_returns=eval_on_benchmark_returns,
                benchmark_proxies=eval_of_proxy_benchmark_returns,
            )
            regret_true_reward = eval_off_level_set['lvl_benchmark_regret_hist'] #shape float[num_levels]
            regret_proxy_reward = eval_off_level_set['proxy_corner']['lvl_benchmark_regret_hist_proxy_corner'] #shape float[num_levels]
            return jnp.maximum(regret_true_reward - regret_proxy_reward,0)
        case "relative_true_regret_dish":
            env = cheese_on_a_dish.Env(
                obs_level_of_detail=0,
                penalize_time=False,
                terminate_after_cheese_and_dish=False,
            )
            level_solver = cheese_on_a_dish.LevelSolver(
                env=env,
                discount_rate=0.999,
            )
            eval_on_benchmark_returns = level_solver.vmap_level_value(
                level_solver.vmap_solve(levels),
                levels
            )
            eval_of_proxy_benchmark_returns = level_solver.vmap_level_value_proxy(
                level_solver.vmap_solve_proxies(levels),
                levels,
            )
            eval_off_level_set = experience.compute_rollout_metrics(
                rollouts=rollouts,
                discount_rate=0.999,
                benchmark_returns=eval_on_benchmark_returns,
                benchmark_proxies=eval_of_proxy_benchmark_returns,
                )
            regret_true_reward = eval_off_level_set['lvl_benchmark_regret_hist']
            regret_proxy_reward = eval_off_level_set['proxy_dish']['lvl_benchmark_regret_hist_proxy_dish']
            return jnp.maximum(regret_true_reward - regret_proxy_reward,0)
        case "true_regret_dish":
            env = cheese_on_a_dish.Env(
                obs_level_of_detail=0,
                penalize_time=False,
                terminate_after_cheese_and_dish= False,
            )
            level_solver = cheese_on_a_dish.LevelSolver(
                env=env,
                discount_rate=0.999,
            )
            #print('transition',rollouts.transitions.env_state.level)
            #levels = rollouts.transitions.env_state.level[:,0]
            eval_on_benchmark_returns = level_solver.vmap_level_value(
                level_solver.vmap_solve(levels),
                levels
            )

            eval_of_proxy_benchmark_returns = level_solver.vmap_level_value_proxy(
                level_solver.vmap_solve_proxies(levels),
                levels,
            )

            eval_off_level_set = experience.compute_rollout_metrics(
                rollouts=rollouts,
                discount_rate=0.999,
                benchmark_returns=eval_on_benchmark_returns,
                benchmark_proxies=eval_of_proxy_benchmark_returns,
                )
            regret_true_reward = eval_off_level_set['lvl_benchmark_regret_hist']
            return jnp.maximum(regret_true_reward ,0)
        case "relative_true_regret_pile":
            env = cheese_on_a_pile.Env(
                obs_level_of_detail=0,
                penalize_time=False,
                terminate_after_cheese_and_pile= False,
            )
            level_solver = cheese_on_a_pile.LevelSolver(
                env=env,
                discount_rate=0.999,
            )
            eval_on_benchmark_returns = level_solver.vmap_level_value(
                level_solver.vmap_solve(levels),
                levels
            )
            eval_of_proxy_benchmark_returns = level_solver.vmap_level_value_proxy(
                level_solver.vmap_solve_proxies(levels),
                levels,
            )
            eval_off_level_set = experience.compute_rollout_metrics(
                rollouts=rollouts,
                discount_rate=0.999,
                benchmark_returns=eval_on_benchmark_returns,
                benchmark_proxies=eval_of_proxy_benchmark_returns,
            )
            regret_true_reward = eval_off_level_set['lvl_benchmark_regret_hist']
            regret_proxy_reward = eval_off_level_set['proxy_pile']['lvl_benchmark_regret_hist_proxy_pile']
            return jnp.maximum(regret_true_reward - regret_proxy_reward,0)
        case "true_regret_pile":
            env = cheese_on_a_pile.Env(
                obs_level_of_detail=0,
                penalize_time=False,
                terminate_after_cheese_and_pile= False,
            )
            level_solver = cheese_on_a_pile.LevelSolver(
                env=env,
                discount_rate=0.999,
            )
            eval_on_benchmark_returns = level_solver.vmap_level_value(
                level_solver.vmap_solve(levels),
                levels
            )
            eval_of_proxy_benchmark_returns = level_solver.vmap_level_value_proxy(
                level_solver.vmap_solve_proxies(levels),
                levels,
            )
            eval_off_level_set = experience.compute_rollout_metrics(
                rollouts=rollouts,
                discount_rate=0.999,
                benchmark_returns=eval_on_benchmark_returns,
                benchmark_proxies=eval_of_proxy_benchmark_returns,
                )
            regret_true_reward = eval_off_level_set['lvl_benchmark_regret_hist']
            return jnp.maximum(regret_true_reward ,0)
        case "true_regret_minigrid_maze":
            env = minigrid_maze.Env(
                obs_level_of_detail=0,
                penalize_time=False,
                terminate_after_cheese=False,
            )
            level_solver = minigrid_maze.LevelSolver(
                env=env,
                discount_rate=0.999,
            )
            eval_on_benchmark_returns = level_solver.vmap_level_value(
                level_solver.vmap_solve(levels),
                levels
            )
            eval_of_proxy_benchmark_returns = level_solver.vmap_level_value_proxy(
                level_solver.vmap_solve_proxies(levels),
                levels,
            )
            eval_off_level_set = experience.compute_rollout_metrics(
                rollouts=rollouts,
                discount_rate=0.999,
                benchmark_returns=eval_on_benchmark_returns,
                benchmark_proxies=eval_of_proxy_benchmark_returns,
            )
            regret_true_reward = eval_off_level_set['lvl_benchmark_regret_hist']
            return jnp.maximum(regret_true_reward ,0)
        case "relative_true_regret_minigrid_maze":
            env = minigrid_maze.Env(
                obs_level_of_detail=0,
                penalize_time=False,
                terminate_after_cheese=False,
            )
            level_solver = minigrid_maze.LevelSolver(
                env=env,
                discount_rate=0.999,
            )
            eval_on_benchmark_returns = level_solver.vmap_level_value(
                level_solver.vmap_solve(levels),
                levels
            )
            eval_of_proxy_benchmark_returns = level_solver.vmap_level_value_proxy(
                level_solver.vmap_solve_proxies(levels),
                levels,
            )
            eval_off_level_set = experience.compute_rollout_metrics(
                rollouts=rollouts,
                discount_rate=0.999,
                benchmark_returns=eval_on_benchmark_returns,
                benchmark_proxies=eval_of_proxy_benchmark_returns,
            )
            regret_true_reward = eval_off_level_set['lvl_benchmark_regret_hist']
            regret_proxy_reward = eval_off_level_set['proxy_corner']['lvl_benchmark_regret_hist_proxy_corner']
            return jnp.maximum(regret_true_reward - regret_proxy_reward,0)
        case "proxy_regret_dish":
            env = cheese_on_a_dish.Env(
                obs_level_of_detail=0,
                penalize_time=False,
                terminate_after_cheese_and_dish=False,
            )
            level_solver = cheese_on_a_dish.LevelSolver(
                env=env,
                discount_rate=0.999,
            )
            eval_on_benchmark_returns = level_solver.vmap_level_value(
                level_solver.vmap_solve(levels),
                levels
            )
            eval_of_proxy_benchmark_returns = level_solver.vmap_level_value_proxy(
                level_solver.vmap_solve_proxies(levels),
                levels,
            )
            eval_off_level_set = experience.compute_rollout_metrics(
                rollouts=rollouts,
                discount_rate=0.999,
                benchmark_returns=eval_on_benchmark_returns,
                benchmark_proxies=eval_of_proxy_benchmark_returns,
                )
            true_reward = eval_off_level_set['lvl_avg_return_hist']
            proxy_reward = eval_off_level_set['proxy_dish']['lvl_avg_return_hist']
            return jnp.maximum(true_reward - proxy_reward, 0)
        case "proxy_regret_pile":
            env = cheese_on_a_pile.Env(
                obs_level_of_detail=0,
                penalize_time=False,
                terminate_after_cheese_and_corner=False,
            )
            level_solver = cheese_on_a_pile.LevelSolver(
                env=env,
                discount_rate=0.999,
            )
            eval_on_benchmark_returns = level_solver.vmap_level_value(
                level_solver.vmap_solve(levels),
                levels
            )
            eval_of_proxy_benchmark_returns = level_solver.vmap_level_value_proxy(
                level_solver.vmap_solve_proxies(levels),
                levels,
            )
            eval_off_level_set = experience.compute_rollout_metrics(
                rollouts=rollouts,
                discount_rate=0.999,
                benchmark_returns=eval_on_benchmark_returns,
                benchmark_proxies=eval_of_proxy_benchmark_returns,
                )
            true_reward = eval_off_level_set['lvl_avg_return_hist']
            proxy_reward = eval_off_level_set['proxy_pile']['lvl_avg_return_hist']
            return jnp.maximum(true_reward - proxy_reward, 0)
        case "proxy_regret_minigrid_maze":
            env = minigrid_maze.Env(
                obs_level_of_detail=0,
                penalize_time=False,
                terminate_after_cheese=False,
            )
            level_solver = minigrid_maze.LevelSolver(
                env=env,
                discount_rate=0.999,
            )
            eval_on_benchmark_returns = level_solver.vmap_level_value(
                level_solver.vmap_solve(levels),
                levels
            )
            eval_of_proxy_benchmark_returns = level_solver.vmap_level_value_proxy(
                level_solver.vmap_solve_proxies(levels),
                levels,
            )
            eval_off_level_set = experience.compute_rollout_metrics(
                rollouts=rollouts,
                discount_rate=0.999,
                benchmark_returns=eval_on_benchmark_returns,
                benchmark_proxies=eval_of_proxy_benchmark_returns,
            )
            true_reward = eval_off_level_set['lvl_avg_return_hist']
            proxy_reward = eval_off_level_set['proxy_corner']['lvl_avg_return_hist']
            return jnp.maximum(regret_true_reward - regret_proxy_reward,0)
        case "maxmc":
            raise NotImplementedError # TODO
        case _:
            raise ValueError("Invalid return estimator name.")


