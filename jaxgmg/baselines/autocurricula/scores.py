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
from jaxgmg.baselines import experience



@functools.partial(jax.jit, static_argnames=["regret_estimator"])
def plr_compute_scores(
    regret_estimator: str,
    rollouts: Rollout,  # Rollout[num_levels] with Transition[num_steps]
    advantages: Array,  # float[num_levels, num_steps]
    level: Level,       # Level[num_levels]
) -> Array:             # float[num_levels]
    match regret_estimator.lower():
        case "absgae":
            return jnp.abs(advantages).mean(axis=1)
        case "pvl":
            return jnp.maximum(advantages, 0).mean(axis=1)
        case "proxy_regret_corner":
            true_reward = rollouts.transitions.reward.sum(axis=1)
            proxy_reward = rollouts.transitions.info['proxy_rewards']['corner'].sum(axis=1)
            return jnp.maximum(true_reward - proxy_reward,0)
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
            levels = level
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
        case "relative_pvl_regret_corner":
            raise NotImplementedError
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
            levels = level
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
            regret_proxy_reward = eval_off_level_set['corner']['lvl_benchmark_regret_hist_corner'] #shape float[num_levels]
            return jnp.maximum(regret_true_reward - regret_proxy_reward,0)
        case "relative_true_regret_dish":
            env = cheese_on_a_dish.Env(
            obs_level_of_detail=0,
            penalize_time=False,
            terminate_after_cheese_and_dish= False,
        )
            level_solver = cheese_on_a_dish.LevelSolver(
                env=env,
                discount_rate=0.999,
            )
            levels = level
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
            regret_proxy_reward = eval_off_level_set['dish']['lvl_benchmark_regret_hist_dish']
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
            levels = level

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
        case "proxy_regret_dish":
            true_reward = rollouts.transitions.reward.sum(axis=1)
            proxy_reward = rollouts.transitions.info['proxy_rewards']['dish'].sum(axis=1)
            return jnp.maximum(true_reward - proxy_reward,0)
        case "proxy_regret_pile":
            true_reward = rollouts.transitions.reward.sum(axis=1)
            proxy_reward = rollouts.transitions.info['proxy_rewards']['proxy_pile'].sum(axis=1)
            return jnp.maximum(true_reward - proxy_reward,0)
        case "maxmc":
            raise NotImplementedError # TODO
        case _:
            raise ValueError("Invalid return estimator.")


