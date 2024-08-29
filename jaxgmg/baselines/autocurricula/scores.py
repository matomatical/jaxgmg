import functools

import jax
import jax.numpy as jnp
from chex import Array

from jaxgmg.baselines.experience import Rollout


@functools.partial(jax.jit, static_argnames=["regret_estimator"])
def plr_compute_scores(
    regret_estimator: str,
    rollouts: Rollout,  # Rollout[num_levels] with Transition[num_steps]
    advantages: Array,  # float[num_levels, num_steps]
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
        case "proxy_regret_corner_wdistance":
            true_reward = rollouts.transitions.reward.sum(axis=1)
            proxy_reward = rollouts.transitions.info['proxy_rewards']['corner'].sum(axis=1)
            mouse_pos = rollouts.transitions.env_state.mouse_pos[:, 0]
            cheese_pos = rollouts.transitions.env_state.level.cheese_pos[:, 0]
            final_distance = jnp.sqrt(jnp.sum((mouse_pos - cheese_pos)**2, axis=-1))
            maze_height = 11
            maze_width = 11
            max_distance = jnp.sqrt(maze_height**2 + maze_width**2)
            normalized_distance = final_distance / max_distance
            
            reward_diff = jnp.maximum(true_reward - proxy_reward,0)
            weight_reward_diff = 0.7
            weight_distance = 0.3
            return weight_reward_diff * reward_diff + weight_distance * normalized_distance
        case "proxy_regret_dish":
            true_reward = rollouts.transitions.reward.sum(axis=1)
            proxy_reward = rollouts.transitions.info['proxy_rewards']['proxy_dish'].sum(axis=1)
            return jnp.maximum(true_reward - proxy_reward,0)
        case "proxy_regret_pile":
            true_reward = rollouts.transitions.reward.sum(axis=1)
            proxy_reward = rollouts.transitions.info['proxy_rewards']['proxy_pile'].sum(axis=1)
            return jnp.maximum(true_reward - proxy_reward,0)
        case "maxmc":
            raise NotImplementedError # TODO
        case _:
            raise ValueError(f"Invalid regret estimator: {regret_estimator}")


