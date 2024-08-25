"""
Types and utilities for collecting experience in an environment.
"""

import functools
from typing import Any

import einops
import jax
import jax.numpy as jnp

# typing
from chex import Array, PRNGKey
from flax import struct
from jaxgmg.environments.base import EnvState, Env, Level, Observation
from jaxgmg.baselines.networks import ActorCriticState, ActorCriticParams
from jaxgmg.baselines.networks import ActorCriticForwardPass


# # # 
# Represent transitions and rollouts


@struct.dataclass
class Transition:
    """
    Captures data involved in one environment step, from the observation to
    the actor/critic response to the reward, termination, and info response
    from the environment. Note: the next env_state and observation etc. is
    represented in the next transition (or for the final transition in the
    rollout, it's represented in the rollout itself.
    """
    env_state: EnvState
    obs: Observation
    net_state: ActorCriticState
    prev_action: int
    value: float
    action: int
    log_prob: float
    reward: float
    done: bool
    info: dict


@struct.dataclass
class Rollout:
    """
    Captures a sequence of environment steps. Fields:

    * trajectories : Transition[num_steps]
            The collected experience as a sequence of transitions.
    * final_value : float
            The network's output for the result of the final transition.
            Useful as a learning target.
    
    Note: Considered adding these fields, but decided to exclude them because
    they are currently unused so we don't need to compute them. They could be
    useful for recomputing the value target but we don't currently do that
    for the last value, it serves only as a static target for computing GAE.

    * final_env_state : jaxgmg.environments.base.EnvState
            The env state arising after the final transition. Useful for
            rendering the full rollout.
    * final_obs : Observation
            The observation arising after the final transition.
    * final_net_state : jaxgmg.baselines.networks.ActorCriticState
            The net state after the final transition.
    * final_prev_action : int
            The prev action after the final transition.
    """
    transitions: Transition             # Transition[num_steps]
    # final_env_state: EnvState
    # final_obs: Observation
    # final_net_state: ActorCriticState
    # final_prev_action: int
    final_value: float


# # # 
# Experience collection


@functools.partial(jax.jit, static_argnames=('net_apply', 'env', 'num_steps'))
def collect_rollout(
    rng: PRNGKey,
    num_steps: int,
    net_apply: ActorCriticForwardPass,
    net_params: ActorCriticParams,
    net_init_state: ActorCriticState,
    env: Env,
    level: Level,
) -> Rollout:
    """
    Reset an environment to `levels` and rollout a policy in these levels for
    `env_steps` steps.

    Parameters:

    * rng : PRNGKey
            Random state (consumed)
    * num_steps : static int
            The environments will run forward for this many steps.
    * net_apply : ActorCriticForwardPass
            The apply function should take params and an observation and
            return an action distribution and value prediction.
    * net_params : ActorCriticParams
            Policy parameter.
    * net_init_state : ActorCriticState
            An initial carry for the network.
    * env : static jaxgmg.environments.base.Env
            Provides functions `reset_to_level` and `step`.
    * level : jaxgmg.environments.base.Level
            Level struct.

    Returns:

    * rollout : Rollout
            The collected experience, a sequence of transitions.
    """
    # reset environments to the given levels
    env_obs, env_state = env.reset_to_level(level=level)
    # reset the net inputs to blank
    default_prev_action = jnp.int32(-1)

    # scan the steps of the rollout
    initial_carry = (
        env_state,
        env_obs,
        net_init_state,
        default_prev_action,
    )
    input_rngs = jax.random.split(rng, num_steps)

    def _env_step(carry, rng):
        env_state, obs, net_state, prev_action = carry

        # select action
        rng_action, rng = jax.random.split(rng)
        (
            action_distribution,
            critic_value,
            next_net_state,
        ) = net_apply(
            net_params,
            obs,
            net_state,
            prev_action,
        )
        action = action_distribution.sample(seed=rng_action)
        log_prob = action_distribution.log_prob(action)

        # step env
        rng_step, rng = jax.random.split(rng)
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step,
            env_state,
            action,
        )

        # reset to net_init_state and prev_action when done
        next_net_state = jax.tree.map(
            lambda r, s: jnp.where(done, r, s),
            net_init_state,
            next_net_state,
        )
        next_prev_action = jnp.where(
            done, 
            default_prev_action,
            action,
        )
        
        # carry to next step
        carry = (next_env_state, next_obs, next_net_state, next_prev_action)
        # output
        transition = Transition(
            env_state=env_state,
            obs=obs,
            net_state=net_state,
            prev_action=prev_action,
            value=critic_value,
            action=action,
            log_prob=log_prob,
            reward=reward,
            done=done,
            info=info,
        )
        return carry, transition

    final_carry, transitions = jax.lax.scan(
        _env_step,
        initial_carry,
        input_rngs,
    )
    # compute final value
    (fin_env_state, fin_obs, fin_net_state, fin_prev_action) = final_carry
    _fin_pi, fin_value, _fin_next_net_state = net_apply(
        net_params,
        fin_obs,
        fin_net_state,
        fin_prev_action,
    )

    return Rollout(
        transitions=transitions,
        # final_env_state=fin_env_state,
        # final_obs=fin_obs,
        # final_net_state=fin_net_stat,
        # final_prev_action=fin_prev_action,
        final_value=fin_value,
    )


@functools.partial(jax.jit, static_argnames=('net_apply', 'env', 'num_steps'))
def collect_rollouts(
    rng: PRNGKey,
    num_steps: int,
    net_apply: ActorCriticForwardPass,
    net_params: ActorCriticParams,
    net_init_state: ActorCriticState,
    env: Env,
    levels: Level,                      # Level[num_levels]
) -> Rollout:                           # Rollout[num_levels]
    """
    Reset an environment to `levels` and rollout a policy in these levels for
    `env_steps` steps.

    Parameters:

    * rng : PRNGKey
            Random state (consumed)
    * num_steps : static int
            The environments will run forward for this many steps.
    * net_apply : ActorCriticForwardPass
            The apply function should take params and an observation and
            return an action distribution and value prediction.
    * net_params : ActorCriticParams
            Policy parameter.
    * net_init_state : ActorCriticState
            An initial carry for the network.
    * env : static jaxgmg.environments.base.Env
            Provides functions `reset_to_level` and `step`.
    * levels : jaxgmg.environments.base.Level[num_levels]
            Vector of Level structs. This many environments will be run in
            parallel.

    Returns:

    * rollouts : Rollout[num_levels]
            The collected experience, one sequence of trajectories for each
            level.
    """
    vectorised_collect_rollout = jax.vmap(
        collect_rollout,
        in_axes=(0,None,None,None,None,None,0),
        out_axes=0,
    )
    num_levels = jax.tree.leaves(levels)[0].shape[0]
    rng_levels = jax.random.split(rng, num_levels)
    rollouts = vectorised_collect_rollout(
        rng_levels,
        num_steps,
        net_apply,
        net_params,
        net_init_state,
        env,
        levels,
    )
    return rollouts


# # # 
# Analysing rollouts


@jax.jit
def compute_single_rollout_metrics(
    rollout: Rollout,
    discount_rate: float,
    benchmark_return: float | None
) -> dict[str, Any]:
    """
    Parameters:

    * rollout: Rollout (with Transition[num_steps] inside)
            The rollout to score.
    * discount_rate : float
            Used in computing the return metric.
    * benchmark_return : float | None
            What is the benchmark (e.g. optimal) return to be aiming for?
            If None, skip this metric.

    Returns:

    * metrics : {str: Any}
            A dictionary of statistics calculated based on the rollout.
    """
    # compute episode lengths
    eps_per_step = rollout.transitions.done.mean()
    steps_per_ep = 1 / (eps_per_step + (1/256))
    # average return for episodes contained in the rollout
    avg_return = compute_average_return(
        rewards=rollout.transitions.reward,
        dones=rollout.transitions.done,
        discount_rate=discount_rate,
    )
    # compute average reward
    reward_per_step = rollout.transitions.reward.mean()
    metrics = {
        'avg_return': avg_return,
        'avg_episode_length': steps_per_ep,
        'reward_per_step': reward_per_step,
    }

    # compare return to benchmark return if provided
    if benchmark_return is not None:
        metrics.update({
            'benchmark_return': benchmark_return,
            'benchmark_regret': benchmark_return - avg_return,
        })
    
    # if there are any proxy rewards, add new metrics for each
    proxy_dict = rollout.transitions.info.get("proxy_rewards", {})
    for proxy_name, proxy_rewards in proxy_dict.items():
        avg_proxy_return = compute_average_return(
            rewards=proxy_rewards,
            dones=rollout.transitions.done,
            discount_rate=discount_rate,
        )
        proxy_reward_per_step = proxy_rewards.mean()
        metrics["proxy_"+proxy_name] = {
            'avg_return': avg_proxy_return,
            'reward_per_step': proxy_reward_per_step,
        }
    
    return metrics


@jax.jit
def compute_rollout_metrics(
    rollouts: Rollout,                  # Rollout[num_levels]
    discount_rate: float,
    benchmark_returns: Array | None,    # float[num_levels]
) -> dict[str, Any]:
    """
    Parameters:

    * rollouts: Rollout[num_levels] (with Transition[num_steps] inside)
            The rollouts to score.
    * discount_rate : float
            Used in computing the return metric.
    * benchmark_returns : float[num_levels] | None
            For each level, what is the benchmark (e.g. optimal) return to be
            aiming for? If None, skip this metric.

    Returns:

    * metrics : {str: Any}
            A dictionary of statistics calculated based on the collected
            experience.
    """
    # note: comments use shorthand L = num_levels, S = num_steps.

    # compute returns
    vmap_avg_return = jax.vmap(compute_average_return, in_axes=(0,0,None))
    avg_returns = vmap_avg_return(
        rollouts.transitions.reward,                # float[L (vmapped), S]
        rollouts.transitions.done,                  # bool[L (vmapped), S]
        discount_rate,                              # float
    )                                               # -> float[L (vmapped)]

    # compute episode lengths
    eps_per_step = (
        rollouts.transitions.done.mean(axis=1)      # bool[L, S] -> float[L]
    )
    steps_per_ep = 1 / (eps_per_step + (1/256))

    # compute average reward
    reward_per_step = (
        rollouts.transitions.reward.mean(axis=1)    # float[L, S] -> float[L]
    )

    metrics = {
        # average over all levels in the batch
        'avg_avg_return': avg_returns.mean(),
        'avg_avg_episode_length': steps_per_ep.mean(),
        'avg_reward_per_step': reward_per_step.mean(),
        # histogram of values for each level in the batch
        'lvl_avg_return_hist': avg_returns,
        'lvl_avg_episode_length_hist': steps_per_ep,
        'lvl_reward_per_step_hist': reward_per_step,
    }

    # compare returns to benchmark returns if provided
    # TODO: allow a dict of different benchmarks (like proxies)
    if benchmark_returns is not None:
        benchmark_regret = benchmark_returns - avg_returns
        metrics.update({
            # average over all levels in the batch
            'avg_benchmark_return': benchmark_returns.mean(),
            'avg_benchmark_regret': benchmark_regret.mean(),
            # histograms of values for each level
            'lvl_benchmark_return_hist': benchmark_returns,
            'lvl_benchmark_regret_hist': benchmark_regret,
        })
    
    # if there are any proxy rewards, add new metrics for each
    proxy_dict = rollouts.transitions.info.get("proxy_rewards", {})
    for proxy_name, proxy_rewards in proxy_dict.items():
        avg_proxy_returns = vmap_avg_return(
            proxy_rewards,              # float[L (vmapped), S]
            rollouts.transitions.done,  # bool[L (vmapped), S]
            discount_rate,              # float
        )                               # -> float[L (vmapped)]
        proxy_reward_per_step = (
            proxy_rewards.mean(axis=1)  # float[L, S] -> float[L]
        )
        metrics["proxy_"+proxy_name] = {
            # average over all levels in the batch
            'avg_avg_return': avg_proxy_returns.mean(),
            'avg_reward_per_step': proxy_reward_per_step.mean(),
            # histrograms of values for each level
            'lvl_avg_return_hist': avg_proxy_returns,
            'lvl_reward_per_step_hist': proxy_reward_per_step,
        }
    
    return metrics


@jax.jit
def compute_average_return(
    rewards: Array,         # float[num_steps]
    dones: Array,           # bool[num_steps]
    discount_rate: float,
) -> float:
    """
    Given a sequence of (reward, done) pairs, compute the average return for
    each episode represented in the sequence.

    Parameters:

    * rewards : float[num_steps]
            Scalar rewards delivered at the conclusion of each timestep.
    * dones : bool[num_steps]
            True indicates the reward was delivered as the episode
            terminated.
    * discount_rate : float
            The return is exponentially discounted sum of future rewards in
            the episode, this is the discount rate for that discounting.

    Returns:

    * average_return : float
            The average of the returns for each episode represented in the
            sequence of (reward, done) pairs.
    """
    # compute per-step returns
    def _accumulate_return(
        next_step_return,
        this_step_reward_and_done,
    ):
        reward, done = this_step_reward_and_done
        this_step_return = reward + (1-done) * discount_rate * next_step_return
        return this_step_return, this_step_return
    _, per_step_returns = jax.lax.scan(
        _accumulate_return,
        0,
        (rewards, dones),
        reverse=True,
    )

    # identify start of each episode
    first_steps = jnp.roll(dones, 1).at[0].set(True)
    
    # average returns at the start of each episode
    total_first_step_returns = jnp.sum(first_steps * per_step_returns)
    num_episodes = jnp.sum(first_steps)
    average_return = total_first_step_returns / num_episodes
    
    return average_return


@jax.jit
def generalised_advantage_estimation(
    rewards: Array,         # float[num_steps]
    dones: Array,           # bool[num_steps]
    values: Array,          # float[num_steps]
    final_value: float,
    lambda_: float,
    discount_rate: float,
) -> Array:
    """
    Given a sequence of (reward, done, value) triples, estimate generalised
    advantages.

    Inputs:
    
    * rewards : float[num_steps]
            Scalar rewards delivered at the conclusion of each timestep.
    * dones : bool[num_steps]
            True indicates that this timestep marks the end of an episode.
    * values : float[num_steps]
            Scalar value estimate from the critic of the state at this
            timestep (before the transition).
    * final_value : float
            Scalar value estimate from the critic after the final timestep.
    * lambda : float
            Used in definition of GAE.
    * discount_rate : float
            Used in definition of GAE.

    Returns:

    * gaes : float[num_steps]
            The generalised advantage estimates across the trajectory.

    Notes:

    * The 'dones' flags indicate episode termination and also truncation.
      When an episode is terminated it's fine to assume there is no value
      left but when an episode is truncated this may not be correct.
      Nevertheless, with the exception of the final episode represented in
      the trajectory (for which we have `final_value` representing the value
      from after the final step) we ignore this distinction for simplicity
      (this also follows the original PPO implementation and PureJaxRL).
    """
    # reverse scan through num_steps axis
    initial_gae_and_next_value = (0, final_value)
    transitions = (rewards, values, dones)
    def _gae_reverse_step(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        reward, this_value, done = transition
        gae = (
            reward
            - this_value
            + (1-done) * discount_rate * (next_value + lambda_ * gae)
        )
        return (gae, this_value), gae
    _final_carry, gaes = jax.lax.scan(
        _gae_reverse_step,
        initial_gae_and_next_value,
        transitions,
        reverse=True,
        unroll=16, # TODO: seems to have no noticable effect on speed on cpu
    )
    return gaes

    
# # # 
# Animating rollouts


@functools.partial(jax.jit, static_argnames=('grid_width','env'))
def animate_rollouts(
    rollouts: Rollout, # Rollout[num_levels] (with Transition[num_steps])
    grid_width: int,
    env: Env = None,
) -> Array:
    """
    Transform a vector of rollouts into a sequence of images showing for each
    timestep a matrix of rendered images.

    Inputs:

    * rollouts : Rollout[num_levels] (each containing Transition[num_steps])
            The rollouts to visualise.
    * grid_width : static int
            How many levels to put in each row of the grid. Must exactly
            divide num_levels (the shape of rollouts).
    * env : Env
            The environment provides the render method used to render the
            states.

    Returns:

    * frames : float[num_steps+4, img_height, img_width, channels]
            The animation.
 
    Notes:

    * In the output type:
      * `num_steps+4` is for 4 frames inserted at the end of the animation to
        mark the end.
      * img_width = grid_width * cell_width + grid_width + 1
      * img_height = grid_height * cell_height + grid_height + 1
      * grid_height = num_levels / grid_width (divides exactly)
      * cell_width and cell_height are dependent on the size of rendered
        images at the given level of detail.
      * the `+ grid_width + 1` and `+ grid_height + 1` come from 1 pixel of
        padding that is inserted separating each rendered image in the grid.
      * channels is usually 3 (rgb) but could be otherwise, it depends on the
        shape of the renders.
    
    TODO:

    * Currently the final result state from the rollout is not shown. We
      could add that!
      * It would require tweaking the rollout class to store the final obs
        and the final env_state.
      * It would slightly complicate the observation aseembly phase of this
        function, see comments.
      * It would change the first output axis to `num_steps + 1`.
    """
    num_levels = jax.tree.leaves(rollouts)[0].shape[0]
    assert (num_levels % grid_width) == 0
    
    # TODO: first stack the final env state

    # assemble observations at desired level of detail
    vrender = jax.vmap(env.render_state)
    vvrender = jax.vmap(vrender)
    images = vvrender(rollouts.transitions.env_state)

    # flash the screen half black for the last frame of each episode
    done_mask = einops.rearrange(
        rollouts.transitions.done,
        'level step -> level step 1 1 1',
    )
    images = images * (1. - .4 * done_mask)
    
    # rearrange into a (padded) grid of renders
    images = jnp.pad(
        images,
        pad_width=(
            (0, 0), # levels
            (0, 0), # steps
            (0, 1), # height
            (0, 1), # width
            (0, 0), # channel
        ),
    )
    grid = einops.rearrange(
        images,
        '(level_h level_w) step h w c -> step (level_h h) (level_w w) c',
        level_w=grid_width,
    )
    grid = jnp.pad(
        grid,
        pad_width=(
            (0, 4), # time
            (1, 0), # height
            (1, 0), # width
            (0, 0), # channel
        ),
    )

    return grid


