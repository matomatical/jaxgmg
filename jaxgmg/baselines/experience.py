"""
Types and utilities for collecting experience in an environment.
"""

import functools

import einops
import jax
import jax.numpy as jnp

# typing
from chex import Array, PRNGKey
from flax import struct
from flax.training.train_state import TrainState # TODO: remove
from jaxgmg.environments.base import EnvState, Env, Level, Observation
from jaxgmg.baselines.networks import ActorCriticState


# # # 
# Represent transitions and rollouts


@struct.dataclass
class Transition:
    """
    Captures data involved in one environment step, from the observation to
    the actor/critic response to the reward, termination, and info response
    from the environment. Note: the next env_state and observation etc. is
    represented in the next transition (or the the final transition in the
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
# Vectorised experience collection


@functools.partial(jax.jit, static_argnames=('env', 'num_steps'))
def collect_rollouts(
    rng: PRNGKey,
    num_steps: int,
    train_state: TrainState,            # TODO: change to a function
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
    * train_state : TrainState
            A flax trainstate object, including the policy parameter
            (`.params`) and application function (`.apply_fn`).
            The policy apply function should take params and an observation
            and return an action distribution and value prediction.
    * net_init_state : ActorCriticState
            An initial carry for the network.
    * env : static jaxgmg.environments.base.Env
            Provides functions `reset` and `step` (actually, vectorised
            versions `vreset` and `vstep`).
    * levels : jaxgmg.environments.base.Level[num_levels]
            Vector of Level structs. This many environments will be run in
            parallel.

    Returns:

    * rollouts : Rollout[num_levels]
            The collected experience, one sequence of trajectories for each
            level.
    """
    # reset environments to the given levels
    env_obs, env_state = env.vreset_to_level(levels=levels)
    num_levels = jax.tree.leaves(levels)[0].shape[0]
    # reset the net inputs to blank
    vec_net_init_state = jax.tree.map(
        lambda c: einops.repeat(
            c,
            '... -> num_levels ...',
            num_levels=num_levels,
        ),
        net_init_state,
    )
    vec_default_prev_action = -jnp.ones(num_levels, dtype=int)

    # scan the steps of the rollout
    initial_carry = (
        env_state,
        env_obs,
        vec_net_init_state,
        vec_default_prev_action,
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
        ) = train_state.apply_fn(
            train_state.params,
            obs,
            net_state,
            prev_action,
        )
        action = action_distribution.sample(seed=rng_action)
        log_prob = action_distribution.log_prob(action)

        # step env
        rng_step, rng = jax.random.split(rng)
        next_obs, next_env_state, reward, done, info = env.vstep(
            rng_step,
            env_state,
            action,
        )

        # reset to net_init_state and prev_action in the parallel envs that are done
        next_net_state = jax.tree.map(
            lambda r, s: jnp.where(
                # reverse broadcast to shape of r (= shape of s)
                done.reshape(-1, *([1] * (len(r.shape)-1))),
                r,
                s,
            ),
            vec_net_init_state,
            next_net_state,
        )
        next_prev_action = jnp.where(
            done, 
            vec_default_prev_action,
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

    final_carry, stepwise_transitions = jax.lax.scan(
        _env_step,
        initial_carry,
        input_rngs,
    )
    # reshape Transn[num_steps, num_levels] -> Transn[num_levels, num_steps]
    transitions = jax.tree.map(
        lambda x: einops.rearrange(x, 'steps levels ... -> levels steps ...'),
        stepwise_transitions,
    )
    # compute final value
    (fin_env_state, fin_obs, fin_net_state, fin_prev_action) = final_carry
    _fin_pi, fin_value, _fin_next_net_state = train_state.apply_fn(
        train_state.params,
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


