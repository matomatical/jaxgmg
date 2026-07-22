import functools

import jax
import jax.numpy as jnp
from chex import Array

from jaxgmg.baselines import experience
from jaxgmg.baselines.experience import Rollout


# # #
# Compute scores for a batch of levels


@functools.partial(
    jax.jit,
    static_argnames=[
        "scoring_method",
    ],
)
def plr_compute_scores(
    # which scoring method?
    scoring_method: str,
    # data for computing scores
    rollouts: Rollout,                      # Rollout[num_levels] with Transition[num_steps]
    max_ever_returns: Array,                # float[num_levels]
    advantages: Array,                      # float[num_levels, num_steps]
    discount_rate: float,
    # data for computing oracle scores
    oracle_returns: Array,                  # float[num_levels]
) -> Array:                                 # float[num_levels]
    """
    Compute prioritisation 'scores' for a batch of levels using a named
    scoring method.

    Inputs:

    * scoring_method : str (static)
            Which level-scoring method to use. The CLI/config surface calls
            this the "regret estimator" (the paper's framing), but internally
            it is deliberately the more general "scoring method": most methods
            estimate regret, but some do not (e.g. `absgae` is an L1 value loss
            and `dro-actor` a distributionally-robust objective), and we may
            add further non-regret methods. See below for the full list.
    * rollouts : Rollout[num_levels] with Transition[num_steps].
            The experience data from which the scores should be computed.
    * max_ever_returns : float[num_levels]
            Used as an estimate of the optimal return achievable for a level
            for maxMC regret estimators methods.
    * advantages : float[num_levels, num_steps]
            When training with PPO, we probably have already computed the
            GAEs from the rollouts. In order to skip computing them again
            for score methods that use them, provide them here.
    * discount_rate: float
            Used in several regret estimation methods.
    * oracle_returns : float[num_levels]
            The analytically-computed optimal return for each level, used only
            by ORACLE scoring methods (used for evaluating estimators). The
            caller computes these once from a configured level solver (see
            `buffer.oracle_returns`); a placeholder is fine for other methods.

    Returns:

    * scores : float[num_levels]
            One score for each level.
    """
    return jax.vmap(
        plr_compute_score,
        in_axes=(
            None,   # scoring method (static, don't vmap)
            0,      # rollouts
            0,      # max ever returns
            0,      # advantages
            None,   # discount rate (don't vmap)
            0,      # oracle returns
        ),
    )(
        scoring_method,         # str (static)
        rollouts,               # Rollout[vmap(num_levels)] with Transition[num_steps]
        max_ever_returns,       # float[vmap(num_levels)]
        advantages,             # float[vmap(num_levels), num_steps]
        discount_rate,          # float
        oracle_returns,         # float[vmap(num_levels)]
    )


# # # 
# Compute scores for a single level


@functools.partial(
    jax.jit,
    static_argnames=[
        "scoring_method",
    ],
)
def plr_compute_score(
    scoring_method: str,
    rollout: Rollout,               # Rollout with Transition[num_steps]
    max_ever_return: float,
    advantages: Array,              # float[num_steps]
    discount_rate: float,
    oracle_return: float,
) -> float:
    # compute the score on the original reward data
    match scoring_method.lower():
        case "absgae":
            original_score = l1_value_loss(
                advantages=advantages,
            )
        case "pvl":
            original_score = regret_pvl(
                advantages=advantages,
            )
        case "maxmc-paper": 
            original_score = regret_maxmc_paper(
                values=rollout.transitions.value,
                max_ever_return=max_ever_return,
            )
        case "maxmc-initial":
            original_score = regret_maxmc_initial(
                values=rollout.transitions.value,
                max_ever_return=max_ever_return,
            )
        case "maxmc-critic":
            original_score = regret_maxmc_critic(
                values=rollout.transitions.value,
                dones=rollout.transitions.done,
                discount_rate=discount_rate,
                max_ever_return=max_ever_return,
            )
        case "maxmc-critic-balanced":
            original_score = regret_maxmc_critic_balanced(
                values=rollout.transitions.value,
                dones=rollout.transitions.done,
                discount_rate=discount_rate,
                max_ever_return=max_ever_return,
            )
        case "maxmc-actor":
            original_score = regret_maxmc_actor(
                rewards=rollout.transitions.reward,
                dones=rollout.transitions.done,
                discount_rate=discount_rate,
                max_ever_return=max_ever_return,
            )
        case "oracle-actor":
            original_score = regret_oracle_actor(
                oracle_return=oracle_return,
                rewards=rollout.transitions.reward,
                dones=rollout.transitions.done,
                discount_rate=discount_rate,
            )
        case "dro-actor":
            original_score = dro_actor(
                rewards=rollout.transitions.reward,
                dones=rollout.transitions.done,
                discount_rate=discount_rate,
                max_ever_return=max_ever_return,
            )
        case _:
            raise ValueError(f"Unknown scoring method {scoring_method!r}")

    return original_score


# # # 
# Different kinds of score functions


def l1_value_loss(
    advantages: Array,          # float[num_steps]
) -> float:
    """
    Estimate replayability as the average of the magnitude of the GAE
    (equivalent to the L1 value loss where PPO trains with the L2 value
    loss).

    Notes:

    * Not an estimate of regret: see positive value loss.
    """
    return jnp.abs(advantages).mean()


def regret_pvl(
    advantages: Array,          # float[num_steps]
) -> float:
    """
    Estimate regret of a level using the average of the positive GAEs. This
    is a biased estimate of regret that is heavily dependent on the value
    network and current policy.
    """
    return jnp.maximum(advantages, 0).mean()
        

def regret_maxmc_paper(
    values: Array,              # float[num_steps]
    max_ever_return: float,
) -> float:
    """
    Estimate regret for a level from a single episode as
        
        Return_{max ever} - 1/T sum_{t=0}^T Value(s_t).
    
    TODO:

    * This is what they suggest in the robust PLR paper, but I think it
      doesn't make sense because we need to discount the value predictions
      from later in the episodes.
    * I think we don't currently handle multiple episode rollouts correctly,
      we should average within episodes and then average over the results
      (which also means we need to know `dones` sequence).
    """
    return max_ever_return - values.mean()


def regret_maxmc_initial(
    values: Array,              # float[num_steps]
    max_ever_return: float,
) -> float:
    """
    Estimate regret for a level as
        
        Return_{max ever} - Value(s_0).
    """
    return max_ever_return - values[0]


def regret_maxmc_critic(
    values: Array,             # float[num_steps]
    dones: Array,              # bool[num_steps]
    discount_rate: float,      # float
    max_ever_return: float,    # float
) -> float:
    """
    Estimate regret for a level from a single episode as
        
        Return_{max ever} - 1/T sum_{t=0}^T \\gamma^t Value(s_t).
    
    Note:

    * For multi-episode rollouts, this takes the average over steps rather
      than over epsiodes, which biases the value-function-based return
      estimate towards the average value prediction of longer episodes within
      the rollout. See `regret_maxmc_critic_balanced` for a corrected version.
    """
    def _step(t, value_and_done):
        value, done = value_and_done
        discounted_value = discount_rate**t * value
        t_next = (1-done) * (t + 1)
        return t_next, discounted_value
    _, discounted_values = jax.lax.scan(
        _step,
        0,
        (values, dones),
    )
    return max_ever_return - discounted_values.mean()


def regret_maxmc_critic_balanced(
    values: Array,             # float[num_steps]
    dones: Array,              # bool[num_steps]
    discount_rate: float,      # float
    max_ever_return: float,    # float
) -> float:
    """
    Estimate regret for a level from a single episode as
        
        Return_{max ever} - 1/T sum_{t=0}^T \\gamma^t Value(s_t).

    Estimate the regret for a multi-episode rollout as the average of the
    per-episode averages.
    """
    # compute discounted values
    def _forward_step(t_and_discount, value_and_done):
        t, discount = t_and_discount
        value, done = value_and_done
        # compute discounted value
        discounted_value = discount * value
        # update carry (reset on done)
        next_discount = jnp.where(done, 1, discount * discount_rate)
        next_t = jnp.where(done, 0, t + 1)
        return (next_t, next_discount), (t + 1, discounted_value)
    _, (ts, discounted_values) = jax.lax.scan(
        _forward_step,
        (0, 1.0),
        (values, dones),
    )
    # compute episode lengths
    def _backward_step(next_step_episode_length, t_and_done):
        t, done = t_and_done
        this_step_episode_length = jnp.where(
            done,
            t,
            next_step_episode_length,
        )
        return this_step_episode_length, this_step_episode_length
    _, Ts = jax.lax.scan(
        _backward_step,
        0,
        (ts, dones),
        reverse=True,
    )
    # now discounted_values are the values at each step and Ts are their
    # respective episode lengths: normalise and compute the average
    mean_discounted_values = (discounted_values / Ts).sum() / dones.sum()
    # turn this into regret
    return max_ever_return - mean_discounted_values


def regret_maxmc_actor(
    rewards: Array,             # float[num_steps]
    dones: Array,               # bool[num_steps]
    discount_rate: float,
    max_ever_return: float,
) -> float:
    """
    Estimate regret for a level from a single episode as

        Return_{max ever} - sum_{t=0}^T \\gamma^t r_t.

    That is, the empirical average return achieved by the policy is used for
    the return of the policy, and the empirical max ever return on this level
    is used for the return of the optimal policy.
    """
    average_return = experience.compute_average_return(
        rewards=rewards,
        dones=dones,
        discount_rate=discount_rate,
    )
    return max_ever_return - average_return


@jax.jit
def regret_oracle_actor(
    oracle_return: float,
    rewards: Array,             # float[num_steps]
    dones: Array,               # float[num_steps]
    discount_rate: float,
) -> float:
    """
    Estimate regret for a level by combining an empirical average of the
    current policy's return with an analytically-computed known optimal
    return.

        Return_{max oracle} - sum_{t=0}^T \\gamma^t r_t.

    The optimal return `oracle_return` is computed by the caller from a
    configured level solver (see `buffer.oracle_returns`), not here. The solver
    is responsible for the oracle's validity constraints (cardinal actions, an
    episode long enough for the optimal path); turn-action environments like
    minigrid_maze are not dispatched to oracle scoring at all.

    Possible future improvement: the oracle return is a property of the level,
    so it need only be computed once when the level enters the buffer (like the
    max-ever return), rather than recomputed on every replay/score pass.
    """
    # no oracle for average return, just use the provided experience
    average_return = experience.compute_average_return(
        rewards=rewards,
        dones=dones,
        discount_rate=discount_rate,
    )

    return oracle_return - average_return


def dro_actor(
    rewards: Array,             # float[num_steps]
    dones: Array,               # bool[num_steps]
    discount_rate: float,
    max_ever_return: float,
) -> float:
    """
    Estimate maximin score for a level from a single episode as

         - sum_{t=0}^T \\gamma^t r_t.

    That is, the empirical average return achieved by the policy is used for
    the return of the policy.
    """
    average_return = experience.compute_average_return(
        rewards=rewards,
        dones=dones,
        discount_rate=discount_rate,
    )
    return  -average_return

