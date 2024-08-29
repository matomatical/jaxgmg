"""
Actor-critic architectures for RL experiments.
"""

import functools
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax

from chex import Array, ArrayTree, PRNGKey
from jaxgmg.environments.base import Observation


# # # 
# Actor critic API


# Types

ActorCriticState = ArrayTree

ActorCriticParams = ArrayTree

ActorCriticForwardPass = Callable[
    [ActorCriticParams, Observation, ActorCriticState, int],
    tuple[distrax.Categorical, float, ActorCriticState],
]


class ActorCriticNetwork(nn.Module):
    """
    Abstract base class for actor-critic architecture for RL experiments.
    
    Fields:

    * num_actions : int (> 0)
            The actor returns a distribution over this many actions.
    * num_values: int (> 0)
            The critic returns a vector of values with this many values.
            (Usually this would only be one value, but this can be used to
            also approximate proxy goals for example.)

    Subclasses must implement `setup` and `__call__` (or just `__call__` with
    `@compact`). The call function should follow this API:
    
    ```
    pi, vs, next_state = architecture.apply(
        params=params,
        obs=obs,
        state=state,
        prev_action=prev_action, # or -1 for first step in episode
    )
    ```

    Then this superclass provides an `init_params_and_state` method that
    follows this API:

    ```
    init_params, init_state = net.init_params_and_state(
        rng=rng_init,
        obs_type=env.obs_type(level=example_level),
    )
    ```
    """
    num_actions: int
    num_values: int


    @property
    def is_recurrent(self) -> bool:
        raise NotImplementedError


    def init_params_and_state(self, rng, obs_type):
        rng_init_state, rng_init_params = jax.random.split(rng)
        init_state = self.initialize_state(rng=rng_init_state)
        params = self.lazy_init(
            rngs=rng_init_params,
            obs=obs_type,
            state=init_state,
            prev_action=-1,
        )
        return params, init_state


    def initialize_state(self, rng):
        raise NotImplementedError


    @functools.partial(jax.jit, static_argnames=['self'])
    def __call__(
        self,
        obs: Observation,
        state: ActorCriticState,
        prev_action: int,
    ) -> tuple[
        distrax.Categorical,    # distribution over range(num_actions)
        Array,                  # float[num_values]
        ActorCriticState,
    ]:
        raise NotImplementedError


# # # 
# Pure function that works with an actor critic network


def evaluate_sequence_recurrent(
    net_params: ActorCriticParams,
    net_apply: ActorCriticForwardPass,
    net_init_state: ActorCriticState,
    obs_sequence: Observation,  # Observation[num_steps]
    done_sequence: Array,       # bool[num_steps]
    action_sequence: Array,     # int[num_steps]
) -> tuple[
    Array, # distrax.Categorical[num_steps] (action_distributions)
    Array, # float[num_steps, num_values] (values)
]:
    # scan through the trajectory
    default_prev_action = -1
    initial_carry = (
        net_init_state,
        default_prev_action,
    )
    transitions = (
        obs_sequence,
        done_sequence,
        action_sequence,
    )
    def _net_step(carry, transition):
        net_state, prev_action = carry
        obs, done, chosen_action = transition
        # apply network
        action_distribution, critic_values, next_net_state = net_apply(
            net_params,
            obs,
            net_state,
            prev_action,
        )
        # reset to net_init_state and default_prev_action when done
        next_net_state, next_prev_action = jax.tree.map(
            lambda r, s: jnp.where(done, r, s),
            (net_init_state, default_prev_action),
            (next_net_state, chosen_action),
        )
        carry = (next_net_state, next_prev_action)
        output = (action_distribution, critic_values)
        return carry, output
    _final_carry, (action_distributions, critic_valuess) = jax.lax.scan(
        _net_step,
        initial_carry,
        transitions,
    )
    return action_distributions, critic_valuess


def evaluate_sequence_parallel(
    net_params: ActorCriticParams,
    net_apply: ActorCriticForwardPass,
    obs_sequence: Observation,              # Observation[num_steps]
    net_state_sequence: ActorCriticState,   # ActorCriticState[num_steps]
    prev_action_sequence: Array,            # int[num_steps]
) -> tuple[
    Array, # distrax.Categorical[num_steps] (action_distributions)
    Array, # float[num_steps, num_values] (values)
]:
    action_distributions, critic_valuess, _net_states = jax.vmap(
        net_apply,
        in_axes=(None, 0, 0, 0),
    )(
        net_params,
        obs_sequence,
        net_state_sequence,
        prev_action_sequence,
    )
    return action_distributions, critic_valuess


# # # 
# IMPALA Architecture


class Impala(ActorCriticNetwork):
    """
    IMPALA Architecture based on Espeholt et al., 2018, "IMPALA: Importance
    Weighted Actor-Learner Architectures" (see Figure 3).

    Fields:

    * num_actions : int (> 0)
            The actor returns a distribution over this many actions.
    * num_values: int (> 0)
            The critic returns a vector of values with this many values.
            (Usually this would only be one value, but this can be used to
            also approximate proxy goals for example.)
    * cnn_type: "mlp", "small" or "large"
            The size of the CNN for embedding observation images.
            * "mlp", a small fully-connected residual ReLU network.
            * "small", the CNN (from figure 3-right in the paper).
            * "large", the CNN (from figure 3-left in the paper).
    * rnn_type: "ff", "lstm", or "gru"
            The type of RNN block to use for processing the embeddings.
            * "ff": feed-forward block (does not use recurrence).
            * "lstm": LSTM, as in Espeholt et al. (2018).
            * "gru": Gated Recurrent Unit, which is supposed to work about
              the same but with fewer parameters.
    * width : int
            The width used for various components (RNN layer, dense layers in
            MLP and CNNs). Note IMPALA uses 256.

    Note: There are small differences in the handling of auxiliary inputs.
    
    * We don't take the previous timestep reward as input. The rationale is
      that we want to train in settings where the reward is not always known.
    * We don't have a 'blue ladder' input with its own embedding and LSTM.
    * We support other aux inputs (e.g. orientation for partially observable
      environments) which are fed directly into the LSTM (etc.) input.
      * TODO: Understand what 'blue ladder' is and try more sophisticated
        embeddings.

    Also, the IMPALA paper uses only one value head.
    """
    cnn_type: Literal["mlp", "small", "large"]
    rnn_type: Literal["ff", "lstm", "gru"]
    width: int

    @nn.compact
    def __call__(
        self,
        obs: Observation,
        state: ActorCriticState,
        prev_action: int,
    ) -> tuple[
        distrax.Categorical,
        Array,                  # float[num_values]
        ActorCriticState,
    ]:
        # embed the image part of the observation
        match self.cnn_type:
            case "mlp":
                obs_embedding = _DenseMLP(width=self.width)(obs.image)
            case "small":
                obs_embedding = _ImpalaSmallCNN(width=self.width)(obs.image)
            case "large":
                obs_embedding = _ImpalaLargeCNN(width=self.width)(obs.image)
            case _:
                raise ValueError(f"Unknown CNN type {self.cnn_type}")
        # combine with everything other than the image
        other_embeddings = [
            # (hack: assume it's already encoded)
            getattr(obs, fieldname).flatten()
            for fieldname in obs.__dataclass_fields__
            if fieldname != 'image'
        ]
        # one-hot embed the previous action
        prev_action_embedding = jax.nn.one_hot(
            x=prev_action,
            num_classes=self.num_actions,
        )
        # stack all this into a single combined embedding vector
        combined_embedding = jnp.concatenate([
            obs_embedding,
            prev_action_embedding,
            *other_embeddings,
        ])

        # recurrent block (or not)
        rnn_in = combined_embedding
        match self.rnn_type:
            case "ff":
                rnn_out = nn.relu(nn.Dense(features=self.width)(rnn_in))
                next_state = state
            case "lstm":
                next_state, rnn_out = nn.OptimizedLSTMCell(
                    features=self.width,
                )(state, rnn_in)
            case "gru":
                next_state, rnn_out = nn.GRUCell(
                    features=self.width,
                )(state, rnn_in)
            case _:
                raise ValueError(f"Unknown RNN type {self.rnn_type}")

        # actor head -> action distribution
        logits = nn.Dense(features=self.num_actions)(rnn_out)
        pi = distrax.Categorical(logits=logits)

        # critic head -> values vector
        vs = nn.Dense(features=self.num_values)(rnn_out)

        return pi, vs, next_state

    
    @property
    def is_recurrent(self) -> bool:
        match self.rnn_type:
            case "ff":
                return False
            case "lstm":
                return True
            case "gru":
                return True
            case _:
                raise ValueError(f"Unknown RNN type {self.rnn_type}")


    @functools.partial(jax.jit, static_argnames=['self'])
    def initialize_state(self, rng: PRNGKey) -> ActorCriticState:
        match self.rnn_type:
            case "ff":
                return None
            case "lstm":
                return nn.OptimizedLSTMCell(
                    features=self.width,
                    parent=None,
                ).initialize_carry(
                    rng=rng,
                    input_shape=(self.width,)
                )
            case "gru":
                return nn.GRUCell(
                    features=self.width,
                    parent=None,
                ).initialize_carry(
                    rng=rng,
                    input_shape=(self.width,)
                )
            case _:
                raise ValueError(f"Unknown RNN type {self.rnn_type}")
        

# # # 
# Helper modules (Observation embedding)


class _ImpalaLargeCNN(nn.Module):
    width: int

    @nn.compact
    def __call__(self, x):
        for ch in (16, 32, 32):
            x = nn.Conv(features=ch, kernel_size=(3,3), strides=(1,1))(x)
            x = nn.max_pool(
                x,
                window_shape=(3,3),
                strides=(2,2),
                padding='SAME',
            )
            for residual_block in (1,2):
                y = nn.relu(x)
                y = nn.Conv(features=ch, kernel_size=(3,3), strides=(1,1))(y)
                y = nn.relu(y)
                y = nn.Conv(features=ch, kernel_size=(3,3), strides=(1,1))(y)
                x = x + y
        x = nn.relu(x)
        x = jnp.ravel(x)
        x = nn.Dense(features=self.width)(x)
        x = nn.relu(x)
        return x


class _ImpalaSmallCNN(nn.Module):
    width: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(8,8), strides=(4,4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4,4), strides=(2,2))(x)
        x = nn.relu(x)
        x = jnp.ravel(x)
        x = nn.Dense(features=self.width)(x)
        x = nn.relu(x)
        return x


class _DenseMLP(nn.Module):
    width: int

    @nn.compact
    def __call__(self, x):
        # flatten for MLP
        x = jnp.ravel(x)
        # start the residual stream
        x = nn.Dense(features=self.width)(x)
        x = nn.relu(x)
        # add with more residual blocks
        for _embedding_residual_block in range(2):
            y = nn.Dense(features=self.width)(x)
            y = nn.relu(y)
            x = x + y
        # done
        return x


