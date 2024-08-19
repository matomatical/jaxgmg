"""
Actor-critic architectures for RL experiments.
"""

from typing import Callable, Literal

import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax

from chex import ArrayTree, PRNGKey
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

    * num_actions : int

    Subclasses must implement `setup` and `__call__` (or just `__call__` with
    `@compact`). The call function should follow this API:
    
    ```
    pi, v, next_state = architecture.apply(
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


    def __call__(
        self,
        obs: Observation,
        state: ActorCriticState,
        prev_action: int,
    ):
        raise NotImplementedError


# # # 
# IMPALA Architecture


class Impala(ActorCriticNetwork):
    """
    IMPALA Architecture based on Espeholt et al., 2018, "IMPALA: Importance
    Weighted Actor-Learner Architectures" (see Figure 3).

    Fields:

    * cnn_size: "mlp", "small" or "large"
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

    Note: There are small differences in the handling of auxiliary inputs.
    
    * We don't take the previous timestep reward as input. The rationale is
      that we want to train in settings where the reward is not always known.
    * We don't have a 'blue ladder' input with its own embedding and LSTM.
    * We support other aux inputs (e.g. orientation for partially observable
      environments) which are fed directly into the LSTM (etc.) input.
      * TODO: Understand what 'blue ladder' is and try more sophisticated
        embeddings.
    """
    cnn_type: Literal["mlp", "small", "large"]
    rnn_type: Literal["ff", "lstm", "gru"]

    @nn.compact
    def __call__(
        self,
        obs: Observation,
        state: ActorCriticState,
        prev_action: int,
    ) -> tuple[
        distrax.Categorical,
        float,
        ActorCriticState,
    ]:
        # embed the image part of the observation
        match self.cnn_type:
            case "mlp":
                obs_embedding = _DenseMLP()(obs.image)
            case "small":
                obs_embedding = _ImpalaSmallCNN()(obs.image)
            case "large":
                obs_embedding = _ImpalaLargeCNN()(obs.image)
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
                rnn_out = nn.relu(nn.Dense(features=256)(rnn_in))
                next_state = state
            case "lstm":
                next_state, rnn_out = nn.OptimizedLSTMCell(
                    features=256,
                )(state, rnn_in)
            case "gru":
                next_state, rnn_out = nn.GRUCell(
                    features=256,
                )(state, rnn_in)
            case _:
                raise ValueError(f"Unknown RNN type {self.rnn_type}")

        # actor head -> action distribution
        logits = nn.Dense(features=self.num_actions)(rnn_out)
        pi = distrax.Categorical(logits=logits)

        # critic head -> value
        v = nn.Dense(features=1)(rnn_out)
        v = jnp.squeeze(v)

        return pi, v, next_state


    def initialize_state(self, rng: PRNGKey) -> ActorCriticState:
        match self.rnn_type:
            case "ff":
                return None
            case "lstm":
                return nn.OptimizedLSTMCell(
                    features=256,
                    parent=None,
                ).initialize_carry(
                    rng=rng,
                    input_shape=(256,)
                )
            case "gru":
                return nn.GRUCell(
                    features=256,
                    parent=None,
                ).initialize_carry(
                    rng=rng,
                    input_shape=(256,)
                )
            case _:
                raise ValueError(f"Unknown RNN type {self.rnn_type}")
        

# # # 
# Helper modules (Observation embedding)


class _ImpalaLargeCNN(nn.Module):
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
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        return x


class _ImpalaSmallCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(8,8), strides=(4,4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4,4), strides=(2,2))(x)
        x = nn.relu(x)
        x = jnp.ravel(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        return x


class _DenseMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        # flatten for MLP
        x = jnp.ravel(x)
        # start the residual stream
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        # add with more residual blocks
        for _embedding_residual_block in range(2):
            y = nn.Dense(features=128)(x)
            y = nn.relu(y)
            x = x + y
        # done
        return x


