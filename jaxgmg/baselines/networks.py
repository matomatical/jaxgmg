"""
Actor-critic architectures for RL experiments.
"""

import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import distrax


# # # 
# Look-up a particular architecture


def get_architecture(
    spec: str,
    num_actions: int,
    **kwargs,
):
    spec = spec.lower().split(":")
    match spec:
        # relu net (optionally with a custom shape)
        case ["relu"]:
            return ReLUFF(num_actions=num_actions)
        case ["relu", layers_by_width]:
            layers, width = layers_by_width.split("x")
            return ReLUFF(
                num_actions=num_actions,
                num_embedding_layers=layers,
                embedding_layer_width=width,
            )
        # impala large (default, lstm, or ff)
        case ["impala"]:
            return ImpalaLarge(num_actions=num_actions) # default LSTM
        case ["impala", "lstm"]:
            return ImpalaLarge(num_actions=num_actions, use_lstm=True)
        case ["impala", "ff"]:
            return ImpalaLarge(num_actions=num_actions, use_lstm=False)
        # impala small (default, lstm, or ff)
        case ["impala", "small"]:
            return ImpalaSmall(num_actions=num_actions) # default LSTM
        case ["impala", "small", "lstm"]:
            return ImpalaSmall(num_actions=num_actions, use_lstm=True)
        case ["impala", "small", "ff"]:
            return ImpalaSmall(num_actions=num_actions, use_lstm=False)
        case _:
            raise ValueError(f"Unknown net architecture spec: {name!r}.")


# # #
# Abstract base class


class ActorCriticNetwork(nn.Module):
    """
    Abstract base class for actor-critic architecture for RL experiments.
    
    Fields:

    * num_actions : int

    Subclasses must implement `setup` and `__call__` (or just `__call__` with
    `@compact`). The call function should follow this API:
    
    ```
    pi, v, carry = architecture.apply(
        {'params': params},
        obs=obs,
        carry=carry,
    )
    ```

    Then this superclass provides an `init_params_and_carry` method that
    follows this API:

    ```
    init_params, init_carry = net.init_params_and_carry(
        rng=rng_init,
        obs_shape=obs.shape,
        obs_dtype=obs.dtype,
    )
    ```
    """
    num_actions: int


    def init_params_and_carry(self, rng, obs_shape, obs_dtype):
        rng_init_carry, rng_init_params = jax.random.split(rng)
        init_carry = self.initialize_carry(rng=rng_init_carry)
        input_shape = jax.ShapeDtypeStruct(shape=obs_shape, dtype=obs_dtype)
        params = self.lazy_init(rngs=rng_init_params, input_shape, init_carry)
        return params, init_carry


    def initialize_carry(self, rng):
        raise NotImplementedError


    def __call__(self, rng):
        raise NotImplementedError


class ImpalaSmall(ActorCriticNetwork):
    """
    Architecture based on Esterholt et al., 2018, "IMPALA: Importance
    Weighted Actor-Learner Architectures". See Figure 3 left (Small
    architecture).

    If 'use_lstm' is false, we replace the LSTM block with 256 output
    features with a dense ReLU layer with 256 output features.
    
    Note: We also don't input the previous action or reward from the
    environment.
    """
    use_lstm: bool


    @nn.compact
    def __call__(self, x, carry):
        # state embedding
        x = nn.Conv(features=16, kernel_size=(8,8), strides=(4,4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4,4), strides=(2,2))(x)
        x = nn.relu(x)
        x = jnp.ravel(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        # TODO: previous reward, previous action embedding?
        
        if use_lstm:
            carry, x = nn.OptimizedLSTMCell(features=256)(carry, x)
        else:
            x = nn.Dense(features=256)(x)
            x = nn.relu(x)

        # actor head
        logits = nn.Dense(self.num_actions)(x)
        pi = distrax.Categorical(logits=logits)

        # critic head
        v = nn.Dense(1)(x)
        v = jnp.squeeze(v)

        return pi, v, carry

    
    def initialize_carry(self, rng):
        if self.use_lstm:
            # TODO: get THE module's LSTM Cell? This spawns a new one
            return nn.OptimizedLSTMCell(features=256).initialize_carry(
                rngs=rng,
                input_shape=(256,)
            )
        else:
            return None


class ImpalaLarge(ActorCriticNetwork):
    """
    Architecture based on Esterholt et al., 2018, "IMPALA: Importance
    Weighted Actor-Learner Architectures". See Figure 3 right (Large
    architecture).

    If 'use_lstm' is false, we replace the LSTM block with 256 output
    features with a dense ReLU layer with 256 output features.
    
    Note: We also don't input the previous action or reward from the
    environment.
    """
    use_lstm: bool


    @nn.compact
    def __call__(self, x, carry):
        # state embedding
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
        x = jnp.ravel(x)
        x = nn.relu(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        
        # TODO: previous reward, previous action embedding?
        
        if use_lstm:
            carry, x = nn.OptimizedLSTMCell(features=256)(carry, x)
        else:
            x = nn.Dense(features=256)(x)
            x = nn.relu(x)

        # actor head
        logits = nn.Dense(self.num_actions)(x)
        pi = distrax.Categorical(logits=logits)

        # critic head
        v = nn.Dense(1)(x)
        v = jnp.squeeze(v)

        return pi, v, carry


    def initialize_carry(self, rng):
        if self.use_lstm:
            # TODO: get THE module's LSTM Cell? This spawns a new one
            return nn.OptimizedLSTMCell(features=256).initialize_carry(
                rngs=rng,
                input_shape=(256,)
            )
        else:
            return None


class ReLUFF(ActorCriticNetwork):
    """
    Simple MLP with ReLU activation. Not recurrent.
    """
    num_embedding_layers: int = 3
    embedding_layer_width: int = 128

    
    @nn.compact
    def __call__(self, x, carry):
        # state embedding
        x = jnp.ravel(x)
        for embedding_residual_block in range(self.num_embedding_layers):
            y = nn.Dense(self.embedding_layer_width)(x)
            y = nn.relu(y)
            x = x + y

        # actor head
        logits = nn.Dense(self.num_actions)(x)
        pi = distrax.Categorical(logits=logits)

        # critic head
        v = nn.Dense(1)(x)
        v = jnp.squeeze(v)

        return pi, v, carry


    def initialize_carry(self, rng):
        return None

