"""
Actor-critic architectures for RL experiments.
"""

import jax
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
            return ReLUFF(num_actions=num_actions) # default layers x width
        case ["relu", layers_by_width]:
            layers, width = layers_by_width.split("x")
            return ReLUFF(
                num_actions=num_actions,
                num_embedding_layers=int(layers),
                embedding_layer_width=int(width),
            )
        # impala large (ff or lstm)
        case ["impala", "ff"]:
            return ImpalaLargeFF(num_actions=num_actions)
        case ["impala", "lstm"]:
            return ImpalaLarge(num_actions=num_actions)
        # impala small (default, lstm, or ff)
        case ["impala-small", "ff"]:
            return ImpalaSmallFF(num_actions=num_actions)
        case ["impala-small", "lstm"]:
            return ImpalaSmall(num_actions=num_actions)
        case ["impala"] | ["impala-small"]:
            raise ValueError(f"Please specify ':ff' or ':lstm'.")
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
    pi, v, next_state = architecture.apply(
        {'params': params},
        obs=obs,
        state=state,
    )
    ```

    Then this superclass provides an `init_params_and_state` method that
    follows this API:

    ```
    init_params, init_state = net.init_params_and_state(
        rng=rng_init,
        obs_shape=obs.shape,
        obs_dtype=obs.dtype,
    )
    ```
    """
    num_actions: int


    def init_params_and_state(self, rng, obs_type):
        rng_init_state, rng_init_params = jax.random.split(rng)
        init_state = self.initialize_state(rng=rng_init_state)
        params = self.lazy_init(
            rngs=rng_init_params,
            x=obs_type,
            state=init_state,
        )
        return params, init_state


    def initialize_state(self, rng):
        raise NotImplementedError


    def __call__(self, rng, x, state):
        raise NotImplementedError


# # # 
# Impala feed-forward architectures


class ImpalaLargeFF(ActorCriticNetwork):
    """
    Architecture based on Esterholt et al., 2018, "IMPALA: Importance
    Weighted Actor-Learner Architectures". See Figure 3 right (Large
    architecture).

    We replace the LSTM block with 256 output features with a dense ReLU
    layer with 256 output features.
    
    Note: We also don't input the previous action or reward from the
    environment.
    """
    _lstm_block: nn.OptimizedLSTMCell = nn.OptimizedLSTMCell(features=256)


    @nn.compact
    def __call__(self, x, state):
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
        
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        # actor head
        logits = nn.Dense(self.num_actions)(x)
        pi = distrax.Categorical(logits=logits)

        # critic head
        v = nn.Dense(1)(x)
        v = jnp.squeeze(v)

        return pi, v, state


    def initialize_state(self, rng):
        return None


class ImpalaSmallFF(ActorCriticNetwork):
    """
    Architecture based on Esterholt et al., 2018, "IMPALA: Importance
    Weighted Actor-Learner Architectures". See Figure 3 left (Small
    architecture).

    We replace the LSTM block with 256 output features with a dense ReLU
    layer with 256 output features.
    
    Note: We also don't input the previous action or reward from the
    environment.
    """


    @nn.compact
    def __call__(self, x, state):
        # state embedding
        x = nn.Conv(features=16, kernel_size=(8,8), strides=(4,4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4,4), strides=(2,2))(x)
        x = nn.relu(x)
        x = jnp.ravel(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        # TODO: previous reward, previous action embedding?
        
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        # actor head
        logits = nn.Dense(self.num_actions)(x)
        pi = distrax.Categorical(logits=logits)

        # critic head
        v = nn.Dense(1)(x)
        v = jnp.squeeze(v)

        return pi, v, state

    
    def initialize_state(self, rng):
        return None


# # # 
# Impala feed-forward architectures


class ImpalaLarge(ActorCriticNetwork):
    """
    Architecture based on Esterholt et al., 2018, "IMPALA: Importance
    Weighted Actor-Learner Architectures". See Figure 3 right (Large
    architecture).

    Note: We don't input the previous action or reward from the environment.
    """
    _lstm_block: nn.OptimizedLSTMCell = nn.OptimizedLSTMCell(features=256)


    @nn.compact
    def __call__(self, x, state):
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
        
        state, x = self._lstm_block(state, x)

        # actor head
        logits = nn.Dense(self.num_actions)(x)
        pi = distrax.Categorical(logits=logits)

        # critic head
        v = nn.Dense(1)(x)
        v = jnp.squeeze(v)

        return pi, v, state


    def initialize_state(self, rng):
        return self._lstm_block.initialize_carry(
            rng=rng,
            input_shape=(256,)
        )


class ImpalaSmall(ActorCriticNetwork):
    """
    Architecture based on Esterholt et al., 2018, "IMPALA: Importance
    Weighted Actor-Learner Architectures". See Figure 3 left (Small
    architecture).

    Note: We don't input the previous action or reward from the environment.
    """
    _lstm_block: nn.OptimizedLSTMCell = nn.OptimizedLSTMCell(features=256)


    @nn.compact
    def __call__(self, x, state):
        # state embedding
        x = nn.Conv(features=16, kernel_size=(8,8), strides=(4,4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4,4), strides=(2,2))(x)
        x = nn.relu(x)
        x = jnp.ravel(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        # TODO: previous reward, previous action embedding?
        
        state, x = self._lstm_block(state, x)

        # actor head
        logits = nn.Dense(self.num_actions)(x)
        pi = distrax.Categorical(logits=logits)

        # critic head
        v = nn.Dense(1)(x)
        v = jnp.squeeze(v)

        return pi, v, state

    
    def initialize_state(self, rng):
        return self._lstm_block.initialize_carry(
            rng=rng,
            input_shape=(256,)
        )


# # # 
# MLP architectures


class ReLUFF(ActorCriticNetwork):
    """
    Simple MLP with ReLU activation. Not recurrent.
    """
    num_embedding_layers: int = 3
    embedding_layer_width: int = 128

    
    @nn.compact
    def __call__(self, x, state):
        # state embedding
        x = jnp.ravel(x)
        # at least one layer (to start the residual stream)
        x = nn.Dense(self.embedding_layer_width)(x)
        x = nn.relu(x)
        # remaining residual layers (adding to residual stream)
        for embedding_residual_block in range(self.num_embedding_layers-1):
            y = nn.Dense(self.embedding_layer_width)(x)
            y = nn.relu(y)
            x = x + y

        # actor head
        logits = nn.Dense(self.num_actions)(x)
        pi = distrax.Categorical(logits=logits)

        # critic head
        v = nn.Dense(1)(x)
        v = jnp.squeeze(v)

        return pi, v, state


    def initialize_state(self, rng):
        return None

