"""
Actor-critic architectures for RL experiments.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax

from chex import ArrayTree
from jaxgmg.environments.base import Observation


# # # 
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
# Impala feed-forward architectures


class ImpalaLargeFF(ActorCriticNetwork):
    """
    Architecture based on Esterholt et al., 2018, "IMPALA: Importance
    Weighted Actor-Learner Architectures". See Figure 3 right (Large
    architecture).

    We replace the LSTM block with 256 output features with a dense ReLU
    layer with 256 output features.
    
    Note: We also don't input the reward from the environment.
    """


    @nn.compact
    def __call__(self, obs, state, prev_action):
        # obs embedding
        x = obs.image
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
        obs_embedding = x

        # optional further embeddings from obs
        other_embeddings = [
            getattr(obs, fieldname) # hack: assume it's a 1d array...?
            for fieldname in obs.__dataclass_fields__
            if fieldname != 'image'
        ]

        # previous action embedding
        prev_action_embedding = jax.nn.one_hot(
            x=prev_action,
            num_classes=self.num_actions,
        )

        # combined embedding
        embedding = jnp.concatenate([
            obs_embedding,
            prev_action_embedding,
            *other_embeddings,
        ])
        
        # dense block in lieu of lstm
        x = nn.Dense(features=256)(embedding)
        lstm_out = nn.relu(x)

        # actor head
        logits = nn.Dense(self.num_actions)(lstm_out)
        pi = distrax.Categorical(logits=logits)

        # critic head
        v = nn.Dense(1)(lstm_out)
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
    
    Note: We also don't input the reward from the environment.
    """


    @nn.compact
    def __call__(self, obs, state, prev_action):
        # obs embedding
        x = obs.image
        x = nn.Conv(features=16, kernel_size=(8,8), strides=(4,4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4,4), strides=(2,2))(x)
        x = nn.relu(x)
        x = jnp.ravel(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        obs_embedding = x
        
        # optional further embeddings from obs
        other_embeddings = [
            getattr(obs, fieldname) # hack: assume it's a 1d array...?
            for fieldname in obs.__dataclass_fields__
            if fieldname != 'image'
        ]
        
        # optional further embeddings from obs
        other_embeddings = [
            getattr(obs, fieldname) # hack: assume it's a 1d array...?
            for fieldname in obs.__dataclass_fields__
            if fieldname != 'image'
        ]

        # previous action embedding
        prev_action_embedding = jax.nn.one_hot(
            x=prev_action,
            num_classes=self.num_actions,
        )

        # combined embedding
        embedding = jnp.concatenate([
            obs_embedding,
            prev_action_embedding,
            *other_embeddings,
        ])
        
        # dense block in lieu of lstm
        x = nn.Dense(features=256)(embedding)
        lstm_out = nn.relu(x)

        # actor head
        logits = nn.Dense(self.num_actions)(lstm_out)
        pi = distrax.Categorical(logits=logits)

        # critic head
        v = nn.Dense(1)(lstm_out)
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

    Note: We don't input the reward from the environment.
    """


    @nn.compact
    def __call__(self, obs, state, prev_action):
        # obs embedding
        x = obs.image
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
        obs_embedding = x
        
        # optional further embeddings from obs
        other_embeddings = [
            getattr(obs, fieldname) # hack: assume it's a 1d array...?
            for fieldname in obs.__dataclass_fields__
            if fieldname != 'image'
        ]
        
        # previous action embedding
        prev_action_embedding = jax.nn.one_hot(
            x=prev_action,
            num_classes=self.num_actions,
        )

        # combined embedding
        embedding = jnp.concatenate([
            obs_embedding,
            prev_action_embedding,
            *other_embeddings,
        ])

        # lstm block
        state, lstm_out = nn.OptimizedLSTMCell(features=256)(state, embedding)

        # actor head
        logits = nn.Dense(self.num_actions)(lstm_out)
        pi = distrax.Categorical(logits=logits)

        # critic head
        v = nn.Dense(1)(lstm_out)
        v = jnp.squeeze(v)

        return pi, v, state


    def initialize_state(self, rng):
        return nn.OptimizedLSTMCell(
            features=526,
            parent=None,
        ).initialize_carry(
            rng=rng,
            input_shape=(256,)
        )


class ImpalaSmall(ActorCriticNetwork):
    """
    Architecture based on Esterholt et al., 2018, "IMPALA: Importance
    Weighted Actor-Learner Architectures". See Figure 3 left (Small
    architecture).

    Note: We don't input the reward from the environment.
    """


    @nn.compact
    def __call__(self, obs, state, prev_action):
        # obs embedding
        x = obs.image
        x = nn.Conv(features=16, kernel_size=(8,8), strides=(4,4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4,4), strides=(2,2))(x)
        x = nn.relu(x)
        x = jnp.ravel(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        obs_embedding = x
        
        # optional further embeddings from obs
        other_embeddings = [
            getattr(obs, fieldname) # hack: assume it's a 1d array...?
            for fieldname in obs.__dataclass_fields__
            if fieldname != 'image'
        ]

        # previous action embedding
        prev_action_embedding = jax.nn.one_hot(
            x=prev_action,
            num_classes=self.num_actions,
        )

        # combined embedding
        embedding = jnp.concatenate([
            obs_embedding,
            prev_action_embedding,
            *other_embeddings,
        ])

        # lstm block
        state, lstm_out = nn.OptimizedLSTMCell(features=256)(state, embedding)

        # actor head
        logits = nn.Dense(self.num_actions)(lstm_out)
        pi = distrax.Categorical(logits=logits)

        # critic head
        v = nn.Dense(1)(lstm_out)
        v = jnp.squeeze(v)

        return pi, v, state

    
    def initialize_state(self, rng):
        return nn.OptimizedLSTMCell(
            features=256,
            parent=None,
        ).initialize_carry(
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
    def __call__(self, obs, state, prev_action):
        # obs embedding
        x = jnp.ravel(obs.image)
        # at least one layer (to start the residual stream)
        x = nn.Dense(self.embedding_layer_width)(x)
        x = nn.relu(x)
        # remaining residual layers (adding to residual stream)
        for embedding_residual_block in range(self.num_embedding_layers-1):
            y = nn.Dense(self.embedding_layer_width)(x)
            y = nn.relu(y)
            x = x + y
        obs_embedding = x
        
        # optional further embeddings from obs
        other_embeddings = [
            getattr(obs, fieldname) # hack: assume it's a 1d array...?
            for fieldname in obs.__dataclass_fields__
            if fieldname != 'image'
        ]

        # previous action embedding
        prev_action_embedding = jax.nn.one_hot(
            x=prev_action,
            num_classes=self.num_actions,
        )

        # combined embedding
        embedding = jnp.concatenate([
            obs_embedding,
            prev_action_embedding,
            *other_embeddings,
        ])

        # actor head
        logits = nn.Dense(self.num_actions)(embedding)
        pi = distrax.Categorical(logits=logits)

        # critic head
        v = nn.Dense(1)(embedding)
        v = jnp.squeeze(v)
        
        return pi, v, state


    def initialize_state(self, rng):
        return None


class ReLU(ActorCriticNetwork):
    """
    Simple MLP with ReLU activation and an LSTM.
    """
    num_embedding_layers: int = 3
    embedding_layer_width: int = 128


    @nn.compact
    def __call__(self, obs, state, prev_action):
        # obs embedding
        x = jnp.ravel(obs.image)
        # at least one layer (to start the residual stream)
        x = nn.Dense(features=self.embedding_layer_width)(x)
        x = nn.relu(x)
        # remaining residual layers (adding to residual stream)
        for _residual_embedding_layer in range(self.num_embedding_layers-1):
            y = nn.Dense(features=self.embedding_layer_width)(x)
            y = nn.relu(y)
            x = x + y
        obs_embedding = x
        
        # optional further embeddings from obs
        other_embeddings = [
            getattr(obs, fieldname) # hack: assume it's a 1d array...?
            for fieldname in obs.__dataclass_fields__
            if fieldname != 'image'
        ]

        # previous action embedding
        prev_action_embedding = jax.nn.one_hot(
            x=prev_action,
            num_classes=self.num_actions,
        )

        # combined embedding
        embedding = jnp.concatenate([
            obs_embedding,
            prev_action_embedding,
            *other_embeddings,
        ])

        # lstm block
        lstm_in = embedding
        state, lstm_out = nn.OptimizedLSTMCell(
            features=self.embedding_layer_width,
        )(state, lstm_in)

        # actor head
        logits = nn.Dense(features=self.num_actions)(lstm_out)
        pi = distrax.Categorical(logits=logits)

        # critic head
        v = nn.Dense(features=1)(lstm_out)
        v = jnp.squeeze(v)

        return pi, v, state


    def initialize_state(self, rng):
        return nn.OptimizedLSTMCell(
            features=self.embedding_layer_width,
            parent=None,
        ).initialize_carry(
            rng=rng,
            input_shape=(self.embedding_layer_width,)
        )


# # # 
# Look-up a particular architecture


def get_architecture(spec: str, num_actions: int) -> ActorCriticNetwork:
    """
    Parse a network specification string and instantiate the appropriate kind
    of network.
    """
    spec = spec.lower().split(":")
    match spec:
        # relu net (optionally with a custom shape)
        case ["relu"]:
            return ReLUFF(num_actions=num_actions) # default layers x width
        case ["relu", "lstm"]:
            return ReLU(num_actions=num_actions)
        case ["relu", layers_by_width]:
            layers, width = layers_by_width.split("x")
            return ReLUFF(
                num_actions=num_actions,
                num_embedding_layers=int(layers),
                embedding_layer_width=int(width),
            )
        # impala large (ff or lstm)
        case ["impala"] | ["impala", "ff"]:
            return ImpalaLargeFF(num_actions=num_actions)
        case ["impala", "lstm"]:
            return ImpalaLarge(num_actions=num_actions)
        # impala small (default, lstm, or ff)
        case ["impala"] | ["impala", "small", "ff"]:
            return ImpalaSmallFF(num_actions=num_actions)
        case ["impala", "small", "lstm"]:
            return ImpalaSmall(num_actions=num_actions)
        case _:
            raise ValueError(f"Unknown net architecture spec: {name!r}.")


