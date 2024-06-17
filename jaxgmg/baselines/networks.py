import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import distrax


class ImpalaSmall(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
        # state embedding
        x = nn.Conv(features=16, kernel_size=(8,8), strides=(4,4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4,4), strides=(2,2))(x)
        x = nn.relu(x)
        x = jnp.ravel(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        # we omit lstm
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        # actor head
        logits = nn.Dense(self.num_actions)(x)
        pi = distrax.Categorical(logits=logits)

        # critic head
        v = nn.Dense(1)(x)
        v = jnp.squeeze(v)

        return pi, v


class ImpalaFull(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
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
        # we omit lstm 256
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        # actor head
        logits = nn.Dense(self.num_actions)(x)
        pi = distrax.Categorical(logits=logits)

        # critic head
        v = nn.Dense(1)(x)
        v = jnp.squeeze(v)

        return pi, v



class ReLUFF(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
        # state embedding
        x = jnp.ravel(x)
        x = nn.Dense(
            128,
            kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            128,
            kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            128,
            kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.relu(x)

        # actor head
        logits = nn.Dense(
            self.num_actions,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        pi = distrax.Categorical(logits=logits)

        # critic head
        v = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        v = jnp.squeeze(v)

        return pi, v


class SigmoidalFF(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
        # state embedding
        x = jnp.ravel(x)
        x = nn.Dense(
            128,
            kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.tanh(x)
        x = nn.Dense(
            128,
            kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.tanh(x)
        x = nn.Dense(
            128,
            kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.tanh(x)

        # actor head
        logits = nn.Dense(
            self.num_actions,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        pi = distrax.Categorical(logits=logits)

        # critic head
        v = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        v = jnp.squeeze(v)

        return pi, v


