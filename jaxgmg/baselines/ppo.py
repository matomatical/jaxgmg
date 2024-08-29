"""
Proximal policy optimisation update on a given data set.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
import einops
from flax.training.train_state import TrainState
from flax import struct
import optax # for global_grad_norm for metrics
from chex import Array, PRNGKey

from jaxgmg.baselines import networks
from jaxgmg.baselines.experience import Transition


@struct.dataclass
class ProximalPolicyOptimisation:
    """
    Configure a PPO updater with the following hyperparameters:

    * clip_eps : float
            TODO document.
    * entropy_coeff : float
            TODO document.
    * critic_coeff : float
            TODO document.
    * do_backprop_thru_time : bool (static)
            Whether to do backpropagation through time during the update.
            Should be set to false for feedforward networks for efficiency.
            Should be set to true for recurrent networks so as to actually
            teach them to use their memory capacity.
    """
    # dynamic fields
    clip_eps: float
    entropy_coeff: float
    critic_coeff: float
    # static fields
    do_backprop_thru_time: bool = struct.field(pytree_node=False)


    @functools.partial(
        jax.jit,
        static_argnames=[
            "num_epochs",
            "num_minibatches_per_epoch",
            "compute_metrics",
        ])
    def update(
        self,
        rng: PRNGKey,
        train_state: TrainState,
        net_init_state: networks.ActorCriticState,
        transitions: Transition, # Transition[num_levels, num_steps]
        advantages: Array, # float[num_levels, num_steps]
        # static
        num_epochs: int,
        num_minibatches_per_epoch: int,
        compute_metrics: bool,
    ) -> tuple[
        TrainState,
        dict[str, Any], # metrics
    ]:
        """
        Given a data set of transition sequences from a batch of levels,
        perform the configured number of PPO updates on this data.

        Returns the updated train state and (optionally) some metrics
        collected from the updates (or an empty dict).
        """
        num_levels, num_steps = advantages.shape
        # value targets based on on-policy values + GAEs
        targets = transitions.values[:,:,0] + advantages
        # compile data set
        data = (transitions, advantages, targets)


        # train on this data for a few epochs
        def _epoch(train_state, rng_epoch):
            # shuffle data
            rng_shuffle, rng_epoch = jax.random.split(rng_epoch)
            permutation = jax.random.permutation(rng_shuffle, num_levels)
            data_shuf = jax.tree.map(lambda x: x[permutation], data)
            # split into minibatches
            data_batched = jax.tree.map(
                lambda x: einops.rearrange(
                    x,
                    '(batch within_batch) ... -> batch within_batch ...',
                    batch=num_minibatches_per_epoch,
                ),
                data_shuf,
            )
            # process each minibatch
            def _minibatch(train_state, minibatch):
                loss_aux_grad = jax.value_and_grad(
                    self.ppo_loss,
                    has_aux=True,
                )
                (loss, diagnostics), grads = loss_aux_grad(
                    train_state.params,
                    net_apply=train_state.apply_fn,
                    net_init_state=net_init_state,
                    transitions=minibatch[0],
                    advantages=minibatch[1],
                    targets=minibatch[2],
                    compute_diagnostics=compute_metrics,
                )
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, (loss, diagnostics, grads)
            train_state, (losses, diagnostics, grads) = jax.lax.scan(
                _minibatch,
                train_state,
                data_batched,
            )
            return train_state, (losses, diagnostics, grads)
        train_state, (losses, diagnostics, grads) = jax.lax.scan(
            _epoch,
            train_state,
            jax.random.split(rng, num_epochs),
        )

        # compute metrics
        if compute_metrics:
            # re-compute global grad norm for metrics
            vvgnorm = jax.vmap(jax.vmap(optax.global_norm))
            grad_norms = vvgnorm(grads) # : float[num_epochs, num_minibatches]

            metrics = {
                'avg_loss': losses.mean(),
                **{'avg_'+d: vs.mean() for d, vs in diagnostics.items()},
                'avg_advantage': advantages.mean(),
                'avg_grad_norm_pre_clip': grad_norms.mean(),
                'max': {
                    'max_loss': losses.max(),
                    **{'max_'+d: vs.max() for d, vs in diagnostics.items()},
                    'max_advantage': advantages.max(),
                    'max_grad_norm_pre_clip': grad_norms.max(),
                },
                'std': {
                    'std_loss': losses.std(),
                    **{'std_'+d: vs.std() for d, vs in diagnostics.items()},
                    'std_advantage': advantages.std(),
                    'std_grad_norm_pre_clip': grad_norms.std(),
                },
            }
        else:
            metrics = {}
        
        return train_state, metrics


    @functools.partial(
        jax.jit,
        static_argnames=[
            "net_apply",
            "compute_diagnostics",
        ],
    )
    def ppo_loss(
        self,
        params: networks.ActorCriticParams,
        net_apply: networks.ActorCriticForwardPass,
        net_init_state: networks.ActorCriticState,
        transitions: Transition,    # Transition[minibatch_size, num_steps]
        advantages: Array,          # float[minibatch_size, num_steps]
        targets: Array,             # float[minibatch_size, num_steps]
        compute_diagnostics: bool,  # if False, return empty metrics dict
    ) -> tuple[
        float,                      # loss
        dict[str, float],           # loss components and other diagnostics
    ]:
        # run latest network to get current value/action predictions
        if self.do_backprop_thru_time:
            # recompute hidden states to allow BPTT
            action_distribution, values = jax.vmap(
                networks.evaluate_sequence_recurrent,
                in_axes=(None, None, None, 0, 0, 0)
            )(
                params,
                net_apply,
                net_init_state,
                transitions.obs,
                transitions.done,
                transitions.action,
            )
        else:
            # use cached inputs, run forward pass in parallel (no BPTT)
            action_distribution, values = jax.vmap(
                networks.evaluate_sequence_parallel,
                in_axes=(None, None, 0, 0, 0),
            )(
                params,
                net_apply,
                transitions.obs,
                transitions.net_state,
                transitions.prev_action,
            )

        # actor loss
        log_prob = action_distribution.log_prob(transitions.action)
        logratio = log_prob - transitions.log_prob
        ratio = jnp.exp(logratio)
        clipped_ratio = jnp.clip(ratio, 1-self.clip_eps, 1+self.clip_eps)
        std_advantages = (
            (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        )
        actor_loss = -jnp.minimum(
            std_advantages * ratio,
            std_advantages * clipped_ratio,
        ).mean()
        if compute_diagnostics:
            # fraction of clipped ratios
            actor_clipfrac = jnp.mean(jnp.abs(ratio - 1) > self.clip_eps)
            # KL estimators k1, k3 (http://joschu.net/blog/kl-approx.html)
            actor_approxkl1 = jnp.mean(-logratio)
            actor_approxkl3 = jnp.mean((ratio - 1) - logratio)

        # TODO: vmap this over reward channels
        value = values[:,:,0]
        transitions_value = transitions.values[:,:,0]
        # critic loss
        value_diff = value - transitions_value
        value_diff_clipped = jnp.clip(value_diff, -self.clip_eps, self.clip_eps)
        value_proximal = transitions_value + value_diff_clipped
        critic_loss = jnp.maximum(
            jnp.square(value - targets),
            jnp.square(value_proximal - targets),
        ).mean() / 2
        if compute_diagnostics:
            # fraction of clipped value diffs
            critic_clipfrac = jnp.mean(jnp.abs(value_diff) > self.clip_eps)

        # entropy regularisation term
        entropy = action_distribution.entropy().mean()

        total_loss = (
            actor_loss
            + self.critic_coeff * critic_loss
            - self.entropy_coeff * entropy
        )

        # auxiliary information for logging
        if compute_diagnostics:
            diagnostics = {
                'actor_loss': actor_loss,
                'actor_clipfrac': actor_clipfrac,
                'actor_approxkl1': actor_approxkl1,
                'actor_approxkl3': actor_approxkl3,
                'critic_loss': critic_loss,
                'critic_clipfrac': critic_clipfrac,
                'entropy': entropy,
            }
        else:
            diagnostics = {}

        return total_loss, diagnostics


