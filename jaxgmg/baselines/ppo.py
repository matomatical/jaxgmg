"""
Proximal policy optimisation for a given network, environment, and set of
training/eval levels.
"""

import collections
import functools
import time
import os

import jax
import jax.numpy as jnp
import einops
from flax.training.train_state import TrainState
from flax import struct
import optax
import orbax.checkpoint as ocp

import tqdm
import wandb

from jaxgmg import util
from jaxgmg.baselines import networks
from jaxgmg.baselines import experience


# # # 
# types

from typing import Any
from chex import Array, PRNGKey
from jaxgmg.environments.base import EnvState, Env, Level, Observation
from jaxgmg.baselines.experience import Transition, Rollout
from jaxgmg.baselines.evals import Eval
from jaxgmg.baselines.autocurricula.base import CurriculumGenerator
from jaxgmg.baselines.autocurricula.base import GeneratorState


# # # 
# training entry point


def run(
    rng: PRNGKey,
    env: Env,
    gen: CurriculumGenerator,
    gen_state: GeneratorState,
    # network
    net: networks.ActorCriticNetwork,
    net_init_params: networks.ActorCriticParams,
    net_init_state: networks.ActorCriticState,
    # evals
    evals_dict: dict[tuple[str, int]: Eval], # {(name, period): eval_obj}
    # PPO hyperparameters
    ppo_lr: float,                      # learning rate
    ppo_gamma: float,                   # discount rate
    ppo_clip_eps: float,
    ppo_gae_lambda: float,
    ppo_entropy_coeff: float,
    ppo_critic_coeff: float,
    ppo_max_grad_norm: float,
    ppo_lr_annealing: bool,
    num_minibatches_per_epoch: int,
    num_epochs_per_cycle: int,
    # training dimensions
    num_total_env_steps: int,
    num_env_steps_per_cycle: int,
    num_parallel_envs: int,
    # logging config
    num_cycles_per_log: int,
    console_log: bool,                  # whether to log metrics to stdout
    wandb_log: bool,                    # whether to log metrics to wandb
    # training animation dimensions
    train_gifs: bool,
    train_gif_grid_width: int,
    # checkpointing config
    checkpointing: bool,
    keep_all_checkpoints: bool,
    max_num_checkpoints: int,
    num_cycles_per_checkpoint: int,
):


    # TODO: Would be a good idea to checkpoint the training levels and eval
    # levels...


    # TODO: Would also be a good idea for this program to initialise and
    # manage the WANDB run. But that means going back to config hell?
    

    # deriving some additional config variables
    num_total_env_steps_per_cycle = num_env_steps_per_cycle * num_parallel_envs
    num_total_cycles = num_total_env_steps // num_total_env_steps_per_cycle
    num_updates_per_cycle = num_epochs_per_cycle * num_minibatches_per_epoch
    

    # define wandb metrics at the end of the first loop (wandb sucks)
    metrics_undefined = True

    
    # initialise the checkpointer
    if checkpointing and not wandb_log:
        print("WARNING: checkpointing requested without wandb logging.")
        print("WARNING: disabling checkpointing!")
        checkpointing = False
    elif checkpointing:
        checkpoint_path = os.path.join(wandb.run.dir, "checkpoints/")
        max_to_keep = None if keep_all_checkpoints else max_num_checkpoints
        checkpoint_manager = ocp.CheckpointManager(
            directory=checkpoint_path,
            options=ocp.CheckpointManagerOptions(
                max_to_keep=max_to_keep,
                save_interval_steps=num_cycles_per_checkpoint,
            ),
        )


    # set up optimiser
    print("setting up optimiser...")
    if ppo_lr_annealing:
        def lr_schedule(updates_count):
            # use a consistent learning rate within each training cycle
            cycles_count = updates_count // num_updates_per_cycle
            frac = updates_count / num_total_cycles
            return (1 - frac) * ppo_lr
        lr = lr_schedule
    else:
        lr = ppo_lr
    

    # init train state
    train_state = TrainState.create(
        apply_fn=net.apply,
        params=net_init_params,
        tx=optax.chain(
            optax.clip_by_global_norm(ppo_max_grad_norm),
            optax.adam(learning_rate=lr),
        ),
    )


    # on-policy training loop
    print("begin training loop!")
    print("(note: first two cycles are slow due to compilation)")
    progress = tqdm.tqdm(
        total=num_total_env_steps,
        unit=" env steps",
        unit_scale=True,
        dynamic_ncols=True,
        colour='magenta',
    )
    for t in range(num_total_cycles):
        rng_t, rng = jax.random.split(rng)
        log_ever = (console_log or wandb_log) 
        log_cycle = log_ever and t % num_cycles_per_log == 0
        eval_cycle = log_ever and any(t % p == 0 for (n, p) in evals_dict)
        metrics = collections.defaultdict(dict)


        # step counting
        if log_cycle:
            t_env_before = t * num_total_env_steps_per_cycle
            t_env_after = t_env_before + num_total_env_steps_per_cycle
            t_ppo_before = t * num_updates_per_cycle
            t_ppo_after = t_ppo_before + num_updates_per_cycle
            metrics['step'].update({
                'ppo-update': t_ppo_before,
                'env-step': t_env_before,
                'ppo-update-after': t_ppo_after,
                'env-step-after': t_env_after,
            })


        # choose levels for this round
        rng_levels, rng_t = jax.random.split(rng_t)
        gen_state, levels_t = gen.get_batch(
            state=gen_state,
            rng=rng_levels,
            num_levels=num_parallel_envs,
        )
        # NOTE: get_batch may return num_levels levels or a number of
        # additional levels (e.g. in the case of parallel robust PLR,
        # 2*num_levels levels are returned). The contract is that we
        # should do rollouts and update UED in all of them, but we should
        # only train in the first num_levels of them.
        # TODO: more fine-grained logging.
    
        
        # collect experience
        rng_env, rng_t = jax.random.split(rng_t)
        if log_cycle:
            env_start_time = time.perf_counter()
        rollouts = experience.collect_rollouts(
            rng=rng_env,
            num_steps=num_env_steps_per_cycle,
            net_apply=train_state.apply_fn,
            net_params=train_state.params,
            net_init_state=net_init_state,
            env=env,
            levels=levels_t,
        )
        if log_cycle:
            env_metrics = experience.compute_rollout_metrics(
                rollouts=rollouts,
                discount_rate=ppo_gamma,
                benchmark_returns=None,
            )
            env_elapsed_time = time.perf_counter() - env_start_time
            metrics['env/train'].update(env_metrics)
            metrics['perf']['env_steps_per_second'] = (
                num_total_env_steps_per_cycle / env_elapsed_time
            )
            # TODO: steps per second is now wrong, doesn't account for actual
            # levels generated and simulated... use size of rollouts
            # TODO: split up each kind of rollouts/metrics?
        if log_cycle and train_gifs:
            frames = experience.animate_rollouts(
                rollouts=rollouts,
                grid_width=train_gif_grid_width,
                env=env,
            )
            metrics['env/train'].update({'rollouts_gif': frames})
            # TODO: split up each kind of rollouts/metrics?
            # TODO: count the number of env steps total along with the number
            # of env steps used for training
        

        # generalised advantage estimation
        advantages = jax.vmap(
            experience.generalised_advantage_estimation,
            in_axes=(0,0,0,0,None,None),
        )(
            rollouts.transitions.reward,
            rollouts.transitions.done,
            rollouts.transitions.value,
            rollouts.final_value,
            ppo_gae_lambda,
            ppo_gamma,
        )
        

        # report experience and performance to level generator
        gen_state = gen.update(
            state=gen_state,
            levels=levels_t,
            rollouts=rollouts,
            advantages=advantages, # shortcut: we did gae already
        )
        if log_cycle:
            metrics['ued'].update(gen.compute_metrics(gen_state))


        # isolate valid levels for ppo updating
        valid_rollouts, valid_advantages = jax.tree.map(
            lambda x: x[:num_parallel_envs],
            (rollouts, advantages),
        )
        # ppo update network on this data a few times
        rng_update, rng_t = jax.random.split(rng_t)
        if log_cycle:
            ppo_start_time = time.perf_counter()
        # ppo step
        train_state, ppo_metrics = ppo_update(
            rng=rng_update,
            train_state=train_state,
            net_init_state=net_init_state,
            rollouts=valid_rollouts,
            advantages=valid_advantages,
            num_epochs=num_epochs_per_cycle,
            num_minibatches_per_epoch=num_minibatches_per_epoch,
            gamma=ppo_gamma,
            clip_eps=ppo_clip_eps,
            entropy_coeff=ppo_entropy_coeff,
            critic_coeff=ppo_critic_coeff,
            do_backprop_thru_time=net.is_recurrent,
            compute_metrics=log_cycle,
        )
        if log_cycle:
            ppo_elapsed_time = time.perf_counter() - ppo_start_time
            metrics['ppo'].update(ppo_metrics)
            metrics['perf']['ppo_updates_per_second'] = (
                num_updates_per_cycle / ppo_elapsed_time
            )
        

        # periodic evaluations
        rng_evals, rng_t = jax.random.split(rng_t)
        for (eval_name, num_cycles_per_eval), eval_obj in evals_dict.items():
            if t % num_cycles_per_eval == 0:
                rng_eval, rng_evals = jax.random.split(rng_evals)
                metrics['eval'][eval_name] = eval_obj.eval(
                    rng=rng_eval,
                    train_state=train_state,
                    net_init_state=net_init_state,
                )


        # periodic logging
        if metrics:
            # log to console
            if console_log:
                progress.write("\n".join([
                    "=" * 59,
                    util.dict2str(metrics),
                    "=" * 59,
                ]))
            
            # log to wandb
            if wandb_log:
                # first time: define metrics
                if metrics_undefined:
                    wandb.define_metric("step/env-step-after")
                    wandb.define_metric("step/ppo-update-after")
                    util.wandb_define_metrics(
                        example_metrics=metrics,
                        step_metric_prefix_mapping={
                            "env/": "step/env-step-after",
                            "eval/": "step/env-step-after",
                            "ued/": "step/env-step-after",
                            "ppo/": "step/ppo-update-after",
                        },
                    )
                    metrics_undefined = False
                # log metrics
                wandb.log(step=t, data=util.wandb_flatten_and_wrap(metrics))

        
        # periodic checkpointing
        if checkpointing and t % num_cycles_per_checkpoint == 0:
            checkpoint_manager.save(
                t,
                args=ocp.args.PyTreeSave(train_state.params),
            )
            progress.write(f"saving checkpoint (wandb will sync at end)...")


        # ending cycle
        progress.update(num_total_env_steps_per_cycle)


    # ending run
    progress.close()
    if checkpointing:
        print("finishing checkpoints...")
        checkpoint_manager.wait_until_finished()
        checkpoint_manager.close()
        # for some reason I have to manually save these files (I thought
        # they would be automatically saved since I put them in the run dir,
        # and the docs say this, but it doesn't seem to be the case...)
        wandb.save(checkpoint_path + "/**", base_path=wandb.run.dir)


# # # 
# PPO update


@functools.partial(
    jax.jit,
    static_argnames=(
        'num_epochs',
        'num_minibatches_per_epoch',
        'do_backprop_thru_time',
        'compute_metrics',
    ),
)
def ppo_update(
    rng: PRNGKey,
    train_state: TrainState,
    net_init_state: networks.ActorCriticState,
    rollouts: Rollout, # Rollout[num_levels] (with Transition[num_steps])
    advantages: Array, # float[num_levels, num_steps]
    # ppo hyperparameters
    num_epochs: int,
    num_minibatches_per_epoch: int,
    gamma: float,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    do_backprop_thru_time: bool,
    # metrics
    compute_metrics: bool,
) -> tuple[
    TrainState,
    dict[str, Any], # metrics
]:
    """
    Given a data set of rollouts, perform GAE followed by a few epochs of PPO
    loss updates on the transitions contained in those rollouts.
    """
    num_levels, num_steps = advantages.shape
    # value targets based on values + GAE estimates
    targets = advantages + rollouts.transitions.value  # -> float[num_levels, num_steps]
    # compile data set
    data = (rollouts.transitions, advantages, targets)


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
            loss_aux_grad = jax.value_and_grad(ppo_loss, has_aux=True)
            (loss, diagnostics), grads = loss_aux_grad(
                train_state.params,
                net_apply=train_state.apply_fn,
                net_init_state=net_init_state,
                transitions=minibatch[0],
                advantages=minibatch[1],
                targets=minibatch[2],
                clip_eps=clip_eps,
                critic_coeff=critic_coeff,
                entropy_coeff=entropy_coeff,
                do_backprop_thru_time=do_backprop_thru_time,
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
        grad_norms = vvgnorm(grads) # -> float[num_epochs, num_minibatches]

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


# # # 
# PPO loss function


@functools.partial(
    jax.jit,
    static_argnames=[
        'net_apply',
        'do_backprop_thru_time',
        'compute_diagnostics',
    ],
)
def ppo_loss(
    params: networks.ActorCriticParams,
    net_apply: networks.ActorCriticForwardPass,
    net_init_state: networks.ActorCriticState,
    transitions: Transition,    # Transition[minibatch_size, num_steps]
    advantages: Array,          # float[minibatch_size, num_steps]
    targets: Array,             # float[minibatch_size, num_steps]
    clip_eps: float,
    critic_coeff: float,
    entropy_coeff: float,
    do_backprop_thru_time: bool,
    compute_diagnostics: bool,  # if False, second return value is empty dict
) -> tuple[
    float,                      # loss
    dict[str, float],           # loss components and other diagnostics
]:
    # run latest network to get current value/action predictions
    if do_backprop_thru_time:
        # recompute hidden states to allow backpropagation thru time (BPTT)
        action_distribution, value = jax.vmap(
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
        # use cached inputs and run forward pass in parallel (no BPTT)
        action_distribution, value = jax.vmap(
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
    clipped_ratio = jnp.clip(ratio, 1-clip_eps, 1+clip_eps)
    std_advantages = (
        (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    )
    actor_loss = -jnp.minimum(
        std_advantages * ratio,
        std_advantages * clipped_ratio,
    ).mean()
    if compute_diagnostics:
        # fraction of clipped ratios
        actor_clipfrac = jnp.mean(jnp.abs(ratio - 1) > clip_eps)
        # KL estimators k1, k3 (http://joschu.net/blog/kl-approx.html)
        actor_approxkl1 = jnp.mean(-logratio)
        actor_approxkl3 = jnp.mean((ratio - 1) - logratio)

    # critic loss
    value_diff = value - transitions.value
    value_diff_clipped = jnp.clip(value_diff, -clip_eps, clip_eps)
    value_proximal = transitions.value + value_diff_clipped
    critic_loss = jnp.maximum(
        jnp.square(value - targets),
        jnp.square(value_proximal - targets),
    ).mean() / 2
    if compute_diagnostics:
        # fraction of clipped value diffs
        critic_clipfrac = jnp.mean(jnp.abs(value_diff) > clip_eps)

    # entropy regularisation term
    entropy = action_distribution.entropy().mean()

    total_loss = (
        actor_loss
        + critic_coeff * critic_loss
        - entropy_coeff * entropy
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


