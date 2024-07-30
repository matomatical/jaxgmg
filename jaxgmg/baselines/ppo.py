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
from jaxgmg.baselines import autocurricula


# # # 
# types

from typing import Any
from chex import Array, PRNGKey
from jaxgmg.environments.base import EnvState, Env, Level, Observation
from jaxgmg.baselines.experience import Transition, Rollout
from jaxgmg.baselines.evals import Eval
from jaxgmg.baselines.autocurricula import CurriculumLevelGenerator


# # # 
# training entry point


def run(
    rng: PRNGKey,
    env: Env,
    gen: CurriculumLevelGenerator,
    gen_state: CurriculumLevelGenerator.State,
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
    save_files_to: str,                 # where to log metrics to disk
    console_log: bool,                  # whether to log metrics to stdout
    wandb_log: bool,                    # whether to log metrics to wandb
    # training animation dimensions
    train_gifs: bool,
    train_gif_grid_width: int,
    train_gif_level_of_detail: int,
    # checkpointing config
    checkpointing: bool,
    keep_all_checkpoints: bool,
    max_num_checkpoints: int,
    num_cycles_per_checkpoint: int,
):


    # initialising file manager
    print("initialising run file manager")
    fileman = util.RunFilesManager(root_path=save_files_to)
    print("  run folder:", fileman.path)

    
    # TODO: Would be a good idea to save the config as a file again


    # TODO: Would be a good idea to checkpoint the training levels and eval
    # levels...


    # TODO: Would also be a good idea for this program to initialise and
    # manage the WANDB run. But that means going back to config hell?
    

    # deriving some additional config variables
    num_total_env_steps_per_cycle = num_env_steps_per_cycle * num_parallel_envs
    num_total_cycles = num_total_env_steps // num_total_env_steps_per_cycle
    num_updates_per_cycle = num_epochs_per_cycle * num_minibatches_per_epoch
    

    # TODO: define wandb axes

    
    # initialise the checkpointer
    checkpoint_path = fileman.get_path("checkpoints")
    checkpoint_manager = ocp.CheckpointManager(
        directory=checkpoint_path,
        options=ocp.CheckpointManagerOptions(
            max_to_keep=None if keep_all_checkpoints else max_num_checkpoints,
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
        apply_fn=jax.vmap(net.apply, in_axes=(None, 0, 0, 0)),
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
                'ppo-update-ater': t_ppo_after,
                'env-step-after': t_env_after,
            })


        # choose levels for this round
        rng_levels, rng_t = jax.random.split(rng_t)
        gen_state, levels_t = gen.get_batch(
            state=gen_state,
            rng=rng_levels,
            num_levels=num_parallel_envs,
        )
    
        
        # collect experience
        rng_env, rng_t = jax.random.split(rng_t)
        if log_cycle:
            env_start_time = time.perf_counter()
        rollouts = experience.collect_rollouts(
            rng=rng_env,
            num_steps=num_env_steps_per_cycle,
            train_state=train_state,
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
        if log_cycle and train_gifs:
            frames = experience.animate_rollouts(
                rollouts=rollouts,
                grid_width=train_gif_grid_width,
                force_lod=train_gif_level_of_detail,
                env=env,
            )
            metrics['env/train'].update({'rollouts_gif': frames})
        

        # generalised advantage estimation
        advantages = jax.vmap(
            experience.generalised_advantage_estimation,
            in_axes=(0,None,None),
        )(
            rollouts,
            ppo_gae_lambda,
            ppo_gamma,
        )


        # ppo update network on this data a few times
        rng_update, rng_t = jax.random.split(rng_t)
        if log_cycle:
            ppo_start_time = time.perf_counter()
        # ppo step
        train_state, ppo_metrics = ppo_update(
            rng=rng_update,
            train_state=train_state,
            rollouts=rollouts,
            advantages=advantages,
            num_epochs=num_epochs_per_cycle,
            num_minibatches_per_epoch=num_minibatches_per_epoch,
            gamma=ppo_gamma,
            clip_eps=ppo_clip_eps,
            entropy_coeff=ppo_entropy_coeff,
            critic_coeff=ppo_critic_coeff,
            compute_metrics=log_cycle,
        )
        if log_cycle:
            ppo_elapsed_time = time.perf_counter() - ppo_start_time
            metrics['ppo'].update(ppo_metrics)
            metrics['perf']['ppo_updates_per_second'] = (
                num_updates_per_cycle / ppo_elapsed_time
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
                metrics_wandb = util.flatten_dict(metrics)
                for key, val in metrics_wandb.items():
                    if key.endswith("_hist"):
                        metrics_wandb[key] = wandb.Histogram(val)
                    elif key.endswith("_gif"):
                        metrics_wandb[key] = util.wandb_gif(val)
                    elif key.endswith("_img"):
                        metrics_wandb[key] = util.wandb_img(val)
                    wandb.log(step=t, data=metrics_wandb)
            
            # either way, log to disk
            metrics_disk = util.flatten_dict(metrics)
            metrics_path = fileman.get_path(f"metrics/{t}/")
            progress.write(f"saving metrics to {metrics_path}...")
            skip_keys = []
            for key, val in metrics_disk.items():
                if key.endswith("_hist"):
                    metrics_disk[key] = val.tolist()
                elif key.endswith("_gif"):
                    path = metrics_path + key[:-4].replace('/','-') + ".gif"
                    util.save_gif(val, path)
                    skip_keys.append(key)
                elif key.endswith("_img"):
                    path = metrics_path + key[:-4].replace('/','-') + ".png"
                    util.save_image(val, path)
                    skip_keys.append(key)
                elif isinstance(val, jax.Array):
                    metrics_disk[key] = val.item()
            for key in skip_keys:
                del metrics_disk[key]
            util.save_json(metrics_disk, metrics_path + "metrics.json")

        
        # periodic checkpointing
        if checkpointing and t % num_cycles_per_checkpoint == 0:
            checkpoint_manager.save(
                t,
                args=ocp.args.PyTreeSave(train_state),
            )
            progress.write(f"saving checkpoint to {checkpoint_path}/{t}...")


        # ending cycle
        progress.update(num_total_env_steps_per_cycle)


    # ending run
    progress.close()
    print("finishing checkpoints...")
    checkpoint_manager.wait_until_finished()
    checkpoint_manager.close()


# # # 
# PPO update


@functools.partial(
    jax.jit,
    static_argnames=(
        'num_epochs',
        'num_minibatches_per_epoch',
        'compute_metrics',
    ),
)
def ppo_update(
    rng: PRNGKey,
    train_state: TrainState,
    rollouts: Rollout, # Rollout[num_levels] (with Transition[num_steps])
    advantages: Array, # float[num_levels, num_steps]
    # ppo hyperparameters
    num_epochs: int,
    num_minibatches_per_epoch: int,
    gamma: float,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    # metrics
    compute_metrics: bool,
) -> tuple[
    TrainState,
    dict[str, Any], # metrics
]:
    """
    Given a data set of rollouts, perform GAE followed by a few epochs of PPO
    loss updates on the transitions contained in those rollouts.

    TODO: document inputs and outputs.
    """
    
    # value targets based on values + GAE estimates
    targets = advantages + rollouts.transitions.value  # -> float[num_levels, num_steps]

    
    # compile data set
    data = (rollouts.transitions, advantages, targets)
    data = jax.tree.map(
        lambda x: einops.rearrange(x, 'lvls steps ... -> (lvls steps) ...'),
        data,
    )
    num_examples, = data[1].shape


    # train on this data for a few epochs
    def _epoch(train_state, rng_epoch):
        # shuffle data
        rng_shuffle, rng_epoch = jax.random.split(rng_epoch)
        permutation = jax.random.permutation(rng_shuffle, num_examples)
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
            ppo_loss_aux_and_grad = jax.value_and_grad(ppo_loss, has_aux=True)
            (loss, diagnostics), grads = ppo_loss_aux_and_grad(
                train_state.params,
                apply_fn=train_state.apply_fn,
                transitions=minibatch[0],
                advantages=minibatch[1],
                targets=minibatch[2],
                clip_eps=clip_eps,
                critic_coeff=critic_coeff,
                entropy_coeff=entropy_coeff,
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
    static_argnames=('apply_fn', 'compute_diagnostics'),
)
def ppo_loss(
    params,
    apply_fn,
    transitions: Transition,    # Transition[minibatch_size]
    advantages: Array,          # float[minibatch_size]
    targets: Array,             # float[minibatch_size]
    clip_eps: float,
    critic_coeff: float,
    entropy_coeff: float,
    compute_diagnostics: bool,  # if True, second return value is {}
) -> tuple[
    float,                      # loss
    dict[str, float],           # loss components and other diagnostics
]:
    # run network to get current value/log_prob prediction
    action_distribution, value, _net_state = apply_fn(
        params,
        transitions.obs,
        transitions.net_state,
        transitions.prev_action,
    )
    log_prob = action_distribution.log_prob(transitions.action)

    # actor loss
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


# # # 
# Restore and evaluate a checkpoint


def eval_checkpoint(
    rng: PRNGKey,
    checkpoint_path: str,
    checkpoint_number: int,
    env: Env,
    net_spec: str,
    example_level: Level,
    evals_dict: dict[str, Eval],
    save_files_to: str,
):
    fileman = util.RunFilesManager(root_path=save_files_to)
    print("  run folder:", fileman.path)
    
    # initialise the checkpointer
    checkpoint_manager = ocp.CheckpointManager(
        directory=os.path.abspath(checkpoint_path),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=None,
            save_interval_steps=0,
        ),
    )
    
    # select agent architecture
    net = networks.get_architecture(net_spec, num_actions=env.num_actions)
    # initialise the network to get the example type
    rng_model_init, rng = jax.random.split(rng, 2)
    net_init_params, net_init_state = net.init_params_and_state(
        rng=rng_model_init,
        obs_type=env.obs_type(level=example_level),
    )

    # reload checkpoint of interest
    train_state = TrainState.create(
        apply_fn=jax.vmap(net.apply, in_axes=(None, 0, 0, 0)),
        params=net_init_params,
        tx=optax.sgd(0), # dummy, will be overridden
    )
    train_state_dtype = jax.tree.map(
        ocp.utils.to_shape_dtype_struct,
        train_state,
    )
    train_state = checkpoint_manager.restore(
        checkpoint_number,
        args=ocp.args.PyTreeRestore(train_state_dtype),
    )

    # perform evals and log metrics
    metrics = {}
    for eval_name, eval_obj in evals_dict.items():
        print("running eval", eval_name, "...")
        rng_eval, rng = jax.random.split(rng)
        metrics[eval_name] = eval_obj.eval(
            rng=rng_eval,
            train_state=train_state,
            net_init_state=net_init_state,
        )
        print(util.dict2str(metrics[eval_name]))
        print("=" * 59)
        
    # log the metrics disk
    metrics_disk = util.flatten_dict(metrics)
    metrics_path = fileman.get_path(f"metrics/")
    print(f"saving metrics to {metrics_path}...")
    skip_keys = []
    for key, val in metrics_disk.items():
        if key.endswith("_hist"):
            metrics_disk[key] = val.tolist()
        elif key.endswith("_gif"):
            path = metrics_path + key[:-4].replace('/','-') + ".gif"
            util.save_gif(val, path)
            skip_keys.append(key)
        elif key.endswith("_img"):
            path = metrics_path + key[:-4].replace('/','-') + ".png"
            util.save_image(val, path)
            skip_keys.append(key)
        elif isinstance(val, jax.Array):
            metrics_disk[key] = val.item()
    for key in skip_keys:
        del metrics_disk[key]
    util.save_json(metrics_disk, metrics_path + "metrics.json")


