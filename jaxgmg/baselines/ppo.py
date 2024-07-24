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
from jaxgmg.environments.base import LevelGenerator # this will go to UED
from jaxgmg.baselines.experience import Transition, Rollout

Metrics = dict[str, Any]


@struct.dataclass
class TrainLevelSet:
    def get_batch(
        self,
        rng: PRNGKey,
        num_levels_in_batch: int,
    ) -> Level: # Level[num_levels_in_batch]
        raise NotImplementedError


@struct.dataclass
class Eval:
    def eval(
        self,
        rng: PRNGKey,
        train_state: TrainState,
        net_init_state: networks.ActorCriticState,
    ) -> Metrics:
        raise NotImplementedError


# # # 
# training entry point


def run(
    rng: PRNGKey,
    env: Env,
    net: networks.ActorCriticNetwork,
    net_init_params: networks.ActorCriticParams,
    net_init_state: networks.ActorCriticState,
    train_level_set: TrainLevelSet,
    evals_dict: dict[str, Eval],
    big_evals_dict: dict[str, Eval],
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
    # evals dimensions
    num_cycles_per_eval: int,
    num_cycles_per_big_eval: int,
    # training animation dimensions
    train_gifs: bool,
    train_gif_grid_width: int,
    train_gif_level_of_detail: int,
    # logging config
    num_cycles_per_log: int,
    save_files_to: str,                 # where to log metrics to disk
    console_log: bool,                  # whether to log metrics to stdout
    wandb_log: bool,                    # whether to log metrics to wandb
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
        eval_cycle = log_ever and t % num_cycles_per_eval == 0
        big_eval_cycle = log_ever and t % num_cycles_per_big_eval == 0
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
        levels_t = train_level_set.get_batch(rng_levels, num_parallel_envs)
    
        
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
            env_metrics = compute_rollout_metrics(
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
            frames = animate_rollouts(
                rollouts=rollouts,
                grid_width=train_gif_grid_width,
                force_lod=train_gif_level_of_detail,
                env=env,
            )
            metrics['env/train'].update({'rollouts_gif': frames})


        # ppo update network on this data a few times
        rng_update, rng_t = jax.random.split(rng_t)
        if log_cycle:
            ppo_start_time = time.perf_counter()
        train_state, advantages, ppo_metrics = ppo_update(
            rng=rng_update,
            train_state=train_state,
            rollouts=rollouts,
            num_epochs=num_epochs_per_cycle,
            num_minibatches_per_epoch=num_minibatches_per_epoch,
            gamma=ppo_gamma,
            clip_eps=ppo_clip_eps,
            gae_lambda=ppo_gae_lambda,
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
        

        # periodic evaluations
        if t % num_cycles_per_eval == 0:
            rng_evals, rng_t = jax.random.split(rng_t)
            eval_metrics = {}
            for eval_name, eval_obj in evals_dict.items():
                rng_eval, rng_evals = jax.random.split(rng_evals)
                eval_metrics[eval_name] = eval_obj.eval(
                    rng=rng_eval,
                    train_state=train_state,
                    net_init_state=net_init_state,
                )
            metrics['eval'].update(eval_metrics)
        
        # periodic big evals
        if t % num_cycles_per_big_eval == 0:
            rng_big_evals, rng_t = jax.random.split(rng_t)
            eval_metrics = {}
            for eval_name, eval_obj in big_evals_dict.items():
                rng_eval, rng_big_evals = jax.random.split(rng_big_evals)
                eval_metrics[eval_name] = eval_obj.eval(
                    rng=rng_eval,
                    train_state=train_state,
                    net_init_state=net_init_state,
                )
            metrics['eval'].update(eval_metrics)
       

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
            
            # log to disk
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
    # ppo hyperparameters
    num_epochs: int,
    num_minibatches_per_epoch: int,
    gamma: float,
    clip_eps: float,
    gae_lambda: float,
    entropy_coeff: float,
    critic_coeff: float,
    # metrics
    compute_metrics: bool,
) -> tuple[
    TrainState,
    Array, # GAE estimates
    Metrics,
]:
    """
    Given a data set of rollouts, perform GAE followed by a few epochs of PPO
    loss updates on the transitions contained in those rollouts.

    TODO: document inputs and outputs.
    """
    # Generalised Advantage Estimation
    initial_carry = (
        jnp.zeros_like(rollouts.final_value),             # float[num_levels]
        rollouts.final_value,                             # float[num_levels]
    )
    stepwise_transitions = jax.tree.map(
        lambda x: einops.rearrange(x, 'levels steps ... -> steps levels ...'),
        rollouts.transitions,
    )                                  # -> Transition[num_steps, num_levels]
    def _gae_accum(carry, transition):
        gae, next_value = carry
        reward = transition.reward                        # float[num_levels] 
        this_value = transition.value                     # float[num_levels]
        done = transition.done                            # bool[num_levels]
        gae = (
            reward
            - this_value
            + (1-done) * gamma * (next_value + gae_lambda * gae)
        )
        return (gae, this_value), gae
    _final_carry, advantages = jax.lax.scan(
        _gae_accum,
        initial_carry,
        stepwise_transitions,
        reverse=True,
        unroll=16, # TODO: parametrise and test this for effect on speed
    )                             # advantages : float[num_steps, num_levels]

    
    # value targets
    stepwise_values = einops.rearrange(
        rollouts.transitions.value,
        "levels steps -> steps levels",
    )                                       # -> float[num_steps, num_levels]
    targets = advantages + stepwise_values  # -> float[num_steps, num_levels]

    
    # compile data set
    data = (stepwise_transitions, advantages, targets)
    data = jax.tree.map(
        lambda x: einops.rearrange(x, 'steps lvls ... -> (steps lvls) ...'),
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
            (loss, loss_components), grads = ppo_loss_aux_and_grad(
                train_state.params,
                apply_fn=train_state.apply_fn,
                data=minibatch,
                clip_eps=clip_eps,
                critic_coeff=critic_coeff,
                entropy_coeff=entropy_coeff,
            )
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, (loss, loss_components, grads)
        train_state, (losses, losses_components, grads) = jax.lax.scan(
            _minibatch,
            train_state,
            data_batched,
        )
        return train_state, (losses, losses_components, grads)
    train_state, (losses, losses_components, grads) = jax.lax.scan(
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
            # avg
            'avg_loss': losses.mean(),
            'avg_loss_actor': losses_components[0].mean(),
            'avg_loss_critic': losses_components[1].mean(),
            'avg_loss_entropy': losses_components[2].mean(),
            'avg_advantage': advantages.mean(),
            'avg_grad_norm_pre_clip': grad_norms.mean(),
            # max
            'max_loss': losses.max(),
            'max_loss_actor': losses_components[0].max(),
            'max_loss_critic': losses_components[1].max(),
            'max_loss_entropy': losses_components[2].max(),
            'max_advantage': advantages.max(),
            'max_grad_norm_pre_clip': grad_norms.max(),
            # std
            'std_loss': losses.std(),
            'std_loss_actor': losses_components[0].std(),
            'std_loss_critic': losses_components[1].std(),
            'std_loss_entropy': losses_components[2].std(),
            'std_advantage': advantages.std(),
            'std_grad_norm_pre_clip': grad_norms.std(),
        }
        # TODO: approx kl
        # TODO: clip proportion
    else:
        metrics = {}
    
    return train_state, advantages, metrics


# # # 
# PPO loss function


@functools.partial(jax.jit, static_argnames=('apply_fn',))
def ppo_loss(
    params,
    apply_fn,
    data: tuple[Transition, Array, Array], # each vectors of length num_data
    clip_eps: float,
    critic_coeff: float,
    entropy_coeff: float,
) -> tuple[
    float,      # loss
    tuple[      # breakdown of loss into three components
        float,
        float,
        float,
    ]
]:
    # unpack minibatch
    trajectories, advantages, targets = data

    # run network to get current value/log_prob prediction
    action_distribution, value, _net_state = apply_fn(
        params,
        trajectories.obs,
        trajectories.net_state,
        trajectories.prev_action,
    )
    log_prob = action_distribution.log_prob(trajectories.action)


    # actor loss
    ratio = jnp.exp(log_prob - trajectories.log_prob)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    actor_loss = -jnp.minimum(
        advantages * ratio,
        advantages * jnp.clip(ratio, 1-clip_eps, 1+clip_eps)
    ).mean()


    # critic loss
    value_diff_clipped = jnp.clip(
        value - trajectories.value,
        -clip_eps,
        clip_eps,
    )
    value_proximal = trajectories.value + value_diff_clipped
    critic_loss = jnp.maximum(
        jnp.square(value - targets),
        jnp.square(value_proximal - targets),
    ).mean() / 2


    # entropy regularisation term
    entropy = action_distribution.entropy().mean()


    total_loss = (
        actor_loss
        + critic_coeff * critic_loss
        - entropy_coeff * entropy
    )
    return total_loss, (actor_loss, critic_loss, entropy)


# # # 
# TRAINING LEVEL SETS


@struct.dataclass
class FixedTrainLevelSet(TrainLevelSet):
    levels: Level       # Level[num_levels]


    @functools.partial(
        jax.jit,
        static_argnames=['num_levels_in_batch'],
    )
    def get_batch(
        self,
        rng: PRNGKey,
        num_levels_in_batch: int,
    ) -> Level: # Level[num_levels_in_batch]
        num_levels = jax.tree.leaves(self.levels)[0].shape[0]
        level_ids = jax.random.choice(
            rng,
            num_levels,
            (num_levels_in_batch,),
            replace=(num_levels_in_batch >= num_levels),
        )
        levels_batch = jax.tree.map(lambda x: x[level_ids], self.levels)
        return levels_batch


@struct.dataclass
class OnDemandTrainLevelSet(TrainLevelSet):
    level_generator: LevelGenerator


    @functools.partial(
        jax.jit,
        static_argnames=['self', 'num_levels_in_batch'],
    )
    def get_batch(
        self,
        rng: PRNGKey,
        num_levels_in_batch: int,
    ) -> Level: # Level[num_levels_in_batch]
        levels_batch = self.level_generator.vsample(
            rng,
            num_levels=num_levels_in_batch,
        )
        return levels_batch


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


# # # 
# EVALUATIONS


@struct.dataclass
class FixedLevelsEval(Eval):
    num_levels: int
    num_steps: int
    discount_rate: float
    env: Env
    levels: Level       # Level[num_levels]


    def eval(
        self,
        rng: PRNGKey,
        train_state: TrainState,
        net_init_state: networks.ActorCriticState,
    ) -> Metrics:
        rollouts = experience.collect_rollouts(
            rng=rng,
            num_steps=self.num_steps,
            train_state=train_state,
            net_init_state=net_init_state,
            env=self.env,
            levels=self.levels,
        )
        eval_metrics = compute_rollout_metrics(
            rollouts=rollouts,
            discount_rate=self.discount_rate,
            benchmark_returns=None,
        )
        return eval_metrics


@struct.dataclass
class FixedLevelsEvalWithBenchmarkReturns(Eval):
    num_levels: int
    num_steps: int
    discount_rate: float
    env: Env
    levels: Level       # Level[num_levels]
    benchmarks: Array   # float[num_levels]


    def eval(
        self,
        rng: PRNGKey,
        train_state: TrainState,
        net_init_state: networks.ActorCriticState,
    ) -> Metrics:
        rollouts = experience.collect_rollouts(
            rng=rng,
            num_steps=self.num_steps,
            train_state=train_state,
            net_init_state=net_init_state,
            env=self.env,
            levels=self.levels,
        )
        eval_metrics = compute_rollout_metrics(
            rollouts=rollouts,
            discount_rate=self.discount_rate,
            benchmark_returns=self.benchmarks,
        )
        return eval_metrics


@struct.dataclass
class AnimatedRolloutsEval(Eval):
    num_levels: int
    levels: Level       # Level[num_levels]
    num_steps: int
    gif_grid_width: int
    gif_level_of_detail: int
    env: Env


    def eval(
        self,
        rng: PRNGKey,
        train_state: TrainState,
        net_init_state: networks.ActorCriticState,
    ) -> Metrics:
        rollouts = experience.collect_rollouts(
            rng=rng,
            num_steps=self.num_steps,
            train_state=train_state,
            net_init_state=net_init_state,
            env=self.env,
            levels=self.levels,
        )
        frames = animate_rollouts(
            rollouts=rollouts,
            grid_width=self.gif_grid_width,
            force_lod=self.gif_level_of_detail,
            env=self.env,
        )
        return {'rollouts_gif': frames}


@struct.dataclass
class HeatmapVisualisationEval(Eval):
    levels: Level
    num_levels: int
    levels_pos: tuple[Array, Array]
    grid_shape: tuple[int, int]
    num_steps: int
    discount_rate: float
    env: Env


    def eval(
        self,
        rng: PRNGKey,
        train_state: TrainState,
        net_init_state: networks.ActorCriticState,
    ) -> Metrics:

        # CHEAP EVALS FROM INIT STATE ONLY
        obs, _ = self.env.vreset_to_level(self.levels)
        vec_net_init_state = jax.tree.map(
            lambda c: einops.repeat(
                c,
                '... -> num_levels ...',
                num_levels=self.num_levels,
            ),
            net_init_state,
        )
        action_distr, values, _net_state = train_state.apply_fn(
            train_state.params,
            obs,
            vec_net_init_state,
            -jnp.ones(self.num_levels, dtype=int),
        )
        # model value -> heatmap
        value_heatmap = generate_heatmap(
            data=values,
            shape=self.grid_shape,
            pos=self.levels_pos,
        )
        # model policy -> diamond map
        action_probs = action_distr.probs
        action_diamond_plot = generate_diamond_plot(
            data=action_probs,
            shape=self.grid_shape,
            pos=self.levels_pos,
        )
    
        # EXPENSIVE EVALS, ROLLOUTS
        # TODO: CONSIDER SEPARATING THESE INTO TWO DIFFERENT EVAL CLASSES
        # model policy rollout returns -> heatmap
        rollouts = experience.collect_rollouts(
            rng=rng,
            num_steps=self.num_steps,
            train_state=train_state,
            net_init_state=net_init_state,
            env=self.env,
            levels=self.levels,
        )
        returns = jax.vmap(compute_average_return, in_axes=(0,0,None))(
            rollouts.transitions.reward,
            rollouts.transitions.done,
            self.discount_rate,
        )
        rollout_heatmap = generate_heatmap(
            data=returns,
            shape=self.grid_shape,
            pos=self.levels_pos,
        )

        return {
            'value_img': value_heatmap,
            'action_probs_img': action_diamond_plot,
            'policy_rollout_return_img': rollout_heatmap,
        }
    

# # # 
# Helper functions


@jax.jit
def compute_rollout_metrics(
    rollouts: Rollout,                  # Rollout[num_levels]
    discount_rate: float,
    benchmark_returns: Array | None,    # float[num_levels]
) -> Metrics:
    """
    Parameters:

    * rollouts: Rollout[num_levels] (with Transition[num_steps] inside)
            The rollouts to score.
    * discount_rate : float
            Used in computing the return metric.
    * benchmark_returns : float[num_levels] | None
            For each level, what is the benchmark (e.g. optimal) return to be
            aiming for? If None, skip this metric.

    Returns:

    * metrics : {str: Any}
            A dictionary of statistics calculated based on the collected
            experience. Each key is prefixed with `metrics_prefix`.
            If `compute_metrics` is False, the dictionary is empty.
    """
    # note: comments use shorthand L = num_levels, S = num_steps.

    # compute returns
    vmap_avg_return = jax.vmap(compute_average_return, in_axes=(0,0,None))
    avg_returns = vmap_avg_return(
        rollouts.transitions.reward,                # float[L (vmapped), S]
        rollouts.transitions.done,                  # bool[L (vmapped), S]
        discount_rate,                              # float
    )                                               # -> float[L (vmapped)]

    # compute episode lengths
    eps_per_step = (
        rollouts.transitions.done.mean(axis=1)      # bool[L, S] -> float[L]
    )
    steps_per_ep = 1 / (eps_per_step + 1e-10)

    # compute average reward
    reward_per_step = (
        rollouts.transitions.reward.mean(axis=1)    # float[L, S] -> float[L]
    )

    metrics = {
        # average over all levels in the batch
        'avg_avg_return': avg_returns.mean(),
        'avg_avg_episode_length': steps_per_ep.mean(),
        'avg_reward_per_step': reward_per_step.mean(),
        # histogram of values for each level in the batch
        'lvl_avg_return_hist': avg_returns,
        'lvl_avg_episode_length_hist': steps_per_ep,
        'lvl_reward_per_step_hist': reward_per_step,
    }

    # compare returns to benchmark returns if provided
    # TODO: allow a dict of different benchmarks (like proxies)
    if benchmark_returns is not None:
        benchmark_regret = benchmark_returns - avg_returns
        metrics.update({
            # average over all levels in the batch
            'avg_benchmark_return': benchmark_returns.mean(),
            'avg_benchmark_regret': benchmark_regret.mean(),
            # histograms of values for each level
            'lvl_benchmark_return_hist': benchmark_returns,
            'lvl_benchmark_regret_hist': benchmark_regret,
        })
    
    # if there are any proxy rewards, add new metrics for each
    proxy_dict = rollouts.transitions.info.get("proxy_rewards", {})
    for proxy_name, proxy_rewards in proxy_dict.items():
        avg_proxy_returns = vmap_avg_return(
            proxy_rewards,              # float[L (vmapped), S]
            rollouts.transitions.done,  # bool[L (vmapped), S]
            discount_rate,              # float
        )                               # -> float[L (vmapped)]
        proxy_reward_per_step = (
            proxy_rewards.mean(axis=1)  # float[L, S] -> float[L]
        )
        metrics[proxy_name] = {
            # average over all levels in the batch
            'avg_avg_return': avg_proxy_returns.mean(),
            'avg_reward_per_step': proxy_reward_per_step.mean(),
            # histrograms of values for each level
            'lvl_avg_return_hist': avg_proxy_returns,
            'lvl_reward_per_step_hist': proxy_reward_per_step,
        }
    
    return metrics


@jax.jit
def compute_average_return(
    rewards: Array,         # float[num_steps]
    dones: Array,           # bool[num_steps]
    discount_rate: float,
) -> float:
    """
    Given a sequence of (reward, done) pairs, compute the average return for
    each episode represented in the sequence.

    Parameters:

    * rewards : float[num_steps]
            Scalar rewards delivered at the conclusion of each timestep.
    * dones : bool[num_steps]
            True indicates the reward was delivered as the episode
            terminated.
    * discount_rate : float
            The return is exponentially discounted sum of future rewards in
            the episode, this is the discount rate for that discounting.

    Returns:

    * average_return : float
            The average of the returns for each episode represented in the
            sequence of (reward, done) pairs.
    """
    # compute per-step returns
    def _accumulate_return(
        next_step_return,
        this_step_reward_and_done,
    ):
        reward, done = this_step_reward_and_done
        this_step_return = reward + (1-done) * discount_rate * next_step_return
        return this_step_return, this_step_return
    _, per_step_returns = jax.lax.scan(
        _accumulate_return,
        0,
        (rewards, dones),
        reverse=True,
    )

    # identify start of each episode
    first_steps = jnp.roll(dones, 1).at[0].set(True)
    
    # average returns at the start of each episode
    total_first_step_returns = jnp.sum(first_steps * per_step_returns)
    num_episodes = jnp.sum(first_steps)
    average_return = total_first_step_returns / num_episodes
    
    return average_return


@functools.partial(jax.jit, static_argnames=('shape',))
def generate_heatmap(data, shape, pos):
    return util.viridis(jnp.zeros(shape).at[pos].set(data))


@functools.partial(jax.jit, static_argnames=('shape',))
def generate_diamond_plot(data, shape, pos):
    color_data = util.viridis(data)
    return einops.rearrange(
        jnp.full((5, 5, *shape, 3), 0.4)
            .at[:, :, pos[0], pos[1], :].set(0.5)
            .at[1, 2, pos[0], pos[1], :].set(color_data[:,0,:])
            .at[2, 1, pos[0], pos[1], :].set(color_data[:,1,:])
            .at[3, 2, pos[0], pos[1], :].set(color_data[:,2,:])
            .at[2, 3, pos[0], pos[1], :].set(color_data[:,3,:]),
        'col row h w rgb -> (h col) (w row) rgb',
    )


@functools.partial(jax.jit, static_argnames=('grid_width','force_lod','env'))
def animate_rollouts(
    rollouts: Rollout, # Rollout[num_levels] (with Transition[num_steps])
    grid_width: int,
    force_lod: int | None = None,
    env: Env | None = None,
) -> Array:
    """
    Transform a vector of rollouts into a sequence of images showing for each
    timestep a matrix of observations.

    Inputs:

    * rollouts : Rollout[num_levels] (each containing Transition[num_steps])
            The rollouts to visualise.
    * grid_width : static int
            How many levels to put in each row of the grid. Must exactly
            divide num_levels (the shape of rollouts).
    * force_lod : optional int (any valid obs_level_of_detail for env)
            Use this level of detail (lod) for the animations.
    * env : optional Env (mandatory if force_lod is provided)
            The environment provides the renderer, used if the level of
            detail is different from the level of detail the obs are already
            encoded at.

    Returns:

    * frames : float[num_steps+4, img_height, img_width, channels]
            The animation.
 
    Notes:

    * In the output type:
      * `num_steps+4` is for 4 frames inserted at the end of the animation to
        mark the end.
      * img_width = grid_width * cell_width + grid_width + 1
      * img_height = grid_height * cell_height + grid_height + 1
      * grid_height = num_levels / grid_width (divides exactly)
      * cell_width and cell_height are dependent on the size of observations
        at the given level of detail.
      * the `+ grid_width + 1` and `+ grid_height + 1` come from 1 pixel of
        padding that is inserted separating each observation in the grid.
      * channels is usually 3 (rgb) but could be otherwise, it depends on the
        shape of the observations.
    
    TODO:

    * Currently the final result state from the rollout is not shown. We
      could add that!
      * It would require tweaking the rollout class to store the final obs
        and the final env_state.
      * It would slightly complicate the observation aseembly phase of this
        function, see comments.
      * It would change the first output axis to `num_steps + 1`.
    """
    num_levels = jax.tree.leaves(rollouts)[0].shape[0]
    assert (num_levels % grid_width) == 0
    assert not (force_lod is not None and env is None)

    # assemble observations at desired level of detail
    if force_lod is not None and force_lod != env.obs_level_of_detail:
        # need to re-render the observations
        vrender = jax.vmap(env.get_obs, in_axes=(0, None,)) # parallel envs
        vvrender = jax.vmap(vrender, in_axes=(0, None,))    # time
        obs = vvrender(rollouts.transitions.env_state, force_lod)
        # TODO: first stack the final env state
    else:
        obs = rollouts.transitions.obs
        # TODO: stack the final observation

    # flash the screen half black for the last frame of each episode
    done_mask = einops.rearrange(
        rollouts.transitions.done,
        'level step -> level step 1 1 1',
    )
    obs = obs * (1. - .4 * done_mask)
    
    # rearrange into a (padded) grid of observations
    obs = jnp.pad(
        obs,
        pad_width=(
            (0, 0), # levels
            (0, 0), # steps
            (0, 1), # height
            (0, 1), # width
            (0, 0), # channel
        ),
    )
    grid = einops.rearrange(
        obs,
        '(level_h level_w) step h w c -> step (level_h h) (level_w w) c',
        level_w=grid_width,
    )
    grid = jnp.pad(
        grid,
        pad_width=(
            (0, 4), # time
            (1, 0), # height
            (1, 0), # width
            (0, 0), # channel
        ),
    )

    return grid


