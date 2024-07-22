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


# # # 
# types

from typing import Tuple, Dict, Any
from chex import Array, ArrayTree, PRNGKey
from jaxgmg.environments.base import EnvState, Env, Level, LevelGenerator
Observation = Array
Metrics = Dict[str, Any]
ActorCriticState = ArrayTree

@struct.dataclass
class TrainLevelSet:
    def get_batch(self, rng, num_levels_in_batch: int) -> Level:
        raise NotImplementedError

@struct.dataclass
class Eval:
    def eval(self, rng: PRNGKey, train_state: TrainState) -> Metrics:
        raise NotImplementedError


# # # 
# training entry point


def run(
    rng: PRNGKey,
    env: Env,
    train_level_set: TrainLevelSet,
    evals_dict: Dict[str, Eval],
    big_evals_dict: Dict[str, Eval],
    # policy config
    net_spec: str,
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

    
    print(f"setting up agent with architecture {net_spec!r}...")
    # select architecture
    net = networks.get_architecture(net_spec, num_actions=env.num_actions)
    # initialise the network
    rng_model_init, rng = jax.random.split(rng, 2)
    net_init_params, net_init_state = net.init_params_and_state(
        rng=rng_model_init,
        obs_shape=env.obs_shape,
        obs_dtype=env.obs_dtype,
    )
            

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
        apply_fn=jax.vmap(net.apply, in_axes=(None, 0, 0)),
        params=net_init_params,
        net_init_state=net_init_state,
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
        (
            trajectories,
            env_obs,
            env_state,
            final_value,
            env_metrics,
        ) = collect_trajectories(
            rng=rng_env,
            train_state=train_state,
            env=env,
            levels=levels_t,
            num_steps=num_env_steps_per_cycle,
            compute_metrics=log_cycle,
            discount_rate=ppo_gamma,
            benchmark_returns=None,
        )
        if log_cycle:
            env_elapsed_time = time.perf_counter() - env_start_time
            metrics['env/train'].update(env_metrics)
            metrics['perf']['env_steps_per_second'] = (
                num_total_env_steps_per_cycle / env_elapsed_time
            )
        if log_cycle and train_gifs:
            frames = animate_trajectories(
                trajectories,
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
            trajectories=trajectories,
            final_obs=env_obs,
            final_value=final_value,
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
# Restore and evaluate a checkpoint


def eval_checkpoint(
    rng: PRNGKey,
    checkpoint_path: str,
    checkpoint_number: int,
    env: Env,
    net_spec: str,
    example_level: Level,
    evals_dict: Dict[str, Eval],
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
    net_init_params, net_init_state = net.init_params_and_carry(
        rng=rng_model,
        obs_shape=env.obs_shape,
        obs_dtype=env.obs_dtype,
    )

    # reload checkpoint of interest
    train_state = TrainState.create(
        apply_fn=jax.vmap(net.apply, in_axes=(None, 0, 0)),
        params=net_init_params,
        net_init_state=net_init_state,
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
# Experience collection / rollouts


@struct.dataclass
class Transition:
    """
    Captures data involved in one environment step, from the observation to
    the actor/critic response to the reward, termination, and info response
    from the environment. Note: the next env_state and observation is
    represented in the next transition.
    """
    env_state: EnvState
    obs: Observation
    rnn_state: ActorCriticState
    value: float
    action: int
    log_prob: float
    reward: float
    done: bool
    info: dict


@functools.partial(
    jax.jit,
    static_argnames=(
        'env',
        'num_steps',
        'compute_metrics',
    ),
)
def collect_trajectories(
    rng: PRNGKey,
    train_state: TrainState,
    env: Env,
    levels: Level,                      # Level[num_levels]
    num_steps: int,
    compute_metrics: bool,
    discount_rate: float | None,
    benchmark_returns: Array | None,    # float[num_levels]
) -> Tuple[
    Transition,                         # Transition[num_steps, num_levels]
    Observation,                        # Observation[num_levels]
    Array,                              # float[num_levels]
    EnvState,
    Metrics,
]:
    """
    Reset an environment to `levels` and rollout a policy in these levels for
    `env_steps` steps.

    Parameters:

    * rng : PRNGKey
            Random state (consumed)
    * train_state : TrainState
            A flax trainstate object, including the policy parameter
            (`.params`) and application function (`.apply_fn`).
            The policy apply function should take params and an observation
            and return an action distribution and value prediction.
    * env : jaxgmg.environments.base.Env
            Provides functions `reset` and `step` (actually, vectorised
            versions `vreset` and `vstep`).
    * net_init_state : ActorCriticState
            An initial carry for the network.
    * levels : jaxgmg.environments.base.Level[num_levels]
            Vector of Level structs. This many environments will be run in
            parallel.
    * num_steps : int
            The environments will run forward for this many steps.
    * compute_metrics : bool (default True)
            Whether to compute metrics.
    * discount_rate : float
            Used in computing the return metric.
    * benchmark_returns : float[num_levels]
            For each level, what is the benchmark (e.g. optimal) return to be
            aiming for? Only used if `compute_metrics` is True.

    Returns:

    * trajectories : Transition[num_steps, num_levels]
            The collected experience.
    * final_obs : Observation[num_levels]
            The observation arising after the final transition in each
            trajectory.
    * final_env_state : jaxgmg.environments.base.EnvState[num_levels]
            The env state arising after the final transition in each
            trajectory.
    * final_value : float[num_levels]
            The network's output for the result of the final transition.
    * metrics : {str: Any}
            A dictionary of statistics calculated based on the collected
            experience. Each key is prefixed with `metrics_prefix`.
            If `compute_metrics` is False, the dictionary is empty.
    """
    # reset environments to these levels
    env_obs, env_state = env.vreset_to_level(levels=levels)
    num_levels = jax.tree.leaves(levels)[0].shape[0]
    vec_net_init_state = jax.tree.map(
        lambda c: einops.repeat(c, '... -> num_levels ...'),
        train_state.net_init_state,
    )
    initial_carry = (env_obs, env_state, vec_net_init_state)

    def _env_step(carry, rng):
        obs, env_state, net_state = carry

        # select action
        rng_action, rng = jax.random.split(rng)
        (
            action_distribution,
            critic_value,
            next_net_state,
        ) = train_state.apply_fn(
            train_state.params,
            obs,
            net_state,
        )
        action = action_distribution.sample(seed=rng_action)
        log_prob = action_distribution.log_prob(action)

        # step env
        rng_step, rng = jax.random.split(rng)
        next_obs, next_env_state, reward, done, info = env.vstep(
            rng_step,
            env_state,
            action,
        )

        # reset to net_init_state in the environmnets that will reset
        next_net_state = jax.tree.map(
            lambda c: jnp.where(
                done,
                vec_net_init_carry,
                next_net_state,
            ),
            next_net_state,
        )
        
        # carry to next step
        carry = (next_obs, next_env_state, next_net_state)
        # output
        transition = Transition(
            env_state=env_state,
            obs=obs,
            net_state=net_state,
            value=critic_value,
            action=action,
            log_prob=log_prob,
            reward=reward,
            done=done,
            info=info,
        )
        return carry, transition

    final_carry, trajectories = jax.lax.scan(
        _env_step,
        initial_carry,
        jax.random.split(rng, num_steps),
    )
    final_obs, final_env_state, final_net_state = final_carry
    final_value, _policy, _carry = train_state.
    final_pi, final_value, _final_carry = train_state.apply_fn(
        train_state.params,
        final_obs,
        final_net_state,
    )


    if compute_metrics:
        # compute returns
        vmap_avg_return = jax.vmap(
            compute_average_return,
            in_axes=(1,1,None),
        )
        actual_returns = vmap_avg_return(
            trajectories.reward,
            trajectories.done,
            discount_rate,
        )
        avg_actual_return = actual_returns.mean()

        metrics = {
            # approx. mean episode completion time (by episode and by level)
            'avg_episode_length_by_episode':
                1 / (trajectories.done.mean() + 1e-10),
            'avg_episode_length_by_level':
                (1 / (trajectories.done.mean(axis=0) + 1e-10)).mean(),
            # average reward per step
            'avg_reward':
                trajectories.reward.mean(),
            # averages returns per episode (by level, vs. benchmark)
            'avg_return_by_level':
                avg_actual_return,
            # return distributions
            'all_returns_by_level_hist':
                actual_returns,
        }
        if benchmark_returns is not None:
            avg_benchmark_return = benchmark_returns.mean()
            metrics.update({
                'avg_benchmark_return_by_level':
                    avg_benchmark_return,
                'benchmark_minus_actual_return':
                    avg_benchmark_return - avg_actual_return,
                'all_benchmark_returns_by_level_hist':
                    benchmark_returns,
            })
        if "proxy_rewards" in trajectories.info:
            for proxy, r_proxy in trajectories.info["proxy_rewards"].items():
                proxy_returns = vmap_avg_return(
                    r_proxy,
                    trajectories.done,
                    discount_rate,
                )
                metrics.update({
                    'proxy_'+proxy+'/avg_reward':
                        r_proxy.mean(),
                    'proxy_'+proxy+'/avg_return_by_level':
                        proxy_returns.mean(),
                    'proxy_'+proxy+'/all_returns_by_level_hist':
                        proxy_returns,
                })
    else:
        metrics = {}

    return (
        trajectories,
        final_obs,
        final_env_state,
        final_value,
        metrics,
    )


@jax.jit
def compute_average_return(
    rewards: Array,
    dones: Array,
    discount_rate: float,
) -> Array:
    """
    Given a sequence of (reward, done) pairs, compute the average return for
    each episode.

    Parameters:

    * rewards : float[t]
            Scalar rewards delivered at the conclusion of each timestep.
    * dones : bool[t]
            True indicates the reward was delivered as the episode
            terminated.
    * discount_rate : float
            The return is exponentially discounted sum of future rewards in
            the episode, this is the discount rate (for one timestep).

    Returns:

    * average_return : float
            The mean of the first-timestep returns for each episode
            represented in the reward/done data.
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


# # # 
# PPO loss function and optimisation


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
    # training state
    train_state: TrainState,
    # data
    trajectories: Transition,   # Transition[num_steps, num_levels]
    final_obs: Observation,     # Observation[num_levels]
    final_value: Array,         # float[num_levels]
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
) -> Tuple[
    TrainState,
    Array, # GAE estimates
    Metrics,
]:
    # generalised advantage estimation
    initial_carry = (
        jnp.zeros_like(final_value),
        final_value,
    )
    def _gae_accum(carry, transition):
        gae, next_value = carry
        reward = transition.reward
        this_value = transition.value
        done = transition.done
        gae = (
            reward
            - this_value
            + (1-done) * gamma * (next_value + gae_lambda * gae)
        )
        return (gae, this_value), gae
    _final_carry, advantages = jax.lax.scan(
        _gae_accum,
        initial_carry,
        trajectories,
        reverse=True,
        unroll=16, # WHY? for speed? test this?
    )


    # value targets
    targets = advantages + trajectories.value
        

    # compile data set
    data = (trajectories, advantages, targets)
    data = jax.tree.map(
        lambda x: einops.rearrange(x, 't parallel ... -> (t parallel) ...'),
        data,
    )
    num_examples, = data[1].shape


    # train on these targets for a few epochs
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
            return train_state, (loss, loss_components)
        train_state, (losses, losses_components) = jax.lax.scan(
            _minibatch,
            train_state,
            data_batched,
        )
        return train_state, (losses, losses_components)
    train_state, (losses, losses_components) = jax.lax.scan(
        _epoch,
        train_state,
        jax.random.split(rng, num_epochs),
    )

    if compute_metrics:
        metrics = {
            'avg_loss': losses.mean(),
            'avg_loss_actor': losses_components[0].mean(),
            'avg_loss_critic': losses_components[1].mean(),
            'avg_loss_entropy': losses_components[2].mean(),
            'avg_advantage': advantages.mean(),
            'std_loss': losses.std(),
            'std_loss_actor': losses_components[0].std(),
            'std_loss_critic': losses_components[1].std(),
            'std_loss_entropy': losses_components[2].std(),
            'std_advantage': advantages.std(),
        }
    else:
        metrics = {}
    
    return train_state, advantages, metrics


@functools.partial(jax.jit, static_argnames=('apply_fn',))
def ppo_loss(
    params,
    apply_fn,
    data: Tuple[Transition, Array, Array],
    clip_eps: float,
    critic_coeff: float,
    entropy_coeff: float,
) -> Tuple[
    float,      # loss
    Tuple[      # breakdown of loss into three components
        float,
        float,
        float,
    ]
]:
    # unpack minibatch
    trajectories, advantages, targets = data

    # run network to get current value/log_prob prediction
    action_distribution, value = apply_fn(
        params,
        trajectories.obs,
        trajectories.net_state,
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
    def get_batch(self, rng, num_levels_in_batch) -> Level:
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
    def get_batch(self, rng, num_levels_in_batch) -> Level:
        levels_batch = self.level_generator.vsample(
            rng,
            num_levels=num_levels_in_batch,
        )
        return levels_batch


# # # 
# EVALUATIONS


@struct.dataclass
class FixedLevelsEval(Eval):
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
    ) -> Metrics:
        *_, eval_metrics = collect_trajectories(
            rng=rng,
            train_state=train_state,
            env=self.env,
            levels=self.levels,
            num_steps=self.num_steps,
            discount_rate=self.discount_rate,
            compute_metrics=True,
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
    ) -> Metrics:
        trajectories, *_ = collect_trajectories(
            rng=rng,
            train_state=train_state,
            env=self.env,
            levels=self.levels,
            num_steps=self.num_steps,
            compute_metrics=False,
            # n/a because compute metrics is false
            discount_rate=None,
            benchmark_returns=None,
        )
        frames = animate_trajectories(
            trajectories,
            grid_width=self.gif_grid_width,
            force_lod=self.gif_level_of_detail,
            env=self.env,
        )
        return {'rollouts_gif': frames}


@struct.dataclass
class HeatmapVisualisationEval(Eval):
    levels: Level
    num_levels: int
    levels_pos: Tuple[Array, Array]
    grid_shape: Tuple[int, int]
    num_steps: int
    discount_rate: float
    env: Env
    # TODO: a flag for 'do rollouts'?


    def eval(
        self,
        rng: PRNGKey,
        train_state: TrainState,
    ) -> Metrics:

        # CHEAP EVALS FROM INIT STATE ONLY
        obs, _ = self.env.vreset_to_level(self.levels)
        vec_net_init_state = jax.tree.map(
            lambda c: einops.repeat(c, '... -> num_levels ...'),
            train_state.net_init_state,
            num_levels=self.num_levels,
        )
        action_distr, values = train_state.apply_fn(
            train_state.params,
            obs,
            vec_net_init_state,
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
        trajectories, *_ = collect_trajectories(
            rng=rng,
            train_state=train_state,
            env=self.env,
            levels=self.levels,
            num_steps=self.num_steps,
            compute_metrics=False,
            # n/a because compute metrics is false
            discount_rate=None,
            benchmark_returns=None,
        )
        returns = jax.vmap(compute_average_return, in_axes=(1,1,None))(
            trajectories.reward,
            trajectories.done,
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


@functools.partial(jax.jit, static_argnames=('grid_width','force_lod','env'))
def animate_trajectories(
    trajectories: Transition,
    grid_width: int,
    force_lod: int = 1,
    env: Env = None,
) -> Array:
    """
    Transform a trajectory into a sequence of images showing for each
    timestep a matrix of observations.
    """
    obs = trajectories.obs

    if force_lod != env.obs_level_of_detail:
        vrender = jax.vmap(env.get_obs, in_axes=(0, None,)) # parallel envs
        vvrender = jax.vmap(vrender, in_axes=(0, None,))    # time
        obs = vvrender(trajectories.env_state, force_lod)

    # flash the screen half black for the last frame of each episode
    done_mask = einops.rearrange(trajectories.done, 't p -> t p 1 1 1')
    obs = obs * (1. - .5 * done_mask)
    
    # rearrange into a (padded) grid of observations
    obs = jnp.pad(
        obs,
        pad_width=(
            (0,0), # time
            (0,0), # parallel
            (0,1), # height
            (0,1), # width
            (0,0), # channel
        ),
    )
    grid = einops.rearrange(
        obs,
        't (p1 p2) h w c -> t (p1 h) (p2 w) c',
        p2=grid_width,
    )
    grid = jnp.pad(
        grid,
        pad_width=(
            (0,8), # time
            (1,0), # height
            (1,0), # width
            (0,0), # channel
        ),
    )

    return grid


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


