"""
Proximal policy optimisation for a given network, environment, and set of
training/eval levels.
"""

import functools
import time

import jax
import jax.numpy as jnp
import einops
from flax.training.train_state import TrainState
from flax import struct
import optax

import tqdm
import wandb

from jaxgmg import util
from jaxgmg.baselines import networks


# # # 
# types

from typing import Tuple, Dict, Any
from chex import Array, PRNGKey
from jaxgmg.environments.base import EnvState, Env, Level
from jaxgmg.environments.base import SplayedLevelSet
Observation = Array
Metrics = Dict[str, Any]


# # # 
# training entry point


def run(
    rng: PRNGKey,
    env: Env,
    train_levels: Level,                            # Level[n]
    eval_levels_dict: Dict[str, Level],             # {str: Level[n]}
    train_benchmark_returns: Array,                 # float[n]
    eval_benchmark_returns_dict: Dict[str, Array],  # {str: float[n]}
    splayset_dict: Dict[str, SplayedLevelSet],
    # policy config
    net: str,
    # training dimensions
    num_total_env_steps: int,
    num_env_steps_per_cycle: int,
    num_parallel_envs: int,
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
    # evaluation config
    num_cycles_per_eval: int,
    num_env_steps_per_eval: int,
    # logging config
    console_log: bool,                  # whether to log metrics to stdout
    wandb_log: bool,                    # whether to log metrics to wandb
    num_cycles_per_log: int,
    train_gifs: bool,
    eval_gifs: bool,
    num_cycles_per_gif: int,
    gif_grid_width: int,
    gif_level_of_detail: int,           # 1, 3, 4, or 8; sprite pixel width
    num_cycles_per_splay: int,
    save_files_to: str,
):


    # initialising file manager
    print("initialising run file manager")
    fileman = util.RunFilesManager(root_path=save_files_to)
    print("  run folder:", fileman.path)

    
    # TODO: Would be a good idea to save the config as a file again


    # TODO: Would also be a good idea for this program to initialise and
    # manage the WANDB run. But that means going back to config hell?
    

    # deriving some additional config variables
    num_total_env_steps_per_cycle = num_env_steps_per_cycle * num_parallel_envs
    num_total_cycles = num_total_env_steps // num_total_env_steps_per_cycle
    num_updates_per_cycle = num_epochs_per_cycle * num_minibatches_per_epoch
    num_train_levels = jax.tree.leaves(train_levels)[0].shape[0]
    

    # alternative axes
    if wandb_log:
        wandb.define_metric("step/env-step")
        wandb.define_metric("env/train/*", step_metric="step/env-step")
        for eval_name in eval_levels_dict.keys():
            wandb.define_metric(
                f"env/eval/{eval_name}/*",
                step_metric="step/env-step",
            )
        wandb.define_metric("step/ppo-update")
        wandb.define_metric("ppo/*", step_metric="step/ppo-update")
        wandb.define_metric("ppo/std/*", step_metric="step/ppo-update")

    
    print(f"setting up agent with {net} architecture...")
    # select agent architecture
    net = networks.get_architecture_class_from_name(net)(
        num_actions=env.num_actions,
    )
    # initialise the network
    rng_model, rng = jax.random.split(rng, 2)
    example_level = jax.tree.map(lambda x: x[0], train_levels)
    example_obs, _ = env.reset_to_level(example_level)
    net_init_params = net.init(rng_model, example_obs)


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
    train_state = TrainState.create(
        apply_fn=jax.vmap(net.apply, in_axes=(None, 0)),
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
        log_cycle = (console_log or wandb_log) and t % num_cycles_per_log == 0
        if log_cycle:
            metrics = {
                'step': {},
                'perf': {},
                'env/train': {},
                'env/eval': {},
                'ppo': {},
            }


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
        level_ids_t = jax.random.choice(
            rng_levels,
            num_train_levels,
            (num_parallel_envs,),
            replace=(num_parallel_envs >= num_train_levels),
        )
        levels_t = jax.tree.map(lambda x: x[level_ids_t], train_levels)
    
        
        # collect experience
        rng_env, rng_t = jax.random.split(rng_t)
        if log_cycle:
            env_start_time = time.perf_counter()
        trajectories, env_obs, env_state, env_metrics = collect_trajectories(
            rng=rng_env,
            train_state=train_state,
            env=env,
            levels=levels_t,
            num_steps=num_env_steps_per_cycle,
            discount_rate=ppo_gamma,
            compute_metrics=log_cycle,
            benchmark_returns=train_benchmark_returns[level_ids_t],
        )
        if log_cycle:
            env_elapsed_time = time.perf_counter() - env_start_time
            metrics['env/train'].update(env_metrics)
            metrics['perf']['env_steps_per_second'] = (
                num_total_env_steps_per_cycle / env_elapsed_time
            )


        # ppo update network on this data a few times
        rng_update, rng_t = jax.random.split(rng_t)
        if log_cycle:
            ppo_start_time = time.perf_counter()
        train_state, advantages, ppo_metrics = ppo_update(
            rng=rng_update,
            train_state=train_state,
            trajectories=trajectories,
            final_obs=env_obs,
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
        

        # periodic evaluation on fixed test levels
        if log_cycle and t % num_cycles_per_eval == 0:
            rng_eval, rng_t = jax.random.split(rng_t)
            # this is a tree map but writing it that way would make it worse
            eval_trajectories = {}
            eval_metrics = {}
            for eval_name, eval_levels in eval_levels_dict.items():
                rng_eval_, rng_eval = jax.random.split(rng_eval)
                eval_trajectories_, *_, eval_metrics_ = collect_trajectories(
                    rng=rng_eval,
                    train_state=train_state,
                    env=env,
                    levels=eval_levels,
                    num_steps=num_env_steps_per_eval,
                    discount_rate=ppo_gamma,
                    compute_metrics=log_cycle,
                    benchmark_returns=eval_benchmark_returns_dict[eval_name],
                )
                eval_trajectories[eval_name] = eval_trajectories_
                eval_metrics[eval_name] = eval_metrics_
            metrics['env/eval'].update(eval_metrics)
        

        # periodic logging
        if log_cycle:
            if console_log:
                progress.write("\n".join([
                    "=" * 59,
                    util.dict2str(metrics),
                    "=" * 59,
                ]))
            if wandb_log:
                wandb.log(step=t, data=util.flatten_dict(metrics))

        
        # periodic training animation saving
        if train_gifs and t % num_cycles_per_gif == 0:
            frames = animate_trajectories(
                trajectories,
                grid_width=gif_grid_width,
                force_lod=gif_level_of_detail,
                env=env,
            )
            gif_path = fileman.get_path(f"gifs/train/{t}.gif")
            util.save_gif(frames, gif_path)
            progress.write("saved gif to " + gif_path)
            
            if wandb_log:
                wandb.log(
                    step=t,
                    data={'gifs/train': util.wandb_gif(frames)},
                )

        
        # periodic eval animation saving
        if eval_gifs and t % num_cycles_per_gif == 0:
            for eval_name, eval_trajectories_ in eval_trajectories.items():
                frames = animate_trajectories(
                    eval_trajectories_,
                    grid_width=gif_grid_width,
                    force_lod=gif_level_of_detail,
                    env=env,
                )
                gif_path = fileman.get_path(f"gifs/eval-{eval_name}/{t}.gif")
                util.save_gif(frames, gif_path)
                progress.write("saved gif to " + gif_path)
                
                if wandb_log:
                    wandb.log(
                        step=t,
                        data={'gifs/eval-'+eval_name: util.wandb_gif(frames)},
                    )
        

        # periodic heatmaps from fixed splayed level sets
        if log_cycle and t % num_cycles_per_splay == 0:
            rng_splay, rng_t = jax.random.split(rng_t)
            images = {}
            for splay_name, splayset in splayset_dict.items():
                rng_splay_, rng_splay = jax.random.split(rng_splay)
                images[splay_name] = analyse_splayset(
                    rng=rng_splay_,
                    train_state=train_state,
                    env=env,
                    splayset=splayset,
                    shape=splayset.grid_shape, # static
                    num_steps=num_env_steps_per_eval,
                    discount_rate=ppo_gamma,
                )
            images = util.flatten_dict(images)
            for img_name, img in images.items():
                img_path = fileman.get_path(f"imgs/splay-{img_name}/{t}.png")
                util.save_image(img, img_path)
            progress.write(
                "saved some splay heatmaps to "
                + fileman.get_path('imgs/splay-*')
            )
            if wandb_log:
                wandb.log(
                    step=t,
                    data={
                        f'splays/{k}': util.wandb_img(v)
                        for k, v in images.items()
                    },
                )


        # ending cycle
        progress.update(num_total_env_steps_per_cycle)


    # ending run
    progress.close()


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
    levels: Level,              # Level[num_levels]
    num_steps: int,
    discount_rate: float,
    compute_metrics: bool,
    benchmark_returns: Array,   # float[num_levels]
) -> Tuple[
    Transition,
    Observation,
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
    * levels : jaxgmg.environments.base.Level[num_levels]
            Vector of Level structs. This many environments will be run in
            parallel.
    * num_steps : int
            The environments will run forward for this many steps.
    * discount_rate : float
            Used in computing the return metric.
    * compute_metrics : bool (default True)
            Whether to compute metrics.
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
    * metrics : {str: Any}
            A dictionary of statistics calculated based on the collected
            experience. Each key is prefixed with `metrics_prefix`.
            If `compute_metrics` is False, the dictionary is empty.
    """
    # reset environments to these levels
    env_obs, env_state = env.vreset_to_level(levels=levels)
    initial_carry = (env_obs, env_state)

    def _env_step(carry, rng):
        obs, env_state = carry

        # select action
        rng_action, rng = jax.random.split(rng)
        action_distribution, critic_value = train_state.apply_fn(
            train_state.params,
            obs,
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
        
        # carry to next step
        carry = (next_obs, next_env_state)
        # output
        transition = Transition(
            env_state=env_state,
            obs=obs,
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
    final_obs, final_env_state = final_carry

    if compute_metrics:
        # compute returns
        avg_actual_return = jax.vmap(
            compute_average_return,
            in_axes=(1,1,None),
        )(
            trajectories.reward,
            trajectories.done,
            discount_rate,
        ).mean()
        avg_benchmark_return = benchmark_returns.mean()
        
        metrics = {
            # approx. mean episode completion time (by episode and by level)
            'avg_episode_length_by_episode':
                1 / (trajectories.done.mean() + 1e-10),
            'avg_episode_length_by_level':
                (1 / (trajectories.done.mean(axis=0) + 1e-10)).mean(),
            # average reward per step
            'avg_reward':
                trajectories.reward.mean(),
            # average return per episode (by level, vs. benchmark)
            'avg_return_by_level':
                avg_actual_return,
            'avg_benchmark_return_by_level':
                avg_benchmark_return,
            'benchmark_minus_actual_return':
                avg_benchmark_return - avg_actual_return,
        }
    else:
        metrics = {}

    return (
        trajectories,
        final_obs,
        final_env_state,
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


@functools.partial(jax.jit, static_argnames=('env', 'shape', 'num_steps'))
def analyse_splayset(
    rng: PRNGKey,
    train_state: TrainState,
    env: Env,
    splayset: SplayedLevelSet,
    shape: int,
    num_steps: int,
    discount_rate: float,
) -> Metrics:
    obs, _ = env.vreset_to_level(splayset.levels)
    action_distr, values = train_state.apply_fn(train_state.params, obs)

    # model value -> heatmap
    value_heatmap = generate_heatmap(
        data=values,
        shape=shape,
        pos=splayset.levels_pos,
    )
    
    # model policy -> diamond map
    action_probs = action_distr.probs
    action_diamond_plot = generate_diamond_plot(
        data=action_probs,
        shape=shape,
        pos=splayset.levels_pos,
    )
    
    # model policy rollout returns -> heatmap
    trajectories, *_ = collect_trajectories(
        rng=rng,
        train_state=train_state,
        env=env,
        levels=splayset.levels,
        num_steps=num_steps,
        discount_rate=discount_rate,
        compute_metrics=False,
        benchmark_returns=None,
    )
    returns = jax.vmap(compute_average_return, in_axes=(1,1,None))(
        trajectories.reward,
        trajectories.done,
        discount_rate,
    ).mean()
    rollout_heatmap = generate_heatmap(
        data=returns,
        shape=shape,
        pos=splayset.levels_pos,
    )

    return {
        'value': value_heatmap,
        'action-probs': action_diamond_plot,
        'policy-rollout-return': rollout_heatmap,
    }
    
    
@functools.partial(jax.jit, static_argnames=('shape',))
def generate_heatmap(data, shape, pos):
    # TODO: colormap
    return einops.repeat(
        jnp.zeros(shape).at[pos].set(data),
        'h w -> h w rgb',
        rgb=3,
    )


@functools.partial(jax.jit, static_argnames=('shape',))
def generate_diamond_plot(data, shape, pos):
    # TODO: colormap
    data = 0.1 + 0.9 * data
    return einops.repeat(
        jnp.full((5, 5, *shape), 0.1)
            .at[:, :, pos[0], pos[1]].set(0.0)
            .at[1, 2, pos[0], pos[1]].set(data[:,0])
            .at[2, 1, pos[0], pos[1]].set(data[:,1])
            .at[3, 2, pos[0], pos[1]].set(data[:,2])
            .at[2, 3, pos[0], pos[1]].set(data[:,3]),
        'col row h w -> (h col) (w row) rgb',
        rgb=3,
    )


# # # 
# RUN FILES MANAGEMENT


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
    trajectories: Transition,
    final_obs: Observation,
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
    _, final_value = train_state.apply_fn(train_state.params, final_obs)
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
    action_distribution, value = apply_fn(params, trajectories.obs)
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

    # TODO: show reward as flashes of colour filters?
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
            (0,16), # time
            (1,0),  # height
            (1,0),  # width
            (0,0),  # channel
        ),
    )

    return grid


