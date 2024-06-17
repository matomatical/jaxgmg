"""
Universal run script allowing configuring:

* training environment
  * maze size and generating algorithm
  * proxy correlation
  * observation (boolean vs. rgb)
* agent architecture (feed forward actor critic)
* UED algorithm (DR or PLR)
* PPO hyperparameters
* wandb logging, local checkpointing, and saving animations

All from the command line or through exposed functions
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
import orbax.checkpoint as ocp

from jaxgmg.baselines import util

from jaxgmg.environments import cheese_in_the_corner
from jaxgmg.environments import keys_and_chests
from jaxgmg.environments import monster_world
from jaxgmg.procgen import maze_generation

from jaxgmg.baselines.networks import (
    ImpalaFull,
    ImpalaSmall,
    SigmoidalFF,
    ReLUFF,
)
from jaxgmg.baselines.ued_algorithms import DR, PLR


# # # 
# types

from typing import Tuple, Dict, Any
from chex import Array, PRNGKey
from jaxgmg.environments.base import EnvState, Env, Level
Observation = Array
Metrics = Dict[str, Any]


# # # 
# "train" entry point


@util.wandb_run # inits wandb and syncs the arguments with wandb.config
def train(
    # randomness
    seed: int = 42,

    # environment config
    env: str = "corner",
    rgb: bool = False,                  # obs are boolean (default) or rgb
    env_height: int = 9,
    env_width: int = 9,
    env_layout: str = 'blocks',
    
    # config for cheese in the corner env
    env_corner_size: int = 1,
    
    # config for keys and chests env
    env_num_keys: int = 2,
    env_num_chests: int = 6,

    # config for monster world env
    env_num_shields: int = 4,
    env_num_monsters: int = 6,
    env_monster_optimality: float = 0.9,
    
    # config agent
    net: str = "relu-ff",

    # # config UED algorithm
    ued: str = "dr",
    plr_temperature: float = 1,         # positive replay distribution temp
    plr_staleness_coeff: float = 0.5,   # staleness mixture weight in [0,1]
    
    # training dimensions
    num_total_env_steps: int = 20_000_000,
    num_env_steps_per_cycle: int = 256,
    num_parallel_envs: int = 32,
    num_train_levels: int = 2048,
    
    # PPO hyperparameters
    ppo_lr: float = 1e-4,               # learning rate
    ppo_gamma: float = 0.995,           # discount rate
    ppo_clip_eps: float = 0.2,
    ppo_gae_lambda: float = 0.98,
    ppo_entropy_coeff: float = 1e-3,
    ppo_critic_coeff: float = 0.5,
    ppo_max_grad_norm: float = 0.5,
    ppo_lr_annealing: bool = False,
    num_minibatches_per_epoch: int = 1,
    num_epochs_per_cycle: int = 5,
    
    # evaluation config
    num_cycles_per_eval: int = 64,
    num_eval_levels: int = 256,
    num_env_steps_per_eval: int = 1024,
    
    # logging
    console_log: bool = True,           # whether to log metrics to stdout
    wandb_log: bool = False,            # whether to log metrics to wandb
    num_cycles_per_log: int = 16,
    
    # wandb config
    wandb_entity: str = None,
    wandb_project: str = "test",
    wandb_group: str = None,
    wandb_name: str = None,
    wandb_notes: str = None,
    
    # checkpointing
    checkpointing: bool = False,
    num_cycles_per_checkpoint: int = 256,
    
    # gif animations for training/eval rollouts
    train_gifs: bool = False,
    eval_gifs: bool = False,
    num_cycles_per_train_gif: int = 256,
    num_cycles_per_eval_gif: int = 256,
    gif_grid_width_train: int = 8,
    gif_grid_width_eval: int = 16,
    rgb_gifs: bool = False,             # force gifs rgb even if obs are bool
    
    # memory profiling
    profiling: bool = False,
    num_cycles_per_profile: int = 256,
    
    # output save directory
    save_files_to: str = "out/",
):


    # config management
    config = dict(locals())
    print("new run with config:")
    print(util.dict2str(config))


    # initialising file manager
    print("initialising run file manager")
    fileman = util.RunFilesManager(root_path=save_files_to)
    print("  run folder:", fileman.path)
    

    # save the config to disk
    config_save_path = fileman.get_path('config.json')
    util.save_json(config, config_save_path)
    print('saved config to', config_save_path)
    

    # deriving some additional config variables
    num_total_env_steps_per_cycle = num_env_steps_per_cycle * num_parallel_envs
    num_total_cycles = num_total_env_steps // num_total_env_steps_per_cycle
    num_updates_per_cycle = num_epochs_per_cycle * num_minibatches_per_epoch
    

    # alternative axes
    if wandb_log:
        wandb.define_metric("env/step")
        wandb.define_metric("env/train/*", step_metric="env/step")
        wandb.define_metric("env/eval/*", step_metric="env/step")
        wandb.define_metric("ued/*", step_metric="env/step")
        wandb.define_metric("ppo/update")
        wandb.define_metric("ppo/*", step_metric="ppo/update")
        wandb.define_metric("ppo/std/*", step_metric="ppo/update")

    
    # initialise prng
    rng = jax.random.PRNGKey(seed)


    print("setting up environment...")
    maze_generator = maze_generation.get_generator_class_from_name(
        name=env_layout,
    )()
    match env.lower():
        case "corner" | "cheese-in-the-corner":
            env = cheese_in_the_corner.Env(
                rgb=rgb,
            )
            level_generator = cheese_in_the_corner.LevelGenerator(
                height=env_height,
                width=env_width,
                maze_generator=maze_generator,
                corner_size=env_corner_size,
            )
        case "keys" | "keys-and-chests":
            env = keys_and_chests.Env(
                rgb=rgb,
            )
            level_generator = keys_and_chests.LevelGenerator(
                height=env_height,
                width=env_width,
                maze_generator=maze_generator,
                num_keys=env_num_keys,
                num_chests=env_num_chests,
            )
        case "monster" | "monsters" | "monster-world":
            env = monster_world.Env(
                rgb=rgb,
            )
            level_generator = monster_world.LevelGenerator(
                height=env_height,
                width=env_width,
                maze_generator=maze_generator,
                num_shields=env_num_shields,
                num_monsters=env_num_monsters,
                monster_optimality=env_monster_optimality,
            )
        case _:
            raise Exception(f"unknown environment: {env}")


    print(f"generating {num_train_levels} training levels...")
    rng_train_levels, rng = jax.random.split(rng)
    train_levels = level_generator.vsample(
        rng_train_levels,
        num_levels=num_train_levels,
    )
    
    
    print(f"setting up {num_eval_levels} evaluation levels...")
    rng_eval_levels, rng = jax.random.split(rng)
    eval_levels = level_generator.vsample(
        rng_eval_levels,
        num_levels=num_eval_levels,
    )


    print(f"set up UED: {ued} algorithm...")
    match ued.lower():
        case "dr" | "domain-randomisation" | "domain-randomization":
            ued = DR(
                num_levels=num_train_levels,
            )
        case "plr" | "prioritised-level-replay" | "prioritized-level-replay":
            raise NotImplementedError # requires testing
            ued = PLR(
                num_levels=num_train_levels,
                temperature=plr_temperature,
                staleness_coeff=plr_staleness_coeff,
            )
        case _:
            raise Exception(f"unknown ued algorithm: {ued}")
    ued_state = ued.init()


    print(f"setting up agent with {net} architecture...")
    # select agent architecture
    match net.lower():
        case "relu-ff" | "relu":
            net = ReLUFF(
                num_actions=env.num_actions,
            )
        case "impala-small":
            net = ImpalaSmall(
                num_actions=env.num_actions,
            )
        case "impala" | "impala-large":
            net = ImpalaFull(
                num_actions=env.num_actions,
            )
        case _:
            raise Exception(f"unknown net architecture: {net}")
    # initialise the network
    rng_model, rng_input, rng_level, rng = jax.random.split(rng, 4)
    example_level = level_generator.sample(rng_level)
    example_obs, _ = env.reset_to_level(rng_input, example_level)
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
        perf_metrics = {}


        rng_levels, rng_t = jax.random.split(rng_t)
        chosen_level_ids = jax.random.choice(
            rng_levels,
            num_train_levels,
            (num_parallel_envs,),
            replace=False,
            p=ued.distribution(ued_state),
        )
        chosen_levels = jax.tree.map(
            lambda x: x[chosen_level_ids],
            train_levels,
        )
    
        
        # collect experience
        rng_env, rng_t = jax.random.split(rng_t)
        if log_cycle:
            env_start_time = time.perf_counter()
        trajectories, env_obs, env_state, env_metrics = collect_trajectories(
            rng=rng_env,
            train_state=train_state,
            env=env,
            levels=chosen_levels,
            num_steps=num_env_steps_per_cycle,
            discount_rate=ppo_gamma,
            compute_metrics=log_cycle,
        )
        if log_cycle:
            env_elapsed_time = time.perf_counter() - env_start_time
            perf_metrics['env_steps_per_second'] = (
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
            perf_metrics['ppo_updates_per_second'] = (
                num_updates_per_cycle / ppo_elapsed_time
            )
    

        # update ued state
        ued_state, ued_metrics = ued.update(
            state=ued_state,
            chosen_level_ids=chosen_level_ids,
            gaes=advantages,
            compute_metrics=log_cycle,
        )
        

        # periodic evaluation on fixed test levels
        if t % num_cycles_per_eval == 0:
            rng_eval, rng_t = jax.random.split(rng_t)
            eval_trajectories, *_, eval_env_metrics = collect_trajectories(
                rng=rng_eval,
                train_state=train_state,
                env=env,
                levels=eval_levels,
                num_steps=num_env_steps_per_eval,
                discount_rate=ppo_gamma,
                compute_metrics=log_cycle,
            )
        else:
            eval_env_metrics = {}
        

        # periodic logging
        if log_cycle:
            # convert cycle count to num steps/updates
            t_env_before = t * num_total_env_steps_per_cycle
            t_env_after = t_env_before + num_total_env_steps_per_cycle
            t_ppo_before = t * num_updates_per_cycle
            t_ppo_after = t_ppo_before + num_updates_per_cycle
            step_metrics = {
                'ppo/update': t_ppo_before,
                'env/step': t_env_before,
            }

            # optionally log to console
            if console_log:
                progress.write("\n".join([
                    "=" * 59,
                    f"training loop cycle {t}:",
                    util.dict2str(step_metrics),
                    f"env step {t_env_before}--{t_env_after} rollout:",
                    util.dict2str(env_metrics),
                    f"ued state @{t_env_after}:",
                    util.dict2str(ued_metrics),
                    f"ppo updates {t_ppo_before}--{t_ppo_after} loss:",
                    util.dict2str(ppo_metrics),
                    f"eval env rollouts (if evaluating this cycle):",
                    util.dict2str(eval_env_metrics),
                    f"performance metrics:",
                    util.dict2str(perf_metrics),
                    "=" * 59,
                ]))

            # optionally log to wandb
            if wandb_log:
                wandb.log(
                    step=t,
                    data=(
                        step_metrics
                        | util.dict_prefix(env_metrics, "env/train/")
                        | util.dict_prefix(ued_metrics, "ued/")
                        | util.dict_prefix(ppo_metrics, "ppo/")
                        | util.dict_prefix(eval_env_metrics, "env/eval/")
                        | util.dict_prefix(perf_metrics, "perf/")
                    ),
                )

        
        # periodic checkpointing
        if checkpointing and t % num_cycles_per_checkpoint == 0:
            checkpoint_path = fileman.get_path(f"checkpoints/{t}.ocp")
            ocp.PyTreeCheckpointer().save(checkpoint_path, train_state)
            progress.write(f"saved checkpoint to {checkpoint_path}")
        

        # periodic training animation saving
        if train_gifs and t % num_cycles_per_train_gif == 0:
            frames = animate_trajectories(
                trajectories,
                grid_width=gif_grid_width_train,
                force_rgb=rgb_gifs,
                env=env,
            )
            gif_path = fileman.get_path(f"gifs/train/{t}.gif")
            util.save_gif(frames, gif_path)
            progress.write("saved gif to " + gif_path)
            
            if wandb_log:
                wandb.log(step=t, data={'gifs/train': util.wandb_gif(frames)})

        
        # periodic eval animation saving
        if eval_gifs and t % num_cycles_per_eval_gif == 0:
            frames = animate_trajectories(
                eval_trajectories,
                grid_width=gif_grid_width_eval,
                force_rgb=rgb_gifs,
                env=env,
            )
            gif_path = fileman.get_path(f"gifs/eval/{t}.gif")
            util.save_gif(frames, gif_path)
            progress.write("saved gif to " + gif_path)
            
            if wandb_log:
                wandb.log(step=t, data={'gifs/eval': util.wandb_gif(frames)})
        

        # periodic memory profiling
        if profiling and t % num_cycles_per_profile == 0:
            profile_path = fileman.get_path(f"profiles/mem-{t}.gif")
            jax.profiler.save_device_memory_profile(profile_path)
            progress.write("saved memory profile to " + profile_path)
        
        
        # ending cycle
        progress.update(num_total_env_steps_per_cycle)


    # ending run
    progress.close()
    # (the decorator finishes the wandb run for us, so no need to do that)
    print("training run complete.")


# # # 
# "eval" entry point


def evaluate_checkpoint(
    # configure the environment
    # ...
    # locate the checkpoint to evaluate
    checkpoint_directory: str = None,   # path to checkpoints folder
    checkpoint_to_eval: int = -1,       # default: final checkpoint
    # evaluation hyperparameters
    # ... num environments and so on
):
    pass
    # set up environment
    # ...

    # load checkpoint
    # ...

    # do some evaluation
    # ...

    # output or save the results somewhere
    # ...


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
    env,
    levels,
    num_steps: int,
    discount_rate: float,
    compute_metrics: bool,
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
    * env : gmg_environments.base.Env
            Provides functions `reset` and `step` (actually, vectorised
            versions `vreset` and `vstep`).
    * levels : gmg_environments.base.Level[num_levels]
            Vector of Level structs. This many environments will be run in
            parallel.
    * num_steps : int
            The environments will run forward for this many steps.
    * discount_rate : float
            Used in computing the return metric.
    * compute_metrics : bool (default True)
            Whether to compute metrics.

    Returns:

    * trajectories : Transition[num_steps, num_levels]
            The collected experience.
    * final_obs : Observation[num_levels]
            The observation arising after the final transition in each
            trajectory.
    * final_env_state : gmg_environments.base.EnvState[num_levels]
            The env state arising after the final transition in each
            trajectory.
    * metrics : {str: Any}
            A dictionary of statistics calculated based on the collected
            experience. Each key is prefixed with `metrics_prefix`.
            If `compute_metrics` is False, the dictionary is empty.
    """
    # reset environments to these levels
    rng_reset, rng = jax.random.split(rng)
    env_obs, env_state = env.vreset_to_level(
        rng=rng_reset,
        levels=levels,
    )
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
        metrics = {
            # average reward per step
            'avg_reward':
                trajectories.reward.mean(),
            # average return per episode
            'avg_episode_return':
                jax.vmap(
                    compute_average_return,
                    in_axes=(1,1,None),
                )(
                    trajectories.reward,
                    trajectories.done,
                    discount_rate,
                ).mean(),
            # return of optimal policy, average by level
            'avg_optimal_return_by_level':
                jax.vmap(
                    env.optimal_value,
                    in_axes=(0,None),
                )(
                    levels,
                    discount_rate,
                ).mean(),
            # approx. mean episode completion time (by episode and by level)
            'avg_episode_length_by_episode':
                1 / (trajectories.done.mean() + 1e-10),
            'avg_episode_length_by_level':
                (1 / (trajectories.done.mean(axis=0) + 1e-10)).mean(),
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
            'std/std_loss': losses.std(),
            'std/std_loss_actor': losses_components[0].std(),
            'std/std_loss_critic': losses_components[1].std(),
            'std/std_loss_entropy': losses_components[2].std(),
            'std/std_advantage': advantages.std(),
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


@functools.partial(jax.jit, static_argnames=('grid_width','force_rgb','env'))
def animate_trajectories(
    trajectories: Transition,
    grid_width: int,
    force_rgb: bool = False,
    env: Env = None,
) -> Array:
    """
    Transform a trajectory into a sequence of images showing for each
    timestep a matrix of observations.

    # TODO: show reward as flashes of colour filters?
    """
    obs = trajectories.obs

    if force_rgb:
        vrender = jax.vmap(env.get_obs, in_axes=(0, None,)) # parallel envs
        vvrender = jax.vmap(vrender, in_axes=(0, None,))    # time
        obs = vvrender(trajectories.env_state, force_rgb)

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


# # # 
# Program entry point

if __name__ == "__main__":
    # 'typer' automatically turns functions into a CLI based on their
    # parameters and type annotations
    import typer
    app = typer.Typer(
        no_args_is_help=True,
        add_completion=False,
        pretty_exceptions_show_locals=False, # can turn on during debugging
    )
    # add the following functions to the CLI as subcommands
    app.command()(train)
    app.command()(evaluate_checkpoint)
    # launch program: parse arguments from command line, pass to the function
    app()
    
