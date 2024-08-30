"""
Run proximal policy optimisation with unsupervised environment design for a
given network, environment, and set of training/eval levels. Integrated with
wandb.
"""

import collections
import functools
import time
import os

import jax
from flax.training.train_state import TrainState
import optax
import orbax.checkpoint as ocp

import tqdm
import wandb

from jaxgmg import util
from jaxgmg.baselines import networks
from jaxgmg.baselines import experience
from jaxgmg.baselines import evals
from jaxgmg.baselines.ppo import ProximalPolicyOptimisation
from jaxgmg.baselines.autocurricula import dr_infinite
from jaxgmg.baselines.autocurricula import dr_finite
from jaxgmg.baselines.autocurricula import plr
from jaxgmg.baselines.autocurricula import accel

# types and abstract base classes used for type annotations
from typing import Any, Callable
from chex import Array, PRNGKey
from jaxgmg.environments.base import EnvState, Env, Level, Observation
from jaxgmg.environments.base import LevelGenerator, LevelMutator
from jaxgmg.environments.base import LevelSolver, LevelMetrics
from jaxgmg.baselines.experience import Transition, Rollout
from jaxgmg.baselines.evals import Eval
from jaxgmg.baselines.autocurricula.base import CurriculumGenerator
from jaxgmg.baselines.autocurricula.base import GeneratorState


# # # 
# training entry point


def run(
    seed: int,
    # environment-specific stuff
    env: Env,
    train_level_generator: LevelGenerator,
    level_mutator: LevelMutator | None,
    level_solver: LevelSolver | None,
    level_metrics: LevelMetrics | None,
    eval_level_generators: dict[str, LevelGenerator],
    fixed_eval_levels: dict[str, Level],
    heatmap_splayer_fn: Callable | None,
    classify_level_is_shift: Callable[[Level], bool] | None,
    # actor critic policy config
    net_cnn_type: str,
    net_rnn_type: str,
    net_width: int,
    # ued config
    ued: str,
    prob_shift: float,
    num_train_levels: int,
    plr_buffer_size: int,
    plr_temperature: float,
    plr_staleness_coeff: float,
    plr_prob_replay: float,
    plr_regret_estimator: str,
    plr_robust: bool,
    # PPO config
    ppo_lr: float,
    ppo_gamma: float,
    ppo_clip_eps: float,
    ppo_gae_lambda: float,
    ppo_entropy_coeff: float,
    ppo_critic_coeff: float,
    ppo_proxy_critic_coeff: float,
    ppo_max_grad_norm: float,
    ppo_lr_annealing: bool,
    train_proxy_critic: bool,
    proxy_name: bool,
    num_minibatches_per_epoch: int,
    num_epochs_per_cycle: int,
    # training run dimensions
    num_total_env_steps: int,
    num_env_steps_per_cycle: int,
    num_parallel_envs: int,
    # training animation dimensions
    train_gifs: bool,
    train_gif_grid_width: int,
    # evals config
    num_cycles_per_eval: int,
    num_eval_levels: int,
    num_env_steps_per_eval: int,
    num_cycles_per_big_eval: int,
    eval_gif_grid_width: int,
    # logging config
    num_cycles_per_log: int,
    console_log: bool,
    wandb_log: bool,
    # checkpointing config
    checkpointing: bool,
    keep_all_checkpoints: bool,
    max_num_checkpoints: int,
    num_cycles_per_checkpoint: int,
):
    # deriving some additional config variables
    num_total_env_steps_per_cycle = num_env_steps_per_cycle * num_parallel_envs
    num_total_cycles = num_total_env_steps // num_total_env_steps_per_cycle
    num_updates_per_cycle = num_epochs_per_cycle * num_minibatches_per_epoch


    print(f"seeding random number generator with {seed=}...")
    rng = jax.random.PRNGKey(seed=seed)
    rng_setup, rng_train = jax.random.split(rng)

    
    print(f"configuring curriculum with {ued=}...")
    rng_train_levels, rng_setup = jax.random.split(rng_setup)
    if ued == "dr":
        gen = dr_infinite.CurriculumGenerator(
            level_generator=train_level_generator,
        )
        gen_state = gen.init()
    elif ued == "dr-finite":
        train_levels = train_level_generator.vsample(
            rng_train_levels,
            num_levels=num_train_levels,
        )
        gen = dr_finite.CurriculumGenerator()
        gen_state = gen.init(
            levels=train_levels,
        )
    elif ued == "plr":
        gen = plr.CurriculumGenerator(
            level_generator=train_level_generator,
            level_metrics=level_metrics,
            buffer_size=plr_buffer_size,
            temperature=plr_temperature,
            staleness_coeff=plr_staleness_coeff,
            prob_replay=plr_prob_replay,
            regret_estimator=plr_regret_estimator,
            robust=plr_robust,
        )
        gen_state = gen.init(
            rng=rng_train_levels,
            batch_size_hint=num_parallel_envs,
        )
    elif ued == "accel":
        assert level_mutator is not None, "accel requires mutator"
        gen = accel.CurriculumGenerator(
            level_generator=train_level_generator,
            level_metrics=level_metrics,
            level_mutator=level_mutator,
            buffer_size=plr_buffer_size,
            temperature=plr_temperature,
            staleness_coeff=plr_staleness_coeff,
            prob_replay=plr_prob_replay,
            regret_estimator=plr_regret_estimator,
            robust=plr_robust,
        )
        gen_state = gen.init(
            rng=rng_train_levels,
            batch_size_hint=num_parallel_envs,
        )
    else:
        raise ValueError(f"unknown UED algorithm: {ued!r}")


    evals_dict = {}


    print(f"configuring eval batches from {len(eval_level_generators)} level generators...")
    rng_evals, rng_setup = jax.random.split(rng_setup)
    for levels_name, level_generator in eval_level_generators.items():
        print(f"  generating {num_eval_levels} {levels_name!r} levels...")
        eval_name = f"batch-{levels_name}"
        rng_eval_levels, rng_evals = jax.random.split(rng_evals)
        levels = level_generator.vsample(
            rng_eval_levels,
            num_levels=num_eval_levels,
        )
        if level_solver is not None:
            print("  also solving generated levels...")
            benchmark_returns = level_solver.vmap_level_value(
                level_solver.vmap_solve(levels),
                levels,
            )
            levels_eval = evals.FixedLevelsEvalWithBenchmarkReturns(
                num_levels=num_eval_levels,
                num_steps=num_env_steps_per_eval,
                discount_rate=ppo_gamma,
                levels=levels,
                benchmarks=benchmark_returns,
                benchmark_proxies=None,
                env=env,
                period=num_cycles_per_eval,
            )
        else:
            levels_eval = evals.FixedLevelsEval(
                num_levels=num_eval_levels,
                num_steps=num_env_steps_per_eval,
                discount_rate=ppo_gamma,
                levels=levels,
                env=env,
                period=num_cycles_per_eval,
            )
        rollouts_eval = evals.AnimatedRolloutsEval(
            num_levels=num_eval_levels,
            levels=levels,
            num_steps=env.max_steps_in_episode,
            gif_grid_width=eval_gif_grid_width,
            env=env,
            period=num_cycles_per_big_eval,
        )
        evals_dict[eval_name] = evals.EvalList.create(
            levels_eval,
            rollouts_eval,
        )


    print(f"configuring evals for {len(fixed_eval_levels)} fixed eval levels...")
    for level_name, level in fixed_eval_levels.items():
        print(f"  registering fixed level {level_name!r}")
        eval_name = f"fixed-{level_name}"
        solo_eval = evals.SingleLevelEval(
            num_steps=num_env_steps_per_eval,
            discount_rate=ppo_gamma,
            level=level,
            env=env,
            period=num_cycles_per_eval,
        )
        if heatmap_splayer_fn is not None:
            print("  also splaying level for heatmap evals...")
            levels, num_levels, levels_pos, grid_shape = (
                heatmap_splayer_fn(level)
            )
            spawn_heatmap_eval = evals.ActorCriticHeatmapVisualisationEval(
                levels=levels,
                num_levels=num_levels,
                levels_pos=levels_pos,
                grid_shape=grid_shape,
                env=env,
                period=num_cycles_per_big_eval,
            )
            rollout_heatmap_eval = evals.RolloutHeatmapVisualisationEval(
                levels=levels,
                num_levels=num_levels,
                levels_pos=levels_pos,
                grid_shape=grid_shape,
                env=env,
                discount_rate=ppo_gamma,
                num_steps=num_env_steps_per_eval,
                period=num_cycles_per_big_eval,
            )
            evals_dict[eval_name] = evals.EvalList.create(
                solo_eval,
                spawn_heatmap_eval,
                rollout_heatmap_eval,
            )
        else:
            evals_dict[eval_name] = solo_eval


    print("configuring actor critic network...")
    # select architecture
    print(f"  {net_cnn_type=}")
    print(f"  {net_rnn_type=}")
    net = networks.Impala(
        num_actions=env.num_actions,
        cnn_type=net_cnn_type,
        rnn_type=net_rnn_type,
        width=net_width,
    )
    # initialise the network
    print("  initialising network...")
    rng_model_init, rng_setup = jax.random.split(rng_setup)
    rng_example_level, rng_setup = jax.random.split(rng_setup)
    example_level = train_level_generator.sample(rng_example_level)
    net_init_params, net_init_state = net.init_params_and_state(
        rng=rng_model_init,
        obs_type=env.obs_type(level=example_level),
    )
    param_count = sum(p.size for p in jax.tree_leaves(net_init_params))
    print("  number of parameters:", param_count)


    # initialise the checkpointer
    if checkpointing and not wandb_log:
        print("WARNING: checkpointing requested without wandb logging.")
        print("WARNING: disabling checkpointing!")
        checkpointing = False
    elif checkpointing:
        print("initialising the checkpointer...")
        checkpoint_path = os.path.join(wandb.run.dir, "checkpoints/")
        max_to_keep = None if keep_all_checkpoints else max_num_checkpoints
        checkpoint_manager = ocp.CheckpointManager(
            directory=checkpoint_path,
            options=ocp.CheckpointManagerOptions(
                max_to_keep=max_to_keep,
                save_interval_steps=num_cycles_per_checkpoint,
            ),
        )
    # TODO: Would be a good idea to checkpoint the training levels and eval
    # levels...


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
    optimiser = optax.chain(
        optax.clip_by_global_norm(ppo_max_grad_norm),
        optax.adam(learning_rate=lr),
    )
    

    # init train state
    print("initialising train state...")
    train_state = TrainState.create(
        apply_fn=net.apply,
        params=net_init_params,
        tx=optimiser,
    )


    print("configuring PPO updater...")
    ppo = ProximalPolicyOptimisation(
        clip_eps=ppo_clip_eps,
        entropy_coeff=ppo_entropy_coeff,
        critic_coeff=ppo_critic_coeff,
        train_proxy_critic=train_proxy_critic,
        proxy_critic_coeff=ppo_proxy_critic_coeff,
        do_backprop_thru_time=net.is_recurrent,
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
    step_counts = collections.defaultdict(int)
    for t in range(num_total_cycles):
        rng_t, rng_train = jax.random.split(rng_train)
        log_cycle = (console_log or wandb_log) and t % num_cycles_per_log == 0
        first_cycle = (t == 0)
        if log_cycle:
            metrics = collections.defaultdict(dict)
        
        
        # mark 'before' step counts
        if log_cycle:
            metrics['step'].update({
                f'{key}-before': count for key, count in step_counts.items()
            })


        # choose levels for this round from curriculum
        rng_levels, rng_t = jax.random.split(rng_t)
        gen_state, levels_t, batch_type = gen.get_batch(
            state=gen_state,
            rng=rng_levels,
            num_levels=num_parallel_envs,
        )
        batch_type_name = gen.batch_type_name(batch_type)
        # level classification
        if log_cycle and classify_level_is_shift is not None:
            shift_levels_mask = jax.vmap(classify_level_is_shift)(levels_t)
            shift_proportion = shift_levels_mask.mean()
            batch_metrics = {'shift_proportion': shift_levels_mask.mean()}
            metrics[f'train-{batch_type_name}'].update(batch_metrics)
            metrics['train-all'].update(batch_metrics)
       

        # collect experience in the levels
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
            env_elapsed_time = time.perf_counter() - env_start_time
            metrics['perf']['env_steps_per_second'] = (
                num_total_env_steps_per_cycle / env_elapsed_time
            )
        step_counts['env-step-all'] += num_total_env_steps_per_cycle
        step_counts[f'env-step-{batch_type_name}'] += num_total_env_steps_per_cycle
        if log_cycle:
            train_metrics = experience.compute_rollout_metrics(
                rollouts=rollouts,
                discount_rate=ppo_gamma,
                benchmark_returns=None,
                benchmark_proxies=None,
            )
            if train_gifs:
                train_metrics['rollouts_gif'] = experience.animate_rollouts(
                    rollouts=rollouts,
                    grid_width=train_gif_grid_width,
                    env=env,
                )
            metrics[f'train-{batch_type_name}'].update(train_metrics)
            metrics['train-all'].update(train_metrics)
        

        # generalised advantage estimation
        advantages = experience.batch_generalised_advantage_estimation(
            rewards=rollouts.transitions.reward,
            dones=rollouts.transitions.done,
            values=rollouts.transitions.value,
            final_values=rollouts.final_value,
            lambda_=ppo_gae_lambda,
            discount_rate=ppo_gamma,
        )
        if train_proxy_critic:
            proxy_advantages = experience.batch_generalised_advantage_estimation(
                rewards=rollouts.transitions.info["proxy_rewards"][proxy_name],
                dones=rollouts.transitions.done,
                values=rollouts.transitions.proxy_value,
                final_values=rollouts.final_proxy_value,
                lambda_=ppo_gae_lambda,
                discount_rate=ppo_gamma,
            )
        else:
            proxy_advantages = None


        # report experience and performance to level generator
        gen_state = gen.update(
            state=gen_state,
            levels=levels_t,
            rollouts=rollouts,
            # shortcut: we did gae already
            advantages=advantages,
            # proxy_advantages=proxy_advantages # TODO!
        )
        if log_cycle:
            ued_metrics = gen.compute_metrics(gen_state)
            metrics['ued'].update(ued_metrics)


        # ppo update network on this data (if curriculum says so, else skip)
        rng_update, rng_t = jax.random.split(rng_t)
        if gen.should_train(batch_type):
            if log_cycle:
                ppo_start_time = time.perf_counter()
            train_state, ppo_metrics = ppo.update(
                rng=rng_update,
                train_state=train_state,
                net_init_state=net_init_state,
                transitions=rollouts.transitions,
                advantages=advantages,
                proxy_advantages=proxy_advantages,
                num_epochs=num_epochs_per_cycle,
                num_minibatches_per_epoch=num_minibatches_per_epoch,
                compute_metrics=log_cycle,
            )
            if log_cycle:
                ppo_elapsed_time = time.perf_counter() - ppo_start_time
                metrics['perf']['ppo_updates_per_second'] = (
                    num_updates_per_cycle / ppo_elapsed_time
                )
            step_counts['env-step-used'] += num_total_env_steps_per_cycle
            step_counts['ppo-update'] += num_updates_per_cycle
            if log_cycle:
                metrics['ppo'].update(ppo_metrics)
        

        # periodic evaluations
        rng_evals, rng_t = jax.random.split(rng_t)
        if log_cycle:
            for eval_name, eval_obj in evals_dict.items():
                rng_eval, rng_evals = jax.random.split(rng_evals)
                eval_start_time = time.perf_counter()
                metrics['eval-'+eval_name] = eval_obj.periodic_eval(
                    rng=rng_eval,
                    step=t,
                    train_state=train_state,
                    net_init_state=net_init_state,
                )
                eval_elapsed_time = time.perf_counter() - eval_start_time
                metrics['perf'][f'eval-{eval_name}-duration'] = eval_elapsed_time

        
        # mark 'after' step counts
        if log_cycle:
            metrics['step'].update({
                f'{key}-after': count for key, count in step_counts.items()
            })


        # define metrics during first cycle
        if first_cycle and wandb_log:
            wandb.define_metric("step/env-step-all-after")
            wandb.define_metric("step/ppo-update-after")
            util.wandb_define_metrics(
                example_metrics=metrics,
                step_metric_prefix_mapping={
                    "train": "step/env-step-all-after",
                    "eval": "step/env-step-all-after",
                    "ued": "step/env-step-all-after",
                    "ppo": "step/ppo-update-after",
                },
            )


        # periodic logging to console/wandb
        if log_cycle and console_log:
            progress.write("\n".join(["="*59, util.dict2str(metrics), "="*59]))
        if log_cycle and wandb_log:
            wandb.log(step=t, data=util.wandb_flatten_and_wrap(metrics))

        
        # periodic checkpointing
        if checkpointing and t % num_cycles_per_checkpoint == 0:
            progress.write("saving checkpoint (wandb will sync at end of run)...")
            checkpoint_manager.save(t, args=ocp.args.PyTreeSave(train_state.params))


        # ending cycle
        progress.update(num_total_env_steps_per_cycle)


    # ending run
    progress.close()
    print("training run complete.")


    if checkpointing:
        print("finishing checkpoints...")
        checkpoint_manager.wait_until_finished()
        checkpoint_manager.close()
        # for some reason I have to manually save these files (I thought
        # they would be automatically saved since I put them in the run dir,
        # and the docs say this, but it doesn't seem to be the case...)
        wandb.save(checkpoint_path + "/**", base_path=wandb.run.dir)


