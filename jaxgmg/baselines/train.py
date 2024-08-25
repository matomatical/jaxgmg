"""
Proximal policy optimisation for a given network, environment, and set of
training/eval levels.
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
from jaxgmg.environments.base import MixtureLevelGenerator
from jaxgmg.baselines import networks
from jaxgmg.baselines import experience
from jaxgmg.baselines import evals
from jaxgmg.baselines.ppo import ProximalPolicyOptimisation
from jaxgmg.baselines.autocurricula import domain_randomisation
from jaxgmg.baselines.autocurricula import finite_domain_randomisation
from jaxgmg.baselines.autocurricula import prioritised_level_replay
from jaxgmg.baselines.autocurricula import parallel_prioritised_level_replay

# types and abstract base classes used for type annotations
from typing import Any, Callable
from chex import Array, PRNGKey
from jaxgmg.environments.base import EnvState, Env, Level, Observation
from jaxgmg.environments.base import LevelGenerator
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
    orig_level_generator: LevelGenerator,
    shift_level_generator: LevelGenerator,
    level_solver: LevelSolver | None,
    level_metrics: LevelMetrics | None,
    fixed_eval_levels: dict[str, Level],
    heatmap_splayer_fn: Callable | None,
    # actor critic policy config
    net_cnn_type: str,
    net_rnn_type: str,
    # ued config
    ued: str,
    prob_shift: float,
    num_train_levels: int,
    plr_buffer_size: int,
    plr_temperature: float,
    plr_staleness_coeff: float,
    plr_prob_replay: float,
    plr_regret_estimator: str,
    # PPO config
    ppo_lr: float,
    ppo_gamma: float,
    ppo_clip_eps: float,
    ppo_gae_lambda: float,
    ppo_entropy_coeff: float,
    ppo_critic_coeff: float,
    ppo_max_grad_norm: float,
    ppo_lr_annealing: bool,
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


    # TODO: Would also be a good idea for this program to initialise and
    # manage the WANDB run. But that means going back to config hell?


    print(f"seeding random number generator with {seed=}...")
    rng = jax.random.PRNGKey(seed=seed)
    rng_setup, rng_train = jax.random.split(rng)


    print(f"mixing training level distributions with {prob_shift=}...")
    if prob_shift > 0.0:
        print("  creating mixture generator...")
        train_level_generator = MixtureLevelGenerator(
            level_generator1=orig_level_generator,
            level_generator2=shift_level_generator,
            prob_level1=1.0-prob_shift,
        )
    else:
        print("  proceeding with original generator unmixed.")
        train_level_generator = orig_level_generator
    
    
    print(f"configuring curriculum with {ued=}...")
    rng_train_levels, rng_setup = jax.random.split(rng_setup)
    if ued == "dr":
        gen = domain_randomisation.CurriculumGenerator(
            level_generator=train_level_generator,
        )
        gen_state = gen.init()
    elif ued == "dr-finite":
        train_levels = train_level_generator.vsample(
            rng_train_levels,
            num_levels=num_train_levels,
        )
        gen = finite_domain_randomisation.CurriculumGenerator()
        gen_state = gen.init(
            levels=train_levels,
        )
    elif ued == "plr":
        gen = prioritised_level_replay.CurriculumGenerator(
            level_generator=train_level_generator,
            level_metrics=level_metrics,
            buffer_size=plr_buffer_size,
            temperature=plr_temperature,
            staleness_coeff=plr_staleness_coeff,
            prob_replay=plr_prob_replay,
            regret_estimator=plr_regret_estimator,
        )
        gen_state = gen.init(
            rng=rng_train_levels,
            batch_size_hint=num_parallel_envs,
        )
    elif ued == "plr-parallel":
        gen = parallel_prioritised_level_replay.CurriculumGenerator(
            level_generator=train_level_generator,
            level_metrics=level_metrics,
            buffer_size=plr_buffer_size,
            temperature=plr_temperature,
            staleness_coeff=plr_staleness_coeff,
            regret_estimator=plr_regret_estimator,
        )
        gen_state = gen.init(
            rng=rng_train_levels,
            batch_size_hint=num_parallel_envs,
        )
    else:
        raise ValueError(f"unknown UED algorithm: {ued!r}")


    print("configuring periodic evals...")
    rng_evals, rng_setup = jax.random.split(rng_setup)
    evals_dict = configure_evals(
        rng=rng_evals,
        env=env,
        orig_level_generator=orig_level_generator,
        shift_level_generator=shift_level_generator,
        level_solver=level_solver,
        fixed_eval_levels=fixed_eval_levels,
        heatmap_splayer_fn=heatmap_splayer_fn,
        discount_rate=ppo_gamma,
        num_cycles_per_eval=num_cycles_per_eval,
        num_eval_levels=num_eval_levels,
        num_env_steps_per_eval=num_env_steps_per_eval,
        num_cycles_per_big_eval=num_cycles_per_big_eval,
        eval_gif_grid_width=eval_gif_grid_width,
    )


    print("configuring actor critic network...")
    # select architecture
    print(f"  {net_cnn_type=}")
    print(f"  {net_rnn_type=}")
    net = networks.Impala(
        num_actions=env.num_actions,
        cnn_type=net_cnn_type,
        rnn_type=net_rnn_type,
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
        do_backprop_thru_time=net.is_recurrent,
    )


    # define wandb metrics at the end of the first loop (wandb sucks)
    metrics_undefined = True


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
        rng_t, rng_train = jax.random.split(rng_train)
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
        train_state, ppo_metrics = ppo.update(
            rng=rng_update,
            train_state=train_state,
            net_init_state=net_init_state,
            transitions=valid_rollouts.transitions,
            advantages=valid_advantages,
            num_epochs=num_epochs_per_cycle,
            num_minibatches_per_epoch=num_minibatches_per_epoch,
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
    print("training run complete.")


    if checkpointing:
        print("finishing checkpoints...")
        checkpoint_manager.wait_until_finished()
        checkpoint_manager.close()
        # for some reason I have to manually save these files (I thought
        # they would be automatically saved since I put them in the run dir,
        # and the docs say this, but it doesn't seem to be the case...)
        wandb.save(checkpoint_path + "/**", base_path=wandb.run.dir)


def configure_evals(
    rng: PRNGKey,
    env: Env,
    orig_level_generator: LevelGenerator,
    shift_level_generator: LevelGenerator | None,
    level_solver: LevelSolver | None,
    fixed_eval_levels: dict[str, Level],
    heatmap_splayer_fn: Callable | None,
    discount_rate: float,
    num_cycles_per_eval: int,
    num_eval_levels: int,
    num_env_steps_per_eval: int,
    num_cycles_per_big_eval: int,
    eval_gif_grid_width: int,
) -> dict[(str, int), Eval]:
    """
    Various config options allow 
    """
    evals_dict = {}


    print("  configuring eval levels from orig level generator...")
    rng_eval_on_levels, rng = jax.random.split(rng)
    eval_on_levels = orig_level_generator.vsample(
        rng_eval_on_levels,
        num_levels=num_eval_levels,
    )
    if level_solver is not None:
        print("    also solving them...")
        eval_on_benchmark_returns = level_solver.vmap_level_value(
            level_solver.vmap_solve(eval_on_levels),
            eval_on_levels,
        )
        eval_on_level_set = evals.FixedLevelsEvalWithBenchmarkReturns(
            num_levels=num_eval_levels,
            levels=eval_on_levels,
            benchmarks=eval_on_benchmark_returns,
            num_steps=num_env_steps_per_eval,
            discount_rate=discount_rate,
            env=env,
        )
    else:
        eval_on_level_set = evals.FixedLevelsEval(
            num_levels=num_eval_levels,
            levels=eval_on_levels,
            num_steps=num_env_steps_per_eval,
            discount_rate=discount_rate,
            env=env,
        )
    evals_dict[('on_dist_levels', num_cycles_per_eval)] = eval_on_level_set

    
    print("  configuring eval levels from shift level generator...")
    if shift_level_generator is not None:
        rng_eval_off_levels, rng = jax.random.split(rng)
        eval_off_levels = shift_level_generator.vsample(
            rng_eval_off_levels,
            num_levels=num_eval_levels,
        )
        if level_solver is not None:
            print("    also solving them...")
            eval_off_benchmark_returns = level_solver.vmap_level_value(
                level_solver.vmap_solve(eval_off_levels),
                eval_off_levels,
            )
            eval_off_level_set = evals.FixedLevelsEvalWithBenchmarkReturns(
                num_levels=num_eval_levels,
                num_steps=num_env_steps_per_eval,
                discount_rate=discount_rate,
                levels=eval_off_levels,
                benchmarks=eval_off_benchmark_returns,
                env=env,
            )
        else:
            eval_off_level_set = evals.FixedLevelsEval(
                num_levels=num_eval_levels,
                num_steps=num_env_steps_per_eval,
                discount_rate=discount_rate,
                levels=eval_off_levels,
                env=env,
            )
        evals_dict[('off_dist_levels', num_cycles_per_eval)] = eval_off_level_set

    
    print("  configuring rollout recorders for orig/shift levels...")
    # orig
    eval_on_rollouts = evals.AnimatedRolloutsEval(
        num_levels=num_eval_levels,
        levels=eval_on_levels,
        num_steps=env.max_steps_in_episode,
        gif_grid_width=eval_gif_grid_width,
        env=env,
    )
    evals_dict[('on_dist_rollouts', num_cycles_per_big_eval)] = eval_on_rollouts
    # shift (optional)
    if shift_level_generator is not None:
        eval_off_rollouts = evals.AnimatedRolloutsEval(
            num_levels=num_eval_levels,
            levels=eval_off_levels,
            num_steps=env.max_steps_in_episode,
            gif_grid_width=eval_gif_grid_width,
            env=env,
        )
        evals_dict[('off_dist_rollouts', num_cycles_per_big_eval)] = eval_off_rollouts
    

    print(f"  configuring evals for {len(fixed_eval_levels)} levels...")
    for level_name, level in fixed_eval_levels.items():
        eval_name = 'fixed/' + level_name
        eval_period = num_cycles_per_eval
        evals_dict[(eval_name, eval_period)] = evals.SingleLevelEval(
            num_steps=num_env_steps_per_eval,
            discount_rate=discount_rate,
            level=level,
            env=env,
        )


    print(f"  configuring heatmap evals for {len(fixed_eval_levels)} levels...")
    for level_name, level in fixed_eval_levels.items():
        if heatmap_splayer_fn is None:
            print("    n/a: no heatmap splayer provided.")
            break
        splayset = heatmap_splayer_fn(level)
        eval_name_static = 'heatmaps/static/' + level_name
        eval_name_rollout = 'heatmaps/rollout/' + level_name
        eval_period = num_cycles_per_big_eval
        evals_dict[(eval_name_static, eval_period)] = (
            evals.ActorCriticHeatmapVisualisationEval(
                *splayset,
                env=env,
            )
        )
        evals_dict[(eval_name_rollout, eval_period)] = (
            evals.RolloutHeatmapVisualisationEval(
                *splayset,
                env=env,
                discount_rate=discount_rate,
                num_steps=num_env_steps_per_eval,
            )
        )

    
    # done!
    return evals_dict


