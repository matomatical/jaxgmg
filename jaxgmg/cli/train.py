"""
Launcher for training runs.
"""

import jax

from jaxgmg.procgen import maze_generation
from jaxgmg.environments import base
from jaxgmg.environments import cheese_in_the_corner
from jaxgmg.environments import keys_and_chests
from jaxgmg.environments import cheese_on_a_dish
from jaxgmg.environments import cheese_on_a_pile
from jaxgmg.baselines import ppo
from jaxgmg.baselines import networks
from jaxgmg.baselines import evals
from jaxgmg.baselines import autocurricula

from jaxgmg import util


@util.wandb_run
def corner(
    # environment config
    env_size: int = 13,
    env_layout: str = 'blocks',
    env_corner_size: int = 1,
    env_terminate_after_corner: bool = False,
    env_level_of_detail: int = 0,           # 0 = bool; 1, 3, 4, or 8 = rgb
    # policy config
    net: str = "impala:lstm",                      # e.g. 'impala:ff', 'impala:lstm'
    # ued config
    ued: str = "plr",                        # 'dr', 'dr-finite', 'plr'
    prob_shift: float = 0.0,
    # for domain randomisation
    num_train_levels: int = 2048,
    # for plr
    plr_buffer_size: int = 2048,
    plr_temperature: float = 0.1,
    plr_staleness_coeff: float = 0.1,
    plr_prob_replay: float = 0.5,
    plr_regret_estimator: str = "proxy_regret",      # "PVL" or "absGAE" (todo "maxMC")
    # PPO hyperparameters
    ppo_lr: float = 0.00005,                # learning rate
    ppo_gamma: float = 0.999,               # discount rate
    ppo_clip_eps: float = 0.1,
    ppo_gae_lambda: float = 0.95,
    ppo_entropy_coeff: float = 0.001,
    ppo_critic_coeff: float = 0.5,
    ppo_max_grad_norm: float = 0.5,
    ppo_lr_annealing: bool = False,
    num_minibatches_per_epoch: int = 8,
    num_epochs_per_cycle: int = 5,
    # training dimensions
    num_total_env_steps: int = 20_000_000,
    num_env_steps_per_cycle: int = 128,
    num_parallel_envs: int = 256,
    # training animation dimensions
    train_gifs: bool = True,
    train_gif_grid_width: int = 16,
    train_gif_level_of_detail: int = 1,
    # evals config
    num_cycles_per_eval: int = 64,
    num_eval_levels: int = 256,
    num_env_steps_per_eval: int = 512,
    # big evals config
    num_cycles_per_big_eval: int = 1024,    # roughly 9M env steps
    eval_gif_grid_width: int = 16,
    eval_gif_level_of_detail: int = 1,      # 1, 3, 4 or 8
    level_splayer: str = 'mouse',           # or 'cheese' or 'cheese-and-mouse'
    # logging
    num_cycles_per_log: int = 64,
    save_files_to: str = "logs/",
    console_log: bool = True,               # whether to log metrics to stdout
    wandb_log: bool = True,                # whether to log metrics to wandb
    wandb_project: str = "test_plr",
    wandb_entity: str = None,               # e.g. 'krueger-lab-cambridge'
    wandb_group: str = None,
    wandb_name: str = None,
    # checkpointing
    checkpointing: bool = True,             # keep checkpoints? (default: yes)
    keep_all_checkpoints: bool = False,     # if so: keep all of them? (no)
    max_num_checkpoints: int = 1,           # if not: keep only latest n (=1)
    num_cycles_per_checkpoint: int = 512,
    # other
    seed: int = 42,
):
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    rng_setup, rng_train = jax.random.split(rng)


    print("setting up environment...")
    env = cheese_in_the_corner.Env(
        obs_level_of_detail=env_level_of_detail,
        penalize_time=False,
        terminate_after_cheese_and_corner=env_terminate_after_corner,
    )


    print(f"generating training level distribution...")
    maze_generator = maze_generation.get_generator_class_from_name(
        name=env_layout,
    )()
    orig_level_generator = cheese_in_the_corner.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        corner_size=env_corner_size,
    )
    shift_level_generator = cheese_in_the_corner.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        corner_size=env_size-2,
    )
    if prob_shift > 0.0:
        train_level_generator = base.MixtureLevelGenerator(
            level_generator1=orig_level_generator,
            level_generator2=shift_level_generator,
            prob_level1=1.0-prob_shift,
        )
    else:
        train_level_generator = orig_level_generator


    print("configuring ued level distributions...")
    rng_train_levels, rng_setup = jax.random.split(rng_setup)
    if ued == "dr":
        gen = autocurricula.InfiniteDomainRandomisation(
            level_generator=train_level_generator,
        )
        gen_state = gen.init()
    elif ued == "dr-finite":
        train_levels = train_level_generator.vsample(
            rng_train_levels,
            num_levels=num_train_levels,
        )
        gen = autocurricula.FiniteDomainRandomisation()
        gen_state = gen.init(
            levels=train_levels,
        )
    elif ued == "plr":
        gen = autocurricula.PrioritisedLevelReplay(
            level_generator=train_level_generator,
            level_metrics=cheese_in_the_corner.LevelMetrics(
                env=env,
                discount_rate=ppo_gamma,
            ),
            level_solver= None,
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
        gen = autocurricula.ParallelRobustPrioritisedLevelReplay(
            level_generator=train_level_generator,
            level_metrics=cheese_in_the_corner.LevelMetrics(
                env=env,
                discount_rate=ppo_gamma,
            ),
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

    
    print(f"setting up agent with architecture {net!r}...")
    # select architecture
    net = networks.get_architecture(net, num_actions=env.num_actions)
    # initialise the network
    rng_model_init, rng = jax.random.split(rng)
    rng_example_level, rng = jax.random.split(rng)
    example_level=train_level_generator.sample(rng_example_level)
    net_init_params, net_init_state = net.init_params_and_state(
        rng=rng_model_init,
        obs_type=env.obs_type(level=example_level),
    )


    print(f"generating some eval levels with baselines...")
    level_solver = cheese_in_the_corner.LevelSolver(
        env=env,
        discount_rate=ppo_gamma,
    )
    # on distribution
    rng_eval_on_levels, rng_setup = jax.random.split(rng_setup)
    eval_on_levels = orig_level_generator.vsample(
        rng_eval_on_levels,
        num_levels=num_eval_levels,
    )
    eval_on_benchmark_returns = level_solver.vmap_level_value(
        level_solver.vmap_solve(eval_on_levels),
        eval_on_levels,
    )
    eval_on_level_set = evals.FixedLevelsEvalWithBenchmarkReturns(
        num_levels=num_eval_levels,
        levels=eval_on_levels,
        benchmarks=eval_on_benchmark_returns,
        num_steps=num_env_steps_per_eval,
        discount_rate=ppo_gamma,
        env=env,
    )


    # off distribution
    rng_eval_off_levels, rng_setup = jax.random.split(rng_setup)
    eval_off_levels = shift_level_generator.vsample(
        rng_eval_off_levels,
        num_levels=num_eval_levels,
    )
    eval_off_benchmark_returns = level_solver.vmap_level_value(
        level_solver.vmap_solve(eval_off_levels),
        eval_off_levels,
    )
    eval_off_level_set = evals.FixedLevelsEvalWithBenchmarkReturns(
        num_levels=num_eval_levels,
        num_steps=num_env_steps_per_eval,
        discount_rate=ppo_gamma,
        levels=eval_off_levels,
        benchmarks=eval_off_benchmark_returns,
        env=env,
    )


    # gif animations from those levels
    eval_on_rollouts = evals.AnimatedRolloutsEval(
        num_levels=num_eval_levels,
        levels=eval_on_levels,
        num_steps=env.max_steps_in_episode,
        gif_grid_width=eval_gif_grid_width,
        gif_level_of_detail=eval_gif_level_of_detail,
        env=env,
    )
    eval_off_rollouts = evals.AnimatedRolloutsEval(
        num_levels=num_eval_levels,
        levels=eval_off_levels,
        num_steps=env.max_steps_in_episode,
        gif_grid_width=eval_gif_grid_width,
        gif_level_of_detail=eval_gif_level_of_detail,
        env=env,
    )


    # splayed eval levels
    match level_splayer:
        case 'mouse':
            splay = cheese_in_the_corner.splay_mouse
        case 'cheese':
            splay = cheese_in_the_corner.splay_cheese
        case 'cheese-and-mouse':
            splay = cheese_in_the_corner.splay_cheese_and_mouse 
        case _:
            raise ValueError(f'unknown level splayer {level_splayer!r}')
    def make_heatmap_evals(level, name):
        splayset = splay(level)
        return {
            (name+"_static_heatmap", num_cycles_per_big_eval):
                evals.ActorCriticHeatmapVisualisationEval(
                    *splayset,
                    env=env,
                ),
            (name+"_rollout_heatmap", num_cycles_per_big_eval):
                evals.RolloutHeatmapVisualisationEval(
                    *splayset,
                    env=env,
                    discount_rate=ppo_gamma,
                    num_steps=num_env_steps_per_eval,
                ),
        }
    heatmap_evals = {
        **make_heatmap_evals(
            name="eval_on_0",
            level=jax.tree.map(lambda x: x[0], eval_on_levels),
        ),
        **make_heatmap_evals(
            name="eval_on_1",
            level=jax.tree.map(lambda x: x[1], eval_on_levels),
        ),
        **make_heatmap_evals(
            name="eval_off_0",
            level=jax.tree.map(lambda x: x[0], eval_off_levels),
        ),
        **make_heatmap_evals(
            name="eval_off_1",
            level=jax.tree.map(lambda x: x[1], eval_off_levels),
        ),
    }


    ppo.run(
        rng=rng_train,
        # environment
        env=env,
        # level distributions
        gen=gen,
        gen_state=gen_state,
        # actor critic network
        net=net,
        net_init_params=net_init_params,
        net_init_state=net_init_state,
        # evals
        evals_dict={
            # small evals
            ('on_dist_levels', num_cycles_per_eval): eval_on_level_set,
            ('off_dist_levels', num_cycles_per_eval): eval_off_level_set,
            # big evals
            ('on_dist_rollouts', num_cycles_per_big_eval): eval_on_rollouts,
            ('off_dist_rollouts', num_cycles_per_big_eval): eval_off_rollouts,
            **heatmap_evals,
        },
        # algorithm
        ppo_lr=ppo_lr,
        ppo_gamma=ppo_gamma,
        ppo_clip_eps=ppo_clip_eps,
        ppo_gae_lambda=ppo_gae_lambda,
        ppo_entropy_coeff=ppo_entropy_coeff,
        ppo_critic_coeff=ppo_critic_coeff,
        ppo_max_grad_norm=ppo_max_grad_norm,
        ppo_lr_annealing=ppo_lr_annealing,
        num_minibatches_per_epoch=num_minibatches_per_epoch,
        num_epochs_per_cycle=num_epochs_per_cycle,
        # training dimensions
        num_total_env_steps=num_total_env_steps,
        num_env_steps_per_cycle=num_env_steps_per_cycle,
        num_parallel_envs=num_parallel_envs,
        # training animation dimensions
        train_gifs=train_gifs,
        train_gif_grid_width=train_gif_grid_width,
        train_gif_level_of_detail=train_gif_level_of_detail,
        # logging
        num_cycles_per_log=num_cycles_per_log,
        save_files_to=save_files_to,
        console_log=console_log,
        wandb_log=wandb_log,
        # checkpointing
        checkpointing=checkpointing,
        keep_all_checkpoints=keep_all_checkpoints,
        max_num_checkpoints=max_num_checkpoints,
        num_cycles_per_checkpoint=num_cycles_per_checkpoint,
    )
    # (the decorator finishes the wandb run for us, so no need to do that)
    print("training run complete.")


@util.wandb_run
def keys(
    # environment config
    env_size: int = 13,
    env_layout: str = 'blocks',
    env_num_keys_min: int = 1,
    env_num_keys_max: int = 3,
    env_num_keys_min_shift: int = 9,
    env_num_keys_max_shift: int = 12,
    env_num_chests_min: int = 12,
    env_num_chests_max: int = 24,
    env_level_of_detail: int = 0,           # 0 = bool; 1, 3, 4, or 8 = rgb
    # policy config
    net: str = "relu",
    # ued config
    ued: str = "dr",                        # 'dr', 'dr-finite', 'plr'
    # for domain randomisation
    num_train_levels: int = 2048,
    # for plr
    plr_buffer_size: int = 2048,
    plr_temperature: float = 0.1,
    plr_staleness_coeff: float = 0.1,
    plr_prob_replay: float = 0.5,
    plr_regret_estimator: str = "PVL",      # "PVL" or "absGAE" (todo "maxMC")
    # PPO hyperparameters
    ppo_lr: float = 0.00005,                # learning rate
    ppo_gamma: float = 0.999,               # discount rate
    ppo_clip_eps: float = 0.1,
    ppo_gae_lambda: float = 0.95,
    ppo_entropy_coeff: float = 0.001,
    ppo_critic_coeff: float = 0.5,
    ppo_max_grad_norm: float = 0.5,
    ppo_lr_annealing: bool = False,
    num_minibatches_per_epoch: int = 8,
    num_epochs_per_cycle: int = 5,
    # training dimensions
    num_total_env_steps: int = 20_000_000,
    num_env_steps_per_cycle: int = 128,
    num_parallel_envs: int = 256,
    # training animation dimensions
    train_gifs: bool = True,
    train_gif_grid_width: int = 16,
    train_gif_level_of_detail: int = 1,
    # evals config
    num_cycles_per_eval: int = 64,
    num_eval_levels: int = 256,
    num_env_steps_per_eval: int = 512,
    # big evals config
    num_cycles_per_big_eval: int = 1024,    # roughly 9M env steps
    eval_gif_grid_width: int = 16,
    eval_gif_level_of_detail: int = 1,      # 1, 3, 4 or 8
    level_splayer: str = 'mouse',           # or 'cheese' or 'cheese-and-mouse'
    # logging
    num_cycles_per_log: int = 64,
    save_files_to: str = "logs/",
    console_log: bool = True,               # whether to log metrics to stdout
    wandb_log: bool = False,                # whether to log metrics to wandb
    wandb_project: str = "test",
    wandb_entity: str = None,
    wandb_group: str = None,
    wandb_name: str = None,
    # checkpointing
    checkpointing: bool = True,             # keep checkpoints? (default: yes)
    keep_all_checkpoints: bool = False,     # if so: keep all of them? (no)
    max_num_checkpoints: int = 1,           # if not: keep only latest n (=1)
    num_cycles_per_checkpoint: int = 512,
    # other
    seed: int = 42,
):
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    rng_setup, rng_train = jax.random.split(rng)


    print("setting up environment...")
    env = keys_and_chests.Env(
        obs_level_of_detail=env_level_of_detail,
        penalize_time=False,
    )


    print(f"generating training level distribution...")
    maze_generator = maze_generation.get_generator_class_from_name(
        name=env_layout,
    )()
    train_level_generator = keys_and_chests.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        num_keys_min=env_num_keys_min,
        num_keys_max=env_num_keys_max,
        num_chests_min=env_num_chests_min,
        num_chests_max=env_num_chests_max,
    )
    
    
    print("configuring ued level distributions...")
    rng_train_levels, rng_setup = jax.random.split(rng_setup)
    if ued == "dr":
        gen = autocurricula.InfiniteDomainRandomisation(
            level_generator=train_level_generator,
        )
        gen_state = gen.init()
    elif ued == "dr-finite":
        train_levels = train_level_generator.vsample(
            rng_train_levels,
            num_levels=num_train_levels,
        )
        gen = autocurricula.FiniteDomainRandomisation()
        gen_state = gen.init(
            levels=train_levels,
        )
    elif ued == "plr":
        gen = autocurricula.PrioritisedLevelReplay(
            level_generator=train_level_generator,
            level_metrics=None,
            # level_metrics=keys_and_chests.LevelMetrics( # TODO: define
            #     env=env,
            #     discount_rate=ppo_gamma,
            # ),
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
        gen = autocurricula.ParallelRobustPrioritisedLevelReplay(
            level_generator=train_level_generator,
            level_metrics=cheese_in_the_corner.LevelMetrics(
                env=env,
                discount_rate=ppo_gamma,
            ),
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


    print(f"setting up agent with architecture {net!r}...")
    # select architecture
    net = networks.get_architecture(net, num_actions=env.num_actions)
    # initialise the network
    rng_model_init, rng = jax.random.split(rng)
    rng_example_level, rng = jax.random.split(rng)
    example_level=train_level_generator.sample(rng_example_level)
    net_init_params, net_init_state = net.init_params_and_state(
        rng=rng_model_init,
        obs_type=env.obs_type(level=example_level),
    )


    print(f"generating some eval levels with baselines...")
    # on distribution
    rng_eval_on_levels, rng_setup = jax.random.split(rng_setup)
    eval_on_levels = train_level_generator.vsample(
        rng_eval_on_levels,
        num_levels=num_eval_levels,
    )
    eval_on_level_set = evals.FixedLevelsEval(
        num_levels=num_eval_levels,
        levels=eval_on_levels,
        num_steps=num_env_steps_per_eval,
        discount_rate=ppo_gamma,
        env=env,
    )


    # off distribution
    shift_level_generator = keys_and_chests.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        num_keys_min=env_num_keys_min_shift,
        num_keys_max=env_num_keys_max_shift,
        num_chests_min=env_num_chests_min,
        num_chests_max=env_num_chests_max,
    )
    rng_eval_off_levels, rng_setup = jax.random.split(rng_setup)
    eval_off_levels = shift_level_generator.vsample(
        rng_eval_off_levels,
        num_levels=num_eval_levels,
    )
    eval_off_level_set = evals.FixedLevelsEval(
        num_levels=num_eval_levels,
        num_steps=num_env_steps_per_eval,
        discount_rate=ppo_gamma,
        levels=eval_off_levels,
        env=env,
    )


    # gif animations from those levels
    eval_on_animation = evals.AnimatedRolloutsEval(
        num_levels=num_eval_levels,
        levels=eval_on_levels,
        num_steps=env.max_steps_in_episode,
        gif_grid_width=eval_gif_grid_width,
        gif_level_of_detail=eval_gif_level_of_detail,
        env=env,
    )
    eval_off_animation = evals.AnimatedRolloutsEval(
        num_levels=num_eval_levels,
        levels=eval_off_levels,
        num_steps=env.max_steps_in_episode,
        gif_grid_width=eval_gif_grid_width,
        gif_level_of_detail=eval_gif_level_of_detail,
        env=env,
    )


    ppo.run(
        rng=rng_train,
        # environment
        env=env,
        # level distributions
        gen=gen,
        gen_state=gen_state,
        # actor critic network
        net=net,
        net_init_params=net_init_params,
        net_init_state=net_init_state,
        # evals
        evals_dict={
            ('on_dist_levels', num_cycles_per_eval): eval_on_level_set,
            ('off_dist_levels', num_cycles_per_eval): eval_off_level_set,
            ('on_dist_animations', num_cycles_per_big_eval): eval_on_animation,
            ('off_dist_animations', num_cycles_per_big_eval): eval_off_animation,
        },
        # ppo algorithm parameters
        ppo_lr=ppo_lr,
        ppo_gamma=ppo_gamma,
        ppo_clip_eps=ppo_clip_eps,
        ppo_gae_lambda=ppo_gae_lambda,
        ppo_entropy_coeff=ppo_entropy_coeff,
        ppo_critic_coeff=ppo_critic_coeff,
        ppo_max_grad_norm=ppo_max_grad_norm,
        ppo_lr_annealing=ppo_lr_annealing,
        num_minibatches_per_epoch=num_minibatches_per_epoch,
        num_epochs_per_cycle=num_epochs_per_cycle,
        # training dimensions
        num_total_env_steps=num_total_env_steps,
        num_env_steps_per_cycle=num_env_steps_per_cycle,
        num_parallel_envs=num_parallel_envs,
        # training animation dimensions
        train_gifs=train_gifs,
        train_gif_grid_width=train_gif_grid_width,
        train_gif_level_of_detail=train_gif_level_of_detail,
        # logging
        num_cycles_per_log=num_cycles_per_log,
        save_files_to=save_files_to,
        console_log=console_log,
        wandb_log=wandb_log,
        # checkpointing
        checkpointing=checkpointing,
        keep_all_checkpoints=keep_all_checkpoints,
        max_num_checkpoints=max_num_checkpoints,
        num_cycles_per_checkpoint=num_cycles_per_checkpoint,
    )
    # (the decorator finishes the wandb run for us, so no need to do that)
    print("training run complete.")


@util.wandb_run
def dish(
    # environment config
    env_size: int = 13,
    env_layout: str = 'blocks',
    env_terminate_after_dish: bool = False,
    max_cheese_radius: int = 0,
    max_cheese_radius_shift: int = 12,
    env_level_of_detail: int = 0,           # 0 = bool; 1, 3, 4, or 8 = rgb
    # policy config
    net: str = "impala:lstm",                      # e.g. 'impala:ff', 'impala:lstm'
    # ued config
    ued: str = "plr",                        # 'dr', 'dr-finite', 'plr'
    prob_shift: float = 0.0,
    # for domain randomisation
    num_train_levels: int = 2048,
    # for plr
    plr_buffer_size: int = 2048,
    plr_temperature: float = 0.1,
    plr_staleness_coeff: float = 0.1,
    plr_prob_replay: float = 0.5,
    plr_regret_estimator: str = "PVL",      # "PVL" or "absGAE" (todo "maxMC")
    # PPO hyperparameters
    ppo_lr: float = 0.00005,                # learning rate
    ppo_gamma: float = 0.999,               # discount rate
    ppo_clip_eps: float = 0.1,
    ppo_gae_lambda: float = 0.95,
    ppo_entropy_coeff: float = 0.001,
    ppo_critic_coeff: float = 0.5,
    ppo_max_grad_norm: float = 0.5,
    ppo_lr_annealing: bool = False,
    num_minibatches_per_epoch: int = 4,
    num_epochs_per_cycle: int = 5,
    # training dimensions
    num_total_env_steps: int = 20_000_000,
    num_env_steps_per_cycle: int = 128,
    num_parallel_envs: int = 256,
    # training animation dimensions
    train_gifs: bool = True,
    train_gif_grid_width: int = 16,
    train_gif_level_of_detail: int = 1,
    # evals config
    num_cycles_per_eval: int = 64,
    num_eval_levels: int = 256,
    num_env_steps_per_eval: int = 512,
    # big evals config
    num_cycles_per_big_eval: int = 1024,    # roughly 9M env steps
    eval_gif_grid_width: int = 16,
    eval_gif_level_of_detail: int = 1,      # 1, 3, 4 or 8
    level_splayer: str = 'mouse',           # or 'cheese' or 'cheese-and-mouse'
    # logging
    num_cycles_per_log: int = 64,
    save_files_to: str = "logs/",
    console_log: bool = True,               # whether to log metrics to stdout
    wandb_log: bool = True,                # whether to log metrics to wandb
    wandb_project: str = "plr_dish",
    wandb_entity: str = None,
    wandb_group: str = None,
    wandb_name: str = None,
    # checkpointing
    checkpointing: bool = True,             # keep checkpoints? (default: yes)
    keep_all_checkpoints: bool = False,     # if so: keep all of them? (no)
    max_num_checkpoints: int = 1,           # if not: keep only latest n (=1)
    num_cycles_per_checkpoint: int = 512,
    # other
    seed: int = 42,
):
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    rng_setup, rng_train = jax.random.split(rng)


    print("setting up environment...")
    env = cheese_on_a_dish.Env(
        obs_level_of_detail=env_level_of_detail,
        penalize_time=False,
        terminate_after_cheese_and_dish= env_terminate_after_dish,
    )


    print(f"generating training level distribution...")
    maze_generator = maze_generation.get_generator_class_from_name(
        name=env_layout,
    )()
    orig_level_generator = cheese_on_a_dish.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        max_cheese_radius=max_cheese_radius,  
    )
    
    shift_level_generator = cheese_on_a_dish.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        max_cheese_radius=max_cheese_radius_shift,  
    )

    if prob_shift > 0.0:
        train_level_generator = base.MixtureLevelGenerator(
            level_generator1=orig_level_generator,
            level_generator2=shift_level_generator,
            prob_level1=1.0-prob_shift,
        )
    else:
        train_level_generator = orig_level_generator
    
    
    print("configuring ued level distributions...")
    rng_train_levels, rng_setup = jax.random.split(rng_setup)
    if ued == "dr":
        gen = autocurricula.InfiniteDomainRandomisation(
            level_generator=train_level_generator,
        )
        gen_state = gen.init()
    elif ued == "dr-finite":
        train_levels = train_level_generator.vsample(
            rng_train_levels,
            num_levels=num_train_levels,
        )
        gen = autocurricula.FiniteDomainRandomisation()
        gen_state = gen.init(
            levels=train_levels,
        )
    elif ued == "plr":
        gen = autocurricula.PrioritisedLevelReplay(
            level_generator=train_level_generator,
            level_metrics = cheese_on_a_dish.LevelMetrics(
                env=env,
                discount_rate=ppo_gamma,
            ),
            level_solver= None,
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
        gen = autocurricula.ParallelRobustPrioritisedLevelReplay(
            level_generator=train_level_generator,
            level_metrics = cheese_on_a_dish.LevelMetrics(
                env=env,
                discount_rate=ppo_gamma,
            ),
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


    print(f"setting up agent with architecture {net!r}...")
    # select architecture
    net = networks.get_architecture(net, num_actions=env.num_actions)
    # initialise the network
    rng_model_init, rng = jax.random.split(rng)
    rng_example_level, rng = jax.random.split(rng)
    example_level=train_level_generator.sample(rng_example_level)
    net_init_params, net_init_state = net.init_params_and_state(
        rng=rng_model_init,
        obs_type=env.obs_type(level=example_level),
    )


    print(f"generating some eval levels with baselines...")
    # on distribution
    rng_eval_on_levels, rng_setup = jax.random.split(rng_setup)
    eval_on_levels = train_level_generator.vsample(
        rng_eval_on_levels,
        num_levels=num_eval_levels,
    )
    eval_on_level_set = evals.FixedLevelsEval(
        num_levels=num_eval_levels,
        levels=eval_on_levels,
        num_steps=num_env_steps_per_eval,
        discount_rate=ppo_gamma,
        env=env,
    )


    # off distribution
    shift_level_generator = cheese_on_a_dish.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        max_cheese_radius=max_cheese_radius_shift,
    )
    rng_eval_off_levels, rng_setup = jax.random.split(rng_setup)
    eval_off_levels = shift_level_generator.vsample(
        rng_eval_off_levels,
        num_levels=num_eval_levels,
    )
    eval_off_level_set = evals.FixedLevelsEval(
        num_levels=num_eval_levels,
        num_steps=num_env_steps_per_eval,
        discount_rate=ppo_gamma,
        levels=eval_off_levels,
        env=env,
    )


    # gif animations from those levels
    eval_on_animation = evals.AnimatedRolloutsEval(
        num_levels=num_eval_levels,
        levels=eval_on_levels,
        num_steps=env.max_steps_in_episode,
        gif_grid_width=eval_gif_grid_width,
        gif_level_of_detail=eval_gif_level_of_detail,
        env=env,
    )
    eval_off_animation = evals.AnimatedRolloutsEval(
        num_levels=num_eval_levels,
        levels=eval_off_levels,
        num_steps=env.max_steps_in_episode,
        gif_grid_width=eval_gif_grid_width,
        gif_level_of_detail=eval_gif_level_of_detail,
        env=env,
    )


    ppo.run(
        rng=rng_train,
        # environment
        env=env,
        # level distributions
        gen=gen,
        gen_state=gen_state,
        # actor critic network
        net=net,
        net_init_params=net_init_params,
        net_init_state=net_init_state,
        # evals
        evals_dict={
            ('on_dist_levels', num_cycles_per_eval): eval_on_level_set,
            ('off_dist_levels', num_cycles_per_eval): eval_off_level_set,
            ('on_dist_animations', num_cycles_per_big_eval): eval_on_animation,
            ('off_dist_animations', num_cycles_per_big_eval): eval_off_animation,
        },
        # ppo algorithm parameters
        ppo_lr=ppo_lr,
        ppo_gamma=ppo_gamma,
        ppo_clip_eps=ppo_clip_eps,
        ppo_gae_lambda=ppo_gae_lambda,
        ppo_entropy_coeff=ppo_entropy_coeff,
        ppo_critic_coeff=ppo_critic_coeff,
        ppo_max_grad_norm=ppo_max_grad_norm,
        ppo_lr_annealing=ppo_lr_annealing,
        num_minibatches_per_epoch=num_minibatches_per_epoch,
        num_epochs_per_cycle=num_epochs_per_cycle,
        # training dimensions
        num_total_env_steps=num_total_env_steps,
        num_env_steps_per_cycle=num_env_steps_per_cycle,
        num_parallel_envs=num_parallel_envs,
        # training animation dimensions
        train_gifs=train_gifs,
        train_gif_grid_width=train_gif_grid_width,
        train_gif_level_of_detail=train_gif_level_of_detail,
        # logging
        num_cycles_per_log=num_cycles_per_log,
        save_files_to=save_files_to,
        console_log=console_log,
        wandb_log=wandb_log,
        # checkpointing
        checkpointing=checkpointing,
        keep_all_checkpoints=keep_all_checkpoints,
        max_num_checkpoints=max_num_checkpoints,
        num_cycles_per_checkpoint=num_cycles_per_checkpoint,
    )
    # (the decorator finishes the wandb run for us, so no need to do that)
    print("training run complete.")


@util.wandb_run
def pile(
    # environment config
    env_size: int = 13,
    env_layout: str = 'blocks',
    env_terminate_after_pile: bool = False,
    split_elements_train:int = 0, # how many objects go on the cheese - not important cuz we want to train with both in the same position
    split_elements_shift:int = 0,
    max_cheese_radius: int = 0,
    max_cheese_radius_shift: int = 12,
    max_dish_radius: int = 0, # this is not relevant and can be ignored for now ( they may come useful in next implementations)
    max_dish_radius_shift: int= 0, # this is not relevant and can be ignored for now ( they may come useful in next implementations)
    env_level_of_detail: int = 0,           # 0 = bool; 1, 3, 4, or 8 = rgb
    #cheese_location: Tuple[int,int] = (1,1) , # default: [1,1], otherwise define a fixed location where you would like your cheese to be placed
    # policy config
    net: str = "impala:lstm",               # e.g. 'impala:ff', 'impala:lstm'
    # ued config
    ued: str = "dr",                        # 'dr', 'dr-finite', 'plr'
    prob_shift: float = 0.0,
    # for domain randomisation
    num_train_levels: int = 2048,
    # for plr
    plr_buffer_size: int = 2048,
    plr_temperature: float = 0.1,
    plr_staleness_coeff: float = 0.1,
    plr_prob_replay: float = 0.5,
    plr_regret_estimator: str = "PVL",      # "PVL" or "absGAE" (todo "maxMC")
    # PPO hyperparameters
    ppo_lr: float = 0.00005,                 # learning rate
    ppo_gamma: float = 0.999,               # discount rate
    ppo_clip_eps: float = 0.1,
    ppo_gae_lambda: float = 0.95,
    ppo_entropy_coeff: float = 0.001,
    ppo_critic_coeff: float = 0.5,
    ppo_max_grad_norm: float = 0.5,
    ppo_lr_annealing: bool = False,
    num_minibatches_per_epoch: int = 4,
    num_epochs_per_cycle: int = 5,
    # training dimensions
    num_total_env_steps: int = 300_000_000,
    num_env_steps_per_cycle: int = 128,
    num_parallel_envs: int = 256,
    fixed_train_levels: bool = False,
    # training animation dimensions
    train_gifs: bool = False,
    train_gif_grid_width: int = 8,
    train_gif_level_of_detail: int = 1,
    # evals config
    num_cycles_per_eval: int = 64,
    num_eval_levels: int = 256,
    num_env_steps_per_eval: int = 512,
    # big evals config
    num_cycles_per_big_eval: int = 1024,    # roughly 9M env steps
    eval_gif_grid_width: int = 16,
    eval_gif_level_of_detail: int = 1,      # 1, 3, 4 or 8
    level_splayer: str = 'mouse',           # or 'cheese' or 'cheese-and-mouse'
    # logging
    num_cycles_per_log: int = 64,
    save_files_to: str = "logs/",
    console_log: bool = True,               # whether to log metrics to stdout
    wandb_log: bool = True,                # whether to log metrics to wandb
    wandb_entity: str = None,
    wandb_project: str = "proxy_test_multiple_dish_final",
    wandb_group: str = None,
    wandb_name: str = None,
    # checkpointing
    checkpointing: bool = True,             # keep checkpoints? (default: yes)
    keep_all_checkpoints: bool = False,     # if so: keep all of them? (no)
    max_num_checkpoints: int = 1,           # if not: keep only latest n (=1)
    num_cycles_per_checkpoint: int = 512,
    # other
    seed: int = 42,
):
    util.print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    rng_setup, rng_train = jax.random.split(rng)


    print("setting up environment...")
    env = cheese_on_a_pile.Env(
        obs_level_of_detail=env_level_of_detail,
        penalize_time=False,
        terminate_after_cheese_and_dish= env_terminate_after_pile,
        split_object_firstgroup = split_elements_train, # check this, if it should be split_elements_train or split_elements_shift or 0
    )


    print(f"generating training level distribution...")
    maze_generator = maze_generation.get_generator_class_from_name(
        name=env_layout,
    )()
    orig_level_generator = cheese_on_a_pile.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        max_cheese_radius=max_cheese_radius,
        max_dish_radius = max_dish_radius,
        split_elements =  split_elements_train,
    )
    
    shift_level_generator = cheese_on_a_pile.LevelGenerator(
        height=env_size,
        width=env_size,
        maze_generator=maze_generator,
        max_cheese_radius=max_cheese_radius_shift,
        max_dish_radius = max_dish_radius_shift,
        split_elements = split_elements_shift,
    )

    if prob_shift > 0.0:
        train_level_generator = base.MixtureLevelGenerator(
            level_generator1=orig_level_generator,
            level_generator2=shift_level_generator,
            prob_level1=1.0-prob_shift,
        )
    else:
        train_level_generator = orig_level_generator
    
       
    print("configuring ued level distributions...")
    rng_train_levels, rng_setup = jax.random.split(rng_setup)
    if ued == "dr":
        gen = autocurricula.InfiniteDomainRandomisation(
            level_generator=train_level_generator,
        )
        gen_state = gen.init()
    elif ued == "dr-finite":
        train_levels = train_level_generator.vsample(
            rng_train_levels,
            num_levels=num_train_levels,
        )
        gen = autocurricula.FiniteDomainRandomisation()
        gen_state = gen.init(
            levels=train_levels,
        )
    elif ued == "plr":
        gen = autocurricula.PrioritisedLevelReplay(
            level_generator=train_level_generator,
            level_metrics = cheese_on_a_pile.LevelMetrics(
                env=env,
                discount_rate=ppo_gamma,
            ),
            level_solver= None,
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
        gen = autocurricula.ParallelRobustPrioritisedLevelReplay(
            level_generator=train_level_generator,
            level_metrics = cheese_on_a_pile.LevelMetrics(
                env=env,
                discount_rate=ppo_gamma,
            ),
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
    
    print(f"setting up agent with architecture {net!r}...")
    # select architecture
    net = networks.get_architecture(net, num_actions=env.num_actions)
    # initialise the network
    rng_model_init, rng = jax.random.split(rng)
    rng_example_level, rng = jax.random.split(rng)
    example_level=train_level_generator.sample(rng_example_level)
    net_init_params, net_init_state = net.init_params_and_state(
        rng=rng_model_init,
        obs_type=env.obs_type(level=example_level),
    )


    print(f"generating some eval levels with baselines...")
    # on distribution
    rng_eval_on_levels, rng_setup = jax.random.split(rng_setup)
    eval_on_levels = train_level_generator.vsample(
        rng_eval_on_levels,
        num_levels=num_eval_levels,
    )
    eval_on_level_set = evals.FixedLevelsEval(
        num_levels=num_eval_levels,
        levels=eval_on_levels,
        num_steps=num_env_steps_per_eval,
        discount_rate=ppo_gamma,
        env=env,
    )

    rng_eval_off_levels, rng_setup = jax.random.split(rng_setup)
    eval_off_levels = shift_level_generator.vsample(
        rng_eval_off_levels,
        num_levels=num_eval_levels,
    )
    eval_off_level_set = evals.FixedLevelsEval(
        num_levels=num_eval_levels,
        num_steps=num_env_steps_per_eval,
        discount_rate=ppo_gamma,
        levels=eval_off_levels,
        env=env,
    )
    
    # gif animations from those levels

    eval_on_animation = evals.AnimatedRolloutsEval(
        num_levels=num_eval_levels,
        levels=eval_on_levels,
        num_steps=env.max_steps_in_episode,
        gif_grid_width=eval_gif_grid_width,
        gif_level_of_detail=eval_gif_level_of_detail,
        env=env,
    )
    eval_off_animation = evals.AnimatedRolloutsEval(
        num_levels=num_eval_levels,
        levels=eval_off_levels,
        num_steps=env.max_steps_in_episode,
        gif_grid_width=eval_gif_grid_width,
        gif_level_of_detail=eval_gif_level_of_detail,
        env=env,
    )

    ppo.run(
        rng=rng_train,
        # environment
        env=env,
        # level distributions
        gen=gen,
        gen_state=gen_state,
        # actor critic network
        net=net,
        net_init_params=net_init_params,
        net_init_state=net_init_state,
        # evals
        evals_dict={
            ('on_dist_levels', num_cycles_per_eval): eval_on_level_set,
            ('off_dist_levels', num_cycles_per_eval): eval_off_level_set,
            ('on_dist_animations', num_cycles_per_big_eval): eval_on_animation,
            ('off_dist_animations', num_cycles_per_big_eval): eval_off_animation,
        },
        # ppo algorithm parameters
        ppo_lr=ppo_lr,
        ppo_gamma=ppo_gamma,
        ppo_clip_eps=ppo_clip_eps,
        ppo_gae_lambda=ppo_gae_lambda,
        ppo_entropy_coeff=ppo_entropy_coeff,
        ppo_critic_coeff=ppo_critic_coeff,
        ppo_max_grad_norm=ppo_max_grad_norm,
        ppo_lr_annealing=ppo_lr_annealing,
        num_minibatches_per_epoch=num_minibatches_per_epoch,
        num_epochs_per_cycle=num_epochs_per_cycle,
        # training dimensions
        num_total_env_steps=num_total_env_steps,
        num_env_steps_per_cycle=num_env_steps_per_cycle,
        num_parallel_envs=num_parallel_envs,
        # training animation dimensions
        train_gifs=train_gifs,
        train_gif_grid_width=train_gif_grid_width,
        train_gif_level_of_detail=train_gif_level_of_detail,
        # logging
        num_cycles_per_log=num_cycles_per_log,
        save_files_to=save_files_to,
        console_log=console_log,
        wandb_log=wandb_log,
        # checkpointing
        checkpointing=checkpointing,
        keep_all_checkpoints=keep_all_checkpoints,
        max_num_checkpoints=max_num_checkpoints,
        num_cycles_per_checkpoint=num_cycles_per_checkpoint,
    )
    # (the decorator finishes the wandb run for us, so no need to do that)
    print("training run complete.")