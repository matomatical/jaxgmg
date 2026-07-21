"""
Load a checkpoint and run some evals on a checkpoint.
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

# types and abstract base classes used for type annotations
from typing import Any, Callable
from chex import Array, PRNGKey
from jaxgmg.environments.base import EnvState, Env, Level, Observation
from jaxgmg.environments.base import LevelGenerator
from jaxgmg.environments.base import LevelSolver
from jaxgmg.baselines.evals import Eval


# # # 
# evals entry point


def run(
    seed: int,
    # checkpoint to load
    checkpoint_folder: str,
    checkpoint_number: int,
    # environment-specific stuff
    env: Env,
    level_solver: LevelSolver | None,
    eval_level_generators: dict[str, LevelGenerator],
    fixed_eval_levels: dict[str, Level],
    arbitrary_level_generator: LevelGenerator,
    heatmap_splayer_fn: Callable | None,
    # stuff i forgot
    ppo_gamma: float,
    # actor critic policy config
    net_cnn_type: str,
    net_rnn_type: str,
    net_width: int,
    # logging and evals config
    evals_num_env_steps: int,
    evals_num_levels: int,
    gif_grid_width: int,
):
    print(f"seeding random number generator with {seed=}...")
    rng = jax.random.PRNGKey(seed=seed)
    rng_setup, rng_train = jax.random.split(rng)

    
    evals_dict = {}


    print(f"configuring eval batches from {len(eval_level_generators)} level generators...")
    rng_evals, rng_setup = jax.random.split(rng_setup)
    for levels_name, level_generator in eval_level_generators.items():
        print(f"  generating {evals_num_levels} {levels_name!r} levels...")
        eval_name = f"batch-{levels_name}"
        rng_eval_levels, rng_evals = jax.random.split(rng_evals)
        levels = level_generator.vsample(
            rng_eval_levels,
            num_levels=evals_num_levels,
        )
        if level_solver is not None:
            print("  also solving generated levels...")
            benchmark_returns = level_solver.vmap_level_value(
                level_solver.vmap_solve(levels),
                levels,
            )
            levels_eval = evals.FixedLevelsEvalWithBenchmarkReturns(
                num_levels=evals_num_levels,
                num_steps=evals_num_env_steps,
                discount_rate=ppo_gamma,
                levels=levels,
                benchmarks=benchmark_returns,
                env=env,
                period=1,
            )
        else:
            levels_eval = evals.FixedLevelsEval(
                num_levels=evals_num_levels,
                num_steps=evals_num_env_steps,
                discount_rate=ppo_gamma,
                levels=levels,
                env=env,
                period=1,
            )
        rollouts_eval = evals.AnimatedRolloutsEval(
            num_levels=evals_num_levels,
            levels=levels,
            num_steps=env.max_steps_in_episode,
            gif_grid_width=gif_grid_width,
            env=env,
            period=1,
        )
        evals_dict[eval_name] = evals.EvalList.create(
            levels_eval,
            rollouts_eval,
        )


    print(f"configuring evals for {len(fixed_eval_levels)} fixed eval levels...")
    for level_name, level in fixed_eval_levels.items():
        print(f"  registering fixed level {level_name!r}")
        # SINGLE ANIMATION EVAL
        rollouts_eval = evals.AnimatedRolloutsEval(
            num_levels=1,
            levels=jax.tree.map(
                lambda x: x[None],
                level,
            ),
            num_steps=env.max_steps_in_episode,
            gif_grid_width=1,
            env=env,
            period=1,
        )
        eval_name = f"fixed-{level_name}"
        evals_dict[eval_name] = rollouts_eval
        # HEATMAP EVALS
        # print("  splaying level for heatmap evals...")
        # levels, num_levels, levels_pos, grid_shape = (
        #     heatmap_splayer_fn(level)
        # )
        # rollout_eval = evals.RolloutHeatmapDataEval(
        #     levels=levels,
        #     num_levels=num_levels,
        #     levels_pos=levels_pos,
        #     grid_shape=grid_shape,
        #     env=env,
        #     discount_rate=ppo_gamma,
        #     num_steps=evals_num_env_steps,
        #     period=1,
        # )
        # eval_name = f"fixed-{level_name}"
        # evals_dict[eval_name] = rollout_eval
        # EVALS WE USED IN TRAINING
        # solo_eval = evals.SingleLevelEval(
        #     num_steps=evals_num_env_steps,
        #     discount_rate=ppo_gamma,
        #     level=level,
        #     env=env,
        #     period=1,
        # )
        # if heatmap_splayer_fn is not None:
        #     print("  also splaying level for heatmap evals...")
        #     levels, num_levels, levels_pos, grid_shape = (
        #         heatmap_splayer_fn(level)
        #     )
        #     spawn_heatmap_eval = evals.ActorCriticHeatmapVisualisationEval(
        #         levels=levels,
        #         num_levels=num_levels,
        #         levels_pos=levels_pos,
        #         grid_shape=grid_shape,
        #         env=env,
        #         period=1,
        #     )
        #     rollout_heatmap_eval = evals.RolloutHeatmapVisualisationEval(
        #         levels=levels,
        #         num_levels=num_levels,
        #         levels_pos=levels_pos,
        #         grid_shape=grid_shape,
        #         env=env,
        #         discount_rate=ppo_gamma,
        #         num_steps=evals_num_env_steps,
        #         period=1,
        #     )
        #     evals_dict[eval_name] = evals.EvalList.create(
        #         solo_eval,
        #         spawn_heatmap_eval,
        #         rollout_heatmap_eval,
        #     )
        #     evals_dict[eval_name] = rollout_heatmap_eval
        # else:
        #     evals_dict[eval_name] = solo_eval


    print("configuring actor critic network...")
    # select architecture
    print(f"  {net_cnn_type=}")
    print(f"  {net_rnn_type=}")
    print(f"  {net_width=}")
    net = networks.Impala(
        num_actions=env.num_actions,
        cnn_type=net_cnn_type,
        rnn_type=net_rnn_type,
        width=net_width,
    )
    # initialise the network
    print("  initialising network (to be overwritten by checkpoint)...")
    rng_model_init, rng_setup = jax.random.split(rng_setup)
    rng_example_level, rng_setup = jax.random.split(rng_setup)
    example_level = arbitrary_level_generator.sample(rng_example_level)
    net_init_params, net_init_state = net.init_params_and_state(
        rng=rng_model_init,
        obs_type=env.obs_type(level=example_level),
    )
    param_count = sum(p.size for p in jax.tree.leaves(net_init_params))
    print("  number of parameters:", param_count)


    # initialise the checkpointer and load the checkpoint
    print("initialising the checkpoint loader...")
    max_to_keep = None
    checkpoint_manager = ocp.CheckpointManager(
        directory=os.path.abspath(checkpoint_folder),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            save_interval_steps=1,
        ),
    )
    # load the checkpoint
    net_params = checkpoint_manager.restore(
        checkpoint_number,
        args=ocp.args.PyTreeRestore(
            net_init_params,
            restore_args=ocp.checkpoint_utils.construct_restore_args(
                net_init_params,
            ),
        )
    )


    # init train state
    print("initialising train state...")
    train_state = TrainState.create(
        apply_fn=net.apply,
        params=net_params,
        tx=optax.sgd(learning_rate=0),
    )


    #  evaluations
    print("doing the evaluations...")
    rng_evals, rng = jax.random.split(rng)
    for eval_name, eval_obj in evals_dict.items():
        print("evaluation:", eval_name)
        rng_eval, rng_evals = jax.random.split(rng_evals)
        results = eval_obj.periodic_eval(
            rng=rng_eval,
            step=0,
            train_state=train_state,
            net_init_state=net_init_state,
        )
        print("* results:", list(results.keys()))
        print("* TODO: save somehow")
        
        # # saving returns from heatmap evaluations
        # returns = results['returns']
        # path = os.path.join(
        #     'evaluations',
        #     f"{checkpoint_folder.replace('/','-')}-{checkpoint_number}",
        #     f'{eval_name}.csv',
        # )
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        # with open(path, 'w') as f:
        #     print("i,j,return", file=f)
        #     for r, i, j in zip(returns, eval_obj.levels_pos[0], eval_obj.levels_pos[1]):
        #         print(f"{i},{j},{r}", file=f)


