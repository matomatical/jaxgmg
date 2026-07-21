"""
Tier-3 training smoke test (see notes/02-cleanup-plan.md test tiers and
notes/04-phase3-design.md). A *wiring* check, not a learning check: run a few
PPO cycles on a tiny Cheese-in-the-Corner env + tiny net, for each curriculum
(DR and PLR), and assert the whole pipeline executes and produces finite,
correctly-structured parameters.

This is the safety net for the Phase-3 config refactor. Everything is kept tiny
so we can leave eval / metrics / logging turned ON (exercising the eval, level
metrics, classifier and console-render paths that the refactor's eval.*/log.*
config groups feed) while still running in seconds. wandb is OFF (console_log
drives the logging path without it); checkpointing is OFF (the backend is slated
for replacement, so it's not worth testing here).

As Phase 3 drops dead flags and then wraps the ~62 args in a TrainConfig, the
`run(...)` invocation here changes with it, but the assertions stay put — that
is exactly the drift guard we want.
"""

import numpy as np

import jax
import jax.numpy as jnp
import pytest

from jaxgmg.baselines import train
from jaxgmg.environments import cheese_in_the_corner as corner
from jaxgmg.procgen import maze_generation


def _tiny_corner_builders():
    """The ~9 builder objects run() needs, tiny + fast, with evals wired up."""
    env = corner.Env(
        penalize_time=False,
        terminate_after_cheese_and_corner=False,
    )
    gen = corner.LevelGenerator(
        height=7,
        width=7,
        maze_generator=maze_generation.TreeMazeGenerator(),
        corner_size=1,
    )
    return dict(
        env=env,
        train_level_generator=gen,
        level_mutator=None,           # DR/PLR don't need one
        level_solver=corner.LevelSolver(env=env, discount_rate=0.99),
        level_metrics=corner.LevelMetrics(env=env, discount_rate=0.99),
        eval_level_generators={'eval': gen},                 # one small eval batch
        fixed_eval_levels={},
        heatmap_splayer_fn=None,                             # skip heatmap media
        classify_level_is_shift=lambda level: (level.cheese_pos[0] == 1),
    )


def _tiny_run_kwargs():
    """The ~48 universal scalars + the soon-to-be-dropped flags, tiny values.

    3 cycles of 8x8 = 192 env steps total. Logging/eval ON, wandb/checkpoint OFF.
    """
    return dict(
        seed=0,
        # net (mlp = fastest, non-recurrent)
        net_cnn_type='mlp',
        net_rnn_type='ff',
        net_width=16,
        # --- debug-stop-gradient: to be removed in Phase 3 (task 10) ---
        debug_stop_gradient=False,
        debug_stop_gradient_after=0.5,
        debug_stop_gradient_oracle=False,
        # ued (overridden per-test)
        ued='dr',
        prob_shift=0.0,
        num_train_levels=16,
        plr_buffer_size=16,
        plr_temperature=1.0,
        plr_staleness_coeff=0.1,
        plr_prob_replay=0.5,
        plr_regret_estimator='maxmc-actor',
        plr_robust=False,
        # ppo
        ppo_lr=1e-3,
        ppo_gamma=0.99,
        ppo_clip_eps=0.2,
        ppo_gae_lambda=0.95,
        ppo_entropy_coeff=0.01,
        ppo_critic_coeff=0.5,
        ppo_max_grad_norm=0.5,
        ppo_lr_annealing=False,
        # dimensions (tiny)
        num_minibatches_per_epoch=2,
        num_epochs_per_cycle=1,
        num_total_env_steps=192,      # // (8*8) = 3 cycles
        num_env_steps_per_cycle=8,
        num_parallel_envs=8,
        # logging + evals: ON (no wandb); media off to stay fast/robust
        console_log=True,
        wandb_log=False,
        log_gifs=False,
        log_imgs=False,
        log_hists=False,
        num_cycles_per_log=1,         # log every cycle -> exercise metrics/render
        num_cycles_per_gifs=1000,
        num_cycles_per_eval=1,        # eval every cycle
        num_cycles_per_big_eval=1000,
        evals_num_env_steps=8,
        evals_num_levels=4,
        gif_grid_width=2,
        # checkpointing: OFF (backend slated for replacement; not tested here)
        checkpointing=False,
        keep_all_checkpoints=False,
        max_num_checkpoints=1,
        num_cycles_per_checkpoint=1000,
    )


def _assert_finite_params(train_state):
    assert train_state is not None
    leaves = jax.tree_util.tree_leaves(train_state.params)
    assert len(leaves) > 0
    for leaf in leaves:
        assert np.all(np.isfinite(np.asarray(leaf))), "non-finite parameter"
    # the optimiser advanced (a few updates happened)
    assert int(train_state.step) > 0


@pytest.mark.parametrize("ued", ["dr", "plr"])
def test_train_run_completes_with_finite_params(ued):
    kwargs = _tiny_run_kwargs()
    kwargs["ued"] = ued
    train_state = train.run(**_tiny_corner_builders(), **kwargs)
    _assert_finite_params(train_state)


def test_train_run_is_deterministic_for_fixed_seed():
    # Same seed -> identical trained params. This is the drift guard the Phase-3
    # config refactor must preserve (behaviour-preserving steps).
    kwargs = _tiny_run_kwargs()
    kwargs["ued"] = "plr"
    ts1 = train.run(**_tiny_corner_builders(), **kwargs)
    ts2 = train.run(**_tiny_corner_builders(), **kwargs)
    l1 = jax.tree_util.tree_leaves(ts1.params)
    l2 = jax.tree_util.tree_leaves(ts2.params)
    for a, b in zip(l1, l2):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
