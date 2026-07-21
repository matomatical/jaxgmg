"""
Tier-3 training smoke test (see notes/02-cleanup-plan.md test tiers and
notes/04-phase3-design.md). A *wiring* check, not a learning check: run a few
PPO cycles on a tiny Cheese-in-the-Corner env + tiny net, for each curriculum
(DR and PLR), and assert the whole pipeline executes and produces finite,
correctly-structured parameters.

This is the safety net for the Phase-3 config refactor: it exercises
collect_rollouts -> GAE -> ppo.update -> curriculum get_batch/update across the
real `baselines.train.run` entry point. `run` is invoked with wandb and
checkpointing OFF (both are wandb-coupled) and logging off (so it stays fast).

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
    """A minimal set of the ~9 builder objects run() needs, tiny + fast."""
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
        level_solver=None,            # skip solving (evals are off anyway)
        level_metrics=None,
        eval_level_generators={},     # no eval batches -> fast
        fixed_eval_levels={},
        heatmap_splayer_fn=None,
        classify_level_is_shift=None,
    )


def _tiny_run_kwargs():
    """The ~48 universal scalars + the soon-to-be-dropped flags, tiny values.

    4 cycles of 8x8 = 256 env steps total. wandb/checkpoint/logging all off.
    """
    return dict(
        seed=0,
        # net (mlp = fastest, non-recurrent)
        net_cnn_type='mlp',
        net_rnn_type='ff',
        net_width=16,
        # --- proxy machinery: to be removed in Phase 3 (kept off here) ---
        train_proxy_critic=False,
        plr_proxy_shaping=False,
        proxy_name='proxy_corner',
        plr_proxy_shaping_coeff=0.0,
        # --- eta schedule: to be removed in Phase 3 ---
        eta_schedule=False,
        eta_schedule_time=0.0,
        # --- debug-stop-gradient: to be removed in Phase 3 ---
        debug_stop_gradient=False,
        debug_stop_gradient_after=0.5,
        debug_stop_gradient_oracle=False,
        # ppo value clipping
        clipping=False,
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
        ppo_proxy_critic_coeff=0.5,
        ppo_max_grad_norm=0.5,
        ppo_lr_annealing=False,
        # dimensions (tiny)
        num_minibatches_per_epoch=2,
        num_epochs_per_cycle=1,
        num_total_env_steps=256,      # // (8*8) = 4 cycles
        num_env_steps_per_cycle=8,
        num_parallel_envs=8,
        # logging + evals: OFF (keeps log_cycle False -> fast, no wandb)
        console_log=False,
        wandb_log=False,
        log_gifs=False,
        log_imgs=False,
        log_hists=False,
        num_cycles_per_log=1,
        num_cycles_per_gifs=1000,
        num_cycles_per_eval=1,
        num_cycles_per_big_eval=1000,
        evals_num_env_steps=8,
        evals_num_levels=4,
        gif_grid_width=2,
        # checkpointing: OFF (wandb-coupled)
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
