"""
Tier-3 training smoke test (see notes/02-cleanup-plan.md test tiers and
notes/04-phase3-design.md). A *wiring* check, not a learning check: run a few
PPO cycles on a tiny Cheese-in-the-Corner env + tiny net, for each curriculum
(DR and PLR), and assert the whole pipeline executes and produces finite,
correctly-structured parameters.

This is the safety net for the Phase-3 config refactor. Everything is kept tiny
so we can leave eval / metrics / logging turned ON (exercising the eval, level
metrics, classifier and console-render paths) while still running in seconds.
"""

import numpy as np

import jax
import jax.numpy as jnp
import pytest

from jaxgmg.baselines import train
from jaxgmg.baselines.config import (
    TrainConfig, NetConfig, PPOConfig, UEDConfig, CollectConfig,
    EvalConfig, LogConfig,
)
from jaxgmg.environments import cheese_in_the_corner as corner
from jaxgmg.procgen import maze_generation


def _tiny_corner_builders(mutator=None):
    """The environment-specific objects run() needs, tiny + fast, evals wired up."""
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
        level_mutator=mutator,        # DR/PLR don't need one; ACCEL does
        level_solver=corner.LevelSolver(env=env, discount_rate=0.99),
        level_metrics=corner.LevelMetrics(env=env, discount_rate=0.99),
        eval_level_generators={'eval': gen},                 # one small eval batch
        fixed_eval_levels={},
        heatmap_splayer_fn=None,                             # skip heatmap media
        classify_level_is_shift=lambda level: (level.cheese_pos[0] == 1),
    )


def _tiny_config(ued, regret_estimator='maxmc-actor'):
    """A tiny TrainConfig: 3 cycles of 8x8 = 192 env steps. Logging/eval ON."""
    return TrainConfig(
        seed=0,
        net=NetConfig(cnn_type='mlp', rnn_type='ff', width=16),
        ppo=PPOConfig(
            lr=1e-3, lr_annealing=False, gamma=0.99, gae_lambda=0.95,
            clip_eps=0.2, entropy_coeff=0.01, critic_coeff=0.5,
            max_grad_norm=0.5, num_epochs_per_cycle=1,
            num_minibatches_per_epoch=2,
        ),
        ued=UEDConfig(
            method=ued, prob_shift=0.0, num_train_levels=16,
            regret_estimator=regret_estimator, robust=False, buffer_size=16,
            temperature=1.0, staleness_coeff=0.1, prob_replay=0.5,
        ),
        collect=CollectConfig(
            num_total_env_steps=192, num_env_steps_per_cycle=8,
            num_parallel_envs=8,
        ),
        eval=EvalConfig(
            num_cycles_per_eval=1, num_cycles_per_big_eval=1000,
            num_env_steps=8, num_levels=4,
        ),
        log=LogConfig(
            console=True, gifs=False, imgs=False, hists=False,
            num_cycles_per_log=1, num_cycles_per_gifs=1000, gif_grid_width=2,
        ),
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
    train_state = train.run(_tiny_config(ued), **_tiny_corner_builders())
    _assert_finite_params(train_state)


@pytest.mark.parametrize("ued", ["plr", "accel"])
def test_train_run_oracle_actor_completes_with_finite_params(ued):
    # The oracle-latest estimator path: run() threads the configured level_solver
    # into the curriculum, which solves each batch (buffer.oracle_returns) to
    # score levels. This exercises that wiring end-to-end under the full loop for
    # both buffer-based curricula (issue #11). ACCEL additionally needs a mutator,
    # so this is also the only integration coverage of the ACCEL training path.
    mutator = corner.ToggleWallLevelMutator() if ued == "accel" else None
    train_state = train.run(
        _tiny_config(ued, regret_estimator='oracle-actor'),
        **_tiny_corner_builders(mutator=mutator),
    )
    _assert_finite_params(train_state)


def test_train_run_is_deterministic_for_fixed_seed():
    # Same seed -> identical trained params. The drift guard the config refactor
    # must preserve (behaviour-preserving steps).
    ts1 = train.run(_tiny_config("plr"), **_tiny_corner_builders())
    ts2 = train.run(_tiny_config("plr"), **_tiny_corner_builders())
    l1 = jax.tree_util.tree_leaves(ts1.params)
    l2 = jax.tree_util.tree_leaves(ts2.params)
    for a, b in zip(l1, l2):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
