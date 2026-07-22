"""
Tier-3 training smoke test for the *recurrent* policy path (see the test tiers
in notes/02-cleanup-plan.md). Companion to test_train_smoke.py, which only
exercises the feedforward path (rnn_type='ff'); this covers the recurrent
machinery that path leaves untested: networks.evaluate_sequence_recurrent, the
`do_backprop_thru_time` branch in ppo.py, and the RNN hidden-state carry/reset
on episode boundaries.

This replaces the former `jaxgmg train memory-test` CLI command: memory-testing
is a check of basic training functionality (does the recurrent algorithm run on
a task that needs memory?), not an experiment, so it belongs here rather than as
a launcher command.

The task is the minigrid Memory-Test level: a partially-observed maze (3x3 obs
window) where the goal spawns on one of two sides and the hero must remember
which — solving it genuinely requires recurrence.

A *wiring* check, not a learning check: it asserts the pipeline runs end-to-end
with finite, advancing parameters for both LSTM and GRU cells. Verifying that
these actually *learn* to solve memory-requiring environments is a deferred
(compute-hungry) learning test — see notes/02-cleanup-plan.md.
"""

import numpy as np

import jax
import pytest

from jaxgmg.baselines import train
from jaxgmg.baselines.config import (
    TrainConfig, NetConfig, PPOConfig, UEDConfig, CollectConfig,
    EvalConfig, LogConfig,
)
from jaxgmg.environments import minigrid_maze


def _tiny_memory_builders():
    """The minigrid Memory-Test task, tiny + fast. DR curriculum, so no solver /
    metrics / evals are needed (mirrors the old memory-test command)."""
    env = minigrid_maze.Env(
        obs_height=3,
        obs_width=3,
        terminate_after_goal=True,
        penalize_time=True,
    )
    gen = minigrid_maze.MemoryTestLevelGenerator()
    return dict(
        env=env,
        train_level_generator=gen,
        level_mutator=None,
        level_solver=None,
        level_metrics=None,
        eval_level_generators={},
        fixed_eval_levels={},
        heatmap_splayer_fn=None,
        classify_level_is_shift=None,
    )


def _tiny_recurrent_config(rnn_type):
    """A tiny TrainConfig with a recurrent cell: 3 cycles of 16x8 = 384 env
    steps. Logging ON (drives train-metric logging), eval OFF."""
    return TrainConfig(
        seed=0,
        net=NetConfig(cnn_type='mlp', rnn_type=rnn_type, width=16),
        ppo=PPOConfig(
            lr=1e-3, lr_annealing=False, gamma=0.99, gae_lambda=0.95,
            clip_eps=0.2, entropy_coeff=0.01, critic_coeff=0.5,
            max_grad_norm=0.5, num_epochs_per_cycle=1,
            num_minibatches_per_epoch=2,
        ),
        ued=UEDConfig(
            method='dr', prob_shift=0.0, num_train_levels=16,
            regret_estimator='maxmc-actor', robust=False, buffer_size=16,
            temperature=1.0, staleness_coeff=0.1, prob_replay=0.5,
        ),
        collect=CollectConfig(
            num_total_env_steps=384, num_env_steps_per_cycle=16,
            num_parallel_envs=8,
        ),
        eval=EvalConfig(
            num_cycles_per_eval=1000, num_cycles_per_big_eval=1000,
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


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_recurrent_training_runs_on_memory_task(rnn_type):
    train_state = train.run(
        _tiny_recurrent_config(rnn_type),
        **_tiny_memory_builders(),
    )
    _assert_finite_params(train_state)
