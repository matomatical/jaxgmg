"""
Configuring various kinds of evaluations that can be run on checkpoints or
during training.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
import einops
from flax import struct

from jaxgmg import util
from jaxgmg.baselines import networks
from jaxgmg.baselines import experience

from flax.training.train_state import TrainState
from chex import PRNGKey, Array
from jaxgmg.environments.base import Env, Level


# # # 
# Abstract base class for evals


@struct.dataclass
class Eval:
    def eval(
        self,
        rng: PRNGKey,
        train_state: TrainState,
        net_init_state: networks.ActorCriticState,
    ) -> dict[str, Any]:
        raise NotImplementedError


# # # 
# Concrete evals


@struct.dataclass
class FixedLevelsEval(Eval):
    num_levels: int
    num_steps: int
    discount_rate: float
    env: Env
    levels: Level       # Level[num_levels]


    def eval(
        self,
        rng: PRNGKey,
        train_state: TrainState,
        net_init_state: networks.ActorCriticState,
    ) -> dict[str, Any]:
        rollouts = experience.collect_rollouts(
            rng=rng,
            num_steps=self.num_steps,
            net_apply=train_state.apply_fn,
            net_params=train_state.params,
            net_init_state=net_init_state,
            env=self.env,
            levels=self.levels,
        )
        eval_metrics = experience.compute_rollout_metrics(
            rollouts=rollouts,
            discount_rate=self.discount_rate,
            benchmark_returns=None,
        )
        return eval_metrics


@struct.dataclass
class FixedLevelsEvalWithBenchmarkReturns(Eval):
    num_levels: int
    num_steps: int
    discount_rate: float
    env: Env
    levels: Level       # Level[num_levels]
    benchmarks: Array   # float[num_levels]


    def eval(
        self,
        rng: PRNGKey,
        train_state: TrainState,
        net_init_state: networks.ActorCriticState,
    ) -> dict[str, Any]:
        rollouts = experience.collect_rollouts(
            rng=rng,
            num_steps=self.num_steps,
            net_apply=train_state.apply_fn,
            net_params=train_state.params,
            net_init_state=net_init_state,
            env=self.env,
            levels=self.levels,
        )
        eval_metrics = experience.compute_rollout_metrics(
            rollouts=rollouts,
            discount_rate=self.discount_rate,
            benchmark_returns=self.benchmarks,
        )
        return eval_metrics


@struct.dataclass
class AnimatedRolloutsEval(Eval):
    num_levels: int
    levels: Level       # Level[num_levels]
    num_steps: int
    gif_grid_width: int
    env: Env


    def eval(
        self,
        rng: PRNGKey,
        train_state: TrainState,
        net_init_state: networks.ActorCriticState,
    ) -> dict[str, Any]:
        rollouts = experience.collect_rollouts(
            rng=rng,
            num_steps=self.num_steps,
            net_apply=train_state.apply_fn,
            net_params=train_state.params,
            net_init_state=net_init_state,
            env=self.env,
            levels=self.levels,
        )
        frames = experience.animate_rollouts(
            rollouts=rollouts,
            grid_width=self.gif_grid_width,
            env=self.env,
        )
        return {'rollouts_gif': frames}


@struct.dataclass
class ActorCriticHeatmapVisualisationEval(Eval):
    levels: Level
    num_levels: int
    levels_pos: tuple[Array, Array]
    grid_shape: tuple[int, int]
    env: Env


    def eval(
        self,
        rng: PRNGKey,
        train_state: TrainState,
        net_init_state: networks.ActorCriticState,
    ) -> dict[str, Any]:
        obs, _ = self.env.vreset_to_level(self.levels)
        vec_net_init_state = jax.tree.map(
            lambda c: einops.repeat(
                c,
                '... -> num_levels ...',
                num_levels=self.num_levels,
            ),
            net_init_state,
        )
        action_distr, values, _net_state = jax.vmap(
            train_state.apply_fn,
            in_axes=(None, 0, 0, 0),
        )(
            train_state.params,
            obs,
            vec_net_init_state,
            -jnp.ones(self.num_levels, dtype=int),
        )
        # model policy -> diamond map
        action_probs = action_distr.probs
        action_diamond_plot = generate_diamond_plot(
            shape=self.grid_shape,
            data=action_probs,
            pos=self.levels_pos,
        )
        # model value -> heatmap
        value_heatmap = generate_heatmap(
            shape=self.grid_shape,
            data=values,
            pos=self.levels_pos,
        )
    
        return {
            'action_probs_img': action_diamond_plot,
            'value_img': value_heatmap,
        }
    

@struct.dataclass
class RolloutHeatmapVisualisationEval(Eval):
    levels: Level
    num_levels: int
    levels_pos: tuple[Array, Array]
    grid_shape: tuple[int, int]
    num_steps: int
    discount_rate: float
    env: Env


    def eval(
        self,
        rng: PRNGKey,
        train_state: TrainState,
        net_init_state: networks.ActorCriticState,
    ) -> dict[str, Any]:

        # model policy rollout returns -> heatmap
        rollouts = experience.collect_rollouts(
            rng=rng,
            num_steps=self.num_steps,
            net_apply=train_state.apply_fn,
            net_params=train_state.params,
            net_init_state=net_init_state,
            env=self.env,
            levels=self.levels,
        )
        returns = jax.vmap(
            experience.compute_average_return,
            in_axes=(0,0,None),
        )(
            rollouts.transitions.reward,
            rollouts.transitions.done,
            self.discount_rate,
        )
        rollout_heatmap = generate_heatmap(
            data=returns,
            shape=self.grid_shape,
            pos=self.levels_pos,
        )

        return {
            'policy_rollout_return_img': rollout_heatmap,
        }
    

# # # 
# Plotting helper functions


@functools.partial(jax.jit, static_argnames=('shape',))
def generate_heatmap(shape, data, pos):
    return util.viridis(jnp.zeros(shape).at[pos].set(data))


@functools.partial(jax.jit, static_argnames=('shape',))
def generate_diamond_plot(shape, data, pos):
    color_data = util.viridis(data)
    return einops.rearrange(
        jnp.full((5, 5, *shape, 3), 0.4)
            .at[:, :, pos[0], pos[1], :].set(0.5)
            .at[1, 2, pos[0], pos[1], :].set(color_data[:,0,:])
            .at[2, 1, pos[0], pos[1], :].set(color_data[:,1,:])
            .at[3, 2, pos[0], pos[1], :].set(color_data[:,2,:])
            .at[2, 3, pos[0], pos[1], :].set(color_data[:,3,:]),
        'col row h w rgb -> (h col) (w row) rgb',
    )


