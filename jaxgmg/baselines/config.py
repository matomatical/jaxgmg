"""
Training configuration for ``jaxgmg.baselines.train.run``.

A single, nested source of truth for the universal training hyperparameters and
their defaults. Previously these were ~all-required arguments duplicated across
every ``cli/train.py`` command, where the defaults had drifted (issues #5/#6 in
notes/02-cleanup-plan.md).

The defaults here are the canonical Cheese-in-the-Corner settings (the reference
environment); individual CLI commands override only the fields that genuinely
differ for their environment.

These are plain frozen dataclasses, NOT flax structs: this is static
configuration, not a JAX pytree. The nesting maps cleanly onto a future tyro CLI
(``--ppo.lr``, ``--ued.temperature``, ...). Environment-specific objects (env,
generators, mutator, solver, metrics, evals) are NOT config — they are passed to
``run`` alongside a ``TrainConfig``.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NetConfig:
    cnn_type: str = "large"           # "mlp" | "small" | "large"
    rnn_type: str = "ff"              # "ff" | "lstm" | "gru"
    width: int = 256


@dataclass(frozen=True)
class PPOConfig:
    lr: float = 0.00005
    lr_annealing: bool = False
    gamma: float = 0.999              # discount rate
    gae_lambda: float = 0.95
    clip_eps: float = 0.1
    entropy_coeff: float = 0.001
    critic_coeff: float = 0.5
    max_grad_norm: float = 0.5
    num_epochs_per_cycle: int = 5
    num_minibatches_per_epoch: int = 4


@dataclass(frozen=True)
class UEDConfig:
    method: str = "plr"               # "dr" | "dr-finite" | "plr" | "accel"
    prob_shift: float = 0.0           # alpha: mixing prob of the shift generator
    num_train_levels: int = 2048      # dr-finite sample count / buffer seeding
    # PLR / ACCEL replay buffer (ignored by dr / dr-finite)
    regret_estimator: str = "maxmc-actor"
    robust: bool = False
    buffer_size: int = 4096
    temperature: float = 0.1
    staleness_coeff: float = 0.1
    prob_replay: float = 0.5


@dataclass(frozen=True)
class CollectConfig:
    num_total_env_steps: int = 20_000_000
    num_env_steps_per_cycle: int = 128
    num_parallel_envs: int = 256


@dataclass(frozen=True)
class EvalConfig:
    num_cycles_per_eval: int = 32
    num_cycles_per_big_eval: int = 1024
    num_env_steps: int = 512
    num_levels: int = 256


@dataclass(frozen=True)
class LogConfig:
    console: bool = True
    wandb: bool = True
    gifs: bool = False
    imgs: bool = True
    hists: bool = False
    num_cycles_per_log: int = 32
    num_cycles_per_gifs: int = 1024
    gif_grid_width: int = 16


@dataclass(frozen=True)
class CheckpointConfig:
    enabled: bool = True
    keep_all: bool = False
    max_num: int = 1
    num_cycles_per: int = 512


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    net: NetConfig = field(default_factory=NetConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    ued: UEDConfig = field(default_factory=UEDConfig)
    collect: CollectConfig = field(default_factory=CollectConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    log: LogConfig = field(default_factory=LogConfig)
    ckpt: CheckpointConfig = field(default_factory=CheckpointConfig)
