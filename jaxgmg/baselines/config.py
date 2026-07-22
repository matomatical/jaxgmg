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
    gifs: bool = False
    imgs: bool = True
    hists: bool = False
    num_cycles_per_log: int = 32
    num_cycles_per_gifs: int = 1024
    gif_grid_width: int = 16


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    net: NetConfig = field(default_factory=NetConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    ued: UEDConfig = field(default_factory=UEDConfig)
    collect: CollectConfig = field(default_factory=CollectConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    log: LogConfig = field(default_factory=LogConfig)

    @staticmethod
    def from_cli(flat: dict) -> "TrainConfig":
        """Build a TrainConfig from a flat dict of CLI parameters (e.g. the
        `locals()` of a `cli/train.py` command).

        Recognised flat parameter names (the historical spellings) are mapped to
        the nested config fields; unrecognised keys (env-specific args, builder
        objects) are ignored, and any recognised key absent from `flat` falls
        back to the canonical TrainConfig default.
        """
        d = flat
        D = TrainConfig()
        g = lambda name, default: d.get(name, default)
        return TrainConfig(
            seed=g("seed", D.seed),
            net=NetConfig(
                cnn_type=g("net_cnn_type", D.net.cnn_type),
                rnn_type=g("net_rnn_type", D.net.rnn_type),
                width=g("net_width", D.net.width),
            ),
            ppo=PPOConfig(
                lr=g("ppo_lr", D.ppo.lr),
                lr_annealing=g("ppo_lr_annealing", D.ppo.lr_annealing),
                gamma=g("ppo_gamma", D.ppo.gamma),
                gae_lambda=g("ppo_gae_lambda", D.ppo.gae_lambda),
                clip_eps=g("ppo_clip_eps", D.ppo.clip_eps),
                entropy_coeff=g("ppo_entropy_coeff", D.ppo.entropy_coeff),
                critic_coeff=g("ppo_critic_coeff", D.ppo.critic_coeff),
                max_grad_norm=g("ppo_max_grad_norm", D.ppo.max_grad_norm),
                num_epochs_per_cycle=g("num_epochs_per_cycle", D.ppo.num_epochs_per_cycle),
                num_minibatches_per_epoch=g("num_minibatches_per_epoch", D.ppo.num_minibatches_per_epoch),
            ),
            ued=UEDConfig(
                method=g("ued", D.ued.method),
                prob_shift=g("prob_shift", D.ued.prob_shift),
                num_train_levels=g("num_train_levels", D.ued.num_train_levels),
                regret_estimator=g("plr_regret_estimator", D.ued.regret_estimator),
                robust=g("plr_robust", D.ued.robust),
                buffer_size=g("plr_buffer_size", D.ued.buffer_size),
                temperature=g("plr_temperature", D.ued.temperature),
                staleness_coeff=g("plr_staleness_coeff", D.ued.staleness_coeff),
                prob_replay=g("plr_prob_replay", D.ued.prob_replay),
            ),
            collect=CollectConfig(
                num_total_env_steps=g("num_total_env_steps", D.collect.num_total_env_steps),
                num_env_steps_per_cycle=g("num_env_steps_per_cycle", D.collect.num_env_steps_per_cycle),
                num_parallel_envs=g("num_parallel_envs", D.collect.num_parallel_envs),
            ),
            eval=EvalConfig(
                num_cycles_per_eval=g("num_cycles_per_eval", D.eval.num_cycles_per_eval),
                num_cycles_per_big_eval=g("num_cycles_per_big_eval", D.eval.num_cycles_per_big_eval),
                num_env_steps=g("evals_num_env_steps", D.eval.num_env_steps),
                num_levels=g("evals_num_levels", D.eval.num_levels),
            ),
            log=LogConfig(
                console=g("console_log", D.log.console),
                gifs=g("log_gifs", D.log.gifs),
                imgs=g("log_imgs", D.log.imgs),
                hists=g("log_hists", D.log.hists),
                num_cycles_per_log=g("num_cycles_per_log", D.log.num_cycles_per_log),
                num_cycles_per_gifs=g("num_cycles_per_gifs", D.log.num_cycles_per_gifs),
                gif_grid_width=g("gif_grid_width", D.log.gif_grid_width),
            ),
        )
