# Phase 3 — config dataclass + single train runner (design)

> Decided with Matthew (2026-07-21). Refines `02-cleanup-plan.md` Phase 3 and
> issues #5/#6. Goal: kill the (now 7-)way `cli/train.py` duplication and the
> all-required `baselines/train.run` signature by introducing one **nested
> `TrainConfig`** with defaults in a single place, and **subtract dead
> exploratory flags** on the way (which also shrinks what we wrap).

## Current surface (from static analysis; see scratchpad/analyze_train.py)

- `baselines/train.run` = **62 params, ALL required** — defaults live in the CLI
  and have **drifted** (`clipping`, `plr_robust`, `checkpointing`, `log_gifs`).
- 7 commands (`corner`/`dish`/`keys` + unmaintained `minimaze`/`memory_test`/
  `follow`/`lava`) each re-declare and forward them.
- 62 = ~9 builder *objects* + ~48 universal scalars + 5 dead flags (below).

## Decided scope

### Drop — dead exploratory flags
- `eta_schedule`, `eta_schedule_time` — dish-only; the η linear-warmup hack
  (ties into issue #12). Hardcoded off in every other command.
- `debug_stop_gradient`, `_after`, `_oracle` — keys-only debug path
  (`train.py:496`). **Cascade:** deletes `scoring_method_override` (only ever
  produced here; already `# IGNORED` in `base.py`/`accel.py`, honored only in
  `plr.py`) from the plr/accel/base update signatures.

### Cut — proxy machinery (turned off permanently)
Flags: `train_proxy_critic`, `plr_proxy_shaping`, `plr_proxy_shaping_coeff`,
`ppo_proxy_critic_coeff`, `proxy_name`.
- **Now (Phase 3), config-scoped + statically-unreachable branches:** the flags;
  `run()`'s `train_proxy_critic`/`plr_proxy_shaping` branches; the `scores.py`
  proxy dispatch block + `proxy_shaping`/`proxy_shaping_coeff` params; the
  `ppo.py` proxy value-loss term.
- **Deferred — dedicated "proxy strip" step (later, with the network `vp`
  strip):** `Rollout.proxy_value`/`final_proxy_value`/`info['proxy_rewards']`,
  buffer `max_ever_proxy_return`, the `networks.py` `vp` head, and the envs'
  `proxy_rewards` emission. These are inert once nothing reads them; ripping them
  out touches tested GAE/buffer/env code and is its own careful change. (Also
  moots the `evals.py` `proxy_value_img` fixed in Phase 2.)

### Keep
- **All** `plr_regret_estimator` options: `maxmc-actor` (default) / `oracle-actor`
  (paper) + `absgae` / `pvl` / `maxmc-{paper,initial,critic,critic-balanced}` /
  `dro-actor`. NB removing the proxy dispatch block also resolves the `dro-actor`
  asymmetry (it only ever existed in the non-proxy block).
- All other universal scalars.

## `TrainConfig` (nested, tyro-friendly)

```
TrainConfig
├─ seed
├─ net    : NetConfig(cnn_type, rnn_type, width)
├─ ppo    : PPOConfig(lr, lr_annealing, gamma, gae_lambda, clip_eps,
│                     entropy_coeff, critic_coeff, max_grad_norm, clipping,
│                     num_epochs_per_cycle, num_minibatches_per_epoch)
├─ ued    : UEDConfig(method, prob_shift, num_train_levels,   # method = dr|plr|accel
│                     regret_estimator, robust, buffer_size,
│                     temperature, staleness_coeff, prob_replay)
├─ collect: CollectConfig(num_total_env_steps, num_env_steps_per_cycle, num_parallel_envs)
├─ eval   : EvalConfig(num_cycles_per_eval, num_cycles_per_big_eval, num_env_steps, num_levels)
├─ log    : LogConfig(console, wandb, gifs, imgs, hists,
│                     num_cycles_per_log, num_cycles_per_gifs, gif_grid_width)
└─ ckpt   : CheckpointConfig(enabled, keep_all, max_num, num_cycles_per)
```

- Frozen dataclasses. `clipping` → `ppo` (PPO value-clipping); `prob_shift` /
  `num_train_levels` → `ued` (curriculum-level; DR just ignores the plr-ish
  siblings). `run(config, env, generators, mutator, solver, metrics, evals, ...)`
  — builder objects stay explicit args.
- Per-env differences become explicit `dataclasses.replace(BASE, ued=replace(
  BASE.ued, method='dr', prob_shift=0.0))` — greppable, no silent drift.

## Sequencing (test-gated)

1. **Tier-3 training smoke test** (corner, DR + PLR): a few PPO cycles on a tiny
   maze/net — runs without error, losses finite, shapes/dtypes correct,
   checkpoint round-trips. The safety net for everything below (there is no
   training test yet).
2. **Subtract dead flags** — proxy machinery, `debug_stop_gradient*` +
   `scoring_method_override`, `eta_schedule*` — from `run()` + all call sites +
   `scores.py`/`ppo.py`. Smoke-test after each.
3. **Introduce nested `TrainConfig`**; `run(config, *builders)` with defaults
   sourced from the config (kills the all-required signature).
4. **Migrate commands:** `corner` (verify vs smoke test) → `dish` → `keys` →
   the 4 unmaintained (mechanical — their call sites move when `run`'s signature
   changes; no correctness investment).
5. **Eval levels → per-env data**; one shared `level_splayer` helper.
6. Update notes + full suite + memory.

## Future refinements (NOT now — Matthew's ideas)

- Split `UEDConfig` into **method-specific sub-configs** — a `DRConfig`,
  `PLRConfig`, `ACCELConfig`, each holding only its own params — rather than one
  flat `ued` group where DR carries unused plr fields. Deferred to keep Phase 3
  focused; the flat `ued` group is a clean stepping stone. See [[jaxgmg-cleanup]].
