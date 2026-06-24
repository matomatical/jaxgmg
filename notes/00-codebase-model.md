# jaxgmg — codebase model

> A detailed map of the `jaxgmg` codebase as of branch `experiments` (the real,
> "far ahead and much messier" research branch behind the RLC 2025 paper; `main`
> is a stale public snapshot). ~27.8k lines of Python across 44 modules, **0 tests**.
> Companion docs: `01-paper-summary.md` (science grounding), `02-cleanup-plan.md`
> (what to do), `raw-audit-*.md` (evidence with file:line).

## 1. What this is

A JAX library of procedurally-generated grid-world environments + RL baselines for
studying **goal misgeneralization**, and the engine for the experiments in
*Mitigating Goal Misgeneralization via Minimax Regret*. The scientific payload: train
agents with **DR** (domain randomization, MEV) vs **PLR⊥ / ACCEL** (regret-based UED,
MMER) across three environments (corner / dish / keys-and-chests), using **max-latest**
and **oracle-latest** regret estimators, and measure whether the adversary amplifies
rare "distinguishing" levels enough to prevent misgeneralization.

## 2. Layered architecture

```
              ┌─────────────────────────────────────────────────────────┐
  CLI layer   │ cli/ (typer)  app.py wires 12 groups                     │
              │   train ★ eval ★  | demos: play mazegen mazesoln noisegen │
              │                   |  mutate parse solve speedtest splay   │
              │                   |  heatmaps                             │
              └───────────────┬─────────────────────────────────────────┘
                              │  (★ = experiment path; rest are demos)
              ┌───────────────▼─────────────────────────────────────────┐
  Training    │ baselines/                                                │
  layer       │   train.run ──loop──▶ experience.collect_rollouts         │
              │                       experience.GAE                      │
              │                       ppo.update                          │
              │                       autocurricula/* .get_batch/.update  │
              │   networks (IMPALA CNN, 2 value heads)  evals  evaluate   │
              │   autocurricula: base | dr_{finite,infinite} | plr |      │
              │                  accel | scores (regret estimators) |     │
              │                  prioritisation | [plr_parallel ✗dead]    │
              └───────────────┬─────────────────────────────────────────┘
              ┌───────────────▼─────────────────────────────────────────┐
  Environment │ environments/  base.Env / Level / EnvState / LevelGenerator│
  layer       │   /LevelSolver(oracle) /LevelMutator /LevelParser /Metrics │
              │   corner ★ dish ★ pile ★ keys ★ | follow lava monster      │
              │   minigrid_maze | [dish_original ✗dead] [scattered ✗dead]  │
              └───────────────┬─────────────────────────────────────────┘
              ┌───────────────▼─────────────────────────────────────────┐
  Procgen     │ procgen/  maze_generation (Tree/Edge/Block/Noise/Open)    │
  layer       │   maze_solving (Floyd–Warshall APSP — oracle core)        │
              │   noise_generation (Perlin/fractal)  combinatorix (Dyck)  │
              └─────────────────────────────────────────────────────────┘
  Support:  util.py (render/io/wandb/colormaps) · graphics/ (sprites) · wrappers/ [✗dead]
```

The dependency direction is clean (CLI → baselines → environments → procgen); the
problems are **within** layers (duplication, dead code, monster files), not in the
layering.

## 3. The critical path: one training run

`cli/train.py:<env>()` (one of 8 copy-pasted commands) builds an `env`, a pair of
level generators (`Λ_nondistg`, `Λ_distg`, mixed by `--prob-shift` = α), a `LevelMutator`
tree (for ACCEL), a `LevelSolver` (for oracle regret), `LevelMetrics`, and eval batches,
then calls `baselines/train.py:run(...)` with ~62 args. `run` is a ~590-line function
that owns the whole loop:

```
init: networks (CNN trunk + actor + 2 critics: true value v, proxy value vp)
      autocurriculum.init()  (DR: stateless | PLR/ACCEL: AnnotatedLevel buffer[4096])
loop over cycles:
  state, levels, batch_type = autocurriculum.get_batch(rng, num_levels=256)
      DR        → fresh sample from Λ_train[α]
      PLR⊥      → coin flip: GENERATE (score new) vs REPLAY (train + rescore)
      ACCEL     → FSM: GENERATE→REPLAY→(always)MUTATE→…   (edit replayed levels)
  rollouts = collect_rollouts(net, env, levels, num_steps=128)   # vmap over 256 levels
  advantages       = GAE(rewards,       value,       γ=0.999, λ=0.95)
  proxy_advantages = GAE(proxy_rewards, proxy_value)             # only if train_proxy_critic
  if autocurriculum.should_train(batch_type):    # PLR⊥ trains only on REPLAY batches
      net = ppo.update(net, rollouts, advantages, proxy_advantages)  # 5 epochs × 4 minibatches
  state = autocurriculum.update(state, levels, rollouts, advantages, proxy_advantages, step)
      → scores.plr_compute_scores(scoring_method)   # regret estimate per level
      → prioritisation.plr_replay_probs (rank + staleness)  → top-k buffer insert
  periodically: evals.* (fixed levels, heatmaps, animated rollouts) → wandb
```

**Regret estimator dispatch** (`scores.py`): `maxmc-actor` (= paper "max-latest",
CLI default) and `oracle-actor` (= "oracle-latest"; calls `LevelSolver`/`maze_solving`
for the exact optimum). Many other cases exist (pvl, absgae, maxmc-{paper,initial,critic,
critic-balanced}, dro-actor) — experimental, not in the paper's headline results.

## 4. Core data structures

| Struct | Where | Role |
|---|---|---|
| `Level` | `environments/base.py:22` + per-env | flax pytree of level params (walls, positions). The thing generators/mutators produce and solvers consume. |
| `EnvState` | `base.py:30` | `level, steps, done` + per-env dynamic fields (mouse pos, inventory…). |
| `Observation` | `base.py:52` | Boolean grid (`15×15×c`); RGB paths exist but `obs_type` only declares BOOLEAN. |
| `Rollout` / `Transition` | `experience.py:46/24` | scan output; **carries both `value` and `proxy_value`**, plus `info["proxy_rewards"]`. |
| `AnnotatedLevel` | `plr.py:24` / `accel.py:42` | buffer entry: `level, last_score, last/first_visit_time, max_ever_return, max_ever_proxy_return` (+accel mutation/replay counts). |
| `GeneratorState` | `base.py:19` + per-curriculum | autocurriculum state (buffer + counters). |
| `LevelSolution` | per-env `LevelSolver` | cached optimal value / action / distances for the oracle. |

## 5. Module health map

Legend: ✅ healthy/central · 🟡 works but messy · 🔴 broken path · ⚰️ dead

| Module | LOC | Health | Note |
|---|---|---|---|
| `procgen/maze_solving.py` | 212 | ✅ (test!) | Floyd–Warshall APSP; **every oracle depends on it**; correctness rests on the 1-thick-border invariant. |
| `procgen/maze_generation.py` | 520 | ✅ | 5 generators; paper uses Block (25% walls). Two Kruskal impls. |
| `procgen/noise_generation.py` | 234 | ✅ | clean, documented. |
| `procgen/combinatorix.py` | 122 | ✅ (test!) | Dyck/perm enumeration for keys oracle. |
| `environments/base.py` | 693 | 🟡 | solid core; proxy solver API is duck-typed (not in base contract); `level_value` redundant arg. |
| `cheese_in_the_corner.py` | 1401 | ✅ | reference impl; everything else cloned from it. |
| `cheese_on_a_dish.py` | 1621 | 🔴 | **proxy solver commented out but called by `scores.py`** → dish regret regimes can't run. |
| `cheese_on_a_pile.py` | 2094 | 🟡/🔴 | largest/messiest; broken `Step/Scatter/MoveObjects` mutators (`NameError`); 120-line render monster. |
| `keys_and_chests.py` | 2020 | 🟡 | distinct combinatorial oracle (✅ test target); ~300-line `# legacy` block. |
| `minigrid_maze.py` | 1468 | 🔴 | `LevelSolver` broken (`state.goal_pos`/`state.got_corner` don't exist) yet dispatched in `scores.py`. |
| `follow_me / lava_land / monster_world` | 665/617/769 | 🟡 | secondary; no/inline solvers; copy-paste docstrings; `follow_me.__post_init` typo. |
| `cheese_on_a_dish_original.py` | 1688 | ⚰️ | dead ancestor fork (0 imports) — salvage its working dish proxy solver, then delete. |
| `scattered.py` | 1109 | ⚰️ | dead, broken, doesn't implement base API. |
| `baselines/train.py` | 633 | 🟡 | `run()` = ~70 positional args, ~590-line monster; all defaults live in CLI, not here. |
| `baselines/ppo.py` | 293 | ✅ | clipped PPO, clipped value + optional proxy-value loss; BPTT for recurrent. |
| `baselines/experience.py` | 783 | ✅ (test!) | rollout scan + GAE + average/maximum-return; multi-episode masking is subtle. |
| `baselines/networks.py` | 399 | 🟡 | IMPALA CNN, **2 value heads**; return-type annotation wrong (3- vs 4-tuple). |
| `baselines/evals.py` | 441 | 🟡 | 5 eval classes repeat rollout boilerplate; `proxy_value_img` bug (`:326`). |
| `baselines/evaluate.py` | 274 | 🟡 | dup of train eval-setup; ~60 commented-out lines. |
| `autocurricula/scores.py` | 1245 | 🔴/⚰️ | **~50% dead** (`plr_compute_scores_old`, undefined-name landmines); live half is the regret-estimator core. |
| `autocurricula/plr.py` | 423 | ✅ | Robust PLR; buffer + branchless replay/generate update. |
| `autocurricula/accel.py` | 559 | 🟡 | ACCEL FSM; ~90% duplicates plr buffer logic. |
| `autocurricula/{dr_finite,dr_infinite,base,prioritisation}.py` | 42-101 | ✅ | small, fine; `prioritisation` has `TODO: is 1+ staleness correct?`. |
| `autocurricula/plr_parallel.py` | 296 | ⚰️ | `raise NotImplementedError` at module load; dead. |
| `cli/train.py` | 2891 | 🔴 | 8 copy-pasted commands, ~70-80% duplicated, ~500 lines of arg-forwarding; default-value drift. |
| `cli/{eval,heatmaps,splay,...}.py` | — | 🟡/🔴 | `splay`/several `speedtest` cmds crash (`NameError`); `eval`/`heatmaps`/`splay` corner-only. |
| `util.py` | 549 | 🟡 | coherent-ish support; dead helpers (`pico8`, `print_img`, `print_histogram`, `save_json`). |
| `wrappers/jaxued_wrappers.py` | 202 | ⚰️ | 0 imports; depends on undeclared `jaxued` → would `ImportError`. |

## 6. Key invariants & cross-cutting coupling (don't break these)

1. **Border invariant** — `maze_solving.maze_distances` is correct *only* because every maze
   has a mandatory 1-cell wall border that masks out edge-wraparound in the flattened
   neighbor indexing. Any new generator must preserve the full border or the oracle goes
   silently wrong. (`raw-audit-environments.md` #1 tricky item.)
2. **Oracle config constraints** — `regret_oracle_actor` / `LevelSolver` give the true
   optimum only under: cardinal (4-dir) actions, no time penalty (`penalize_time` handled
   consistently), γ ∈ (0,1), episode-length ≥ max path. Currently enforced by **asserts in
   `cli/train.py`**, not by the solver. Wrong for `minigrid_maze` (turn actions).
3. **Proxy-critic dependency** — proxy-shaped regret estimators need `train_proxy_critic=True`
   or the `vp` head is untrained and scores are garbage (warning only, `train.py:130-132`).
4. **`should_train`** — PLR⊥ updates the policy *only* on REPLAY batches; GENERATE/MUTATE
   batches collect rollouts purely to score levels. Getting this wrong changes the method.
5. **Defaults live in the CLI**, not in `baselines/train.run` (which has none). The 8 CLI
   commands are the de-facto config spec — and they have already drifted (`plr_robust`,
   `clipping`, `eta_schedule`). There is no single source of truth for a "default run".

## 7. Headline assessment

The **algorithmic core is sound and well-structured at the seams** (clean layering, JAX
pytrees, branchless curriculum updates, a real PLR/ACCEL implementation). The debt is
**accreted research drift**: (a) ~4.3k lines of outright dead code (~15%), (b) heavy
copy-paste (the 8-way `cli/train.py`, the corner→dish→pile env fork, plr↔accel buffers),
(c) a handful of **broken-but-reachable** code paths that silently disable experiments
(dish proxy solver, minigrid solver), and (d) **zero tests** over genuinely subtle
correctness-critical code (Floyd–Warshall oracle, keys Dyck oracle, GAE, return/regret
estimators). The cleanup is mostly *subtraction and consolidation*, not rewriting — but
it must be **gated behind tests for the science core** so the paper's results stay
reproducible. See `02-cleanup-plan.md`.
