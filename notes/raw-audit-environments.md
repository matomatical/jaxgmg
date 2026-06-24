# Raw audit: environments + procgen subsystem

> Verbatim deep-dive from the environments/procgen mapping pass. Preserved as evidence.
> Line numbers as of the `experiments` branch at review time — re-verify before editing.
> Cross-cutting synthesis lives in `00-codebase-model.md` / `02-cleanup-plan.md`.

## Core abstractions (base.py, 693 lines)

Everything is a `@flax.struct.dataclass` (registered JAX pytree); fields typed `chex.Array`.

- **`Env`** (`base.py:64-339`): config-only struct (`max_steps_in_episode=128`, `penalize_time=True`,
  `automatically_reset=True`, `obs_level_of_detail`, `img_level_of_detail`). Jitted API:
  `reset_to_level`, `step`, `get_obs`, `render_state`, `render_level` + vmapped `vreset_to_level`/`vstep`.
  Subclass contract (`base.py:110-119`): `num_actions`, `obs_type`, `_reset`, `_step`,
  `_render_obs_bool/_rgb`, `_render_state_bool/_rgb`. Base `step` (`:201-258`) owns time-tracking,
  timeout, time-penalty (`1 − 0.9·t/T`), proxy-reward penalty propagation, auto-reset via `lax.cond`.
- **Structs**: `Level` (`:22`, empty), `EnvState` (`:30`: `level, steps, done`), `Observation` (`:52`, empty).
  Each env redefines all three.
- **`LevelGenerator`** (`:346`): `sample(rng)->Level` + `vsample`. `MixtureLevelGenerator` (`:389`).
- **`LevelMutator`** (`:426`) + combinators `Identity/Mixture/Iterated/Chain` (`:457-528`) — ACCEL/minimax.
- **`LevelParser`** (`:535`): `parse(str)->Level`, base `parse_batch`.
- **`LevelSolver`** (`:587`): oracle. `solve->LevelSolution`, `state_value`, `state_action`,
  `state_action_values`, `level_value`; plus **bolted-on proxy track** (`solve_proxy`/`vmap_solve_proxies`/
  `level_value_proxy`/`vmap_level_value_proxy`, `:625-676`). `LevelMetrics` (`:683`): `compute_metrics`.

Specialization is uniform: each env defines `Level/EnvState/Observation/Action(IntEnum)/Channel(IntEnum)/
Env/LevelGenerator/LevelParser` and optionally `LevelSolver/LevelMutator*/splay/LevelMetrics`.

**Base defects:** `level_value` takes a redundant `level` arg (author-flagged wrong, `:619-621`);
`LevelSolver` proxy methods (`solve_proxy`, `state_value_proxies`) are used by the public proxy API
but **not declared in the base class** — implicit/duck-typed proxy contract.

## Per-file summary

- **`base.py`** (693): core abstractions. Live, central.
- **`procgen/maze_generation.py`** (520): 5 generators (Tree/Kruskal, Edge, Noise, Block, Open) behind
  `MazeGenerator` + `get_generator_class_from_name`. Tree ships two Kruskal impls (`_kruskal`, `_kruskal_alt`). Live.
- **`procgen/maze_solving.py`** (212): tricky oracle core. `maze_distances` (Floyd–Warshall APSP,
  `float[h,w,h,w]`), `maze_directional_distances` (`[...,5]`), `maze_optimal_directions`. Highest test value.
- **`procgen/noise_generation.py`** (234): `generate_perlin_noise` + `generate_fractal_noise` + `smootherstep`. Clean. Live.
- **`procgen/combinatorix.py`** (122): `combinations`/`permutations`/`associations` (Catalan/Dyck). Used only by keys. Live.
- **`cheese_in_the_corner.py`** (1401): paper-central; the reference implementation everything is cloned from.
  Full `LevelSolver` with **live** proxy track (`proxy_corner`), 7 mutators, 3 splay fns, `LevelMetrics`. Heavily used.
- **`cheese_on_a_dish.py`** (1621): paper-central. Adds `dish_pos` + multichannel obs + **dish-as-barrier solver
  trick** (`:1127-1141`). 8 mutators. **Entire proxy solver commented out** (`:1086-1095, 1158-1214, 1261-1352`).
- **`cheese_on_a_dish_original.py`** (1688): **DEAD** (0 imports). Ancestor fork of dish. Only unique asset: a
  live, working dish proxy solver that is commented-out in the live file.
- **`cheese_on_a_pile.py`** (2094): paper-central but **largest, messiest**. Fork of dish → 7 hard-coded
  distractors; 9-channel obs; live proxy solver (keyed `napkin`). Broken mutators + 120-line render monster.
- **`keys_and_chests.py`** (2020): paper-central. Distinct solver: enumerates visitation sequences via
  `combinatorix` (`_evaluate_all_visitation_sequences`, `FullLevelSolver`, `LevelSolverInit/Filtered`).
  Carries a `# legacy` block (`:1632-1929`, ~300 lines) author marks unmaintained.
- **`follow_me.py`** (665): leader-follower beacons; caches `dir_map` in Level. No solver/metrics/mutators.
  Bug: `__post_init` typo (`:432`, missing `__`, undefined `num_beacons`).
- **`lava_land.py`** (617): cheese + perlin lava. Solver is inline `Env.optimal_value` (`:323-399`), not a `LevelSolver`. Secondary.
- **`monster_world.py`** (769): apples/shields/monsters, softmax-pursuing monsters (cached `dist_map`). No solver. Secondary. Docstring wrongly says "Keys and Chests" (`:514`).
- **`minigrid_maze.py`** (1468): egocentric FOV maze (turn/forward actions, oriented obs slicing `:367-463`).
  Paper-pipeline-adjacent. **Solver broken** (see smells).
- **`scattered.py`** (1109): **DEAD** (0 imports). Abandoned prototype; method names `_get_obs_*` don't match
  base API (`_render_*`) — cannot satisfy `Env`. Delete.

## Duplication & dead code (quantified)

- **Cheese family ~80–95% copy-paste.** corner→dish→pile successive forks. `state_value`/`state_action`/
  `state_action_values` byte-identical across corner/dish/pile/minigrid (only field names differ). `_step`
  ~90% identical. The 4-direction `steps = jnp.array(...)` literal duplicated **>20×**.
- **Mutators**: `ToggleWallLevelMutator` ~identical in 5 envs; `Step*`/`Scatter*` share ~85% boilerplate.
- **`LevelMetrics.compute_metrics`** ~90% shared (minigrid admits `"copied from Cheese in the Corner"` `:1365`).
- **`cheese_on_a_dish_original.py`** (1688) DEAD ancestor fork.
- **`scattered.py`** (1109) DEAD, broken.
- **keys legacy block** (`:1624-1929`, ~300 dead lines) inc. `original_optimal_value` re-impl with `itertools`.
- **Dish proxy solver in 3 states**: commented-out in live dish, live in dead `_original`, re-forked in pile.

## Code smells (file:line)

- **CRITICAL — `minigrid_maze.LevelSolver` broken**: `state_action` indexes `state.goal_pos` (`:902-903`)
  but `goal_pos` is on `Level` not `EnvState` → AttributeError; also indexes dist table by *goal* not *hero*.
  `state_value_proxies` reads `state.got_corner` (`:781`) but the field is `got_proxy`. Dispatched in `scores.py:861/887/1024/1050/1131`.
- **CRITICAL — `cheese_on_a_dish` proxy solver missing but called**: `scores.py:751-941` calls
  `cheese_on_a_dish.LevelSolver.vmap_solve_proxies`/`vmap_level_value_proxy`, but `solve_proxy`/
  `state_value_proxies`/`LevelSolutionProxies` are commented out (`:1086-1352`) → `relative_true_regret_dish` cannot run.
- **`cheese_on_a_pile` broken mutators**: `StepPile` (`:1298`)/`ScatterPile` (`:1364`) reference undefined
  `new_cheese_pos` → NameError when `split_elements>0`; `MoveObjectsPile` `static_argnames=('self')` (`:1383`) is a str not tuple.
- **Monster function**: `cheese_on_a_pile._render_obs_rgb` (`:393-513`, ~120 lines) hand-unrolled 22-entry sprite priority.
- Commented-out debug block in `CornerCheeseLevelMutator` with note `"ADDING THIS TO SEE IF THIS IS THE
  PROBLEM FOR ACCEL PERFORMING POORLY"` (`cheese_in_the_corner.py:708-716`).
- **JAX foot-guns**: `maze_distances` recomputed in `sample`, solver, and `LevelMetrics` for the same level.
  `FixedCheeseLevelMutator` (`:734`) no-op stub. Python-list + static-loop logic in jitted pile `sample`/mutators (`:713-721`).
- **Magic numbers/encodings**: `char_map` uses `len(Channel)`/`+1` as raw tile codes (TODOs); pile hard-codes
  `6`; `state_value` indexes `[...,4]` ("stay") as bare literal; dish render reads `[:,:,2]`/`[:,:,-1]` ("any would work").
- **Misleading docstrings**: follow_me/monster_world LevelGenerator docstrings say "Cheese on a Dish"/"Keys and Chests"; pile docstrings still describe 1-distractor dish.
- **`obs_type` only supports BOOLEAN** in every env (`# TODO: only works for boolean observations`).
- **`follow_me.__post_init`** (`:432`) misspelled + undefined `num_beacons` → silently never runs.

## Tricky / test-worthy logic (ranked)

1. **`maze_solving.maze_distances`** (`:42-71`) — APSP correctness hinges on subtle invariant: flattened
   neighbor writes `idx±1`/`idx±w` wrap across grid edges and clamp OOB; only masked correctly by the
   **mandatory 1-thick wall border** (the `grid|grid` mask sets border to `inf`). Generators without a full
   border would silently produce wrong distances. **Highest test value** — every regret oracle depends on it.
2. **`maze_directional_distances` / `maze_optimal_directions`** (`:74-210`) — 5-direction padding-and-slice,
   asymmetric source/target axes (warned `:177-181`), stay-action/edge masking, tie-breaking.
3. **`keys_and_chests._evaluate_all_visitation_sequences`** (`:1026-1258`) — Dyck-path oracle; inventory
   accounting, discounting, truncation; `LevelSolverFiltered` "silently breaks if level format wrong" (`:982-986`).
4. **`cheese_on_a_dish.LevelSolver.solve` dish-as-barrier** (`:1127-1141`) — verify oracle return matches env
   when dish terminates; verify `cheese_on_a_pile.solve` (`:1590`) which **omits** the barrier (likely optimistic).
5. **`maze_generation.TreeMazeGenerator._kruskal`** (`:138-190`) — valid spanning tree; `_kruskal` vs `_kruskal_alt` agree.
6. **`combinatorix.associations`** (`:51-121`) — Catalan/Dyck enumeration; count = C(n), prefix-balance, dedup.
7. **Splay functions** (`cheese_in_the_corner.py:1163-1275`) — metamaze indexing; non-jittable index math.
8. **Level classification** (distinguishing vs not) + **generators produce valid levels** (no item on wall/border, mouse≠cheese, reachability).
9. **Rendering invariants** — channel counts match `obs_type`; sprite-priority `argmax` never drops an object.

## Top cleanup opportunities (ranked)

1. **Delete dead files** — `scattered.py` (1109) + `cheese_on_a_dish_original.py` (1688), after salvaging the
   dish proxy solver from `_original`. ~2.8k lines gone. *Effort: low* (0 imports confirmed).
2. **Fix broken oracle paths before refactor** (correctness): minigrid `state_action`/`state_value_proxies`;
   restore dish proxy solver so `scores.py` dish regimes run; fix/remove pile broken mutators. *Effort: medium.*
3. **Extract shared `gridworld` mixin/base** for cheese family: common `_step` movement, `steps` literal,
   bool→RGB render idiom, boilerplate mutators, `LevelMetrics`. Collapses ~80% of corner/dish/pile + secondary
   envs. *Effort: high* — gate behind tests (paper reproducibility surface).
4. **Unify solver/proxy API in `base.py`**: declare `solve_proxy`/`state_value_proxies`/`LevelSolutionProxies`
   in base; fix `level_value(level)` redundant arg. *Effort: medium.*
5. **Regularize `cheese_on_a_pile`**: array-of-objects instead of 6 hard-coded fields + python-list + 120-line
   render monster; fix docstrings; remove dead machinery. *Effort: high* (worst file, actively used).
6. **Add `tests/`** starting with `maze_solving` (#1), generators-produce-valid-levels, oracle-return-matches-rollout, combinatorix counts. *Effort: medium* — prerequisite that de-risks 3 & 5.
7. **Promote secondary envs** (lava/monster/follow) to standard interface (proper `LevelSolver`s) or mark out-of-scope. *Effort: medium.*
