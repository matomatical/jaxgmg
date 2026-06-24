# jaxgmg — cleanup plan & test strategy

> What to clean up, in what order, and how to protect correctness while doing it.
> Builds on `00-codebase-model.md` and the `raw-audit-*.md` evidence files.
> Goal context: clean up + add tests for the tricky parts, **to enable new research
> with these algorithms** (regret estimators, UED, the oracle). So the bar is: the
> science core must be *correct, tested, and easy to extend* afterwards.

## Guiding principle: net first, then subtract, then consolidate

The codebase has subtle, untested, correctness-critical code (oracles, GAE, regret
estimators) *and* heavy duplication tempting a big refactor. Refactoring untested
science code is how reproducibility quietly breaks. So the sequence is deliberate:

1. **Build a correctness net** (tests) over the tricky core — cheap, high-value, and the
   prerequisite for every risky change.
2. **Subtract** — delete unambiguously dead code and fix outright bugs. Low risk, shrinks
   the surface ~15% before any restructuring.
3. **Consolidate** — collapse duplication (config dataclass, shared PLR buffer, shared
   gridworld base), each gated behind the tests from step 1.
4. **Polish** — API/naming unification, enums, demo-surface decisions.

## Scope: cleanup now vs rewrite later

This plan is **in-place stabilization**, not the bigger rewrite. Matthew's
`mfr-wishlist.md` captures the *post-cleanup* vision — reward functions as first-class
objects (add/mix/scale/shape, potential functions; envs take reward fns; oracles support
a subset) and a dependency-stack migration (chex→jaxtyping, flax/distrax→`strux`+hijax,
typer→tyro, orbax→`strux`, wandb→TBD). We don't pursue those here, but we avoid painting
into a corner the rewrite would have to undo:
- The Phase-3 **config dataclass is forward-compatible with tyro** (tyro builds CLIs from
  dataclasses), so it's worth doing now and carries over.
- Don't over-invest in flax/typer-specific deep rewrites (esp. Phase 5 env-base work and
  network internals) that the port will redo — prefer deletion/bug-fixing there for now.
- The bolted-on **proxy machinery** (proxy reward channel, `vp` head, duck-typed proxy
  solver track, `proxy_shaping_coeff` schedule-hack) is exactly what first-class reward
  functions will replace — so in Phases 2/4 fix it minimally to keep experiments running,
  rather than gold-plating it.
- **Tests transfer fully** to the rewrite and are the safety net for the port — so the
  testing investment below is worth making regardless of when the rewrite happens.

## Key issues, ranked (severity × effort)

| # | Issue | Severity | Effort | Phase |
|---|---|---|---|---|
| 1 | **No tests** over oracle/GAE/regret/keys-combinatorics | 🔴 high | M | 0 |
| 2 | **Broken reachable paths**: dish proxy solver commented-out but called (`scores.py`); `minigrid_maze.LevelSolver` (`state.goal_pos`/`got_corner`); pile mutators `NameError` | 🔴 high | M | 2 |
| 3 | **`scores.py` ~50% dead** (`plr_compute_scores_old`, undefined `maxmc_critic`, dup cases) | 🟠 med | S | 1 |
| 4 | **Dead files**: `scattered.py`, `cheese_on_a_dish_original.py`, `plr_parallel.py`, `wrappers/jaxued_wrappers.py` (~3.3k lines, 0 imports — verified) | 🟠 med | S | 1 |
| 5 | **`cli/train.py` 8-way duplication** (~2891 lines, ~70-80% copy-paste, default drift) | 🟠 med | L | 3 |
| 6 | **No config object**: ~62 params typed 3× (CLI sig / `train.run` call / `run` sig); defaults only in CLI and drifting | 🟠 med | M | 3 |
| 7 | **plr↔accel buffer dup** (~90%; max-return block 4×) — science core in two copies | 🟠 med | M | 4 |
| 8 | **Crashing demo commands**: `splay corner` (`obs` undef), several `speedtest` (`level_of_detail`) | 🟡 low | XS | 2 |
| 9 | **Cosmetic bugs**: 7× `{prob_shift=}` missing `f`; `evals.py:326` `proxy_value_img` discards heatmap; networks return annotation | 🟡 low | XS | 2 |
| 10 | **Cheese family ~80-95% copy-paste** (corner→dish→pile; state_value/step/render/mutators/metrics) | 🟡 low | L | 5 |
| 11 | **`regret_oracle_actor` smell**: `isinstance` dispatch + re-solve each call + magic numbers + `assert False` | 🟡 low | M | 4 |
| 12 | **`proxy_shaping_coeff` schedule smuggled through `static_argnames`** (jit footgun) | 🟡 low | S | 4 |
| 13 | **Magic strings** (`ued`, `proxy_name`, `level_splayer`, estimator) no enum/validation; 4× duplicated `match` | 🟡 low | S | 6 |
| 14 | **API inconsistencies**: proxy-solver duck-typing not in base; `scoring_method_override` IGNORED-vs-honored; `regret_estimator` vs `scoring_method`; `splay_*` vs `LevelSplayer.splay_*` | 🟡 low | M | 6 |
| 15 | **Stale `jobs/plr.jobs`** (`proxy_regret` crashes live scorer); ~27 bespoke SLURM scripts; dead util helpers | 🟢 nit | S | 1/6 |

## Phased plan

### Phase 0 — Correctness net (do first)
Stand up `tests/` (none exists) with pytest, CPU-only JAX. Cover the ranked tricky logic
(see Test strategy below). This is the safety net for Phases 2-5. Add a minimal CI hook
(or at least a `make test`). *Effort: M.*

### Phase 1 — Subtract: delete dead code (low risk, ~4.3k lines / ~15%)
- `git rm` `environments/scattered.py`, `baselines/autocurricula/plr_parallel.py`,
  `wrappers/jaxued_wrappers.py` (+ empty `wrappers/`). All verified 0-import.
- Salvage the working **dish proxy solver** out of `cheese_on_a_dish_original.py` into the
  live `cheese_on_a_dish.py` (replacing its commented-out block — ties into issue #2), then
  delete `_original`.
- Delete `plr_compute_scores_old` (`scores.py:627-1243`) and remove the keys `# legacy`
  block (`keys_and_chests.py:~1624-1929`) once tests pin the live solvers.
- Remove dead util helpers (`pico8`, `print_img`, `print_histogram`, `save_json`) after a
  notebook/script grep; fix or retire `jobs/plr.jobs`.
*Effort: S. Each deletion verified by import grep + green tests.*

### Phase 2 — Fix outright bugs (low risk)
- **Reachable/correctness** (issue #2): restore dish proxy solver; fix `minigrid_maze`
  solver field refs (`state.goal_pos`→`Level`, `got_corner`→`got_proxy`) or explicitly mark
  minigrid out-of-scope and remove its `scores.py` dispatch; fix pile `Step/Scatter/MoveObjects`
  mutators (`new_cheese_pos` NameError, `static_argnames=('self')` str-not-tuple).
- **Cosmetic** (issues #8, #9): `splay.py:68` `obs`→`image`; drop bogus `level_of_detail`
  checks in `speedtest.py`; add `f` to the 7 `{prob_shift=}` prints; `evals.py:326`
  `value_heatmap`→`proxy_value_heatmap`; fix `networks.py` return annotation.
*Effort: M (mostly the solver fixes). Add a regression test per fix where feasible.*

### Phase 3 — Consolidate: config dataclass + single train runner (biggest readability win)
- Introduce a `TrainConfig` dataclass (grouped: `net_*`, `ppo_*`, `ued/plr_*`, `proxy_*`,
  `eval_*`, logging) holding the ~50 universal params + their defaults **once**.
- Give `baselines/train.run` its defaults from the dataclass (kills the all-required
  signature). Reduce each `cli/train.py` command to: env-specific args + the 4 builder blocks
  (env, generators, mutator, evals) + one `run(config, env_objects)` call.
- Move the ~160 lines of hardcoded ASCII eval levels out of `train.py` into per-env data;
  replace the 4 duplicated `match level_splayer` blocks with one helper.
*Effort: L. Highest payoff for day-to-day research ergonomics; eliminates default drift.*

### Phase 4 — Consolidate: the science core
- Extract a shared **PLR buffer module** used by `plr.py` and `accel.py` (max-ever-return
  computation, score computation, top-k insertion) — removes the 4×/2× duplication so the
  estimator/buffer logic has one source of truth (issue #7).
- Refactor `regret_oracle_actor` to the design its own docstring proposes: solve the level
  once at creation via a configured `LevelSolver` and pass oracle returns into the scorer,
  instead of `isinstance` + re-solve + magic `min_keys=3`/`128`/`assert False` (issue #11).
- Replace the `proxy_shaping_coeff`-as-static-schedule hack with a per-step float (issue #12).
*Effort: M-L. Gate strictly behind Phase-0 tests; this is the paper's reproducibility core.*

### Phase 5 — Consolidate: shared gridworld base (optional, highest risk)
- Extract a `gridworld` mixin/base for the cheese family: common `_step` movement + the
  `steps` literal (duplicated >20×), the bool→RGB `argmax(priority)→spritemap` render, the
  boilerplate mutators (`ToggleWall`/`Step*`/`Scatter*`), and `LevelMetrics`. Regularize
  `cheese_on_a_pile`'s 6 hard-coded object fields into an array-of-objects (issues #10).
*Effort: L. Only attempt with env-level golden tests in place (Phase 0). Defer if time-boxed.*

### Phase 6 — Polish: API & surface
- Typed enums + central validation for `ued`/`level_splayer`/`proxy_name`/estimator;
  normalize option naming (issue #13).
- Declare the proxy-solver methods in `base.LevelSolver`; unify `scoring_method`
  naming and the `scoring_method_override` semantics; reconcile `splay_*` APIs (issue #14).
- Decide the demo surface: keep 10 demo groups, fold under `jaxgmg demo …`, or move to an
  optional extra. Generalize `eval`/`heatmaps` beyond corner, or document the limitation.
- Restore `__init__.py` files / confirm namespace-package intent; refresh README/roadmap;
  archive the ~27 bespoke SLURM scripts (`scripts/`, `jobs/`) into `scripts/archive/`.

## Test strategy (the "tricky parts")

Tooling: `pytest`, JAX on CPU (`JAX_PLATFORM_NAME=cpu`), tiny fixtures (3×3–7×7 mazes),
fixed PRNG seeds. Mix of **golden** (hand-computed expected values), **property/invariant**,
and **cross-check** (oracle vs brute-force / vs actual rollout) tests. Ranked by value:

1. **`procgen/maze_solving.maze_distances`** — golden APSP on hand-laid 4×4/5×5 mazes
   (incl. unreachable → ∞); **property**: symmetry `d(a,b)=d(b,a)`, triangle inequality,
   `d(a,a)=0`; **border-invariant regression**: a maze missing its border must NOT silently
   pass (document/guard the invariant). Also `maze_directional_distances`/`optimal_directions`:
   the implied path length equals `maze_distances`, tie-breaking is deterministic.
2. **Oracle ↔ rollout cross-check** — for corner & dish, on N random levels: `LevelSolver`
   max return `== γ^(d-1)` and equals the return of an optimal policy actually stepped
   through `env.step`. Validates the whole oracle-latest estimator end-to-end. Include the
   dish-as-barrier case (dish terminates the episode).
3. **`keys_and_chests` oracle** — (a) `combinatorix.associations` yields exactly Catalan(m)
   Dyck paths, all prefix-balanced, deduped; perms count `= P(n,m)`; (b) the 21,600-sequence
   count for (k,c)=(3,10); (c) on small hand-built levels, the enumerated optimum matches a
   slow `itertools` brute force AND a real optimal rollout; (d) unreachable keys/chests → ∞.
4. **`experience.GAE`** — golden against a hand-computed 3–4 step example; **property**:
   λ=0 ⇒ advantage `= r+γV'−V` (TD error); `done` zeroes bootstrap across episode boundaries.
5. **`experience.compute_average_return` / `compute_maximum_return`** — multi-episode-per-
   rollout masking (`first_steps` roll), since these underpin the default `maxmc-actor`
   estimator. Golden on a 2-episode rollout.
6. **Regret estimators** (`scores.py` live cases) — `maxmc-actor` = `max_ever − avg_return`;
   `oracle-actor` = `oracle − avg_return`; `pvl` = `mean(relu(GAE))`; pin `maxmc-critic` vs
   `maxmc-critic-balanced` behavior (the documented multi-episode bias).
7. **PLR buffer mechanics** — `prioritisation.plr_replay_probs` rank+staleness (monotone in
   score; staleness mix sums to 1); top-k insert keeps the highest-score levels; ACCEL FSM
   produces the intended GENERATE/REPLAY/MUTATE proportions.
8. **Level generators produce valid levels** — property tests over many seeds: no
   object on a wall/border, mouse ≠ cheese (per env rules), correct key/chest counts;
   distinguishing vs non-distinguishing classification matches the env's own predicate.
9. **Env dynamics, reward accumulation & termination edge cases** (Matthew flagged these
   explicitly) — collecting cheese/dish/chest gives exactly the right reward at the right
   step; time penalty and timeout behave; **discounted return accumulates correctly across
   multi-step/multi-episode rollouts**; auto-reset boundaries; proxy reward emitted in
   `info`. Edge cases: unreachable goal (zero return), goal at spawn-adjacent, episode that
   times out without collection, keys-and-chests inventory/`min(k,c)` termination.

### Test tiers under a tight compute budget (decided with Matthew)
Matthew is short on compute (NUC = CPU-only; paper used ~1.2k A100-hours) and has no saved
checkpoints — only the paper's published numbers. So the refactor safety net is built to
**not require training compute**, with expensive validation deferred and gated:

- **Tier 1 — correctness (free, CPU, do now):** the *training-free* parts (cheapest to run,
  per Matthew). `maze_solving`, `combinatorix`, oracle returns, env termination/reward
  accumulation, generators, GAE, return/regret math. Expected values come from hand
  calculation or slow brute-force (`itertools`) cross-checks. This is most of the science core.
- **Tier 2 — golden / characterization (free, CPU, do now):** for logic where the "right"
  answer is hard to state up front (full scoring pass, PLR/ACCEL buffer evolution), **capture
  the current output on a fixed seed BEFORE a refactor and assert it's unchanged AFTER.**
  This is the practical drift-guard for Phases 1–4 without needing ground truth or training.
- **Tier 3 — training smoke (cheap, CPU, minutes):** a few PPO cycles on a trivial Open-maze
  + tiny net: runs without error, loss finite, shapes/dtypes correct, checkpoint round-trips.
  A wiring check, not a learning check.
- **Tier 4 — faithful paper reproduction (needs GPU, DEFERRED):** the paper plots are the
  reference (headline numbers captured in `01-paper-summary.md`: corner/keys DR misgeneralizes
  below ~α=1e‑1; dish robust from α=1e‑2; ACCEL+oracle robust for all positive α). Document the
  exact config; run only when GPU is available. A downscaled **mini-repro** (smaller maze/steps)
  is a cheaper qualitative stand-in. **Not a blocker for the cleanup.**

Suggested layout: `tests/procgen/`, `tests/environments/`, `tests/baselines/`,
`tests/integration/`, with a shared `conftest.py` of small-maze fixtures and a
`rollout_optimal_policy` helper.

## Scope decisions (resolved with Matthew, 2026-06-24)

1. **First-class envs: `corner`, `dish`, `keys`** (the paper's three). Focus the cleanup here.
2. **`cheese_on_a_pile` → archive/remove.** Not needed for the next research; messiest live
   file; a clean version can be re-derived in the rewrite. Also drop pile-specific scripts.
3. **`follow_me`, `lava_land`, `monster_world`, `minigrid_maze` → keep but unmaintained.**
   Matthew wants them eventually, but there's no baseline now. Leave them unused and **drop
   their broken solver paths out of `scores.py`** so they can't silently break the 3 paper
   envs. Don't invest until they're revived.
4. **Phase 5 (shared gridworld base) → deferred.** Undecided; revisit later. The env API is
   slated for the rewrite, so a deep in-place flax-coupled extraction is likely wasted now.
   During cleanup, only de-duplicate cheaply/safely.
5. **Demo CLI → keep, simplify lightly.** Demos are useful for new research. Fix the crashers
   (issue #8); don't restructure (typer→tyro is a rewrite-era change).
6. **Reproducibility → no compute / no checkpoints; paper numbers only.** Adopt the 4-tier
   test plan above: Tiers 1–3 (free, CPU) are the cleanup safety net; Tier 4 (GPU paper
   repro) is deferred and non-blocking.

## Still-open (lower stakes, decide later)

- **`main` vs `experiments`**: whether cleaned-`experiments` eventually becomes the new `main`,
  and whether to archive the ~27 SLURM scripts / stale branches into `scripts/archive/`.
