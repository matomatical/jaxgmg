# The rewrite arc — plan

> Plan for the post-cleanup rewrite, agreed 2026-07-22 (Matthew + Claude).
> Vision doc: `mfr-wishlist.md`. Prior state: cleanup complete on `cleanup2`
> (see `02-cleanup-plan.md`), 139 CPU tests green.

## Decisions (settled)

* **Bottom-up phased port**, not an in-place migration. Each phase ends with
  its portion of the test suite green and a concrete deliverable (demo +
  README section), so there is always a working system and a feedback loop.
* **End-state: three repos** — game engine ("gg"), environments, baselines —
  mirroring the craftax-style engine/baselines split. But the split happens
  *after* the rewrite: develop everything as subpackages inside this repo,
  then repackage submodules as their own libraries once the boundaries have
  proven themselves.
* **Venue:** new branch `rewrite` off `cleanup2`. No merge into `main` until
  the very end of the arc.
* **Dependencies:** chex → jaxtyping; flax/distrax → vanilla JAX with strux +
  the hijax convention; typer → tyro; orbax-checkpoint → strux; terminal
  rendering → matthewplotlib. strux is simple and stable (we may add features
  to it as needs arise); hijax is a *reference convention*, not a dependency.
  Clone both locally for reference when needed.
* **wandb is cut**, as a final pre-arc task on `cleanup2` (see below). The
  current plumbing is untested and would drag on the refactor. Logging gets
  replumbed later (replacement TBD — deliberately deferred, one migration at
  a time).
* Research-enabling speed vs. bottom-up quality: bottom-up wins on balance;
  new reward-function research starts after the port.

## Pre-arc: cut wandb (on `cleanup2`)

Scope (from a grep audit, 2026-07-22):

* `util.py`: `wandb_img`, `wandb_gif`, `wandb_flatten_and_wrap_metrics`,
  `wandb_run` decorator, `wandb_define_metrics` (~180 lines).
* `cli/train.py`: `@util.wandb_run` + five `wandb_*` args on each of the 6
  train commands.
* `baselines/train.py`: the wandb logging branch of the train loop, metric
  definitions, end-of-run `wandb.save`.
* `baselines/config.py`: `log.wandb` flag; the gif/img/hist logging levers
  feed wandb-only sinks and go with it.
* `baselines/evaluate.py`: stray unused `import wandb`.
* `pyproject.toml`: the dependency.

**Coupling to resolve:** checkpointing writes to `wandb.run.dir` and is
force-disabled when wandb is off (`baselines/train.py:301–307`). Options:
(a) redirect checkpoints to a local run directory (small change, keeps
checkpointing usable for interim GPU runs); (b) cut orbax checkpointing too
(it's equally untested and gets rebuilt on strux in Phase 3 anyway).
**Decision: TBD.**

The train smoke tests already run with wandb off, so the suite should stay
green through this cut.

## Phase 0 — scaffolding

* Create the `rewrite` branch off `cleanup2`.
* Clone strux (and hijax for reference) locally.
* Set up the new subpackage skeleton and conventions: jaxtyping annotations,
  tyro CLI entry points, test layout mirroring the new structure.
* Decide the engine subpackage's name (working name: `gg` / "gridgames").

## Phase 1 — the engine (gg)

A self-contained JAX grid-game engine with **no environment rules** (no keys,
chests, termination, or rewards):

* Procgen: maze generation (tree/edge/block/noise/open), noise generation.
* Maze solving: APSP distances/directions (Floyd–Warshall core).
* Grid mechanics for both movement regimes: fully observable udlr and
  partially observable forward/turn; generic object interaction / inventory
  support to the extent it earns its keep.
* Rendering: sprites, boolean/RGB observations; terminal output via
  matthewplotlib.

Notes: this layer has no flax/distrax today, so it's the gentlest place to
establish the new idioms. The procgen and maze-solving tests port nearly
verbatim.

**Deliverable:** gg demo(s) + a README that makes sense standalone (mural,
solving visualisation, speedtests) — the seed of the future gg repo README.

## Phase 2 — environments on the engine, reward-first

* Design **first-class reward-function objects**: add / mix / scale / shape,
  potential-based shaping; environments take reward functions as part of
  their API. This is the distinctive research API of the rewrite — for goal
  misgeneralisation, "same dynamics, different reward" *is* the experimental
  manipulation, so it should be a parameter, not a subclass.
* Rebuild the paper environments (corner, dish, keys) as thin compositions of
  gg + reward fn + termination + level distribution. Absorbs the deferred
  Phase-5 dedup and the typed-enum CLI work (issue #13).
* Oracles (level solvers) generalise across reward functions where possible;
  where not, supporting only a subset of reward fns is acceptable.
* Open question: which non-paper environments to port (follow / lava /
  monster / minigrid-maze) vs. drop or defer.

**Deliverable:** environments demo + README (play, splay, oracles). Oracle
and env-dynamics tests are the correctness net.

## Phase 3 — baselines

* Networks on strux + vanilla JAX (drop flax/distrax); PPO, GAE, rollouts.
* Autocurricula: DR, PLR⊥, ACCEL on the shared buffer library, plus a clean
  rebuild of the **parallel + robust PLR** variant (see `mfr-wishlist.md` for
  the preserved behavioural spec: rollout on replay+new batches, train on
  replay only).
* Checkpointing on strux. Console logging only for now; the wandb replacement
  is a separate later decision.
* CLI: tyro, one `TrainConfig`-driven entry point.

**Deliverable:** train demos on CPU; GAE / regret / buffer / smoke tests
ported and green.

## Phase 4 — validation

* The expensive integration test: **replicate the paper's main plot(s)** on
  GPU. This simultaneously settles the three Tier-4 caveats from the cleanup
  (oracle discount fix, keys min_keys/min_chests fix, value-head RNG change)
  — see the cleanup memory / `03-bug-log.md`.
* Recurrent *learning* tests: LSTM/GRU actually solve memory-requiring envs
  (the current smoke test only checks the path runs).
* Optional: rebuild the jaxued-conformant wrapper for external UED interop.
* Refresh README speedtest tables (and finally run them on a GPU).

## Post-arc

* Split into three repos (names TBD); jaxgmg proper becomes the environments
  package depending on gg; baselines depend on both.
* Merge to `main` / release; archive `experiments` as the paper-era branch.
* Then: the new reward-function research, and the logging replacement.

## Open questions (running list)

1. Pre-arc wandb cut: keep checkpointing via a local run dir, or cut orbax
   checkpointing too?
2. Engine subpackage/repo name ("gg"? "gridgames"? something else?).
3. Which non-paper environments make the cut in Phase 2.
4. wandb replacement (deferred by design).
