# Bug log — correctness issues found during cleanup

Running record of genuine correctness bugs (not just code smells) surfaced by
the `cleanup2` effort, mostly via the test suite. **Policy (per Matthew):**

- We do **not** hotfix these on `cleanup2` as we find them — several touch the
  science core and changing them alters published numbers.
- We **fix them all as part of the refactor**, and keep this log so we know
  what changed and can reason about the effect on the paper's results.
- Each bug that we can pin with a test is pinned as `xfail(strict=True)` (or a
  characterization test of current behaviour). When we fix the bug, the xfail
  should flip to xpass — that's our signal to drop the xfail and check this
  entry off.
- **Target: this log is fully resolved (all `[ ]` → `[x]`) by the end of the
  cleanup.**

Status legend: `[ ]` open (fix pending) · `[x]` fixed & verified.

---

## BUG-1 — Oracle discount off-by-one (γ^d vs γ^(d-1))  `[ ]`

- **Found:** 2026-06-25, cross-checking the oracle against an optimal rollout.
- **Where:** `jaxgmg/environments/cheese_in_the_corner.py:997`
  ```python
  discounted_reward = (self.discount_rate**optimal_dist) * valid_reward
  ```
  and the analogous `state_value` in the other environments' `LevelSolver`s
  (`cheese_on_a_pile.py`, `cheese_on_a_dish.py`, `keys_and_chests.py`,
  `minigrid_maze.py`, `base.py:599`) — **the fix must sweep all of them**, not
  just corner.
- **Bug:** `state_value` discounts the terminal reward by `γ^d`, where `d` is
  the mouse→cheese shortest-path distance. But the reward for reaching the
  cheese lands on the **arrival step** (trajectory index `d-1`, since the first
  step is index 0), so the *realised* discounted return an optimal agent
  collects is `γ^(d-1)`. The oracle applies **one extra discount factor**.
- **Impact:** This is the oracle-latest regret **benchmark**. Because it
  undervalues the optimum, an optimal agent shows slightly **negative**
  oracle-regret (e.g. d=1: collects 1.0, oracle says 0.9). Systematic bias in
  the `oracle-actor` estimator. Magnitude ≈ (1-γ); at the paper's γ=0.999 it's
  ~0.1% — almost certainly why it was never noticed, and unlikely to have
  changed paper conclusions, but worth confirming when we fix it.
- **Fix sketch:** discount by `γ^(optimal_dist - 1)`. **Watch the edge cases**
  before committing: `d=0` (mouse starts on the cheese — does it collect at
  step 0? then `γ^(-1)` is wrong and the convention needs care) and
  `optimal_dist = ∞` (unreachable — `γ^∞ = 0` must stay 0). This is exactly why
  it's deferred to the refactor rather than hotfixed.
- **Pinned by:** `tests/environments/test_corner_oracle.py`
  - `test_oracle_level_value_is_gamma_pow_d` — characterizes current γ^d.
  - `test_realised_return_is_gamma_pow_d_minus_1` — establishes the true target.
  - `test_oracle_value_should_equal_realised_return` — `xfail(strict)`; flips
    to xpass when fixed.
- **Also confirmed in keys-and-chests** (2026-07-21): `FullLevelSolver`
  discounts each chest reward by `γ^(cumulative distance to chest)` while the
  realised return uses `γ^(distance-1)` — same one-extra-factor bias, now
  *per chest*. Pinned by
  `tests/environments/test_keys_oracle.py::test_oracle_value_should_equal_realised_return`
  (3 golden corridor levels, `xfail(strict)`).
- **Also confirmed in the `oracle-actor` regret estimator** (2026-07-21):
  `scores.regret_oracle_actor` independently computes `oracle_max_return =
  γ^goal_dist` (its own `maze_distances` lookup, *not* via `LevelSolver`), so an
  optimal agent gets **negative** oracle-latest regret. This is the estimator
  the paper's `oracle-actor` runs actually use, so the off-by-one is baked into
  three separate code paths (corner solver, keys solver, oracle-actor). The fix
  must touch the estimator too, not just the solvers. Pinned by
  `tests/baselines/autocurricula/test_scores.py::test_oracle_actor_optimal_agent_has_zero_regret`
  (`xfail(strict)`).

---

## BUG-2 (candidate, UNCONFIRMED) — `LevelSolverFiltered` selects on hidden-key count  `[ ]`

- **Found:** 2026-07-21, reading the `oracle-actor` keys path.
- **Where:** `jaxgmg/environments/keys_and_chests.py:1015-1020`
  ```python
  num_real_keys = level.hidden_keys.sum()          # actually counts HIDDEN keys
  value = jnp.where(
      num_real_keys == self.min_keys,
      value_filtered_keys,
      value_filtered_chests,
  )
  ```
- **Concern:** `hidden_keys[i]=True` marks slot `i` as *hidden* (`LevelGenerator`:
  `hidden_keys = arange(num_keys_max) >= num_keys`), so `hidden_keys.sum()` is the
  number of **hidden** keys = `num_keys_max − num_keys`, not the number of real
  keys. The variable is misnamed and the branch selector looks inverted: to pick
  the key-filtered solve when keys are the small dimension (`num_keys == min_keys`)
  the test should be `(~level.hidden_keys).sum() == self.min_keys`. As written it
  only coincidentally selects correctly when `num_keys_max == 2*min_keys`.
- **Reachability:** `LevelSolverFiltered` is used only by `scores.regret_oracle_actor`
  for keys-and-chests, instantiated with hardcoded `min_keys=3, min_chests=3` (itself
  issue #11 in the cleanup plan). So this only bites keys experiments run with the
  `oracle-actor` estimator (default is `maxmc-actor`).
- **NOT YET CONFIRMED — needs Matthew:** what `num_keys_max` did the oracle-actor
  keys runs use? If always 6, the selection happens to be correct and this is "only"
  a misnamed-variable landmine for the refactor. Otherwise it's a live correctness
  bug in those runs. Also the class carries the loud docstring warning "THIS WILL
  SILENTLY BREAK IF YOU PASS IT A LEVEL THAT DOES NOT HAVE THE EXPECTED FORMAT".
- **Not pinned by a test yet** (would need the level-format invariant nailed down
  first). Revisit when cleaning up `regret_oracle_actor` / issue #11 in Phase 4.
