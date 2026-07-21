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

## BUG-2 — `LevelSolverFiltered` selects on hidden-key count (affects paper results)  `[ ]`

> **CONFIRMED (2026-07-21).** Matthew confirmed the oracle-latest (`oracle-actor`)
> keys-and-chests experiments are in the **main paper**, and the appendix
> corroborates `num_keys_max=10`, `min_keys=3`. Since `num_keys_max=10 ≠ 6`, the
> buggy and correct selectors do NOT coincide, so this **did affect the published
> keys oracle-latest numbers** on the training distribution. Not just a latent
> landmine — a real bug in reported results.


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
- **Effect with the paper's config** (`num_keys=3, num_keys_shift=10, num_chests=10,
  num_chests_shift=3` ⟹ `num_keys_max=num_chests_max=10`, `min_keys=min_chests=3`):
  - **Train / original levels** (3 real keys, 10 real chests): correct branch is
    `value_filtered_keys`; buggy `hidden_keys.sum()=7 ≠ 3` selects
    `value_filtered_chests`, which truncates to the first 3 of 10 chests and thus
    **undervalues the optimum** (oracle regret too low / negative). WRONG.
  - **Shift / eval levels** (10 real keys, 3 real chests): both selectors land on
    `value_filtered_chests`, which is correct here. OK by luck.
- **Reachability:** `LevelSolverFiltered` is used only by `scores.regret_oracle_actor`
  for keys-and-chests, hardcoded `min_keys=3, min_chests=3` (issue #11). Reached via
  `--plr-regret-estimator oracle-actor`, or the `--debug-stop-gradient
  --debug-stop-gradient-oracle` path (`baselines/train.py:499`; honored in `plr.py`,
  IGNORED in `accel.py`/`base.py`). No *committed* script enables it, but the runs
  exist off-repo (see above).
- **Fix:** one line — `num_real_keys = (~level.hidden_keys).sum()`. (Also fold
  `LevelSolverFiltered` into a cleaner oracle when addressing issue #11 in Phase 4;
  consider re-checking whether the paper's keys oracle-latest numbers shift.)
- **Pinned by:** `tests/environments/test_keys_oracle.py`
  - `test_filtered_solver_matches_full_solver_on_train_format` — `xfail(strict)`;
    `LevelSolverFiltered` should equal the trusted `FullLevelSolver` optimum on a
    keys-small level but returns the wrong branch. Flips to xpass when fixed.
  - `test_filtered_solver_currently_picks_wrong_branch` — characterizes the current
    (buggy) 0-value behaviour so the bug is documented even before the fix.

---

## BUG-3 — staleness has an extra `+1` vs the reference PLR algorithm  `[ ]`

> Not a crash or a clearly-wrong result — an **unintended deviation from the
> published algorithm**, documented here (per Matthew) so we can later test
> whether it had a meaningful effect. Verified 2026-07-21 against both papers'
> equations and both reference implementations (see citations).

- **Where:** `jaxgmg/baselines/autocurricula/prioritisation.py:33`
  ```python
  staleness = 1 + current_time - last_visit_times   # TODO: is 1+ correct?
  ```
- **Reference algorithm.** Both PLR papers define the staleness distribution as
  `P_C(l_i) = (c − C_i) / Σ_j (c − C_j)`, where `c` is the current episode/step
  count and `C_i` is the timestamp at which level `i` was last sampled — so a
  **just-visited level has staleness 0**:
  - *Prioritized Level Replay* (Jiang, Grefenstette, Rocktäschel; arXiv
    2010.03934), Eq. (staleness): `~/agents/papers/Jiang+2020-plr/core/3_methods.tex:148`.
  - *Replay-Guided Adversarial Environment Design* / Robust PLR (arXiv
    2110.02439), "PLR level-buffer update rule" algorithm: `P_C = (c−C_i)/Σ(c−C_j)`
    at `~/agents/papers/Jiang+2021-dcd/main.tex:~515`.
  - Reference code, `_update_staleness` = `seed_staleness += 1;
    seed_staleness[selected] = 0` (just-visited → 0):
    - `facebookresearch/level-replay` @ ccecf45, `level_replay/level_sampler.py:202-205`
    - `facebookresearch/dcd` @ cefd881, `level_replay/level_sampler.py:627-630`
- **The deviation.** jaxgmg computes `1 + current − last_visit`, i.e.
  `staleness_jaxgmg = staleness_reference + 1`, uniformly (verified by simulating
  both update rules over an identical replay sequence: the difference is exactly
  `[1,1,…]` at every step). Consequences:
  1. A **just-replayed level gets nonzero staleness weight** (should be 0), so it
     is not fully de-prioritised on the staleness axis the way the paper intends.
  2. Adding 1 to every entry before normalising **compresses the staleness
     distribution toward uniform**, weakening staleness prioritisation overall
     (e.g. reference `[0,5]→[0,1]`; jaxgmg `[1,6]→[1/7,6/7]`).
- **Reachability / blast radius.** Affects every run with `staleness_coeff > 0`.
  jaxgmg default is `0.1`; the Robust-PLR paper used `0.3`/`0.7`. So this touches
  the main PLR/ACCEL results, though the effect is likely small (it only *softens*
  a secondary mixture term). Magnitude to be measured, not assumed.
- **Fix:** `staleness = current_time - last_visit_times` (drop the `1 +`), and
  double-check the clock alignment: last-visit is stamped `num_replay_batches + 1`
  in `plr._replay_update`, and `current_time = num_replay_batches`, so with the
  `+1` removed a just-visited level reads staleness `-1` — the stamp should then
  be `num_replay_batches` (no `+1`) so it reads 0. Fix both together.
- **Pinned by:** `tests/baselines/autocurricula/test_prioritisation.py`
  - `test_just_visited_level_has_zero_staleness_weight_ref` — `xfail(strict)`;
    asserts the reference behaviour (just-visited ⇒ 0 staleness weight). Flips to
    xpass when fixed.
  - `test_staleness_offset_is_one_not_zero` — characterizes current behaviour.

### Checked and NOT a bug: rank tie-breaking

The rank prioritisation breaks ties between equal scores by index (via `argsort`)
rather than averaging — but this **matches the reference** exactly: both
`level-replay` (`level_sampler.py:295-299`) and `dcd` (`:791-795`) use
`temp = np.flip(scores.argsort()); ranks[temp] = arange+1`, ordinal ranks with an
index tie-break, identical in form to jaxgmg's `prioritisation.py:25-28`. (Only a
cosmetic difference in *which* tied index wins — `np.flip(argsort)` favours the
higher index, jaxgmg's `argsort(descending=True)` the lower — immaterial given
random buffer order.) Documented via `test_equal_scores_broken_by_index_not_uniform`;
no change needed.
