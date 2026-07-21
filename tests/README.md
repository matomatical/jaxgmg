# jaxgmg tests

Phase-0 correctness net for the cleanup (see `notes/02-cleanup-plan.md`). These
are the **Tier-1** training-free correctness tests over the science core — they
run on CPU in ~10s and need no training compute or checkpoints. They are the
safety net that gates the later refactor phases.

## Running

```sh
pip install -e ".[dev]"     # installs pytest
pytest                       # from the repo root; auto-discovers tests/
```

JAX is pinned to CPU in `tests/conftest.py` (before JAX is imported) so the
suite is portable. `xfail_strict` is on: an `xfail` test that starts passing is
a failure, which flags fixed bugs for follow-up.

## Layout

| File | Covers | Kind |
|---|---|---|
| `procgen/test_maze_solving.py` | Floyd–Warshall APSP (`maze_distances`), directional distances, optimal directions | golden, invariants (symmetry / triangle ineq. / walls=∞), BFS cross-check, **border-invariant guard** |
| `procgen/test_combinatorix.py` | Dyck/Catalan + combination/permutation enumeration (keys oracle primitives) | counts vs closed forms, set-equality vs `itertools` brute force |
| `baselines/test_experience.py` | GAE + `compute_average/maximum_return` | hand-computed golden, NumPy reference, λ=0 ⇒ TD error, `done` cuts bootstrap, multi-episode masking |
| `environments/test_corner_oracle.py` | Cheese-in-the-Corner `LevelSolver` ↔ optimal rollout | end-to-end oracle cross-check, unreachable-cheese edge case |
| `environments/test_keys_oracle.py` | Keys-and-Chests `FullLevelSolver` enumerate-and-argmax oracle | plan-set cardinality (incl. 21,600 for k=3,c=10), golden corridor levels ↔ `env.step` rollout |

`conftest.py` holds shared fixtures: a fixed PRNG `key`, a `make_mazes` factory
(generates border-respecting wall grids), and a `generator_name` parametrisation
across the four border-respecting generators.

## Known findings (pinned as xfail)

Correctness bugs found during the cleanup are logged in `notes/03-bug-log.md`
and pinned here as `xfail(strict)` so a fix flips them to xpass (our signal to
drop the marker and check the bug off). Currently:

**BUG-1 — oracle discount off-by-one.** The `LevelSolver` oracles discount the
terminal reward by one extra factor of γ: corner returns `γ^d` where the
realised optimal return is `γ^(d-1)` (reward lands on the arrival step), and
keys-and-chests over-discounts each chest reward the same way. So an optimal
agent shows slightly *negative* oracle-regret. Tiny at γ=0.999 but a real bias
in the oracle-latest estimator. Pinned by
`test_corner_oracle.py::test_oracle_value_should_equal_realised_return` and
`test_keys_oracle.py::test_oracle_value_should_equal_realised_return`. To be
fixed (across all envs) during the refactor.

## Natural next steps (not yet covered)

- **Dish oracle**: blocked on the broken/commented-out dish proxy solver
  (cleanup-plan issue #2 / Phase 2); add once that path is restored.
- **Regret estimators** in `autocurricula/scores.py` (`maxmc-actor`,
  `oracle-actor`, `pvl`) and **PLR/ACCEL buffer** mechanics (Tier-2
  characterization) — the next layer up from these unit tests.
