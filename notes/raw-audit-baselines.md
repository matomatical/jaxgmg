# Raw audit: baselines + UED subsystem

> Verbatim deep-dive from the baselines-mapping pass. Preserved as evidence; the
> distilled cross-cutting view lives in `00-codebase-model.md` / `02-cleanup-plan.md`.
> Line numbers are as of the `experiments` branch at review time — re-verify before editing.

## Autocurricula abstraction (base interface + how each method implements it)

**Base interface** — `base.py`. Two `@struct.dataclass`es:
- `GeneratorState` (`base.py:19`) — empty marker; subclasses add buffers/counters.
- `CurriculumGenerator` (`base.py:27`) with methods: `init() -> GeneratorState` (`:35`); `get_batch(state, rng, num_levels) -> (state, Level[num_levels], batch_type:int)` (`:40`); `batch_type_name(batch_type) -> str` (`:66`); `should_train(batch_type) -> bool` (`:74`); `update(state, levels, rollouts, advantages, proxy_advantages, step, scoring_method_override) -> state` (`:83`, default no-op); `compute_metrics(state) -> dict` (`:97`).

The contract is well-defined but **not enforced** — `init` has different signatures per subclass (no args for `dr_infinite`; `levels` for `dr_finite`; `rng, default_score, batch_size_hint` for `plr`/`accel`), so the base `init` is purely decorative. `get_batch`'s third return is typed `int` in base/`dr` but `bool` in `dr_infinite`/`plr` docstrings — actually all return ints (`replay_choice.astype(int)`, `batch_type`).

**Implementations:**
- `dr_infinite.py` — stateless. `get_batch` (`:24`) samples fresh from the generator, returns `(state, levels, 0)`. No `update`/`init` override. This is the paper's **DR** baseline (`ued="dr"`, `train.py:148`).
- `dr_finite.py` — fixed level pool. `init(levels)` (`:30`) stores `Level[num_levels]` + `visit_counts`. `get_batch` (`:39`) samples ids with `replace=(num_levels > num_levels_total)`, increments visit counts. Used as `ued="dr-finite"` (`train.py:153`).
- `plr.py` — Robust PLR. State (`:36`): `buffer: AnnotatedLevel[buffer_size]`, `num_replay_batches`, `num_generate_batches`, `prev_P_replay`, `prev_batch_was_replay`, `prev_batch_level_ids`. `get_batch` (`:102`) builds both a fresh batch and a replay batch, flips a `prob_replay` coin, returns the chosen one. `update` (`:181`) computes **both** `_replay_update` and `_new_update` and `jnp.where`-selects by `prev_batch_was_replay` (branchless dispatch). `should_train` returns True always unless `robust`, then only on replay batches (`:174`).
- `accel.py` — ACCEL = PLR + mutation. State (`:56`) adds `prev_batch_type: BatchType{GENERATE,REPLAY,MUTATE}`, `prev_batch_mutate_counts`, separate counters; buffer adds `num_mutations`, `num_replays`. `get_batch` (`:130`) builds generate/replay/mutate candidate batches and selects via a **3-state Markov transition matrix** (`:183-196`): GENERATE→{generate,replay}, REPLAY→mutate (always), MUTATE→{generate,replay}. `update` (`:235`) computes all three branch states and `select_n`s by `prev_batch_type`.
- `plr_parallel.py` — **DEAD**. Line 16 is a module-level `raise NotImplementedError`; whole body unreachable. Its `update` (`:166`) calls the old `plr_compute_scores(regret_estimator=..., level=...)` API that no longer exists. It is the superseded "return 2×num_levels, train on first half" design that `plr.py` ("select one batch") replaced. Decommission note at `plr_parallel.py:13-44`.

## Regret estimators / scores.py

Live dispatcher `plr_compute_scores` (`:34`) → vmaps `plr_compute_score` (`:155`), a `match` on `scoring_method.lower()` (true-reward branch `:171-227`, mirrored proxy branch `:234-283`):

| Case (`:171`) | Function | Math | Paper mapping |
|---|---|---|---|
| `absgae` | `l1_value_loss` (`:296`) | `mean(|GAE|)` | Original PLR "value loss"; not regret. Experimental. |
| `pvl` | `regret_pvl` (`:311`) | `mean(max(GAE,0))` | Robust-PLR positive value loss (PVL). Baseline estimator. |
| `maxmc-paper` | `regret_maxmc_paper` (`:322`) | `R_maxever − mean_t V(s_t)` | MaxMC as literally stated in PLR paper; docstring flags likely wrong (no discounting). |
| `maxmc-initial` | `regret_maxmc_initial` (`:343`) | `R_maxever − V(s_0)` | MaxMC variant. |
| `maxmc-critic` | `regret_maxmc_critic` (`:355`) | `R_maxever − mean_t γ^t V(s_t)` (single t-counter, not reset per episode — biased multi-episode) | MaxMC (critic-based). |
| `maxmc-critic-balanced` | `regret_maxmc_critic_balanced` (`:386`) | per-episode discounted-value avg, then avg over episodes (corrected) | MaxMC (critic, multi-episode-correct). |
| `maxmc-actor` | `regret_maxmc_actor` (`:437`) | `R_maxever − avg_return` | **paper's "max-latest". CLI default** (`cli/train.py:49`). |
| `oracle-actor` | `regret_oracle_actor` (`:461`) | `R_oracle(level) − avg_return`, oracle from `maze_solving.maze_distances` → `γ^goal_dist` | **paper's "oracle-latest"** — exact graph optimum. |
| `dro-actor` | `dro_actor` (`:600`) | `−avg_return` (maximin, no optimal term) | DRO/minimax-return baseline. Experimental. |

`max_ever_return` = running max over rollouts in the buffer (`compute_maximum_return`, `experience.py:501`).

**Proxy shaping** (`:285-289`): `original_score − coeff(step)·proxy_score`, optionally clipped at 0. `proxy_shaping_coeff` is a **schedule function passed via `static_argnames`** (HACK comments `:31,152,285`); it's the paper's η-schedule (built in `train.py:135-147`).

**DEAD code in scores.py:** `plr_compute_scores_old` (`:626-1243`, ~620 lines = half the file) — old string-keyed API, **no callers**. Calls `maxmc_critic(...)` (`:1166/1175/1185/1196`) which is **not defined anywhere** (NameError). Duplicate unreachable `case` labels (`true_regret_corner` `:692` after `:667`; `true_regret_dish` `:935`/`:772`; `relative_true_regret_pile` `:965`/`:802`; `proxy_regret_dish` `:1071`/`:908`). Undefined-name bugs at `:1151` and `:691`. Hardcoded `discount_rate=0.999`, `'proxy_corner'` (`:1217,1235`).

**Stale jobs file:** `jobs/plr.jobs` active line uses `--plr-regret-estimator='proxy_regret'` which does **not** exist in the live matcher → `ValueError` at `:227`. `'PVL'` works (lowercased). `proxy_regret`/`proxy_regret_weighted_dist` only existed in the old API.

## PLR buffer + ACCEL editing

`AnnotatedLevel` (`plr.py:24`, `accel.py:42`): SoA struct `level, last_score, last_visit_time, first_visit_time, max_ever_return, max_ever_proxy_return` (+accel `num_mutations, num_replays`). Fixed-size `buffer_size`, seeded with random levels at `default_score=0` (`plr.py:73-99`).

Prioritisation `prioritisation.py:plr_replay_probs` (`:12`): **rank-based** (`(1/rank)^(1/temperature)`, `:25-30`) mixed with **staleness** `(1−c)·tempered + c·staleness`, `staleness = 1 + current_time − last_visit_time` (`:33`, `TODO: is 1+ correct?`). No proportional option. Replay decision = biased coin `prob_replay` (`plr.py:144`).

Update (`plr.py:181`): replay → update `last_score`, bump `last_visit_time`, update max-ever returns for `prev_batch_level_ids` (`_replay_update :220`); generate → top-k insertion: take `num_levels` lowest-`prev_P_replay`, concat challengers, keep top-`num_levels` by score (`_new_update :304-395`).

ACCEL editing `level_mutator.mutate_levels` (`accel.py:176`) mutates the **previous** replay batch (`# mutate the previous(! not current?) batch` `:170`); after REPLAY the FSM forces MUTATE (`:188`); mutated levels rolled out, scored, inserted via same `_buffer_insert_update` (`:422`).

## PPO / GAE / experience core (+ true vs proxy reward)

- Rollout `experience.collect_rollout` (`:88`) `lax.scan` over `num_steps`; `collect_rollouts` (`:221`) vmaps over levels. RNN state + `prev_action` reset on `done` in scan (`:167-176`). `Transition` (`:24`) stores `value` AND `proxy_value`; `Rollout` (`:46`) stores `final_value`/`final_proxy_value`.
- GAE `generalised_advantage_estimation` (`:555`): reverse `lax.scan`, `gae = r − V + (1−done)·γ·(V' + λ·gae)` (`:605-609`). Standard; bootstraps `final_value`; truncation==termination except final step (PureJaxRL convention). `batch_…` (`:621`) vmaps.
- PPO `ppo.py:...update` (`:54`): targets `value+advantages` (`:86`); epochs→minibatches via shuffle+rearrange (`:97-134`); **minibatches split along level axis only** (keeps full `num_steps` for BPTT). `ppo_loss` (`:171`): clipped surrogate, **advantage standardisation over whole minibatch** (`:224`), clipped value loss (`:238-245`), optional clipped **proxy critic loss** (`:251-258`), entropy. Recurrent recomputes hidden via `evaluate_sequence_recurrent`; FF uses cached state via `evaluate_sequence_parallel` (`networks.py:107/157`), selected by `net.is_recurrent`.
- **True vs proxy:** network has **two value heads** (`networks.py:291`, `nn.Dense(2)` → `v, vp`). Proxy reward from `rollouts.transitions.info["proxy_rewards"][proxy_name]` (`train.py:483`). Proxy advantages only when `train_proxy_critic` (`train.py:472-491`). **Footgun** (`train.py:130-132`): `plr_proxy_shaping` without `train_proxy_critic` → proxy head untrained → "invalidates most estimators" (printed warning only).

## Duplication & dead code (quantified)

- `scores.py:626-1243` — ~618 dead lines (~50% of file), undefined-name calls, 5+ unreachable dup cases, hardcoded constants.
- `plr_parallel.py` — entire 297-line file dead.
- PLR vs ACCEL: `plr._new_update` ≈ `accel._buffer_insert_update`; `plr._replay_update` ≈ `accel._replay_update`; ~90% identical (`plr.py:220-395` vs `accel.py:309-520`). Max-ever-return block copy-pasted **4×**.
- DR finite vs infinite reimplement same `get_batch` shape.
- `evals.py` rollout boilerplate repeated across 5 eval classes (`:146,181,214,349,396`).
- `train.py:217-266` vs `evaluate.py:67-113` duplicate eval-setup; `evaluate.py:133-188` ~60 commented-out lines.

## Code smells

- `train.py:46-117` — `run()` takes **~70 positional config args**, ~590-line monster (setup + full training loop).
- `scores.py:31,152,285` — schedule function via `static_argnames` (jit footgun).
- `scores.py:461-597` — `regret_oracle_actor` does `isinstance` dispatch over 5 env types, **re-solves maze each call**, magic numbers (`min_keys=3, min_chests=3, max_steps=128`), `assert False` for keys proxy (`:586`). Validity guarded only by asserts in `cli/train.py` (`:581,1733-1739`).
- `scores.py:1217,1235` — hardcoded `'proxy_corner'` (dead fn).
- `prioritisation.py:33` — `1 + current_time − last_visit_time` (`TODO: is 1+ correct?`).
- `experience.py:307,384` — `steps_per_ep = 1/(eps_per_step + 1/256)` magic `256`.
- `evals.py:323-327` — **BUG**: returns `'proxy_value_img': value_heatmap` instead of `proxy_value_heatmap` (computed at `:317`, discarded).
- `evals.py:104-106` — duplicate-key handling via `print` not error.
- `base.py:92` — `scoring_method_override` "IGNORED" in base/accel but honored in plr (`:269,340`).
- `networks.py:230-239` — return annotation says 3-tuple, returns 4-tuple.
- naming: `plr_regret_estimator` (train param) vs `scoring_method` (plr/accel).

## Tricky / test-worthy logic (ranked)

1. `regret_maxmc_critic_balanced` (`scores.py:386`) — per-episode discounted-value double-scan.
2. `generalised_advantage_estimation` (`experience.py:555`) — GAE recursion + bootstrap + done masking.
3. `compute_average_return` / `compute_maximum_return` (`experience.py:448/501`) — reverse-scan returns + episode-start masking; underpins default `maxmc-actor`.
4. `regret_maxmc_critic` (`scores.py:355`) — documented multi-episode bias.
5. `plr_replay_probs` (`prioritisation.py:12`) — rank + staleness mix.
6. PLR `_new_update` / ACCEL `_buffer_insert_update` top-k insert.
7. ACCEL FSM transition matrix + `select_n` dispatch.
8. PPO clipped value/proxy-value loss + advantage standardisation.
9. `regret_oracle_actor` per-env oracle returns.

## Top cleanup opportunities (ranked)

1. **Delete dead scoring code** — `plr_compute_scores_old` + `plr_parallel.py`; fix/retire `jobs/plr.jobs`. *S.*
2. **Extract shared PLR buffer module** — factor max-ever-return / score / top-k insert shared by plr+accel (removes 4× + 2× duplication). *M.* High value: science core in two copies.
3. **Replace `proxy_shaping_coeff`-as-static-schedule hack** — pass the float per step. *S–M.*
4. **Refactor `regret_oracle_actor`** to solve-once-at-creation (per its own docstring `:485-493`); remove magic numbers, `assert False`, env-config coupling. *M–L.*
5. **Config dataclass for `train.run`** — group ~70 args (net/ppo/ued/proxy/eval); unify naming. *M.*
6. **Dedup eval boilerplate** (`evals.py`, train/evaluate setup); clean commented code; fix `proxy_value_img` bug. *S–M.*
7. **Unit tests** for ranked tricky fns before refactor. *M.*

## Correctness caveats for the human reviewer
(a) CLI default `maxmc-actor` = paper "max-latest"; `oracle-actor` = "oracle-latest".
(b) `maxmc-critic` documented-biased multi-episode (prefer `maxmc-critic-balanced`).
(c) proxy shaping silently garbage if `train_proxy_critic=False` (warning only).
(d) `regret_oracle_actor` valid only under no-time-penalty, cardinal-action, no-time-limit; wrong for `minigrid_maze` turn actions.
