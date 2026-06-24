# Raw audit: CLI + wrappers + util subsystem

> Verbatim deep-dive from the CLI/glue mapping pass. Preserved as evidence.
> Line numbers as of the `experiments` branch at review time — re-verify before editing.
> Cross-cutting synthesis lives in `00-codebase-model.md` / `02-cleanup-plan.md`.

## CLI command inventory

App assembled in `cli/app.py`: **no root callback**; every group added via
`app.add_typer(make_typer_app(...))` (`app.py:39-52`). Options auto-derived by Typer from each function's
kwargs (no shared option model). Structure: `jaxgmg <group> <command> [options]`.

| Group / command | Purpose | Demo/Experiment | Paper? |
|---|---|---|---|
| `heatmaps corner` | Heatmap viz of splayed level value/policy | Demo | No |
| `noisegen perlin/fractal` | Visualize noise | Demo | No |
| `mazegen tree/edges/noise/blocks/open/mural` | Visualize generators | Demo | No (figures) |
| `mazesoln distances/directions/...` | Visualize solver outputs | Demo | No |
| `mutate corner/dish/minimaze/pile/keys` | Animate mutators (ACCEL building blocks) | Demo | No |
| `parse corner/dish/follow/keys/lava/monsters/minimaze` | Test ASCII parsers | Demo | No |
| `play ...` | Interactive keyboard play | Demo | No |
| `solve corner/dish/keys` | Animate optimal solver rollouts | Demo | No |
| `speedtest ...` | Throughput profiling | Demo | No |
| `splay corner` | Test splayers | Demo | No (**broken**) |
| **`train corner/dish/pile/follow/keys/lava/minimaze/memory_test`** | **PPO/UED training driver** | **Experiment** | **Yes — primary** |
| **`eval corner`** | Evaluate a checkpoint | **Experiment** | **Yes (corner only)** |

Wiring gaps in `app.py`: `solve.follow/lava/monsters` commented "not yet implemented" (`:174-177`);
`train.monsters`/`train.scatter` commented out (`:227,229`, and no such functions exist). `parse.dish_multichannel`
(`parse.py:93`) defined but never wired. `eval`/`heatmaps`/`splay` are **corner-only** despite 7 environments.

## cli/train.py anatomy (2891 lines)

**Structure:** flat module — 8 top-level `@util.wandb_run`-decorated functions, one per env: `corner`
(27-491), `dish` (493-985), `pile` (988-1485), `keys` (1486-1820), `minimaze` (1821-2314), `memory_test`
(2315-2464), `follow` (2467-2681), `lava` (2682-2891). No shared helper / config object / dispatch table.

Each "fat" command follows an identical 7-block skeleton, copy-pasted with per-env substitutions:
1. `config = locals(); util.print_config(config)`
2. configure `env = <env>.Env(...)`
3. configure level generators (`orig`/`shift`/`tree`, optional `MixtureLevelGenerator` if `prob_shift>0`)
4. `classify_level_is_shift(level)` nested closure
5. configure level mutator (largest divergent block — `Mixture`/`Chain`/`Iterated` trees, ~40-60 lines)
6. configure solver/metrics/eval generators + **inline ASCII fixed-eval levels** (only when `env_size==15`;
   corner alone ~160 lines of hardcoded maze strings, `:248-411`)
7. one ~63-line `train.run(...)` call.

`memory_test` is the structural outlier (~150 lines, passes `None`/`{}` for mutator/solver/metrics/evals).

**Duplication quantified:**
- **8 near-identical signatures**, 56-70 params each. The bulk (~50 params: all `net_*`, `ued`, `plr_*`,
  `proxy_*`, 11 `ppo_*`, dims, logging/eval/checkpoint) is identical across all 8 — defaults *and* comments.
- **8 near-identical `train.run(...)` blocks** (`:426,921,1422,1754,2248,2400,2615,2828`), ~63 lines each ≈
  **~500 lines of pure keyword-forwarding boilerplate**, plus hardcoded `debug_stop_gradient*` literals.
- **~70-80% of the file is duplicated glue.** One default change must be made in up to 8 places.

**Hyperparameter threading:** CLI arg (Typer) → `locals()` dump for logging → manually re-listed in
`train.run(...)`. No `**kwargs`, no dataclass. Every param typed **3×** (CLI signature, `train.run` call,
`baselines/train.py:run` signature `:47-116`).

## Config & argument handling

- **No config dataclass anywhere.** Hyperparameters are Typer kwargs with inline literal defaults.
- `baselines/train.py:run()` (`:46-117`) declares ~62 params **with no defaults** (all required). So **all
  defaults live only in the CLI layer**, duplicated across 8 commands — and **already drifting**: `plr_robust`
  default `False` in corner/follow but `True` in dish/keys/minimaze/memory_test; `clipping` `False` in
  corner/dish vs `True` in follow/memory_test; `eta_schedule` a real param only in `dish`, hardcoded elsewhere.
- `cli/parse.py` is **not** config handling — it's the `parse` *demo* group (confusing name). Real arg parsing is 100% Typer-from-signature.
- Magic strings, no enum/validation at CLI boundary: `ued` ∈ {dr, dr-finite, plr, plr-parallel} (comment-only,
  `:41`), `proxy_name` (8 free-text values), `level_splayer`/`splayer` validated by repeated `match` blocks in
  4 files (`eval.py:144`, `heatmaps.py:155`, `splay.py:38`, `train.py:415`).

## util.py contents (mostly coherent)

Sections: string/image rendering (`img2str:22`, `print_config:63`, `print_histogram:71`, `print_legend:82`,
`print_img:91`, `filter_and_render_metrics:99`); dict transforms (`flatten_dict:134`, `linearise_dict:145`);
disk I/O (`save_image:172`, `save_gif:208`, `save_json:257`); wandb (`wandb_img:266`, `wandb_gif:289`,
`wandb_flatten_and_wrap_metrics:315`, `wandb_run:348` decorator, `wandb_define_metrics:426`); colormaps
(`viridis:448`, `sweetie16:522`, `pico8:536`). Reasonable support module; mixes pure-presentation helpers with
wandb-coupled training plumbing (`wandb_run` arguably belongs near `baselines/`).

**Dead/near-dead helpers (no callers):** `pico8` (`:536`), `print_img` (`:91`), `print_histogram` (`:71`),
`save_json` (`:257`). `flatten_dict`/`linearise_dict` used only internally. `viridis`/`sweetie16` only by demos.

## wrappers/jaxued_wrappers.py — vestigial

`wrappers/` has **no `__init__.py`**; only file is `jaxued_wrappers.py` defining `UnderspecifiedEnvWrapper`
(`:42`) + `LogWrapper` (`:104`) over jaxued's `UnderspecifiedEnv`.
- **0 importers** anywhere. `jaxued` is imported only here and is **not in `pyproject.toml` deps** → would
  `ImportError` if invoked. Unfinished docstring (`:14`). Internal step inconsistency (`:160`). Dead jaxued-era prototype.

## Code smells (file:line)

- **Broken — NameError:** `splay.py:67-68` assigns `image = env.render_level(level)` then `util.img2str(obs)`
  — `obs` undefined. `splay corner` crashes immediately. *(verified)*
- **Broken — NameError:** `speedtest.py:228,256,284,314` (`mazegen_tree/edges/blocks/noise`) and `:668`
  (`envstep_monsters`) reference `level_of_detail`, not a param of those fns → crash. (`mazegen_open:332` is fine.)
- **F-string bug (×7):** `train.py:142,610,1111,1612,1928,2575,2788` — `print("...{prob_shift=}...")` missing `f`. *(verified ×3 shown)*
- **Dead/commented wiring:** `app.py:174-177,227,229`; unwired `parse.dish_multichannel` (`parse.py:93`).
- **Hardcoded debug knobs:** `debug_stop_gradient*` literals in all 8 `train.run` calls; `eta_schedule` only in `dish`.
- **Duplicated `splayer` match blocks:** `eval.py:144`, `heatmaps.py:155`, `splay.py:38`, `train.py:415`
  (eval/heatmaps/train use `cheese_in_the_corner.splay_*`; splay uses `LevelSplayer.splay_*` — divergent API).
- **Inconsistent option naming:** `mutate_cheese` vs `mutate_cheese_on_dish`; `env_layout` (train/eval) vs
  `layout` (demos); `num_keys_min/max` (speedtest) vs `num_keys/num_keys_max` (play/solve).
- **Default-value drift across 8 copies:** `plr_robust`, `clipping`, `wandb_project`, `log_gifs` differ.
- **Commented-out env imports** repeated as boilerplate: `heatmaps.py:11-15`, `splay.py:11-15`, `solve.py:13-15`.
- **Dead util helpers:** `pico8`, `print_img`, `print_histogram`, `save_json`.

## Top cleanup opportunities (ranked)

1. **Collapse 8 train commands onto a shared config + single runner.** *Effort: L, highest payoff.* Introduce
   `TrainConfig` dataclass holding the ~50 universal params + defaults *once*; give `baselines/train.py:run()`
   defaults from it (kills the second copy). Reduce each command to env-specific params + 4 builder blocks +
   one `train.run(config, env_objects)` call. Eliminates ~500 lines forwarding + default drift.
2. **Extract fixed-eval ASCII levels + splayer-selection out of train.py.** *Effort: S-M.* Move ~160 lines of
   hardcoded mazes into per-env data; replace 4 duplicated `match level_splayer` blocks with one helper.
3. **Delete vestigial code.** *Effort: XS.* `wrappers/jaxued_wrappers.py` + empty `wrappers/`; dead util helpers; commented `app.py` wiring.
4. **Fix outright bugs.** *Effort: XS.* `splay.py:68` `obs`→`image`; remove bogus `level_of_detail` checks in
   speedtest; add `f` prefix to 7 prints. (Several advertised commands currently crash.)
5. **Replace magic strings with enums + central validation.** *Effort: S.* Typed enums for `ued`,
   `level_splayer`, `plr_regret_estimator`; normalize option naming.
6. **Decide the demo surface.** *Effort: S (product decision).* ~10 of 12 groups are demos, none paper-critical;
   could move under `jaxgmg demo ...` or optional install. `eval`/`heatmaps`/`splay` are corner-only — generalize or document.

Key files: `cli/train.py` (monster), `baselines/train.py` (`run()` target for shared defaults), `cli/app.py`
(wiring), `cli/eval.py`, `util.py`, `wrappers/jaxued_wrappers.py` (delete).
