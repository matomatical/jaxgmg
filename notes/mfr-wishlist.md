Some musings on some things I want to eventually do with the rewrite.

* Reward functions should be a first-class object in the new environment
  design. This will be distinctive part of this API and useful for new
  research.
  * Operations for adding, mixing, scaling, shaping reward functions (also
    represent potential functions for example).
  * Environments should take reward functions as part of the API.
  * Oracles may only support certain reward functions, we can generalise them
    to the extent possible but otherwise that's fine.
* Dependency changes:
  * Anywhere where we are implementing terminal rendering we can use
    matthewplotlib (https://github.com/matomatical/matthewplotlib).
  * We should switch from using chex to jaxtyping entirely.
  * We should switch from using flax and distrax to vanilla jax entirely, using
    the strux (https://github.com/matomatical/strux) and hijax convention for
    modular networks.
  * We should switch from using typer to tyro for CLI.
  * We should switch from using orbax-checkpoint to strux for checkpointing.
  * We eventually want to switch from wandb to something different.
* [vamp] Was unsure how much to unify the environment mechanics. I now think we
  could have a module that handles both types of mazes (fully observable udlr,
  partially observable forward/turn movement), including step functions and
  rendering given sprites, maybe some generic support for object interaction
  and inventory (unsure), plus procedural generation and solving of
  mazes/gridworlds themselves, all bundled together with no particular
  environment rules like the num keys or chests or thougths about termination
  and rewards; a little JAX game engine that makes sense by itself. The
  environments can 'import gridgames as gg' and fill in the environment
  distribution / RL API with the specifics for each environment. This I think I
  would be satisfied with. The difference from refactoring this to some shared
  blob within jaxgmg is that gg can aim to be more general and reusable by
  other projects involving game simulation, not just goal misgeneralisation.
  Thinking about it this way may also suggest features and demos that belong in
  gg that would make this library complete by itself.[/vamp]
* Testing:
  * Complex JAX algorithms like procedural generation, maze solving, reward
    accumulation, GAE, etc., need tests. Edge cases of environment termination
    and rewards too.
  * Training algorithms can be end-to-end integration tested on simple
    environment distributions.
  * An expensive integration test of the whole port is to replicate the main
    plot(s) of the paper.
* Baselines & interop to (re)build (deleted during the cleanup because the
  existing code was non-functional/unused, but the *intent* is worth keeping;
  the old code is recoverable from git history — e.g. `git show
  895d537:jaxgmg/baselines/autocurricula/plr_parallel.py`):
  * **Parallel + robust PLR baseline** (my implementation of the variant
    proposed in the minimax paper, `~/agents/papers/Jiang+2023-minimax`).
    Rebuild cleanly rather than porting — the old version duplicated most of
    `plr.py`. The distinctive behaviour to preserve: `get_batch` returns `2 *
    num_levels` levels (a `replay` batch *and* a `new` batch), rollouts + UED
    buffer updates run on **all** of them, but PPO trains on **only** the first
    `num_levels` (the replay ones). i.e. collect experience in parallel from
    both batches, but keep the "robust" property of training on replay only.
    (The old module carried this as a decommissioned `NotImplementedError` stub
    with a training-loop integration snippet in its docstring.)
  * **JaxUED-conformant environment wrapper** for interop with external UED
    baselines (wraps our `Env`/`Level` in jaxued's `UnderspecifiedEnv` API).
    Will need a comprehensive rewrite; deleting the old wrapper also lets us
    drop the external `jaxued` dependency until the feature is rebuilt.
