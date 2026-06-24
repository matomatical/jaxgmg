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
* Testing:
  * Complex JAX algorithms like procedural generation, maze solving, reward
    accumulation, GAE, etc., need tests. Edge cases of environment termination
    and rewards too.
  * Training algorithms can be end-to-end integration tested on simple
    environment distributions.
  * An expensive integration test of the whole port is to replicate the main
    plot(s) of the paper.
