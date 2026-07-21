"""
Shared pytest fixtures and configuration for the jaxgmg test suite.

The whole suite is designed to run on CPU with no training compute (see the
"test tiers" discussion in ``notes/02-cleanup-plan.md``): these are the Tier-1
correctness tests over the training-free science core.
"""

import os

# Pin JAX to CPU before it (or any jaxgmg module) is imported anywhere in the
# test process. The nook is CPU-only anyway; this just makes the suite portable
# and avoids accidentally grabbing an accelerator on other machines.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import pytest

from jaxgmg.procgen import maze_generation


@pytest.fixture
def key():
    """A fixed-seed PRNG key, for reproducible randomised tests."""
    return jax.random.PRNGKey(0)


# Maze generators that respect the mandatory 1-cell wall border. ("open" is
# trivial but useful; "tree"/"edges"/"blocks" exercise varied connectivity.)
# Tree mazes require odd height/width, so we stick to odd sizes throughout.
_GENERATOR_NAMES = ("open", "blocks", "edges", "tree")


@pytest.fixture
def make_mazes():
    """
    Factory returning a list of randomly generated wall grids (``bool[s, s]``).

    Usage::

        grids = make_mazes(name="blocks", count=8, size=7, seed=0)

    Every generator produces a full 1-cell wall border, so the results are
    valid inputs for the maze-solving oracle.
    """
    def _make(name="blocks", count=8, size=7, seed=0):
        gen = maze_generation.get_generator_class_from_name(name)()
        base = jax.random.PRNGKey(seed)
        grids = []
        for k in jax.random.split(base, count):
            grids.append(gen.generate(k, height=size, width=size))
        return grids
    return _make


@pytest.fixture(params=_GENERATOR_NAMES)
def generator_name(request):
    """Parametrise a test across each border-respecting maze generator."""
    return request.param
