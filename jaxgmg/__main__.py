"""
Entry point for jaxgmg CLI.

The functions that are invoked by each subcommand are in the jaxgmg.cli
module. This script just imports them and transforms them into a CLI
using the Typer library.
"""

import jax
import jax.numpy as jnp
import numpy as np
import einops

import typer

from jaxgmg.cli import noisegen
from jaxgmg.cli import mazegen
from jaxgmg.cli import mazesoln
from jaxgmg.cli import parse
from jaxgmg.cli import play
from jaxgmg.cli import solve


TYPER_CONFIG = {
    'no_args_is_help': True,
    'add_completion': False,
    'pretty_exceptions_show_locals': False, # can turn on during debugging
}

def make_typer_app(name, help, subcommands):
    """
    Transform a list of functions into a typer application.
    """
    app = typer.Typer(name=name, help=help, **TYPER_CONFIG)
    for subcommand in subcommands:
        app.command()(subcommand)
    return app


# # # 
# Assemble various entrypoints into a Typer app

app = typer.Typer(**TYPER_CONFIG)


# noise generation
app.add_typer(make_typer_app(
    name='noisegen',
    help=noisegen.__doc__,
    subcommands=(
        noisegen.perlin,
        noisegen.fractal,
    ),
))


# maze generation
app.add_typer(make_typer_app(
    name='mazegen',
    help=mazegen.__doc__,
    subcommands=(
        mazegen.tree,
        mazegen.edges,
        mazegen.noise,
        mazegen.blocks,
        mazegen.open,
    ),
))


# maze solving
app.add_typer(make_typer_app(
    name='mazesoln',
    help=mazesoln.__doc__,
    subcommands=(
        mazesoln.distance,
        mazesoln.direction,
    ),
))


# play environments
app.add_typer(make_typer_app(
    name='play',
    help=play.__doc__,
    subcommands=(
        play.corner,
        play.dish,
        play.follow,
        play.keys,
        play.lava,
        play.monsters,
    ),
))


# testing parsers
app.add_typer(make_typer_app(
    name='parse',
    help=parse.__doc__,
    subcommands=(
        parse.corner,
        parse.dish,
        parse.follow,
        parse.keys,
        parse.lava,
        parse.monsters,
    ),
))


# solve environments
app.add_typer(make_typer_app(
    name='solve',
    help=solve.__doc__,
    subcommands=(
        solve.corner,
        # solve.dish, # not yet implemented
        # solve.follow, # not yet implemented
        solve.keys,
        # solve.lava, # not yet implemented
        # solve.monsters, # not yet implemented
    ),
))


# # # 
# Launch the Typer application

app()

