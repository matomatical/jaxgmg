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


# # # 
# Assemble entry points into a Typer application object

app = typer.Typer(**TYPER_CONFIG)


# noise generation
app_noisegen = typer.Typer(**TYPER_CONFIG)
app_noisegen.command()(noisegen.perlin)
app_noisegen.command()(noisegen.fractal)
app.add_typer(app_noisegen, name='noisegen')


# maze generation
app_mazegen = typer.Typer(**TYPER_CONFIG)
app_mazegen.command()(mazegen.tree)
app_mazegen.command()(mazegen.edges)
app_mazegen.command()(mazegen.noise)
app_mazegen.command()(mazegen.blocks)
app_mazegen.command()(mazegen.open)
app.add_typer(app_mazegen, name='mazegen')


# maze solving
app_mazesoln = typer.Typer(**TYPER_CONFIG)
app_mazesoln.command()(mazesoln.distance)
app_mazesoln.command()(mazesoln.direction)
app.add_typer(app_mazesoln, name='mazesoln')


# play environments
app_play = typer.Typer(**TYPER_CONFIG)
app_play.command()(play.corner)
app_play.command()(play.dish)
app_play.command()(play.follow)
app_play.command()(play.keys)
app_play.command()(play.lava)
app_play.command()(play.monsters)
app.add_typer(app_play, name='play')


# testing parsers
app_parse = typer.Typer(**TYPER_CONFIG)
app_parse.command()(parse.corner)
app_parse.command()(parse.dish)
app_parse.command()(parse.follow)
app_parse.command()(parse.keys)
app_parse.command()(parse.monsters)
app_parse.command()(parse.lava)
app.add_typer(app_parse, name='parse')


# solve environments
app_solve = typer.Typer(**TYPER_CONFIG)
app_solve.command()(solve.corner)
app_solve.command()(solve.keys)
app.add_typer(app_solve, name='solve')


# let's go!
app()

