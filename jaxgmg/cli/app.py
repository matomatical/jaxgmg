"""
Entry point for jaxgmg CLI.

The functions that are invoked by each subcommand are elsewhere in the
jaxgmg.cli module.

This script just imports them and transforms them into a CLI application
using the Typer library.

The export of this module is a callable object `app`, that should be called
to launch the program. This is used in two places:

* It is imported and called in `jaxgmg.__main__`, supporting invocation of
  the CLI via `python -m jaxgmg`.
* It is referenced as entry point `jaxgmg.cli.app:app` inside pyproject.toml,
  supporting invocation of the CLI via the bare command `jaxgmg`.
"""

import typer

from jaxgmg.cli import util

from jaxgmg.cli import noisegen
from jaxgmg.cli import mazegen
from jaxgmg.cli import mazesoln
from jaxgmg.cli import parse
from jaxgmg.cli import play
from jaxgmg.cli import solve
from jaxgmg.cli import speedtest


# # #
# Configure the Typer application

TYPER_CONFIG = {
    'no_args_is_help': True,
    'add_completion': False,
    'pretty_exceptions_show_locals': False, # can turn on during debugging
}


# # # 
# Create the Typer application

app = typer.Typer(**TYPER_CONFIG)


# # # 
# Assemble various entrypoints into a Typer app

# noise generation
app.add_typer(util.make_typer_app(
    name='noisegen',
    help=noisegen.__doc__,
    subcommands=(
        noisegen.perlin,
        noisegen.fractal,
    ),
    **TYPER_CONFIG,
))


# maze generation
app.add_typer(util.make_typer_app(
    name='mazegen',
    help=mazegen.__doc__,
    subcommands=(
        mazegen.tree,
        mazegen.edges,
        mazegen.noise,
        mazegen.blocks,
        mazegen.open,
        mazegen.mural,
    ),
    **TYPER_CONFIG,
))


# maze solving
app.add_typer(util.make_typer_app(
    name='mazesoln',
    help=mazesoln.__doc__,
    subcommands=(
        mazesoln.distances,
        mazesoln.directions,
        mazesoln.distances_and_directions,
    ),
    **TYPER_CONFIG,
))


# play environments
app.add_typer(util.make_typer_app(
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
    **TYPER_CONFIG,
))


# testing parsers
app.add_typer(util.make_typer_app(
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
    **TYPER_CONFIG,
))


# solve environments
app.add_typer(util.make_typer_app(
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
    **TYPER_CONFIG,
))


# speedtests
app.add_typer(util.make_typer_app(
    name='speedtest',
    help=speedtest.__doc__,
    subcommands=(
        speedtest.envstep_corner,
        speedtest.envstep_dish,
        speedtest.envstep_follow,
        speedtest.envstep_keys,
        speedtest.envstep_lava,
        speedtest.envstep_monsters,
        speedtest.mazegen_tree,
        speedtest.mazegen_edges,
        speedtest.mazegen_blocks,
        speedtest.mazegen_noise,
        speedtest.mazegen_open,
        speedtest.mazesoln_distances,
        speedtest.mazesoln_directional_distances,
        speedtest.mazesoln_optimal_directions,
    ),
    **TYPER_CONFIG,
))


# training (FIXME: MOVE ENTRY POINT TO CLI MODULE)
from jaxgmg.baselines.run import train
app.command()(train)
