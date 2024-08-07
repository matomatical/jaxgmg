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
  supporting invocation of the CLI via the command `jaxgmg`.
"""

import typer

from jaxgmg.cli import heatmaps
from jaxgmg.cli import noisegen
from jaxgmg.cli import mazegen
from jaxgmg.cli import mazesoln
from jaxgmg.cli import parse
from jaxgmg.cli import play
from jaxgmg.cli import reevaluate
from jaxgmg.cli import solve
from jaxgmg.cli import speedtest
from jaxgmg.cli import splay
from jaxgmg.cli import train


# # #
# Helper function: Create a Typer application.


def make_typer_app(name, help, subcommands):
    """
    Transform a list of functions into a typer application.
    """
    app = typer.Typer(
        name=name,
        help=help,
        no_args_is_help=True,
        add_completion=False,
        pretty_exceptions_show_locals=False,
    )
    for subcommand in subcommands:
        app.command()(subcommand)
    return app


# # # 
# Create the Typer application


app = make_typer_app(
    name='jaxgmg',
    help="""
        JAX-based environments and RL baselines for studying goal
        misgeneralisation.
    """,
    subcommands=(),
)


# # # 
# Assemble various entrypoints as subapplications/subcommands.


# heatmaps
app.add_typer(make_typer_app(
    name='heatmaps',
    help=heatmaps.__doc__,
    subcommands=(
        heatmaps.corner,
    ),
))


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
        mazegen.mural,
    ),
))


# maze solving
app.add_typer(make_typer_app(
    name='mazesoln',
    help=mazesoln.__doc__,
    subcommands=(
        mazesoln.distances,
        mazesoln.directions,
        mazesoln.distances_and_directions,
    ),
))


# parsers
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


# re-evaluation
app.add_typer(make_typer_app(
    name='reevaluate',
    help=reevaluate.__doc__,
    subcommands=(
        reevaluate.corner,
        # reevaluate.dish,
        # reevaluate.follow,
        # reevaluate.keys,
        # reevaluate.lava,
        # reevaluate.monsters,
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
        # solve.keys, # not yet implemented
        # solve.lava, # not yet implemented
        # solve.monsters, # not yet implemented
    ),
))


# speedtests
app.add_typer(make_typer_app(
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
))


# splayer demonstration
app.add_typer(make_typer_app(
    name='splay',
    help=splay.__doc__,
    subcommands=(
        splay.corner,
    ),
))


# training
app.add_typer(make_typer_app(
    name='train',
    help=train.__doc__,
    subcommands=(
        train.corner,
        train.dish,
        train.pile,
        # train.follow,
        train.keys,
        # train.lava,
        # train.monsters,
    ),
))


