`jaxgmg`: JAX-accelerated Goal MisGeneralisation
================================================

JAX-based environments and RL baselines for studying goal misgeneralisation.


JAX-accelerated environments
----------------------------

TODO: demo gifs.

TODO: speedtests.


RL baselines
---------

TODO: implement baselines.


Procedural level generation
---------------------------

TODO: demo gifs.

TODO: speedtests.


Install
-------

Install the latest master version from GitHub:

```
pip install git+ssh://git@github.com/matomatical/jaxgmg.git
```

Install from a local clone:

```
git clone git@github.com:matomatical/jaxgmg.git
cd jaxgmg
pip install -e .
```


Explore
-------

After installing run the following:

```
jaxgmg --help
```

You can try the various subcommands to see demonstrations of the library's
functionality. For example:

* To play with an interactive demonstration of the environments, try
  `jaxgmg play ENV_NAME` (see `jaxgmg play --help` for options).
* To procedurally generate mazes , try `jaxgmg mazegen LAYOUT` (see `jaxgmg
  mazegen --help` for options).

Note: Most of the demos display colour images to the terminal using ANSI
control codes, which may not work in some environments (e.g. on Windows?).


Roadmap
-------

Core maze generation methods (JAX accelerated):

* [x] Kruskal's algorithm
* [x] Random block mazes
* [x] Perlin noise
* [ ] More: Rooms? BSP? Tunnelling? Cellular automata? See notes
      [here](https://christianjmills.com/posts/procedural-map-generation-techniques-notes/).

Core environments (JAX accelerated):

* [x] Cheese in the corner
* [x] Keys and chests
* [x] Monster world
* [x] Cheese on a dish
* [x] Lava land
* [x] Follow the leader (simplified 'cultural transmission')
* [ ] Forest recovery

Environment features:

* [ ] Variable-resolution RGB rendering
* [ ] Partially observable versions of environments

RL baselines:

* [ ] Train PPO agents in the above environments (symbolic)
* [ ] Train PPO agents in the above environments (small pixels)
* [ ] Qualitative and quantitative demonstration of goal misgeneralisation

Release:

* [x] Create this repository
* [x] Format project as an installable Python package
* [ ] Document experiments in a report
* [ ] Release v1 on arXiv and PyPI

---

Stretch environments:

* [ ] Coin at the end (simplified 'coinrun'-style platformer)
* [ ] Survivor ('crafter'-style mining/farming grid world)
* [ ] Dungeon (a simple roguelike)
* [ ] More procgen games

Stretch environment features:

* [ ] Procgen-style variable-size mazes
* [ ] Procgen-style sprite and background diversity


