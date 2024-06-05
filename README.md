Goal misgeneralisation benchmark environments
=============================================

JAX-based environments and RL baselines for studying goal misgeneralisation.

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

Roadmap
-------

Core maze generation methods:

* [ ]

Core environments:

* [x] Cheese in the corner
* [x] Keys and chests
* [x] Monster world
* [ ] Cheese on a dish
* [ ] Lava land
* [ ] Follow the leader (simplified 'cultural transmission')
* [ ] Forest recovery

Environment features:

* [ ] Variable-resolution RGB rendering
* [ ] Partially observable versions of environments

RL baselines:

* [ ] Train PPO agents in the above environments (symbolic)
* [ ] Train PPO agents in the above environments (small pixels)
* [ ] Qualitative and quantitative demonstration of goal misgeneralisation

Release:

* [ ] Create this repository
* [ ] Installable python package
* [ ] Document experiments in a report
* [ ] Release v1 on arXiv and PyPI

Stretch environments:

* [ ] Coin at the end (simplified 'coinrun'-style platformer)
* [ ] Survivor ('crafter'-style mining/farming grid world)
* [ ] Dungeon (a simple roguelike)
* [ ] More procgen games

Stretch environment features:

* [ ] Procgen-style variable-size mazes
* [ ] Procgen-style sprite and background diversity

