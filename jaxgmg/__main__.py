import jax
import jax.numpy as jnp
import numpy as np
import einops
import readchar
import time

import typer

from jaxgmg.procgen import maze_generation

from jaxgmg.environments import keys_and_chests
from jaxgmg.environments import monster_world
from jaxgmg.environments import cheese_in_the_corner


# # # 
# HELPER FUNCTIONS: PLAY AND VISUALISATION


def print_config(config):
    for key, value in config.items():
        print(f"  {key:30s}: {value}")


def play_forever(rng, env, level_generator, debug=False):
    """
    Helper function to interact with an environment.
    """
    while True:
        print("generating level...")
        rng_level, rng = jax.random.split(rng)
        level = level_generator.sample(rng_level)
        obs, state = env.reset_to_level(rng, level)

        print("playing level...")
        print("initial state")
        image = img2str(env.get_obs(state))
        lines = len(str(image).splitlines())
        print(
            image,
            "",
            "controls: w = up | a = left | s = down | d = right | q = quit",
            sep="\n",
        )
    
        rng_steps, rng = jax.random.split(rng)
        while True:
            key = readchar.readkey()
            if key == "q":
                print("bye!")
                return
            if key == "r":
                break
            if key not in "wsda":
                continue
            a = "wasd".index(key)
            rng_step, rng_steps = jax.random.split(rng_steps)
            _, state, r, d, _ = env.step(rng_step, state, a)
            print(
                "" if debug else f"\x1b[{lines+4}A",
                f"action: {a} ({'uldr'[a]})",
                img2str(env.get_obs(state)),
                f"reward: {r:.2f} done: {d}",
                "controls: w = up | a = left | s = down | d = right | q = quit",
                sep="\n",
            )
            if d:
                break
        print(f"\x1b[{lines+6}A")


def mutate_forever(
    rng,
    env,
    level_generator,
    level_mutator,
    fps,
    debug=False,
):
    """
    Helper function to repeatedly mutate and display a level.
    """
    # initial level
    rng_initial_level, rng_reset, rng = jax.random.split(rng, 3)
    level = level_generator.sample(rng_initial_level)
    obs, _ = env.reset_to_level(rng_reset, level)
    img = img2str(obs)
    lines = len(img.splitlines())
    print("initial level:", img, sep="\n")

    # mutation levels
    i = 1
    while True:
        rng_mutate, rng_reset, rng = jax.random.split(rng, 3)
        level = level_mutator.mutate(rng_mutate, level)
        obs, _ = env.reset_to_level(rng_reset, level)
        img = img2str(obs)
        print(
            "" if debug else f"\x1b[{lines+2}A",
            f"level after {i} mutations:",
            img,
            sep="\n",
        )
        time.sleep(1/fps)
        i += 1


def img2str(im, colormap=None):
    """
    Render a small image using a grid of unicode half-characters with
    different foreground and background colours to represent pairs of
    pixels.
    """
    im = np.asarray(im)
    # convert to RGB
    if len(im.shape) == 2 and colormap is None:
        im = einops.repeat(im, 'h w -> h w 3') # uniform colorization
    elif colormap is not None:
        im = colormap(im) # colormap: h w bw -> h w [r g b]
    # pad to even height (and width, latter not strictly necessary)
    im = np.pad(
        array=im,
        pad_width=(
            (0, im.shape[0] % 2),
            (0, im.shape[1] % 2),
            (0, 0),
        ),
        mode='constant',
        constant_values=0.,
    )
    # stack to fg/bg
    im = einops.rearrange(im, '(h h2) w c -> h w h2 c', h2=2)
    # render the image as a string with ansi color codes
    def _switch_color(fg, bg):
        fgr, fgg, fgb = (255 * fg).astype(np.uint8)
        bgr, bgg, bgb = (255 * bg).astype(np.uint8)
        return f"\033[38;2;{fgr};{fgg};{fgb}m\033[48;2;{bgr};{bgg};{bgb}m"
    _reset = "\033[0m"
    return f"{_reset}\n".join([
        "".join([_switch_color(fg, bg) + "▀" for fg, bg in row])
        for row in im
    ]) + _reset


def viridis(x):
    """
    https://youtu.be/xAoljeRJ3lU
    """
    return np.array([
        [.267,.004,.329],[.268,.009,.335],[.269,.014,.341],[.271,.019,.347],
        [.272,.025,.353],[.273,.031,.358],[.274,.037,.364],[.276,.044,.370],
        [.277,.050,.375],[.277,.056,.381],[.278,.062,.386],[.279,.067,.391],
        [.280,.073,.397],[.280,.078,.402],[.281,.084,.407],[.281,.089,.412],
        [.282,.094,.417],[.282,.100,.422],[.282,.105,.426],[.283,.110,.431],
        [.283,.115,.436],[.283,.120,.440],[.283,.125,.444],[.283,.130,.449],
        [.282,.135,.453],[.282,.140,.457],[.282,.145,.461],[.281,.150,.465],
        [.281,.155,.469],[.280,.160,.472],[.280,.165,.476],[.279,.170,.479],
        [.278,.175,.483],[.278,.180,.486],[.277,.185,.489],[.276,.190,.493],
        [.275,.194,.496],[.274,.199,.498],[.273,.204,.501],[.271,.209,.504],
        [.270,.214,.507],[.269,.218,.509],[.267,.223,.512],[.266,.228,.514],
        [.265,.232,.516],[.263,.237,.518],[.262,.242,.520],[.260,.246,.522],
        [.258,.251,.524],[.257,.256,.526],[.255,.260,.528],[.253,.265,.529],
        [.252,.269,.531],[.250,.274,.533],[.248,.278,.534],[.246,.283,.535],
        [.244,.287,.537],[.243,.292,.538],[.241,.296,.539],[.239,.300,.540],
        [.237,.305,.541],[.235,.309,.542],[.233,.313,.543],[.231,.318,.544],
        [.229,.322,.545],[.227,.326,.546],[.225,.330,.547],[.223,.334,.548],
        [.221,.339,.548],[.220,.343,.549],[.218,.347,.550],[.216,.351,.550],
        [.214,.355,.551],[.212,.359,.551],[.210,.363,.552],[.208,.367,.552],
        [.206,.371,.553],[.204,.375,.553],[.203,.379,.553],[.201,.383,.554],
        [.199,.387,.554],[.197,.391,.554],[.195,.395,.555],[.194,.399,.555],
        [.192,.403,.555],[.190,.407,.556],[.188,.410,.556],[.187,.414,.556],
        [.185,.418,.556],[.183,.422,.556],[.182,.426,.557],[.180,.429,.557],
        [.179,.433,.557],[.177,.437,.557],[.175,.441,.557],[.174,.445,.557],
        [.172,.448,.557],[.171,.452,.557],[.169,.456,.558],[.168,.459,.558],
        [.166,.463,.558],[.165,.467,.558],[.163,.471,.558],[.162,.474,.558],
        [.160,.478,.558],[.159,.482,.558],[.157,.485,.558],[.156,.489,.557],
        [.154,.493,.557],[.153,.497,.557],[.151,.500,.557],[.150,.504,.557],
        [.149,.508,.557],[.147,.511,.557],[.146,.515,.556],[.144,.519,.556],
        [.143,.522,.556],[.141,.526,.555],[.140,.530,.555],[.139,.533,.555],
        [.137,.537,.554],[.136,.541,.554],[.135,.544,.554],[.133,.548,.553],
        [.132,.552,.553],[.131,.555,.552],[.129,.559,.551],[.128,.563,.551],
        [.127,.566,.550],[.126,.570,.549],[.125,.574,.549],[.124,.578,.548],
        [.123,.581,.547],[.122,.585,.546],[.121,.589,.545],[.121,.592,.544],
        [.120,.596,.543],[.120,.600,.542],[.119,.603,.541],[.119,.607,.540],
        [.119,.611,.538],[.119,.614,.537],[.119,.618,.536],[.120,.622,.534],
        [.120,.625,.533],[.121,.629,.531],[.122,.633,.530],[.123,.636,.528],
        [.124,.640,.527],[.126,.644,.525],[.128,.647,.523],[.130,.651,.521],
        [.132,.655,.519],[.134,.658,.517],[.137,.662,.515],[.140,.665,.513],
        [.143,.669,.511],[.146,.673,.508],[.150,.676,.506],[.153,.680,.504],
        [.157,.683,.501],[.162,.687,.499],[.166,.690,.496],[.170,.694,.493],
        [.175,.697,.491],[.180,.701,.488],[.185,.704,.485],[.191,.708,.482],
        [.196,.711,.479],[.202,.715,.476],[.208,.718,.472],[.214,.722,.469],
        [.220,.725,.466],[.226,.728,.462],[.232,.732,.459],[.239,.735,.455],
        [.246,.738,.452],[.252,.742,.448],[.259,.745,.444],[.266,.748,.440],
        [.274,.751,.436],[.281,.755,.432],[.288,.758,.428],[.296,.761,.424],
        [.304,.764,.419],[.311,.767,.415],[.319,.770,.411],[.327,.773,.406],
        [.335,.777,.402],[.344,.780,.397],[.352,.783,.392],[.360,.785,.387],
        [.369,.788,.382],[.377,.791,.377],[.386,.794,.372],[.395,.797,.367],
        [.404,.800,.362],[.412,.803,.357],[.421,.805,.351],[.430,.808,.346],
        [.440,.811,.340],[.449,.813,.335],[.458,.816,.329],[.468,.818,.323],
        [.477,.821,.318],[.487,.823,.312],[.496,.826,.306],[.506,.828,.300],
        [.515,.831,.294],[.525,.833,.288],[.535,.835,.281],[.545,.838,.275],
        [.555,.840,.269],[.565,.842,.262],[.575,.844,.256],[.585,.846,.249],
        [.595,.848,.243],[.606,.850,.236],[.616,.852,.230],[.626,.854,.223],
        [.636,.856,.216],[.647,.858,.209],[.657,.860,.203],[.668,.861,.196],
        [.678,.863,.189],[.688,.865,.182],[.699,.867,.175],[.709,.868,.169],
        [.720,.870,.162],[.730,.871,.156],[.741,.873,.149],[.751,.874,.143],
        [.762,.876,.137],[.772,.877,.131],[.783,.879,.125],[.793,.880,.120],
        [.804,.882,.114],[.814,.883,.110],[.824,.884,.106],[.835,.886,.102],
        [.845,.887,.099],[.855,.888,.097],[.866,.889,.095],[.876,.891,.095],
        [.886,.892,.095],[.896,.893,.096],[.906,.894,.098],[.916,.896,.100],
        [.926,.897,.104],[.935,.898,.108],[.945,.899,.112],[.955,.901,.118],
        [.964,.902,.123],[.974,.903,.130],[.983,.904,.136],[.993,.906,.143],
    ])[(np.clip(x, 0., 1.) * (255)).astype(int)]


def sweetie16(x):
    """
    https://lospec.com/palette-list/sweetie-16
    """
    return np.array([
        [.101,.109,.172],[.364,.152,.364],[.694,.243,.325],[.937,.490,.341],
        [.999,.803,.458],[.654,.941,.439],[.219,.717,.392],[.145,.443,.474],
        [.160,.211,.435],[.231,.364,.788],[.254,.650,.964],[.450,.937,.968],
        [.956,.956,.956],[.580,.690,.760],[.337,.423,.525],[.2  ,.235,.341],
    ])[x]


def print_legend(legend, colormap=None):
    print("legend:")
    for value, name in legend.items():
        print(img2str(jnp.full((2,2), value), colormap=colormap,), name)


def print_histogram(data, bins=10, range=None, width=40):
    hist, bin_edges = jnp.histogram(data, bins=bins, range=range)
    norm_counts = hist / hist.max()
    for count, lo, hi in zip(norm_counts, bin_edges, bin_edges[1:]):
        print(f"  {lo:.2f} to {hi:.2f} | {int(count * width + 1) * '*'}")


# # # 
# MAZE GENERATION/SOLUTION FUNCTIONALITY


def maze_gen(
    layout: str = 'tree',
    height: int = 15,
    width:  int = 79,
    seed: int = 42,
):
    print("maze-gen: generate and visualise a random maze")
    print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    gen = maze_generation.get_generator_function(layout)
    maze = gen(
        key=rng,
        h=height,
        w=width,
    )
    print(img2str(maze * .25))


def maze_distance(
    layout: str = 'tree',
    height: int = 5,
    width:  int = 5,
    seed: int = 42,
):
    print(
        "maze-distance: solve a maze and plot the optimal distance "
        "from any source to any destination"
    )
    print_config(locals())

    print("generating maze...")
    rng = jax.random.PRNGKey(seed=seed)
    gen = maze_generation.get_generator_function(layout)
    maze = gen(key=rng, h=height, w=width)
    print(img2str(maze * .25))

    print("solving maze...")
    dist = maze_generation.maze_distances(maze)

    print("visualising solution...")
    # transform inf -> -inf, [0,max] -> [0.5,1.0] for visualisation
    dist_finite = jnp.nan_to_num(dist, posinf=-jnp.inf) # to allow max
    maxd = dist_finite.max()
    dist_01 = (maxd + jnp.nan_to_num(dist_finite, neginf=-maxd)) / (2*maxd)
    # rearrange into macro/micro maze format
    print(img2str(
        einops.rearrange(dist_01, 'H W h w -> (H h) (W w)'),
        colormap=viridis,
    ))
    print_legend({
        0.0: "wall/unreachable",
        0.5: "distance 0",
        1.0: f"distance {maxd.astype(int)}",
    }, colormap=viridis)
    print("the source is represented by the square in the macromaze")
    print("the target is represented by the square in the micromaze")


def maze_direction(
    layout: str = 'tree',
    height: int = 5,
    width:  int = 5,
    stay_action: bool = True,
    seed: int = 42,
):
    print(
        "maze-direction: solve a maze and plot the optimal direction "
        "from any source to any destination"
    )
    print_config(locals())

    print("generating maze...")
    rng = jax.random.PRNGKey(seed=seed)
    gen = maze_generation.get_generator_function(layout)
    maze = gen(key=rng, h=height, w=width)
    print(img2str(maze * .25))

    print("solving maze...")
    soln = maze_generation.maze_optimal_directions(maze, stay_action=stay_action)

    print("visualising directions...")
    # transform {0,1,2,3} -> {1,2,3,4} and walls -> 0
    soln = (1 + soln) * ~maze * ~maze[:,:,None,None]
    # rearrange into macro/micro maze format
    print(img2str(
        einops.rearrange(soln, 'H W h w -> (H h) (W w)'),
        colormap=sweetie16,
    ))
    print_legend({
        0: "n/a",
        1: "up",
        2: "left",
        3: "down",
        4: "right",
        5: "(stay)",
    }, colormap=sweetie16)
    print("the source is represented by the square in the macromaze")
    print("the target is represented by the square in the micromaze")


# # # 
# ENVIRONMENTS


def play_corner(
    height: int                 = 13,
    width: int                  = 9,
    layout: str                 = 'tree',
    corner_size: int            = 3,
    seed: int                   = 42,
    debug: bool                 = False,
):
    print("corner: interact with a random cheese in the corner level")
    print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    env = cheese_in_the_corner.Env(rgb=True)
    level_generator = cheese_in_the_corner.LevelGenerator(
        height=height,
        width=width,
        layout=layout,
        corner_size=corner_size,
    )
    play_forever(
        rng=rng,
        env=env,
        level_generator=level_generator,
        debug=debug,
    )


def play_keys(
    height: int                 = 13,
    width: int                  = 9,
    layout: str                 = 'tree',
    num_keys_min: int           = 2,
    num_keys_max: int           = 6,
    num_chests_min: int         = 6,
    num_chests_max: int         = 6,
    seed: int                   = 42,
    debug: bool                 = False,
):
    print("keys: interact with a random keys and chests level")
    print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    env = keys_and_chests.Env(rgb=True)
    level_generator = keys_and_chests.LevelGenerator(
        height=height,
        width=width,
        layout=layout,
        num_keys_min=num_keys_min,
        num_keys_max=num_keys_max,
        num_chests_min=num_chests_min,
        num_chests_max=num_chests_max,
    )
    play_forever(
        rng=rng,
        env=env,
        level_generator=level_generator,
        debug=debug,
    )


def play_monsters(
    height: int                 = 13,
    width: int                  = 9,
    layout: str                 = 'open',
    num_apples: int             = 5,
    num_shields: int            = 5,
    num_monsters: int           = 5,
    monster_optimality: float   = 3,
    seed: int                   = 42,
    debug: bool                 = False,
):
    print("monsters: interact with a random monster world level")
    print_config(locals())

    rng = jax.random.PRNGKey(seed=seed)
    env = monster_world.Env(rgb=True)
    level_generator = monster_world.LevelGenerator(
        height=height,
        width=width,
        layout=layout,
        num_apples=num_apples,
        num_shields=num_shields,
        num_monsters=num_monsters,
        monster_optimality=monster_optimality,
    )
    play_forever(
        rng=rng,
        env=env,
        level_generator=level_generator,
        debug=debug,
    )
    

# # # 
# SOLVING ENVIRONMENTS


def solve_corner(
    height: int                 = 13,
    width: int                  = 13,
    layout: str                 = 'tree',
    corner_size: int            = 3,
    penalize_time: bool         = True,
    max_steps_in_episode: int   = 128,
    discount_rate: float        = 0.995,
    num_levels: int             = 64,
    seed: int                   = 42,
):
    print(
        "solve-corner: optimal value for a random cheese in the corner level"
    )
    print_config(locals())

    print("initialising...")
    rng = jax.random.PRNGKey(seed=seed)
    env = cheese_in_the_corner.Env(
        rgb=False,
        penalize_time=penalize_time,
        max_steps_in_episode=max_steps_in_episode,
    )
    level_generator = cheese_in_the_corner.LevelGenerator(
        height=height,
        width=width,
        layout=layout,
        corner_size=corner_size,
    )

    print("generating levels...")
    rng_level, rng = jax.random.split(rng)
    levels = level_generator.vsample(rng_level, num_levels)
    rng_reset, rng = jax.random.split(rng)
    obs, state = env.vreset_to_level(rng_reset, levels)

    print("visualising first level (indicative)...")
    img = (
        1 * obs[0,:,:,env.Channel.WALL]
        + 4 * obs[0,:,:,env.Channel.CHEESE]
        + 12 * obs[0,:,:,env.Channel.MOUSE]
    )
    print(img2str(img, colormap=sweetie16))
    print_legend({
        1: "wall",
        4: "cheese",
        12: "mouse",
    }, colormap=sweetie16)

    print("solving levels...")
    v = jax.vmap(
        env.optimal_value,
        in_axes=(0,None),
    )(
        levels,
        discount_rate,
    )

    print('optimal values:')
    print_histogram(v)
    print('average optimal value:', v.mean())
    print('std dev optimal value:', v.std())


def solve_keys(
    height: int                 = 13,
    width: int                  = 13,
    layout: str                 = 'tree',
    num_keys_min: int           = 2,
    num_keys_max: int           = 4,
    num_chests_min: int         = 4,
    num_chests_max: int         = 4,
    penalize_time: bool         = True,
    max_steps_in_episode: int   = 128,
    discount_rate: float        = 0.995,
    num_levels: int             = 64,
    seed: int                   = 42,
):
    print("solve-keys: optimal value for a random keys and chests level")
    print_config(locals())
    if num_keys_max + num_chests_max > 9:
        print(
            f"WARNING: attempting to solve environments with "
            f"{num_keys_max} + {num_chests_max} "
            f"= {num_keys_max + num_chests_max} > 9 keys/chests. "
            "This may take a while."
        )

    print("initialising...")
    rng = jax.random.PRNGKey(seed=seed)
    level_generator = keys_and_chests.LevelGenerator(
        height=height,
        width=width,
        layout=layout,
        num_keys_min=num_keys_min,
        num_keys_max=num_keys_max,
        num_chests_min=num_chests_min,
        num_chests_max=num_chests_max,
    )
    env = keys_and_chests.Env(
        rgb=False,
        penalize_time=penalize_time,
        max_steps_in_episode=max_steps_in_episode,
    )

    print("generating levels...")
    rng_level, rng = jax.random.split(rng)
    levels = level_generator.vsample(rng_level, num_levels)
    rng_reset, rng = jax.random.split(rng)
    obs, state = env.vreset_to_level(rng_reset, levels)
    
    print("visualising first level (indicative)...")
    img = (
        1 * obs[0,:,:,env.Channel.WALL]
        + 2 * obs[0,:,:,env.Channel.CHEST]
        + 4 * obs[0,:,:,env.Channel.KEY]
        + 12 * obs[0,:,:,env.Channel.MOUSE]
    )
    print(img2str(img, colormap=sweetie16))
    print_legend({
        1: "wall",
        2: "chest",
        4: "key",
        12: "mouse",
    }, colormap=sweetie16)

    print("solving levels...")
    
    v = jax.vmap(
        env.optimal_value,
        in_axes=(0,None),
    )(
        levels,
        discount_rate,
    )

    print('optimal values:')
    print_histogram(v)
    print('average optimal value:', v.mean())
    print('std dev optimal value:', v.std())


def solve_monsters(
    height: int                 = 13,
    width: int                  = 9,
    layout: str                 = 'open',
    num_apples: int             = 5,
    num_shields: int            = 5,
    num_monsters: int           = 5,
    monster_optimality: float   = 3,
    penalize_time: bool         = True,
    max_steps_in_episode: int   = 128,
    discount_rate: float        = 0.995,
    seed: int                   = 42,
):
    print("solve-monsters: not yet implemented, sorry")

    
# # # 
# ENVIRONMENT MUTATORS


def mutate_corner(
    height: int                 = 13,
    width: int                  = 13,
    layout: str                 = 'open',
    corner_size: int            = 11,
    prob_wall_spawn: float      = 0.04,
    prob_wall_despawn: float    = 0.05,
    mouse_scatter: bool         = False,
    max_mouse_steps: int        = 2,
    cheese_scatter: bool        = True,
    max_cheese_steps: int       = 0,
    mut_corner_size: int        = 11,
    seed: int                   = 42,
    fps: int                    = 25,
    debug: bool                 = False,
):
    print(
        "mutate-corner: generate mutations of a random cheese in the corner "
        "level"
    )
    print_config(locals())

    print("initialising...")
    rng = jax.random.PRNGKey(seed=seed)
    env = cheese_in_the_corner.Env(
        rgb=True,
    )
    level_generator = cheese_in_the_corner.LevelGenerator(
        height=height,
        width=width,
        layout=layout,
        corner_size=corner_size,
    )
    level_mutator = cheese_in_the_corner.LevelMutator(
        prob_wall_spawn=prob_wall_spawn,
        prob_wall_despawn=prob_wall_despawn,
        mouse_scatter=mouse_scatter,
        max_mouse_steps=max_mouse_steps,
        cheese_scatter=cheese_scatter,
        max_cheese_steps=max_cheese_steps,
        corner_size=mut_corner_size,
    )

    mutate_forever(
        rng=rng,
        env=env,
        level_generator=level_generator,
        level_mutator=level_mutator,
        fps=fps,
        debug=debug,
    )


def mutate_keys(
    height: int                 = 13,
    width: int                  = 13,
    layout: str                 = 'open',
    num_keys_min: int           = 2,
    num_keys_max: int           = 6,
    num_chests_min: int         = 4,
    num_chests_max: int         = 4,
    prob_wall_spawn: float      = 0.04,
    prob_wall_despawn: float    = 0.05,
    prob_scatter: float         = 0.1,
    max_steps: int              = 1,
    prob_num_keys_step: float   = 0.1,
    mut_num_keys_min: int       = 2,
    mut_num_keys_max: int       = 6,
    prob_num_chests_step: float = 0.1,
    mut_num_chests_min: int     = 1,
    mut_num_chests_max: int     = 4,
    seed: int                   = 42,
    fps: int                    = 25,
    debug: bool                 = False,
):
    print(
        "mutate-keys: generate mutations of a random keys and chests level"
    )
    print_config(locals())

    print("initialising...")
    rng = jax.random.PRNGKey(seed=seed)
    env = keys_and_chests.Env(
        rgb=True,
    )
    level_generator = keys_and_chests.LevelGenerator(
        height=height,
        width=width,
        layout=layout,
        num_keys_min=num_keys_min,
        num_keys_max=num_keys_max,
        num_chests_min=num_chests_min,
        num_chests_max=num_chests_max,
    )
    level_mutator = keys_and_chests.LevelMutator(
        prob_wall_spawn=prob_wall_spawn,
        prob_wall_despawn=prob_wall_despawn,
        prob_scatter=prob_scatter,
        max_steps=max_steps,
        prob_num_keys_step=prob_num_keys_step,
        num_keys_min=mut_num_keys_min,
        num_keys_max=mut_num_keys_max,
        prob_num_chests_step=prob_num_chests_step,
        num_chests_min=mut_num_chests_min,
        num_chests_max=mut_num_chests_max,
    )

    mutate_forever(
        rng=rng,
        env=env,
        level_generator=level_generator,
        level_mutator=level_mutator,
        fps=fps,
        debug=debug,
    )


def mutate_monsters(
    height: int                     = 13,
    width: int                      = 13,
    layout: str                     = 'open',
    num_apples: int                 = 5,
    num_shields: int                = 5,
    num_monsters: int               = 5,
    monster_optimality: int         = 3,
    prob_wall_spawn: float          = 0.01,
    prob_wall_despawn: float        = 0.3,
    prob_scatter: float             = 0.1,
    max_steps: int                  = 1,
    monster_optimality_step: float  = 0.5,
    seed: int                       = 42,
    fps: int                        = 25,
):
    print(
        "mutate-monsters: generate mutations of a random monster world level"
    )
    print_config(locals())

    print("initialising...")
    rng = jax.random.PRNGKey(seed=seed)
    env = monster_world.Env(
        rgb=True,
    )
    level_generator = monster_world.LevelGenerator(
        height=height,
        width=width,
        layout=layout,
        num_apples=num_apples,
        num_shields=num_shields,
        num_monsters=num_monsters,
        monster_optimality=monster_optimality,
    )
    level_mutator = monster_world.LevelMutator(
        prob_wall_spawn=prob_wall_spawn,
        prob_wall_despawn=prob_wall_despawn,
        prob_scatter=prob_scatter,
        max_steps=max_steps,
        monster_optimality_step=monster_optimality_step,
    )

    mutate_forever(
        rng=rng,
        env=env,
        level_generator=level_generator,
        level_mutator=level_mutator,
        fps=fps,
    )


# # # 
# ENTRY POINT
app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False, # can turn on during debugging
)

# maze generation and solving
app.command()(maze_gen)
app.command()(maze_distance)
app.command()(maze_direction)

# play environments
app.command()(play_corner)
app.command()(play_keys)
app.command()(play_monsters)

# solve environments
app.command()(solve_corner)
app.command()(solve_keys)
app.command()(solve_monsters) # TODO

# mutate environments
app.command()(mutate_corner)
app.command()(mutate_keys)
app.command()(mutate_monsters)

# let's go!
app()

