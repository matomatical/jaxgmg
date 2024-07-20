"""
Utilities for transforming data or rendering it to strings/stdout/disk/wandb
in support of jaxgmg CLI application and training scripts.
"""

import functools
import os
import datetime
import numpy as np
import jax
import jax.numpy as jnp
import einops
import PIL.Image as pillow
import wandb
import json


# # # 
# Rendering data / images / plots to strings


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


def dict2str(dct):
    def dict2lines(dct, depth):
        for key, value in dct.items():
            if isinstance(value, dict):
                yield (depth, key, '')
                yield from dict2lines(value, depth+1)
            else:
                yield (depth, key, value)
    def render(value):
        if isinstance(value, jax.Array) and value.shape != ():
            return f'{value.dtype}{value.shape}'
        else:
            return str(value)
    return '\n'.join([
        '  '*depth + (key + ':').ljust(48-2*depth) + render(value)
        for depth, key, value in dict2lines(dct, 1)
    ])


# # # 
# Rendering data / images / plots to stdout


def print_config(config: dict):
    """
    Dump a dictionary's contents to stdout.
    """
    for key, value in config.items():
        print(f"  {key:30s}: {value}")


def print_histogram(data, bins=10, range=None, width=40):
    """
    Bin and count a sequence of values and print them to stdout as an ASCII
    histogram.
    """
    hist, bin_edges = jnp.histogram(data, bins=bins, range=range)
    norm_counts = hist / hist.max()
    for count, lo, hi in zip(norm_counts, bin_edges, bin_edges[1:]):
        print(f"  {lo:.2f} to {hi:.2f} | {int(count * width + 1) * '*'}")


def print_legend(legend, colormap=None):
    """
    Render a mapping from colors to values to stdout.
    """
    print("legend:")
    for value, name in legend.items():
        print(img2str(jnp.full((2,2), value), colormap=colormap,), name)


def print_img(im, colormap=None):
    """
    Render a small image to stdout using unicode half-block characters to
    represent pairs of pixels.
    """
    print(img2str, colormap=colormap)


# # # 
# Saving images or animations to disk


def save_image(
    image,
    path,
    upscale=1,
):
    """
    Take a (height, width, rgb) matrix and save it as a png.
    
    Parameters:

    * image: float[h, w, rgb]
            The image. Axes represent the image data. Each point should be a
            float between 0 and 1.
    * path : str
            Where to save the image.
    * upscale : int (>=1, default is 1)
            Width/height of pixel representation of each matrix entry.
    """
    image = np.asarray(image)
    H, W, C = image.shape
    assert C == 3, f"Wrong number of channels for GIF (C={C}!=3)"
        
    # preprocess image data
    image_u8 = (image * 255).astype(np.uint8)
    image_u8_upscaled = einops.repeat(
        image_u8,
        'h w rgb -> (h sh) (w sw) rgb',
        sh=upscale,
        sw=upscale,
    )
    # PIL images for each frame
    img = pillow.fromarray(image_u8_upscaled)
    # save!
    img.save(path)


def save_gif(
    frames,
    path,
    upscale=1,
    fps=12,
    repeat=True,
):
    """
    Take a (time, height, width, rgb) matrix and save it as an animated gif.
    
    Parameters:

    * frames : float[t, h, w, rgb]
            The animation. First axis is time, remaining axes represent the
            image data. Each point should be a float between 0 and 1.
    * path : str
            Where to save the gif.
    * upscale : int (>=1, default is 1)
            Width/height of pixel representation of each matrix entry.
    * fps : int
            Approx. frames per second encoded into the gif.
    * repeat : bool (default True)
            Whether the gif loops indefinitely (True, default) or only plays
            once (False).
    """
    frames = np.asarray(frames)
    T, H, W, C = frames.shape
    assert C == 3, f"Wrong number of channels for GIF (C={C}!=3)"
        
    # preprocess image data
    frames_u8 = (frames * 255).astype(np.uint8)
    frames_u8_upscaled = einops.repeat(
        frames_u8,
        't h w rgb -> t (h sh) (w sw) rgb',
        sh=upscale,
        sw=upscale,
    )
    # PIL images for each frame
    imgs = [pillow.fromarray(i) for i in frames_u8_upscaled]
    # compile gif
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=1000 // fps,
        loop=1-repeat, # 1 = loop once, 0 = loop forever
    )


def save_json(dct, path):
    with open(path, 'w') as outfile:
        json.dump(dct, outfile, indent=True)


# # # 
# wandb wrappers and formatting functions


def wandb_img(image):
    """
    Format a gif as a video for wandb.

    Parameters:

    * image : float[h, w, rgb]
            RGB floats each channel in range [0,1].

    Returns:

    * image : wandb.Image (contains uint8[h, w, c]).
            RGB Image including this data in the required format.

    """
    return wandb.Image(
        np.asarray(
            255 * image,
            dtype=np.uint8,
        ),
    )


def wandb_gif(frames, fps=12):
    """
    Format a gif as a video for wandb.

    Parameters:

    * frames : float[t h w c]
            RGB floats each channel in range [0,1].
    * fps : int = 12
            Frames per second for the wandb video.

    Returns:

    * video : wandb.Video (contains uint8[t c h w]).
            RGB video including this data in the required format.

    """
    return wandb.Video(
        np.asarray(
            255 * einops.rearrange(frames, 't h w c -> t c h w'),
            dtype=np.uint8,
        ),
        fps=fps,
    )


# # # 
# Training run file/wandb management


def wandb_run(f):
    """
    Decorator to initialise (and finish) wandb runs associated with a
    function, while syncing the keyword arguments of the function and the
    wandb run config.

    With this you can write functions that are configured by their arguments
    rather than a `config` dictionary. In turn this allows using annotation-
    based CLI generators like Typer or plac to neatly call the function from
    the command line.
    
    The function `f` must satisfy some mild requirements:

    1. The function *must* have at least one keyword argument `wandb_log`,
       a Boolean indicating whether to activate wandb. The wrapper will call
       `wandb.init` if and only if this argument is True, and accordingly the
       function should only call `wandb.log` in this case.
    
    2. The function *may* have additional keyword arguments matching those of
       `wandb.init` (https://docs.wandb.ai/ref/python/init) with an
       additional prefix `wandb_`. These arguments will be passed to
       `wandb.init`.

       For example, if there is an argument `wandb_project` then `wandb.init`
       will be called with `project` set to its value.

       Note: not all `wandb_init` parameters are supported right now, check
       source code; it's pretty easy to add more.

    Note: The function can include positional arguments, but these are not
    passed to wandb.init.
    """
    @functools.wraps(f)
    def g(*args, **kwargs):
        if kwargs['wandb_log']:
            # convert kwargs into config dictionary
            # exclude those that are passed to the init function directly
            config = wandb.helper.parse_config(
                kwargs,
                exclude=(
                    'wandb_log',
                    'wandb_entity',
                    'wandb_project',
                    'wandb_group',
                    'wandb_name',
                    'wandb_notes',
                    'wandb_tags',
                    'wandb_config_exclude_keys',
                    'wandb_config_include_keys',
                    *kwargs.get('wandb_config_exclude_keys', ()),
                ),
                include=[
                    *kwargs.get('wandb_config_include_keys', ()),
                ],
            )
            with wandb.init(
                config=config,
                # locate / describe the run
                entity=kwargs.get('wandb_entity', None),
                project=kwargs.get('wandb_project', None),
                group=kwargs.get('wandb_group', None),
                name=kwargs.get('wandb_name', None),
                notes=kwargs.get('wandb_notes', None),
                tags=kwargs.get('wandb_tags', None),
            ):
                # for the function call, update kwargs with any changes from wandb
                # (e.g. during sweeps) before passing to the run function
                kwargs.update(wandb.config)
                return f(*args, **kwargs)
        else:
            return f(*args, **kwargs)
    return g


class RunFilesManager:
    def __init__(self, root_path="out/"):
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.path = os.path.abspath(os.path.join(root_path, f'run_{now}'))
        os.makedirs(self.path, exist_ok=False)


    def get_path(self, suffix):
        path = os.path.join(self.path, suffix)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path


# # # 
# Transforming dictionaries


def flatten_dict(nested_dict, separator='/'):
    merged_dict = {}
    for key, inner_dict in nested_dict.items():
        if isinstance(inner_dict, dict):
            for inner_key, value in flatten_dict(inner_dict).items():
                merged_dict[key + separator + inner_key] = value
        else:
            merged_dict[key] = inner_dict
    return merged_dict


# # # 
# Colormaps


def viridis(x):
    """
    Viridis colormap.

    Details: https://youtu.be/xAoljeRJ3lU
    """
    return jnp.array([
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
    ])[(jnp.clip(x, 0., 1.) * (255)).astype(int)]


def sweetie16(x):
    """
    Sweetie-16 colour palette.

    Details: https://lospec.com/palette-list/sweetie-16
    """
    return jnp.array([
        [.101,.109,.172],[.364,.152,.364],[.694,.243,.325],[.937,.490,.341],
        [.999,.803,.458],[.654,.941,.439],[.219,.717,.392],[.145,.443,.474],
        [.160,.211,.435],[.231,.364,.788],[.254,.650,.964],[.450,.937,.968],
        [.956,.956,.956],[.580,.690,.760],[.337,.423,.525],[.2  ,.235,.341],
    ])[x]


def pico8(x):
    """
    PICO-8 colour palette.

    Details: https://pico-8.fandom.com/wiki/Palette
    """
    return (jnp.array([
        [  0,   0,   0], [ 29,  43,  83], [126,  37,  83], [  0, 135,  81],
        [171,  82,  54], [ 95,  87,  79], [194, 195, 199], [255, 241, 232],
        [255,   0,  77], [255, 163,   0], [255, 236,  39], [  0, 228,  54],
        [ 41, 173, 255], [131, 118, 156], [255, 119, 168], [255, 204, 170],
    ]) / 255)[x]


