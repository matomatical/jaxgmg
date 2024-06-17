import functools
import datetime
import os
import json
from PIL import Image
import numpy as np
import einops
import wandb


# # # 
# WANDB WRAPPERS


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


# # # 
# RUN FILES MANAGEMENT


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
# RENDERING AND SAVING DICTIONARIES


def dict_prefix(dct, prefix):
    return {prefix + key: value for key, value in dct.items()}


def dict2str(dct):
    return "\n".join([f"  {key:33}: {value}" for key, value in dct.items()])


def save_json(dct, path):
    with open(path, 'w') as outfile:
        json.dump(dct, outfile, indent=True)


# # # 
# RENDERING AND SAVING IMAGES AND ANIMATIONS


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
                The animation. First axis is time, remaining axes represent
                the image data. Each point should be a float between 0 and 1.
    * path : str
                Where to save the gif.
    * upscale : int (>=1, default is 1)
                Width/height of pixel representation of each matrix entry.
    * fps : int
                Approx. frames per second encoded into the gif.
    * repeat : bool (default True)
                Whether the gif loops indefinitely (True, default) or only
                plays once (False).
    """
    T, H, W, C = frames.shape
    assert C == 3, f"too many channels ({C}>3) to create gif"
        
    # preprocess image data
    frames_u8 = (np.asarray(frames) * 255).astype(np.uint8)
    frames_u8_upscaled = einops.repeat(
        frames_u8,
        't h w rgb -> t (h sh) (w sw) rgb',
        sh=upscale,
        sw=upscale,
    )
    # PIL images for each frame
    imgs = [Image.fromarray(i) for i in frames_u8_upscaled]
    # compile gif
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=1000 // fps,
        loop=1-repeat,
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
