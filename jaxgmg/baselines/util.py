import functools
import datetime
import os
import json
from PIL import Image
import numpy as np
import einops
import wandb


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
