import imageio
import einops
import jax.numpy as jnp
from importlib.resources import files

_SPRITESHEET_PATH = files('jaxgmg.environments').joinpath('spritesheet.png')
_SPRITESHEET = jnp.array(
    einops.rearrange(
        imageio.v2.imread(_SPRITESHEET_PATH) / 255,
        '(H h) (W w) c -> H W h w c',
        h=8,
        w=8,
    )
)


BLANK               = _SPRITESHEET[0,0]
PATH                = _SPRITESHEET[0,1]
WALL                = _SPRITESHEET[0,2]
MOUSE               = _SPRITESHEET[0,3]
CHEESE              = _SPRITESHEET[0,4]

KEY                 = _SPRITESHEET[1,0]
CHEST               = _SPRITESHEET[1,1]
MOUSE_ON_CHEST      = _SPRITESHEET[1,2]
KEY_ON_WALL         = _SPRITESHEET[1,3]

APPLE               = _SPRITESHEET[2,0]
SHIELD              = _SPRITESHEET[2,1]
MONSTER             = _SPRITESHEET[2,2]
MONSTER_ON_APPLE    = _SPRITESHEET[2,3]
MONSTER_ON_SHIELD   = _SPRITESHEET[2,4]
MOUSE_ON_MONSTER    = _SPRITESHEET[2,5]
SHIELD_ON_WALL      = _SPRITESHEET[2,6]

GRASS               = _SPRITESHEET[5,0]
TREE                = _SPRITESHEET[5,1]
LAVA                = _SPRITESHEET[5,2]
MOUSE_ON_LAVA       = _SPRITESHEET[5,3]
