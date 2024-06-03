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

LEADER_MOUSE        = _SPRITESHEET[3,0]
BEACON_INACTIVE_1   = _SPRITESHEET[3,1]
BEACON_INACTIVE_2   = _SPRITESHEET[3,2]
BEACON_INACTIVE_3   = _SPRITESHEET[3,3]
BEACON_ACTIVE_1     = _SPRITESHEET[3,4]
BEACON_ACTIVE_2     = _SPRITESHEET[3,5]
BEACON_ACTIVE_3     = _SPRITESHEET[3,6]
MOUSE_ON_LEADER     = _SPRITESHEET[4,0]
LEADER_ON_BEACON_1  = _SPRITESHEET[4,1]
LEADER_ON_BEACON_2  = _SPRITESHEET[4,2]
LEADER_ON_BEACON_3  = _SPRITESHEET[4,3]
