import enum
import imageio
import einops
import jax.numpy as jnp
from importlib.resources import files


class LevelOfDetail(enum.IntEnum):
    BOOLEAN = 0
    RGB_1PX = 1
    RGB_3PX = 3
    RGB_4PX = 4
    RGB_8PX = 8


_MIPMAP = {
    px: jnp.array(einops.rearrange(
        imageio.v2.imread(
            files('jaxgmg.graphics').joinpath(f'spritesheet{px}.png')
        ) / 255,
        '(H h) (W w) c -> H W h w c',
        h=px,
        w=px,
    ))
    for px in (
        LevelOfDetail.RGB_1PX,
        LevelOfDetail.RGB_3PX,
        LevelOfDetail.RGB_4PX,
        LevelOfDetail.RGB_8PX,
    )
}


def spritesheet(lod: LevelOfDetail):
    _spritesheet = _MIPMAP[lod]
    return {
        'BLANK':                   _spritesheet[0,0],
        'PATH':                    _spritesheet[0,1],
        'WALL':                    _spritesheet[0,2],
        'MOUSE':                   _spritesheet[0,3],
        'CHEESE':                  _spritesheet[0,4],
        'DISH':                    _spritesheet[0,5],
        'SMALL_CHEESE':            _spritesheet[0,6],
        'CHEESE_ON_DISH':          _spritesheet[0,7],
        'KEY':                     _spritesheet[1,0],
        'CHEST':                   _spritesheet[1,1],
        'MOUSE_ON_CHEST':          _spritesheet[1,2],
        'KEY_ON_WALL':             _spritesheet[1,3],
        'GRASS':                   _spritesheet[1,4],
        'TREE':                    _spritesheet[1,5],
        'LAVA':                    _spritesheet[1,6],
        'MOUSE_ON_LAVA':           _spritesheet[1,7],
        'APPLE':                   _spritesheet[2,0],
        'SHIELD':                  _spritesheet[2,1],
        'MONSTER':                 _spritesheet[2,2],
        'MONSTER_ON_APPLE':        _spritesheet[2,3],
        'MONSTER_ON_SHIELD':       _spritesheet[2,4],
        'MOUSE_ON_MONSTER':        _spritesheet[2,5],
        'SHIELD_ON_WALL':          _spritesheet[2,6],
        'LEADER_MOUSE':            _spritesheet[3,0],
        'BEACON_OFF':              _spritesheet[3,1],
        'BEACON_ON':               _spritesheet[3,2],
        'LEADER_ON_BEACON_OFF':    _spritesheet[3,3],
        'LEADER_ON_BEACON_ON':     _spritesheet[3,4],
        'MOUSE_ON_BEACON_OFF':     _spritesheet[3,5],
        'MOUSE_ON_LEADER':         _spritesheet[3,6],
        'BOTH_MICE_ON_BEACON_OFF': _spritesheet[3,7],
    }

