import functools
import dataclasses
from typing import Sequence
import jax

def struct(Class, static_argnames: Sequence[str]=()):
    Dataclass = dataclasses.dataclass(Class)
    meta_fields = []
    data_fields = []
    for field_info in dataclasses.fields(Dataclass):
        if field_info.name in static_argnames:
            meta_fields.append(field_info.name)
        else:
            data_fields.append(field_info.name)

    # add some convenience methods for mutation and serialisation
    def replace(self, **updates):
        """
        Returns a new object replacing the specified fields with new
        values.
        """
        return dataclasses.replace(self, **updates)
    Dataclass.replace = replace
    # TODO: more methods

    jax.tree_util.register_dataclass(
        nodetype=Dataclass,
        data_fields=data_fields,
        meta_fields=meta_fields,
    )

    return Dataclass

# @dataclasses.dataclass        # doesn't work as b is an array
# @struct                       # doesn't work as size depends on a
@functools.partial(struct, static_argnames=['a']) # works great!
class Env:
    a: int
    b: jax.Array


@jax.jit
def func(env: Env):
    return jax.numpy.arange(env.a), env.b

print(func(Env(a=1, b=jax.numpy.arange(10)).replace(a=10)))
