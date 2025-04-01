from typing import ClassVar, BinaryIO, cast
from io import BytesIO
from os import PathLike
from functools import partial
from tarfile import open as tar_open, TarInfo
from pickle import dump as pickle_dump, load as pickle_load
from numpy import savez as npsavez, load as npload
from jax.numpy import asarray as asjaxarray
from jax.tree_util import tree_flatten, tree_unflatten
from ..ftypes import Config, State, External, Const
from ..struct import Struct
from .utils import host_callback


# save and load any pytree (pickle treedef, numpy.savez pytree leaves)


def save(
    otree,
    save_path: str | bytes | PathLike,
    tar_compress_mode: str = "",
    overwrites: bool = False,
) -> None:
    treeleaves, treedef = tree_flatten(otree)

    def _host_save(tleaves):
        if tar_compress_mode not in ("", "gz", "bz2", "xz"):
            raise ValueError(
                f"invalid tarfile compression mode {tar_compress_mode}"
            )
        mode = f"{'w' if overwrites else 'x'}:{tar_compress_mode}"
        with tar_open(save_path, mode=mode) as tfp:
            with BytesIO() as bp:
                pickle_dump(treedef, cast(BinaryIO, bp))
                tarinfo = TarInfo(name="treedef.pickle")
                tarinfo.size = len(bp.getbuffer())
                bp.seek(0)
                tfp.addfile(tarinfo=tarinfo, fileobj=bp)
            with BytesIO() as bp:
                npsavez(cast(BinaryIO, bp), *tleaves)
                bp.seek(0)  # get back to start
                tarinfo = TarInfo(name="leaves.npz")
                tarinfo.size = len(bp.getbuffer())
                tfp.addfile(tarinfo=tarinfo, fileobj=bp)

    host_callback(_host_save, debug_impl=False)(treeleaves)


def load(
    saved_path: str | bytes | PathLike,
    return_shape_dtypes=NotImplemented,
    **hcb_kwargs,
):

    def _host_load():
        with tar_open(saved_path, "r") as tfp:
            with tfp.extractfile("treedef.pickle") as bp:
                tdef = pickle_load(bp)
            with npload(tfp.extractfile("leaves.npz")) as npz:
                tleaves = [npz[n] for n in npz.files]
        return tdef, tleaves

    if return_shape_dtypes is NotImplemented:  # no host callback for jit
        treedef, leaves = _host_load()
        return tree_unflatten(treedef, [asjaxarray(t) for t in leaves])
    else:  # host callback to support jit
        return host_callback(
            partial(tree_unflatten, *_host_load()),
            debug_impl=False,
            result_shape_dtypes=return_shape_dtypes,
            **hcb_kwargs,
        )()


class IOBase(Struct):
    admissible_field_types: ClassVar = (Config, State, Const, External)
