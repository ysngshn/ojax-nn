from . import ftypes, struct, utils, modules  # , optim, io

# field types
from .ftypes import (
    StructField,
    Config,
    State,
    Const,
    Child,
    Parameter,
    Buffer,
    Schedulable,
    External,
    config,
    state,
    const,
    child,
    parameter,
    buffer,
    schedulable,
    external,
)

# basic structure
from .struct import (
    Struct,
    new,
)

# utility functions
from .utils import maybe_split

# module system to define neural networks
from .modules import *


__version__ = "0.0.0"
