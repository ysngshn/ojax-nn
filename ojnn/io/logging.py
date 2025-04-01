from __future__ import annotations
import logging
from logging import Logger as HostLogger
from typing_extensions import Self
from functools import partial
from jax import ShapeDtypeStruct
from ..ftypes import external
from .base import IOBase
from .utils import host_callback


def _host_flush_all(host_logger):
    for h in host_logger.handlers:
        h.flush()


def _host_close_all(host_logger):
    for h in host_logger.handlers:
        h.flush()
        h.close()


class UnformatInfo(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._info_formatter = logging.Formatter("%(message)s")

    def format(self, record):
        if record.levelno == logging.INFO:
            return self._info_formatter.format(record)
        else:
            return super().format(record)


class Logger(IOBase):
    host_logger: HostLogger = external()

    def __init__(
        self, host_logger: HostLogger | str | None = None, save_to=None
    ):
        # if host_logger is None:
        #     host_logger = "__ojnn__"
        #     str_format = "%(levelname)s: %(message)s"
        # else:
        #     str_format = "%(levelname)s@%(name)s: %(message)s"
        host_logger = "__ojnn__" if host_logger is None else host_logger
        if host_logger is None or isinstance(host_logger, str):
            host_logger = logging.getLogger(host_logger)
            host_logger.setLevel(1)
            ch = logging.StreamHandler()
            ch.setFormatter(UnformatInfo())
            host_logger.addHandler(ch)
            if save_to is not None:
                fh = logging.FileHandler(save_to)
                fh.setFormatter(UnformatInfo())
                host_logger.addHandler(fh)
        elif not isinstance(host_logger, HostLogger):
            raise ValueError(f"invalid host logger {host_logger}")
        self.assign_(host_logger=host_logger)

    def setLevel(self: Self, level: int | str) -> Self:
        if isinstance(level, str):
            host_callback(partial(self.host_logger.setLevel, level))()
        else:
            host_callback(self.host_logger.setLevel)(level)
        return self

    def isEnabledFor(self, level: int) -> bool:
        return host_callback(
            self.host_logger.isEnabledFor,
            debug_impl=False,
            result_shape_dtypes=ShapeDtypeStruct(shape=(), dtype=bool),
        )(level)

    def getEffectiveLevel(self) -> int:
        return host_callback(
            self.host_logger.getEffectiveLevel,
            debug_impl=False,
            result_shape_dtypes=ShapeDtypeStruct(shape=(), dtype=int),
        )()

    def debug(self: Self, msg: str, *args, **kwargs) -> Self:
        host_callback(partial(self.host_logger.debug, msg))(*args, **kwargs)
        return self

    def info(self: Self, msg: str, *args, **kwargs) -> Self:
        host_callback(partial(self.host_logger.info, msg))(*args, **kwargs)
        return self

    def warning(self: Self, msg: str, *args, **kwargs) -> Self:
        host_callback(partial(self.host_logger.warning, msg))(*args, **kwargs)
        return self

    def error(self: Self, msg: str, *args, **kwargs) -> Self:
        host_callback(partial(self.host_logger.error, msg))(*args, **kwargs)
        return self

    def critical(self: Self, msg: str, *args, **kwargs) -> Self:
        host_callback(partial(self.host_logger.critical, msg))(*args, **kwargs)
        return self

    def log(self: Self, level: int, msg: str, *args, **kwargs) -> Self:
        host_callback(partial(self.host_logger.log, level, msg))(
            *args, **kwargs
        )
        return self

    def exception(self: Self, msg: str, *args, **kwargs) -> Self:
        host_callback(partial(self.host_logger.exception, msg))(
            *args, **kwargs
        )
        return self

    def flush(self: Self) -> Self:
        host_callback(partial(_host_flush_all, self.host_logger))()
        return Self

    def close(self: Self) -> Self:
        host_callback(
            partial(_host_close_all, self.host_logger),
            debug_impl=False,
        )()
        return Self
