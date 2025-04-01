from __future__ import annotations
from typing import Any, cast
from typing_extensions import Self
import abc
from functools import partial
import csv
import time
from os.path import exists as opexists, join as opjoin
from contextlib import contextmanager
from queue import Empty
import multiprocessing
from multiprocessing import Process, JoinableQueue
from numpy import append as npappend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from jax.typing import DTypeLike, ArrayLike
from ..struct import new
from ..ftypes import external, config, state, const
from ..modules.misc import TrackAverage, TrackEMA
from .base import IOBase
from .utils import host_callback


class HostMonitor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def start(self, *metric_names: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def log(self, **values):
        raise NotImplementedError

    def stop(self) -> None:
        pass

    def jaxify(self) -> Monitor:
        return Monitor(self)


class HostCSVMonitor(HostMonitor):
    def __init__(
        self,
        csv_path,
        append: bool | None = None,
    ):
        self.csv_path = csv_path
        self.append = append
        self._fp: Any = None
        self._csv_writer: csv.DictWriter | None = None
        self._fieldnames: tuple[str, ...] = ()

    def start(self, *fieldnames: str) -> None:
        self._fieldnames = tuple(fieldnames)
        append, csv_path = self.append, self.csv_path
        if append is None:
            append = opexists(csv_path)
        self._fp = open(csv_path, "a" if append else "w")
        self._csv_writer = csv.DictWriter(
            self._fp, fieldnames=self._fieldnames
        )
        if not append:
            self._csv_writer.writeheader()
            self._fp.flush()

    def log(self, **data: float) -> None:
        cast(csv.DictWriter, self._csv_writer).writerow(data)
        self._fp.flush()

    def stop(self) -> None:
        self._fp.close()
        self._fp = None


def _maybe_draw(figures, t_previous, refresh_interval):
    t = time.time()
    if t - t_previous > refresh_interval:
        HostPlotMonitor.draw_figures(figures)
        return t
    else:
        return t_previous


# less intrusive than plt.pause() which might trigger GUI as active
def _plt_idle(interval: float) -> None:
    if plt.get_fignums():
        # has running figures, unfreeze current one
        plt.gcf().canvas.start_event_loop(interval)
    else:
        time.sleep(interval)


class HostPlotMonitor(HostMonitor):
    plot_job_queue: JoinableQueue | None = None
    plot_process: Process | None = None

    @staticmethod
    def init_figure(
        figname: str,
        axenames: tuple[str, ...],
        axe_rows_cols: tuple[int, int] | None = None,
        fig_kwargs: dict[str, Any] | None = None,
    ) -> Figure:
        num_axes = len(axenames)
        axerows, axecols = (
            (1, num_axes) if axe_rows_cols is None else axe_rows_cols
        )
        assert axerows * axecols >= num_axes, "axe_shape is too small"
        fig_kwargs = {} if fig_kwargs is None else fig_kwargs
        if "figsize" not in fig_kwargs:  # set okayish default figure size
            fig_kwargs["figsize"] = [4.8 * axecols, 4.8 * axerows]
        fig = plt.figure(figname, **fig_kwargs)
        fig.subplots(axerows, axecols)
        fig.set_layout_engine("constrained")
        for ax, n in zip(fig.axes, axenames):
            ax.set_title(n)
            ax.plot([], [])
        plt.pause(0.5)  # give matplotlib some time to show the GUI
        return fig

    @staticmethod
    def update_line(fig: Figure, data: dict[str, tuple[float, float]]) -> None:
        axes = fig.axes
        for i, (n, (x, y)) in enumerate(data.items()):
            ax = axes[i]
            line = ax.get_lines()[0]
            line.set_xdata(npappend(line.get_xdata(), x))
            line.set_ydata(npappend(line.get_ydata(), y))
            # auto rescale and draw plot
            ax.relim()
            ax.autoscale_view()

    @staticmethod
    def execute_plot_job(
        figures: dict[str, Figure], figname: str, args
    ) -> dict[str, Figure]:
        if isinstance(args, tuple):
            figures[figname] = HostPlotMonitor.init_figure(figname, *args)
        elif isinstance(args, dict):
            HostPlotMonitor.update_line(figures[figname], args)
        else:
            raise NotImplementedError
        return figures

    @staticmethod
    def draw_figures(figures: dict[str, Figure]) -> None:
        """redraw figures and update matplotlib GUI display"""
        for fig in figures.values():
            fignum: int = getattr(fig, "number")
            if not plt.fignum_exists(fignum):
                continue
            fig.canvas.draw()
            fig.canvas.flush_events()

    @staticmethod
    def run_plot_process(
        q: JoinableQueue, refresh_interval: float, save_dir: str, block: bool
    ):
        # unfreeze GUI regularly when waiting for next plot job
        def _idle_till_get(q: JoinableQueue, figures, t_previous):
            while True:
                try:
                    plot_job = q.get(block=False)
                except Empty:
                    _plt_idle(0.1)
                    t_previous = _maybe_draw(
                        figures, t_previous, refresh_interval
                    )
                else:
                    break
            return plot_job, t_previous

        t_previous = time.time()
        figures: dict[str, plt.Figure] = {}
        plot_job, t_previous = _idle_till_get(q, figures, t_previous)
        while plot_job is not None:
            figname, args = plot_job
            figures = HostPlotMonitor.execute_plot_job(figures, figname, args)
            q.task_done()
            t_previous = _maybe_draw(figures, t_previous, refresh_interval)
            plot_job, t_previous = _idle_till_get(q, figures, t_previous)
        q.task_done()
        if save_dir:
            plt.show(block=False)
            for n, f in figures.items():
                f.savefig(opjoin(save_dir, f"{n}.pdf"))
        if block:
            print("\n### Close all figure windows to continue. ###\n")
            plt.show(block=True)

    def __init__(
        self,
        name: str,
        step_name: str = "step",
        axe_rows_cols: tuple[int, int] | None = None,
        fig_kwargs: dict[str, Any] | None = None,
    ):
        self.name = name
        self.step_name = step_name
        self.axe_rows_cols = axe_rows_cols
        self.fig_kwargs = fig_kwargs
        self._fieldnames: tuple[str, ...] = ()

    def start(self, *fieldnames: str) -> None:
        if self.plot_job_queue is None:
            raise RuntimeError("need to be inside the run_with_plot() context")
        if self.step_name not in fieldnames:
            raise ValueError(
                f"{self.step_name} requested, received {fieldnames}"
            )
        fieldnames = tuple(n for n in fieldnames if n != self.step_name)
        self._fieldnames = fieldnames
        q = self.plot_job_queue
        q.put([self.name, (fieldnames, self.axe_rows_cols, self.fig_kwargs)])

    def log(self, **data) -> None:
        step = data[self.step_name]
        q = cast(JoinableQueue, self.plot_job_queue)
        q.put([self.name, {n: (step, data[n]) for n in self._fieldnames}])

    def stop(self) -> None:
        pass


@contextmanager
def run_with_plot(
    save_folder: str = "",
    block: bool = False,
    refresh_interval=2.0,  # default refresh interval in seconds
):
    if HostPlotMonitor.plot_job_queue is not None:
        raise RuntimeError("plot monitor process already exists.")
    ctx = multiprocessing.get_context("spawn")
    q = HostPlotMonitor.plot_job_queue = ctx.JoinableQueue()
    p = HostPlotMonitor.plot_process = ctx.Process(
        target=HostPlotMonitor.run_plot_process,
        args=(q, refresh_interval, save_folder, block),
    )
    p.start()
    try:
        yield
    except Exception as err:
        # clear up then raise the Exception
        q.put(None)
        p.join(0.5)
        p.terminate()
        HostPlotMonitor.plot_job_queue = None
        HostPlotMonitor.plot_process = None
        raise err
    else:
        # finish plot jobs, final display then clear the queue
        q.put(None)
        p.join()
        HostPlotMonitor.plot_job_queue = None
        HostPlotMonitor.plot_process = None


class IOMonitor(IOBase, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def start(self: Self, *metric_names: str) -> Self:
        raise NotImplementedError

    @abc.abstractmethod
    def log(self: Self, **values) -> Self:
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self: Self) -> Self:
        raise NotImplementedError


class Monitor(IOMonitor):
    host_monitor: HostMonitor = external()

    def __init__(self, host_monitor: HostMonitor):
        self.assign_(host_monitor=host_monitor)

    def start(self: Self, *metric_names: str) -> Self:
        host_callback(partial(self.host_monitor.start, *metric_names))()
        return self

    def log(self: Self, **values) -> Self:
        host_callback(self.host_monitor.log)(**values)
        return self

    def stop(self: Self) -> Self:
        host_callback(self.host_monitor.stop)()
        return self


class MetricTracker(IOMonitor):
    momentum: float | None = config()
    init_value: ArrayLike | None = const()
    dtype: DTypeLike | None = config()
    _trackers: dict[str, TrackEMA | TrackAverage] | None = state()

    def __init__(
        self,
        momentum: float | None = None,
        init_value: ArrayLike | None = None,
        dtype: DTypeLike | None = None,
    ):
        if momentum is None and init_value is None:
            init_value = 0.0
        self.assign_(
            momentum=momentum,
            init_value=init_value,
            dtype=dtype,
            _trackers=None,
        )

    def start(self: Self, *metric_names: str) -> Self:
        if self.momentum is None:
            trackers = {
                k: TrackAverage(self.dtype).init(()) for k in metric_names
            }
        else:
            trackers = {
                k: TrackEMA(self.momentum, self.init_value, self.dtype).init(
                    ()
                )
                for k in metric_names
            }

        return new(self, _trackers=trackers)

    def log(self: Self, **values) -> Self:
        trackers = {k: self._trackers[k](v)[0] for k, v in values.items()}
        return self.update(_trackers=trackers)

    def stop(self: Self) -> Self:
        return new(self, _trackers=None)

    def __getitem__(self, item: str) -> ArrayLike:
        return self._trackers[item].average

    def __getattr__(self, name: str) -> ArrayLike:
        if name == "_trackers":
            raise AttributeError
        elif name in self._trackers:
            return self._trackers[name].average
        else:
            raise AttributeError


class MultiMonitor(IOMonitor):
    _monitors: dict[str, Monitor] = state()

    def __init__(self, **monitors: IOMonitor):
        for k, v in monitors.items():
            if not isinstance(v, IOMonitor):
                raise ValueError(f"{k} is not an instance of IOMonitor")
        self.assign_(_monitors=monitors)

    def start(self: Self, *metric_names: str) -> Self:
        monitors = {
            k: v.start(*metric_names) for k, v in self._monitors.items()
        }
        return self.update(_monitors=monitors)

    def log(self: Self, **values) -> Self:
        monitors = {k: v.log(**values) for k, v in self._monitors.items()}
        return self.update(_monitors=monitors)

    def stop(self: Self) -> Self:
        monitors = {k: v.stop() for k, v in self._monitors.items()}
        return self.update(_monitors=monitors)

    def __getitem__(self, item: str) -> Monitor:
        return self._monitors[item]

    def __getattr__(self, name: str) -> Monitor:
        if name == "_monitors":
            raise AttributeError
        elif name in self._monitors:
            return self._monitors[name]
        else:
            raise AttributeError
